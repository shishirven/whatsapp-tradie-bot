"""Regression set: multi-turn conversations built from real failures.

The owner's golden set (`golden_set.yaml`) is 30 single messages with reference replies,
and stays as written. This set is different. It holds short conversations that each pin
one behaviour, most of them taken from bugs found in live use, with code checks on both
the reply and what was stored on the job:

    python -m evals.regression                 # all cases offline; live Gemini on a random 10 if GEMINI_API_KEY is set
    python -m evals.regression --offline       # offline only (what CI runs)
    python -m evals.regression --category nonsense_first_message
    python -m evals.regression --case A1 --repeats 3 --rpm 5
    python -m evals.regression --sample 0      # live on every case (about 100 model calls)
    python -m evals.regression --seed 42       # repeat an earlier live sample exactly

Live runs cost model calls, so by default they cover a random sample of LIVE_SAMPLE cases,
spread across the categories (one from each, then extras at random). The seed is printed
and saved in the report, so a sample can be run again. Offline runs every case; it's free.

Each conversation runs through the real app the way the simulator does: Riley greets first,
then each customer message in turn. Offline uses the keyword qualifier. Live uses Gemini;
a turn where Gemini failed and the offline qualifier answered is reported as a fallback,
not a pass. Turns with a `model:` block replay a recorded model output (for example Gemini
writing "\\u2013"), so how the app handles a model quirk is tested without calling a model.

Cases live in `regression_set.yaml`; the checks a turn can use are listed in CHECKS below.
Reports are written to evals/results/regression_latest.json and .md.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import yaml

ROOT = Path(__file__).resolve().parent
CASES_PATH = ROOT / "regression_set.yaml"
BUSINESS_PATH = ROOT / "fixtures" / "regression_business.json"

LIVE_SAMPLE = 10

CATEGORIES = {
    "nonsense_first_message": "A. Nonsense first message",
    "vague_request": "B. Vague request",
    "not_electrical": "C. Not electrical or spam",
    "unclear_answer_mid_chat": "D. Unclear answer mid-chat",
    "escalation_path": "E. Escalation path",
    "must_not_clarify": "F. Look-alikes that must not be clarified",
    "hazard_contrast": "G. Hazard contrast pairs",
    "model_output_hygiene": "H. Model output hygiene (replayed)",
    "wording": "I. Wording",
    "business_questions": "J. Business questions",
}

_ASKS = {
    "job_type": ("electrician for", "electrical job", "what needs doing", "what is it", "reply with a number",
                 "what do you need", "what can we help", "what work"),
    "suburb": ("suburb",),
    "contact_name": ("name",),
    "urgency": ("urgent", "emergency", "can it wait"),
    "description": ("what's going on", "what is going on", "what exactly", "describe"),
}
_GENDERED = re.compile(r"\b(?:he|him|his|he'll|he's|he'd)\b", re.IGNORECASE)
_FACTS = ("job_type", "suburb", "urgency", "description", "preferred_time", "contact_name")


# --- loading -------------------------------------------------------------------------


def load_cases(path: Path | None = None) -> list[dict]:
    data = yaml.safe_load((path or CASES_PATH).read_text(encoding="utf-8"))
    cases = data.get("cases") or []
    seen = set()
    for case in cases:
        for key in ("id", "category", "why", "turns"):
            if not case.get(key):
                raise ValueError(f"case {case.get('id')} is missing {key}")
        if case["id"] in seen:
            raise ValueError(f"duplicate case id {case['id']}")
        seen.add(case["id"])
        if case["category"] not in CATEGORIES:
            raise ValueError(f"case {case['id']}: unknown category {case['category']}")
        for turn in case["turns"]:
            if "say" not in turn:
                raise ValueError(f"case {case['id']}: every turn needs 'say'")
            for check in _check_items(turn.get("expect") or []):
                if check[0] not in CHECKS:
                    raise ValueError(f"case {case['id']}: unknown check {check[0]}")
    return cases


def load_business():
    from app.business import parse_business

    return parse_business(json.loads(BUSINESS_PATH.read_text(encoding="utf-8")))


def is_scripted(case: dict) -> bool:
    return any("model" in turn for turn in case["turns"])


# --- checks -------------------------------------------------------------------------


@dataclass
class Turn:
    said: str
    reply: str
    lead: dict
    calls: list[dict]
    alerts: list[dict]
    business: object


def _value(lead: dict, key: str):
    value = lead.get(key)
    return value if value not in ("", None) else None


def _fact(key: str):
    def check(turn: Turn, expected) -> tuple[bool, str]:
        actual = _value(turn.lead, key)
        if expected == "any":
            return actual is not None, f"{key}={actual!r}, expected something"
        if expected == "clean":
            return actual is None or "\\u" not in str(actual), f"{key}={actual!r} has an escape"
        return actual == expected, f"{key}={actual!r}, expected {expected!r}"

    return check


def _contains(turn: Turn, expected) -> tuple[bool, str]:
    missing = [item for item in _as_list(expected) if item.casefold() not in turn.reply.casefold()]
    return not missing, f"reply lacks {missing}"


def _contains_any(turn: Turn, expected) -> tuple[bool, str]:
    options = _as_list(expected)
    return any(item.casefold() in turn.reply.casefold() for item in options), f"reply has none of {options}"


def _lacks(turn: Turn, expected) -> tuple[bool, str]:
    found = [item for item in _as_list(expected) if item.casefold() in turn.reply.casefold()]
    return not found, f"reply contains {found}"


def _stores_nothing(turn: Turn, _=None) -> tuple[bool, str]:
    stored = {key: turn.lead.get(key) for key in _FACTS if _value(turn.lead, key) is not None}
    return not stored, f"stored {stored}"


def _asks_about(turn: Turn, field_name) -> tuple[bool, str]:
    cues = _ASKS[field_name]
    reply = turn.reply.casefold()
    asked = ("?" in reply or "reply with a number" in reply) and any(cue in reply for cue in cues)
    return asked, f"reply doesn't ask about {field_name}"


def _one_question(turn: Turn, _=None) -> tuple[bool, str]:
    return turn.reply.count("?") <= 1, f"{turn.reply.count('?')} questions"


def _mentions_000(turn: Turn, expected=None) -> tuple[bool, str]:
    expected = True if expected is None else expected
    has = "000" in turn.reply
    return has is bool(expected), "000 missing" if expected else "mentions 000"


def _no_echo(turn: Turn, _=None) -> tuple[bool, str]:
    said = " ".join(turn.said.split()).casefold()
    if len(said.split()) < 3:
        return True, ""
    return said not in turn.reply.casefold(), "reply reads the customer's message back"


def _no_gendered_owner(turn: Turn, _=None) -> tuple[bool, str]:
    match = _GENDERED.search(turn.reply)
    return match is None, f"reply says {match.group(0)!r}" if match else ""


def _owner_capitalised(turn: Turn, _=None) -> tuple[bool, str]:
    bad = turn.reply.startswith("the owner") or re.search(r"[.!?]\s+the owner\b", turn.reply)
    return not bad, "sentence starts with lowercase 'the owner'"


def _bool_field(key: str):
    def check(turn: Turn, expected) -> tuple[bool, str]:
        actual = bool(turn.lead.get(key))
        return actual is bool(expected), f"{key}={actual}, expected {bool(expected)}"

    return check


def _string_field(key: str):
    def check(turn: Turn, expected) -> tuple[bool, str]:
        actual = turn.lead.get(key)
        return actual == expected, f"{key}={actual!r}, expected {expected!r}"

    return check


def _alerts(turn: Turn, expected) -> tuple[bool, str]:
    return len(turn.alerts) == int(expected), f"{len(turn.alerts)} owner alerts, expected {expected}"


def _booked(turn: Turn, expected=None) -> tuple[bool, str]:
    expected = True if expected is None else expected
    status = (turn.lead.get("booking") or {}).get("status")
    return (status == "booked") is bool(expected), f"booking status {status!r}"


def _said_once(turn: Turn, expected) -> tuple[bool, str]:
    count = turn.reply.casefold().count(str(expected).casefold())
    return count == 1, f"{expected!r} appears {count} times"


CHECKS = {
    "stores_nothing": _stores_nothing,
    "job_type": _fact("job_type"),
    "suburb": _fact("suburb"),
    "urgency": _fact("urgency"),
    "description": _fact("description"),
    "preferred_time": _fact("preferred_time"),
    "contact_name": _fact("contact_name"),
    "reply_contains": _contains,
    "reply_contains_any": _contains_any,
    "reply_lacks": _lacks,
    "asks_about": _asks_about,
    "one_question": _one_question,
    "mentions_000": _mentions_000,
    "no_echo": _no_echo,
    "no_gendered_owner": _no_gendered_owner,
    "owner_capitalised": _owner_capitalised,
    "handoff": _bool_field("handoff"),
    "handoff_reason": _string_field("handoff_reason"),
    "escalated": _bool_field("escalated"),
    "escalation_reason": _string_field("escalation_reason"),
    "alerts": _alerts,
    "booked": _booked,
    "said_once": _said_once,
}


def _check_items(expect) -> list[tuple[str, object]]:
    """`expect` entries are a name ("stores_nothing") or a one-key mapping ({"job_type": "fault"})."""
    items = []
    for entry in expect:
        if isinstance(entry, str):
            items.append((entry, None))
        elif isinstance(entry, dict) and len(entry) == 1:
            items.append(next(iter(entry.items())))
        else:
            raise ValueError(f"bad check {entry!r}")
    return items


def _as_list(value) -> list[str]:
    return [str(item) for item in (value if isinstance(value, list) else [value])]


# --- running ------------------------------------------------------------------------


class _Script:
    """Replays a recorded model output for turns that have one; other turns use `fallback`."""

    def __init__(self, outputs: list[dict | None], fallback) -> None:
        self.outputs = list(outputs)
        self.fallback = fallback

    def qualify(self, turn):
        output = self.outputs.pop(0) if self.outputs else None
        if output is None:
            return self.fallback.qualify(turn)
        from app.llm import CallMeta, parse_gemini_qualification

        args = {
            "reply": "", "request_human": False, "job_type": None, "suburb": None, "urgency": None,
            "hazard": "none", "intent": None, "clarifying_question": None, "description": None,
            "preferred_time": None, "contact_name": None, **output,
        }
        call = SimpleNamespace(name="record_job_intake", args=args)
        draft = parse_gemini_qualification(SimpleNamespace(function_calls=[call], text=None, candidates=[]))
        draft.meta = CallMeta(provider="scripted")
        return draft


class _Paced:
    def __init__(self, inner, rpm: int | None) -> None:
        self.inner = inner
        self.gap = 60.0 / rpm if rpm else 0.0
        self.next = 0.0

    def qualify(self, turn):
        if self.gap:
            wait = self.next - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self.next = time.monotonic() + self.gap
        return self.inner.qualify(turn)


def run_case(case: dict, *, mode: str, business, api_key: str = "", model: str = "", rpm: int | None = None) -> dict:
    """Run one conversation. Returns {"passed", "fallback", "turns": [...]}."""
    from fastapi.testclient import TestClient

    from app.config import Settings
    from app.llm import GeminiQualifier, RuleBasedQualifier
    from app.main import create_app
    from app.notify import ListNotifier
    from app.routing import build_intent
    from evals.harness import EVAL_NOW

    offline = RuleBasedQualifier(business)
    if mode == "live":
        base = _Paced(GeminiQualifier(api_key, model, business=business), rpm)
        intent = build_intent("gemini", "", model, gemini_api_key=api_key, gemini_model=model)
    else:
        base = offline
        intent = build_intent("mock", "", "")
    qualifier = _Script([turn.get("model") for turn in case["turns"]], base) if is_scripted(case) else base

    with tempfile.TemporaryDirectory(prefix="regression-") as tmp:
        settings = Settings(
            anthropic_api_key="", anthropic_model="claude-sonnet-5-5", llm_provider="mock",
            messaging_provider="simulated", whatsapp_verify_token="", whatsapp_token="",
            whatsapp_phone_number_id="", whatsapp_app_secret="", sqlite_path=str(Path(tmp) / "case.db"),
        )
        notifier = ListNotifier()
        logging.getLogger("tradie").setLevel(logging.ERROR)
        app = create_app(
            settings=settings, qualifier=qualifier, notifier=notifier, business=business,
            now=lambda: EVAL_NOW, intent=intent,
        )
        logging.getLogger("tradie").setLevel(logging.ERROR)
        client = TestClient(app)
        start = client.post("/api/conversations", json={"profile_name": case.get("profile_name")}).json()
        conversation_id = start["conversation_id"]
        results, passed, fallback = [], True, False
        for index, step in enumerate(case["turns"], start=1):
            calls_before = len(app.state.db.list_llm_calls())
            alerts_before = len(notifier.events)
            response = client.post("/api/chat", json={"conversation_id": conversation_id, "text": step["say"]}).json()
            calls = app.state.db.list_llm_calls()[calls_before:]
            turn = Turn(
                said=step["say"], reply=response.get("reply") or "", lead=response.get("lead") or {},
                calls=calls, alerts=notifier.events[alerts_before:], business=business,
            )
            if mode == "live" and "model" not in step and any(call.get("status") == "fallback" for call in calls):
                fallback = True
            failures = []
            for name, argument in _check_items(step.get("expect") or []):
                ok, detail = CHECKS[name](turn, argument)
                if not ok:
                    failures.append(f"{name}: {detail}")
            passed = passed and not failures
            results.append({"turn": index, "said": step["say"], "reply": turn.reply, "failures": failures})
    return {"passed": passed, "fallback": fallback, "turns": results}


def run(
    *,
    offline_only: bool = False,
    category: str | None = None,
    case_id: str | None = None,
    repeats: int = 1,
    rpm: int | None = None,
    results_dir: str | Path = "evals/results",
    sample: int = LIVE_SAMPLE,
    seed: int | None = None,
) -> dict:
    business = load_business()
    cases = [
        case for case in load_cases()
        if (not category or case["category"] == category) and (not case_id or case["id"] == case_id)
    ]
    api_key = "" if offline_only else os.getenv("GEMINI_API_KEY", "").strip()
    model = os.getenv("GEMINI_MODEL", "").strip() or "gemini-3.5-flash-lite"
    modes = ["offline"] + (["live"] if api_key else [])
    seed = seed if seed is not None else random.randrange(1, 10**6)
    live_ids = {case["id"] for case in sample_live(cases, sample, seed)} if api_key else set()
    rows = []
    for case in cases:
        row = {"id": case["id"], "category": case["category"], "why": case["why"], "modes": {}}
        for mode in modes:
            if mode == "live" and is_scripted(case):
                row["modes"][mode] = {"verdict": "replayed", "runs": []}
                continue
            if mode == "live" and case["id"] not in live_ids:
                row["modes"][mode] = {"verdict": "not sampled", "runs": []}
                continue
            runs = [
                run_case(case, mode=mode, business=business, api_key=api_key, model=model, rpm=rpm)
                for _ in range(max(1, repeats if mode == "live" else 1))
            ]
            row["modes"][mode] = {"verdict": _verdict(runs), "runs": runs}
        rows.append(row)
    report = {"modes": modes, "repeats": repeats, "model": model if api_key else None, "cases": rows,
              "summary": _summary(rows, modes), "seed": seed if api_key else None,
              "live_sample": sorted(live_ids)}
    _write(report, Path(results_dir))
    return report


def sample_live(cases: list[dict], size: int, seed: int) -> list[dict]:
    """A random sample for live runs: one case per category first, then extras at random.

    Replayed cases (with a recorded `model:` output) never call a model, so they're left out.
    `size` 0 or less means every live-eligible case.
    """
    eligible = [case for case in cases if not is_scripted(case)]
    if size <= 0 or size >= len(eligible):
        return eligible
    rng = random.Random(seed)
    by_category: dict[str, list[dict]] = defaultdict(list)
    for case in eligible:
        by_category[case["category"]].append(case)
    for group in by_category.values():
        rng.shuffle(group)
    order = list(by_category)
    rng.shuffle(order)
    picked: list[dict] = []
    while len(picked) < size:
        for category in order:
            if by_category[category] and len(picked) < size:
                picked.append(by_category[category].pop())
    return picked


def _verdict(runs: list[dict]) -> str:
    if any(run["fallback"] for run in runs):
        return "fallback"
    outcomes = {run["passed"] for run in runs}
    if outcomes == {True}:
        return "pass"
    if outcomes == {False}:
        return "fail"
    return "unstable"


def _summary(rows: list[dict], modes: list[str]) -> dict:
    summary = {}
    for mode in modes:
        by_category: dict[str, Counter] = defaultdict(Counter)
        for row in rows:
            verdict = row["modes"][mode]["verdict"]
            if verdict == "not sampled":
                continue
            by_category[row["category"]][verdict] += 1
        total = Counter()
        for counts in by_category.values():
            total.update(counts)
        summary[mode] = {"total": dict(total), "by_category": {key: dict(value) for key, value in by_category.items()}}
    return summary


def _write(report: dict, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "regression_latest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (directory / "regression_latest.md").write_text(render_markdown(report), encoding="utf-8")


def render_markdown(report: dict) -> str:
    lines = ["# Regression set", ""]
    for mode in report["modes"]:
        total = report["summary"][mode]["total"]
        counted = sum(value for key, value in total.items() if key != "replayed")
        label = (
            f"live ({report['model']}, {len(report['live_sample'])} sampled, seed {report['seed']})"
            if mode == "live" else mode
        )
        lines.append(f"**{label}:** {total.get('pass', 0)}/{counted} pass"
                     + "".join(f", {total[key]} {key}" for key in ("fail", "unstable", "fallback") if total.get(key)))
    lines += ["", "| Category | " + " | ".join(report["modes"]) + " |", "| --- |" + " --- |" * len(report["modes"])]
    for key, title in CATEGORIES.items():
        cells = []
        for mode in report["modes"]:
            counts = report["summary"][mode]["by_category"].get(key)
            if not counts:
                cells.append("—")
                continue
            counted = sum(value for name, value in counts.items() if name != "replayed")
            cells.append(f"{counts.get('pass', 0)}/{counted}" if counted else "replayed")
        lines.append(f"| {title} | " + " | ".join(cells) + " |")
    failures = [
        (row, mode, run)
        for row in report["cases"]
        for mode, result in row["modes"].items()
        for run in result["runs"]
        if not run["passed"] or run["fallback"]
    ]
    if failures:
        lines += ["", "## Failures", ""]
        for row, mode, run in failures:
            lines.append(f"**{row['id']}** ({mode}) {row['why']}")
            for turn in run["turns"]:
                for failure in turn["failures"]:
                    lines.append(f"- turn {turn['turn']} \"{turn['said']}\": {failure}")
                    lines.append(f"  - reply: {turn['reply'][:200]}")
            if run["fallback"]:
                lines.append("- Gemini failed on a turn and the offline qualifier answered")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    from dotenv import load_dotenv

    load_dotenv(override=False)
    parser = argparse.ArgumentParser(prog="python -m evals.regression", description="Run the regression set.")
    parser.add_argument("--offline", action="store_true", help="Offline only, even if GEMINI_API_KEY is set.")
    parser.add_argument("--category", choices=sorted(CATEGORIES), help="Only this category.")
    parser.add_argument("--case", help="Only this case id, e.g. A1.")
    parser.add_argument("--repeats", type=int, default=1, help="Live runs per case; flips are 'unstable'.")
    parser.add_argument("--rpm", type=int, default=0, help="Max Gemini calls per minute (5 on the free tier).")
    parser.add_argument("--sample", type=int, default=LIVE_SAMPLE,
                        help=f"Live cases to sample at random (default {LIVE_SAMPLE}); 0 runs every case.")
    parser.add_argument("--seed", type=int, help="Seed for the live sample, to repeat an earlier one.")
    parser.add_argument("--results-dir", default="evals/results")
    args = parser.parse_args(argv)
    report = run(
        offline_only=args.offline, category=args.category, case_id=args.case,
        repeats=max(1, args.repeats), rpm=args.rpm or None, results_dir=args.results_dir,
        sample=args.sample, seed=args.seed,
    )
    print(render_markdown(report))
    print(f"Wrote {args.results_dir}/regression_latest.json and regression_latest.md")
    failed = any(
        result["verdict"] in ("fail", "unstable")
        for row in report["cases"] for result in row["modes"].values()
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
