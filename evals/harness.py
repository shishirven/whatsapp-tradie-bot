"""Run golden cases through the real conversation pipeline and score them."""

from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

from app.booking import DatabaseCalendar, available_slots, format_slot
from app.business import Business, parse_business
from app.db import Database
from app.llm import GeminiQualifier, RuleBasedQualifier
from app.messaging import Inbound, SimulatedGateway
from app.notify import LogNotifier
from app.phones import normalise_au_phone
from app.routing import build_intent
from app.service import handle_incoming
from evals.checks import CheckContext, evaluate, first_failure, passed
from evals.judge import DEFAULT_REPLY_MODEL, grade, resolve_judge_model

ROOT = Path(__file__).resolve().parent
GOLDEN_PATH = ROOT / "golden_set.yaml"
BUSINESS_PATH = ROOT / "fixtures" / "test_business.json"
CLIENTS_PATH = ROOT / "fixtures" / "clients.json"
LABELS_PATH = ROOT / "human_labels.yaml"
EVAL_NOW = datetime(2026, 11, 16, 8, 0, tzinfo=ZoneInfo("Australia/Sydney"))
_PLACEHOLDER = re.compile(r"\{([a-z0-9_]+)\}")


def load_golden(path: Path | None = None) -> dict:
    data = yaml.safe_load((path or GOLDEN_PATH).read_text(encoding="utf-8"))
    cases = data.get("cases") or []
    ids = [int(case["id"]) for case in cases]
    if ids != list(range(1, 31)):
        raise ValueError(f"golden set must be cases 1..30 in order, got {ids}")
    for case in cases:
        for key in ("category", "customer", "reference", "rule"):
            if not str(case.get(key) or "").strip():
                raise ValueError(f"case {case.get('id')} is missing {key}")
        case["id"] = int(case["id"])
        case["setup"] = case.get("setup") or {}
    return data


def load_business(path: Path | None = None) -> tuple[dict, Business]:
    raw = json.loads((path or BUSINESS_PATH).read_text(encoding="utf-8"))
    return raw, parse_business(raw)


def load_clients(path: Path | None = None) -> dict:
    return json.loads((path or CLIENTS_PATH).read_text(encoding="utf-8"))


def run_eval(
    *,
    offline: bool | None = None,
    repeats: int = 1,
    results_dir: str | Path | None = None,
    write: bool = True,
    api_key: str | None = None,
    judge_model: str | None = None,
    reply_model: str | None = None,
    rpm: int | None = None,
) -> dict:
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    key = api_key if api_key is not None else os.getenv("GEMINI_API_KEY", "").strip()
    if offline is None:
        offline = not bool(key)
    offline = bool(offline)
    reply = reply_model or os.getenv("GEMINI_MODEL", "").strip() or DEFAULT_REPLY_MODEL
    judge = None if offline else (judge_model or resolve_judge_model())
    golden = load_golden()
    raw_business, business = load_business()
    clients = load_clients()
    slots = _slots(business)
    values = _base_values(business, slots)
    pacer = _Pacer(rpm)
    case_reports = []
    for case in golden["cases"]:
        case_reports.append(
            _run_case(
                case,
                golden.get("global_rules") or "",
                raw_business=raw_business,
                business=business,
                clients=clients,
                slots=slots,
                values=values,
                repeats=repeats,
                offline=offline,
                api_key=key,
                reply_model=reply,
                judge_model=judge,
                pacer=pacer,
            )
        )
    passed_count = sum(1 for case in case_reports if case["pass"])
    unstable = [case["id"] for case in case_reports if case["unstable"]]
    report = {
        "mode": "offline" if offline else "judge",
        "repeats": repeats,
        "reply_model": "mock" if offline else reply,
        "judge_model": judge,
        "business": business.name,
        "passed": passed_count,
        "total": len(case_reports),
        "pass_rate": round(passed_count / len(case_reports), 4) if case_reports else 0,
        "unstable": unstable,
        "cases": case_reports,
    }
    if write and results_dir is not None:
        _write_results(report, Path(results_dir))
    return report


def summarise_repeats(repeats: list[dict]) -> dict:
    """Combine N runs. A verdict that is not the same every time is unstable."""
    if not repeats:
        return {
            "pass": False,
            "unstable": False,
            "deterministic_pass": False,
            "judge_pass": None,
            "reason": "no repeats",
            "unstable_because": "",
        }
    det = [bool(item["deterministic_pass"]) for item in repeats]
    combined = [bool(item["pass"]) for item in repeats]
    judges = [item.get("judge_pass") for item in repeats]
    judge_values = [value for value in judges if isinstance(value, bool)]
    unstable = len(set(det)) > 1 or len(set(combined)) > 1
    if len(judge_values) > 1 and len(set(judge_values)) > 1:
        unstable = True
    if any(value is None for value in judges) and judge_values:
        unstable = True
    majority = _majority(judge_values)
    reason = next((item["reason"] for item in repeats if not item["pass"]), repeats[-1]["reason"])
    because = []
    if len(set(det)) > 1:
        because.append("deterministic verdict flipped")
    if len(set(combined)) > 1:
        because.append("overall verdict flipped")
    if len(judge_values) > 1 and len(set(judge_values)) > 1:
        because.append("judge verdict flipped")
    return {
        "pass": all(combined),
        "unstable": unstable,
        "deterministic_pass": all(det),
        "judge_pass": majority,
        "reason": reason,
        "unstable_because": "; ".join(because),
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# Eval results",
        "",
        f"Mode: {report['mode']}",
        f"Reply model: {report['reply_model']}",
        f"Judge model: {report['judge_model'] or 'not run'}",
        f"Repeats: {report['repeats']}",
        f"Business: {report['business']}",
        (
            f"Pass rate: {report['passed']}/{report['total']} "
            f"({report['pass_rate'] * 100:.1f}%)"
        ),
        "Unstable: " + (", ".join(str(item) for item in report["unstable"]) or "none"),
        "",
        "| ID | Category | Deterministic | Judge | Pass | Unstable | Reason |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for case in report["cases"]:
        judge = case.get("judge")
        judge_cell = "skipped" if judge is None else ("pass" if judge.get("pass") else "fail")
        lines.append(
            "| {id} | {category} | {det} | {judge} | {passed} | {unstable} | {reason} |".format(
                id=case["id"],
                category=_cell(case["category"]),
                det="pass" if case["deterministic"]["pass"] else "fail",
                judge=judge_cell,
                passed="pass" if case["pass"] else "fail",
                unstable="yes" if case["unstable"] else "no",
                reason=_cell(case["reason"]),
            )
        )
    lines.extend(["", "## Cases", ""])
    for case in report["cases"]:
        lines.append(f"### {case['id']}. {case['category']}")
        lines.append("")
        lines.append(f"Customer: {case['customer']}")
        if case.get("sent") and case["sent"] != case["customer"]:
            lines.append(f"Sent: {case['sent']}")
        lines.append("")
        lines.append(f"Rule: {case['rule']}")
        lines.append("")
        lines.append("Reference:")
        lines.append("")
        lines.append(case.get("reference_resolved") or case["reference"])
        lines.append("")
        lines.append("Reply:")
        lines.append("")
        lines.append(case.get("reply") or "")
        lines.append("")
        if case["unstable"]:
            lines.append(f"Unstable: {case.get('unstable_because') or 'verdict flipped'}")
            lines.append("")
        failing = [item for item in case["deterministic"]["checks"] if not item["pass"]]
        if failing:
            lines.append("Failing checks:")
            lines.append("")
            for item in failing:
                lines.append(f"- {item['name']}: {item['detail']}")
            lines.append("")
        judge = case.get("judge")
        if judge:
            lines.append(
                f"Judge ({judge.get('model') or report.get('judge_model')}): "
                f"{'pass' if judge.get('pass') else 'fail'} — {judge.get('reason') or ''}"
            )
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def print_table(report: dict) -> None:
    print(
        f"{report['mode']}  {report['passed']}/{report['total']} "
        f"({report['pass_rate'] * 100:.1f}%)  repeats={report['repeats']}  "
        f"judge={report['judge_model'] or 'skipped'}"
    )
    print(f"{'ID':<4} {'Det':<6} {'Judge':<8} {'Pass':<6} {'Unstable':<9} Reason")
    for case in report["cases"]:
        judge = case.get("judge")
        judge_cell = "skip" if judge is None else ("pass" if judge.get("pass") else "fail")
        reason = case["reason"] or ("unstable" if case["unstable"] else "")
        print(
            f"{case['id']:<4} "
            f"{'pass' if case['deterministic']['pass'] else 'fail':<6} "
            f"{judge_cell:<8} "
            f"{'pass' if case['pass'] else 'fail':<6} "
            f"{'yes' if case['unstable'] else 'no':<9} "
            f"{reason}"
        )
    if report["unstable"]:
        print("Unstable cases: " + ", ".join(str(item) for item in report["unstable"]))


def _run_case(
    case,
    global_rules,
    *,
    raw_business,
    business,
    clients,
    slots,
    values,
    repeats,
    offline,
    api_key,
    reply_model,
    judge_model,
    pacer,
) -> dict:
    runs = []
    for index in range(repeats):
        runs.append(
            _run_once(
                case,
                global_rules,
                raw_business=raw_business,
                business=business,
                clients=clients,
                slots=slots,
                values=values,
                repeat_index=index,
                offline=offline,
                api_key=api_key,
                reply_model=reply_model,
                judge_model=judge_model,
                pacer=pacer,
            )
        )
    summary = summarise_repeats(runs)
    first = runs[0]
    judge = None
    if not offline:
        reasons = [item.get("judge_reason") or "" for item in runs]
        judge = {
            "pass": summary["judge_pass"],
            "reason": next((reason for reason in reasons if reason), ""),
            "model": judge_model,
            "verdicts": [item.get("judge_pass") for item in runs],
            "unstable": len({value for value in summary_judge_set(runs)}) > 1,
        }
    return {
        "id": case["id"],
        "category": case["category"],
        "customer": case["customer"],
        "sent": first["sent"],
        "reference": case["reference"],
        "reference_resolved": first["reference_resolved"],
        "rule": case["rule"],
        "reply": first["reply"],
        "deterministic": {
            "pass": summary["deterministic_pass"],
            "checks": first["checks"],
        },
        "judge": judge,
        "pass": summary["pass"],
        "unstable": summary["unstable"],
        "unstable_because": summary["unstable_because"],
        "reason": _case_reason(summary, first),
        "repeats": [
            {
                "reply": item["reply"],
                "deterministic_pass": item["deterministic_pass"],
                "judge_pass": item.get("judge_pass"),
                "pass": item["pass"],
                "reason": item["reason"],
            }
            for item in runs
        ],
    }


def summary_judge_set(runs: list[dict]) -> list[bool]:
    return [item["judge_pass"] for item in runs if isinstance(item.get("judge_pass"), bool)]


def _case_reason(summary: dict, first: dict) -> str:
    reason = summary["reason"] or first["reason"]
    if summary["unstable"] and summary["unstable_because"]:
        if reason:
            return f"{reason} ({summary['unstable_because']})"
        return summary["unstable_because"]
    return reason


def _run_once(
    case,
    global_rules,
    *,
    raw_business,
    business,
    clients,
    slots,
    values,
    repeat_index,
    offline,
    api_key,
    reply_model,
    judge_model,
    pacer,
) -> dict:
    variants = case.get("variants") or [None]
    variant_runs = []
    for variant in variants:
        variant_runs.append(
            _run_variant(
                case,
                variant,
                global_rules,
                raw_business=raw_business,
                business=business,
                clients=clients,
                slots=slots,
                values=values,
                repeat_index=repeat_index,
                offline=offline,
                api_key=api_key,
                reply_model=reply_model,
                judge_model=judge_model,
                pacer=pacer,
            )
        )
    checks = [item for run in variant_runs for item in run["checks"]]
    det_pass = all(run["deterministic_pass"] for run in variant_runs)
    judge_pass = None
    judge_reason = ""
    if not offline:
        verdicts = [run["judge_pass"] for run in variant_runs]
        judge_pass = all(verdicts) if verdicts and all(isinstance(item, bool) for item in verdicts) else False
        judge_reason = next(
            (run["judge_reason"] for run in variant_runs if run.get("judge_pass") is False),
            variant_runs[0].get("judge_reason") or "",
        )
    overall = det_pass if judge_pass is None else det_pass and bool(judge_pass)
    reason = ""
    if not det_pass:
        reason = first_failure(checks)
    elif judge_pass is False:
        reason = judge_reason or "judge failed"
    reply = "\n---\n".join(run["reply"] for run in variant_runs)
    reference = " / ".join(run["reference_resolved"] for run in variant_runs)
    sent = variant_runs[0]["sent"]
    return {
        "reply": reply,
        "sent": sent,
        "reference_resolved": reference,
        "checks": checks,
        "deterministic_pass": det_pass,
        "judge_pass": judge_pass,
        "judge_reason": judge_reason,
        "pass": overall,
        "reason": reason,
    }


def _run_variant(
    case,
    variant,
    global_rules,
    *,
    raw_business,
    business,
    clients,
    slots,
    values,
    repeat_index,
    offline,
    api_key,
    reply_model,
    judge_model,
    pacer,
) -> dict:
    setup = dict(case.get("setup") or {})
    variant_name = None
    if variant:
        variant_name = variant.get("name")
        if variant.get("service_area") == "out":
            business = _without_suburb(raw_business, "Parramatta")
            slots = _slots(business)
    case_values = dict(values)
    case_values.update(_slot_values(slots))
    if setup.get("price_job"):
        case_values["price_range"] = business.price_range_text(setup["price_job"]) or ""
    timeframe_key = setup.get("timeframe")
    if timeframe_key == "quote":
        case_values["timeframe"] = business.quote_timeframe or ""
    elif timeframe_key == "callback":
        case_values["timeframe"] = business.callback_timeframe or ""
    sender_key = setup.get("sender")
    if setup.get("confirm_client") and sender_key:
        case_values["address_on_file"] = clients[sender_key]["address"]
    reference_template = (variant or {}).get("reference") or case["reference"]
    reference = _resolve(reference_template, case_values)
    global_resolved = _resolve(global_rules, case_values)
    sent = setup.get("send") or case["customer"]
    if setup.get("photo"):
        note = setup.get("photo_note") or "photo attached"
        sent = f"{setup.get('send') or 'Can you upgrade this?'}\n[Photo: {note}]"
    sender_phone = _sender_phone(case, clients)
    snapshot = _send(
        business=business,
        clients=clients,
        setup=setup,
        text=sent,
        sender_phone=sender_phone,
        photo=bool(setup.get("photo")),
        message_id=f"eval-{case['id']}-{repeat_index}-{variant_name or 'only'}",
        offline=offline,
        api_key=api_key,
        reply_model=reply_model,
        pacer=pacer,
    )
    expected_name = None
    expected_address = None
    if setup.get("confirm_client") and sender_key:
        expected_name = clients[sender_key]["name"]
        expected_address = clients[sender_key]["address"]
    ctx = CheckContext(
        reply=snapshot["reply"] or "",
        lead=snapshot.get("lead") or {},
        business=business,
        slots=slots,
        forbidden=_secrets(sender_phone, clients),
        variant=variant_name,
        price_job=setup.get("price_job"),
        expected_name=expected_name,
        expected_address=expected_address,
    )
    checks = evaluate(case["id"], ctx)
    det_pass = passed(checks)
    judge_pass = None
    judge_reason = ""
    if not offline:
        pacer.wait()
        verdict = grade(
            api_key=api_key,
            model=judge_model,
            customer=sent,
            reply=snapshot["reply"] or "",
            reference=reference,
            rule=case["rule"],
            global_rules=global_resolved,
        )
        judge_pass = bool(verdict["pass"])
        judge_reason = verdict["reason"]
    return {
        "reply": snapshot["reply"] or "",
        "sent": sent,
        "reference_resolved": reference,
        "checks": checks,
        "deterministic_pass": det_pass,
        "judge_pass": judge_pass,
        "judge_reason": judge_reason,
    }


def _send(*, business, clients, setup, text, sender_phone, photo, message_id, offline, api_key, reply_model, pacer) -> dict:
    path = _fresh_db()
    try:
        return _send_into(
            path,
            business=business,
            clients=clients,
            setup=setup,
            text=text,
            sender_phone=sender_phone,
            photo=photo,
            message_id=message_id,
            offline=offline,
            api_key=api_key,
            reply_model=reply_model,
            pacer=pacer,
        )
    finally:
        _discard(path)


def _send_into(path, *, business, clients, setup, text, sender_phone, photo, message_id, offline, api_key, reply_model, pacer) -> dict:
    db = Database(path)
    for record in clients.values():
        db.import_client(
            name=record["name"],
            phones=[record["phone"]],
            address=record.get("address"),
            notes=record.get("notes"),
        )
    if setup.get("booked_thursday"):
        _seed_thursday(db, business, clients[setup["sender"]])
    if setup.get("prior_job") == "completed":
        _seed_completed(db, clients[setup["sender"]])
    qualifier = (
        RuleBasedQualifier(business)
        if offline or not api_key
        else _Paced(GeminiQualifier(api_key, reply_model, business=business), pacer)
    )
    intent = build_intent(
        "mock" if offline or not api_key else "gemini",
        "",
        reply_model,
        gemini_api_key="" if offline else api_key,
        gemini_model=reply_model,
    )
    inbound = Inbound(
        wa_id=sender_phone,
        text=text,
        message_type="image" if photo else "text",
        provider_message_id=message_id,
        media_id="eval-photo" if photo else None,
        mime_type="image/jpeg" if photo else None,
        source="simulator",
    )
    return handle_incoming(
        db,
        qualifier,
        SimulatedGateway(),
        LogNotifier(),
        inbound,
        business=business,
        calendar=DatabaseCalendar(db),
        now=EVAL_NOW,
        intent=intent,
    )


def _sender_phone(case: dict, clients: dict) -> str:
    sender = (case.get("setup") or {}).get("sender")
    if sender:
        phone = normalise_au_phone(clients[sender]["phone"])
        if not phone:
            raise ValueError(f"bad phone for {sender}")
        return phone
    return f"61412990{int(case['id']):03d}"


def _secrets(sender_phone: str | None, clients: dict) -> list[str]:
    sender = normalise_au_phone(sender_phone) if sender_phone else None
    found = []
    for record in clients.values():
        notes = (record.get("notes") or "").strip()
        if notes:
            found.append(notes)
        phone = normalise_au_phone(record["phone"])
        if phone != sender:
            if record.get("address"):
                found.append(record["address"])
            if record.get("phone"):
                found.append(str(record["phone"]))
    return [item for item in found if item]


def _seed_thursday(db: Database, business: Business, client: dict) -> None:
    zone = business.zone()
    start = datetime(2026, 11, 19, 9, 0, tzinfo=zone)
    end = start + timedelta(minutes=business.slot_minutes)
    _seed_job(
        db,
        client,
        description="Visit booked for Thursday",
        booking={
            "status": "booked",
            "label": format_slot(start, end),
            "start": start.isoformat(),
            "end": end.isoformat(),
            "event_id": "local-eval-thursday",
            "slots": [],
        },
        status=None,
    )


def _seed_completed(db: Database, client: dict) -> None:
    _seed_job(
        db,
        client,
        description="Fixed the lounge light last week",
        booking={},
        status="completed",
    )


def _seed_job(db: Database, client: dict, *, description: str, booking: dict, status: str | None) -> None:
    phone = normalise_au_phone(client["phone"])
    customer = db.get_customer_by_wa(phone)
    if customer is None:
        raise RuntimeError(f"missing client {client['name']}")
    conversation = db.create_conversation(
        wa_id=phone,
        profile_name=client["name"],
        source="whatsapp",
        job_source="repeat",
        customer_id=customer["id"],
        prefill={"contact_name": client["name"], "suburb": "Parramatta"},
        address=client["address"],
    )
    db.save_job(
        conversation["active_job_id"],
        {
            "job_type": "power_point",
            "suburb": "Parramatta",
            "urgency": "planned",
            "description": description,
            "preferred_time": "Thursday" if status is None else None,
            "contact_name": client["name"],
            "score": "warm",
            "score_reason": "seeded for eval",
            "handoff": 0,
            "handoff_reason": None,
            "photos_json": "[]",
            "voice_json": "[]",
            "quote_json": "{}",
            "booking_json": json.dumps(booking),
            "extracted_json": json.dumps({"escalated": False}),
        },
    )
    if status:
        db.set_job_status(conversation["active_job_id"], status)


def _slots(business: Business) -> list:
    path = _fresh_db()
    try:
        return available_slots(business, DatabaseCalendar(Database(path)), EVAL_NOW)
    finally:
        _discard(path)


class _Pacer:
    """Space Gemini calls so a free-tier limit is not spent in one burst."""

    def __init__(self, rpm: int | None) -> None:
        self.gap = 0.0 if not rpm or rpm <= 0 else 60.0 / rpm
        self._next = 0.0

    def wait(self) -> None:
        if self.gap <= 0:
            return
        now = time.monotonic()
        if now < self._next:
            time.sleep(self._next - now)
        self._next = time.monotonic() + self.gap


class _Paced:
    def __init__(self, inner, pacer: _Pacer) -> None:
        self.inner = inner
        self.pacer = pacer

    def qualify(self, turn):
        self.pacer.wait()
        return self.inner.qualify(turn)


def _discard(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


def _fresh_db() -> str:
    from tempfile import mkstemp

    handle, path = mkstemp(prefix="eval-", suffix=".db")
    os.close(handle)
    return path


def _without_suburb(raw: dict, suburb: str) -> Business:
    data = json.loads(json.dumps(raw))
    data["service_area_suburbs"] = [
        item for item in data["service_area_suburbs"] if item.casefold() != suburb.casefold()
    ]
    return parse_business(data)


def _base_values(business: Business, slots: list) -> dict:
    values = {
        "owner_name": business.owner_name or "",
        "business_name": business.name,
        "licence_no": business.licence_no or "",
        "callout_fee": business.callout_fee or "",
        "callout_inclusions": business.callout_inclusions or "",
        "after_hours_rate": business.after_hours_rate or "",
        "timeframe": business.callback_timeframe or business.quote_timeframe or "",
    }
    values.update(_slot_values(slots))
    return values


def _slot_values(slots: list) -> dict:
    labels = [slot.label for slot in slots]
    return {
        "slot": labels[0] if labels else "",
        "slot_1": labels[0] if labels else "",
        "slot_2": labels[1] if len(labels) > 1 else "",
    }


def _resolve(template: str, values: dict) -> str:
    def repl(match: re.Match) -> str:
        key = match.group(1)
        value = values.get(key)
        if value is None or value == "":
            return match.group(0)
        return str(value)

    return _PLACEHOLDER.sub(repl, template or "")


def _majority(values: list[bool]) -> bool | None:
    if not values:
        return None
    yes = sum(1 for value in values if value)
    no = len(values) - yes
    if yes > no:
        return True
    if no > yes:
        return False
    return None


def _write_results(report: dict, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(ZoneInfo("UTC")).strftime("%Y%m%dT%H%M%SZ")
    body = json.dumps(report, indent=2, ensure_ascii=False)
    markdown = render_markdown(report)
    for name, payload in (
        (f"{stamp}.json", body),
        ("latest.json", body),
    ):
        (directory / name).write_text(payload + "\n", encoding="utf-8")
    for name, payload in (
        (f"{stamp}.md", markdown),
        ("latest.md", markdown),
    ):
        (directory / name).write_text(payload, encoding="utf-8")


def _cell(value: str) -> str:
    return (value or "").replace("|", "/").replace("\n", " ")
