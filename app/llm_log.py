"""Record what the model did on each turn, and read those records back.

Every customer message the bot answers writes one row to `llm_calls`: who
answered (Gemini, Claude, or the offline stand-in), which model, how long it
took, tokens, every fallback attempt, the raw tool output, the reply the model
wrote, the reply actually sent, the keyword and model hazard verdicts, and
which guardrail rule (if any) replaced the reply. The system prompt is stored
once in `prompts`, keyed by hash, so a turn can be replayed against a new
prompt. One structured log line per turn carries the same summary without
phone numbers or message text.

Writing a record never breaks a reply: any error is logged and swallowed.

    python -m app.llm_log summary            # counts, fallback rate, latency, tokens, disagreements
    python -m app.llm_log export calls.jsonl # one JSON object per turn, for eval cases
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import statistics
import sys
from collections import Counter

from app.llm import QUALIFY_TOOL
from app.telemetry import current_span, set_attributes

logger = logging.getLogger("tradie.llm_calls")


def text_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]


def prompt_version(base_prompt: str) -> str:
    """A short id for the prompt template plus the tool schema.

    `base_prompt` is the system prompt built with no customer facts, so it changes
    only when the template, the business config, or the tool definition changes.
    """
    material = base_prompt + "\n" + json.dumps(QUALIFY_TOOL, sort_keys=True)
    return text_hash(material)[:12]


def record_turn(
    db,
    *,
    turn_id: str,
    conversation_id: int | None,
    job_id: int | None,
    inbound_message_id: int | None,
    system_prompt: str,
    version: str | None,
    history_len: int,
    draft=None,
    latency_ms: int | None = None,
    decision=None,
    rule_hazard=None,
    sent_reply: str | None = None,
    urgency: str | None = None,
    handoff: bool = False,
    error: str | None = None,
) -> int | None:
    """Store one turn and log a one-line summary. Returns the row id, or None if storing failed."""
    try:
        meta = getattr(draft, "meta", None)
        status = "error" if error else (getattr(meta, "status", None) or ("ok" if draft else "error"))
        hazard_final = getattr(getattr(decision, "hazard", None), "kind", None)
        values = {
            "turn_id": turn_id,
            "conversation_id": conversation_id,
            "job_id": job_id,
            "inbound_message_id": inbound_message_id,
            "purpose": "qualify",
            "provider": getattr(meta, "provider", None) or ("unknown" if draft else None),
            "model": getattr(meta, "model", None),
            "status": status,
            "fallback_from": getattr(meta, "fallback_from", None),
            "attempts_json": json.dumps(getattr(meta, "attempts", None) or []),
            "latency_ms": latency_ms,
            "input_tokens": getattr(meta, "input_tokens", None),
            "output_tokens": getattr(meta, "output_tokens", None),
            "prompt_version": version,
            "system_hash": text_hash(system_prompt) if system_prompt else None,
            "history_len": history_len,
            "raw_output_json": _json_or_none(getattr(draft, "raw", None)),
            "model_reply": getattr(draft, "reply", None),
            "sent_reply": sent_reply,
            "hazard_rules": getattr(rule_hazard, "kind", None),
            "hazard_model": getattr(draft, "hazard", None),
            "hazard_final": hazard_final,
            "policy_rule": getattr(decision, "rule", None),
            "urgency": urgency,
            "escalation_reason": getattr(decision, "escalation_reason", None),
            "handoff": handoff,
            "error": (error or "")[:500] or None,
        }
        row_id = db.record_llm_call(values, system_prompt=system_prompt)
    except Exception:
        logger.exception("could not record llm call for turn %s", turn_id)
        return None
    summary = {
        key: values[key]
        for key in (
            "turn_id", "conversation_id", "job_id", "provider", "model", "status", "fallback_from",
            "latency_ms", "input_tokens", "output_tokens", "prompt_version", "hazard_rules",
            "hazard_model", "hazard_final", "policy_rule", "urgency", "escalation_reason", "handoff",
        )
    }
    logger.info("turn %s", json.dumps(summary, separators=(",", ":")))
    # The same summary on the trace, so a trace and its llm_calls row share turn_id.
    set_attributes(current_span(), {f"tradie.{key}": value for key, value in summary.items()})
    return row_id


def summarise(calls: list[dict]) -> dict:
    """Numbers worth watching: volume, who answered, fallbacks, latency, tokens, disagreements."""
    total = len(calls)
    latencies = sorted(c["latency_ms"] for c in calls if isinstance(c.get("latency_ms"), int))
    disagreements = [
        c for c in calls if (c.get("hazard_rules") or None) != (c.get("hazard_model") or None)
        and c.get("provider") not in (None, "offline")
    ]
    return {
        "turns": total,
        "by_provider": dict(Counter(c.get("provider") or "none" for c in calls)),
        "by_model": dict(Counter(c.get("model") or "none" for c in calls)),
        "by_status": dict(Counter(c.get("status") for c in calls)),
        "fallback_rate": round(sum(c.get("status") == "fallback" for c in calls) / total, 3) if total else 0.0,
        "latency_ms_p50": _percentile(latencies, 50),
        "latency_ms_p95": _percentile(latencies, 95),
        "input_tokens": sum(c.get("input_tokens") or 0 for c in calls),
        "output_tokens": sum(c.get("output_tokens") or 0 for c in calls),
        "hazards": dict(Counter(c.get("hazard_final") for c in calls if c.get("hazard_final"))),
        "policy_rules": dict(Counter(c.get("policy_rule") for c in calls if c.get("policy_rule"))),
        "hazard_disagreements": len(disagreements),
        "prompt_versions": dict(Counter(c.get("prompt_version") or "none" for c in calls)),
    }


def export_rows(calls: list[dict]) -> list[dict]:
    """One eval-ready object per turn: what the customer said, what the model did, what was sent."""
    rows = []
    for call in calls:
        rows.append(
            {
                "turn_id": call.get("turn_id"),
                "created_at": call.get("created_at"),
                "conversation_id": call.get("conversation_id"),
                "customer": call.get("inbound_text"),
                "model_reply": call.get("model_reply"),
                "sent_reply": call.get("sent_reply"),
                "provider": call.get("provider"),
                "model": call.get("model"),
                "status": call.get("status"),
                "prompt_version": call.get("prompt_version"),
                "hazard_rules": call.get("hazard_rules"),
                "hazard_model": call.get("hazard_model"),
                "hazard_final": call.get("hazard_final"),
                "policy_rule": call.get("policy_rule"),
                "urgency": call.get("urgency"),
                "raw_output": _loads(call.get("raw_output_json")),
                "attempts": _loads(call.get("attempts_json")) or [],
                "latency_ms": call.get("latency_ms"),
                # Filled in by a person during error analysis.
                "label": None,
                "note": None,
            }
        )
    return rows


def _percentile(sorted_values: list[int], pct: int) -> int | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    cuts = statistics.quantiles(sorted_values, n=100, method="inclusive")
    return int(round(cuts[pct - 1]))


def _json_or_none(value) -> str | None:
    if value is None:
        return None
    try:
        return json.dumps(value, default=str)
    except (TypeError, ValueError):
        return None


def _loads(value):
    if not value:
        return None
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return None


def main(argv: list[str] | None = None) -> int:
    from app.config import Settings
    from app.db import Database

    parser = argparse.ArgumentParser(prog="python -m app.llm_log", description=__doc__.split("\n\n")[0])
    parser.add_argument("--db", help="SQLite path (default: SQLITE_PATH)")
    parser.add_argument("--since", help='Only calls at or after this time, e.g. "2026-11-01"')
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("summary", help="Print counts, fallback rate, latency, tokens, disagreements")
    export = sub.add_parser("export", help="Write one JSON object per turn")
    export.add_argument("out", help="Output .jsonl path, or - for stdout")
    args = parser.parse_args(argv)

    path = args.db or Settings.from_env().sqlite_path
    calls = Database(path).list_llm_calls(since=args.since)
    if args.command == "summary":
        print(json.dumps(summarise(calls), indent=2))
        return 0
    rows = export_rows(calls)
    handle = sys.stdout if args.out == "-" else open(args.out, "w", encoding="utf-8")
    try:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    finally:
        if handle is not sys.stdout:
            handle.close()
    if args.out != "-":
        print(f"Wrote {len(rows)} turns to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
