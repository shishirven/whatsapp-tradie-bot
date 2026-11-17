"""Every turn is recorded in llm_calls and traced, without real Gemini or Anthropic calls."""

import json
import logging
from types import SimpleNamespace

import pytest

from app.business import load_business
from app.db import Database
from app.llm import AnthropicQualifier, GeminiQualifier, RuleBasedQualifier
from app.llm_log import export_rows, main as llm_log_main, summarise
from app.messaging import Inbound, SimulatedGateway
from app.notify import ListNotifier
from app.service import handle_incoming
from evals.harness import EVAL_NOW

PHONE = "61412998800"


def _send(db, qualifier, text, *, wa_id=PHONE, message_id=None):
    return handle_incoming(
        db,
        qualifier,
        SimulatedGateway(),
        ListNotifier(),
        Inbound(
            wa_id=wa_id,
            text=text,
            message_type="text",
            provider_message_id=message_id or f"llm-{abs(hash(text)) % 10**8}",
            source="simulator",
        ),
        business=load_business(),
        now=EVAL_NOW,
    )


@pytest.fixture
def db(tmp_path):
    return Database(str(tmp_path / "calls.db"))


def _args(**overrides):
    args = {
        "reply": "Which suburb is the job in?", "request_human": False, "job_type": "power_point",
        "suburb": None, "urgency": None, "hazard": "none", "description": None,
        "preferred_time": None, "contact_name": None,
    }
    args.update(overrides)
    return args


def _gemini_response(args, prompt_tokens=812, output_tokens=64):
    call = SimpleNamespace(name="record_job_intake", args=args)
    return SimpleNamespace(
        function_calls=[call], text=None, candidates=[],
        usage_metadata=SimpleNamespace(prompt_token_count=prompt_tokens, candidates_token_count=output_tokens),
    )


class _GeminiModels:
    def __init__(self, outcomes):
        self.outcomes = dict(outcomes)
        self.called = []

    def generate_content(self, **kwargs):
        model = kwargs["model"]
        self.called.append(model)
        outcome = self.outcomes[model]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _error(code):
    error = Exception(f"{code} UNAVAILABLE")
    error.code = code
    return error


def _gemini(outcomes, business=None):
    models = _GeminiModels(outcomes)
    qualifier = GeminiQualifier(
        "test-key",
        "gemini-3.8-flash",
        client=SimpleNamespace(models=models),
        business=business or load_business(),
        fallback_models=["gemini-3.5-flash-lite"],
    )
    return qualifier, models


# --- what one turn records ----------------------------------------------------


def test_offline_turn_is_recorded_with_hazards_rule_and_sent_reply(db):
    result = _send(db, RuleBasedQualifier(load_business()), "live wire hanging nearby")
    [call] = db.list_llm_calls()
    assert call["provider"] == "offline"
    assert call["status"] == "ok"
    assert call["inbound_text"] == "live wire hanging nearby"
    assert call["sent_reply"] == result["reply"]
    assert call["model_reply"]  # what the qualifier wrote before guardrails
    assert call["hazard_rules"] == "wire"
    assert call["hazard_model"] == "wire"
    assert call["hazard_final"] == "wire"
    assert call["policy_rule"] == "safety:wire"
    assert call["urgency"] == "emergency"
    assert call["escalation_reason"] == "danger"
    assert call["conversation_id"] and call["job_id"] and call["inbound_message_id"]
    assert isinstance(call["latency_ms"], int)
    assert len(call["turn_id"]) == 16
    # The exact system prompt is stored once and can be fetched for a replay.
    assert "Northside Electrical" in db.get_prompt(call["system_hash"])


def test_prompt_version_is_stable_across_turns_and_customers(db):
    _send(db, RuleBasedQualifier(load_business()), "need a power point", wa_id="61412998801")
    _send(db, RuleBasedQualifier(load_business()), "my name is Sam, in Newtown", wa_id="61412998802")
    versions = {call["prompt_version"] for call in db.list_llm_calls()}
    assert len(versions) == 1 and None not in versions


def test_gemini_success_records_model_tokens_and_attempts(db):
    qualifier, _ = _gemini({"gemini-3.8-flash": _gemini_response(_args())})
    _send(db, qualifier, "need a new power point")
    [call] = db.list_llm_calls()
    assert (call["provider"], call["model"], call["status"]) == ("gemini", "gemini-3.8-flash", "ok")
    assert (call["input_tokens"], call["output_tokens"]) == (812, 64)
    assert call["hazard_model"] is None
    attempts = json.loads(call["attempts_json"])
    assert [a["model"] for a in attempts] == ["gemini-3.8-flash"]
    assert json.loads(call["raw_output_json"])["job_type"] == "power_point"


def test_gemini_second_model_answers_after_a_503(db):
    qualifier, models = _gemini(
        {"gemini-3.8-flash": _error(503), "gemini-3.5-flash-lite": _gemini_response(_args())}
    )
    _send(db, qualifier, "need a new power point")
    [call] = db.list_llm_calls()
    assert models.called == ["gemini-3.8-flash", "gemini-3.5-flash-lite"]
    assert call["model"] == "gemini-3.5-flash-lite"
    attempts = json.loads(call["attempts_json"])
    assert attempts[0]["error"].startswith("503")
    assert "error" not in attempts[1]


def test_every_gemini_model_failing_is_recorded_as_a_fallback(db):
    qualifier, _ = _gemini({"gemini-3.8-flash": _error(503), "gemini-3.5-flash-lite": _error(429)})
    _send(db, qualifier, "need a new power point")
    [call] = db.list_llm_calls()
    assert call["status"] == "fallback"
    assert call["provider"] == "offline"
    assert call["fallback_from"] == "gemini"
    assert [a["error"][:3] for a in json.loads(call["attempts_json"])] == ["503", "429"]


def test_model_hazard_disagreement_is_visible_in_the_row(db):
    text = "the thing from the pole to my roof is lying across the driveway"
    qualifier, _ = _gemini({"gemini-3.8-flash": _gemini_response(_args(hazard="powerline", job_type=None))})
    _send(db, qualifier, text)
    [call] = db.list_llm_calls()
    assert call["hazard_rules"] is None
    assert call["hazard_model"] == "powerline"
    assert call["hazard_final"] == "powerline"
    assert summarise([call])["hazard_disagreements"] == 1


def test_anthropic_records_tokens_and_served_model(db):
    response = SimpleNamespace(
        content=[SimpleNamespace(type="tool_use", name="record_job_intake", input=_args())],
        model="claude-sonnet-5-5-20260901",
        usage=SimpleNamespace(input_tokens=1500, output_tokens=90),
    )
    client = SimpleNamespace(messages=SimpleNamespace(create=lambda **kwargs: response))
    _send(db, AnthropicQualifier("test-key", "claude-sonnet-5-5", client=client), "need a power point")
    [call] = db.list_llm_calls()
    assert (call["provider"], call["model"]) == ("anthropic", "claude-sonnet-5-5-20260901")
    assert (call["input_tokens"], call["output_tokens"]) == (1500, 90)


def test_qualifier_crash_is_recorded_as_an_error(db):
    class Broken:
        def qualify(self, turn):
            raise RuntimeError("socket closed")

    result = _send(db, Broken(), "need a power point")
    [call] = db.list_llm_calls()
    assert call["status"] == "error"
    assert "socket closed" in call["error"]
    assert call["sent_reply"] == result["reply"]


def test_a_failing_record_never_breaks_the_reply(db, monkeypatch, caplog):
    def explode(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(db, "record_llm_call", explode)
    with caplog.at_level(logging.ERROR, logger="tradie.llm_calls"):
        result = _send(db, RuleBasedQualifier(load_business()), "need a power point")
    assert result["reply"]
    assert "could not record llm call" in caplog.text


def test_turn_log_line_has_no_phone_number_or_message_text(db, caplog):
    with caplog.at_level(logging.INFO, logger="tradie.llm_calls"):
        _send(db, RuleBasedQualifier(load_business()), "secret words about my kitchen")
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("turn ")]
    assert len(lines) == 1
    summary = json.loads(lines[0][len("turn "):])
    assert summary["provider"] == "offline" and summary["turn_id"]
    assert PHONE not in lines[0]
    assert "kitchen" not in lines[0]


# --- reading the records back -------------------------------------------------


def test_summary_and_export(db, tmp_path, capsys):
    qualifier = RuleBasedQualifier(load_business())
    _send(db, qualifier, "need a power point", wa_id="61412998803")
    _send(db, qualifier, "live wire hanging nearby", wa_id="61412998804")
    calls = db.list_llm_calls()
    stats = summarise(calls)
    assert stats["turns"] == 2
    assert stats["by_provider"] == {"offline": 2}
    assert stats["hazards"] == {"wire": 1}
    assert stats["policy_rules"] == {"safety:wire": 1}
    assert stats["latency_ms_p50"] is not None

    rows = export_rows(calls)
    assert rows[1]["customer"] == "live wire hanging nearby"
    assert rows[1]["label"] is None  # filled in during error analysis

    out = tmp_path / "calls.jsonl"
    assert llm_log_main(["--db", db.path, "export", str(out)]) == 0
    assert len(out.read_text().splitlines()) == 2
    assert llm_log_main(["--db", db.path, "summary"]) == 0
    assert '"turns": 2' in capsys.readouterr().out


# --- OpenTelemetry --------------------------------------------------------------


@pytest.fixture(scope="module")
def spans():
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    provider = trace.get_tracer_provider()
    if not isinstance(provider, TracerProvider):
        provider = TracerProvider()
        trace.set_tracer_provider(provider)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return exporter


def test_turn_and_model_call_are_traced_with_genai_attributes(db, spans):
    spans.clear()
    qualifier, _ = _gemini({"gemini-3.8-flash": _gemini_response(_args())})
    _send(db, qualifier, "my kitchen power point is dead", wa_id="61412998805")
    by_name = {span.name: span for span in spans.get_finished_spans()}
    turn = by_name["whatsapp.turn"]
    chat = by_name["chat gemini-3.8-flash"]
    assert chat.parent.span_id == turn.context.span_id
    assert chat.attributes["gen_ai.operation.name"] == "chat"
    assert chat.attributes["gen_ai.provider.name"] == "gcp.gemini"
    assert chat.attributes["gen_ai.response.model"] == "gemini-3.8-flash"
    assert chat.attributes["gen_ai.usage.input_tokens"] == 812
    [call] = db.list_llm_calls()
    assert turn.attributes["tradie.turn_id"] == call["turn_id"]  # trace <-> llm_calls row
    everything = json.dumps({k: str(v) for s in (turn, chat) for k, v in s.attributes.items()})
    assert "61412998805" not in everything
    assert "kitchen" not in everything


def test_tracing_stays_off_without_an_endpoint(monkeypatch):
    from app import telemetry

    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", raising=False)
    monkeypatch.setattr(telemetry, "_configured", False)
    assert telemetry.setup_tracing() is False
