"""The model's `hazard` field. Either the rules or the model can raise a hazard; the model cannot lower one.

No real Gemini or Anthropic calls: a fake qualifier returns a fixed draft.
"""

import logging
from types import SimpleNamespace

import pytest

from app.business import load_business
from app.db import Database
from app.hazards import MODEL_HAZARDS
from app.llm import QUALIFY_TOOL, QualificationDraft, _gemini_tool, parse_gemini_qualification, parse_qualification
from app.messaging import Inbound, SimulatedGateway
from app.notify import ListNotifier
from app.policy import apply_policy, resolve_hazard
from app.service import handle_incoming
from evals.harness import EVAL_NOW


class FakeQualifier:
    """Stands in for Gemini or Claude. Returns a plain intake reply plus a hazard."""

    def __init__(self, hazard, urgency=None):
        self.hazard = hazard
        self.urgency = urgency

    def qualify(self, turn):
        return QualificationDraft(
            reply="Thanks. Which suburb is the job in?",
            hazard=self.hazard,
            urgency=self.urgency,
            raw={"hazard": self.hazard},
        )


def _send(tmp_path, text, qualifier, wa_id="61412997700"):
    notifier = ListNotifier()
    result = handle_incoming(
        Database(str(tmp_path / "model-hazard.db")),
        qualifier,
        SimulatedGateway(),
        notifier,
        Inbound(wa_id=wa_id, text=text, message_type="text", provider_message_id=f"mh-{wa_id}", source="simulator"),
        business=load_business(),
        now=EVAL_NOW,
    )
    return result, notifier


# --- schema -----------------------------------------------------------------


def test_hazard_is_a_required_enum_in_the_tool_schema():
    schema = QUALIFY_TOOL["input_schema"]
    assert "hazard" in schema["required"]
    assert schema["properties"]["hazard"]["enum"] == list(MODEL_HAZARDS)
    assert set(MODEL_HAZARDS) == {
        "none", "wire", "powerline", "shock", "person_down", "fire", "alarm",
        "water", "overheating", "battery", "street_outage",
    }


def test_gemini_declaration_carries_the_hazard_enum():
    declaration = _gemini_tool().function_declarations[0]
    schema = declaration.parameters_json_schema
    assert "hazard" in schema["required"]
    assert schema["properties"]["hazard"]["enum"] == list(MODEL_HAZARDS)


# --- parsing ----------------------------------------------------------------


def _args(hazard):
    return {
        "reply": "Which suburb?", "request_human": False, "job_type": None, "suburb": None,
        "urgency": None, "hazard": hazard, "description": None, "preferred_time": None,
        "contact_name": None,
    }


@pytest.mark.parametrize(
    "value,expected",
    [("wire", "wire"), ("FIRE", "fire"), (" water ", "water"), ("none", None), (None, None),
     ("lava", None), ("life_support", "life_support"), (3, None)],
)
def test_hazard_is_parsed_from_claude_and_gemini(value, expected):
    claude = parse_qualification(
        SimpleNamespace(content=[SimpleNamespace(type="tool_use", name="record_job_intake", input=_args(value))])
    )
    gemini = parse_gemini_qualification(
        SimpleNamespace(
            function_calls=[SimpleNamespace(name="record_job_intake", args=_args(value))],
            text=None,
            candidates=[],
        )
    )
    assert claude.hazard == expected
    assert gemini.hazard == expected


def test_old_tool_output_without_hazard_still_parses():
    args = _args(None)
    del args["hazard"]
    draft = parse_qualification(
        SimpleNamespace(content=[SimpleNamespace(type="tool_use", name="record_job_intake", input=args)])
    )
    assert draft.hazard is None


# --- either can raise, the model cannot lower --------------------------------


def test_model_catches_a_hazard_the_rules_miss(tmp_path):
    text = "the thing from the pole to my roof is lying across the driveway"
    assert resolve_hazard(text, None) is None  # rules alone miss this phrasing
    result, notifier = _send(tmp_path, text, FakeQualifier("powerline"))
    reply = result["reply"]
    assert "8 metres" in reply and "000" in reply  # the vetted script, not model prose
    assert "Which suburb" not in reply
    assert result["lead"]["urgency"] == "emergency"
    assert notifier.events, "owner is alerted"


def test_model_cannot_lower_a_hazard_the_rules_found(tmp_path):
    result, _ = _send(tmp_path, "live wire hanging nearby", FakeQualifier("none", urgency="planned"))
    reply = result["reply"]
    assert "Treat that wire as live" in reply
    assert "call 000" in reply.lower()
    assert result["lead"]["urgency"] == "emergency"


def test_rules_pick_the_script_when_both_flag_different_kinds():
    decision = apply_policy("there's smoke coming from the switchboard", load_business(), model_hazard="water")
    assert decision.hazard.kind == "fire"
    assert decision.ensure_000


def test_model_urgent_hazard_gives_urgent_script_without_forcing_000(tmp_path):
    text = "the outlet in the bathroom looks wet inside"
    result, _ = _send(tmp_path, text, FakeQualifier("water"), wa_id="61412997701")
    reply = result["reply"]
    assert "don't touch the switchboard" in reply.lower()
    assert not reply.lower().startswith("call 000 now")
    assert result["lead"]["urgency"] == "urgent"


def test_model_street_outage_redirects_without_escalating():
    decision = apply_policy("nobody around here has electricity", load_business(), model_hazard="street_outage")
    assert "distributor" in decision.replacement
    assert not decision.escalated
    assert not decision.is_hazard


def test_no_hazard_from_either_side_leaves_the_reply_alone(tmp_path):
    result, _ = _send(tmp_path, "need a new power point in the garage", FakeQualifier(None), wa_id="61412997702")
    assert result["reply"].startswith("Thanks. Which suburb")
    assert "000" not in result["reply"]


def test_disagreement_is_logged(caplog):
    with caplog.at_level(logging.INFO, logger="tradie.policy"):
        apply_policy("the thing from the pole is on the driveway", load_business(), model_hazard="powerline")
    assert "hazard disagreement rules=none model=powerline" in caplog.text


def test_agreement_is_not_logged(caplog):
    with caplog.at_level(logging.INFO, logger="tradie.policy"):
        apply_policy("live wire hanging nearby", load_business(), model_hazard="wire")
    assert "hazard disagreement" not in caplog.text
