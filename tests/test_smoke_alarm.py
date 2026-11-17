"""A beeping smoke alarm is a job, not a fire. Real smoke still gets the 000 reply."""

import pytest

from app.business import load_business
from app.extract import extract_job_type, extract_urgency
from app.policy import apply_policy, is_safety_message
from tests.test_policy import _incoming

ALARM_JOBS = [
    "smoke alarm beeping",
    "my smoke detector keeps chirping",
    "smoke alarms need replacing",
    "My smoke detectors keep beeping every minute",
    "Smoke alarm is chirping, think the battery is low",
]

HAZARDS = [
    "there's smoke coming from the switchboard",
    "my smoke alarm is going off and I can smell burning",
    "Smoke alarm went off and there's smoke in the kitchen",
]


@pytest.mark.parametrize("text", ALARM_JOBS)
def test_beeping_smoke_alarm_is_not_a_hazard(text):
    business = load_business()
    assert not is_safety_message(text)
    assert extract_urgency(text) != "emergency"
    assert extract_job_type(text, business) == "smoke_alarm"
    decision = apply_policy(text, business)
    assert not decision.ensure_000
    assert decision.escalation_reason != "danger"
    assert "000" not in (decision.replacement or "")


@pytest.mark.parametrize("text", ALARM_JOBS)
def test_beeping_smoke_alarm_is_recorded_as_a_job_without_000(tmp_path, text):
    wa_id = f"61412997{ALARM_JOBS.index(text):04d}"
    result = _incoming(tmp_path, load_business(), text, wa_id)
    assert "000" not in result["reply"]
    assert result["lead"]["job_type"] == "smoke_alarm"
    assert result["lead"]["urgency"] != "emergency"


@pytest.mark.parametrize("text", HAZARDS)
def test_real_smoke_still_gets_the_safety_reply(tmp_path, text):
    business = load_business()
    assert is_safety_message(text)
    assert extract_urgency(text) == "emergency"
    decision = apply_policy(text, business)
    assert decision.ensure_000
    assert decision.escalation_reason == "danger"
    wa_id = f"61412998{HAZARDS.index(text):04d}"
    reply = _incoming(tmp_path, business, text, wa_id)["reply"]
    assert "call 000" in reply.lower()


SOUNDING = [
    "fire alarma keeps activating",
    "fire alarm keeps going off",
    "My smoke alarm is going off",
    "smoke alarm keeps going off for no reason",
]


@pytest.mark.parametrize("text", SOUNDING)
def test_sounding_alarm_gets_the_alarm_script_not_the_switchboard_fire_script(text):
    business = load_business()
    decision = apply_policy(text, business)
    assert decision.hazard.kind == "alarm"
    reply = decision.replacement
    assert "call 000" in reply  # any sign of fire: get out and call 000
    assert "fire brigade" in reply  # answers "you or the fire department?"
    assert "keep clear of the board" not in reply
    assert not decision.ensure_000  # no unconditional "Call 000 now"
    assert decision.escalation_reason == "danger"
    assert extract_urgency(text) == "urgent"
    assert extract_job_type(text, business) == "smoke_alarm"


@pytest.mark.parametrize("text", ["fire alarm beeping", "fire alarms need replacing"])
def test_beeping_fire_alarm_is_an_ordinary_job(text):
    assert not is_safety_message(text)
    assert extract_job_type(text, load_business()) == "smoke_alarm"


def test_model_reading_fire_alarm_as_a_fire_gets_the_alarm_script():
    decision = apply_policy("fire alarma keeps activating", load_business(), model_hazard="fire")
    assert decision.hazard.kind == "alarm"
    assert "fire brigade" in decision.replacement


def test_model_fire_with_real_burning_keeps_the_fire_script():
    decision = apply_policy(
        "fire alarm going off and I can smell burning", load_business(), model_hazard="fire"
    )
    assert decision.hazard.kind == "fire"
    assert decision.ensure_000


def test_live_conversation_fire_alarma_keeps_activating(tmp_path):
    """The conversation from the live app, with the model saying hazard=fire as it did there."""
    from tests.test_model_hazard import FakeQualifier, _send

    result, notifier = _send(tmp_path, "fire alarma keeps activating", FakeQualifier("fire"))
    reply = result["reply"]
    assert "fire brigade" in reply
    assert "keep clear of the board" not in reply
    assert not reply.lower().startswith("call 000 now")
    assert result["lead"]["urgency"] == "urgent"
    assert notifier.events


def test_fire_brigade_or_electrician_question_gets_a_clear_answer():
    text = "should it be you or fire departjnent that investigates? how to decide"
    decision = apply_policy(text, load_business())
    reply = decision.replacement
    assert "fire brigade on 000" in reply
    assert "faulty alarm" in reply
    assert not decision.ensure_000
    assert not decision.escalated
