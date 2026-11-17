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
    "My smoke alarm is going off",
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
