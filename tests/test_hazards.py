"""Electrical hazards are caught without the model, and ordinary jobs are not."""

import pytest

from app.business import load_business
from app.extract import extract_job_type, extract_urgency
from app.hazards import classify_hazard
from app.policy import apply_policy, is_safety_message
from tests.test_policy import _incoming

# (message, kind, tier)
HAZARDS = [
    # Fallen / damaged wires and powerlines
    ("my wire looke broken and someone might step on i. i want it fixed now", "wire", "emergency"),
    ("live wire hanging nearby", "wire", "emergency"),
    ("tree fell on my wire", "wire", "emergency"),
    ("branch came down on the wires to the house", "wire", "emergency"),
    ("there's a cable hanging out of the wall in the hallway", "wire", "emergency"),
    ("exposed wires in the ceiling after the storm", "wire", "emergency"),
    ("tree fell on pworline", "powerline", "emergency"),
    ("A powerline is down in our street after the storm", "powerline", "emergency"),
    ("the power pole out front is leaning after a car hit it", "powerline", "emergency"),
    ("service wire from the pole snapped", "powerline", "emergency"),
    # A person in contact with power
    ("my son is stuck to the heater and can't let go", "person_down", "emergency"),
    ("dad collapsed after touching the power point", "person_down", "emergency"),
    # Shocks and tingles
    ("I got a shock off the kitchen tap", "shock", "emergency"),
    ("I got a tingle from the shower", "shock", "emergency"),
    ("got zapped by the light switch", "shock", "emergency"),
    # Fire
    ("There's a burning smell coming from my switchboard", "fire", "emergency"),
    ("sparks came out of the socket", "fire", "emergency"),
    ("the oven caught fire", "fire", "emergency"),
    # Urgent but not (yet) a 000 call
    ("the power point is melting", "overheating", "urgent"),
    ("light switch is hot to touch and buzzing", "overheating", "urgent"),
    ("there's scorch marks on the power point", "overheating", "urgent"),
    ("water dripping from the ceiling light", "water", "urgent"),
    ("the house flooded and the power is back on", "water", "urgent"),
    ("water leaking into the switchboard", "water", "urgent"),
    ("solar battery is swelling and hissing", "battery", "urgent"),
]

ORDINARY = [
    "rewire my garage",
    "need a cable run for the new tv",
    "nbn cable cut by the landscaper",
    "can you run a wire down the wall for a new light",
    "hot water system not working",
    "smoke alarm beeping",
    "I was shocked by the invoice",
    "static shock from the carpet",
    "install fire rated downlights",
    "need a new power point near the kitchen sink",
    "the fence wire is broken",
    "Another sparky quoted $200 less. Can you match it?",
    "Just tell me which wire goes where and I'll swap the power point myself",
    "want to install a solar battery",
    "Safety switch keeps tripping, how do I bypass it?",
]


@pytest.mark.parametrize("text,kind,tier", HAZARDS)
def test_hazard_is_classified(text, kind, tier):
    hazard = classify_hazard(text)
    assert hazard is not None, text
    assert (hazard.kind, hazard.tier) == (kind, tier)
    assert is_safety_message(text)
    assert extract_urgency(text) == tier
    decision = apply_policy(text, load_business())
    assert decision.escalation_reason == "danger"
    assert decision.ensure_000 is (tier == "emergency")
    assert "000" in decision.replacement


@pytest.mark.parametrize("text", ORDINARY)
def test_ordinary_job_is_not_a_hazard(text):
    assert classify_hazard(text) is None
    assert not is_safety_message(text)
    decision = apply_policy(text, load_business())
    assert not decision.ensure_000
    assert decision.escalation_reason != "danger"


def test_whole_street_outage_is_redirected_not_escalated():
    text = "Power's out in the whole street"
    assert classify_hazard(text).kind == "street_outage"
    assert not is_safety_message(text)
    decision = apply_policy(text, load_business())
    assert "distributor" in decision.replacement
    assert not decision.escalated


@pytest.mark.parametrize(
    "index,text",
    list(enumerate(
        [
            "my wire looke broken and someone might step on i. i want it fixed now",
            "live wire hanging nearby",
            "tree fell on my wire",
        ]
    )),
)
def test_wire_hazard_reply_end_to_end(tmp_path, index, text):
    result = _incoming(tmp_path, load_business(), text, f"6141299600{index}")
    reply = result["reply"]
    assert "call 000" in reply.lower()
    assert "8 metres" in reply
    assert "distributor" in reply
    assert "once it's safe" in reply
    lead = result["lead"]
    assert lead["urgency"] == "emergency"
    assert lead["job_type"] == "fault"
    assert lead["suburb"] in (None, "")


def test_misspelled_powerline_is_not_read_as_a_suburb(tmp_path):
    result = _incoming(tmp_path, load_business(), "tree fell on pworline", "61412996009")
    reply = result["reply"]
    assert "8 metres" in reply and "000" in reply
    assert "outside the area" not in reply
    assert result["lead"]["suburb"] in (None, "")


def test_urgent_hazard_does_not_force_call_000_now(tmp_path):
    result = _incoming(tmp_path, load_business(), "the power point is melting", "61412996010")
    reply = result["reply"]
    assert not reply.lower().startswith("call 000 now")
    assert "stop using it" in reply.lower()
    assert result["lead"]["urgency"] == "urgent"


def test_hazard_with_no_job_word_is_a_fault_job():
    assert extract_job_type("live wire hanging nearby", load_business()) == "fault"
