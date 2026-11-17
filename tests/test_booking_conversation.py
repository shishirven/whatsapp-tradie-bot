"""Replays a live Gemini conversation from the simulator (17 Nov 2026) with a fake model.

The fake returns what Gemini returned: a preferred time with the dash written as the
literal text "\\u2013", a reply that restates the booked time, and a lowercase name.
"""

from app.business import load_business
from app.db import Database
from types import SimpleNamespace

from app.llm import parse_gemini_qualification
from app.messaging import Inbound, SimulatedGateway
from app.notify import ListNotifier
from app.service import handle_incoming
from evals.harness import EVAL_NOW

WA = "61412995700"


class ScriptedGemini:
    def __init__(self, turns):
        self.turns = list(turns)

    def qualify(self, turn):
        return self.turns.pop(0)


def _draft(reply, **facts):
    """Parse the tool call exactly as a real Gemini response is parsed."""
    args = {
        "reply": reply, "request_human": False, "job_type": None, "suburb": None, "urgency": None,
        "hazard": "none", "description": None, "preferred_time": None, "contact_name": None, **facts,
    }
    call = SimpleNamespace(name="record_job_intake", args=args)
    return parse_gemini_qualification(SimpleNamespace(function_calls=[call], text=None, candidates=[]))


def test_live_booking_conversation_reads_cleanly(tmp_path):
    db = Database(str(tmp_path / "booking.db"))
    model = ScriptedGemini(
        [
            _draft("Thanks!", job_type="power_point", suburb="Newtown"),
            _draft(
                "Got it, Wed 18 Nov between 9:00 am and 10:30 am. Could I please grab your name?",
                preferred_time="Wed 18 Nov, 9:00 am \\u2013 10:30 am",
                urgency="planned",
            ),
            _draft("Thanks Robbie! What's going on, in a sentence or two?", contact_name="robbie williams"),
        ]
    )

    def send(text, n):
        return handle_incoming(
            db, model, SimulatedGateway(), ListNotifier(),
            Inbound(wa_id=WA, text=text, message_type="text", provider_message_id=f"live-{n}", source="simulator"),
            business=load_business(), now=EVAL_NOW,
        )

    first = send("install lighting in Newtown", 1)
    assert "reply 1, 2, or 3" in first["reply"]

    second = send("2", 2)
    reply = second["reply"]
    assert reply.count("You're booked for") == 1
    assert "Got it, Wed" not in reply  # the booked time is not said twice
    assert reply.endswith("Could I please grab your name?")
    assert "\\u2013" not in (second["lead"]["preferred_time"] or "")
    assert "–" in second["lead"]["preferred_time"]

    third = send("robbie williams", 3)
    lead = third["lead"]
    assert lead["contact_name"] == "Robbie Williams"
    assert lead["score_reason"] == "Job type is known; still missing: description."


def test_score_reason_names_only_what_is_missing():
    from app.scoring import score_lead

    business = load_business()
    score = score_lead({"job_type": "power_point", "suburb": "Newtown"}, business)
    assert score.reason == "Job type is known; still missing: urgency, description, name."


def test_a_capitalised_name_is_left_alone():
    from app.scoring import canonical_facts

    business = load_business()
    assert canonical_facts({"contact_name": "Robbie McDonald"}, business)["contact_name"] == "Robbie McDonald"
    assert canonical_facts({"contact_name": "robbie mcdonald"}, business)["contact_name"] == "Robbie Mcdonald"


def test_urgency_field_tells_the_model_a_booking_is_not_an_urgency():
    from app.llm import QUALIFY_TOOL

    assert "picking a visit time is not an urgency" in QUALIFY_TOOL["input_schema"]["properties"]["urgency"]["description"]
