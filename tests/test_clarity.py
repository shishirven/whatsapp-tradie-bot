"""The clarity layer: ask instead of guessing when a message isn't a clear electrical request.

No real model calls: the offline rules, plus a scripted Gemini parsed through the real parser.
"""

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.business import load_business
from app.clarity import assess, looks_like_gibberish, rule_intent, usable_question
from app.llm import QUALIFY_TOOL, _gemini_tool, parse_gemini_qualification
from app.main import create_app
from app.notify import ListNotifier, alert_params
from tests.conftest import make_settings

GREETING_ASKS = ["job_type", "suburb"]  # Riley's greeting asks what needs doing and which suburb


def _gemini(reply="", **fields):
    args = {
        "reply": reply, "request_human": False, "job_type": None, "suburb": None, "urgency": None,
        "hazard": "none", "intent": None, "clarifying_question": None, "description": None,
        "preferred_time": None, "contact_name": None, **fields,
    }
    call = SimpleNamespace(name="record_job_intake", args=args)
    return parse_gemini_qualification(SimpleNamespace(function_calls=[call], text=None, candidates=[]))


class Scripted:
    def __init__(self, *drafts):
        self.drafts = list(drafts)

    def qualify(self, turn):
        return self.drafts.pop(0)


def _chat(tmp_path, messages, qualifier=None, notifier=None):
    client = TestClient(create_app(settings=make_settings(tmp_path), qualifier=qualifier, notifier=notifier))
    cid = client.post("/api/conversations", json={}).json()["conversation_id"]
    return [client.post("/api/chat", json={"conversation_id": cid, "text": text}).json() for text in messages], client


# --- the screenshot -------------------------------------------------------------


def test_hha_offline_asks_what_they_need_and_stores_nothing(tmp_path):
    [result], _ = _chat(tmp_path, ["hha"])
    assert result["lead"]["job_type"] is None and result["lead"]["suburb"] is None
    assert result["reply"].startswith("Sorry, I didn't quite catch that. What do you need an electrician for?")
    assert "fault finding and power loss" in result["reply"]


def test_hha_with_gemini_uses_its_clarifying_question(tmp_path):
    model = Scripted(_gemini(
        "Hey there!", job_type="hha", suburb="Hha", intent="unclear",
        clarifying_question="Sorry, I didn't get that. What electrical work can we help with?",
    ))
    [result], _ = _chat(tmp_path, ["hha"], qualifier=model)
    assert result["reply"] == "Sorry, I didn't get that. What electrical work can we help with?"
    assert result["lead"]["job_type"] is None and result["lead"]["suburb"] is None


def test_hha_is_caught_even_when_the_model_calls_it_a_job(tmp_path):
    model = Scripted(_gemini("Got it!", job_type="hha", intent="service_request"))
    [result], _ = _chat(tmp_path, ["hha"], qualifier=model)
    assert result["lead"]["job_type"] is None
    assert "What do you need an electrician for?" in result["reply"]


# --- general cases ----------------------------------------------------------------


@pytest.mark.parametrize(
    "text,intent",
    [
        ("hha", "unclear"),
        ("asdfgh", "unclear"),
        ("?", "unclear"),
        ("hi", "greeting"),
        ("thanks mate", "greeting"),
        ("need an electrician", "vague_request"),
        ("can you help", "vague_request"),
        ("my toilet is blocked", "not_electrical"),
        ("can you help me with SEO for my website", "not_electrical"),
        ("lights keep flickering in the kitchen", "service_request"),
        ("wiring is the issue", "service_request"),
        ("do you work weekends?", "question"),
    ],
)
def test_offline_intent_of_a_first_message(text, intent):
    assert rule_intent(text, known={}, pending=GREETING_ASKS, business=load_business()) == intent


@pytest.mark.parametrize(
    "text,model_intent,expected",
    [
        ("need 6 downlights installed", "unclear", "service_request"),  # configured job word
        ("sparks!!", "unclear", "service_request"),  # a hazard is never clarified
        ("Can someone come tomorrow morning?", "vague_request", "question"),  # booking handles it
        ("Newtown", "unclear", "answer"),  # configured suburb, and the suburb was asked
        ("the thing on the verandah died", "vague_request", "vague_request"),  # model beats rules
    ],
)
def test_deterministic_evidence_beats_the_model_and_the_model_beats_rules(text, model_intent, expected):
    result = assess(text, model_intent=model_intent, known={}, pending=GREETING_ASKS, business=load_business())
    assert result.intent == expected


def test_a_guardrail_reply_is_never_replaced_by_a_clarifying_question():
    result = assess(
        "I want a full refund", model_intent="unclear", known={}, pending=GREETING_ASKS,
        business=load_business(), handled=True,
    )
    assert not result.clarify and result.strict  # replied to by policy, and no junk job stored


def test_vague_request_keeps_only_confirmed_facts(tmp_path):
    model = Scripted(_gemini("Sure", job_type="electrical stuff", suburb="Newtown", intent="vague_request"))
    [result], _ = _chat(tmp_path, ["need some electrical stuff done at my place in Newtown"], qualifier=model)
    assert result["lead"]["job_type"] is None
    assert result["lead"]["suburb"] == "Newtown"  # configured suburb kept
    assert result["reply"].startswith("Happy to help. What's the electrical job?")


def test_not_electrical_is_declined_politely_and_never_escalates(tmp_path):
    results, _ = _chat(tmp_path, ["my toilet is blocked", "the toilet!", "toilet is blocked again"])
    for result in results:
        assert "we're electricians" in result["reply"]
        assert result["lead"]["handoff"] is False
        assert result["lead"]["job_type"] is None


def test_unclear_three_times_gives_a_menu_then_a_person(tmp_path):
    notifier = ListNotifier()
    (first, second, third), _ = _chat(tmp_path, ["hha", "asdf", "zzz"], notifier=notifier)
    assert "What do you need an electrician for?" in first["reply"]
    assert "Reply with a number" in second["reply"]
    assert "1. Fault finding and power loss" in second["reply"]
    assert third["lead"]["handoff"] is True
    assert third["lead"]["handoff_reason"] == "unclear_request"
    assert "a person from the" in third["reply"]
    assert notifier.events[-1]["reason"] == "unclear_request"
    assert alert_params(notifier.events[-1])[0] == "Bot couldn't work out what the customer needs"


def test_a_menu_number_picks_the_job_and_intake_carries_on(tmp_path):
    (_, menu, picked), _ = _chat(tmp_path, ["hha", "asdf", "2"])
    assert "2. Power point or lighting" in menu["reply"]
    assert picked["lead"]["job_type"] == "power_point"
    assert "Power point or lighting — I'll put that on the job." in picked["reply"]


def test_a_clear_message_resets_the_count(tmp_path):
    results, _ = _chat(tmp_path, ["hha", "need a new power point", "zzz"])
    assert results[1]["lead"]["job_type"] == "power_point"
    assert "Reply with a number" not in results[2]["reply"]  # not a second unclear in a row


def test_an_unclear_answer_mid_intake_repeats_the_question(tmp_path):
    model = Scripted(
        _gemini("Got it. Which suburb is the job in?", job_type="power_point", intent="service_request"),
        _gemini("Okay!", suburb="Hha", intent="unclear"),
    )
    (_, second), _ = _chat(tmp_path, ["need a new power point", "hha"], qualifier=model)
    assert second["reply"] == "Sorry, I didn't quite catch that. Which suburb is the job in?"
    assert second["lead"]["suburb"] is None
    assert second["lead"]["job_type"] == "power_point"  # what was on file stays


def test_a_three_letter_suburb_is_fine_mid_conversation_offline():
    business = load_business()
    assert rule_intent("kew", known={"job_type": "power_point"}, pending=["suburb"], business=business) == "answer"
    assert looks_like_gibberish("kew") and not looks_like_gibberish("kew", short_is_unclear=False)


def test_clarifying_turns_are_recorded(tmp_path):
    _, client = _chat(tmp_path, ["hha"])
    from app.llm_log import summarise

    calls = client.app.state.db.list_llm_calls()
    assert calls[-1]["policy_rule"] == "clarify:unclear"
    assert summarise(calls)["clarified"] == 1


# --- the model's fields -------------------------------------------------------------


def test_intent_and_clarifying_question_are_required_tool_fields():
    schema = QUALIFY_TOOL["input_schema"]
    assert {"intent", "clarifying_question"} <= set(schema["required"])
    assert "unclear" in schema["properties"]["intent"]["enum"]
    declaration = _gemini_tool().function_declarations[0].parameters_json_schema
    assert declaration["properties"]["intent"]["enum"] == schema["properties"]["intent"]["enum"]


def test_unknown_intent_values_are_ignored():
    assert _gemini(intent="banana").intent is None
    assert _gemini(intent="UNCLEAR").intent == "unclear"


@pytest.mark.parametrize(
    "question,usable",
    [
        ("What electrical work do you need done?", True),
        ("Tell me more.", False),  # not a question
        ("That's $200. Want to book?", False),  # a price
        ("x" * 300 + "?", False),  # too long
    ],
)
def test_only_a_short_safe_question_from_the_model_is_used(question, usable):
    assert (usable_question(question) is not None) is usable
