"""Loose intake answers are stored, and the same question is not asked again."""

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.extract import extract_facts
from app.intake import question_for
from app.llm import GeminiQualifier, QualificationDraft, RuleBasedQualifier, TurnContext
from app.main import create_app
from app.scoring import FACT_KEYS
from tests.conftest import make_settings


@pytest.mark.parametrize(
    ("text", "description", "job_type"),
    [
        ("wiring is the issue", "wiring", "wiring"),
        ("power keeps tripping", "power keeps tripping", "fault"),
        ("lights not working", "lights not working", "power_point"),
    ],
)
def test_loose_problem_answer_is_stored_and_the_next_detail_is_asked(client, text, description, job_type):
    started = client.post("/api/conversations", json={}).json()
    conversation_id = started["conversation_id"]

    first = client.post("/api/chat", json={"conversation_id": conversation_id, "text": text}).json()
    assert first["lead"]["description"] == description
    assert first["lead"]["job_type"] == job_type
    reply = first["reply"].lower()
    assert "what needs doing" not in reply
    assert "what's going on" not in reply
    assert "suburb" in reply

    second = client.post(
        "/api/chat",
        json={"conversation_id": conversation_id, "text": "Marrickville"},
    ).json()
    assert second["lead"]["description"] == description
    assert second["lead"]["job_type"] == job_type
    assert second["lead"]["suburb"] == "Marrickville"
    again = second["reply"].lower()
    assert "what needs doing" not in again
    assert "what's going on" not in again
    assert "what is the problem" not in again
    assert question_for("job_type") not in second["reply"]
    assert question_for("description") not in second["reply"]


def test_a_short_problem_is_not_a_keyword_description_until_intake_accepts_it():
    assert extract_facts("wiring is the issue")["description"] is None
    assert extract_facts("wiring is the issue")["job_type"] is None
    assert extract_facts("Newtown")["description"] is None

    draft = RuleBasedQualifier().qualify(_turn("Newtown"))
    assert draft.suburb == "Newtown"
    assert draft.description is None
    assert draft.job_type is None


def test_unclear_answer_gets_one_different_followup_then_moves_on(client):
    conversation_id = client.post("/api/conversations", json={}).json()["conversation_id"]
    job_clarify = question_for("job_type", clarify=True)
    suburb_ask = question_for("suburb")
    suburb_clarify = question_for("suburb", clarify=True)

    first = client.post("/api/chat", json={"conversation_id": conversation_id, "text": "yes"}).json()
    assert first["lead"]["job_type"] is None
    assert first["lead"]["description"] is None
    assert job_clarify in first["reply"]
    assert question_for("job_type") not in first["reply"]

    second = client.post("/api/chat", json={"conversation_id": conversation_id, "text": "nope"}).json()
    assert second["lead"]["job_type"] is None
    assert suburb_ask in second["reply"]
    assert job_clarify not in second["reply"]
    assert question_for("job_type") not in second["reply"]

    third = client.post("/api/chat", json={"conversation_id": conversation_id, "text": "nope"}).json()
    assert suburb_clarify in third["reply"]
    assert suburb_ask not in third["reply"]
    assert third["lead"]["suburb"] is None

    fourth = client.post("/api/chat", json={"conversation_id": conversation_id, "text": "nope"}).json()
    assert suburb_clarify not in fourth["reply"]
    assert suburb_ask not in fourth["reply"]
    assert question_for("urgency") in fourth["reply"]
    assert fourth["lead"]["job_type"] is None
    assert fourth["lead"]["suburb"] is None


def test_a_loose_answer_after_the_followup_is_accepted(client):
    conversation_id = client.post("/api/conversations", json={}).json()["conversation_id"]
    client.post("/api/chat", json={"conversation_id": conversation_id, "text": "huh"})
    answered = client.post(
        "/api/chat",
        json={"conversation_id": conversation_id, "text": "wiring is the issue"},
    ).json()
    assert answered["lead"]["description"] == "wiring"
    assert answered["lead"]["job_type"] == "wiring"
    assert question_for("job_type") not in answered["reply"]
    assert question_for("job_type", clarify=True) not in answered["reply"]
    assert "suburb" in answered["reply"].lower()


def test_a_model_that_repeats_the_question_still_keeps_the_loose_answer(tmp_path, notifier):
    class Repeater:
        def qualify(self, turn):
            return QualificationDraft(
                reply=question_for("job_type"),
                request_human=False,
            )

    client = TestClient(
        create_app(settings=make_settings(tmp_path), qualifier=Repeater(), notifier=notifier)
    )
    conversation_id = client.post("/api/conversations", json={}).json()["conversation_id"]
    first = client.post(
        "/api/chat",
        json={"conversation_id": conversation_id, "text": "wiring is the issue"},
    ).json()
    assert first["lead"]["description"] == "wiring"
    assert first["lead"]["job_type"] == "wiring"
    assert question_for("job_type") not in first["reply"]
    assert "suburb" in first["reply"].lower()

    second = client.post(
        "/api/chat",
        json={"conversation_id": conversation_id, "text": "lights not working"},
    ).json()
    # The problem is already on file, so a later phrase does not get the same question.
    assert second["lead"]["description"] == "wiring"
    assert second["lead"]["job_type"] == "wiring"
    assert question_for("job_type") not in second["reply"]
    assert "what's going on" not in second["reply"].lower()


@pytest.mark.parametrize(
    ("text", "description", "job_type"),
    [
        ("wiring is the issue", "wiring", "wiring"),
        ("power keeps tripping", "power keeps tripping", "fault"),
        ("lights not working", "lights not working", "power_point"),
    ],
)
def test_gemini_timeout_uses_the_keyword_path_and_keeps_the_loose_answer(text, description, job_type):
    class Models:
        def generate_content(self, **kwargs):
            raise TimeoutError("The read operation timed out")

    qualifier = GeminiQualifier(
        "test-key",
        "gemini-3.8-flash",
        client=SimpleNamespace(models=Models()),
    )
    draft = qualifier.qualify(
        _turn(
            text,
            history=[
                {"role": "assistant", "content": question_for("job_type")},
                {"role": "user", "content": text},
            ],
        )
    )
    assert draft.description == description
    assert draft.job_type == job_type
    assert question_for("job_type") not in draft.reply
    assert "suburb" in draft.reply.lower()


def _turn(text, history=None, known=None):
    base = {key: None for key in FACT_KEYS}
    base.update(known or {})
    return TurnContext(
        history=history or [{"role": "user", "content": text}],
        known=base,
        ad=None,
        profile_name=None,
        system="system",
    )
