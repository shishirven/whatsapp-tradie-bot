"""Reply wording problems found in the llm_calls records."""

import pytest

from app.business import load_business
from app.extract import extract_job_type
from app.service import _tidy_reply
from tests.test_policy import _incoming


@pytest.mark.parametrize(
    "text,job",
    [
        ("need 6 downlights installed in Newtown", "power_point"),
        ("two new power points in the kitchen", "power_point"),
        ("quote for 3 ceiling fans", "ceiling_fan"),
        ("our smoke alarms need replacing", "smoke_alarm"),
        ("need a downlight replaced", "power_point"),
    ],
)
def test_plural_job_words_match_their_job(text, job):
    assert extract_job_type(text, load_business()) == job


def test_downlights_message_is_a_named_job_not_an_echo(tmp_path):
    result = _incoming(tmp_path, load_business(), "need 6 downlights installed in Newtown", "61412995501")
    reply = result["reply"]
    assert result["lead"]["job_type"] == "power_point"
    assert "need 6 downlights" not in reply
    assert reply.startswith("Power point or lighting — I'll put that on the job.")


def test_unrecognised_job_is_acknowledged_without_reading_it_back():
    from app.intake import _acknowledgement
    from app.llm import _job_acknowledgement

    business = load_business()
    free_text = "need my pool pump wired up in Newtown"
    for reply in (_job_acknowledgement(free_text, business), _acknowledgement({"job_type": free_text}, business)):
        assert reply == "Got it, I'll put that on the job."
    for reply in (_job_acknowledgement("ceiling_fan", business), _acknowledgement({"job_type": "ceiling_fan"}, business)):
        assert reply == "Ceiling fan — I'll put that on the job."


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Typically $180–$450. the owner confirms the final price.", "Typically $180–$450. The owner confirms the final price."),
        ("the owner will call you.", "The owner will call you."),
        ("I'll pass this to the owner.", "I'll pass this to the owner."),
    ],
)
def test_the_owner_is_capitalised_only_at_a_sentence_start(raw, expected):
    assert _tidy_reply(raw) == expected


def test_price_reply_starts_the_owner_sentence_with_a_capital(tmp_path):
    reply = _incoming(tmp_path, load_business(), "how much for a new power point?", "61412995503")["reply"]
    assert "The owner confirms the final price" in reply
    assert ". the owner" not in reply


def test_replies_do_not_assume_the_owners_gender(tmp_path):
    business = load_business()
    for index, text in enumerate(
        ["I want a full refund", "The invoice is $300 more than the quote!", "Am I talking to a real person?"]
    ):
        reply = _incoming(tmp_path, business, text, f"6141299551{index}")["reply"]
        assert " him " not in f" {reply} " and "he'll" not in reply, reply
