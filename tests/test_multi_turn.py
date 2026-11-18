"""Multi-turn conversations around starting a second job in the same chat.

Each test replays a short chat. The intent classifier is a fake that is far
too eager to call something a new job, the way a small model can be, so the
code-side guards are what keep these conversations on the right job.
"""

from fastapi.testclient import TestClient

from app.llm import QualificationDraft
from app.main import create_app
from tests.conftest import make_settings


class EagerIntent:
    """Says 'new job' to anything except an obvious answer to our last question."""

    def __init__(self):
        self.summaries: list[str] = []

    def classify(self, text, job_summary=""):
        self.summaries.append(job_summary)
        return True


class Scripted:
    """Answers like a model that always re-introduces itself."""

    def __init__(self, job_types: dict[str, str]):
        self.job_types = job_types
        self.systems: list[str] = []

    def qualify(self, turn):
        self.systems.append(turn.system)
        latest = turn.history[-1]["content"].lower()
        job_type = turn.known.get("job_type")
        for cue, kind in self.job_types.items():
            if cue in latest:
                job_type = kind
        if job_type:
            ask = "Which suburb is the job in?"
        else:
            ask = "What needs doing?"
        return QualificationDraft(
            reply=f"Hi King Brown, I'm Riley, the AI assistant for Northside Electrical. {ask}",
            job_type=job_type,
            suburb=turn.known.get("suburb"),
        )


def _chat(tmp_path, notifier, intent, qualifier):
    client = TestClient(
        create_app(
            settings=make_settings(tmp_path),
            notifier=notifier,
            qualifier=qualifier,
            intent=intent,
        )
    )
    conversation_id = client.post("/api/conversations", json={}).json()["conversation_id"]

    def say(text):
        return client.post(
            "/api/chat", json={"conversation_id": conversation_id, "text": text}
        ).json()

    return client, say


def _job_count(client):
    return len(client.app.state.db.list_jobs())


def test_joke_answer_to_the_suburb_question_stays_on_the_new_job(tmp_path, notifier):
    """The reported chat: new request, then 'your moms house' opened a third job."""
    intent = EagerIntent()
    client, say = _chat(tmp_path, notifier, intent, Scripted({"ev charger": "ev_charger"}))

    first = say("I need an EV charger installed")
    assert first["lead"]["job_type"] == "ev_charger"

    second = say("what if i have a new request")
    assert second["lead"]["id"] != first["lead"]["id"]
    assert "separate job" in second["reply"].lower()

    third = say("your moms house")
    assert third["lead"]["id"] == second["lead"]["id"]
    assert "separate job" not in third["reply"].lower()
    assert _job_count(client) == 2


def test_a_real_suburb_after_a_new_request_stays_on_the_new_job(tmp_path, notifier):
    intent = EagerIntent()
    client, say = _chat(tmp_path, notifier, intent, Scripted({"ev charger": "ev_charger"}))
    say("I need an EV charger installed")
    second = say("got another job for you")
    third = say("Parramatta")
    assert third["lead"]["id"] == second["lead"]["id"]
    assert _job_count(client) == 2


def test_rude_non_answer_does_not_open_a_job(tmp_path, notifier):
    intent = EagerIntent()
    client, say = _chat(tmp_path, notifier, intent, Scripted({"ev charger": "ev_charger"}))
    say("I need an EV charger installed")
    second = say("new job please")
    for text in ("none of your business", "lol", "idk mate"):
        reply = say(text)
        assert reply["lead"]["id"] == second["lead"]["id"], text
    assert _job_count(client) == 2


def test_a_new_job_does_not_reintroduce_the_assistant(tmp_path, notifier):
    intent = EagerIntent()
    qualifier = Scripted({"ev charger": "ev_charger"})
    client, say = _chat(tmp_path, notifier, intent, qualifier)
    first = say("I need an EV charger installed")
    assert "I'm Riley" in first["reply"]

    second = say("what if i have a new request")
    assert "I'm Riley" not in second["reply"]
    assert not second["reply"].lower().startswith("i'll open this as a separate job. hi")
    assert "What needs doing?" in second["reply"]
    assert "already mid-conversation" in qualifier.systems[-1]
    assert "already mid-conversation" not in qualifier.systems[0]


def test_classifier_sees_the_question_we_just_asked(tmp_path, notifier):
    seen: list[str] = []

    class Recording:
        def classify(self, text, job_summary=""):
            seen.append(job_summary)
            return False

    client, say = _chat(tmp_path, notifier, Recording(), Scripted({"ev charger": "ev_charger"}))
    say("I need an EV charger installed")
    say("it's at my place in Parramatta near the station")
    assert seen, "classifier was not consulted on a typed job"
    assert "Last message we sent:" in seen[-1]
    assert "Which suburb is the job in?" in seen[-1]


def test_a_clear_new_job_on_a_typed_job_still_splits(tmp_path, notifier):
    client, say = _chat(
        tmp_path,
        notifier,
        EagerIntent(),
        Scripted({"ev charger": "ev_charger", "ceiling fan": "ceiling_fan"}),
    )
    first = say("I need an EV charger installed")
    second = say("different job, a ceiling fan in the bedroom")
    assert second["lead"]["id"] != first["lead"]["id"]
    assert second["lead"]["job_type"] == "ceiling_fan"
    # Once the second job has a type, answering its suburb question stays put
    # only if the classifier says so; the eager fake would split again, which
    # is why the classifier now sees our last question.
    assert _job_count(client) == 2


def test_joke_is_not_stored_as_the_suburb(tmp_path, notifier):
    client, say = _chat(
        tmp_path, notifier, EagerIntent(), Scripted({"ev charger": "ev_charger"})
    )
    say("I need an EV charger installed")
    after = say("your moms house")
    assert not after["lead"].get("suburb")
    real = say("Parramatta")
    assert real["lead"]["suburb"] == "Parramatta"


def test_non_answers_are_recognised_and_real_answers_are_kept():
    from app.intake import is_non_answer

    for text in ("your moms house", "ur mums", "none of your business", "lol", "haha!", "rack off"):
        assert is_non_answer(text), text
    for text in (
        "Parramatta",
        "whatever time suits",
        "doesn't matter, anytime",
        "lol the light's broken",
        "my mum's place in Ryde has no power",
    ):
        assert not is_non_answer(text), text


def test_plural_aliases_match_the_job_type():
    from app.business import load_business
    from app.extract import extract_job_type

    business = load_business()
    assert extract_job_type("need 6 downlights put in", business) == "power_point"
    assert extract_job_type("two power points in the kitchen", business) == "power_point"
    assert extract_job_type("swap the light fittings", business) == "power_point"
