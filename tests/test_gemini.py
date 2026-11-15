"""Gemini qualifier and new-job classifier. The client is a fake. No network."""

from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.llm import GeminiQualifier, TurnContext
from app.main import build_qualifier, create_app, llm_status
from app.routing import GeminiIntentClassifier, build_intent
from app.scoring import FACT_KEYS
from tests.conftest import make_settings


def test_gemini_function_call_is_forced_and_prices_are_scrubbed(tmp_path, notifier):
    captured = {}

    class Models:
        def generate_content(self, **kwargs):
            captured.update(kwargs)
            return _function_response(
                {
                    "reply": "A switchboard is $1,800 to $4,500. Which suburb?",
                    "request_human": False,
                    "job_type": "switchboard",
                    "suburb": None,
                    "urgency": None,
                    "description": None,
                    "preferred_time": None,
                    "contact_name": None,
                }
            )

    qualifier = GeminiQualifier(
        "test-key",
        "gemini-2.5-flash",
        client=SimpleNamespace(models=Models()),
    )
    client = TestClient(
        create_app(settings=make_settings(tmp_path), qualifier=qualifier, notifier=notifier)
    )
    conversation_id = client.post("/api/conversations", json={}).json()["conversation_id"]
    result = client.post(
        "/api/chat",
        json={"conversation_id": conversation_id, "text": "switchboard please"},
    )
    assert "$" not in result.json()["reply"]
    assert "Which suburb?" in result.json()["reply"]
    assert result.json()["lead"]["job_type"] == "switchboard"
    assert result.json()["lead"]["quote"]["status"] == "draft"
    assert captured["model"] == "gemini-2.5-flash"
    config = captured["config"]
    mode = config.tool_config.function_calling_config.mode
    assert str(mode) == "ANY" or getattr(mode, "value", None) == "ANY"
    declaration = config.tools[0].function_declarations[0]
    assert declaration.name == "record_job_intake"
    assert "price" in declaration.description.lower()
    assert config.automatic_function_calling.disable is True


def test_gemini_rate_limit_uses_the_offline_qualifier(tmp_path, notifier):
    class Models:
        def generate_content(self, **kwargs):
            error = Exception("429 RESOURCE_EXHAUSTED")
            error.code = 429
            raise error

    qualifier = GeminiQualifier("test-key", "gemini-2.5-flash", client=SimpleNamespace(models=Models()))
    client = TestClient(
        create_app(settings=make_settings(tmp_path), qualifier=qualifier, notifier=notifier)
    )
    conversation_id = client.post("/api/conversations", json={}).json()["conversation_id"]
    result = client.post(
        "/api/chat",
        json={
            "conversation_id": conversation_id,
            "text": "The safety switch keeps tripping in Newtown",
        },
    )
    assert result.status_code == 200
    assert "once more" not in result.json()["reply"]
    assert result.json()["lead"]["job_type"] == "fault"
    assert result.json()["lead"]["suburb"] == "Newtown"
    assert "$" not in result.json()["reply"]


def test_gemini_plain_text_stores_no_new_facts():
    class Models:
        def generate_content(self, **kwargs):
            return SimpleNamespace(text="Which suburb?", function_calls=[], candidates=[])

    qualifier = GeminiQualifier("test-key", "gemini-2.5-flash", client=SimpleNamespace(models=Models()))
    draft = qualifier.qualify(_turn("Hi"))
    assert draft.reply == "Which suburb?"
    assert draft.job_type is None
    assert draft.raw == {"unstructured": True}


def test_gemini_intent_reads_yes_or_no_and_a_429_uses_the_phrase_list():
    class Models:
        def __init__(self, answer=None, boom=False):
            self.answer = answer
            self.boom = boom
            self.calls = 0

        def generate_content(self, **kwargs):
            self.calls += 1
            if self.boom:
                error = Exception("quota")
                error.code = 429
                raise error
            return SimpleNamespace(text=self.answer, candidates=[])

    yes = Models("YES")
    classifier = GeminiIntentClassifier(
        "test-key", "gemini-2.5-flash", client=SimpleNamespace(models=yes)
    )
    assert classifier.classify("the other house", job_summary="fault, Newtown") is True
    assert yes.calls == 1

    no = GeminiIntentClassifier(
        "test-key",
        "gemini-2.5-flash",
        client=SimpleNamespace(models=Models("NO")),
    )
    assert no.classify("the oven still trips it") is False

    limited = GeminiIntentClassifier(
        "test-key",
        "gemini-2.5-flash",
        client=SimpleNamespace(models=Models(boom=True)),
    )
    assert limited.classify("this is a different job") is True
    assert limited.classify("Also, a switchboard in Newtown") is False


def test_provider_switch_and_missing_gemini_key(tmp_path, monkeypatch, notifier):
    from app.config import Settings

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "env.db"))
    assert Settings.from_env().llm_provider == "mock"

    monkeypatch.setenv("GEMINI_API_KEY", "gem-test")
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    chosen = Settings.from_env()
    assert chosen.llm_provider == "gemini"
    assert chosen.gemini_model == "gemini-2.5-flash"

    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
    assert Settings.from_env().gemini_model == "gemini-2.5-flash-lite"

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    assert Settings.from_env().llm_provider == "gemini"

    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    assert Settings.from_env().llm_provider == "anthropic"

    business_settings = make_settings(
        tmp_path,
        llm_provider="gemini",
        gemini_api_key="",
        gemini_model="gemini-2.5-flash",
    )
    missing = build_qualifier(business_settings, None)
    status = llm_status(business_settings, missing)
    assert status["provider"] == "unconfigured"
    assert "GEMINI_API_KEY" in status["label"]

    client = TestClient(create_app(settings=business_settings, notifier=notifier))
    conversation_id = client.post("/api/conversations", json={}).json()["conversation_id"]
    result = client.post("/api/chat", json={"conversation_id": conversation_id, "text": "Hi"})
    assert "GEMINI_API_KEY" in result.json()["reply"]
    assert "aistudio.google.com" in result.json()["reply"]

    wired = build_intent(
        "gemini",
        "",
        "claude-sonnet-5-5",
        gemini_api_key="gem-test",
        gemini_model="gemini-2.5-flash",
    )
    assert isinstance(wired, GeminiIntentClassifier)
    assert isinstance(build_intent("mock", "", "claude-sonnet-5-5"), type(build_intent("anthropic", "", "x")))


def _function_response(args):
    call = SimpleNamespace(name="record_job_intake", args=args)
    return SimpleNamespace(function_calls=[call], text=None, candidates=[])


def _turn(text):
    known = {key: None for key in FACT_KEYS}
    return TurnContext(
        history=[{"role": "user", "content": text}],
        known=known,
        ad=None,
        profile_name=None,
        system="Do not mention prices.",
    )
