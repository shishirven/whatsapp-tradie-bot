"""Owner alerts go to the owner's own WhatsApp. No network: gateways record sends."""

import logging

import httpx
import pytest
from fastapi.testclient import TestClient

from app.business import load_business
from app.db import Database
from app.llm import RuleBasedQualifier
from app.main import build_notifier, create_app
from app.messaging import DeliveryError, Inbound, WhatsAppCloudGateway
from app.notify import LogNotifier, WhatsAppOwnerNotifier, alert_params, alert_text
from app.service import handle_incoming
from evals.harness import EVAL_NOW
from tests.conftest import make_settings
from tests.fakes import MemoryGateway

OWNER = "61400111222"


import itertools

_ids = itertools.count(1)


class RecordingGateway:
    def __init__(self, fail=None):
        self.sent = []
        self.fail = fail

    def send_text(self, to, body):
        if self.fail:
            raise self.fail
        self.sent.append({"kind": "text", "to": to, "body": body})
        return f"wamid.text{next(_ids)}"

    def send_template(self, to, name, params, language="en"):
        if self.fail:
            raise self.fail
        self.sent.append({"kind": "template", "to": to, "name": name, "params": list(params), "language": language})
        return f"wamid.tpl{next(_ids)}"


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def _notifier(gateway, **kwargs):
    kwargs.setdefault("template", "owner_job_alert")
    kwargs.setdefault("base_url", "https://bot.example.com")
    return WhatsAppOwnerNotifier(gateway, OWNER, background=False, **kwargs)


def _customer(db, notifier, text, wa_id="61412345678", message_id=None):
    customer_gateway = RecordingGateway()
    result = handle_incoming(
        db,
        RuleBasedQualifier(load_business()),
        customer_gateway,
        notifier,
        Inbound(wa_id=wa_id, text=text, message_type="text",
                provider_message_id=message_id or f"oa-{abs(hash(text)) % 10**8}", source="whatsapp"),
        business=load_business(),
        now=EVAL_NOW,
    )
    return result


# --- the alert itself -----------------------------------------------------------


def test_live_wire_alerts_the_owner_on_whatsapp_with_the_template(tmp_path):
    gateway = RecordingGateway()
    result = _customer(Database(str(tmp_path / "a.db")), _notifier(gateway), "live wire hanging nearby")
    assert "8 metres" in result["reply"]  # the customer still gets the safety reply
    [alert] = gateway.sent
    assert alert["kind"] == "template" and alert["to"] == OWNER
    assert alert["name"] == "owner_job_alert" and alert["language"] == "en"
    headline, job, customer, message, link = alert["params"]
    assert headline == "URGENT SAFETY: possible live wire"
    assert job.startswith("#") and "emergency" in job
    assert customer == "+61 412 345 678"  # the owner needs the number to call back
    assert message == "live wire hanging nearby"
    assert link.startswith("https://bot.example.com/staff/")


def test_without_a_template_the_alert_is_free_text(tmp_path):
    gateway = RecordingGateway()
    _customer(Database(str(tmp_path / "b.db")), _notifier(gateway, template=None), "there's smoke coming from the switchboard")
    [alert] = gateway.sent
    assert alert["kind"] == "text"
    assert alert["body"].startswith("URGENT SAFETY: smoke, fire or burning\nJob: #")
    assert 'They said: "there\'s smoke coming from the switchboard"' in alert["body"]


def test_complaint_alert_says_complaint(tmp_path):
    gateway = RecordingGateway()
    _customer(Database(str(tmp_path / "c.db")), _notifier(gateway), "I want a full refund")
    [alert] = gateway.sent
    assert alert["params"][0] == "Complaint"


def test_ordinary_job_sends_no_alert(tmp_path):
    gateway = RecordingGateway()
    _customer(Database(str(tmp_path / "d.db")), _notifier(gateway), "need a new power point")
    assert gateway.sent == []


def test_repeat_messages_about_one_hazard_alert_once_then_again_later(tmp_path):
    gateway = RecordingGateway()
    clock = Clock()
    notifier = _notifier(gateway, clock=clock)
    db = Database(str(tmp_path / "e.db"))
    _customer(db, notifier, "live wire hanging nearby", message_id="m1")
    _customer(db, notifier, "it's still hanging there, live wire", message_id="m2")
    assert len(gateway.sent) == 1
    clock.now += 31 * 60
    _customer(db, notifier, "live wire still hanging, please hurry", message_id="m3")
    assert len(gateway.sent) == 2


def test_a_failed_alert_is_logged_and_the_customer_still_gets_a_reply(tmp_path, caplog):
    gateway = RecordingGateway(fail=DeliveryError("400 code 131047: Re-engagement message"))
    with caplog.at_level(logging.ERROR, logger="tradie.handoff"):
        result = _customer(Database(str(tmp_path / "f.db")), _notifier(gateway, template=None), "live wire hanging nearby")
    assert "8 metres" in result["reply"]
    assert "owner alert FAILED" in caplog.text
    assert "131047" in caplog.text
    assert "OWNER_ALERT_TEMPLATE" in caplog.text  # tells you what to fix
    assert OWNER not in caplog.text  # owner number masked in logs


def test_gateway_without_a_language_argument_still_works():
    gateway = MemoryGateway()
    notifier = WhatsAppOwnerNotifier(gateway, OWNER, template="owner_job_alert", background=False)
    notifier.handoff({"lead_id": 3, "reason": "danger", "hazard": "fire", "wa_id": "61412345678", "message": "smoke"})
    assert gateway.sent[0][0:3] == ("template", OWNER, "owner_job_alert")


def test_background_send_does_not_block(tmp_path):
    import threading

    started = threading.Event()
    release = threading.Event()

    class Slow(RecordingGateway):
        def send_template(self, *args, **kwargs):
            started.set()
            release.wait(5)
            return super().send_template(*args, **kwargs)

    gateway = Slow()
    notifier = WhatsAppOwnerNotifier(gateway, OWNER, template="owner_job_alert", background=True)
    notifier.handoff({"lead_id": 4, "reason": "danger", "hazard": "wire", "message": "live wire"})
    assert started.wait(5)  # the send is running in the background...
    assert gateway.sent == []  # ...and handoff() already returned
    release.set()


# --- parameters ------------------------------------------------------------------


def test_template_parameters_are_cleaned():
    params = alert_params({"lead_id": 9, "reason": "pricing", "message": "line one\n\nline\ttwo" + "x" * 400})
    assert params[0] == "Price or discount request"
    assert "\n" not in params[3] and "\t" not in params[3]
    assert len(params[3]) <= 300
    assert params[2] == "unknown"
    assert params[4] == "the staff desk"
    assert all(params)  # Meta rejects empty parameters
    assert alert_text(params).count("\n") == 4


# --- wiring ----------------------------------------------------------------------


def test_build_notifier_uses_whatsapp_only_when_the_owner_number_is_set(tmp_path):
    assert isinstance(build_notifier(make_settings(tmp_path), RecordingGateway()), LogNotifier)
    notifier = build_notifier(
        make_settings(tmp_path, owner_whatsapp="0400 111 222", owner_alert_template="owner_job_alert"),
        RecordingGateway(),
    )
    assert isinstance(notifier, WhatsAppOwnerNotifier)
    assert notifier.owner_number == OWNER  # local format normalised to WhatsApp digits
    assert notifier.template == "owner_job_alert"


def test_messages_from_the_owner_number_get_no_customer_reply(tmp_path):
    from tests.test_messaging import _payload

    gateway = RecordingGateway()
    settings = make_settings(tmp_path, owner_whatsapp="+1 555 123 0000")  # _payload sends from 15551230000
    client = TestClient(create_app(settings=settings, gateway=gateway, notifier=LogNotifier()))
    response = client.post("/webhook", json=_payload("thanks, on my way"))
    assert response.json() == {"status": "ok", "processed": 0, "duplicates": 0, "owner": 1}
    assert gateway.sent == []


def test_an_australian_owner_number_in_any_format_is_recognised(tmp_path):
    from app.main import is_owner_number

    for configured in ("0400 111 222", "+61 400 111 222", "61400111222"):
        settings = make_settings(tmp_path, owner_whatsapp=configured)
        assert is_owner_number(settings, "61400111222")
        assert not is_owner_number(settings, "61400111223")
    assert not is_owner_number(make_settings(tmp_path), "61400111222")


def test_template_payload_uses_the_configured_language():
    payload = WhatsAppCloudGateway("t", "123").build_template_payload(OWNER, "owner_job_alert", ["a"], "en_US")
    assert payload["template"]["language"] == {"code": "en_US"}


def test_meta_error_code_is_kept_in_the_delivery_error(monkeypatch):
    def fake_post(url, **kwargs):
        return httpx.Response(
            400,
            json={"error": {"code": 131047, "message": "Re-engagement message",
                            "error_data": {"details": "More than 24 hours have passed"}}},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    with pytest.raises(DeliveryError) as raised:
        WhatsAppCloudGateway("token", "123").send_text(OWNER, "hello")
    assert "131047" in str(raised.value)
    assert "24 hours" in str(raised.value)
    assert "token" not in str(raised.value)
