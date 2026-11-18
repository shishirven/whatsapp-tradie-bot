"""Staff notification seam.

The durable record of a handoff is the lead row. This notifier is only the
side effect (a log line today, a Slack or email call later).
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Protocol

logger = logging.getLogger("tradie.handoff")


class StaffNotifier(Protocol):
    def handoff(self, summary: dict) -> None:
        """Tell the sales desk that a conversation needs a person."""


def mask_number(value) -> str:
    """Last three digits only, so logs can tell numbers apart without holding them."""
    digits = "".join(ch for ch in str(value or "") if ch.isalnum())
    return f"***{digits[-3:]}" if digits else "none"


class LogNotifier:
    """TODO: post `summary` to the desk Slack channel or email the tradie."""

    def handoff(self, summary: dict) -> None:
        logger.info(
            "HANDOFF lead=%s wa_id=%s score=%s reason=%s job_type=%s suburb=%s",
            summary.get("lead_id"),
            mask_number(summary.get("wa_id")),
            summary.get("score"),
            summary.get("reason"),
            summary.get("job_type"),
            summary.get("suburb"),
        )


class ListNotifier:
    """Test double that remembers handoff summaries."""

    def __init__(self) -> None:
        self.events: list[dict] = []

    def handoff(self, summary: dict) -> None:
        self.events.append(summary)


# What the owner sees for each hazard, most serious first in the headline.
_HAZARD_HEADLINES = {
    "life_support": "URGENT SAFETY: life-support customer without power",
    "person_down": "URGENT SAFETY: person in contact with power",
    "powerline": "URGENT SAFETY: fallen powerline or pole",
    "wire": "URGENT SAFETY: possible live wire",
    "shock": "URGENT SAFETY: electric shock or tingle",
    "fire": "URGENT SAFETY: smoke, fire or burning",
    "alarm": "Smoke or fire alarm going off",
    "water": "Urgent: water near electrics",
    "overheating": "Urgent: overheating or melting fitting",
    "battery": "Urgent: battery or inverter problem",
}
_REASON_HEADLINES = {
    "danger": "URGENT SAFETY",
    "complaint": "Complaint",
    "pricing": "Price or discount request",
    "abusive": "Upset customer",
    "hot_lead": "Hot lead ready to book",
    "human_requested": "Customer asked for a person",
    "unclear_request": "Bot couldn't work out what the customer needs",
}


class WhatsAppOwnerNotifier:
    """Send each alert to the owner's own WhatsApp, and keep the log line.

    WhatsApp only allows free text to someone who messaged the business number in the
    last 24 hours. Alerts usually fall outside that, so they go out as an approved
    utility template (OWNER_ALERT_TEMPLATE). Without a template, free text is tried
    and a failure is logged with what to fix. The customer's reply is never held up:
    the send runs in a background thread. The same job and reason alert at most once
    every `repeat_after_seconds`, so a customer sending three messages about one live
    wire produces one alert, not three.
    """

    def __init__(
        self,
        gateway,
        owner_number: str,
        *,
        template: str | None = None,
        language: str = "en",
        base_url: str | None = None,
        background: bool = True,
        repeat_after_seconds: int = 30 * 60,
        clock=None,
    ) -> None:
        self.gateway = gateway
        self.owner_number = owner_number
        self.template = template or None
        self.language = language or "en"
        self.base_url = (base_url or "").rstrip("/") or None
        self.background = background
        self.repeat_after_seconds = repeat_after_seconds
        self.clock = clock or time.monotonic
        self._log = LogNotifier()
        self._last_sent: dict[tuple, float] = {}
        self._lock = threading.Lock()

    def handoff(self, summary: dict) -> None:
        self._log.handoff(summary)
        key = (summary.get("lead_id"), summary.get("hazard") or summary.get("reason"))
        now = self.clock()
        with self._lock:
            last = self._last_sent.get(key)
            if last is not None and now - last < self.repeat_after_seconds:
                logger.info("owner alert suppressed (repeat) lead=%s reason=%s", key[0], key[1])
                return
            self._last_sent[key] = now
        params = alert_params(summary, self.base_url)
        if self.background:
            threading.Thread(target=self._send, args=(params, summary), daemon=True).start()
        else:
            self._send(params, summary)

    def _send(self, params: list[str], summary: dict) -> None:
        owner = mask_number(self.owner_number)
        try:
            if self.template:
                send_template = getattr(self.gateway, "send_template")
                try:
                    message_id = send_template(self.owner_number, self.template, params, language=self.language)
                except TypeError:  # a gateway without the language argument
                    message_id = send_template(self.owner_number, self.template, params)
                mode = f"template {self.template}"
            else:
                message_id = self.gateway.send_text(self.owner_number, alert_text(params))
                mode = "text"
        except Exception as exc:
            hint = (
                "" if self.template else
                " Free text only reaches the owner within 24 hours of the owner messaging the business "
                "number. Set OWNER_ALERT_TEMPLATE to an approved utility template so alerts always deliver."
            )
            logger.error(
                "owner alert FAILED lead=%s owner=%s: %s.%s",
                summary.get("lead_id"), owner, " ".join(str(exc).split())[:300], hint,
            )
            return
        logger.info("owner alert sent lead=%s owner=%s via %s id=%s", summary.get("lead_id"), owner, mode, message_id)


def alert_params(summary: dict, base_url: str | None = None) -> list[str]:
    """The five values for the owner_job_alert template, in order."""
    hazard = summary.get("hazard")
    reason = summary.get("reason") or "escalated"
    headline = _HAZARD_HEADLINES.get(hazard) or _REASON_HEADLINES.get(reason) or reason.replace("_", " ").capitalize()
    job_bits = [f"#{summary.get('lead_id')}"]
    for key in ("job_label", "suburb"):
        if summary.get(key):
            job_bits.append(str(summary[key]))
    if summary.get("urgency"):
        job_bits.append(str(summary["urgency"]))
    customer = " ".join(
        part for part in (summary.get("contact_name"), _display_number(summary.get("wa_id"))) if part
    )
    message = summary.get("message") or "(no text: photo or voice note)"
    if base_url and summary.get("lead_id") is not None:
        link = f"{base_url}/staff/{summary['lead_id']}"
    else:
        link = "the staff desk"
    return [_param(headline), _param(", ".join(job_bits)), _param(customer or "unknown"), _param(message, 300), _param(link)]


def alert_text(params: list[str]) -> str:
    """The same alert as free text, for when no template is configured."""
    headline, job, customer, message, link = params
    return (
        f"{headline}\n"
        f"Job: {job}\n"
        f"Customer: {customer}\n"
        f"They said: \"{message}\"\n"
        f"Open the job: {link}"
    )


def _param(value: object, limit: int = 160) -> str:
    """Template parameters can't hold newlines, tabs or long runs of spaces, or be empty."""
    text = " ".join(str(value or "").split())
    if len(text) > limit:
        text = text[: limit - 1].rstrip() + "…"
    return text or "-"


def _display_number(wa_id) -> str | None:
    digits = "".join(ch for ch in str(wa_id or "") if ch.isdigit())
    if not digits:
        return None
    if digits.startswith("61") and len(digits) == 11:
        return f"+61 {digits[2:5]} {digits[5:8]} {digits[8:]}"
    return f"+{digits}"
