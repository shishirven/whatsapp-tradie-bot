"""Staff notification seam.

The durable record of a handoff is the lead row. This notifier is only the
side effect (a log line today, a Slack or email call later).
"""

from __future__ import annotations

import logging
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
