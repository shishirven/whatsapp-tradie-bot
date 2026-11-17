"""Shared wording rules for fire hazards.

"Smoke" inside "smoke alarm" or "smoke detector" names a device, not a fire.
A beeping or chirping alarm is almost always a low battery or an expired
unit, so it is an ordinary smoke_alarm job. An alarm that is actually going
off is still treated as a possible fire.
"""

from __future__ import annotations

import re

_ALARM_DEVICE = re.compile(r"\bsmoke[\s-]*(?:alarm|detector)s?\b")
_ALARM_SOUNDING = re.compile(
    r"\b(?:going off|went off|gone off|goes off|go off|sounding|blaring|screaming|activated)\b"
)
_ALARM_MAINTENANCE = re.compile(
    r"\b(?:beep\w*|chirp\w*|batter(?:y|ies)|replac\w*|install\w*|expir\w*|"
    r"end of life|flashing|blinking|test\w*|upgrad\w*|compliance|"
    r"no reason|false alarm|nuisance|randomly)\b"
)


def without_alarm_devices(lowered: str) -> str:
    """Remove 'smoke alarm' / 'smoke detector' so the device name is not read as smoke."""
    return _ALARM_DEVICE.sub(" ", lowered)


def smoke_alarm_sounding(lowered: str) -> bool:
    """A smoke alarm that is going off, not one that is chirping for a battery."""
    if not _ALARM_DEVICE.search(lowered):
        return False
    return bool(_ALARM_SOUNDING.search(lowered)) and not _ALARM_MAINTENANCE.search(lowered)
