"""Electrical hazard detection that does not depend on the model.

Every customer message is checked here before the reply goes out. A hazard
has a kind (which safety script to use) and a tier:

- "emergency": tell the customer to call 000 now and escalate.
- "urgent": a real hazard that is not (yet) a 000 call. Make it safe, stop
  using it, and treat the job as urgent. The reply still says when to call 000.
- "redirect": not our job (a whole-street outage). Point them to the
  distributor and do not escalate.

The wording follows Australian distributor, regulator and first-aid guidance:
treat fallen or damaged wires as live and keep at least 8 metres away, don't
touch a person who may still be in contact with the power, don't touch wet
electrics or turn flood-affected power back on, and leave and call 000 if a
lithium battery smokes. See the module tests for the phrases that must and
must not trigger.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

EMERGENCY = "emergency"
URGENT = "urgent"
REDIRECT = "redirect"


@dataclass(frozen=True)
class Hazard:
    kind: str
    tier: str


# --- smoke alarms -----------------------------------------------------------
# "Smoke" inside "smoke alarm" names a device, not a fire. A beeping alarm is
# a battery or an expired unit. An alarm that is actually going off may be a fire.

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


# --- patterns ---------------------------------------------------------------

_NEAR = r"[^.!?\n]{0,40}?"  # same sentence, within a few words

_LIFE_SUPPORT = re.compile(r"\b(?:oxygen|life[\s-]?support|dialysis|ventilator)\b")

# A person who may still be in contact with the power, or is down after a shock.
_PERSON_DOWN = re.compile(
    r"\b(?:unconscious|not breathing|isn'?t breathing|stopped breathing|passed out|collapsed|"
    r"can'?t let go|stuck to|still holding|not responding|unresponsive)\b"
)
_ELECTRIC_CONTEXT = re.compile(
    r"\b(?:shock\w*|electr\w*|zapped|wires?|cables?|power ?points?|sockets?|outlets?|"
    r"switch\w*|plug\w*|leads?|cords?|appliance\w*|powerlines?|live|heater|kettle|toaster|"
    r"fridge|oven|stove|microwave|iron|drill|tools?|taps?|heater|dryer|washing machine)\b"
)

_STREET = re.compile(r"\b(?:whole street|our street|the street|in the street|whole suburb|whole area|neighbou?rs)\b")
_OUTAGE = re.compile(r"\b(?:power|outage|blackout|no electricity)\b")

_POWERLINE = re.compile(
    r"\b(?:power[\s-]?lines?|electricity lines?|overhead lines?|downed lines?|lines? (?:is|are) down|"
    r"(?:power|electricity|light|stobie) poles?|service (?:line|wire|cable)|"
    r"wires? (?:from|to) the pole|pole\b" + _NEAR + r"\b(?:fell|fallen|down|snapped|leaning|hit))\b"
)

_WIRE = r"(?:wires?|wiring|cables?|cabling)"
_WIRE_STATE = (
    r"(?:live|exposed|bare|hanging|dangling|broken|snapped|severed|cut|fallen|fell|"
    r"(?:came|come|is|are|has come|have come) down|on the ground|sticking out|hanging out|"
    r"pulled out|ripped|torn|step(?:ped)? on|tread on|trip over|touching the water)"
)
_WIRE_HAZARD = re.compile(
    rf"\b{_WIRE_STATE}\b{_NEAR}\b{_WIRE}\b|\b{_WIRE}\b{_NEAR}\b{_WIRE_STATE}\b"
)
_NOT_POWER_CABLE = re.compile(
    r"\b(?:nbn|internet|tv|telly|antenna|aerial|foxtel|phone|data|ethernet|network cable|"
    r"hdmi|usb|speaker|fence|fencing|clothes ?line|fishing)\b"
)
_TREE = re.compile(r"\b(?:tree|trees|branch|branches|limb)\b")
_WIRE_WORD = re.compile(rf"\b{_WIRE}\b")

_SHOCK = re.compile(
    r"\bshocks?\b|\b(?:got|been|getting|get) (?:a )?(?:shock|shocked|zapped|jolt)\w*|"
    r"\btingl\w*|\bzapped\b|\belectrocut\w*"
)
_NOT_SHOCK = re.compile(r"\bstatic\b|\bshock absorb\w*|\bshell[\s-]?shock")

_FIRE = re.compile(
    r"\bburning\b|\bsmoke\b|\bsmoking\b|\bflames?\b|\bsparks?\b|\bsparking\b|\bsparked\b|\barcing\b|"
    r"\b(?:on|caught|catching|started a|there'?s a|small|electrical) fire\b|\bfire (?:in|at|from|coming)\b"
)

_BATTERY = re.compile(r"\b(?:batter(?:y|ies)|powerwall|inverter|solar)\b")
_BATTERY_STATE = re.compile(
    r"\b(?:swollen|swelling|bulging|puffed|hissing|popping|leaking|overheating|"
    r"hot to (?:the )?touch|very hot|really hot|melt\w*)\b"
)

_WATER = r"(?:water|flood\w*|leak\w*|drip\w*|wet|soaked|rain\w*|under ?water|submerged)"
_ELECTRICS = (
    r"(?:switchboard|meter box|fuse ?box|power ?points?|sockets?|outlets?|gpos?|lights?|"
    r"downlights?|light fittings?|wiring|wires?|electrics|electrical|inverter|appliances?|"
    r"heater|plugs?|ceiling fan)"
)
_WATER_HAZARD = re.compile(
    rf"\b{_WATER}\b{_NEAR}\b{_ELECTRICS}\b|\b{_ELECTRICS}\b{_NEAR}\b{_WATER}\b"
)
_FLOODED = re.compile(r"\b(?:house|home|place|garage|unit|we) (?:is |was |got |has been )?flooded\b|\bflooded\b")

_OVERHEAT = re.compile(
    r"\bmelt\w*|\bscorch\w*|\bcharred\b|\bblacken\w*|\bburn marks?\b|\bscorch marks?\b|"
    r"\bhot to (?:the )?touch\b|\b(?:really|very|too) hot\b|\bsmells? (?:hot|like plastic|fishy)\b|"
    r"\bfishy smell\b|\bbuzz\w*|\bcrackl\w*|\bsizzl\w*|\bfrayed\b"
)
_OVERHEAT_CONTEXT = re.compile(
    rf"\b{_ELECTRICS}\b|\b(?:switch\w*|plug\w*|leads?|cords?|extension|charger|board|breaker\w*)\b"
)


# --- classification ---------------------------------------------------------


def classify_hazard(text: str | None) -> Hazard | None:
    """The hazard in one customer message, most serious first. None for an ordinary job."""
    lowered = " ".join((text or "").casefold().split())
    if not lowered:
        return None
    # "hot water" is a job type, not hot or water near the electrics.
    lowered = re.sub(r"\bhot[\s-]water\b", " ", lowered)
    plain = without_alarm_devices(lowered)

    if _LIFE_SUPPORT.search(plain):
        return Hazard("life_support", EMERGENCY)
    if _PERSON_DOWN.search(plain) and _ELECTRIC_CONTEXT.search(plain):
        return Hazard("person_down", EMERGENCY)
    if _is_powerline(plain):
        return Hazard("powerline", EMERGENCY)
    if _STREET.search(plain) and _OUTAGE.search(plain):
        return Hazard("street_outage", REDIRECT)
    if _is_wire_hazard(plain):
        return Hazard("wire", EMERGENCY)
    if _SHOCK.search(plain) and not _NOT_SHOCK.search(plain):
        return Hazard("shock", EMERGENCY)
    if _FIRE.search(plain) or smoke_alarm_sounding(lowered):
        return Hazard("fire", EMERGENCY)
    if _BATTERY.search(plain) and _BATTERY_STATE.search(plain):
        return Hazard("battery", URGENT)
    if _WATER_HAZARD.search(plain) or _FLOODED.search(plain):
        if not re.search(r"\blight rain\b", plain) or _FLOODED.search(plain):
            return Hazard("water", URGENT)
    if _OVERHEAT.search(plain) and _OVERHEAT_CONTEXT.search(plain):
        return Hazard("overheating", URGENT)
    return None


def is_hazard(text: str | None) -> bool:
    """A real hazard (emergency or urgent). A whole-street outage is not."""
    hazard = classify_hazard(text)
    return hazard is not None and hazard.tier != REDIRECT


def _is_powerline(lowered: str) -> bool:
    if _POWERLINE.search(lowered):
        return True
    # Typos: "pworline", "powrline", "power lnie".
    words = re.findall(r"[a-z]+", lowered)
    candidates = words + [a + b for a, b in zip(words, words[1:])]
    for word in candidates:
        token = word[:-1] if word.endswith("s") else word
        if token.startswith("p") and 7 <= len(token) <= 11:
            if SequenceMatcher(None, token, "powerline").ratio() >= 0.8:
                return True
    return False


def _is_wire_hazard(lowered: str) -> bool:
    if _NOT_POWER_CABLE.search(lowered):
        return False
    if _WIRE_HAZARD.search(lowered):
        return True
    return bool(_TREE.search(lowered) and _WIRE_WORD.search(lowered))
