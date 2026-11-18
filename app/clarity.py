"""Is the customer asking for electrical work clearly enough to take the job? If not, ask.

Every turn ends with one intent:

    service_request  they describe electrical work or a fault ("downlights in the kitchen")
    answer           they answer the question just asked ("Newtown", "2", "Sam", "yes")
    question         a question or request the bot handles elsewhere (price, area, a visit time)
    follow_up        about a job already on file
    greeting         hello, thanks, ok
    vague_request    they want an electrician but not what for ("need a sparky")
    unclear          we can't tell what they mean ("hha", "asdf", "?")
    not_electrical   another trade or nothing to do with us ("blocked toilet", selling SEO)

The last three get a clarifying question instead of intake. Nothing from that message is
stored except facts a strict reader can confirm (a configured suburb, a time, "my name is ...").
"question", "greeting" and "follow_up" keep the reply but also store only strict facts, so
"do you work weekends?" never becomes a job type.

Who decides, strongest first:
1. Deterministic evidence: a hazard, a guardrail reply, a configured job word, a slot or
   menu number, a configured suburb when the suburb was asked, a visit or cancel request,
   a request for a person. These are tested and never second-guessed by the model.
2. The model's `intent` field (Gemini or Claude), which reads the whole conversation.
3. Keyword rules, only when there is no model (offline) or it gave no intent.

Asking again has a limit: the second unclear message in a row gets a numbered menu of the
configured jobs, the third hands the chat to a person, so nobody is stuck in a loop.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.business import Business
from app.extract import extract_facts, extract_job_type, extract_preferred_time, extract_suburb, extract_urgency
from app.hazards import classify_hazard

INTENTS = (
    "service_request",
    "answer",
    "question",
    "follow_up",
    "greeting",
    "vague_request",
    "unclear",
    "not_electrical",
)
CLARIFY = frozenset({"vague_request", "unclear", "not_electrical"})
# Intents whose message adds only strictly confirmed facts (no loose reading of free text).
STRICT = frozenset({"question", "follow_up", "greeting"}) | CLARIFY

MENU_AFTER = 2  # the second unclear message in a row gets the numbered job menu
HANDOFF_AFTER = 3  # the third hands the chat to a person

_GREETING_WORDS = {
    "hi", "hello", "hey", "hiya", "gday", "g'day", "morning", "afternoon", "evening", "thanks",
    "thank", "you", "cheers", "ta", "ok", "okay", "k", "yes", "yep", "yeah", "no", "nope", "nah",
    "sure", "great", "good", "cool", "there", "mate", "please",
}
_YES_NO = {"yes", "yep", "yeah", "y", "no", "nope", "nah", "n", "ok", "okay", "sure"}
# Wanting an electrician, without saying for what.
_VAGUE = re.compile(
    r"\b(?:need|want|looking for|after|get)\b.{0,20}\b(?:an? )?(?:electrician|sparky|sparkie|tradie|quote|job done|"
    r"some work|work done|help|someone)\b|^(?:help|electrician|sparky|quote|urgent|asap)[.!?]*$|"
    r"\b(?:electrical (?:work|stuff|issue|issues|problem|job)|power (?:issue|problem)s?|"
    r"can you (?:help|come)|are you available|do you do electrical)\b"
)
# Words that say there is electrical work or a fault to talk about.
_ELECTRICAL = re.compile(
    r"\b(?:wir(?:e|es|ing)|cables?|lights?|lighting|lamps?|globes?|bulbs?|downlights?|switch\w*|sockets?|"
    r"outlets?|power ?points?|plugs?|gpos?|breakers?|fuses?|rcds?|safety switch|tripp\w*|trips|tripped|"
    r"meter|switchboard|board|power|electric\w*|volt\w*|amps?|circuit\w*|earth\w*|surge|"
    r"oven|cooktop|stove|heater|aircon|air ?con\w*|split system|fans?|exhaust|hot water|appliance\w*|"
    r"doorbell|intercom|sensor|cctv|antenna|data point|solar|inverter|battery|charger|ev|"
    r"flicker\w*|buzz\w*|spark(?:s|ing|ed)?|smok\w*|burn\w*|shock\w*|tingl\w*|"
    r"not working|stopped working|doesn'?t work|won'?t (?:work|turn on)|dead|broken|faulty|fault|blown)\b"
)
# Other trades and things we don't do. A hazard check runs first, so water near electrics still wins.
_OTHER_TRADE = re.compile(
    r"(?:leak\w*|drip\w*)[^.!?]{0,30}\b(?:taps?|pipes?|toilet|cistern|gutters?)\b|"
    r"\b(?:taps?|pipes?)\b[^.!?]{0,20}(?:leak\w*|drip\w*)|"
    r"\b(?:plumb\w*|toilet|blocked (?:drain|sink|toilet|pipe)|drains?|leaking (?:tap|pipe|roof)|dripping tap|"
    r"gutters?|roof(?:er|ing)?|tiles? (?:need|to be)|tiling|tiler|paint(?:er|ing)|carpet\w*|carpenter|"
    r"locksmith|locked out|pest\w*|termites?|lawn\w*|garden(?:er|ing)|landscap\w*|fence repair|"
    r"mechanic|car service|rego|removal\w*|cleaner|cleaning service|concret\w*|bricklay\w*|"
    r"seo|marketing services|web ?design|backlinks|crypto|investment opportunity)\b"
)
_VISIT = re.compile(
    r"\b(?:can someone come|come (?:out|over|today|tomorrow)|next available|book (?:me )?in|book a visit|"
    r"available (?:today|tomorrow)|when can you come)\b"
)
_CANCEL = re.compile(r"\b(?:cancel|reschedule|move my (?:booking|appointment))\b")
_NAME_ANSWER = re.compile(r"^(?:it'?s |i'?m |my name is |this is )?[a-z][a-z'\-]+(?: [a-z][a-z'\-]+){0,2}[.!]?$")


@dataclass(frozen=True)
class Assessment:
    intent: str
    source: str  # "evidence", "model" or "rules"
    clarify: bool
    strict: bool


def assess(
    text: str | None,
    *,
    model_intent: str | None,
    known: dict,
    pending: list[str],
    business: Business,
    handled: bool = False,
    booked: bool = False,
    menu_choice: str | None = None,
    wants_person: bool = False,
    opened_new_job: bool = False,
    previous: str | None = None,
) -> Assessment:
    """The final intent for one message, with deterministic evidence first, then the model, then rules.

    `handled` means a guardrail already replaces the reply (price, complaint, DIY, ...).
    `pending` is the intake fields the bot's last message asked for.
    """
    evidence = evidence_intent(
        text, known=known, pending=pending, business=business, handled=handled, booked=booked,
        menu_choice=menu_choice, wants_person=wants_person, opened_new_job=opened_new_job,
    )
    if evidence:
        intent, source = evidence, "evidence"
    elif model_intent in INTENTS:
        intent, source = model_intent, "model"
    else:
        intent, source = rule_intent(text, known=known, pending=pending, business=business), "rules"
    lowered = " ".join((text or "").casefold().split())
    if (
        source == "model"
        and intent in ("service_request", "answer")
        and not business.resolve_job_type(known.get("job_type"))
        and looks_like_gibberish(lowered)
        and not _ELECTRICAL.search(lowered)
    ):
        # Rules may overrule the model only towards asking: a clarifying question is cheap, a junk job isn't.
        intent, source = "unclear", "rules"
    if intent == "unclear" and source == "rules" and previous and _same(previous, text):
        # They sent the same thing again after we said we didn't catch it: believe them.
        intent = "answer"
    # A guardrail reply or a person stepping in already answers the message: never clarify over it.
    clarify = intent in CLARIFY and not handled and not wants_person
    return Assessment(intent=intent, source=source, clarify=clarify, strict=intent in STRICT)


def evidence_intent(
    text: str | None,
    *,
    known: dict,
    pending: list[str],
    business: Business,
    handled: bool = False,
    booked: bool = False,
    menu_choice: str | None = None,
    wants_person: bool = False,
    opened_new_job: bool = False,
) -> str | None:
    """Deterministic signs that a message is clear. None when there are none."""
    raw = (text or "").strip()
    if not raw:
        return None
    lowered = " ".join(raw.casefold().split())
    hazard = classify_hazard(raw)
    if hazard is not None:
        # Safety first: "sparks!!" is short but perfectly clear. A street outage is a question for the distributor.
        return "question" if hazard.tier == "redirect" else "service_request"
    if booked or menu_choice:
        return "answer"
    if extract_job_type(raw, business):
        return "service_request"
    if handled or wants_person:
        return "question"
    if opened_new_job:
        # "I've got another job for you": routing already opened a new job and the model asks what it is.
        return "follow_up"
    if _VISIT.search(lowered) or _CANCEL.search(lowered):
        return "question"
    if "suburb" in pending and _configured_suburb(raw, business) and len(lowered.split()) <= 4:
        # "Newtown" or "it's in Newtown". A longer message may also be a vague request, so the model decides.
        return "answer"
    if "preferred_time" in pending and extract_preferred_time(raw):
        return "answer"
    if "urgency" in pending and extract_urgency(raw):
        return "answer"
    return None


def rule_intent(text: str | None, *, known: dict, pending: list[str], business: Business) -> str:
    """A keyword guess for when there is no model. Weaker than the model's judgement."""
    raw = (text or "").strip()
    lowered = " ".join(raw.casefold().split())
    words = re.findall(r"[a-z0-9']+", lowered)
    job_known = bool(business.resolve_job_type(known.get("job_type")))
    if not words:
        return "unclear"
    if pending and set(words) <= _YES_NO:
        return "answer"
    if set(words) <= _GREETING_WORDS:
        return "greeting"
    if _OTHER_TRADE.search(lowered):
        return "not_electrical"
    if _VAGUE.search(lowered) and not _ELECTRICAL.search(_VAGUE.sub(" ", lowered)):
        # "need some electrical stuff done": the only electrical word is in the vague phrase itself.
        return "vague_request"
    if _ELECTRICAL.search(lowered):
        return "service_request" if not job_known else "answer"
    if job_known and pending[:1] == ["suburb"] and len(words) == 1 and len(words[0]) <= 3 and raw == raw.lower():
        # "hha" for the suburb. A real short suburb typed in capitals ("Kew") is an answer, and a
        # customer who repeats a lowercase one is believed (see `assess`).
        return "unclear"
    if job_known:
        # Mid-intake, a plain answer to "what name?" or "what's going on?" is given the benefit of the doubt.
        if ("contact_name" in pending and _NAME_ANSWER.match(lowered)) or len(words) >= 3:
            return "answer"
        # A real three-letter suburb like "Kew" is a fine answer here; only clear nonsense is unclear.
        return "unclear" if looks_like_gibberish(lowered, short_is_unclear=False) else "answer"
    if "?" in raw:
        return "question"
    if len(words) <= 3:
        return "unclear"
    return "vague_request"


def _same(left: str | None, right: str | None) -> bool:
    return " ".join((left or "").casefold().split()) == " ".join((right or "").casefold().split())


def looks_like_gibberish(lowered: str, *, short_is_unclear: bool = True) -> bool:
    """A short message that can't be read: "hha", "asdfgh", "?", "kkkk".

    With `short_is_unclear`, any lone word of three letters or fewer counts too ("hha" as a
    first message). Mid-conversation it doesn't, because "Kew" or "Ryde" can be an answer.
    """
    words = re.findall(r"[a-z]+", lowered)
    if not words:
        return True
    if len(words) > 3:
        return False
    for word in words:
        if word in _GREETING_WORDS:
            continue
        if not re.search(r"[aeiouy]", word) or re.search(r"(.)\1\1", word) or re.search(r"[^aeiouy]{5,}", word):
            return True
        if short_is_unclear and len(word) <= 3:
            return True
    return False


def strict_facts(text: str | None, business: Business) -> dict:
    """Only facts a strict reader can confirm from this message: configured job and suburb words,
    a time, an urgency, a stated name, a long description."""
    facts = extract_facts(text or "", business)
    facts["suburb"] = _configured_suburb(text or "", business)
    # A description comes from a real answer about the job, never from a question or an unclear message.
    facts["description"] = None
    return {key: value for key, value in facts.items() if value}


def menu_choice(text: str | None, menu: list[str] | None) -> str | None:
    """The job id picked from the numbered menu ("2" or "2." or "option 2"), if one is open."""
    if not menu:
        return None
    match = re.fullmatch(r"\s*(?:option\s*)?(\d{1,2})[.)]?\s*", text or "", flags=re.IGNORECASE)
    if not match:
        return None
    index = int(match.group(1)) - 1
    return menu[index] if 0 <= index < len(menu) else None


def clarify_reply(
    intent: str,
    *,
    business: Business,
    known: dict,
    pending_question: str | None,
    model_question: str | None,
    streak: int,
) -> tuple[str, list[str], bool]:
    """(reply, job menu to remember, hand to a person?) for a message that needs clarifying.

    `streak` counts unclear or vague messages in a row, this one included.
    """
    job_known = bool(business.resolve_job_type(known.get("job_type")))
    question = usable_question(model_question)
    if intent == "not_electrical":
        return (
            question or "Sorry, we're electricians, so that one isn't something we can help with. "
            "Is there any electrical work you need done?",
            [],
            False,
        )
    if streak >= HANDOFF_AFTER:
        # The service adds the hand-over line ("a person from the desk will follow up ...").
        return ("No worries, let's get a person to help.", [], True)
    if job_known:
        follow = pending_question or "Could you say that another way?"
        return (question or f"Sorry, I didn't quite catch that. {follow}", [], False)
    if streak >= MENU_AFTER:
        menu = [job.id for job in business.job_types]
        lines = "\n".join(f"{index}. {job.label}" for index, job in enumerate(business.job_types, start=1))
        return (
            "Sorry, I'm still not sure what you need. Reply with a number, or tell me in your own words:\n" + lines,
            menu,
            False,
        )
    examples = job_examples(business)
    if question:
        return (question, [], False)
    if intent == "vague_request":
        return (f"Happy to help. What's the electrical job? For example: {examples}.", [], False)
    return (f"Sorry, I didn't quite catch that. What do you need an electrician for? For example: {examples}.", [], False)


def usable_question(text: str | None) -> str | None:
    """The model's clarifying question, if it is one short question and nothing else risky."""
    cleaned = " ".join((text or "").split())
    if not cleaned or not cleaned.endswith("?") or len(cleaned) > 240:
        return None
    if "$" in cleaned or len(re.split(r"(?<=[.!?])\s+", cleaned)) > 3:
        return None
    return cleaned


def job_examples(business: Business, limit: int = 4) -> str:
    labels = [job.label.lower() for job in business.job_types[:limit]]
    if len(labels) <= 1:
        return "".join(labels)
    return ", ".join(labels[:-1]) + " or " + labels[-1]


def _configured_suburb(text: str, business: Business) -> str | None:
    found = extract_suburb(text, business)
    if found and found.casefold() in {suburb.casefold() for suburb in business.suburbs}:
        return business.canonical_suburb(found)
    return None
