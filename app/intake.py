"""Accept a loose answer to the intake question that was just asked.

The keyword reader only stores a job when it hits an exact alias, and it only
stores a description once the message is six words or longer. "wiring is the
issue" matches neither, so the field stays empty and the same question goes
out again. Gemini is given the thread, but a null tool result or a timeout
falls through to that reader and the loop continues.

This layer treats the customer's words as the answer to the question they
were just asked. A partial answer is stored. A reply that still means nothing
gets one different, more specific question, then we keep what we have and
move on.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.business import Business, load_business
from app.extract import extract_facts, extract_job_type, extract_name, extract_preferred_time, extract_suburb, extract_urgency
from app.scoring import merge_facts

_SENTENCE = re.compile(r"(?<=[.!?])\s+")

# Filler that does not name a job, a place, or a person.
_FILLER = {
    "a",
    "about",
    "am",
    "an",
    "and",
    "are",
    "be",
    "cheers",
    "did",
    "do",
    "doesnt",
    "doesn't",
    "dont",
    "don't",
    "dunno",
    "fine",
    "for",
    "hello",
    "help",
    "hey",
    "hi",
    "hiya",
    "hmm",
    "hm",
    "huh",
    "i",
    "idk",
    "idea",
    "im",
    "i'm",
    "is",
    "isn't",
    "isnt",
    "issue",
    "it",
    "its",
    "it's",
    "just",
    "k",
    "know",
    "maybe",
    "my",
    "nah",
    "no",
    "nope",
    "not",
    "of",
    "ok",
    "okay",
    "or",
    "our",
    "please",
    "pls",
    "problem",
    "something",
    "stuff",
    "sure",
    "thank",
    "thanks",
    "thankyou",
    "that",
    "the",
    "there",
    "things",
    "this",
    "to",
    "uh",
    "um",
    "unsure",
    "whatever",
    "what",
    "work",
    "working",
    "yeah",
    "yep",
    "yes",
    "you",
    "yup",
}

_UNCLEAR_PHRASES = {
    "a problem",
    "an issue",
    "dont know",
    "don't know",
    "dunno",
    "hello",
    "hey",
    "hi",
    "huh",
    "i don't know",
    "i dont know",
    "idk",
    "n/a",
    "na",
    "nah",
    "no",
    "no idea",
    "nope",
    "not sure",
    "not working",
    "ok",
    "ok thanks",
    "okay",
    "sure",
    "thank you",
    "thanks",
    "the issue",
    "the problem",
    "unsure",
    "what",
    "whatever",
    "yes",
    "yes please",
    "yep",
}

_PROBLEM_WORDS = {
    "breaker",
    "breakers",
    "broken",
    "burning",
    "dead",
    "fault",
    "faulty",
    "flickering",
    "globe",
    "globes",
    "lamp",
    "lamps",
    "light",
    "lights",
    "outage",
    "outlet",
    "power",
    "rewire",
    "rewiring",
    "shock",
    "smoke",
    "smoking",
    "socket",
    "spark",
    "sparking",
    "switchboard",
    "tripped",
    "trips",
    "tripping",
    "wire",
    "wires",
    "wiring",
}

_PROBLEM_PHRASES = (
    "not working",
    "not work",
    "n't working",
    "doesn't work",
    "doesnt work",
    "don't work",
    "dont work",
    "is the issue",
    "is the problem",
    "is the fault",
)

# Words the configured aliases miss. "lights" is not "lighting".
_LOOSE_JOBS = (
    ("power_point", ("light", "lights", "lamp", "lamps", "globe", "globes")),
    ("fault", ("trips", "tripped", "breaker", "breakers", "outage")),
)

_TIME_WORDS = {
    "afternoon",
    "arvo",
    "asap",
    "evening",
    "later",
    "morning",
    "soon",
    "today",
    "tomorrow",
    "tonight",
    "whenever",
}

_ISSUE_TAIL = re.compile(
    r"^(?:(?:it'?s|it is|the|a|my|our)\s+)*(?P<body>[a-z0-9][a-z0-9' -]*?)\s+"
    r"(?:is|are)\s+(?:(?:the|a)\s+)?(?:issue|problem|fault|trouble)\b[.!]*$",
    re.IGNORECASE,
)
_ISSUE_HEAD = re.compile(
    r"^(?:(?:the|a)\s+)?(?:issue|problem|fault|trouble)\s+(?:is|are)\s+(?P<body>.+?)[.!]*$",
    re.IGNORECASE,
)
_ITS = re.compile(
    r"^it'?s\s+(?:(?:the|a)\s+)?(?P<body>[a-z0-9][a-z0-9' -]*)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class _Field:
    key: str
    ask: str
    clarify: str
    cues: tuple[str, ...]


FIELDS = (
    _Field(
        "job_type",
        "What needs doing — a fault, a power point, a switchboard, or something else?",
        "What is it — wiring, lights, a power point, a switchboard, or something else?",
        (
            "what needs doing",
            "what's the problem",
            "what is the problem",
            "what's wrong",
            "what is wrong",
            "what sort of work",
            "what is it",
        ),
    ),
    _Field(
        "suburb",
        "Which suburb is the job in?",
        "Which suburb name should I write down? Newtown or Marrickville is enough.",
        (
            "which suburb",
            "what suburb",
            "suburb you're",
            "suburb name",
            "suburb is the job",
        ),
    ),
    _Field(
        "urgency",
        "Is this an emergency, is it urgent, or can it wait?",
        "Should someone come today, is it urgent, or can it wait for a free visit?",
        (
            "is this an emergency",
            "can it wait",
            "how urgent",
            "is it urgent",
            "come today",
        ),
    ),
    _Field(
        "description",
        "What's going on, in a sentence or two?",
        "What exactly is wrong — wiring, the lights, or a breaker tripping?",
        (
            "what's going on",
            "what is going on",
            "sentence or two",
            "what exactly is wrong",
        ),
    ),
    _Field(
        "contact_name",
        "What name should I put on the job?",
        "What first name should the electrician ask for?",
        (
            "what name",
            "your name",
            "name should",
            "first name",
            "put on the job",
        ),
    ),
)

_BY_KEY = {field.key: field for field in FIELDS}


@dataclass
class IntakeTurn:
    """Facts gained from this reply, the one question to ask next, and cursor state."""

    facts: dict
    question: str | None
    intake: dict
    force_question: bool
    prompt: str | None


def question_for(key: str, *, clarify: bool = False) -> str:
    field = _BY_KEY[key]
    return field.clarify if clarify else field.ask


def empty_intake() -> dict:
    return {"probed": [], "waived": []}


def normalise_intake(raw: dict | None) -> dict:
    """Copy the probe / waive cursor. Unknown keys are dropped."""
    source = raw or {}
    state = empty_intake()
    for key in ("probed", "waived"):
        for item in source.get(key) or []:
            if item in _BY_KEY and item not in state[key]:
                state[key].append(item)
    return state


def apply_intake(
    text: str,
    known: dict | None,
    history: list[dict] | None,
    business: Business | None = None,
    intake: dict | None = None,
    model_facts: dict | None = None,
) -> IntakeTurn:
    """Fill empty fields from this reply and choose the next question.

    `model_facts` wins over the keyword reader. Nothing already stored is cleared.
    """
    business = business or load_business()
    state = normalise_intake(intake)
    base = merge_facts(known or {}, model_facts or {})
    extracted = extract_facts(text or "", business)
    additions: dict = {}
    for key, value in extracted.items():
        if value and not _filled(base, key):
            additions[key] = value
    working = merge_facts(base, additions)
    asked = asked_fields(_last_assistant(history))
    for key, value in _loose_additions(text or "", business, asked, working, extracted).items():
        if value and not _filled(working, key):
            additions[key] = value
            working = merge_facts(base, additions)

    missed = _missed_field(asked, additions, working, state, known or {})
    force = False
    if missed and not _already_probed(missed, state, history):
        if missed not in state["probed"]:
            state["probed"].append(missed)
        question = question_for(missed, clarify=True)
        force = True
    elif missed:
        effort = best_effort(missed, text or "", business)
        if effort and not _filled(working, missed):
            additions[missed] = effort
            if missed == "job_type" and not _filled(working, "description"):
                additions.setdefault("description", problem_description(text or "") or effort)
            working = merge_facts(base, additions)
        elif missed not in state["waived"]:
            state["waived"].append(missed)
        question = next_question(working, state)
        force = True
    else:
        question = next_question(working, state)

    return IntakeTurn(
        facts=additions,
        question=question,
        intake=state,
        force_question=force,
        prompt=question if force else None,
    )


def next_question(facts: dict, intake: dict | None = None) -> str | None:
    """The next standard question, skipping facts we have or have given up on."""
    waived = set(normalise_intake(intake)["waived"])
    for field in FIELDS:
        if field.key in waived or _filled(facts, field.key):
            continue
        return field.ask
    return None


def asked_fields(message: str | None) -> list[str]:
    """Intake fields the last assistant message was asking about, in ask order."""
    if not message:
        return []
    lowered = message.casefold()
    found: list[str] = []
    for field in FIELDS:
        if field.ask.casefold() in lowered or field.clarify.casefold() in lowered:
            found.append(field.key)
            continue
        if any(cue in lowered for cue in field.cues):
            found.append(field.key)
    return found


def render_reply(model_reply: str | None, outcome: IntakeTurn, business: Business, filled: dict) -> str:
    """Keep a model reply that already asks for the right gap. Replace a repeated one."""
    reply = (model_reply or "").strip()
    question = outcome.question
    desired = _field_of_question(question)
    targets = _question_targets(reply)
    filled_keys = {key for key in _BY_KEY if _filled(filled, key)}
    waived = set(outcome.intake.get("waived") or [])
    blocked = filled_keys | waived

    if outcome.force_question:
        return _join_question(reply, question, outcome.facts, business)
    if question and desired and desired in targets and desired not in blocked:
        return reply or question
    if question and (targets & blocked or (targets and desired not in targets) or not targets):
        return _join_question(reply, question, outcome.facts, business)
    if not question and targets & blocked:
        return _statements(reply) or "Thanks — I've noted that."
    return reply


def delivered_intake(before: dict | None, after: dict | None, sent_reply: str | None) -> dict:
    """Drop a follow-up probe that never reached the customer.

    A slot offer or a different question can replace the reply after intake
    decides to clarify. The probe only counts once the specific question was sent.
    Giving up on a field still sticks, so a swallowed question cannot loop.
    """
    state = normalise_intake(after)
    prior = set(normalise_intake(before)["probed"])
    sent = sent_reply or ""
    kept = []
    for field in state["probed"]:
        if field not in prior and question_for(field, clarify=True) not in sent:
            continue
        kept.append(field)
    state["probed"] = kept
    return state


def problem_description(text: str) -> str | None:
    """The problem in the customer's words. 'wiring is the issue' becomes 'wiring'."""
    if is_unclear(text):
        return None
    core = issue_core(text)
    if core:
        return core
    if not looks_like_problem(text):
        return None
    phrase = _tidy(text)
    if not phrase:
        return None
    return phrase[:240]


def issue_core(text: str) -> str | None:
    """The noun in 'wiring is the issue' or 'the problem is wiring'."""
    tidy = _tidy(text)
    if not tidy:
        return None
    for pattern in (_ISSUE_TAIL, _ISSUE_HEAD, _ITS):
        match = pattern.match(tidy)
        if not match:
            continue
        body = _tidy(match.group("body"))
        if not body or is_unclear(body):
            continue
        if pattern is _ITS and not _problem_body(body):
            continue
        return body[:80]
    return None


def looks_like_problem(text: str, business: Business | None = None) -> bool:
    if issue_core(text):
        return True
    lowered = (text or "").casefold()
    if any(phrase in lowered for phrase in _PROBLEM_PHRASES):
        return True
    words = set(re.findall(r"[a-z0-9']+", lowered))
    if words & _PROBLEM_WORDS:
        return True
    if business is not None and extract_job_type(text, business):
        return True
    return False


def is_unclear(text: str | None) -> bool:
    """True when the reply does not name a thing we can store."""
    cleaned = _norm(text)
    if not cleaned:
        return True
    if not re.search(r"[a-z0-9]", cleaned):
        return True
    if cleaned in _UNCLEAR_PHRASES:
        return True
    words = re.findall(r"[a-z0-9']+", cleaned)
    if not words:
        return True
    return all(word in _FILLER for word in words)


def best_effort(field: str, text: str, business: Business) -> str | None:
    """Last chance after the specific follow-up. Unclear text is not stored."""
    if field not in _BY_KEY or is_unclear(text):
        return None
    if field == "urgency":
        return _loose_urgency(text)
    if field == "contact_name":
        return _loose_name(text, business)
    if field == "suburb":
        return _place_name(text, business)
    if field == "job_type":
        return loose_job_id(text, business) or issue_core(text) or _tidy(text)[:80] or None
    if field == "description":
        return problem_description(text) or _tidy(text)[:240] or None
    return None


def loose_job_id(text: str, business: Business) -> str | None:
    found = extract_job_type(text, business)
    if found:
        return found
    lowered = (text or "").casefold()
    for job_id, words in _LOOSE_JOBS:
        for word in words:
            if re.search(rf"\b{re.escape(word)}\b", lowered):
                return job_id
    return None


def _loose_additions(text: str, business: Business, asked: list[str], working: dict, extracted: dict) -> dict:
    if not text or is_unclear(text):
        return {}
    updates: dict = {}
    elsewhere = _other_only(text, extracted, business)
    described = None if _filled(working, "description") else problem_description(text)
    if not _filled(working, "job_type"):
        job_id = loose_job_id(text, business)
        if job_id:
            updates["job_type"] = job_id
        elif not elsewhere:
            core = issue_core(text)
            if core:
                updates["job_type"] = core
            elif "job_type" in asked:
                updates["job_type"] = _tidy(text)[:80]
    if described and not elsewhere and not _filled(working, "description"):
        updates["description"] = described
    elif (
        "description" in asked
        and not _filled(working, "description")
        and "description" not in updates
        and not elsewhere
    ):
        phrase = _tidy(text)
        if phrase:
            updates["description"] = phrase[:240]
    if "suburb" in asked and not _filled(working, "suburb") and "suburb" not in updates:
        place = _place_name(text, business)
        if place:
            updates["suburb"] = place
    if not _filled(working, "urgency") and "urgency" not in updates:
        urgency = _loose_urgency(text)
        if urgency:
            updates["urgency"] = urgency
    if "contact_name" in asked and not _filled(working, "contact_name"):
        name = _loose_name(text, business)
        if name:
            updates["contact_name"] = name
    if updates.get("job_type") and not _filled(working, "description") and "description" not in updates:
        extra = described or issue_core(text)
        if extra:
            updates["description"] = extra
    return {key: value for key, value in updates.items() if value}


def _loose_urgency(text: str) -> str | None:
    found = extract_urgency(text)
    if found:
        return found
    lowered = (text or "").casefold()
    if re.search(r"\b(can wait|it can wait|able to wait|no hurry|when you can)\b", lowered):
        return "planned"
    if re.search(r"\b(soon|straight away|straightaway)\b", lowered):
        return "urgent"
    return None


def _loose_name(text: str, business: Business) -> str | None:
    found = extract_name(text)
    if found:
        return found
    if is_unclear(text) or looks_like_problem(text, business):
        return None
    cleaned = _tidy(text)
    cleaned = re.sub(r"^(?:it'?s|this is|i am|i'm)\s+", "", cleaned, flags=re.IGNORECASE)
    if not re.fullmatch(r"[A-Za-z]+(?:\s+[A-Za-z]+){0,2}", cleaned or ""):
        return None
    parts = cleaned.split()
    if any(len(part) < 2 for part in parts):
        return None
    suburbs = {suburb.casefold() for suburb in business.suburbs}
    for part in parts:
        token = part.casefold()
        if token in _FILLER or token in _PROBLEM_WORDS or token in suburbs or token in _TIME_WORDS:
            return None
    return " ".join(part.capitalize() for part in parts)


def _place_name(text: str, business: Business) -> str | None:
    found = extract_suburb(text, business)
    if found and (found in business.suburbs or found.casefold() in {item.casefold() for item in business.suburbs}):
        return found
    if found and len(_tidy(text).split()) <= 4 and not looks_like_problem(text, business):
        return found
    if is_unclear(text) or looks_like_problem(text, business):
        return None
    tidy = _tidy(text)
    if not re.fullmatch(r"[A-Za-z]+(?:\s+[A-Za-z]+){0,3}", tidy or ""):
        return None
    return " ".join(part.capitalize() for part in tidy.split())


def _other_only(text: str, extracted: dict, business: Business) -> bool:
    """A suburb, a time, a name, or an urgency, with no problem in it."""
    if looks_like_problem(text, business) or issue_core(text):
        return False
    if _time_only(text):
        return True
    words = _tidy(text).split()
    if extracted.get("urgency") and len(words) <= 4:
        return True
    if extracted.get("contact_name") and not extracted.get("job_type") and len(words) <= 6:
        return True
    if extracted.get("suburb") and not extracted.get("job_type") and len(words) <= 5:
        return True
    if extracted.get("preferred_time") and len(words) <= 4:
        return True
    return False


def _time_only(text: str) -> bool:
    if looks_like_problem(text):
        return False
    words = re.findall(r"[a-z']+", (text or "").casefold())
    if not words:
        return False
    if extract_preferred_time(text) and len(_tidy(text).split()) <= 4:
        return True
    return all(word in _TIME_WORDS or word in _FILLER for word in words) and any(word in _TIME_WORDS for word in words)


def _problem_body(body: str) -> bool:
    words = set(re.findall(r"[a-z0-9']+", body.casefold()))
    return bool(words & _PROBLEM_WORDS) or any(phrase in body.casefold() for phrase in _PROBLEM_PHRASES)


def _missed_field(asked: list[str], additions: dict, working: dict, state: dict, known: dict) -> str | None:
    """The asked field this reply still didn't fill, if it filled none of them.

    A fact the model stored from this same message counts as an answer, even
    though it is not in `additions` (those are only the gaps we filled).
    """
    if not asked:
        return None
    for field in asked:
        if additions.get(field):
            return None
        if _filled(working, field) and not _filled(known, field):
            return None
    waived = set(state.get("waived") or [])
    for field in asked:
        if field in waived or _filled(working, field):
            continue
        return field
    return None


def _already_probed(field: str, state: dict, history: list[dict] | None) -> bool:
    if field in (state.get("probed") or []):
        return True
    clarify = question_for(field, clarify=True).casefold()
    for item in history or []:
        if item.get("role") != "assistant":
            continue
        if clarify in str(item.get("content") or "").casefold():
            return True
    return False


def _last_assistant(history: list[dict] | None) -> str:
    for item in reversed(history or []):
        if item.get("role") == "assistant":
            return str(item.get("content") or "")
    return ""


def _field_of_question(question: str | None) -> str | None:
    if not question:
        return None
    for field in FIELDS:
        if question == field.ask or question == field.clarify:
            return field.key
    return None


def _question_targets(reply: str) -> set[str]:
    targets: set[str] = set()
    for sentence in _sentences(reply):
        if "?" not in sentence:
            continue
        lowered = sentence.casefold()
        for field in FIELDS:
            if field.ask.casefold() in lowered or field.clarify.casefold() in lowered:
                targets.add(field.key)
                continue
            if any(cue in lowered for cue in field.cues):
                targets.add(field.key)
    return targets


def _join_question(model_reply: str, question: str | None, additions: dict, business: Business) -> str:
    body = _statements(model_reply)
    ack = _acknowledgement(additions, business)
    if ack and ack not in body:
        body = f"{body} {ack}".strip() if body else ack
    if question:
        if question in body:
            return body
        return f"{body} {question}".strip() if body else question
    return body or "Thanks — I've noted that."


def _acknowledgement(facts: dict, business: Business) -> str:
    job = facts.get("job_type")
    if job:
        label = business.job_label(job) or _cap(str(job))
        return f"{label} — I'll put that on the job."
    description = facts.get("description")
    if description:
        return f"{_cap(str(description))} — I've got that."
    suburb = facts.get("suburb")
    if suburb:
        return f"{suburb} — noted."
    name = facts.get("contact_name")
    if name:
        return f"Thanks, {name}."
    return ""


def _statements(text: str) -> str:
    return " ".join(sentence for sentence in _sentences(text) if "?" not in sentence).strip()


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in _SENTENCE.split((text or "").strip()) if part.strip()]


def _filled(facts: dict | None, key: str) -> bool:
    if not facts:
        return False
    value = facts.get(key)
    return value is not None and str(value).strip() != ""


def _tidy(text: str | None) -> str:
    cleaned = " ".join((text or "").strip().split())
    cleaned = cleaned.strip(" \t\"'`.,!?;:")
    cleaned = re.sub(r"\b(?:please|thanks|thank you|cheers)\b", "", cleaned, flags=re.IGNORECASE)
    return " ".join(cleaned.split()).strip(" \t\"'`.,!?;:")


def _norm(text: str | None) -> str:
    cleaned = (text or "").replace("’", "'").replace("`", "'").casefold()
    cleaned = " ".join(cleaned.split())
    return cleaned.strip(" \t\"'`.,!?;:")


def _cap(text: str) -> str:
    text = text.strip()
    if not text:
        return text
    return text[0].upper() + text[1:]
