"""Qualifiers: Gemini or Claude via a tool call, and a keyword stand-in.

Gemini is the recommended provider. It is forced to call `record_job_intake`
(`function_calling_config` mode ANY) so each turn returns a reply plus facts.
A rate limit or any other Gemini error uses the offline qualifier for that
turn instead of failing the webhook.

Claude Sonnet 5.5 rejects forced tool choice (`tool_choice` any/tool). That
call uses `auto` plus a strict schema, and the prompt tells the assistant to
call the tool every turn. If the model answers in plain text anyway, that
text is the reply and no new facts are stored.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import anthropic

from app.business import Business, load_business
from app.extract import extract_facts
from app.hazards import MODEL_HAZARDS, classify_hazard, hazard_from_kind
from app.intake import apply_intake, normalise_intake
from app.scoring import canonical_facts, merge_facts, wants_human

logger = logging.getLogger("tradie.llm")

QUALIFY_TOOL = {
    "name": "record_job_intake",
    "description": (
        "Record what the customer has actually said about the job, and the "
        "exact WhatsApp reply to send. Call this on every turn. Use null for "
        "any fact they have not stated. Do not invent a suburb, a time, or a name. "
        "Never put a price in reply."
    ),
    "strict": True,
    "input_schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "reply": {
                "type": "string",
                "description": (
                    "Customer-facing WhatsApp message. One or two short sentences. "
                    "Ask at most one question. Do not mention prices or dollar amounts."
                ),
            },
            "request_human": {
                "type": "boolean",
                "description": "True only when the customer asks to speak with a person.",
            },
            "job_type": {
                "anyOf": [{"type": "string"}, {"type": "null"}],
                "description": (
                    "Configured job type id, or their words when they describe the work "
                    "without a listed name. A short answer still counts. Null only when "
                    "they have not said. The ad is not confirmation."
                ),
            },
            "suburb": {
                "anyOf": [{"type": "string"}, {"type": "null"}],
                "description": "Suburb of the job, or null.",
            },
            "urgency": {
                "anyOf": [{"type": "string"}, {"type": "null"}],
                "description": "emergency, urgent, planned, or null.",
            },
            "hazard": {
                "type": "string",
                "enum": list(MODEL_HAZARDS),
                "description": (
                    "Electrical danger in the customer's latest message. "
                    "wire: a fallen, hanging, exposed, cut or damaged wire or cable, or a tree on one. "
                    "powerline: street powerlines, a power pole, or the service line from the pole. "
                    "shock: a shock or tingle from a tap, appliance or fitting. "
                    "person_down: someone stuck to, or collapsed after touching, something electrical. "
                    "fire: smoke, flames, sparks, burning smell, or a smoke alarm going off for a fire. "
                    "water: water, a leak or flooding near the switchboard, lights, power points or wiring. "
                    "overheating: melting, scorch marks, a hot, buzzing or crackling fitting, a frayed lead. "
                    "battery: a swollen, hissing, leaking or very hot home battery or inverter. "
                    "street_outage: the whole street or area has no power. "
                    "none: anything else, including a beeping smoke alarm, new wiring work, or a price question. "
                    "When unsure between none and a hazard, choose the hazard."
                ),
            },
            "description": {
                "anyOf": [{"type": "string"}, {"type": "null"}],
                "description": (
                    "What they say is wrong, in their words, even a short phrase. "
                    "'wiring is the issue' means wiring. Null only when they have not said."
                ),
            },
            "preferred_time": {
                "anyOf": [{"type": "string"}, {"type": "null"}],
                "description": "When they would like the visit, in their words, or null.",
            },
            "contact_name": {
                "anyOf": [{"type": "string"}, {"type": "null"}],
                "description": "Name for the job, or null. Do not copy the WhatsApp profile.",
            },
        },
        "required": [
            "reply",
            "request_human",
            "job_type",
            "suburb",
            "urgency",
            "hazard",
            "description",
            "preferred_time",
            "contact_name",
        ],
    },
}

_FALLBACK_REPLY = "Could you tell me a little more about the job?"
_MAX_HISTORY = 20
_SAFETY = "If anyone is in danger, or you can smell burning, get out and call 000."


@dataclass
class QualificationDraft:
    """One turn: the WhatsApp reply, plus any facts stated in that turn."""

    reply: str
    request_human: bool = False
    job_type: str | None = None
    suburb: str | None = None
    urgency: str | None = None
    description: str | None = None
    preferred_time: str | None = None
    contact_name: str | None = None
    hazard: str | None = None
    raw: dict | None = None
    intake: dict | None = None
    intake_prompt: str | None = None

    def facts(self) -> dict:
        return {
            "job_type": self.job_type,
            "suburb": self.suburb,
            "urgency": self.urgency,
            "description": self.description,
            "preferred_time": self.preferred_time,
            "contact_name": self.contact_name,
        }


@dataclass
class TurnContext:
    """What a qualifier may look at for a single customer message."""

    history: list[dict]
    known: dict
    ad: dict | None
    profile_name: str | None
    system: str
    intake: dict | None = None


class AnthropicQualifier:
    def __init__(self, api_key: str, model: str, client=None) -> None:
        self.model = model
        self.client = client if client is not None else anthropic.Anthropic(api_key=api_key)

    def qualify(self, turn: TurnContext) -> QualificationDraft:
        # between_tools skips the long up-front thinking pass. Sonnet 5.5
        # rejects thinking type "disabled", and it also rejects forced tool_choice.
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1500,
            system=turn.system,
            tools=[QUALIFY_TOOL],
            tool_choice={"type": "auto"},
            thinking={"type": "between_tools"},
            messages=to_api_messages(turn.history[-_MAX_HISTORY:]),
        )
        return parse_qualification(response)


class GeminiQualifier:
    """Gemini function call. A 404, 429, 503, or 504 tries the next model, then the offline qualifier."""

    def __init__(
        self,
        api_key: str,
        model: str,
        client=None,
        business: Business | None = None,
        fallback_models: list[str] | None = None,
    ) -> None:
        self.model = model
        self.models = model_chain(model, fallback_models or [])
        self._offline = RuleBasedQualifier(business)
        if client is not None:
            self.client = client
        else:
            self.client = gemini_client(api_key)

    def qualify(self, turn: TurnContext) -> QualificationDraft:
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=turn.system,
            tools=[_gemini_tool()],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode="ANY",
                    allowed_function_names=[QUALIFY_TOOL["name"]],
                )
            ),
            max_output_tokens=1500,
            temperature=0,
        )
        contents = to_gemini_contents(turn.history[-_MAX_HISTORY:])
        last_error: Exception | None = None
        for index, model in enumerate(self.models):
            try:
                response = self.client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config,
                )
            except Exception as exc:
                last_error = exc
                if should_try_next_model(exc) and index < len(self.models) - 1:
                    logger.warning(
                        "gemini qualify model %s unavailable (%s); trying the next model",
                        model,
                        exc,
                    )
                    continue
                log_gemini_failure("qualify", exc, model)
                return self._offline.qualify(turn)
            try:
                draft = parse_gemini_qualification(response)
            except Exception as exc:
                log_gemini_failure("qualify parse", exc, model)
                return self._offline.qualify(turn)
            logger.info("gemini qualify answered with model %s", model)
            return draft
        log_gemini_failure(
            "qualify",
            last_error or RuntimeError("no gemini model configured"),
            self.model,
        )
        return self._offline.qualify(turn)


class RuleBasedQualifier:
    """Offline qualifier. Keywords, then a loose read of the question just asked."""

    def __init__(self, business: Business | None = None) -> None:
        self.business = business or load_business()

    def qualify(self, turn: TurnContext) -> QualificationDraft:
        latest = _latest_user_text(turn.history)
        extracted = extract_facts(latest, self.business)
        asked = wants_human(latest)
        if asked:
            reply = (
                f"Absolutely — I'll bring a person from the {self.business.name} desk into this chat."
            )
            raw = {"reply": reply, "request_human": True, **extracted}
            return QualificationDraft(
                reply=reply,
                request_human=True,
                job_type=extracted["job_type"],
                suburb=extracted["suburb"],
                urgency=extracted["urgency"],
                description=extracted["description"],
                preferred_time=extracted["preferred_time"],
                contact_name=extracted["contact_name"],
                hazard=_rule_hazard(latest),
                raw=raw,
                intake=normalise_intake(turn.intake),
            )
        outcome = apply_intake(
            latest,
            turn.known,
            turn.history,
            self.business,
            intake=turn.intake,
            model_facts=extracted,
        )
        combined = merge_facts(extracted, outcome.facts)
        merged = canonical_facts(merge_facts(turn.known, combined), self.business)
        reply = _compose_reply(
            business=self.business,
            latest=latest,
            extracted=combined,
            known=turn.known,
            merged=merged,
            ad=turn.ad,
            profile_name=turn.profile_name,
            history=turn.history,
            asked=False,
            question=outcome.question,
        )
        raw = {"reply": reply, "request_human": False, **combined}
        return QualificationDraft(
            reply=reply,
            request_human=False,
            job_type=combined.get("job_type"),
            suburb=combined.get("suburb"),
            urgency=combined.get("urgency"),
            description=combined.get("description"),
            preferred_time=combined.get("preferred_time"),
            contact_name=combined.get("contact_name"),
            hazard=_rule_hazard(latest),
            raw=raw,
            intake=outcome.intake,
            intake_prompt=outcome.prompt,
        )


def _rule_hazard(text: str) -> str | None:
    hazard = classify_hazard(text)
    return hazard.kind if hazard else None


class MissingKeyQualifier:
    """Used when a provider is selected but its API key is empty."""

    def __init__(self, provider: str = "anthropic") -> None:
        self.provider = provider

    def qualify(self, turn: TurnContext) -> QualificationDraft:
        if self.provider == "gemini":
            reply = (
                "Gemini is selected (LLM_PROVIDER=gemini) but GEMINI_API_KEY is empty. "
                "Create a free key at https://aistudio.google.com/apikey and add it to your .env file, "
                "or unset LLM_PROVIDER to use the offline qualifier."
            )
        else:
            reply = (
                "Claude is selected (LLM_PROVIDER=anthropic) but ANTHROPIC_API_KEY is empty. "
                "Add the key to your .env file, or unset LLM_PROVIDER to use the offline qualifier."
            )
        # intake is set so the service does not replace this explanation with a question.
        return QualificationDraft(reply=reply, raw={"unconfigured": True}, intake=normalise_intake(None))


def parse_qualification(response) -> QualificationDraft:
    tool_input = None
    texts: list[str] = []
    for block in getattr(response, "content", None) or []:
        block_type = block["type"] if isinstance(block, dict) else getattr(block, "type", None)
        if block_type == "tool_use":
            name = block["name"] if isinstance(block, dict) else getattr(block, "name", None)
            if name == QUALIFY_TOOL["name"]:
                tool_input = block["input"] if isinstance(block, dict) else getattr(block, "input", None)
        elif block_type == "text":
            text = block["text"] if isinstance(block, dict) else getattr(block, "text", "")
            if text:
                texts.append(text)
    if isinstance(tool_input, dict):
        return _draft_from_tool(tool_input, texts)
    reply = " ".join(piece.strip() for piece in texts if piece and piece.strip())
    if not reply:
        logger.warning("model returned no tool call and no text")
        reply = _FALLBACK_REPLY
    return QualificationDraft(reply=reply, raw={"unstructured": True})


def to_gemini_contents(history: list[dict]) -> list[dict]:
    """Gemini roles are user and model. The first turn must be a user turn."""
    contents = []
    for message in to_api_messages(history):
        role = "model" if message["role"] == "assistant" else "user"
        contents.append({"role": role, "parts": [{"text": message["content"]}]})
    return contents


def parse_gemini_qualification(response) -> QualificationDraft:
    """Read a function call, or the plain text if the model did not call one."""
    texts = _gemini_texts(response)
    for call in _gemini_function_calls(response):
        name = call.get("name")
        if name == QUALIFY_TOOL["name"]:
            return _draft_from_tool(call.get("args") or {}, texts)
    reply = " ".join(piece.strip() for piece in texts if piece and piece.strip())
    if not reply:
        logger.warning("gemini returned no function call and no text")
        reply = _FALLBACK_REPLY
    return QualificationDraft(reply=reply, raw={"unstructured": True})


def to_api_messages(history: list[dict]) -> list[dict]:
    """Anthropic requires a user message first and alternating roles."""
    cleaned: list[dict] = []
    for item in history:
        role = item.get("role")
        content = str(item.get("content") or "").strip()
        if role not in {"user", "assistant"} or not content:
            continue
        if cleaned and cleaned[-1]["role"] == role:
            cleaned[-1]["content"] += "\n" + content
        else:
            cleaned.append({"role": role, "content": content})
    if cleaned and cleaned[0]["role"] == "assistant":
        cleaned.insert(
            0,
            {
                "role": "user",
                "content": "(The customer opened the chat. Your greeting was already sent.)",
            },
        )
    if not cleaned:
        cleaned = [{"role": "user", "content": "(no message)"}]
    return cleaned


def _draft_from_tool(data: dict, texts: list[str]) -> QualificationDraft:
    reply = _clean_str(data.get("reply"))
    if not reply:
        reply = " ".join(piece.strip() for piece in texts if piece and piece.strip())
    if not reply:
        reply = _FALLBACK_REPLY
    urgency = _clean_str(data.get("urgency"))
    if urgency:
        urgency = urgency.lower()
    return QualificationDraft(
        reply=reply,
        request_human=_clean_bool(data.get("request_human")),
        job_type=_clean_str(data.get("job_type")),
        suburb=_clean_str(data.get("suburb")),
        urgency=urgency,
        description=_clean_str(data.get("description")),
        preferred_time=_clean_str(data.get("preferred_time")),
        contact_name=_clean_str(data.get("contact_name")),
        hazard=_clean_hazard(data.get("hazard")),
        raw=data,
    )


def _clean_hazard(value) -> str | None:
    """A known hazard kind, or None for "none", null, or a name we do not know."""
    hazard = hazard_from_kind(value if isinstance(value, str) else None)
    return hazard.kind if hazard else None


def _compose_reply(
    *,
    business: Business,
    latest: str,
    extracted: dict,
    known: dict,
    merged: dict,
    ad: dict | None,
    profile_name: str | None,
    history: list[dict],
    asked: bool,
    question: str | None,
) -> str:
    if asked:
        return (
            f"Absolutely — I'll bring a person from the {business.name} desk into this chat."
        )

    parts: list[str] = []
    if merged.get("urgency") == "emergency" and "000" not in latest:
        parts.append(_SAFETY)

    first_user_turn = _user_turns(history) <= 1
    if profile_name and not known.get("contact_name") and first_user_turn:
        parts.append(f"Thanks, {profile_name}.")

    new_job = extracted.get("job_type")
    if new_job and new_job != known.get("job_type"):
        label = business.job_label(new_job) or _label(new_job)
        parts.append(f"{label} — I'll put that on the job.")
    elif ad and not merged.get("job_type") and first_user_turn:
        headline = ad.get("headline") or "the ad you tapped"
        parts.append(f'I can help with "{headline}", or with something else.')

    if merged.get("suburb") and not business.in_area(merged.get("suburb")) and extracted.get("suburb"):
        parts.append(
            f"{merged['suburb']} is outside the area we cover. "
            "I'll note it so someone can refer you."
        )

    ready = _checklist_complete(merged, business)
    emergency_handoff = bool(
        merged.get("urgency") == "emergency"
        and business.in_area(merged.get("suburb"))
        and business.resolve_job_type(merged.get("job_type"))
    )
    if question and not emergency_handoff and not ready:
        parts.append(question)
        return " ".join(parts)

    if ready or emergency_handoff:
        label = business.job_label(merged.get("job_type")) or merged.get("job_type")
        who = merged.get("contact_name") or "you"
        parts.append(
            f"I have a {label} in {merged.get('suburb')} ({merged.get('urgency') or 'emergency'}) "
            f"under the name {who}. A person from the {business.name} desk will take it from here."
        )
        return " ".join(parts)

    return " ".join(parts) or "Thanks — I've noted that."


def _checklist_complete(facts: dict, business: Business) -> bool:
    return bool(
        business.resolve_job_type(facts.get("job_type"))
        and business.in_area(facts.get("suburb"))
        and facts.get("urgency")
        and facts.get("description")
        and facts.get("contact_name")
    )


def _label(value: str) -> str:
    text = str(value).strip()
    if not text:
        return text
    return text[0].upper() + text[1:]


def _latest_user_text(history: list[dict]) -> str:
    for item in reversed(history):
        if item.get("role") == "user":
            return str(item.get("content") or "")
    return ""


def _user_turns(history: list[dict]) -> int:
    return sum(1 for item in history if item.get("role") == "user")


def _clean_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.casefold() in {"null", "none", "undefined", "nil", "n/a"}:
        return None
    return text


def _clean_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return False


def _gemini_tool():
    from google.genai import types

    properties = {}
    for name, spec in QUALIFY_TOOL["input_schema"]["properties"].items():
        schema = {"description": spec.get("description") or ""}
        if spec.get("type") == "boolean":
            schema["type"] = "boolean"
        elif spec.get("type") == "string":
            schema["type"] = "string"
            if spec.get("enum"):
                schema["enum"] = list(spec["enum"])
        else:
            schema["type"] = "string"
            schema["nullable"] = True
        properties[name] = schema
    declaration = types.FunctionDeclaration(
        name=QUALIFY_TOOL["name"],
        description=QUALIFY_TOOL["description"],
        parameters_json_schema={
            "type": "object",
            "properties": properties,
            "required": list(QUALIFY_TOOL["input_schema"]["required"]),
        },
    )
    return types.Tool(function_declarations=[declaration])


def _gemini_function_calls(response) -> list[dict]:
    found = []
    calls = getattr(response, "function_calls", None) or []
    for call in calls:
        found.append(_call_dict(call))
    if found:
        return found
    for candidate in getattr(response, "candidates", None) or []:
        content = candidate.get("content") if isinstance(candidate, dict) else getattr(candidate, "content", None)
        parts = content.get("parts") if isinstance(content, dict) else getattr(content, "parts", None)
        for part in parts or []:
            call = part.get("function_call") if isinstance(part, dict) else getattr(part, "function_call", None)
            if call:
                found.append(_call_dict(call))
    return found


def _call_dict(call) -> dict:
    if isinstance(call, dict):
        name = call.get("name")
        args = call.get("args") or {}
    else:
        name = getattr(call, "name", None)
        args = getattr(call, "args", None) or {}
    if not isinstance(args, dict):
        try:
            args = dict(args)
        except (TypeError, ValueError):
            args = {}
    return {"name": name, "args": args}


def _gemini_texts(response) -> list[str]:
    texts: list[str] = []
    for candidate in getattr(response, "candidates", None) or []:
        content = candidate.get("content") if isinstance(candidate, dict) else getattr(candidate, "content", None)
        parts = content.get("parts") if isinstance(content, dict) else getattr(content, "parts", None)
        for part in parts or []:
            text = part.get("text") if isinstance(part, dict) else getattr(part, "text", None)
            if text:
                texts.append(str(text))
    if texts:
        return texts
    try:
        text = getattr(response, "text", None)
    except Exception:
        text = None
    if isinstance(text, str) and text.strip():
        return [text]
    return []


GEMINI_REQUEST_TIMEOUT_MS = 20_000


def gemini_client(api_key: str):
    """One Gemini client. Each request waits at most 20s and is not retried.

    The SDK's default HTTP timeout is unlimited, so a long thinking pass holds
    the webhook until the model finishes. A 404 is not retried either. After
    the timeout the caller falls back to the offline qualifier.
    """
    from google import genai
    from google.genai import types

    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=GEMINI_REQUEST_TIMEOUT_MS,
            retry_options=types.HttpRetryOptions(attempts=1),
        ),
    )


def log_gemini_failure(where: str, exc: Exception, model: str) -> None:
    if _is_not_found(exc):
        logger.error(
            "Gemini model %s returned 404 NOT_FOUND and is not available to this key. "
            "Set GEMINI_MODEL to a current model id (the default is gemini-3.8-flash). %s",
            model,
            exc,
        )
        return
    if _is_rate_limit(exc):
        logger.warning(
            "gemini %s hit a rate limit for model %s; using the offline qualifier: %s",
            where,
            model,
            exc,
        )
        return
    logger.exception("gemini %s failed for model %s; using the offline qualifier", where, model)


def _is_not_found(exc: Exception) -> bool:
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    status = str(getattr(exc, "status", "") or "")
    if code == 404 or status.upper() == "NOT_FOUND":
        return True
    text = str(exc).lower()
    return ("404" in text and "not_found" in text) or "no longer available" in text


def model_chain(primary: str, fallbacks: list[str] | None = None) -> list[str]:
    """Primary model, then each fallback, with blanks and duplicates removed."""
    chain: list[str] = []
    for name in [primary, *(fallbacks or [])]:
        text = (name or "").strip()
        if text and text not in chain:
            chain.append(text)
    return chain or ["gemini-3.8-flash"]


def should_try_next_model(exc: Exception) -> bool:
    """True when another model might answer: not found, rate limit, or the model is down."""
    code = _status_code(exc)
    if code in {404, 429, 503, 504}:
        return True
    text = str(exc).lower()
    if any(token in text for token in ("503", "504", "unavailable", "deadline exceeded", "timed out")):
        return True
    return _is_not_found(exc) or _is_rate_limit(exc)


def _status_code(exc: Exception) -> int | None:
    for attr in ("code", "status_code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    return None


def _is_rate_limit(exc: Exception) -> bool:
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if code == 429:
        return True
    text = str(exc).lower()
    return "429" in text or "resource_exhausted" in text or "rate limit" in text or "quota" in text
