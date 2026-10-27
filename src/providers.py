"""Provider transport. Dispatch and turn lifecycle belong to loop.metta."""
import copy
import json
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)
_llmProviderRegistry = {}
_llmprovider = None
_provider_ready = False
_provider_error = None
_attempt = 0
_pending_results = []


@dataclass
class ProviderResponse:
    kind: str
    status: str
    raw: object = None
    calls: list = field(default_factory=list)
    final_text: str = ""
    commentary: str = ""
    refusal: str = ""
    error: object = None
    attempt: int = 0


class LLMProvider:
    def start(self):
        pass

    def stop(self):
        pass

    def chat(self, prompt, max_tokens=6000, reasoning_mode="medium"):
        raise NotImplementedError()

    def respond(self, prompt, max_tokens=6000, reasoning_mode="medium", tools=None):
        # Only this explicit compatibility boundary authorizes command parsing.
        return {"kind": "legacy_commands", "status": "completed",
                "text": self.chat(prompt, max_tokens, reasoning_mode)}

    def tool_results(self, results):
        pass

    def end_turn(self, status):
        pass


# Schemas describe the existing string-argument native skills, not host functions.
_STATIC = {
    "remember": ["string"], "query": ["string"], "episodes": ["time_string"],
    "pin": ["string"], "shell": ["string"], "read-file": ["filename"],
    "write-file": ["filename", "string"],
    "write-file-b64": ["filename", "base64string"],
    "append-file": ["filename", "string"], "get-io-policy": [],
    "delete-file": ["filename"], "send": ["string"],
    "websearch": ["string"], "version": [], "metta": ["sexpression"],
}
_skills = {name: (name, args) for name, args in _STATIC.items()}


def register_skill(name, description, arguments):
    name, arguments = str(name), [str(a) for a in arguments]
    if not name or len(arguments) != len(set(arguments)) or any(not a for a in arguments):
        raise ValueError("Invalid skill schema")
    _skills[name] = (str(description), arguments)
    return True


def remove_skill(name):
    if str(name) not in _STATIC:
        _skills.pop(str(name), None)
    return True


def tool_schemas():
    return [{"type": "function", "name": name, "description": description,
             "parameters": {"type": "object", "properties": {
                 a: {"type": "string"} for a in arguments},
                 "required": arguments, "additionalProperties": False}, "strict": True}
            for name, (description, arguments) in _skills.items()]


def _arguments(name, arguments):
    if name not in _skills:
        raise ValueError("Unknown skill: " + str(name))
    names = _skills[name][1]
    if not isinstance(arguments, dict) or set(arguments) != set(names):
        raise ValueError("Wrong argument names for " + name)
    if any(not isinstance(arguments[a], str) for a in names):
        raise ValueError("Skill arguments must be strings")
    return [arguments[a] for a in names]


def _json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON argument")
        result[key] = value
    return result


def _quoted_argument(text):
    """Decode the pinned PeTTa string escapes, returning value and consumed size."""
    value, index = [], 1
    while index < len(text):
        char = text[index]
        index += 1
        if char == '"':
            return "".join(value), index
        if char == "\\":
            if index == len(text):
                break
            char = text[index]
            index += 1
            char = {"n": "\n", "r": "\r", "t": "\t"}.get(char, char)
        value.append(char)
    raise ValueError("Unterminated quoted argument")


def _legacy_argument(text):
    # Match helper.quote_arg -> pinned sread without generating program text.
    # A multiline quoted block is literal data, including its outer quotes.
    if text.startswith('"') and text.endswith('"') and "\n" not in text:
        value, consumed = _quoted_argument(text)
        if consumed != len(text):
            raise ValueError("Unexpected text after quoted argument")
        return value
    return text


def _legacy_calls(text):
    import helper
    calls = []
    text = text.replace("_quote_", '"').replace("_newline_", "\n")
    for block in helper.split_command_blocks(text):
        block = block.strip()
        if block.startswith("(-"):
            block = "(pin " + block[2:]
        elif block.startswith("-"):
            block = "pin " + block[1:]
        if block.startswith("(") and block.endswith(")"):
            block = block[1:-1].strip()
        elif block.startswith("("):
            block = block[1:].strip()
        bits = block.split(None, 1)
        if not bits:
            continue
        name = bits[0]
        if name not in _skills:
            raise ValueError("Unknown legacy skill")
        names = _skills[name][1]
        rest = bits[1].strip() if len(bits) == 2 else ""
        if len(names) <= 1:
            values = [_legacy_argument(rest)] if rest else []
        elif name in helper.TWO_ARG_COMMANDS:
            if rest.startswith('"'):
                first, end = _quoted_argument(rest)
                remaining = rest[end:].strip()
            else:
                parts = rest.split(None, 1)
                first = parts[0] if parts else ""
                remaining = parts[1] if len(parts) > 1 else ""
            values = [first, _legacy_argument(remaining)] if remaining else [first]
        else:
            # The registered schema extends the old one-argument dynamic form
            # to multiple explicitly separated string arguments.
            values = []
            while rest:
                if rest.startswith('"'):
                    value, end = _quoted_argument(rest)
                    if end < len(rest) and not rest[end].isspace():
                        raise ValueError("Missing argument separator")
                    rest = rest[end:].lstrip()
                else:
                    parts = rest.split(None, 1)
                    value = parts[0]
                    rest = parts[1] if len(parts) == 2 else ""
                values.append(value)
        if len(values) != len(names):
            raise ValueError("Wrong legacy argument count")
        calls.append(["legacy-" + str(len(calls)), name, values])
    return calls


def validate_response(raw):
    """Validate the entire batch before exposing any executable call to MeTTa.

    Keep the complete transport object even when validation fails. Extracted
    fields are views, not replacements for ordered output/reasoning/usage.
    """
    response = ProviderResponse("structured", "malformed", raw=copy.deepcopy(raw))
    try:
        if not isinstance(raw, dict):
            raise ValueError("Response must be a complete transport object")
        if raw.get("kind") == "legacy_commands":
            response.kind = "legacy_commands"
            if raw.get("status") != "completed" or not isinstance(raw.get("text"), str):
                raise ValueError("Invalid legacy envelope")
            response.calls = _legacy_calls(raw["text"])
            response.status = "completed"
            return response
        status = raw.get("status")
        if status not in ("completed", "incomplete", "cancelled", "failed", "queued", "in_progress"):
            raise ValueError("Missing or unsupported response status")
        if not isinstance(raw.get("id"), str) or not isinstance(raw.get("output"), list):
            raise ValueError("Missing response ID or ordered output")
        if status != "completed":
            response.status = status
            response.error = raw.get("error")
            return response  # Partial output is retained, never executable.
        ids, calls, final, commentary, refusals = set(), [], [], [], []
        for item in raw["output"]:
            if not isinstance(item, dict):
                raise ValueError("Invalid output item")
            kind = item.get("type")
            if kind == "reasoning":
                continue  # Retained in raw and in the provider's replay, unchanged.
            if kind == "function_call":
                call_id, name = item.get("call_id"), item.get("name")
                if not isinstance(call_id, str) or not call_id or call_id in ids:
                    raise ValueError("Missing or duplicate call ID")
                ids.add(call_id)
                if item.get("status", "completed") != "completed":
                    raise ValueError("Partial tool call")
                arguments = item.get("arguments")
                if not isinstance(arguments, str):
                    raise ValueError("Arguments must be serialized JSON")
                args = json.loads(arguments, object_pairs_hook=_json_object)
                calls.append([call_id, name, _arguments(name, args)])
            elif kind == "message":
                if item.get("role") != "assistant" or item.get("status", "completed") != "completed":
                    raise ValueError("Invalid assistant message")
                phase = item.get("phase")
                if phase not in (None, "final_answer", "commentary"):
                    raise ValueError("Unsupported assistant phase")
                if not isinstance(item.get("content"), list):
                    raise ValueError("Invalid message content")
                for part in item["content"]:
                    if not isinstance(part, dict):
                        raise ValueError("Invalid content part")
                    if part.get("type") == "output_text" and isinstance(part.get("text"), str):
                        (commentary if phase == "commentary" else final).append(part["text"])
                    elif part.get("type") == "refusal" and isinstance(part.get("refusal"), str):
                        refusals.append(part["refusal"])
                    else:
                        raise ValueError("Unsupported content part")
            else:
                raise ValueError("Unsupported output item")
        response.status = status
        response.calls = calls
        response.final_text = "\n".join(final)
        response.commentary = "\n".join(commentary)
        response.refusal = "\n".join(refusals)
        response.error = raw.get("error")
        if status == "completed" and (response.error is not None or raw.get("incomplete_details") is not None):
            raise ValueError("Completed response carries failure details")
    except Exception as exc:
        response.status = "malformed"
        response.calls = []
        response.error = failure_detail("validation_error", exc)
    return response


def registerLLMProvider(id, provider):
    _llmProviderRegistry[id] = provider


def llmProviderStart(provider):
    global _llmprovider, _provider_ready, _provider_error
    _provider_ready = False
    _llmprovider = _llmProviderRegistry.get(provider)
    try:
        if _llmprovider is None:
            raise RuntimeError("Unregistered provider: " + str(provider))
        _llmprovider.start()
        _provider_ready, _provider_error = True, None
        return "ready"
    except Exception as exc:
        _provider_error = failure_detail("start_error", exc)
        logger.error("Provider start failed: %s", _provider_error)
        return "start_error"


def llmProviderChat(prompt, max_tokens, reasoning_mode):
    return _llmprovider.chat(prompt, max_tokens, reasoning_mode)


def llmProviderRespond(prompt, max_tokens=6000, reasoning_mode="medium", tools=None):
    global _attempt
    _attempt += 1
    if not _provider_ready:
        return ProviderResponse("transport", "start_error", error=_provider_error, attempt=_attempt)
    if flush_tool_results() != "recorded":
        return ProviderResponse("transport", "replay_error", attempt=_attempt,
                                error={"category": "replay_error", "usage": None})
    try:
        raw = _llmprovider.respond(prompt, int(max_tokens), reasoning_mode,
                                   tool_schemas() if tools is None else tools)
        response = validate_response(raw)
        response.attempt = _attempt
        return response
    except Exception as exc:
        return ProviderResponse("transport", "provider_error", attempt=_attempt,
                                error=failure_detail("provider_error", exc))


def response_field(response, name):
    return getattr(response, name)


def response_record(response):
    return json.dumps(vars(response), ensure_ascii=False, default=str)


def failure_detail(category, exc):
    # Exception text/tracebacks can embed credentials or full request bodies.
    # Never place them in a response, native history, or a subsequent prompt.
    return {"category": category, "type": type(exc).__name__, "usage": None}


def flush_tool_results():
    """Mechanical delivery of already chosen native outcomes; never reexecute."""
    global _pending_results
    if not _pending_results:
        return "recorded"
    try:
        _llmprovider.tool_results(copy.deepcopy(_pending_results))
        _pending_results = []
        return "recorded"
    except Exception as exc:
        logger.error("Provider replay failed: %s", failure_detail("replay_error", exc))
        return "replay_error"


def llmProviderToolResults(results):
    global _pending_results
    # Retain the complete result set before the delegate can fail, including
    # after a partial append. Providers must accept identical replay by call ID.
    _pending_results.extend(copy.deepcopy(results))
    return flush_tool_results()


def provider_ready():
    return _provider_ready


def llmProviderEndTurn(status):
    # A missing delegate receipt cannot erase known native effects/results.
    if flush_tool_results() != "recorded":
        return "replay_error"
    try:
        if _provider_ready:
            _llmprovider.end_turn(status)
        return "recorded"
    except Exception as exc:
        logger.error("End-turn replay failed: %s", failure_detail("replay_error", exc))
        return "replay_error"
