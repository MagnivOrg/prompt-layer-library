import copy
import logging
import threading
import time
from string import Formatter
from typing import Any, Dict, List, Optional, Tuple

import jinja2
from jinja2.sandbox import SandboxedEnvironment

logger = logging.getLogger(__name__)

CacheKey = Tuple[str, Optional[int], Optional[str], Optional[str], Optional[str]]

_NON_RENDERABLE_TTL = 60
_MAX_ENTRIES = 1000


class _CacheEntry:
    __slots__ = ("response", "timestamp")

    def __init__(self, response: dict, timestamp: float):
        self.response = response
        self.timestamp = timestamp


class PromptTemplateCache:
    """Thread-safe in-memory TTL cache for prompt templates.

    Stores unrendered API responses keyed by prompt identifier fields.
    Supports stale-while-error: if TTL has expired but the API is unreachable,
    the stale entry is still usable as a stability fallback.
    """

    def __init__(self, ttl_seconds: int, max_size: int = _MAX_ENTRIES):
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self._entries: Dict[CacheKey, _CacheEntry] = {}
        self._non_renderable: Dict[CacheKey, float] = {}
        self._lock = threading.Lock()

    @staticmethod
    def make_key(prompt_name: str, params: Optional[dict] = None) -> CacheKey:
        if not params:
            return (prompt_name, None, None, None, None)
        return (
            prompt_name,
            params.get("version"),
            params.get("label"),
            params.get("provider"),
            params.get("model"),
        )

    def get(self, key: CacheKey) -> Tuple[Optional[dict], bool]:
        """Return (cached_response | None, is_fresh).

        The returned response is a deep copy safe to mutate.
        """
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None, False
            is_fresh = (time.monotonic() - entry.timestamp) < self.ttl_seconds
            return copy.deepcopy(entry.response), is_fresh

    def put(self, key: CacheKey, response: dict):
        with self._lock:
            if len(self._entries) >= self.max_size and key not in self._entries:
                self._evict_oldest_entry()
            self._entries[key] = _CacheEntry(copy.deepcopy(response), time.monotonic())

    def is_non_renderable(self, key: CacheKey) -> bool:
        with self._lock:
            ts = self._non_renderable.get(key)
            if ts is None:
                return False
            if (time.monotonic() - ts) >= _NON_RENDERABLE_TTL:
                del self._non_renderable[key]
                return False
            return True

    def mark_non_renderable(self, key: CacheKey):
        with self._lock:
            if len(self._non_renderable) >= _MAX_ENTRIES and key not in self._non_renderable:
                self._evict_oldest_non_renderable()
            self._non_renderable[key] = time.monotonic()

    def clear(self):
        """Remove all cached entries and non-renderable markers."""
        with self._lock:
            self._entries.clear()
            self._non_renderable.clear()

    def invalidate(self, prompt_name: str):
        """Remove all entries whose cache key starts with *prompt_name*."""
        with self._lock:
            keys_to_remove = [k for k in self._entries if k[0] == prompt_name]
            for k in keys_to_remove:
                del self._entries[k]
            nr_to_remove = [k for k in self._non_renderable if k[0] == prompt_name]
            for k in nr_to_remove:
                del self._non_renderable[k]

    def _evict_oldest_entry(self):
        if not self._entries:
            return
        oldest_key = min(self._entries, key=lambda k: self._entries[k].timestamp)
        del self._entries[oldest_key]

    def _evict_oldest_non_renderable(self):
        if not self._non_renderable:
            return
        oldest_key = min(self._non_renderable, key=self._non_renderable.get)
        del self._non_renderable[oldest_key]


# ── public helpers used by TemplateManager ──────────────────────────


def should_skip_cache(params: Optional[dict]) -> bool:
    """Return True when the request params make caching inappropriate."""
    if not params:
        return False
    if params.get("metadata_filters") or params.get("model_parameter_overrides"):
        return True
    return False


def has_list_input_variables(params: Optional[dict]) -> bool:
    """Detect list-typed input variables (placeholder messages / tool variables)."""
    ivars = params.get("input_variables") if params else None
    if not ivars:
        return False
    return any(isinstance(v, list) for v in ivars.values())


def is_locally_renderable(response: dict) -> bool:
    """Check whether the template can be rendered client-side.

    Rejects templates that use placeholder messages or tool-variable
    expansion — these require server-side logic we do not replicate.
    """
    pt = response.get("prompt_template")
    if not pt:
        return False

    if pt.get("type") == "chat":
        for msg in pt.get("messages", []):
            if msg.get("role") == "placeholder":
                return False

    for tool in pt.get("tools") or []:
        if tool.get("type") == "variable":
            return False

    return True


def make_cache_params(params: Optional[dict]) -> dict:
    """Build API request params for fetching a cacheable (unrendered) template."""
    result: dict = {}
    if params:
        for k, v in params.items():
            if k in ("input_variables", "metadata_filters", "model_parameter_overrides"):
                continue
            result[k] = v
    result["skip_input_variable_rendering"] = True
    return result


# ── response rendering ──────────────────────────────────────────────


def render_response(response: dict, input_variables: Optional[Dict[str, Any]] = None) -> dict:
    """Render input variables in a response dict.

    Mutates *response* in-place and returns it.  Callers that need to
    preserve the original must pass a copy (e.g. the one returned by
    ``PromptTemplateCache.get``).
    """
    has_llm_kwargs = response.get("llm_kwargs") is not None
    if has_llm_kwargs:
        variables = input_variables if input_variables else {}
    else:
        if not input_variables:
            return response
        variables = input_variables

    pt = response.get("prompt_template")
    if pt:
        _render_prompt_template(pt, variables)
        message_formats = _get_message_formats(pt)
        if response.get("llm_kwargs"):
            _render_llm_kwargs(response["llm_kwargs"], message_formats, variables)

    return response


# ── internal rendering helpers ──────────────────────────────────────


def _get_message_formats(prompt_template: dict) -> List[str]:
    """Extract template_format from each message in the prompt template."""
    if prompt_template.get("type") == "chat":
        return [msg.get("template_format", "f-string") for msg in prompt_template.get("messages", [])]
    elif prompt_template.get("type") == "completion":
        return [prompt_template.get("template_format", "f-string")]
    return ["f-string"]


def _render_text(text: str, template_format: str, variables: dict) -> str:
    try:
        if template_format == "jinja2":
            return _jinja2_render(text, variables)
        return _fstring_render(text, variables)
    except Exception:
        logger.debug("Failed to render template text, returning original", exc_info=True)
        return text


def _fstring_render(template: str, variables: dict) -> str:
    """Match server-side ``fstring_formatter`` behaviour."""
    resolved: Dict[str, Any] = {}
    for _, field_name, _, _ in Formatter().parse(template):
        if field_name:
            if field_name in variables:
                val = variables[field_name]
                resolved[field_name] = val if val is not None else ""
            else:
                resolved[field_name] = ""
    return template.format(**resolved)


_JINJA2_ENV = SandboxedEnvironment(undefined=jinja2.ChainableUndefined)


def _jinja2_render(template: str, variables: dict) -> str:
    """Match server-side ``jinja2_formatter`` behaviour (no-warnings path)."""
    return _JINJA2_ENV.from_string(template).render(**variables)


# ── prompt_template rendering ───────────────────────────────────────


def _render_prompt_template(prompt_template: dict, variables: dict):
    """Render input variables in ``prompt_template`` text content (in-place)."""
    if prompt_template.get("type") == "chat":
        for message in prompt_template.get("messages", []):
            fmt = message.get("template_format", "f-string")
            for block in message.get("content", []):
                if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str):
                    block["text"] = _render_text(block["text"], fmt, variables)
    elif prompt_template.get("type") == "completion":
        fmt = prompt_template.get("template_format", "f-string")
        for block in prompt_template.get("content", []):
            if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str):
                block["text"] = _render_text(block["text"], fmt, variables)


# ── llm_kwargs rendering (provider-agnostic) ────────────────────────


_TEXT_CONTENT_TYPES = {"text", "input_text"}


def _render_content_field(value, template_format: str, variables: dict):
    """Render template variables in a content value (string *or* list of blocks)."""
    if isinstance(value, str):
        return _render_text(value, template_format, variables)
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict) and "text" in item and isinstance(item["text"], str):
                if item.get("type", "text") in _TEXT_CONTENT_TYPES:
                    item["text"] = _render_text(item["text"], template_format, variables)
        return value
    return value


def _render_parts(parts, fmt: str, variables: dict):
    """Render text in a list of Google-style ``parts`` dicts (in-place)."""
    for part in parts:
        if isinstance(part, dict) and "text" in part and isinstance(part["text"], str):
            part["text"] = _render_text(part["text"], fmt, variables)


def _find_system_instruction(llm_kwargs: dict):
    """Return the Google ``system_instruction`` value regardless of nesting.

    Google mappers may place it at the top level or inside
    ``generation_config``.
    """
    si = llm_kwargs.get("system_instruction")
    if si is not None:
        return si
    gc = llm_kwargs.get("generation_config")
    if isinstance(gc, dict):
        return gc.get("system_instruction")
    return None


def _render_llm_kwargs(llm_kwargs: dict, message_formats: List[str], variables: dict):
    """Render input variables in ``llm_kwargs`` text content (in-place).

    Handles the message/content structures of all major providers
    (OpenAI, Anthropic, Google, Bedrock, Mistral).

    A single ``fmt`` derived from the first prompt-template message is
    used for every field.  Per-message format alignment is intentionally
    avoided because providers like Anthropic and Bedrock extract system
    messages into a separate top-level key, which shifts positional
    indices and makes per-index lookup incorrect.  In practice all
    messages in a template share the same format.
    """
    fmt = message_formats[0] if message_formats else "f-string"

    # Messages — OpenAI Chat Completions / Anthropic / Mistral / Bedrock
    for msg in llm_kwargs.get("messages", []):
        if "content" in msg:
            msg["content"] = _render_content_field(msg["content"], fmt, variables)

    # Input — OpenAI Responses API
    for msg in llm_kwargs.get("input", []):
        if not isinstance(msg, dict):
            continue
        if "content" in msg:
            msg["content"] = _render_content_field(msg["content"], fmt, variables)

    # Top-level system — Anthropic / Bedrock
    if "system" in llm_kwargs:
        llm_kwargs["system"] = _render_content_field(llm_kwargs["system"], fmt, variables)

    # Contents — Google (completion)
    for item in llm_kwargs.get("contents", []):
        if isinstance(item, dict):
            _render_parts(item.get("parts", []), fmt, variables)

    # History — Google (chat)
    for item in llm_kwargs.get("history", []):
        if isinstance(item, dict):
            _render_parts(item.get("parts", []), fmt, variables)

    # System instruction — Google (top-level or inside generation_config)
    si = _find_system_instruction(llm_kwargs)
    if isinstance(si, dict):
        _render_parts(si.get("parts", []), fmt, variables)
    elif isinstance(si, list):
        _render_parts(si, fmt, variables)

    # Prompt field — completion-type models
    if isinstance(llm_kwargs.get("prompt"), str):
        llm_kwargs["prompt"] = _render_text(llm_kwargs["prompt"], fmt, variables)
