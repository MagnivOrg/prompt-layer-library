from __future__ import annotations

import base64
import json
import logging
import os
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

_BEDROCK_RUNTIME_SERVICE = "bedrock-runtime"
_CONVERSE_OPERATIONS = {"Converse", "ConverseStream"}
_GEN_AI_INPUT_MESSAGES = "gen_ai.input.messages"
_GEN_AI_OUTPUT_MESSAGES = "gen_ai.output.messages"
_GEN_AI_SYSTEM_INSTRUCTIONS = "gen_ai.system_instructions"
_MESSAGE_CONTENT_ENV = "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"
_SPAN_CAPTURE_MODES = {
    "1",
    "on",
    "span",
    "span_only",
    "span_and_event",
    "true",
    "yes",
}


def bedrock_request_hook(
    span: Any,
    service_name: str,
    operation_name: str,
    api_params: Dict[str, Any],
) -> None:
    """Attach Bedrock Converse input content using GenAI span conventions."""

    if not _should_capture(span, service_name, operation_name):
        return

    try:
        messages = _normalize_messages(api_params.get("messages", ()))
        if messages:
            _set_json_attribute(span, _GEN_AI_INPUT_MESSAGES, messages)

        system_instructions = _normalize_parts(api_params.get("system", ()))
        if system_instructions:
            _set_json_attribute(span, _GEN_AI_SYSTEM_INSTRUCTIONS, system_instructions)
    except Exception:
        logger.debug("PromptLayer could not capture an AWS Bedrock request body", exc_info=True)


def bedrock_response_hook(
    span: Any,
    service_name: str,
    operation_name: str,
    result: Optional[Dict[str, Any]],
) -> None:
    """Attach a non-streaming Bedrock Converse response to its GenAI span."""

    if not _should_capture(span, service_name, operation_name) or not isinstance(result, dict):
        return

    try:
        output = result.get("output")
        message = output.get("message") if isinstance(output, dict) else None
        normalized = _normalize_message(message, default_role="assistant")
        if normalized is None:
            return

        finish_reason = result.get("stopReason")
        if finish_reason is not None:
            normalized["finish_reason"] = str(finish_reason)
        _set_json_attribute(span, _GEN_AI_OUTPUT_MESSAGES, [normalized])
    except Exception:
        logger.debug("PromptLayer could not capture an AWS Bedrock response body", exc_info=True)


def _should_capture(span: Any, service_name: str, operation_name: str) -> bool:
    if service_name != _BEDROCK_RUNTIME_SERVICE or operation_name not in _CONVERSE_OPERATIONS:
        return False
    if not _capture_message_content_enabled():
        return False
    is_recording = getattr(span, "is_recording", None)
    return not callable(is_recording) or is_recording()


def _capture_message_content_enabled() -> bool:
    configured = os.environ.get(_MESSAGE_CONTENT_ENV, "")
    return configured.strip().lower() in _SPAN_CAPTURE_MODES


def _normalize_messages(messages: Any) -> List[Dict[str, Any]]:
    if not isinstance(messages, Iterable) or isinstance(messages, (str, bytes, dict)):
        return []

    normalized = []
    for message in messages:
        normalized_message = _normalize_message(message)
        if normalized_message is not None:
            normalized.append(normalized_message)
    return normalized


def _normalize_message(message: Any, *, default_role: Optional[str] = None) -> Optional[Dict[str, Any]]:
    if not isinstance(message, dict):
        return None

    role = message.get("role") or default_role
    parts = _normalize_parts(message.get("content", ()))
    if not role or not parts:
        return None
    return {"role": str(role), "parts": parts}


def _normalize_parts(content: Any) -> List[Dict[str, Any]]:
    if isinstance(content, (str, bytes)) or isinstance(content, dict):
        content = [content]
    if not isinstance(content, Iterable):
        return []

    parts = []
    for block in content:
        part = _normalize_part(block)
        if part is not None:
            parts.append(part)
    return parts


def _normalize_part(block: Any) -> Optional[Dict[str, Any]]:
    if isinstance(block, str):
        return {"type": "text", "content": block}
    if isinstance(block, bytes):
        return {
            "type": "generic",
            "value": base64.b64encode(block).decode("ascii"),
        }
    if not isinstance(block, dict):
        return None

    if "text" in block:
        return {"type": "text", "content": str(block["text"])}

    tool_use = block.get("toolUse")
    if isinstance(tool_use, dict):
        return {
            "type": "tool_call",
            "id": tool_use.get("toolUseId"),
            "name": tool_use.get("name", ""),
            "arguments": _json_safe(tool_use.get("input")),
        }

    tool_result = block.get("toolResult")
    if isinstance(tool_result, dict):
        return {
            "type": "tool_call_response",
            "id": tool_result.get("toolUseId"),
            "response": _json_safe(tool_result.get("content")),
        }

    reasoning = block.get("reasoningContent")
    if isinstance(reasoning, dict):
        reasoning_text = reasoning.get("reasoningText")
        if isinstance(reasoning_text, dict) and reasoning_text.get("text") is not None:
            return {"type": "reasoning", "content": str(reasoning_text["text"])}

    return {"type": "generic", "value": _json_safe(block)}


def _json_safe(value: Any) -> Any:
    if isinstance(value, bytes):
        return base64.b64encode(value).decode("ascii")
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _set_json_attribute(span: Any, name: str, value: Any) -> None:
    span.set_attribute(
        name,
        json.dumps(value, ensure_ascii=False, separators=(",", ":")),
    )
