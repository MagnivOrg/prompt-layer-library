"""Derive eval ``output`` from a Trace cell (last assistant message)."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, NamedTuple, Optional

from promptlayer.evaluations.utils import parse_cell_value

_TOOL_PREFIXES = ("Tool: ", "Tool:")


class ToolSpan(NamedTuple):
    tool: str
    output: Any
    span: Dict[str, Any]


def parse_json_dict(value: Any) -> Optional[Dict[str, Any]]:
    """Parse a JSON object from a dict or JSON string; otherwise return ``None``."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def collect_tool_spans(trace: Any) -> List[ToolSpan]:
    """Collect ``Tool:`` spans in chronological order."""
    if not isinstance(trace, dict):
        return []
    tools: List[ToolSpan] = []
    for span in iter_spans_chrono(trace):
        tool = _tool_name_from_span(span)
        if tool is None:
            continue
        tools.append(ToolSpan(tool=tool, output=span.get("output"), span=span))
    return tools


def extract_tool_names(trace: Any) -> List[str]:
    return [entry.tool for entry in collect_tool_spans(trace)]


def _tool_name_from_span(span: Dict[str, Any]) -> Optional[str]:
    name = span.get("name")
    if not isinstance(name, str):
        return None
    for prefix in _TOOL_PREFIXES:
        if name.startswith(prefix):
            return name[len(prefix) :].strip()
    return None


def extract_last_assistant_message(trace: Any) -> Any:
    """Return the last assistant message found in a Trace span tree."""
    if not isinstance(trace, dict):
        return None

    candidates: List[Any] = []
    for span in iter_spans_chrono(trace):
        message = _assistant_from_span(span)
        if message is not None:
            candidates.append(message)
    return candidates[-1] if candidates else None


def resolve_output_from_trace_row(
    row: Optional[Dict[str, Any]],
    columns_by_title_map: Dict[str, Any],
    *,
    fallback: Any = None,
) -> Any:
    """Use a non-null runner output; otherwise derive the assistant output from Trace."""
    if fallback is not None:
        return fallback
    trace = _trace_cell_value(row, columns_by_title_map)
    return extract_last_assistant_message(trace)


def _trace_cell_value(
    row: Optional[Dict[str, Any]],
    columns_by_title_map: Dict[str, Any],
) -> Any:
    if not row:
        return None
    column = columns_by_title_map.get("Trace")
    if not column or column.get("id") is None:
        return None
    cells = row.get("cells") or {}
    cell = cells.get(str(column["id"]))
    return parse_cell_value(cell if isinstance(cell, dict) else None)


def iter_spans_chrono(root: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    spans = list(_walk_spans(root))
    spans.sort(key=_span_sort_key)
    return spans


def _walk_spans(node: Any) -> Iterable[Dict[str, Any]]:
    if not isinstance(node, dict):
        return
    yield node
    children = node.get("children") or []
    if isinstance(children, list):
        for child in children:
            yield from _walk_spans(child)


def _span_sort_key(span: Dict[str, Any]) -> tuple:
    start = span.get("start") or ""
    span_id = span.get("span_id") or ""
    return (str(start), str(span_id))


def _assistant_from_span(span: Dict[str, Any]) -> Any:
    request_log = span.get("request_log")
    if not isinstance(request_log, dict):
        return None

    response = request_log.get("request_response")
    message = _assistant_from_request_response(response)
    if message is not None:
        return message

    kwargs = request_log.get("function_kwargs")
    if isinstance(kwargs, dict):
        messages = kwargs.get("messages")
        if isinstance(messages, list):
            for msg in reversed(messages):
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    return _normalize_assistant_message(msg)
    return None


def _assistant_from_request_response(response: Any) -> Any:
    if not isinstance(response, dict):
        return None

    choices = response.get("choices")
    if isinstance(choices, list) and choices:
        choice = choices[0]
        if isinstance(choice, dict):
            message = choice.get("message")
            if isinstance(message, dict):
                role = message.get("role") or "assistant"
                if role == "assistant":
                    return _normalize_assistant_message(message)

    # Anthropic Messages API. Backend-normalized responses intentionally omit
    # ``type`` and may omit ``stop_reason``, so role + content is the stable shape.
    role = response.get("role")
    if role in (None, "assistant") and isinstance(response.get("content"), (str, list)):
        return _normalize_anthropic_response(response)

    # Legacy Anthropic Completions API.
    completion = response.get("completion")
    if isinstance(completion, str):
        return _maybe_parse_json(completion)

    # Google Gemini / Vertex generate-content.
    candidates = response.get("candidates")
    if isinstance(candidates, list) and candidates:
        candidate = candidates[0]
        if isinstance(candidate, dict):
            message = _normalize_google_content(candidate.get("content"))
            if message is not None:
                return message

    # OpenAI Responses API
    output_list = response.get("output")
    if isinstance(output_list, list):
        for item in reversed(output_list):
            if not isinstance(item, dict):
                continue
            if item.get("type") == "message" and item.get("role", "assistant") == "assistant":
                return _normalize_responses_message(item)
            if item.get("type") == "function_call":
                return {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": item.get("call_id") or item.get("id"),
                            "type": "function",
                            "function": {
                                "name": item.get("name"),
                                "arguments": item.get("arguments"),
                            },
                        }
                    ],
                }

    # Amazon Bedrock Converse.
    if isinstance(response.get("output"), dict):
        output = response["output"]
        message = output.get("message")
        normalized = _normalize_bedrock_message(message)
        if normalized is not None:
            return normalized

    return None


def _normalize_assistant_message(message: Dict[str, Any]) -> Any:
    content = message.get("content")
    tool_calls = message.get("tool_calls")
    function_call = message.get("function_call")
    text = _content_as_text(content)

    if tool_calls:
        return {"content": text if text is not None else content, "tool_calls": tool_calls}
    if function_call:
        return {"content": text if text is not None else content, "function_call": function_call}
    return _maybe_parse_json(text if text is not None else content)


def _normalize_anthropic_response(response: Dict[str, Any]) -> Any:
    content = response.get("content")
    text_parts: List[str] = []
    tool_calls: List[Dict[str, Any]] = []

    if isinstance(content, str):
        return _maybe_parse_json(content)

    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "text" and isinstance(block.get("text"), str):
                text_parts.append(block["text"])
            elif block_type == "tool_use":
                tool_calls.append(
                    {
                        "id": block.get("id"),
                        "type": "function",
                        "function": {
                            "name": block.get("name"),
                            "arguments": json.dumps(block.get("input") or {}),
                        },
                    }
                )

    text = "\n".join(text_parts) if text_parts else None
    if tool_calls:
        return {"content": text, "tool_calls": tool_calls}
    if text is not None:
        return _maybe_parse_json(text)
    return None


def _normalize_google_content(content: Any) -> Any:
    if not isinstance(content, dict):
        return None
    if content.get("role") != "model":
        return None
    parts = content.get("parts")
    if not isinstance(parts, list):
        return None

    text_parts: List[str] = []
    tool_calls: List[Dict[str, Any]] = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        if isinstance(part.get("text"), str) and not part.get("thought"):
            text_parts.append(part["text"])
        function_call = part.get("function_call")
        if isinstance(function_call, dict):
            arguments = function_call.get("args") or {}
            tool_calls.append(
                {
                    "id": function_call.get("id"),
                    "type": "function",
                    "function": {
                        "name": function_call.get("name"),
                        "arguments": arguments if isinstance(arguments, str) else json.dumps(arguments),
                    },
                }
            )

    text = "\n".join(text_parts) if text_parts else None
    if tool_calls:
        return {"content": text, "tool_calls": tool_calls}
    return _maybe_parse_json(text) if text is not None else None


def _normalize_bedrock_message(message: Any) -> Any:
    if not isinstance(message, dict) or message.get("role", "assistant") != "assistant":
        return None
    content = message.get("content")
    if not isinstance(content, list):
        return None

    text_parts: List[str] = []
    tool_calls: List[Dict[str, Any]] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if isinstance(block.get("text"), str):
            text_parts.append(block["text"])
        tool_use = block.get("toolUse")
        if isinstance(tool_use, dict):
            tool_calls.append(
                {
                    "id": tool_use.get("toolUseId"),
                    "type": "function",
                    "function": {
                        "name": tool_use.get("name"),
                        "arguments": json.dumps(tool_use.get("input") or {}),
                    },
                }
            )

    text = "\n".join(text_parts) if text_parts else None
    if tool_calls:
        return {"content": text, "tool_calls": tool_calls}
    return _maybe_parse_json(text) if text is not None else None


def _normalize_responses_message(item: Dict[str, Any]) -> Any:
    content = item.get("content")
    text_parts: List[str] = []
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") in ("output_text", "text"):
                if isinstance(block.get("text"), str):
                    text_parts.append(block["text"])
    elif isinstance(content, str):
        text_parts.append(content)
    text = "\n".join(text_parts) if text_parts else None
    return _maybe_parse_json(text) if text is not None else None


def _content_as_text(content: Any) -> Optional[str]:
    if content is None:
        return None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts) if parts else None
    return None


def _maybe_parse_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped or stripped[0] not in "{[":
        return value
    try:
        return json.loads(stripped)
    except (TypeError, ValueError, json.JSONDecodeError):
        return value
