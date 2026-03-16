import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from opentelemetry.trace import SpanKind

_UNSET = object()


def telemetry_source_version() -> str:
    try:
        from importlib.metadata import version

        return version("promptlayer")
    except Exception:
        return "unknown"


def base_trace_attributes(trace, *, include_raw_payloads: bool) -> dict[str, Any]:
    attrs: dict[str, Any] = {
        "promptlayer.telemetry.source": "openai-agents-python",
        "promptlayer.telemetry.source_version": telemetry_source_version(),
        "openai_agents.trace_id_original": trace.trace_id,
        "openai_agents.workflow_name": trace.name,
    }

    if getattr(trace, "group_id", None):
        attrs["openai_agents.group_id"] = trace.group_id

    metadata = getattr(trace, "metadata", None)
    if isinstance(metadata, Mapping):
        for key, value in metadata.items():
            if _is_scalar(value):
                attrs[f"openai_agents.metadata.{_sanitize_key(str(key))}"] = value
        if include_raw_payloads and metadata:
            attrs["openai_agents.metadata_json"] = _json_dumps(metadata)

    return attrs


def span_name_for(span) -> str:
    span_type = span.span_data.type
    if span_type == "function":
        return f"Function: {span.span_data.name}"
    if span_type == "agent":
        return f"Agent: {span.span_data.name}"
    if span_type == "guardrail":
        return f"Guardrail: {span.span_data.name}"
    if span_type == "custom":
        return str(span.span_data.name)
    return span_type.replace("_", " ").title()


def span_kind_for(span) -> SpanKind:
    if span.span_data.type in {"generation", "response"}:
        return SpanKind.CLIENT
    return SpanKind.INTERNAL


def base_span_attributes(span) -> dict[str, Any]:
    attrs: dict[str, Any] = {
        "promptlayer.telemetry.source": "openai-agents-python",
        "promptlayer.telemetry.source_version": telemetry_source_version(),
        "openai_agents.span_id_original": span.span_id,
        "openai_agents.span_type": span.span_data.type,
    }
    if span.parent_id:
        attrs["openai_agents.parent_id_original"] = span.parent_id
    return attrs


def span_data_attributes(span_data, *, include_raw_payloads: bool) -> dict[str, Any]:
    span_type = span_data.type
    if span_type == "generation":
        return _generation_attributes(span_data, include_raw_payloads=include_raw_payloads)
    if span_type == "response":
        return _response_attributes(span_data, include_raw_payloads=include_raw_payloads)
    if span_type == "function":
        return _function_attributes(span_data, include_raw_payloads=include_raw_payloads)
    if span_type == "agent":
        return _agent_attributes(span_data, include_raw_payloads=include_raw_payloads)
    if span_type == "handoff":
        return _handoff_attributes(span_data)
    if span_type == "guardrail":
        return _guardrail_attributes(span_data)
    if span_type == "custom":
        return _custom_attributes(span_data, include_raw_payloads=include_raw_payloads)
    exported = _export_if_possible(span_data)
    return {"openai_agents.raw_json": _json_dumps(exported)} if include_raw_payloads and exported else {}


def _generation_attributes(span_data, *, include_raw_payloads: bool) -> dict[str, Any]:
    attrs: dict[str, Any] = {
        "gen_ai.provider.name": "openai.responses",
    }
    if span_data.model:
        attrs["gen_ai.request.model"] = span_data.model

    usage = span_data.usage or {}
    input_tokens = usage.get("input_tokens", usage.get("prompt_tokens", _UNSET))
    if input_tokens is not _UNSET and input_tokens is not None:
        attrs["gen_ai.usage.input_tokens"] = int(input_tokens)
    output_tokens = usage.get("output_tokens", usage.get("completion_tokens", _UNSET))
    if output_tokens is not _UNSET and output_tokens is not None:
        attrs["gen_ai.usage.output_tokens"] = int(output_tokens)

    _flatten_indexed_messages("gen_ai.prompt", normalize_messages(span_data.input), attrs)
    _flatten_indexed_messages("gen_ai.completion", normalize_messages(span_data.output), attrs)

    if include_raw_payloads and span_data.model_config:
        attrs["openai_agents.model_config_json"] = _json_dumps(span_data.model_config)
    if include_raw_payloads and span_data.input is not None:
        attrs["openai_agents.generation.raw_input_json"] = _json_dumps(_jsonable(span_data.input))
    if include_raw_payloads and span_data.output is not None:
        attrs["openai_agents.generation.raw_output_json"] = _json_dumps(_jsonable(span_data.output))

    return attrs


def _response_attributes(span_data, *, include_raw_payloads: bool) -> dict[str, Any]:
    attrs: dict[str, Any] = {
        "gen_ai.provider.name": "openai.responses",
    }
    response_obj = _jsonable(getattr(span_data, "response", None))
    if isinstance(response_obj, Mapping):
        model = response_obj.get("model")
        if model:
            attrs["gen_ai.request.model"] = str(model)
            attrs["gen_ai.response.model"] = str(model)

        response_id = response_obj.get("id") or response_obj.get("response_id")
        if response_id:
            attrs["gen_ai.response.id"] = str(response_id)

        usage = response_obj.get("usage") or {}
        if isinstance(usage, Mapping):
            input_tokens = usage.get("input_tokens", usage.get("prompt_tokens", _UNSET))
            if input_tokens is not _UNSET and input_tokens is not None:
                attrs["gen_ai.usage.input_tokens"] = int(input_tokens)
            output_tokens = usage.get("output_tokens", usage.get("completion_tokens", _UNSET))
            if output_tokens is not _UNSET and output_tokens is not None:
                attrs["gen_ai.usage.output_tokens"] = int(output_tokens)

        _flatten_indexed_messages(
            "gen_ai.prompt",
            normalize_response_input_items(getattr(span_data, "input", None) or response_obj.get("input")),
            attrs,
        )
        _flatten_indexed_messages(
            "gen_ai.completion", normalize_response_output_items(response_obj.get("output")), attrs
        )

        if include_raw_payloads:
            attrs["openai_agents.response.raw_json"] = _json_dumps(response_obj)
            if response_obj.get("object") is not None:
                attrs["openai_agents.response.object"] = str(response_obj["object"])

    return attrs


def _function_attributes(span_data, *, include_raw_payloads: bool) -> dict[str, Any]:
    attrs: dict[str, Any] = {
        "openai_agents.function.name": span_data.name,
    }
    if span_data.input is not None:
        attrs["openai_agents.function.input"] = str(span_data.input)
    if span_data.output is not None:
        output = _jsonable(span_data.output)
        if isinstance(output, str):
            attrs["openai_agents.function.output"] = output
        elif include_raw_payloads:
            attrs["openai_agents.function.output_json"] = _json_dumps(output)
    if include_raw_payloads and span_data.mcp_data is not None:
        attrs["openai_agents.function.mcp_data_json"] = _json_dumps(_jsonable(span_data.mcp_data))
    return attrs


def _agent_attributes(span_data, *, include_raw_payloads: bool) -> dict[str, Any]:
    attrs: dict[str, Any] = {
        "openai_agents.agent.name": span_data.name,
    }
    if span_data.output_type is not None:
        attrs["openai_agents.agent.output_type"] = str(span_data.output_type)
    if include_raw_payloads and span_data.handoffs is not None:
        attrs["openai_agents.agent.handoffs_json"] = _json_dumps(_jsonable(span_data.handoffs))
    if include_raw_payloads and span_data.tools is not None:
        attrs["openai_agents.agent.tools_json"] = _json_dumps(_jsonable(span_data.tools))
    return attrs


def _handoff_attributes(span_data) -> dict[str, Any]:
    attrs: dict[str, Any] = {}
    if span_data.from_agent is not None:
        attrs["openai_agents.handoff.from_agent"] = span_data.from_agent
    if span_data.to_agent is not None:
        attrs["openai_agents.handoff.to_agent"] = span_data.to_agent
    return attrs


def _guardrail_attributes(span_data) -> dict[str, Any]:
    return {
        "openai_agents.guardrail.name": span_data.name,
        "openai_agents.guardrail.triggered": span_data.triggered,
    }


def _custom_attributes(span_data, *, include_raw_payloads: bool) -> dict[str, Any]:
    attrs = {"openai_agents.custom.name": span_data.name}
    if include_raw_payloads:
        attrs["openai_agents.custom.data_json"] = _json_dumps(_jsonable(span_data.data))
    return attrs


def normalize_messages(items: Sequence[Mapping[str, Any]] | None) -> list[dict[str, Any]]:
    if not isinstance(items, Sequence):
        return []

    messages: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, Mapping):
            continue

        normalized: dict[str, Any] = {}
        role = item.get("role")
        if role is not None:
            normalized["role"] = str(role)

        content = _extract_text_content(item.get("content"))
        if content:
            normalized["content"] = content

        tool_calls = _normalize_tool_calls(item.get("tool_calls"))
        if tool_calls:
            normalized["tool_calls"] = tool_calls

        tool_call_id = item.get("tool_call_id")
        if tool_call_id:
            normalized["tool_call_id"] = str(tool_call_id)

        if normalized:
            messages.append(normalized)

    return messages


def normalize_response_input_items(items: Any) -> list[dict[str, Any]]:
    return _normalize_response_items(items)


def normalize_response_output_items(items: Any) -> list[dict[str, Any]]:
    return _normalize_response_items(items)


def _normalize_response_items(items: Any) -> list[dict[str, Any]]:
    if not isinstance(items, Sequence):
        return []

    messages: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, Mapping):
            continue

        item_type = item.get("type")
        if item_type == "message":
            role = str(item.get("role") or "assistant")
            content = _extract_response_content(item.get("content"))
            message: dict[str, Any] = {"role": role}
            if content:
                message["content"] = content
            if message:
                messages.append(message)
            continue

        if item_type == "function_call":
            tool_call = _normalize_response_function_call(item)
            if tool_call:
                messages.append({"role": "assistant", "tool_calls": [tool_call]})
            continue

        if item_type == "function_call_output":
            call_id = item.get("call_id")
            output = item.get("output")
            message = {
                "role": "tool",
                "tool_call_id": str(call_id or ""),
                "content": "" if output is None else str(output),
            }
            messages.append(message)

    return messages


def _normalize_response_function_call(item: Mapping[str, Any]) -> dict[str, Any] | None:
    call_id = item.get("call_id") or item.get("id")
    name = item.get("name")
    if not call_id and not name:
        return None

    arguments = item.get("arguments", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except Exception:
            pass

    return {
        "id": "" if call_id is None else str(call_id),
        "type": "tool_call",
        "name": "tool" if name is None else str(name),
        "arguments": _jsonable(arguments),
    }


def _extract_response_content(content: Any) -> str | None:
    if isinstance(content, str):
        return content
    if not isinstance(content, Sequence):
        return None

    text_parts: list[str] = []
    for part in content:
        if not isinstance(part, Mapping):
            continue
        part_type = part.get("type")
        if part_type in {"input_text", "output_text", "text"}:
            text = part.get("text")
            if text is not None:
                text_parts.append(str(text))
    return "\n".join(text_parts) if text_parts else None


def _extract_text_content(content: Any) -> str | None:
    if content is None:
        return None
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence) and not isinstance(content, (str, bytes, bytearray)):
        text_parts: list[str] = []
        for part in content:
            if not isinstance(part, Mapping):
                continue
            part_type = part.get("type")
            if part_type in {"text", "input_text", "output_text"}:
                text = part.get("text", part.get("content"))
                if text is not None:
                    text_parts.append(str(text))
        return "\n".join(text_parts) if text_parts else None
    return str(content)


def _normalize_tool_calls(tool_calls: Any) -> list[dict[str, Any]]:
    if not isinstance(tool_calls, Sequence):
        return []

    normalized: list[dict[str, Any]] = []
    for call in tool_calls:
        if not isinstance(call, Mapping):
            continue

        if "function" in call and isinstance(call["function"], Mapping):
            function = call["function"]
            name = function.get("name", call.get("name", "tool"))
            arguments = function.get("arguments", call.get("arguments", {}))
        else:
            name = call.get("name", "tool")
            arguments = call.get("arguments", {})

        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except Exception:
                pass

        normalized.append(
            {
                "id": str(call.get("id", "")),
                "type": "tool_call",
                "name": str(name),
                "arguments": _jsonable(arguments),
            }
        )

    return normalized


def _flatten_indexed_messages(prefix: str, messages: list[dict[str, Any]], attrs: dict[str, Any]) -> None:
    for index, message in enumerate(messages):
        for key, value in message.items():
            attr_key = f"{prefix}.{index}.{key}"
            if _is_scalar(value):
                attrs[attr_key] = value
            else:
                attrs[attr_key] = _json_dumps(value)


def _export_if_possible(value: Any) -> Any:
    exported = getattr(value, "export", None)
    if callable(exported):
        return _jsonable(exported())
    return _jsonable(value)


def _jsonable(value: Any) -> Any:
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _jsonable(model_dump(mode="json", exclude_none=True))

    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _jsonable(to_dict())

    if isinstance(value, Mapping):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_jsonable(item) for item in value]
    return value


def _json_dumps(value: Any) -> str:
    return json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True)


def _sanitize_key(key: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", key)


def _is_scalar(value: Any) -> bool:
    return isinstance(value, (str, bool, int, float))
