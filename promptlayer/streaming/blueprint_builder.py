"""
Blueprint builders for various LLM providers

This module contains functions to build prompt blueprints from LLM responses
and streaming events for different providers (OpenAI, Anthropic, etc.)
"""

from typing import Any, Dict, List, Optional

_OUTPUT_FORMAT_TO_MIME = {
    "png": "image/png",
    "webp": "image/webp",
    "jpeg": "image/jpeg",
    "jpg": "image/jpeg",
    "gif": "image/gif",
}


def _create_tool_call(call_id: str, function_name: str, arguments: Any, tool_id: str = None) -> Dict[str, Any]:
    """Create a standardized tool call structure"""
    tool_call = {"id": call_id, "type": "function", "function": {"name": function_name, "arguments": arguments}}
    if tool_id:
        tool_call["tool_id"] = tool_id
    return tool_call


def _create_content_item(content_type: str, item_id: str = None, **kwargs) -> Dict[str, Any]:
    """Create a standardized content item"""
    content_item = {"type": content_type}
    if item_id:
        content_item["id"] = item_id
    content_item.update(kwargs)
    return content_item


def _build_assistant_message(
    content: List[Dict], tool_calls: Optional[List[Dict]] = None, template_format: str = "f-string"
) -> Dict[str, Any]:
    """Build a standardized assistant message structure"""
    message = {"role": "assistant", "content": content, "input_variables": [], "template_format": template_format}

    if tool_calls:
        message["tool_calls"] = tool_calls

    return message


def _build_prompt_blueprint(assistant_message: Dict[str, Any], metadata: Any) -> Dict[str, Any]:
    """Build a standardized prompt blueprint structure"""
    prompt_template = {"type": "chat", "messages": [assistant_message], "input_variables": []}

    blueprint = {"prompt_template": prompt_template, "metadata": metadata}

    return blueprint


def build_prompt_blueprint_from_openai_chunk(chunk, metadata):
    """Build a prompt blueprint from an OpenAI chat completion chunk"""

    assistant_content = []
    tool_calls = []

    if hasattr(chunk, "choices") and len(chunk.choices) > 0:
        delta = chunk.choices[0].delta

        if hasattr(delta, "content") and delta.content:
            assistant_content.append(_create_content_item("text", text=delta.content))

        if hasattr(delta, "tool_calls") and delta.tool_calls:
            for tool_call in delta.tool_calls:
                tool_calls.append(
                    _create_tool_call(
                        getattr(tool_call, "id", ""),
                        getattr(tool_call.function, "name", "") if tool_call.function else "",
                        getattr(tool_call.function, "arguments", "") if tool_call.function else "",
                    )
                )

    assistant_message = _build_assistant_message(assistant_content, tool_calls or None)
    return _build_prompt_blueprint(assistant_message, metadata)


def _map_output_item_to_blueprint_content(item, assistant_content, tool_calls, metadata=None):
    """Map an OpenAI Responses output item to blueprint content items."""
    item_type = item.get("type")
    item_id = item.get("id")

    if item_type == "reasoning":
        summary = item.get("summary", [])
        for summary_part in summary:
            if summary_part.get("type") == "summary_text":
                text = summary_part.get("text", "")
                if text:
                    assistant_content.append(
                        _create_content_item("thinking", item_id=item_id, thinking=text, signature=None)
                    )

    elif item_type == "function_call":
        tool_calls.append(
            _create_tool_call(item.get("call_id", ""), item.get("name", ""), item.get("arguments", ""), tool_id=item_id)
        )

    elif item_type == "message":
        content = item.get("content", [])
        for content_part in content:
            if content_part.get("type") == "output_text":
                text = content_part.get("text", "")
                assistant_content.append(_create_content_item("text", item_id=item_id, text=text))

    elif item_type == "code_interpreter_call":
        assistant_content.append(
            _create_content_item(
                "code", item_id=item_id, code=item.get("code", ""), container_id=item.get("container_id")
            )
        )

    elif item_type in ("web_search_call", "file_search_call"):
        pass

    elif item_type == "shell_call":
        assistant_content.append(
            _create_content_item(
                "shell_call",
                item_id=item_id,
                call_id=item.get("call_id"),
                action=item.get("action", {}),
                status=item.get("status", "completed"),
            )
        )

    elif item_type == "shell_call_output":
        assistant_content.append(
            _create_content_item(
                "shell_call_output",
                item_id=item_id,
                call_id=item.get("call_id"),
                output=item.get("output", []),
                status=item.get("status"),
            )
        )

    elif item_type == "apply_patch_call":
        assistant_content.append(
            _create_content_item(
                "apply_patch_call",
                item_id=item_id,
                call_id=item.get("call_id"),
                operation=item.get("operation", {}),
                status=item.get("status", "completed"),
            )
        )

    elif item_type == "apply_patch_call_output":
        assistant_content.append(
            _create_content_item(
                "apply_patch_call_output",
                item_id=item_id,
                call_id=item.get("call_id"),
                output=item.get("output"),
                status=item.get("status"),
            )
        )

    elif item_type == "mcp_list_tools":
        assistant_content.append(
            _create_content_item(
                "mcp_list_tools",
                item_id=item_id,
                server_label=item.get("server_label", ""),
                tools=item.get("tools", []),
                error=item.get("error"),
            )
        )

    elif item_type == "mcp_call":
        assistant_content.append(
            _create_content_item(
                "mcp_call",
                item_id=item_id,
                call_id=item.get("call_id"),
                name=item.get("name", ""),
                server_label=item.get("server_label", ""),
                arguments=item.get("arguments", ""),
                output=item.get("output"),
                error=item.get("error"),
                approval_request_id=item.get("approval_request_id"),
                status=item.get("status", "completed"),
            )
        )

    elif item_type == "mcp_approval_request":
        assistant_content.append(
            _create_content_item(
                "mcp_approval_request",
                item_id=item_id,
                name=item.get("name", ""),
                arguments=item.get("arguments", ""),
                server_label=item.get("server_label", ""),
            )
        )

    elif item_type == "mcp_approval_response":
        assistant_content.append(
            _create_content_item(
                "mcp_approval_response",
                item_id=item_id,
                approval_request_id=item.get("approval_request_id", ""),
                approve=item.get("approve", False),
            )
        )

    elif item_type == "image_generation_call":
        result = item.get("result", "")
        if result:
            output_format = item.get("output_format")
            if output_format:
                mime_type = _OUTPUT_FORMAT_TO_MIME.get(output_format, "image/png")
            else:
                model_params = metadata.get("model", {}).get("parameters", {}) if metadata else {}
                fmt = model_params.get("output_format", "png") or "png"
                mime_type = _OUTPUT_FORMAT_TO_MIME.get(fmt, "image/png")
            provider_metadata = {}
            for key in ("revised_prompt", "background", "size", "quality", "output_format"):
                val = item.get(key)
                if val is not None:
                    provider_metadata[key] = val
            content_kwargs = dict(
                item_id=item_id,
                url=result,
                mime_type=mime_type,
                media_type="image",
            )
            if provider_metadata:
                content_kwargs["provider_metadata"] = provider_metadata
            assistant_content.append(_create_content_item("output_media", **content_kwargs))


def build_prompt_blueprint_from_openai_responses_event(event, metadata):
    """Build a prompt blueprint from an OpenAI responses event"""

    assistant_content = []
    tool_calls = []

    event_dict = event.model_dump() if hasattr(event, "model_dump") else event
    event_type = event_dict.get("type")

    if event_type == "response.reasoning_summary_text.delta":
        delta = event_dict.get("delta", "")
        item_id = event_dict.get("item_id")
        if delta:
            assistant_content.append(_create_content_item("thinking", item_id=item_id, thinking=delta, signature=None))

    elif event_type == "response.reasoning_summary_text.done":
        final_text = event_dict.get("text", "")
        item_id = event_dict.get("item_id")
        if final_text:
            assistant_content.append(
                _create_content_item("thinking", item_id=item_id, thinking=final_text, signature=None)
            )

    elif event_type == "response.reasoning_summary_part.added":
        part = event_dict.get("part", {})
        item_id = event_dict.get("item_id")
        if part.get("type") == "summary_text":
            text = part.get("text", "")
            assistant_content.append(_create_content_item("thinking", item_id=item_id, thinking=text, signature=None))

    elif event_type == "response.reasoning_summary_part.done":
        part = event_dict.get("part", {})
        item_id = event_dict.get("item_id")
        if part.get("type") == "summary_text":
            text = part.get("text", "")
            if text:
                assistant_content.append(
                    _create_content_item("thinking", item_id=item_id, thinking=text, signature=None)
                )

    elif event_type == "response.function_call_arguments.delta":
        item_id = event_dict.get("item_id")
        delta = event_dict.get("delta", "")
        if delta:
            tool_calls.append(_create_tool_call("", "", delta, tool_id=item_id))

    elif event_type == "response.function_call_arguments.done":
        item_id = event_dict.get("item_id")
        final_arguments = event_dict.get("arguments", "")
        if final_arguments:
            tool_calls.append(_create_tool_call("", "", final_arguments, tool_id=item_id))

    elif event_type == "response.output_item.added":
        item = event_dict.get("item", {})
        item_type = item.get("type")
        item_id = item.get("id")

        if item_type == "reasoning":
            assistant_content.append(_create_content_item("thinking", item_id=item_id, thinking="", signature=None))
        elif item_type == "function_call":
            tool_calls.append(_create_tool_call(item.get("call_id", ""), item.get("name", ""), "", tool_id=item_id))
        elif item_type == "message":
            assistant_content.append(_create_content_item("text", item_id=item_id, text=""))
        elif item_type == "code_interpreter_call":
            assistant_content.append(
                _create_content_item(
                    "code", item_id=item_id, code=item.get("code", ""), container_id=item.get("container_id")
                )
            )
        elif item_type in ("web_search_call", "file_search_call"):
            assistant_content.append(_create_content_item("text", item_id=item_id, text="", annotation=[]))
        elif item_type == "shell_call":
            assistant_content.append(
                _create_content_item(
                    "shell_call",
                    item_id=item_id,
                    call_id=item.get("call_id"),
                    action=item.get("action", {}),
                    status=item.get("status", "in_progress"),
                )
            )
        elif item_type == "shell_call_output":
            assistant_content.append(
                _create_content_item(
                    "shell_call_output",
                    item_id=item_id,
                    call_id=item.get("call_id"),
                    output=item.get("output", []),
                    status=item.get("status"),
                )
            )
        elif item_type == "apply_patch_call":
            assistant_content.append(
                _create_content_item(
                    "apply_patch_call",
                    item_id=item_id,
                    call_id=item.get("call_id"),
                    operation=item.get("operation", {}),
                    status=item.get("status", "in_progress"),
                )
            )
        elif item_type == "apply_patch_call_output":
            assistant_content.append(
                _create_content_item(
                    "apply_patch_call_output",
                    item_id=item_id,
                    call_id=item.get("call_id"),
                    output=item.get("output"),
                    status=item.get("status"),
                )
            )
        elif item_type == "mcp_list_tools":
            assistant_content.append(
                _create_content_item(
                    "mcp_list_tools",
                    item_id=item_id,
                    server_label=item.get("server_label", ""),
                    tools=item.get("tools", []),
                    error=item.get("error"),
                )
            )
        elif item_type == "mcp_call":
            assistant_content.append(
                _create_content_item(
                    "mcp_call",
                    item_id=item_id,
                    call_id=item.get("call_id"),
                    name=item.get("name", ""),
                    server_label=item.get("server_label", ""),
                    arguments=item.get("arguments", ""),
                    error=item.get("error"),
                    approval_request_id=item.get("approval_request_id"),
                    status=item.get("status", "in_progress"),
                )
            )
        elif item_type == "mcp_approval_request":
            assistant_content.append(
                _create_content_item(
                    "mcp_approval_request",
                    item_id=item_id,
                    name=item.get("name", ""),
                    arguments=item.get("arguments", ""),
                    server_label=item.get("server_label", ""),
                )
            )
        elif item_type == "mcp_approval_response":
            assistant_content.append(
                _create_content_item(
                    "mcp_approval_response",
                    item_id=item_id,
                    approval_request_id=item.get("approval_request_id", ""),
                    approve=item.get("approve", False),
                )
            )
        elif item_type == "image_generation_call":
            result = item.get("result") or ""
            output_format = item.get("output_format")
            if output_format:
                mime_type = _OUTPUT_FORMAT_TO_MIME.get(output_format, "image/png")
            else:
                model_params = metadata.get("model", {}).get("parameters", {}) if metadata else {}
                fmt = model_params.get("output_format", "png") or "png"
                mime_type = _OUTPUT_FORMAT_TO_MIME.get(fmt, "image/png")
            provider_metadata = {}
            for key in ("revised_prompt", "background", "size", "quality", "output_format"):
                val = item.get(key)
                if val is not None:
                    provider_metadata[key] = val
            content_kwargs = dict(
                item_id=item_id,
                url=result,
                mime_type=mime_type,
                media_type="image",
            )
            if provider_metadata:
                content_kwargs["provider_metadata"] = provider_metadata
            assistant_content.append(_create_content_item("output_media", **content_kwargs))

    elif event_type == "response.content_part.added":
        item_id = event_dict.get("item_id")
        part = event_dict.get("part", {})
        part_type = part.get("type", "output_text")

        if part_type == "output_text":
            text = part.get("text", "")
            assistant_content.append(_create_content_item("text", item_id=item_id, text=text if text else ""))

    elif event_type == "response.content_part.done":
        item_id = event_dict.get("item_id")
        part = event_dict.get("part", {})
        part_type = part.get("type", "output_text")

        if part_type == "output_text":
            text = part.get("text", "")
            annotations = part.get("annotations", [])
            content_item = _create_content_item("text", item_id=item_id, text=text if text else "")
            if annotations:
                mapped_annotations = []
                for ann in annotations:
                    atype = ann.get("type")
                    if atype == "url_citation":
                        mapped_annotations.append(
                            {
                                "type": "url_citation",
                                "title": ann.get("title"),
                                "url": ann.get("url"),
                                "start_index": ann.get("start_index"),
                                "end_index": ann.get("end_index"),
                            }
                        )
                    elif atype == "file_citation":
                        mapped_annotations.append(
                            {
                                "type": "file_citation",
                                "index": ann.get("index"),
                                "file_id": ann.get("file_id"),
                                "filename": ann.get("filename"),
                            }
                        )
                    else:
                        mapped_annotations.append(ann)
                content_item["annotations"] = mapped_annotations
            assistant_content.append(content_item)

    elif event_type == "response.output_text.annotation.added":
        annotation = event_dict.get("annotation", {}) or {}
        atype = annotation.get("type")
        mapped_annotation = None

        if atype == "url_citation":
            mapped_annotation = {
                "type": "url_citation",
                "title": annotation.get("title"),
                "url": annotation.get("url"),
                "start_index": annotation.get("start_index"),
                "end_index": annotation.get("end_index"),
            }
        elif atype == "file_citation":
            mapped_annotation = {
                "type": "file_citation",
                "index": annotation.get("index"),
                "file_id": annotation.get("file_id"),
                "filename": annotation.get("filename"),
            }
        else:
            mapped_annotation = annotation

        assistant_content.append(
            _create_content_item("text", item_id=event_dict.get("item_id"), text="", annotation=[mapped_annotation])
        )

    elif event_type == "response.code_interpreter_call.in_progress":
        item_id = event_dict.get("item_id")
        assistant_content.append(
            _create_content_item(
                "code", item_id=item_id, code=event_dict.get("code"), container_id=event_dict.get("container_id")
            )
        )

    elif event_type == "response.code_interpreter_call_code.delta":
        item_id = event_dict.get("item_id")
        delta_code = event_dict.get("delta", "")
        if delta_code:
            assistant_content.append(
                _create_content_item(
                    "code", item_id=item_id, code=delta_code, container_id=event_dict.get("container_id")
                )
            )

    elif event_type == "response.code_interpreter_call_code.done":
        item_id = event_dict.get("item_id")
        final_code = event_dict.get("code", "")
        if final_code:
            assistant_content.append(
                _create_content_item(
                    "code", item_id=item_id, code=final_code, container_id=event_dict.get("container_id")
                )
            )

    elif event_type == "response.code_interpreter_call.interpreting":
        item_id = event_dict.get("item_id")
        assistant_content.append(
            _create_content_item(
                "code", item_id=item_id, code=event_dict.get("code"), container_id=event_dict.get("container_id")
            )
        )

    elif event_type == "response.code_interpreter_call.completed":
        item_id = event_dict.get("item_id")
        assistant_content.append(
            _create_content_item(
                "code", item_id=item_id, code=event_dict.get("code"), container_id=event_dict.get("container_id")
            )
        )

    elif event_type == "response.shell_call_command.added":
        item_id = event_dict.get("item_id")
        command = event_dict.get("command", "")
        assistant_content.append(
            _create_content_item("shell_call", item_id=item_id, action={"commands": [command] if command else []})
        )

    elif event_type == "response.shell_call_command.delta":
        item_id = event_dict.get("item_id")
        delta_command = event_dict.get("delta", "")
        if delta_command:
            assistant_content.append(
                _create_content_item("shell_call", item_id=item_id, action={"commands": [delta_command]})
            )

    elif event_type == "response.shell_call_command.done":
        item_id = event_dict.get("item_id")
        final_command = event_dict.get("command", "")
        if final_command:
            assistant_content.append(
                _create_content_item("shell_call", item_id=item_id, action={"commands": [final_command]})
            )

    elif event_type == "response.shell_call_output_content.delta":
        item_id = event_dict.get("item_id")
        delta = event_dict.get("delta", {})
        if delta:
            assistant_content.append(
                _create_content_item(
                    "shell_call_output",
                    item_id=item_id,
                    output=[delta] if isinstance(delta, dict) else [{"stdout": delta}],
                )
            )

    elif event_type == "response.shell_call_output_content.done":
        item_id = event_dict.get("item_id")
        output = event_dict.get("output", [])
        if output:
            assistant_content.append(_create_content_item("shell_call_output", item_id=item_id, output=output))

    elif event_type == "response.apply_patch_call_operation_diff.delta":
        item_id = event_dict.get("item_id")
        delta_diff = event_dict.get("delta", "")
        if delta_diff:
            assistant_content.append(
                _create_content_item("apply_patch_call", item_id=item_id, operation={"diff": delta_diff})
            )

    elif event_type == "response.apply_patch_call_operation_diff.done":
        item_id = event_dict.get("item_id")
        final_diff = event_dict.get("diff", "")
        if final_diff:
            assistant_content.append(
                _create_content_item("apply_patch_call", item_id=item_id, operation={"diff": final_diff})
            )

    elif event_type == "response.mcp_call_arguments.delta":
        item_id = event_dict.get("item_id")
        delta_args = event_dict.get("delta", "")
        if delta_args:
            assistant_content.append(_create_content_item("mcp_call", item_id=item_id, arguments=delta_args))

    elif event_type == "response.mcp_call_arguments.done":
        item_id = event_dict.get("item_id")
        final_args = event_dict.get("arguments", "")
        if final_args:
            assistant_content.append(_create_content_item("mcp_call", item_id=item_id, arguments=final_args))

    elif event_type == "response.output_text.delta":
        item_id = event_dict.get("item_id")
        delta_text = event_dict.get("delta", "")
        if delta_text:
            assistant_content.append(_create_content_item("text", item_id=item_id, text=delta_text))

    elif event_type == "response.output_text.done":
        item_id = event_dict.get("item_id")
        final_text = event_dict.get("text", "")
        assistant_content.append(_create_content_item("text", item_id=item_id, text=final_text))

    elif event_type == "response.image_generation_call.partial_image":
        item_id = event_dict.get("item_id")
        partial_b64 = event_dict.get("partial_image_b64", "")
        if partial_b64:
            output_format = event_dict.get("output_format")
            if output_format:
                partial_mime = _OUTPUT_FORMAT_TO_MIME.get(output_format, "image/png")
            else:
                model_params = metadata.get("model", {}).get("parameters", {}) if metadata else {}
                fmt = model_params.get("output_format", "png") or "png"
                partial_mime = _OUTPUT_FORMAT_TO_MIME.get(fmt, "image/png")
            assistant_content.append(
                _create_content_item(
                    "output_media", item_id=item_id, url=partial_b64, mime_type=partial_mime, media_type="image"
                )
            )

    elif event_type == "response.output_item.done":
        item = event_dict.get("item", {})
        _map_output_item_to_blueprint_content(item, assistant_content, tool_calls, metadata)

    elif event_type == "response.completed":
        response_info = event_dict.get("response", {})
        for output_item in response_info.get("output", []):
            _map_output_item_to_blueprint_content(output_item, assistant_content, tool_calls, metadata)

    assistant_message = _build_assistant_message(assistant_content, tool_calls or None)
    return _build_prompt_blueprint(assistant_message, metadata)


def _anthropic_content_block_to_dict(content_block) -> Dict[str, Any]:
    """Convert an Anthropic content block to a dict for the blueprint."""
    block_type = getattr(content_block, "type", None)
    result = {"type": block_type}
    if block_type == "server_tool_use":
        result["id"] = getattr(content_block, "id", "")
        result["name"] = getattr(content_block, "name", "")
        result["input"] = getattr(content_block, "input", {})
    elif block_type == "web_search_tool_result":
        result["tool_use_id"] = getattr(content_block, "tool_use_id", "")
        content = getattr(content_block, "content", [])
        if hasattr(content, "model_dump"):
            result["content"] = content.model_dump() if callable(getattr(content, "model_dump", None)) else content
        elif isinstance(content, list):
            result["content"] = [
                item.model_dump() if hasattr(item, "model_dump") else (item if isinstance(item, dict) else {})
                for item in content
            ]
        else:
            result["content"] = content
    elif block_type == "bash_code_execution_tool_result":
        result["tool_use_id"] = getattr(content_block, "tool_use_id", "")
        content = getattr(content_block, "content", {})
        result["content"] = content.model_dump() if hasattr(content, "model_dump") else content
    elif block_type == "text_editor_code_execution_tool_result":
        result["tool_use_id"] = getattr(content_block, "tool_use_id", "")
        content = getattr(content_block, "content", {})
        result["content"] = content.model_dump() if hasattr(content, "model_dump") else content
    else:
        # Generic fallback: try model_dump, else just use what we have
        if hasattr(content_block, "model_dump"):
            return content_block.model_dump()
        return result
    return result


def _anthropic_citation_to_annotation(citation) -> Optional[Dict[str, Any]]:
    """Convert an Anthropic citation object to an annotation dict."""
    citation_type = getattr(citation, "type", "")
    if citation_type == "web_search_result_location":
        ann: Dict[str, Any] = {
            "type": "url_citation",
            "url": getattr(citation, "url", ""),
            "title": getattr(citation, "title", "") or "",
            "start_index": getattr(citation, "start_index", 0),
            "end_index": getattr(citation, "end_index", 0),
        }
        cited_text = getattr(citation, "cited_text", None)
        if cited_text:
            ann["cited_text"] = cited_text
        return ann
    # Generic fallback: include what we can
    ann = {"type": citation_type}
    for field in ("url", "title", "cited_text", "source"):
        val = getattr(citation, field, None)
        if val is not None:
            ann[field] = val
    return ann


def build_prompt_blueprint_from_anthropic_event(event, metadata):
    """Build a prompt blueprint from an Anthropic stream event"""

    assistant_content = []
    tool_calls = []

    if hasattr(event, "type"):
        if event.type == "content_block_start" and hasattr(event, "content_block"):
            block_type = getattr(event.content_block, "type", None)
            if block_type == "thinking":
                assistant_content.append(_create_content_item("thinking", thinking="", signature=None))
            elif block_type == "text":
                assistant_content.append(_create_content_item("text", text=""))
            elif block_type == "tool_use":
                tool_calls.append(
                    _create_tool_call(
                        getattr(event.content_block, "id", ""),
                        getattr(event.content_block, "name", ""),
                        getattr(event.content_block, "input", ""),
                    )
                )
            elif block_type == "server_tool_use":
                assistant_content.append(
                    _create_content_item(
                        "server_tool_use",
                        id=getattr(event.content_block, "id", ""),
                        name=getattr(event.content_block, "name", ""),
                        input=getattr(event.content_block, "input", {}),
                    )
                )
            elif block_type in (
                "web_search_tool_result",
                "bash_code_execution_tool_result",
                "text_editor_code_execution_tool_result",
            ):
                assistant_content.append(_anthropic_content_block_to_dict(event.content_block))
        elif event.type == "content_block_delta" and hasattr(event, "delta"):
            if hasattr(event.delta, "text"):
                assistant_content.append(_create_content_item("text", text=event.delta.text))
            elif hasattr(event.delta, "thinking"):
                assistant_content.append(
                    _create_content_item(
                        "thinking", thinking=event.delta.thinking, signature=getattr(event.delta, "signature", None)
                    )
                )
            elif hasattr(event.delta, "partial_json"):
                tool_calls.append(
                    _create_tool_call(
                        getattr(event.delta, "id", ""),
                        getattr(event.delta, "name", ""),
                        getattr(event.delta, "input", event.delta.partial_json),
                    )
                )
            elif hasattr(event.delta, "citation"):
                ann = _anthropic_citation_to_annotation(event.delta.citation)
                if ann:
                    assistant_content.append({"type": "text", "text": "", "annotations": [ann]})

    assistant_message = _build_assistant_message(assistant_content, tool_calls or None)
    return _build_prompt_blueprint(assistant_message, metadata)


def _get_chunk_web_info(chunk: dict) -> tuple:
    """Extract (uri, title) from a Google grounding chunk dict."""
    web = chunk.get("web") or chunk.get("webChunk")
    if isinstance(web, dict):
        return web.get("uri", ""), web.get("title", "")
    return chunk.get("uri", ""), chunk.get("title", "")


def _grounding_metadata_to_annotations(grounding_metadata) -> List[Dict[str, Any]]:
    """Convert Google grounding_metadata to annotation dicts.

    Handles web chunks (url_citation), maps chunks (map_citation), and
    retrieved_context chunks from file search (file_citation).
    """
    if not grounding_metadata:
        return []

    # Support both SDK objects (with attributes) and plain dicts
    def _get(obj, key, default=None):
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    def _to_dict(obj):
        if obj is None:
            return None
        if isinstance(obj, dict):
            return obj
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        if hasattr(obj, "__dict__"):
            return vars(obj)
        return None

    gm = _to_dict(grounding_metadata) or {}
    chunks = gm.get("grounding_chunks") or gm.get("groundingChunks") or []
    chunks = [_to_dict(c) or c for c in chunks]
    supports = gm.get("grounding_supports") or gm.get("groundingSupports") or []
    supports = [_to_dict(s) or s for s in supports]

    if supports and chunks:
        annotations: List[Dict[str, Any]] = []
        for support in supports:
            if not isinstance(support, dict):
                continue
            segment = support.get("segment") or {}
            if not isinstance(segment, dict) and hasattr(segment, "model_dump"):
                segment = segment.model_dump()
            elif not isinstance(segment, dict):
                segment = {}
            chunk_indices = support.get("grounding_chunk_indices") or support.get("groundingChunkIndices") or []
            start_index = segment.get("start_index") or segment.get("startIndex") or 0
            end_index = segment.get("end_index") or segment.get("endIndex") or 0
            cited_text = segment.get("text")

            for chunk_idx in chunk_indices:
                if not isinstance(chunk_idx, int) or chunk_idx >= len(chunks):
                    continue
                chunk = chunks[chunk_idx]
                if not isinstance(chunk, dict):
                    continue

                maps = chunk.get("maps") or chunk.get("mapsChunk")
                if isinstance(maps, dict):
                    uri = maps.get("uri") or ""
                    title = maps.get("title") or ""
                    place_id = maps.get("place_id") or maps.get("placeId") or None
                    if uri or title:
                        ann = {
                            "type": "map_citation",
                            "url": uri,
                            "title": title or uri,
                            "start_index": start_index,
                            "end_index": end_index,
                        }
                        if cited_text:
                            ann["cited_text"] = cited_text
                        if place_id:
                            ann["place_id"] = place_id
                        annotations.append(ann)
                    continue

                retrieved = chunk.get("retrieved_context") or chunk.get("retrievedContext")
                if isinstance(retrieved, dict):
                    title = retrieved.get("title") or ""
                    if title:
                        annotations.append(
                            {
                                "type": "file_citation",
                                "file_id": title,
                                "filename": title,
                                "index": chunk_idx,
                            }
                        )
                    continue

                uri, title = _get_chunk_web_info(chunk)
                if uri:
                    ann = {
                        "type": "url_citation",
                        "url": uri,
                        "title": title or uri,
                        "start_index": start_index,
                        "end_index": end_index,
                    }
                    if cited_text:
                        ann["cited_text"] = cited_text
                    annotations.append(ann)
        return annotations

    # Fallback: chunks only, no segment info
    annotations = []
    for idx, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            continue

        maps = chunk.get("maps") or chunk.get("mapsChunk")
        if isinstance(maps, dict):
            uri = maps.get("uri") or ""
            title = maps.get("title") or ""
            place_id = maps.get("place_id") or maps.get("placeId") or None
            if uri or title:
                ann = {
                    "type": "map_citation",
                    "url": uri,
                    "title": title or uri,
                    "start_index": 0,
                    "end_index": 0,
                }
                if place_id:
                    ann["place_id"] = place_id
                annotations.append(ann)
            continue

        retrieved = chunk.get("retrieved_context") or chunk.get("retrievedContext")
        if isinstance(retrieved, dict):
            title = retrieved.get("title") or ""
            if title:
                annotations.append(
                    {
                        "type": "file_citation",
                        "file_id": title,
                        "filename": title,
                        "index": idx,
                    }
                )
            continue

        uri, title = _get_chunk_web_info(chunk)
        if uri:
            annotations.append(
                {
                    "type": "url_citation",
                    "url": uri,
                    "title": title or uri,
                    "start_index": 0,
                    "end_index": 0,
                }
            )
    return annotations


def build_prompt_blueprint_from_google_event(event, metadata):
    """
    Build a prompt blueprint from a Google (Gemini) streaming event (raw dict or GenerateContentResponse).
    """
    assistant_content = []
    tool_calls = []
    candidate = event.candidates[0]

    if candidate and hasattr(candidate, "content") and candidate.content and hasattr(candidate.content, "parts"):
        for part in candidate.content.parts:
            # Check specific part types before the thought flag, since
            # executable_code/code_execution_result can also have thought=True
            if hasattr(part, "executable_code") and part.executable_code:
                exec_code = part.executable_code
                language = getattr(exec_code, "language", "PYTHON")
                if hasattr(language, "value"):
                    language = language.value
                assistant_content.append(
                    _create_content_item(
                        "code",
                        code=getattr(exec_code, "code", ""),
                        language=language,
                    )
                )
            elif hasattr(part, "code_execution_result") and part.code_execution_result:
                exec_result = part.code_execution_result
                outcome = getattr(exec_result, "outcome", "OUTCOME_OK")
                if hasattr(outcome, "value"):
                    outcome = outcome.value
                assistant_content.append(
                    _create_content_item(
                        "code_execution_result",
                        output=getattr(exec_result, "output", ""),
                        outcome=outcome,
                    )
                )
            elif hasattr(part, "inline_data") and part.inline_data:
                if hasattr(part, "model_dump"):
                    part_dict = part.model_dump(mode="json")
                    inline_dict = part_dict.get("inline_data", {})
                    data = inline_dict.get("data", "")
                    mime_type = inline_dict.get("mime_type", "image/png")
                    thought_sig = part_dict.get("thought_signature")
                else:
                    inline = part.inline_data
                    data = getattr(inline, "data", "")
                    mime_type = getattr(inline, "mime_type", "image/png")
                    thought_sig = getattr(part, "thought_signature", None)
                content_kwargs = dict(url=data, mime_type=mime_type, media_type="image")
                if thought_sig is not None:
                    content_kwargs["provider_metadata"] = {"thought_signature": thought_sig}
                assistant_content.append(_create_content_item("output_media", **content_kwargs))
            elif hasattr(part, "thought") and part.thought is True and hasattr(part, "text") and part.text:
                assistant_content.append(_create_content_item("thinking", thinking=part.text, signature=None))
            elif hasattr(part, "text") and part.text is not None:
                assistant_content.append(_create_content_item("text", text=part.text))
            elif hasattr(part, "function_call") and part.function_call:
                tool_calls.append(
                    _create_tool_call(
                        getattr(part.function_call, "id", ""),
                        getattr(part.function_call, "name", ""),
                        getattr(part.function_call, "args", {}),
                    )
                )

    # Attach grounding annotations (Google Search / Maps / File Search) to text content
    grounding_metadata = getattr(candidate, "grounding_metadata", None)
    annotations = _grounding_metadata_to_annotations(grounding_metadata)

    # Attach url_context_metadata as url_citation annotations
    url_context_metadata = getattr(candidate, "url_context_metadata", None)
    if url_context_metadata:
        url_metadata_list = getattr(url_context_metadata, "url_metadata", None) or []
        for url_meta in url_metadata_list:
            retrieved_url = getattr(url_meta, "retrieved_url", None)
            if retrieved_url is not None:
                annotations.append(
                    {
                        "type": "url_citation",
                        "url": retrieved_url,
                        "title": None,
                        "start_index": None,
                        "end_index": None,
                    }
                )

    if annotations and assistant_content:
        for item in assistant_content:
            if item.get("type") == "text":
                item["annotations"] = (item.get("annotations") or []) + annotations
                break

    assistant_message = _build_assistant_message(assistant_content, tool_calls or None, template_format="f-string")
    return _build_prompt_blueprint(assistant_message, metadata)


def build_prompt_blueprint_from_bedrock_event(result, metadata):
    """
    Build a prompt blueprint from an Amazon Bedrock streaming event.
    """
    assistant_content = []
    tool_calls = []

    if "contentBlockDelta" in result:
        delta = result["contentBlockDelta"].get("delta", {})

        if "reasoningContent" in delta:
            reasoning_text = delta["reasoningContent"].get("text", "")
            signature = delta["reasoningContent"].get("signature")
            assistant_content.append(_create_content_item("thinking", thinking=reasoning_text, signature=signature))

        elif "text" in delta:
            assistant_content.append(_create_content_item("text", text=delta["text"]))

        elif "toolUse" in delta:
            tool_use = delta["toolUse"]
            assistant_content.append(
                _create_tool_call(tool_use.get("toolUseId", ""), tool_use.get("name", ""), tool_use.get("input", ""))
            )

    elif "contentBlockStart" in result:
        start_block = result["contentBlockStart"].get("start", {})

        if "toolUse" in start_block:
            tool_use = start_block["toolUse"]
            tool_calls.append(_create_tool_call(tool_use.get("toolUseId", ""), tool_use.get("name", ""), ""))

    assistant_message = _build_assistant_message(assistant_content, tool_calls or None)
    return _build_prompt_blueprint(assistant_message, metadata)


def build_prompt_blueprint_from_openai_images_event(event_dict, metadata):
    """Build a prompt blueprint from an OpenAI Images API streaming event."""
    assistant_content = []

    if not isinstance(event_dict, dict):
        event_dict = event_dict.model_dump() if hasattr(event_dict, "model_dump") else {}

    event_type = event_dict.get("type", "")

    model_params = metadata.get("model", {}).get("parameters", {}) if metadata else {}
    output_format = model_params.get("output_format", "png") or "png"
    mime_type = _OUTPUT_FORMAT_TO_MIME.get(output_format, "image/png")

    if event_type == "image_generation.partial_image":
        b64 = event_dict.get("b64_json", "")
        if b64:
            assistant_content.append(
                _create_content_item("output_media", url=b64, mime_type=mime_type, media_type="image")
            )

    elif event_type == "image_generation.completed":
        b64 = event_dict.get("b64_json", "")
        if b64:
            provider_metadata = {}
            for key in ("revised_prompt", "background", "size", "quality", "output_format"):
                val = event_dict.get(key)
                if val is not None:
                    provider_metadata[key] = val
            content_kwargs = dict(url=b64, mime_type=mime_type, media_type="image")
            if provider_metadata:
                content_kwargs["provider_metadata"] = provider_metadata
            assistant_content.append(_create_content_item("output_media", **content_kwargs))

    assistant_message = _build_assistant_message(assistant_content)
    return _build_prompt_blueprint(assistant_message, metadata)
