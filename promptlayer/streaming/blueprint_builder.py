"""
Blueprint builders for various LLM providers

This module contains functions to build prompt blueprints from LLM responses
and streaming events for different providers (OpenAI, Anthropic, etc.)
"""

from typing import Any, Dict, List, Optional


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
            assistant_content.append(_create_content_item("text", item_id=item_id, text="[Message started]"))

    elif event_type == "response.content_part.added":
        item_id = event_dict.get("item_id")
        part = event_dict.get("part", {})
        part_type = part.get("type", "output_text")

        if part_type == "output_text":
            text = part.get("text", "")
            assistant_content.append(
                _create_content_item("text", item_id=item_id, text=text if text else "[Content part added]")
            )

    elif event_type == "response.output_text.delta":
        item_id = event_dict.get("item_id")
        delta_text = event_dict.get("delta", "")
        if delta_text:
            assistant_content.append(_create_content_item("text", item_id=item_id, text=delta_text))

    elif event_type == "response.output_text.done":
        item_id = event_dict.get("item_id")
        final_text = event_dict.get("text", "")
        if final_text:
            assistant_content.append(_create_content_item("text", item_id=item_id, text=final_text))

    elif event_type == "response.output_item.done":
        item = event_dict.get("item", {})
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
                _create_tool_call(
                    item.get("call_id", ""), item.get("name", ""), item.get("arguments", ""), tool_id=item_id
                )
            )

        elif item_type == "message":
            content = item.get("content", [])
            for content_part in content:
                if content_part.get("type") == "output_text":
                    text = content_part.get("text", "")
                    if text:
                        assistant_content.append(_create_content_item("text", item_id=item_id, text=text))

    assistant_message = _build_assistant_message(assistant_content, tool_calls or None)
    return _build_prompt_blueprint(assistant_message, metadata)


def build_prompt_blueprint_from_anthropic_event(event, metadata):
    """Build a prompt blueprint from an Anthropic stream event"""

    assistant_content = []
    tool_calls = []

    if hasattr(event, "type"):
        if event.type == "content_block_start" and hasattr(event, "content_block"):
            if event.content_block.type == "thinking":
                assistant_content.append(_create_content_item("thinking", thinking="", signature=None))
            elif event.content_block.type == "text":
                assistant_content.append(_create_content_item("text", text=""))
            elif event.content_block.type == "tool_use":
                tool_calls.append(
                    _create_tool_call(
                        getattr(event.content_block, "id", ""),
                        getattr(event.content_block, "name", ""),
                        getattr(event.content_block, "input", ""),
                    )
                )
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

    assistant_message = _build_assistant_message(assistant_content, tool_calls or None)
    return _build_prompt_blueprint(assistant_message, metadata)


def build_prompt_blueprint_from_google_event(event, metadata):
    """
    Build a prompt blueprint from a Google (Gemini) streaming event (raw dict or GenerateContentResponse).
    """
    assistant_content = []
    tool_calls = []
    candidate = event.candidates[0]

    if candidate and hasattr(candidate, "content") and candidate.content and hasattr(candidate.content, "parts"):
        for part in candidate.content.parts:
            # "thought" is a boolean attribute on Part for Gemini
            if hasattr(part, "thought") and part.thought is True:
                assistant_content.append(
                    _create_content_item("thinking", thinking=getattr(part, "text", ""), signature=None)
                )
            elif hasattr(part, "text") and part.text:
                assistant_content.append(_create_content_item("text", text=part.text))
            elif hasattr(part, "function_call"):
                tool_calls.append(
                    _create_tool_call(
                        getattr(part.function_call, "id", ""),
                        getattr(part.function_call, "name", ""),
                        getattr(part.function_call, "args", {}),
                    )
                )

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
