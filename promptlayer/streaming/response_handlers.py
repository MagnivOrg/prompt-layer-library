"""
Response handlers for different LLM providers

This module contains handlers that process streaming responses from various
LLM providers and return both the final response and prompt blueprint.
"""

import json
from typing import Any, AsyncIterable, List


def openai_stream_chat(results: list):
    """Process OpenAI streaming chat results and return response + blueprint"""
    from openai.types.chat import (
        ChatCompletion,
        ChatCompletionChunk,
        ChatCompletionMessage,
        ChatCompletionMessageToolCall,
    )
    from openai.types.chat.chat_completion import Choice
    from openai.types.chat.chat_completion_message_tool_call import Function

    chat_completion_chunks: List[ChatCompletionChunk] = results
    response: ChatCompletion = ChatCompletion(
        id="",
        object="chat.completion",
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(role="assistant"),
            )
        ],
        created=0,
        model="",
    )
    last_result = chat_completion_chunks[-1]
    response.id = last_result.id
    response.created = last_result.created
    response.model = last_result.model
    response.system_fingerprint = last_result.system_fingerprint
    response.usage = last_result.usage
    content = ""
    tool_calls: List[ChatCompletionMessageToolCall] = []

    for result in chat_completion_chunks:
        choices = result.choices
        if len(choices) == 0:
            continue
        if choices[0].delta.content:
            content = f"{content}{result.choices[0].delta.content}"

        delta = choices[0].delta
        if delta.tool_calls:
            last_tool_call = None
            if len(tool_calls) > 0:
                last_tool_call = tool_calls[-1]
            tool_call = delta.tool_calls[0]
            if not tool_call.function:
                continue
            if not last_tool_call or tool_call.id:
                tool_calls.append(
                    ChatCompletionMessageToolCall(
                        id=tool_call.id or "",
                        function=Function(
                            name=tool_call.function.name or "",
                            arguments=tool_call.function.arguments or "",
                        ),
                        type=tool_call.type or "function",
                    )
                )
                continue
            last_tool_call.function.name = f"{last_tool_call.function.name}{tool_call.function.name or ''}"
            last_tool_call.function.arguments = (
                f"{last_tool_call.function.arguments}{tool_call.function.arguments or ''}"
            )

    response.choices[0].message.content = content
    response.choices[0].message.tool_calls = tool_calls if tool_calls else None
    return response


async def aopenai_stream_chat(generator: AsyncIterable[Any]) -> Any:
    """Async version of openai_stream_chat"""
    from openai.types.chat import (
        ChatCompletion,
        ChatCompletionChunk,
        ChatCompletionMessage,
        ChatCompletionMessageToolCall,
    )
    from openai.types.chat.chat_completion import Choice
    from openai.types.chat.chat_completion_message_tool_call import Function

    chat_completion_chunks: List[ChatCompletionChunk] = []
    response: ChatCompletion = ChatCompletion(
        id="",
        object="chat.completion",
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(role="assistant"),
            )
        ],
        created=0,
        model="",
    )
    content = ""
    tool_calls: List[ChatCompletionMessageToolCall] = []

    async for result in generator:
        chat_completion_chunks.append(result)
        choices = result.choices
        if len(choices) == 0:
            continue
        if choices[0].delta.content:
            content = f"{content}{choices[0].delta.content}"

        delta = choices[0].delta
        if delta.tool_calls:
            last_tool_call = None
            if len(tool_calls) > 0:
                last_tool_call = tool_calls[-1]
            tool_call = delta.tool_calls[0]
            if not tool_call.function:
                continue
            if not last_tool_call or tool_call.id:
                tool_calls.append(
                    ChatCompletionMessageToolCall(
                        id=tool_call.id or "",
                        function=Function(
                            name=tool_call.function.name or "",
                            arguments=tool_call.function.arguments or "",
                        ),
                        type=tool_call.type or "function",
                    )
                )
                continue
            last_tool_call.function.name = f"{last_tool_call.function.name}{tool_call.function.name or ''}"
            last_tool_call.function.arguments = (
                f"{last_tool_call.function.arguments}{tool_call.function.arguments or ''}"
            )

    # After collecting all chunks, set the response attributes
    if chat_completion_chunks:
        last_result = chat_completion_chunks[-1]
        response.id = last_result.id
        response.created = last_result.created
        response.model = last_result.model
        response.system_fingerprint = getattr(last_result, "system_fingerprint", None)
        response.usage = last_result.usage

    response.choices[0].message.content = content
    response.choices[0].message.tool_calls = tool_calls if tool_calls else None
    return response


def _initialize_openai_response_data():
    """Initialize the response data structure for OpenAI responses"""
    return {
        "id": None,
        "object": "response",
        "created_at": None,
        "status": None,
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "max_output_tokens": None,
        "model": None,
        "output": [],
        "parallel_tool_calls": True,
        "previous_response_id": None,
        "reasoning": {"effort": None, "summary": None},
        "store": True,
        "temperature": 1,
        "text": {"format": {"type": "text"}},
        "tool_choice": "auto",
        "tools": [],
        "top_p": 1,
        "truncation": "disabled",
        "usage": None,
        "user": None,
        "metadata": {},
    }


def _process_openai_response_event(chunk_dict, response_data, current_items):
    """Process a single OpenAI response event and update the response data"""
    event_type = chunk_dict.get("type")
    has_reasoning = False

    if event_type == "response.created":
        response_info = chunk_dict.get("response", {})
        response_data["id"] = response_info.get("id")
        response_data["created_at"] = response_info.get("created_at")
        response_data["model"] = response_info.get("model")
        response_data["status"] = response_info.get("status")
        response_data["parallel_tool_calls"] = response_info.get("parallel_tool_calls", True)
        response_data["temperature"] = response_info.get("temperature", 1)
        response_data["tool_choice"] = response_info.get("tool_choice", "auto")
        response_data["tools"] = response_info.get("tools", [])
        response_data["top_p"] = response_info.get("top_p", 1)
        response_data["truncation"] = response_info.get("truncation", "disabled")
        response_data["max_output_tokens"] = response_info.get("max_output_tokens")
        response_data["previous_response_id"] = response_info.get("previous_response_id")
        response_data["store"] = response_info.get("store", True)
        response_data["user"] = response_info.get("user")
        response_data["metadata"] = response_info.get("metadata", {})

        text_config = response_info.get("text", {})
        if text_config:
            response_data["text"] = text_config

        reasoning = response_info.get("reasoning", {})
        if reasoning:
            response_data["reasoning"] = reasoning
            has_reasoning = True

    elif event_type == "response.in_progress":
        response_info = chunk_dict.get("response", {})
        response_data["status"] = response_info.get("status")

    elif event_type == "response.output_item.added":
        item = chunk_dict.get("item", {})
        item_id = item.get("id")
        item_type = item.get("type")

        if item_type == "reasoning":
            current_items[item_id] = {
                "type": "reasoning",
                "id": item_id,
                "summary": [],
                "status": item.get("status", "in_progress"),
            }
            has_reasoning = True

        elif item_type == "function_call":
            current_items[item_id] = {
                "type": "function_call",
                "id": item_id,
                "call_id": item.get("call_id"),
                "name": item.get("name"),
                "arguments": "",
                "status": item.get("status", "in_progress"),
            }

        elif item_type == "message":
            current_items[item_id] = {
                "type": "message",
                "id": item_id,
                "role": item.get("role", "assistant"),
                "content": [],
                "status": item.get("status", "in_progress"),
            }

    elif event_type == "response.reasoning_summary_part.added":
        item_id = chunk_dict.get("item_id")
        part = chunk_dict.get("part", {})

        if item_id in current_items and current_items[item_id]["type"] == "reasoning":
            summary_part = {"type": part.get("type", "summary_text"), "text": part.get("text", "")}
            current_items[item_id]["summary"].append(summary_part)

    elif event_type == "response.reasoning_summary_text.delta":
        item_id = chunk_dict.get("item_id")
        delta = chunk_dict.get("delta", "")
        summary_index = chunk_dict.get("summary_index", 0)

        if item_id in current_items and current_items[item_id]["type"] == "reasoning":
            while len(current_items[item_id]["summary"]) <= summary_index:
                current_items[item_id]["summary"].append({"type": "summary_text", "text": ""})

            current_items[item_id]["summary"][summary_index]["text"] += delta

    elif event_type == "response.reasoning_summary_text.done":
        item_id = chunk_dict.get("item_id")
        final_text = chunk_dict.get("text", "")
        summary_index = chunk_dict.get("summary_index", 0)

        if item_id in current_items and current_items[item_id]["type"] == "reasoning":
            while len(current_items[item_id]["summary"]) <= summary_index:
                current_items[item_id]["summary"].append({"type": "summary_text", "text": ""})

            current_items[item_id]["summary"][summary_index]["text"] = final_text

    elif event_type == "response.reasoning_summary_part.done":
        item_id = chunk_dict.get("item_id")
        part = chunk_dict.get("part", {})

        if item_id in current_items and current_items[item_id]["type"] == "reasoning":
            summary_index = chunk_dict.get("summary_index", 0)
            if summary_index < len(current_items[item_id]["summary"]):
                current_items[item_id]["summary"][summary_index] = {
                    "type": part.get("type", "summary_text"),
                    "text": part.get("text", ""),
                }

    elif event_type == "response.function_call_arguments.delta":
        item_id = chunk_dict.get("item_id")
        delta = chunk_dict.get("delta", "")

        if item_id in current_items:
            current_items[item_id]["arguments"] += delta

    elif event_type == "response.function_call_arguments.done":
        item_id = chunk_dict.get("item_id")
        final_arguments = chunk_dict.get("arguments", "")

        if item_id in current_items:
            current_items[item_id]["arguments"] = final_arguments

    elif event_type == "response.content_part.added":
        part = chunk_dict.get("part", {})

        message_item = None
        for item in current_items.values():
            if item.get("type") == "message":
                message_item = item
                break

        if message_item:
            content_part = {
                "type": part.get("type", "output_text"),
                "text": part.get("text", ""),
                "annotations": part.get("annotations", []),
            }
            message_item["content"].append(content_part)

    elif event_type == "response.output_text.delta":
        delta_text = chunk_dict.get("delta", "")

        for item in current_items.values():
            if item.get("type") == "message" and item.get("content"):
                if item["content"] and item["content"][-1].get("type") == "output_text":
                    item["content"][-1]["text"] += delta_text
                break

    elif event_type == "response.output_text.done":
        final_text = chunk_dict.get("text", "")

        for item in current_items.values():
            if item.get("type") == "message" and item.get("content"):
                if item["content"] and item["content"][-1].get("type") == "output_text":
                    item["content"][-1]["text"] = final_text
                break

    elif event_type == "response.output_item.done":
        item = chunk_dict.get("item", {})
        item_id = item.get("id")

        if item_id in current_items:
            current_items[item_id]["status"] = item.get("status", "completed")

            if item.get("type") == "reasoning":
                current_items[item_id].update({"summary": item.get("summary", current_items[item_id]["summary"])})
            elif item.get("type") == "function_call":
                current_items[item_id].update(
                    {
                        "arguments": item.get("arguments", current_items[item_id]["arguments"]),
                        "call_id": item.get("call_id", current_items[item_id]["call_id"]),
                        "name": item.get("name", current_items[item_id]["name"]),
                    }
                )
            elif item.get("type") == "message":
                current_items[item_id].update(
                    {
                        "content": item.get("content", current_items[item_id]["content"]),
                        "role": item.get("role", current_items[item_id]["role"]),
                    }
                )

            response_data["output"].append(current_items[item_id])

    elif event_type == "response.completed":
        response_info = chunk_dict.get("response", {})
        response_data["status"] = response_info.get("status", "completed")
        response_data["usage"] = response_info.get("usage")
        response_data["output"] = response_info.get("output", response_data["output"])

        if response_info.get("reasoning"):
            response_data["reasoning"] = response_info["reasoning"]

    return has_reasoning


def openai_responses_stream_chat(results: list):
    """Process OpenAI Responses streaming chat results and return response"""
    from openai.types.responses import Response

    response_data = _initialize_openai_response_data()
    current_items = {}

    for chunk in results:
        chunk_dict = chunk.model_dump()
        _process_openai_response_event(chunk_dict, response_data, current_items)

    return Response(**response_data)


async def aopenai_responses_stream_chat(generator: AsyncIterable[Any]) -> Any:
    """Async version of openai_responses_stream_chat"""
    from openai.types.responses import Response

    response_data = _initialize_openai_response_data()
    current_items = {}

    async for chunk in generator:
        chunk_dict = chunk.model_dump()
        _process_openai_response_event(chunk_dict, response_data, current_items)

    return Response(**response_data)


def anthropic_stream_message(results: list):
    """Process Anthropic streaming message results and return response + blueprint"""
    from anthropic.types import Message, MessageStreamEvent, Usage

    from promptlayer.utils import build_anthropic_content_blocks

    message_stream_events: List[MessageStreamEvent] = results
    response: Message = Message(
        id="",
        model="",
        content=[],
        role="assistant",
        type="message",
        stop_reason="stop_sequence",
        stop_sequence=None,
        usage=Usage(input_tokens=0, output_tokens=0),
    )

    for event in message_stream_events:
        if event.type == "message_start":
            response = event.message
            break

    content_blocks, usage, stop_reason = build_anthropic_content_blocks(message_stream_events)
    response.content = content_blocks
    if usage:
        response.usage.output_tokens = usage.output_tokens
    if stop_reason:
        response.stop_reason = stop_reason

    return response


async def aanthropic_stream_message(generator: AsyncIterable[Any]) -> Any:
    """Async version of anthropic_stream_message"""
    from anthropic.types import Message, MessageStreamEvent, Usage

    from promptlayer.utils import build_anthropic_content_blocks

    message_stream_events: List[MessageStreamEvent] = []
    response: Message = Message(
        id="",
        model="",
        content=[],
        role="assistant",
        type="message",
        stop_reason="stop_sequence",
        stop_sequence=None,
        usage=Usage(input_tokens=0, output_tokens=0),
    )

    async for event in generator:
        if event.type == "message_start":
            response = event.message
        message_stream_events.append(event)

    content_blocks, usage, stop_reason = build_anthropic_content_blocks(message_stream_events)
    response.content = content_blocks
    if usage:
        response.usage.output_tokens = usage.output_tokens
    if stop_reason:
        response.stop_reason = stop_reason

    return response


def openai_stream_completion(results: list):
    from openai.types.completion import Completion, CompletionChoice

    completions: List[Completion] = results
    last_chunk = completions[-1]
    response = Completion(
        id=last_chunk.id,
        created=last_chunk.created,
        model=last_chunk.model,
        object="text_completion",
        choices=[CompletionChoice(finish_reason="stop", index=0, text="")],
    )
    text = ""
    for completion in completions:
        usage = completion.usage
        system_fingerprint = completion.system_fingerprint
        if len(completion.choices) > 0 and completion.choices[0].text:
            text = f"{text}{completion.choices[0].text}"
        if usage:
            response.usage = usage
        if system_fingerprint:
            response.system_fingerprint = system_fingerprint
    response.choices[0].text = text
    return response


async def aopenai_stream_completion(generator: AsyncIterable[Any]) -> Any:
    from openai.types.completion import Completion, CompletionChoice

    completions: List[Completion] = []
    text = ""
    response = Completion(
        id="",
        created=0,
        model="",
        object="text_completion",
        choices=[CompletionChoice(finish_reason="stop", index=0, text="")],
    )

    async for completion in generator:
        completions.append(completion)
        usage = completion.usage
        system_fingerprint = getattr(completion, "system_fingerprint", None)
        if len(completion.choices) > 0 and completion.choices[0].text:
            text = f"{text}{completion.choices[0].text}"
        if usage:
            response.usage = usage
        if system_fingerprint:
            response.system_fingerprint = system_fingerprint

    # After collecting all completions, set the response attributes
    if completions:
        last_chunk = completions[-1]
        response.id = last_chunk.id
        response.created = last_chunk.created
        response.model = last_chunk.model

    response.choices[0].text = text
    return response


def anthropic_stream_completion(results: list):
    from anthropic.types import Completion

    completions: List[Completion] = results
    last_chunk = completions[-1]
    response = Completion(
        id=last_chunk.id,
        completion="",
        model=last_chunk.model,
        stop_reason="stop",
        type="completion",
    )

    text = ""
    for completion in completions:
        text = f"{text}{completion.completion}"
    response.completion = text
    return response


async def aanthropic_stream_completion(generator: AsyncIterable[Any]) -> Any:
    from anthropic.types import Completion

    completions: List[Completion] = []
    text = ""
    response = Completion(
        id="",
        completion="",
        model="",
        stop_reason="stop",
        type="completion",
    )

    async for completion in generator:
        completions.append(completion)
        text = f"{text}{completion.completion}"

    # After collecting all completions, set the response attributes
    if completions:
        last_chunk = completions[-1]
        response.id = last_chunk.id
        response.model = last_chunk.model

    response.completion = text
    return response


def _build_google_response_from_parts(thought_content: str, regular_content: str, function_calls: list, last_result):
    """Helper function to build Google response with thought, regular, and function call parts."""
    from google.genai.chats import Part

    response = last_result.model_copy()
    final_parts = []

    if thought_content:
        thought_part = Part(text=thought_content, thought=True)
        final_parts.append(thought_part)

    if regular_content:
        text_part = Part(text=regular_content, thought=None)
        final_parts.append(text_part)

    for function_call in function_calls:
        function_part = Part(function_call=function_call, thought=None)
        final_parts.append(function_part)

    if final_parts:
        response.candidates[0].content.parts = final_parts

    return response


async def amap_google_stream_response(generator: AsyncIterable[Any]):
    from google.genai.chats import GenerateContentResponse

    response = GenerateContentResponse()

    thought_content = ""
    regular_content = ""
    function_calls = []
    last_result = None

    async for result in generator:
        last_result = result
        if result.candidates and result.candidates[0].content.parts:
            for part in result.candidates[0].content.parts:
                if hasattr(part, "text") and part.text:
                    if hasattr(part, "thought") and part.thought:
                        thought_content = f"{thought_content}{part.text}"
                    else:
                        regular_content = f"{regular_content}{part.text}"
                elif hasattr(part, "function_call") and part.function_call:
                    function_calls.append(part.function_call)

    if not last_result:
        return response

    return _build_google_response_from_parts(thought_content, regular_content, function_calls, last_result)


async def agoogle_stream_chat(generator: AsyncIterable[Any]):
    return await amap_google_stream_response(generator)


async def agoogle_stream_completion(generator: AsyncIterable[Any]):
    return await amap_google_stream_response(generator)


def map_google_stream_response(results: list):
    from google.genai.chats import GenerateContentResponse

    response = GenerateContentResponse()
    if not results:
        return response
    results: List[GenerateContentResponse] = results

    thought_content = ""
    regular_content = ""
    function_calls = []

    for result in results:
        if result.candidates and result.candidates[0].content.parts:
            for part in result.candidates[0].content.parts:
                if hasattr(part, "text") and part.text:
                    if hasattr(part, "thought") and part.thought:
                        thought_content = f"{thought_content}{part.text}"
                    else:
                        regular_content = f"{regular_content}{part.text}"
                elif hasattr(part, "function_call") and part.function_call:
                    function_calls.append(part.function_call)

    return _build_google_response_from_parts(thought_content, regular_content, function_calls, results[-1])


def google_stream_chat(results: list):
    return map_google_stream_response(results)


def google_stream_completion(results: list):
    return map_google_stream_response(results)


def mistral_stream_chat(results: list):
    from openai.types.chat import ChatCompletion, ChatCompletionMessage, ChatCompletionMessageToolCall
    from openai.types.chat.chat_completion import Choice
    from openai.types.chat.chat_completion_message_tool_call import Function

    last_result = results[-1]
    response = ChatCompletion(
        id=last_result.data.id,
        object="chat.completion",
        choices=[
            Choice(
                finish_reason=last_result.data.choices[0].finish_reason or "stop",
                index=0,
                message=ChatCompletionMessage(role="assistant"),
            )
        ],
        created=last_result.data.created,
        model=last_result.data.model,
    )

    content = ""
    tool_calls = None

    for result in results:
        choices = result.data.choices
        if len(choices) == 0:
            continue

        delta = choices[0].delta
        if delta.content is not None:
            content = f"{content}{delta.content}"

        if delta.tool_calls:
            tool_calls = tool_calls or []
            for tool_call in delta.tool_calls:
                if len(tool_calls) == 0 or tool_call.id:
                    tool_calls.append(
                        ChatCompletionMessageToolCall(
                            id=tool_call.id or "",
                            function=Function(
                                name=tool_call.function.name,
                                arguments=tool_call.function.arguments,
                            ),
                            type="function",
                        )
                    )
                else:
                    last_tool_call = tool_calls[-1]
                    if tool_call.function.name:
                        last_tool_call.function.name = f"{last_tool_call.function.name}{tool_call.function.name}"
                    if tool_call.function.arguments:
                        last_tool_call.function.arguments = (
                            f"{last_tool_call.function.arguments}{tool_call.function.arguments}"
                        )

    response.choices[0].message.content = content
    response.choices[0].message.tool_calls = tool_calls
    response.usage = last_result.data.usage
    return response


async def amistral_stream_chat(generator: AsyncIterable[Any]) -> Any:
    from openai.types.chat import ChatCompletion, ChatCompletionMessage, ChatCompletionMessageToolCall
    from openai.types.chat.chat_completion import Choice
    from openai.types.chat.chat_completion_message_tool_call import Function

    completion_chunks = []
    response = ChatCompletion(
        id="",
        object="chat.completion",
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(role="assistant"),
            )
        ],
        created=0,
        model="",
    )
    content = ""
    tool_calls = None

    async for result in generator:
        completion_chunks.append(result)
        choices = result.data.choices
        if len(choices) == 0:
            continue
        delta = choices[0].delta
        if delta.content is not None:
            content = f"{content}{delta.content}"

        if delta.tool_calls:
            tool_calls = tool_calls or []
            for tool_call in delta.tool_calls:
                if len(tool_calls) == 0 or tool_call.id:
                    tool_calls.append(
                        ChatCompletionMessageToolCall(
                            id=tool_call.id or "",
                            function=Function(
                                name=tool_call.function.name,
                                arguments=tool_call.function.arguments,
                            ),
                            type="function",
                        )
                    )
                else:
                    last_tool_call = tool_calls[-1]
                    if tool_call.function.name:
                        last_tool_call.function.name = f"{last_tool_call.function.name}{tool_call.function.name}"
                    if tool_call.function.arguments:
                        last_tool_call.function.arguments = (
                            f"{last_tool_call.function.arguments}{tool_call.function.arguments}"
                        )

    if completion_chunks:
        last_result = completion_chunks[-1]
        response.id = last_result.data.id
        response.created = last_result.data.created
        response.model = last_result.data.model
        response.usage = last_result.data.usage

    response.choices[0].message.content = content
    response.choices[0].message.tool_calls = tool_calls
    return response


def bedrock_stream_message(results: list):
    """Process Amazon Bedrock streaming message results and return response + blueprint"""

    response = {"ResponseMetadata": {}, "output": {"message": {}}, "stopReason": "end_turn", "metrics": {}, "usage": {}}

    content_blocks = []
    current_tool_call = None
    current_tool_input = ""
    current_text = ""
    current_signature = ""
    current_thinking = ""

    for event in results:
        if "contentBlockStart" in event:
            content_block = event["contentBlockStart"]
            if "start" in content_block and "toolUse" in content_block["start"]:
                tool_use = content_block["start"]["toolUse"]
                current_tool_call = {"toolUse": {"toolUseId": tool_use["toolUseId"], "name": tool_use["name"]}}
                current_tool_input = ""

        elif "contentBlockDelta" in event:
            delta = event["contentBlockDelta"]["delta"]
            if "text" in delta:
                current_text += delta["text"]
            elif "reasoningContent" in delta:
                reasoning_content = delta["reasoningContent"]
                if "text" in reasoning_content:
                    current_thinking += reasoning_content["text"]
                elif "signature" in reasoning_content:
                    current_signature += reasoning_content["signature"]
            elif "toolUse" in delta:
                if "input" in delta["toolUse"]:
                    input_chunk = delta["toolUse"]["input"]
                    current_tool_input += input_chunk
                    if not input_chunk.strip():
                        continue

        elif "contentBlockStop" in event:
            if current_tool_call and current_tool_input:
                try:
                    current_tool_call["toolUse"]["input"] = json.loads(current_tool_input)
                except json.JSONDecodeError:
                    current_tool_call["toolUse"]["input"] = {}
                content_blocks.append(current_tool_call)
                current_tool_call = None
                current_tool_input = ""
            elif current_text:
                content_blocks.append({"text": current_text})
                current_text = ""
            elif current_thinking and current_signature:
                content_blocks.append(
                    {
                        "reasoningContent": {
                            "reasoningText": {"text": current_thinking, "signature": current_signature},
                        }
                    }
                )
                current_thinking = ""
                current_signature = ""

        elif "messageStop" in event:
            response["stopReason"] = event["messageStop"]["stopReason"]

        elif "metadata" in event:
            metadata = event["metadata"]
            response["usage"] = metadata.get("usage", {})
            response["metrics"] = metadata.get("metrics", {})

    response["output"]["message"] = {"role": "assistant", "content": content_blocks}
    return response


async def abedrock_stream_message(generator: AsyncIterable[Any]) -> Any:
    """Async version of bedrock_stream_message"""

    response = {"ResponseMetadata": {}, "output": {"message": {}}, "stopReason": "end_turn", "metrics": {}, "usage": {}}

    content_blocks = []
    current_tool_call = None
    current_tool_input = ""
    current_text = ""
    current_signature = ""
    current_thinking = ""

    async for event in generator:
        if "contentBlockStart" in event:
            content_block = event["contentBlockStart"]
            if "start" in content_block and "toolUse" in content_block["start"]:
                tool_use = content_block["start"]["toolUse"]
                current_tool_call = {"toolUse": {"toolUseId": tool_use["toolUseId"], "name": tool_use["name"]}}
                current_tool_input = ""

        elif "contentBlockDelta" in event:
            delta = event["contentBlockDelta"]["delta"]
            if "text" in delta:
                current_text += delta["text"]
            elif "reasoningContent" in delta:
                reasoning_content = delta["reasoningContent"]
                if "text" in reasoning_content:
                    current_thinking += reasoning_content["text"]
                elif "signature" in reasoning_content:
                    current_signature += reasoning_content["signature"]
            elif "toolUse" in delta:
                if "input" in delta["toolUse"]:
                    input_chunk = delta["toolUse"]["input"]
                    current_tool_input += input_chunk
                    if not input_chunk.strip():
                        continue

        elif "contentBlockStop" in event:
            if current_tool_call and current_tool_input:
                try:
                    current_tool_call["toolUse"]["input"] = json.loads(current_tool_input)
                except json.JSONDecodeError:
                    current_tool_call["toolUse"]["input"] = {}
                content_blocks.append(current_tool_call)
                current_tool_call = None
                current_tool_input = ""
            elif current_text:
                content_blocks.append({"text": current_text})
                current_text = ""
            elif current_thinking and current_signature:
                content_blocks.append(
                    {
                        "reasoningContent": {
                            "reasoningText": {"text": current_thinking, "signature": current_signature},
                        }
                    }
                )
                current_thinking = ""
                current_signature = ""

        elif "messageStop" in event:
            response["stopReason"] = event["messageStop"]["stopReason"]

        elif "metadata" in event:
            metadata = event["metadata"]
            response["usage"] = metadata.get("usage", {})
            response["metrics"] = metadata.get("metrics", {})

    response["output"]["message"] = {"role": "assistant", "content": content_blocks}
    return response
