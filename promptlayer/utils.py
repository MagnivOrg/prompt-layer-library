import asyncio
import contextvars
import datetime
import functools
import json
import os
import sys
import types
from copy import deepcopy
from enum import Enum
from typing import (
    Any,
    AsyncGenerator,
    AsyncIterable,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Union,
)

import httpx
import requests
from ably import AblyRealtime
from ably.types.message import Message
from opentelemetry import context, trace

from promptlayer.types import RequestLog
from promptlayer.types.prompt_template import (
    GetPromptTemplate,
    GetPromptTemplateResponse,
    ListPromptTemplateResponse,
    PublishPromptTemplate,
    PublishPromptTemplateResponse,
)

URL_API_PROMPTLAYER = os.environ.setdefault(
    "URL_API_PROMPTLAYER", "https://api.promptlayer.com"
)


async def arun_workflow_request(
    *,
    workflow_name: str,
    input_variables: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
    workflow_label_name: Optional[str] = None,
    workflow_version_number: Optional[int] = None,
    api_key: str,
    return_all_outputs: Optional[bool] = False,
    timeout: Optional[int] = 120,
) -> Dict[str, Any]:
    payload = {
        "input_variables": input_variables,
        "metadata": metadata,
        "workflow_label_name": workflow_label_name,
        "workflow_version_number": workflow_version_number,
        "return_all_outputs": return_all_outputs,
    }

    url = f"{URL_API_PROMPTLAYER}/workflows/{workflow_name}/run"
    headers = {"X-API-KEY": api_key}

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, headers=headers)
            if response.status_code != 201:
                raise_on_bad_response(
                    response,
                    "PromptLayer had the following error while running your workflow",
                )

            result = response.json()
            warning = result.get("warning")
            if warning:
                print(f"WARNING: {warning}")

    except Exception as e:
        error_message = f"Failed to run workflow: {str(e)}"
        print(error_message)
        raise Exception(error_message)

    execution_id = result.get("workflow_version_execution_id")
    if not execution_id:
        raise Exception("No execution ID returned from workflow run")

    channel_name = f"workflow_updates:{execution_id}"

    # Get WebSocket token
    try:
        async with httpx.AsyncClient() as client:
            ws_response = await client.post(
                f"{URL_API_PROMPTLAYER}/ws-token-request-library",
                headers=headers,
                params={"capability": channel_name},
            )
            if ws_response.status_code != 201:
                raise_on_bad_response(
                    ws_response,
                    "PromptLayer had the following error while getting WebSocket token",
                )
            token_details = ws_response.json()["token_details"]
    except Exception as e:
        error_message = f"Failed to get WebSocket token: {e}"
        print(error_message)
        raise Exception(error_message)

    # Initialize Ably client
    ably_client = AblyRealtime(token=token_details["token"])

    # Subscribe to the channel named after the execution ID
    channel = ably_client.channels.get(channel_name)

    final_output = {}
    message_received_event = asyncio.Event()

    async def message_listener(message: Message):
        if message.name == "set_workflow_node_output":
            data = json.loads(message.data)
            if data.get("status") == "workflow_complete":
                final_output.update(data.get("final_output", {}))
                message_received_event.set()

    # Subscribe to the channel
    await channel.subscribe("set_workflow_node_output", message_listener)

    # Wait for the message or timeout
    try:
        await asyncio.wait_for(message_received_event.wait(), timeout)
    except asyncio.TimeoutError:
        channel.unsubscribe("set_workflow_node_output", message_listener)
        await ably_client.close()
        raise Exception("Workflow execution did not complete properly")

    # Unsubscribe from the channel and close the client
    channel.unsubscribe("set_workflow_node_output", message_listener)
    await ably_client.close()

    return final_output


def promptlayer_api_handler(
    function_name,
    provider_type,
    args,
    kwargs,
    tags,
    response,
    request_start_time,
    request_end_time,
    api_key,
    return_pl_id=False,
    llm_request_span_id=None,
):
    if (
        isinstance(response, types.GeneratorType)
        or isinstance(response, types.AsyncGeneratorType)
        or type(response).__name__
        in [
            "Stream",
            "AsyncStream",
            "AsyncMessageStreamManager",
            "MessageStreamManager",
        ]
    ):
        return GeneratorProxy(
            generator=response,
            api_request_arguments={
                "function_name": function_name,
                "provider_type": provider_type,
                "args": args,
                "kwargs": kwargs,
                "tags": tags,
                "request_start_time": request_start_time,
                "request_end_time": request_end_time,
                "return_pl_id": return_pl_id,
                "llm_request_span_id": llm_request_span_id,
            },
            api_key=api_key,
        )
    else:
        request_id = promptlayer_api_request(
            function_name=function_name,
            provider_type=provider_type,
            args=args,
            kwargs=kwargs,
            tags=tags,
            response=response,
            request_start_time=request_start_time,
            request_end_time=request_end_time,
            api_key=api_key,
            return_pl_id=return_pl_id,
            llm_request_span_id=llm_request_span_id,
        )
        if return_pl_id:
            return response, request_id
        return response


async def promptlayer_api_handler_async(
    function_name,
    provider_type,
    args,
    kwargs,
    tags,
    response,
    request_start_time,
    request_end_time,
    api_key,
    return_pl_id=False,
    llm_request_span_id=None,
):
    return await run_in_thread_async(
        None,
        promptlayer_api_handler,
        function_name,
        provider_type,
        args,
        kwargs,
        tags,
        response,
        request_start_time,
        request_end_time,
        api_key,
        return_pl_id=return_pl_id,
        llm_request_span_id=llm_request_span_id,
    )


def convert_native_object_to_dict(native_object):
    if isinstance(native_object, dict):
        return {k: convert_native_object_to_dict(v) for k, v in native_object.items()}
    if isinstance(native_object, list):
        return [convert_native_object_to_dict(v) for v in native_object]
    if isinstance(native_object, Enum):
        return native_object.value
    if hasattr(native_object, "__dict__"):
        return {
            k: convert_native_object_to_dict(v)
            for k, v in native_object.__dict__.items()
        }
    return native_object


def promptlayer_api_request(
    *,
    function_name,
    provider_type,
    args,
    kwargs,
    tags,
    response,
    request_start_time,
    request_end_time,
    api_key,
    return_pl_id=False,
    metadata=None,
    llm_request_span_id=None,
):
    if isinstance(response, dict) and hasattr(response, "to_dict_recursive"):
        response = response.to_dict_recursive()
    request_response = None
    if hasattr(
        response, "dict"
    ):  # added this for anthropic 3.0 changes, they return a completion object
        response = response.dict()
    try:
        request_response = requests.post(
            f"{URL_API_PROMPTLAYER}/track-request",
            json={
                "function_name": function_name,
                "provider_type": provider_type,
                "args": args,
                "kwargs": convert_native_object_to_dict(kwargs),
                "tags": tags,
                "request_response": response,
                "request_start_time": request_start_time,
                "request_end_time": request_end_time,
                "metadata": metadata,
                "api_key": api_key,
                "span_id": llm_request_span_id,
            },
        )
        if not hasattr(request_response, "status_code"):
            warn_on_bad_response(
                request_response,
                "WARNING: While logging your request PromptLayer had the following issue",
            )
        elif request_response.status_code != 200:
            warn_on_bad_response(
                request_response,
                "WARNING: While logging your request PromptLayer had the following error",
            )
    except Exception as e:
        print(
            f"WARNING: While logging your request PromptLayer had the following error: {e}",
            file=sys.stderr,
        )
    if request_response is not None and return_pl_id:
        return request_response.json().get("request_id")


def track_request(**body):
    try:
        response = requests.post(
            f"{URL_API_PROMPTLAYER}/track-request",
            json=body,
        )
        if response.status_code != 200:
            warn_on_bad_response(
                response,
                f"PromptLayer had the following error while tracking your request: {response.text}",
            )
        return response.json()
    except requests.exceptions.RequestException as e:
        print(
            f"WARNING: While logging your request PromptLayer had the following error: {e}",
            file=sys.stderr,
        )
        return {}


async def atrack_request(**body: Any) -> Dict[str, Any]:
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{URL_API_PROMPTLAYER}/track-request",
                json=body,
            )
        if response.status_code != 200:
            warn_on_bad_response(
                response,
                f"PromptLayer had the following error while tracking your request: {response.text}",
            )
        return response.json()
    except httpx.RequestError as e:
        print(
            f"WARNING: While logging your request PromptLayer had the following error: {e}",
            file=sys.stderr,
        )
        return {}


def promptlayer_api_request_async(
    function_name,
    provider_type,
    args,
    kwargs,
    tags,
    response,
    request_start_time,
    request_end_time,
    api_key,
    return_pl_id=False,
):
    return run_in_thread_async(
        None,
        promptlayer_api_request,
        function_name=function_name,
        provider_type=provider_type,
        args=args,
        kwargs=kwargs,
        tags=tags,
        response=response,
        request_start_time=request_start_time,
        request_end_time=request_end_time,
        api_key=api_key,
        return_pl_id=return_pl_id,
    )


def promptlayer_get_prompt(
    prompt_name, api_key, version: int = None, label: str = None
):
    """
    Get a prompt from the PromptLayer library
    version: version of the prompt to get, None for latest
    label: The specific label of a prompt you want to get. Setting this will supercede version
    """
    try:
        request_response = requests.get(
            f"{URL_API_PROMPTLAYER}/library-get-prompt-template",
            headers={"X-API-KEY": api_key},
            params={"prompt_name": prompt_name, "version": version, "label": label},
        )
    except Exception as e:
        raise Exception(
            f"PromptLayer had the following error while getting your prompt: {e}"
        )
    if request_response.status_code != 200:
        raise_on_bad_response(
            request_response,
            "PromptLayer had the following error while getting your prompt",
        )

    return request_response.json()


def promptlayer_publish_prompt(
    prompt_name, prompt_template, commit_message, tags, api_key, metadata=None
):
    try:
        request_response = requests.post(
            f"{URL_API_PROMPTLAYER}/library-publish-prompt-template",
            json={
                "prompt_name": prompt_name,
                "prompt_template": prompt_template,
                "commit_message": commit_message,
                "tags": tags,
                "api_key": api_key,
                "metadata": metadata,
            },
        )
    except Exception as e:
        raise Exception(
            f"PromptLayer had the following error while publishing your prompt: {e}"
        )
    if request_response.status_code != 200:
        raise_on_bad_response(
            request_response,
            "PromptLayer had the following error while publishing your prompt",
        )
    return True


def promptlayer_track_prompt(
    request_id, prompt_name, input_variables, api_key, version, label
):
    try:
        request_response = requests.post(
            f"{URL_API_PROMPTLAYER}/library-track-prompt",
            json={
                "request_id": request_id,
                "prompt_name": prompt_name,
                "prompt_input_variables": input_variables,
                "api_key": api_key,
                "version": version,
                "label": label,
            },
        )
        if request_response.status_code != 200:
            warn_on_bad_response(
                request_response,
                "WARNING: While tracking your prompt PromptLayer had the following error",
            )
            return False
    except Exception as e:
        print(
            f"WARNING: While tracking your prompt PromptLayer had the following error: {e}",
            file=sys.stderr,
        )
        return False
    return True


async def apromptlayer_track_prompt(
    request_id: str,
    prompt_name: str,
    input_variables: Dict[str, Any],
    api_key: Optional[str] = None,
    version: Optional[int] = None,
    label: Optional[str] = None,
) -> bool:
    url = f"{URL_API_PROMPTLAYER}/library-track-prompt"
    payload = {
        "request_id": request_id,
        "prompt_name": prompt_name,
        "prompt_input_variables": input_variables,
        "api_key": api_key,
        "version": version,
        "label": label,
    }
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload)
        if response.status_code != 200:
            warn_on_bad_response(
                response,
                "WARNING: While tracking your prompt, PromptLayer had the following error",
            )
            return False
    except httpx.RequestError as e:
        print(
            f"WARNING: While tracking your prompt PromptLayer had the following error: {e}",
            file=sys.stderr,
        )
        return False

    return True


def promptlayer_track_metadata(request_id, metadata, api_key):
    try:
        request_response = requests.post(
            f"{URL_API_PROMPTLAYER}/library-track-metadata",
            json={
                "request_id": request_id,
                "metadata": metadata,
                "api_key": api_key,
            },
        )
        if request_response.status_code != 200:
            warn_on_bad_response(
                request_response,
                "WARNING: While tracking your metadata PromptLayer had the following error",
            )
            return False
    except Exception as e:
        print(
            f"WARNING: While tracking your metadata PromptLayer had the following error: {e}",
            file=sys.stderr,
        )
        return False
    return True


async def apromptlayer_track_metadata(
    request_id: str, metadata: Dict[str, Any], api_key: Optional[str] = None
) -> bool:
    url = f"{URL_API_PROMPTLAYER}/library-track-metadata"
    payload = {
        "request_id": request_id,
        "metadata": metadata,
        "api_key": api_key,
    }
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload)
        if response.status_code != 200:
            warn_on_bad_response(
                response,
                "WARNING: While tracking your metadata, PromptLayer had the following error",
            )
            return False
    except httpx.RequestError as e:
        print(
            f"WARNING: While tracking your metadata PromptLayer had the following error: {e}",
            file=sys.stderr,
        )
        return False

    return True


def promptlayer_track_score(request_id, score, score_name, api_key):
    try:
        data = {"request_id": request_id, "score": score, "api_key": api_key}
        if score_name is not None:
            data["name"] = score_name
        request_response = requests.post(
            f"{URL_API_PROMPTLAYER}/library-track-score",
            json=data,
        )
        if request_response.status_code != 200:
            warn_on_bad_response(
                request_response,
                "WARNING: While tracking your score PromptLayer had the following error",
            )
            return False
    except Exception as e:
        print(
            f"WARNING: While tracking your score PromptLayer had the following error: {e}",
            file=sys.stderr,
        )
        return False
    return True


async def apromptlayer_track_score(
    request_id: str,
    score: float,
    score_name: Optional[str],
    api_key: Optional[str] = None,
) -> bool:
    url = f"{URL_API_PROMPTLAYER}/library-track-score"
    data = {
        "request_id": request_id,
        "score": score,
        "api_key": api_key,
    }
    if score_name is not None:
        data["name"] = score_name
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=data)
        if response.status_code != 200:
            warn_on_bad_response(
                response,
                "WARNING: While tracking your score, PromptLayer had the following error",
            )
            return False
    except httpx.RequestError as e:
        print(
            f"WARNING: While tracking your score PromptLayer had the following error: {str(e)}",
            file=sys.stderr,
        )
        return False

    return True


class GeneratorProxy:
    def __init__(self, generator, api_request_arguments, api_key):
        self.generator = generator
        self.results = []
        self.api_request_arugments = api_request_arguments
        self.api_key = api_key

    def __iter__(self):
        return self

    def __aiter__(self):
        return self

    async def __aenter__(self):
        api_request_arguments = self.api_request_arugments
        if hasattr(self.generator, "_AsyncMessageStreamManager__api_request"):
            return GeneratorProxy(
                await self.generator._AsyncMessageStreamManager__api_request,
                api_request_arguments,
                self.api_key,
            )

    def __enter__(self):
        api_request_arguments = self.api_request_arugments
        if hasattr(self.generator, "_MessageStreamManager__api_request"):
            stream = self.generator.__enter__()
            return GeneratorProxy(
                stream,
                api_request_arguments,
                self.api_key,
            )

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    async def __anext__(self):
        result = await self.generator.__anext__()
        return self._abstracted_next(result)

    def __next__(self):
        result = next(self.generator)
        return self._abstracted_next(result)

    def __getattr__(self, name):
        if name == "text_stream":  # anthropic async stream
            return GeneratorProxy(
                self.generator.text_stream, self.api_request_arugments, self.api_key
            )
        return getattr(self.generator, name)

    def _abstracted_next(self, result):
        self.results.append(result)
        provider_type = self.api_request_arugments["provider_type"]
        end_anthropic = False

        if provider_type == "anthropic":
            if hasattr(result, "stop_reason"):
                end_anthropic = result.stop_reason
            elif hasattr(result, "message"):
                end_anthropic = result.message.stop_reason
            elif hasattr(result, "type") and result.type == "message_stop":
                end_anthropic = True

        end_openai = provider_type == "openai" and (
            result.choices[0].finish_reason == "stop"
            or result.choices[0].finish_reason == "length"
        )

        if end_anthropic or end_openai:
            request_id = promptlayer_api_request(
                function_name=self.api_request_arugments["function_name"],
                provider_type=self.api_request_arugments["provider_type"],
                args=self.api_request_arugments["args"],
                kwargs=self.api_request_arugments["kwargs"],
                tags=self.api_request_arugments["tags"],
                response=self.cleaned_result(),
                request_start_time=self.api_request_arugments["request_start_time"],
                request_end_time=self.api_request_arugments["request_end_time"],
                api_key=self.api_key,
                return_pl_id=self.api_request_arugments["return_pl_id"],
                llm_request_span_id=self.api_request_arugments.get(
                    "llm_request_span_id"
                ),
            )

            if self.api_request_arugments["return_pl_id"]:
                return result, request_id

        if self.api_request_arugments["return_pl_id"]:
            return result, None

        return result

    def cleaned_result(self):
        provider_type = self.api_request_arugments["provider_type"]
        if provider_type == "anthropic":
            response = ""
            for result in self.results:
                if hasattr(result, "completion"):
                    response = f"{response}{result.completion}"
                elif hasattr(result, "message") and isinstance(result.message, str):
                    response = f"{response}{result.message}"
                elif (
                    hasattr(result, "content_block")
                    and hasattr(result.content_block, "text")
                    and "type" in result
                    and result.type != "message_stop"
                ):
                    response = f"{response}{result.content_block.text}"
                elif hasattr(result, "delta") and hasattr(result.delta, "text"):
                    response = f"{response}{result.delta.text}"
            if (
                hasattr(self.results[-1], "type")
                and self.results[-1].type == "message_stop"
            ):  # this is a message stream and not the correct event
                final_result = deepcopy(self.results[0].message)
                final_result.usage = None
                content_block = deepcopy(self.results[1].content_block)
                content_block.text = response
                final_result.content = [content_block]
            else:
                final_result = deepcopy(self.results[-1])
                final_result.completion = response
            return final_result
        if hasattr(self.results[0].choices[0], "text"):  # this is regular completion
            response = ""
            for result in self.results:
                response = f"{response}{result.choices[0].text}"
            final_result = deepcopy(self.results[-1])
            final_result.choices[0].text = response
            return final_result
        elif hasattr(
            self.results[0].choices[0], "delta"
        ):  # this is completion with delta
            response = {"role": "", "content": ""}
            for result in self.results:
                if (
                    hasattr(result.choices[0].delta, "role")
                    and result.choices[0].delta.role is not None
                ):
                    response["role"] = result.choices[0].delta.role
                if (
                    hasattr(result.choices[0].delta, "content")
                    and result.choices[0].delta.content is not None
                ):
                    response["content"] = response[
                        "content"
                    ] = f"{response['content']}{result.choices[0].delta.content}"
            final_result = deepcopy(self.results[-1])
            final_result.choices[0] = response
            return final_result
        return ""


async def run_in_thread_async(executor, func, *args, **kwargs):
    """https://github.com/python/cpython/blob/main/Lib/asyncio/threads.py"""
    loop = asyncio.get_running_loop()
    ctx = contextvars.copy_context()
    func_call = functools.partial(ctx.run, func, *args, **kwargs)
    res = await loop.run_in_executor(executor, func_call)
    return res


def warn_on_bad_response(request_response, main_message):
    if hasattr(request_response, "json"):
        try:
            print(
                f"{main_message}: {request_response.json().get('message')}",
                file=sys.stderr,
            )
        except json.JSONDecodeError:
            print(
                f"{main_message}: {request_response}",
                file=sys.stderr,
            )
    else:
        print(f"{main_message}: {request_response}", file=sys.stderr)


def raise_on_bad_response(request_response, main_message):
    if hasattr(request_response, "json"):
        try:
            raise Exception(
                f"{main_message}: {request_response.json().get('message') or request_response.json().get('error')}"
            )
        except json.JSONDecodeError:
            raise Exception(f"{main_message}: {request_response}")
    else:
        raise Exception(f"{main_message}: {request_response}")


async def async_wrapper(
    coroutine_obj,
    return_pl_id,
    request_start_time,
    function_name,
    provider_type,
    tags,
    api_key: str = None,
    llm_request_span_id: str = None,
    tracer=None,
    *args,
    **kwargs,
):
    current_context = context.get_current()
    token = context.attach(current_context)

    try:
        response = await coroutine_obj
        request_end_time = datetime.datetime.now().timestamp()
        result = await promptlayer_api_handler_async(
            function_name,
            provider_type,
            args,
            kwargs,
            tags,
            response,
            request_start_time,
            request_end_time,
            api_key,
            return_pl_id=return_pl_id,
            llm_request_span_id=llm_request_span_id,
        )

        if tracer:
            current_span = trace.get_current_span()
            if current_span:
                current_span.set_attribute("function_output", str(result))

        return result
    finally:
        context.detach(token)


def promptlayer_create_group(api_key: str = None):
    try:
        request_response = requests.post(
            f"{URL_API_PROMPTLAYER}/create-group",
            json={
                "api_key": api_key,
            },
        )
        if request_response.status_code != 200:
            warn_on_bad_response(
                request_response,
                "WARNING: While creating your group PromptLayer had the following error",
            )
            return False
    except requests.exceptions.RequestException as e:
        # I'm aiming for a more specific exception catch here
        raise Exception(
            f"PromptLayer had the following error while creating your group: {e}"
        )
    return request_response.json()["id"]


async def apromptlayer_create_group(api_key: Optional[str] = None) -> str:
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{URL_API_PROMPTLAYER}/create-group",
                json={
                    "api_key": api_key,
                },
            )
        if response.status_code != 200:
            warn_on_bad_response(
                response,
                "WARNING: While creating your group, PromptLayer had the following error",
            )
            return False
        return response.json()["id"]
    except httpx.RequestError as e:
        raise Exception(
            f"PromptLayer had the following error while creating your group: {str(e)}"
        ) from e


def promptlayer_track_group(request_id, group_id, api_key: str = None):
    try:
        request_response = requests.post(
            f"{URL_API_PROMPTLAYER}/track-group",
            json={
                "api_key": api_key,
                "request_id": request_id,
                "group_id": group_id,
            },
        )
        if request_response.status_code != 200:
            warn_on_bad_response(
                request_response,
                "WARNING: While tracking your group PromptLayer had the following error",
            )
            return False
    except requests.exceptions.RequestException as e:
        # I'm aiming for a more specific exception catch here
        raise Exception(
            f"PromptLayer had the following error while tracking your group: {e}"
        )
    return True


async def apromptlayer_track_group(request_id, group_id, api_key: str = None):
    try:
        payload = {
            "api_key": api_key,
            "request_id": request_id,
            "group_id": group_id,
        }
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{URL_API_PROMPTLAYER}/track-group",
                headers={"X-API-KEY": api_key},
                json=payload,
            )
        if response.status_code != 200:
            warn_on_bad_response(
                response,
                "WARNING: While tracking your group, PromptLayer had the following error",
            )
            return False
    except httpx.RequestError as e:
        print(
            f"WARNING: While tracking your group PromptLayer had the following error: {e}",
            file=sys.stderr,
        )
        return False

    return True


def get_prompt_template(
    prompt_name: str, params: Union[GetPromptTemplate, None] = None, api_key: str = None
) -> GetPromptTemplateResponse:
    try:
        json_body = {"api_key": api_key}
        if params:
            json_body = {**json_body, **params}
        response = requests.post(
            f"{URL_API_PROMPTLAYER}/prompt-templates/{prompt_name}",
            headers={"X-API-KEY": api_key},
            json=json_body,
        )
        if response.status_code != 200:
            raise Exception(
                f"PromptLayer had the following error while getting your prompt template: {response.text}"
            )

        warning = response.json().get("warning", None)
        if warning is not None:
            warn_on_bad_response(
                warning,
                "WARNING: While getting your prompt template",
            )
        return response.json()
    except requests.exceptions.RequestException as e:
        raise Exception(
            f"PromptLayer had the following error while getting your prompt template: {e}"
        )


async def aget_prompt_template(
    prompt_name: str,
    params: Union[GetPromptTemplate, None] = None,
    api_key: str = None,
) -> GetPromptTemplateResponse:
    try:
        json_body = {"api_key": api_key}
        if params:
            json_body.update(params)
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{URL_API_PROMPTLAYER}/prompt-templates/{prompt_name}",
                headers={"X-API-KEY": api_key},
                json=json_body,
            )
            if response.status_code != 200:
                raise_on_bad_response(
                    response,
                    "PromptLayer had the following error while getting your prompt template",
                )
        warning = response.json().get("warning", None)
        if warning:
            warn_on_bad_response(
                warning,
                "WARNING: While getting your prompt template",
            )
        return response.json()
    except httpx.RequestError as e:
        raise Exception(
            f"PromptLayer had the following error while getting your prompt template: {str(e)}"
        ) from e


def publish_prompt_template(
    body: PublishPromptTemplate,
    api_key: str = None,
) -> PublishPromptTemplateResponse:
    try:
        response = requests.post(
            f"{URL_API_PROMPTLAYER}/rest/prompt-templates",
            headers={"X-API-KEY": api_key},
            json={
                "prompt_template": {**body},
                "prompt_version": {**body},
                "release_labels": body.get("release_labels"),
            },
        )
        if response.status_code == 400:
            raise Exception(
                f"PromptLayer had the following error while publishing your prompt template: {response.text}"
            )
        return response.json()
    except requests.exceptions.RequestException as e:
        raise Exception(
            f"PromptLayer had the following error while publishing your prompt template: {e}"
        )


async def apublish_prompt_template(
    body: PublishPromptTemplate,
    api_key: str = None,
) -> PublishPromptTemplateResponse:
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{URL_API_PROMPTLAYER}/rest/prompt-templates",
                headers={"X-API-KEY": api_key},
                json={
                    "prompt_template": {**body},
                    "prompt_version": {**body},
                    "release_labels": body.get("release_labels"),
                },
            )
        if response.status_code == 400:
            raise Exception(
                f"PromptLayer had the following error while publishing your prompt template: {response.text}"
            )
        if response.status_code != 201:
            raise_on_bad_response(
                response,
                "PromptLayer had the following error while publishing your prompt template",
            )
        return response.json()
    except httpx.RequestError as e:
        raise Exception(
            f"PromptLayer had the following error while publishing your prompt template: {str(e)}"
        ) from e


def get_all_prompt_templates(
    page: int = 1, per_page: int = 30, api_key: str = None
) -> List[ListPromptTemplateResponse]:
    try:
        response = requests.get(
            f"{URL_API_PROMPTLAYER}/prompt-templates",
            headers={"X-API-KEY": api_key},
            params={"page": page, "per_page": per_page},
        )
        if response.status_code != 200:
            raise Exception(
                f"PromptLayer had the following error while getting all your prompt templates: {response.text}"
            )
        items = response.json().get("items", [])
        return items
    except requests.exceptions.RequestException as e:
        raise Exception(
            f"PromptLayer had the following error while getting all your prompt templates: {e}"
        )


async def aget_all_prompt_templates(
    page: int = 1, per_page: int = 30, api_key: str = None
) -> List[ListPromptTemplateResponse]:
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{URL_API_PROMPTLAYER}/prompt-templates",
                headers={"X-API-KEY": api_key},
                params={"page": page, "per_page": per_page},
            )
        if response.status_code != 200:
            raise_on_bad_response(
                response,
                "PromptLayer had the following error while getting all your prompt templates",
            )
        items = response.json().get("items", [])
        return items
    except httpx.RequestError as e:
        raise Exception(
            f"PromptLayer had the following error while getting all your prompt templates: {str(e)}"
        ) from e


def openai_stream_chat(results: list):
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
    tool_calls: Union[List[ChatCompletionMessageToolCall], None] = None
    for result in chat_completion_chunks:
        choices = result.choices
        if len(choices) == 0:
            continue
        if choices[0].delta.content:
            content = f"{content}{result.choices[0].delta.content}"

        delta = choices[0].delta
        if delta.tool_calls:
            tool_calls = tool_calls or []
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
            last_tool_call.function.name = (
                f"{last_tool_call.function.name}{tool_call.function.name or ''}"
            )
            last_tool_call.function.arguments = f"{last_tool_call.function.arguments}{tool_call.function.arguments or ''}"

    response.choices[0].message.content = content
    response.choices[0].message.tool_calls = tool_calls
    return response


async def aopenai_stream_chat(generator: AsyncIterable[Any]) -> Any:
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
    tool_calls: Union[List[ChatCompletionMessageToolCall], None] = None

    async for result in generator:
        chat_completion_chunks.append(result)
        choices = result.choices
        if len(choices) == 0:
            continue
        if choices[0].delta.content:
            content = f"{content}{choices[0].delta.content}"

        delta = choices[0].delta
        if delta.tool_calls:
            tool_calls = tool_calls or []
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
            last_tool_call.function.name = (
                f"{last_tool_call.function.name}{tool_call.function.name or ''}"
            )
            last_tool_call.function.arguments = f"{last_tool_call.function.arguments}{tool_call.function.arguments or ''}"

    # After collecting all chunks, set the response attributes
    if chat_completion_chunks:
        last_result = chat_completion_chunks[-1]
        response.id = last_result.id
        response.created = last_result.created
        response.model = last_result.model
        response.system_fingerprint = getattr(last_result, "system_fingerprint", None)
        response.usage = last_result.usage

    response.choices[0].message.content = content
    response.choices[0].message.tool_calls = tool_calls
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


def anthropic_stream_message(results: list):
    from anthropic.types import Message, MessageStreamEvent, TextBlock, Usage

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
    content = ""
    for result in message_stream_events:
        if result.type == "message_start":
            response = result.message
        elif result.type == "content_block_delta":
            if result.delta.type == "text_delta":
                content = f"{content}{result.delta.text}"
        elif result.type == "message_delta":
            if hasattr(result, "usage"):
                response.usage.output_tokens = result.usage.output_tokens
            if hasattr(result.delta, "stop_reason"):
                response.stop_reason = result.delta.stop_reason
    response.content.append(TextBlock(type="text", text=content))
    return response


async def aanthropic_stream_message(generator: AsyncIterable[Any]) -> Any:
    from anthropic.types import Message, MessageStreamEvent, TextBlock, Usage

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
    content = ""

    async for result in generator:
        message_stream_events.append(result)
        if result.type == "message_start":
            response = result.message
        elif result.type == "content_block_delta":
            if result.delta.type == "text_delta":
                content = f"{content}{result.delta.text}"
        elif result.type == "message_delta":
            if hasattr(result, "usage"):
                response.usage.output_tokens = result.usage.output_tokens
            if hasattr(result.delta, "stop_reason"):
                response.stop_reason = result.delta.stop_reason

    response.content.append(TextBlock(type="text", text=content))
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


def stream_response(
    generator: Generator, after_stream: Callable, map_results: Callable
):
    data = {
        "request_id": None,
        "raw_response": None,
        "prompt_blueprint": None,
    }
    results = []
    for result in generator:
        results.append(result)
        data["raw_response"] = result
        yield data
    request_response = map_results(results)
    response = after_stream(request_response=request_response.model_dump())
    data["request_id"] = response.get("request_id")
    data["prompt_blueprint"] = response.get("prompt_blueprint")
    yield data


async def astream_response(
    generator: AsyncIterable[Any],
    after_stream: Callable[..., Any],
    map_results: Callable[[Any], Any],
) -> AsyncGenerator[Dict[str, Any], None]:
    data = {
        "request_id": None,
        "raw_response": None,
        "prompt_blueprint": None,
    }
    results = []
    async for result in generator:
        results.append(result)
        data["raw_response"] = result
        yield data

    async def async_generator_from_list(lst):
        for item in lst:
            yield item

    request_response = await map_results(async_generator_from_list(results))
    after_stream_response = await after_stream(
        request_response=request_response.model_dump()
    )
    data["request_id"] = after_stream_response.get("request_id")
    data["prompt_blueprint"] = after_stream_response.get("prompt_blueprint")
    yield data


def openai_chat_request(client, **kwargs):
    return client.chat.completions.create(**kwargs)


def openai_completions_request(client, **kwargs):
    return client.completions.create(**kwargs)


MAP_TYPE_TO_OPENAI_FUNCTION = {
    "chat": openai_chat_request,
    "completion": openai_completions_request,
}


def openai_request(prompt_blueprint: GetPromptTemplateResponse, **kwargs):
    from openai import OpenAI

    client = OpenAI(base_url=kwargs.pop("base_url", None))
    request_to_make = MAP_TYPE_TO_OPENAI_FUNCTION[
        prompt_blueprint["prompt_template"]["type"]
    ]
    return request_to_make(client, **kwargs)


async def aopenai_chat_request(client, **kwargs):
    return await client.chat.completions.create(**kwargs)


async def aopenai_completions_request(client, **kwargs):
    return await client.completions.create(**kwargs)


AMAP_TYPE_TO_OPENAI_FUNCTION = {
    "chat": aopenai_chat_request,
    "completion": aopenai_completions_request,
}


async def aopenai_request(prompt_blueprint: GetPromptTemplateResponse, **kwargs):
    from openai import AsyncOpenAI

    client = AsyncOpenAI(base_url=kwargs.pop("base_url", None))
    request_to_make = AMAP_TYPE_TO_OPENAI_FUNCTION[
        prompt_blueprint["prompt_template"]["type"]
    ]
    return await request_to_make(client, **kwargs)


def azure_openai_request(prompt_blueprint: GetPromptTemplateResponse, **kwargs):
    from openai import AzureOpenAI

    client = AzureOpenAI(azure_endpoint=kwargs.pop("base_url", None))
    request_to_make = MAP_TYPE_TO_OPENAI_FUNCTION[
        prompt_blueprint["prompt_template"]["type"]
    ]
    return request_to_make(client, **kwargs)


async def aazure_openai_request(prompt_blueprint: GetPromptTemplateResponse, **kwargs):
    from openai import AsyncAzureOpenAI

    client = AsyncAzureOpenAI(azure_endpoint=kwargs.pop("base_url", None))
    request_to_make = AMAP_TYPE_TO_OPENAI_FUNCTION[
        prompt_blueprint["prompt_template"]["type"]
    ]
    return await request_to_make(client, **kwargs)


def anthropic_chat_request(client, **kwargs):
    return client.messages.create(**kwargs)


def anthropic_completions_request(client, **kwargs):
    return client.completions.create(**kwargs)


MAP_TYPE_TO_ANTHROPIC_FUNCTION = {
    "chat": anthropic_chat_request,
    "completion": anthropic_completions_request,
}


def anthropic_request(prompt_blueprint: GetPromptTemplateResponse, **kwargs):
    from anthropic import Anthropic

    client = Anthropic(base_url=kwargs.pop("base_url", None))
    request_to_make = MAP_TYPE_TO_ANTHROPIC_FUNCTION[
        prompt_blueprint["prompt_template"]["type"]
    ]
    return request_to_make(client, **kwargs)


async def aanthropic_chat_request(client, **kwargs):
    return await client.messages.create(**kwargs)


async def aanthropic_completions_request(client, **kwargs):
    return await client.completions.create(**kwargs)


AMAP_TYPE_TO_ANTHROPIC_FUNCTION = {
    "chat": aanthropic_chat_request,
    "completion": aanthropic_completions_request,
}


async def aanthropic_request(prompt_blueprint: GetPromptTemplateResponse, **kwargs):
    from anthropic import AsyncAnthropic

    client = AsyncAnthropic(base_url=kwargs.pop("base_url", None))
    request_to_make = AMAP_TYPE_TO_ANTHROPIC_FUNCTION[
        prompt_blueprint["prompt_template"]["type"]
    ]
    return await request_to_make(client, **kwargs)


# do not remove! This is used in the langchain integration.
def get_api_key():
    # raise an error if the api key is not set
    api_key = os.environ.get("PROMPTLAYER_API_KEY")
    if not api_key:
        raise Exception(
            "Please set your PROMPTLAYER_API_KEY environment variable or set API KEY in code using 'promptlayer.api_key = <your_api_key>' "
        )
    return api_key


def util_log_request(api_key: str, **kwargs) -> Union[RequestLog, None]:
    try:
        response = requests.post(
            f"{URL_API_PROMPTLAYER}/log-request",
            headers={"X-API-KEY": api_key},
            json=kwargs,
        )
        if response.status_code != 201:
            warn_on_bad_response(
                response,
                "WARNING: While logging your request PromptLayer had the following error",
            )
            return None
        return response.json()
    except Exception as e:
        print(
            f"WARNING: While tracking your prompt PromptLayer had the following error: {e}",
            file=sys.stderr,
        )
        return None


async def autil_log_request(api_key: str, **kwargs) -> Union[RequestLog, None]:
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{URL_API_PROMPTLAYER}/log-request",
                headers={"X-API-KEY": api_key},
                json=kwargs,
            )
        if response.status_code != 201:
            warn_on_bad_response(
                response,
                "WARNING: While logging your request PromptLayer had the following error",
            )
            return None
        return response.json()
    except Exception as e:
        print(
            f"WARNING: While tracking your prompt PromptLayer had the following error: {e}",
            file=sys.stderr,
        )
        return None


def mistral_request(
    prompt_blueprint: GetPromptTemplateResponse,
    **kwargs,
):
    from mistralai import Mistral

    client = Mistral(api_key=os.environ.get("MISTRAL_API_KEY"))
    if "stream" in kwargs and kwargs["stream"]:
        kwargs.pop("stream")
        return client.chat.stream(**kwargs)
    if "stream" in kwargs:
        kwargs.pop("stream")
    return client.chat.complete(**kwargs)


async def amistral_request(
    prompt_blueprint: GetPromptTemplateResponse,
    **kwargs,
):
    from mistralai import Mistral

    client = Mistral(api_key=os.environ.get("MISTRAL_API_KEY"))
    if "stream" in kwargs and kwargs["stream"]:
        return await client.chat.stream_async(**kwargs)
    return await client.chat.complete_async(**kwargs)


def mistral_stream_chat(results: list):
    from openai.types.chat import (
        ChatCompletion,
        ChatCompletionMessage,
        ChatCompletionMessageToolCall,
    )
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
                        last_tool_call.function.name = (
                            f"{last_tool_call.function.name}{tool_call.function.name}"
                        )
                    if tool_call.function.arguments:
                        last_tool_call.function.arguments = f"{last_tool_call.function.arguments}{tool_call.function.arguments}"

    response.choices[0].message.content = content
    response.choices[0].message.tool_calls = tool_calls
    response.usage = last_result.data.usage
    return response


async def amistral_stream_chat(generator: AsyncIterable[Any]) -> Any:
    from openai.types.chat import (
        ChatCompletion,
        ChatCompletionMessage,
        ChatCompletionMessageToolCall,
    )
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
                        last_tool_call.function.name = (
                            f"{last_tool_call.function.name}{tool_call.function.name}"
                        )
                    if tool_call.function.arguments:
                        last_tool_call.function.arguments = f"{last_tool_call.function.arguments}{tool_call.function.arguments}"

    if completion_chunks:
        last_result = completion_chunks[-1]
        response.id = last_result.data.id
        response.created = last_result.data.created
        response.model = last_result.data.model
        response.usage = last_result.data.usage

    response.choices[0].message.content = content
    response.choices[0].message.tool_calls = tool_calls
    return response
