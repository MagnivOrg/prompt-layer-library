import asyncio
import contextvars
import datetime
import functools
import json
import logging
import os
import sys
import types
from contextlib import asynccontextmanager
from copy import deepcopy
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional, Union
from uuid import uuid4

import httpx
import requests
import urllib3
import urllib3.util
from ably import AblyRealtime
from ably.types.message import Message
from centrifuge import (
    Client,
    PublicationContext,
    SubscriptionEventHandler,
    SubscriptionState,
)
from opentelemetry import context, trace

from promptlayer.types import RequestLog
from promptlayer.types.prompt_template import (
    GetPromptTemplate,
    GetPromptTemplateResponse,
    ListPromptTemplateResponse,
    PublishPromptTemplate,
    PublishPromptTemplateResponse,
)

# Configuration

RERAISE_ORIGINAL_EXCEPTION = os.getenv("PROMPTLAYER_RE_RAISE_ORIGINAL_EXCEPTION", "False").lower() == "true"
RAISE_FOR_STATUS = os.getenv("PROMPTLAYER_RAISE_FOR_STATUS", "False").lower() == "true"
DEFAULT_HTTP_TIMEOUT = 5

WORKFLOW_RUN_URL_TEMPLATE = "{base_url}/workflows/{workflow_id}/run"
WORKFLOW_RUN_CHANNEL_NAME_TEMPLATE = "workflows:{workflow_id}:run:{channel_name_suffix}"
SET_WORKFLOW_COMPLETE_MESSAGE = "SET_WORKFLOW_COMPLETE"
WS_TOKEN_REQUEST_LIBRARY_URL = (
    f"{os.getenv('PROMPTLAYER_BASE_URL', 'https://api.promptlayer.com')}/ws-token-request-library"
)


logger = logging.getLogger(__name__)


class FinalOutputCode(Enum):
    OK = "OK"
    EXCEEDS_SIZE_LIMIT = "EXCEEDS_SIZE_LIMIT"


def _get_http_timeout():
    try:
        return float(os.getenv("PROMPTLAYER_HTTP_TIMEOUT", DEFAULT_HTTP_TIMEOUT))
    except (ValueError, TypeError):
        return DEFAULT_HTTP_TIMEOUT


def _make_httpx_client():
    return httpx.AsyncClient(timeout=_get_http_timeout())


def _make_simple_httpx_client():
    return httpx.Client(timeout=_get_http_timeout())


def _get_workflow_workflow_id_or_name(workflow_id_or_name, workflow_name):
    # This is backward compatibility code
    if (workflow_id_or_name := workflow_name if workflow_id_or_name is None else workflow_id_or_name) is None:
        raise ValueError('Either "workflow_id_or_name" or "workflow_name" must be provided')

    return workflow_id_or_name


async def _get_final_output(
    base_url: str, execution_id: int, return_all_outputs: bool, *, headers: Dict[str, str]
) -> Dict[str, Any]:
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{base_url}/workflow-version-execution-results",
            headers=headers,
            params={"workflow_version_execution_id": execution_id, "return_all_outputs": return_all_outputs},
        )
        response.raise_for_status()
        return response.json()


# TODO(dmu) MEDIUM: Consider putting all these functions into a class, so we do not have to pass
#                   `authorization_headers` into each function
async def _resolve_workflow_id(base_url: str, workflow_id_or_name: Union[int, str], headers):
    if isinstance(workflow_id_or_name, int):
        return workflow_id_or_name

    # TODO(dmu) LOW: Should we warn user here to avoid using workflow names in favor of workflow id?
    async with _make_httpx_client() as client:
        # TODO(dmu) MEDIUM: Generalize the way we make async calls to PromptLayer API and reuse it everywhere
        response = await client.get(f"{base_url}/workflows/{workflow_id_or_name}", headers=headers)
        if RAISE_FOR_STATUS:
            response.raise_for_status()
        elif response.status_code != 200:
            raise_on_bad_response(response, "PromptLayer had the following error while running your workflow")

        return response.json()["workflow"]["id"]


async def _get_ably_token(base_url: str, channel_name, authentication_headers):
    try:
        async with _make_httpx_client() as client:
            response = await client.post(
                f"{base_url}/ws-token-request-library",
                headers=authentication_headers,
                params={"capability": channel_name},
            )
            if RAISE_FOR_STATUS:
                response.raise_for_status()
            elif response.status_code != 201:
                raise_on_bad_response(
                    response,
                    "PromptLayer had the following error while getting WebSocket token",
                )
            return response.json()
    except Exception as ex:
        error_message = f"Failed to get WebSocket token: {ex}"
        print(error_message)  # TODO(dmu) MEDIUM: Remove prints in favor of logging
        logger.exception(error_message)
        if RERAISE_ORIGINAL_EXCEPTION:
            raise
        else:
            raise Exception(error_message)


def _make_message_listener(base_url: str, results_future, execution_id_future, return_all_outputs, headers):
    # We need this function to be mocked by unittests
    async def message_listener(message: Message):
        if results_future.cancelled() or message.name != SET_WORKFLOW_COMPLETE_MESSAGE:
            return  # TODO(dmu) LOW: Do we really need this check?

        execution_id = await asyncio.wait_for(execution_id_future, _get_http_timeout() * 1.1)
        message_data = json.loads(message.data)
        if message_data["workflow_version_execution_id"] != execution_id:
            return

        if (result_code := message_data.get("result_code")) in (FinalOutputCode.OK.value, None):
            results = message_data["final_output"]
        elif result_code == FinalOutputCode.EXCEEDS_SIZE_LIMIT.value:
            results = await _get_final_output(base_url, execution_id, return_all_outputs, headers=headers)
        else:
            raise NotImplementedError(f"Unsupported final output code: {result_code}")

        results_future.set_result(results)

    return message_listener


async def _subscribe_to_workflow_completion_channel(
    base_url: str, channel, execution_id_future, return_all_outputs, headers
):
    results_future = asyncio.Future()
    message_listener = _make_message_listener(
        base_url, results_future, execution_id_future, return_all_outputs, headers
    )
    await channel.subscribe(SET_WORKFLOW_COMPLETE_MESSAGE, message_listener)
    return results_future, message_listener


async def _post_workflow_id_run(
    *,
    base_url: str,
    authentication_headers,
    workflow_id,
    input_variables: Dict[str, Any],
    metadata: Dict[str, Any],
    workflow_label_name: str,
    workflow_version_number: int,
    return_all_outputs: bool,
    channel_name_suffix: str,
    _url_template: str = WORKFLOW_RUN_URL_TEMPLATE,
):
    url = _url_template.format(base_url=base_url, workflow_id=workflow_id)
    payload = {
        "input_variables": input_variables,
        "metadata": metadata,
        "workflow_label_name": workflow_label_name,
        "workflow_version_number": workflow_version_number,
        "return_all_outputs": return_all_outputs,
        "channel_name_suffix": channel_name_suffix,
    }
    try:
        async with _make_httpx_client() as client:
            response = await client.post(url, json=payload, headers=authentication_headers)
            if RAISE_FOR_STATUS:
                response.raise_for_status()
            elif response.status_code != 201:
                raise_on_bad_response(response, "PromptLayer had the following error while running your workflow")

            result = response.json()
            if warning := result.get("warning"):
                print(f"WARNING: {warning}")
    except Exception as ex:
        error_message = f"Failed to run workflow: {str(ex)}"
        print(error_message)  # TODO(dmu) MEDIUM: Remove prints in favor of logging
        logger.exception(error_message)
        if RERAISE_ORIGINAL_EXCEPTION:
            raise
        else:
            raise Exception(error_message)

    return result.get("workflow_version_execution_id")


async def _wait_for_workflow_completion(channel, results_future, message_listener, timeout):
    # We need this function for mocking in unittests
    try:
        return await asyncio.wait_for(results_future, timeout)
    except asyncio.TimeoutError:
        raise Exception("Workflow execution did not complete properly")
    finally:
        channel.unsubscribe(SET_WORKFLOW_COMPLETE_MESSAGE, message_listener)


def _make_channel_name_suffix():
    # We need this function for mocking in unittests
    return uuid4().hex


MessageCallback = Callable[[Message], Coroutine[None, None, None]]


class SubscriptionEventLoggerHandler(SubscriptionEventHandler):
    def __init__(self, callback: MessageCallback):
        self.callback = callback

    async def on_publication(self, ctx: PublicationContext):
        message_name = ctx.pub.data.get("message_name", "unknown")
        data = ctx.pub.data.get("data", "")
        message = Message(name=message_name, data=data)
        await self.callback(message)


@asynccontextmanager
async def centrifugo_client(address: str, token: str):
    client = Client(address, token=token)
    try:
        await client.connect()
        yield client
    finally:
        await client.disconnect()


@asynccontextmanager
async def centrifugo_subscription(client: Client, topic: str, message_listener: MessageCallback):
    subscription = client.new_subscription(
        topic,
        events=SubscriptionEventLoggerHandler(message_listener),
    )
    try:
        await subscription.subscribe()
        yield
    finally:
        if subscription.state == SubscriptionState.SUBSCRIBED:
            await subscription.unsubscribe()


async def arun_workflow_request(
    *,
    api_key: str,
    base_url: str,
    workflow_id_or_name: Optional[Union[int, str]] = None,
    input_variables: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
    workflow_label_name: Optional[str] = None,
    workflow_version_number: Optional[int] = None,
    return_all_outputs: Optional[bool] = False,
    timeout: Optional[int] = 3600,
    # `workflow_name` deprecated, kept for backward compatibility only.
    workflow_name: Optional[str] = None,
):
    headers = {"X-API-KEY": api_key}
    workflow_id = await _resolve_workflow_id(
        base_url, _get_workflow_workflow_id_or_name(workflow_id_or_name, workflow_name), headers
    )
    channel_name_suffix = _make_channel_name_suffix()
    channel_name = WORKFLOW_RUN_CHANNEL_NAME_TEMPLATE.format(
        workflow_id=workflow_id, channel_name_suffix=channel_name_suffix
    )
    ably_token = await _get_ably_token(base_url, channel_name, headers)
    token = ably_token["token_details"]["token"]

    execution_id_future = asyncio.Future[int]()

    if ably_token.get("messaging_backend") == "centrifugo":
        address = urllib3.util.parse_url(base_url)._replace(scheme="wss", path="/connection/websocket").url
        async with centrifugo_client(address, token) as client:
            results_future = asyncio.Future[dict[str, Any]]()
            async with centrifugo_subscription(
                client,
                channel_name,
                _make_message_listener(base_url, results_future, execution_id_future, return_all_outputs, headers),
            ):
                execution_id = await _post_workflow_id_run(
                    base_url=base_url,
                    authentication_headers=headers,
                    workflow_id=workflow_id,
                    input_variables=input_variables,
                    metadata=metadata,
                    workflow_label_name=workflow_label_name,
                    workflow_version_number=workflow_version_number,
                    return_all_outputs=return_all_outputs,
                    channel_name_suffix=channel_name_suffix,
                )
                execution_id_future.set_result(execution_id)
                await asyncio.wait_for(results_future, timeout)
                return results_future.result()

    async with AblyRealtime(token=token) as ably_client:
        # It is crucial to subscribe before running a workflow, otherwise we may miss a completion message
        channel = ably_client.channels.get(channel_name)
        results_future, message_listener = await _subscribe_to_workflow_completion_channel(
            base_url, channel, execution_id_future, return_all_outputs, headers
        )

        execution_id = await _post_workflow_id_run(
            base_url=base_url,
            authentication_headers=headers,
            workflow_id=workflow_id,
            input_variables=input_variables,
            metadata=metadata,
            workflow_label_name=workflow_label_name,
            workflow_version_number=workflow_version_number,
            return_all_outputs=return_all_outputs,
            channel_name_suffix=channel_name_suffix,
        )
        execution_id_future.set_result(execution_id)

        return await _wait_for_workflow_completion(channel, results_future, message_listener, timeout)


def promptlayer_api_handler(
    api_key: str,
    base_url: str,
    function_name,
    provider_type,
    args,
    kwargs,
    tags,
    response,
    request_start_time,
    request_end_time,
    return_pl_id=False,
    llm_request_span_id=None,
):
    if (
        isinstance(response, types.GeneratorType)
        or isinstance(response, types.AsyncGeneratorType)
        or type(response).__name__ in ["Stream", "AsyncStream", "AsyncMessageStreamManager", "MessageStreamManager"]
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
            base_url=base_url,
        )
    else:
        request_id = promptlayer_api_request(
            base_url=base_url,
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
    api_key: str,
    base_url: str,
    function_name,
    provider_type,
    args,
    kwargs,
    tags,
    response,
    request_start_time,
    request_end_time,
    return_pl_id=False,
    llm_request_span_id=None,
):
    return await run_in_thread_async(
        None,
        promptlayer_api_handler,
        api_key,
        base_url,
        function_name,
        provider_type,
        args,
        kwargs,
        tags,
        response,
        request_start_time,
        request_end_time,
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
        return {k: convert_native_object_to_dict(v) for k, v in native_object.__dict__.items()}
    return native_object


def promptlayer_api_request(
    *,
    base_url: str,
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
    if hasattr(response, "dict"):  # added this for anthropic 3.0 changes, they return a completion object
        response = response.dict()
    try:
        request_response = requests.post(
            f"{base_url}/track-request",
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
                request_response, "WARNING: While logging your request PromptLayer had the following issue"
            )
        elif request_response.status_code != 200:
            warn_on_bad_response(
                request_response, "WARNING: While logging your request PromptLayer had the following error"
            )
    except Exception as e:
        print(f"WARNING: While logging your request PromptLayer had the following error: {e}", file=sys.stderr)
    if request_response is not None and return_pl_id:
        return request_response.json().get("request_id")


def track_request(base_url: str, **body):
    try:
        response = requests.post(
            f"{base_url}/track-request",
            json=body,
        )
        if response.status_code != 200:
            warn_on_bad_response(
                response, f"PromptLayer had the following error while tracking your request: {response.text}"
            )
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"WARNING: While logging your request PromptLayer had the following error: {e}", file=sys.stderr)
        return {}


async def atrack_request(base_url: str, **body: Any) -> Dict[str, Any]:
    try:
        async with _make_httpx_client() as client:
            response = await client.post(
                f"{base_url}/track-request",
                json=body,
            )
            if RAISE_FOR_STATUS:
                response.raise_for_status()
            elif response.status_code != 200:
                warn_on_bad_response(
                    response, f"PromptLayer had the following error while tracking your request: {response.text}"
                )
        return response.json()
    except httpx.RequestError as e:
        print(f"WARNING: While logging your request PromptLayer had the following error: {e}", file=sys.stderr)
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


def promptlayer_get_prompt(api_key: str, base_url: str, prompt_name, version: int = None, label: str = None):
    """
    Get a prompt from the PromptLayer library
    version: version of the prompt to get, None for latest
    label: The specific label of a prompt you want to get. Setting this will supercede version
    """
    try:
        request_response = requests.get(
            f"{base_url}/library-get-prompt-template",
            headers={"X-API-KEY": api_key},
            params={"prompt_name": prompt_name, "version": version, "label": label},
        )
    except Exception as e:
        raise Exception(f"PromptLayer had the following error while getting your prompt: {e}")
    if request_response.status_code != 200:
        raise_on_bad_response(
            request_response,
            "PromptLayer had the following error while getting your prompt",
        )

    return request_response.json()


def promptlayer_publish_prompt(
    api_key: str, base_url: str, prompt_name, prompt_template, commit_message, tags, metadata=None
):
    try:
        request_response = requests.post(
            f"{base_url}/library-publish-prompt-template",
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
        raise Exception(f"PromptLayer had the following error while publishing your prompt: {e}")
    if request_response.status_code != 200:
        raise_on_bad_response(
            request_response,
            "PromptLayer had the following error while publishing your prompt",
        )
    return True


def promptlayer_track_prompt(api_key: str, base_url: str, request_id, prompt_name, input_variables, version, label):
    try:
        request_response = requests.post(
            f"{base_url}/library-track-prompt",
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
    api_key: str,
    base_url: str,
    request_id: str,
    prompt_name: str,
    input_variables: Dict[str, Any],
    version: Optional[int] = None,
    label: Optional[str] = None,
) -> bool:
    url = f"{base_url}/library-track-prompt"
    payload = {
        "request_id": request_id,
        "prompt_name": prompt_name,
        "prompt_input_variables": input_variables,
        "api_key": api_key,
        "version": version,
        "label": label,
    }
    try:
        async with _make_httpx_client() as client:
            response = await client.post(url, json=payload)

        if RAISE_FOR_STATUS:
            response.raise_for_status()
        elif response.status_code != 200:
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


def promptlayer_track_metadata(api_key: str, base_url: str, request_id, metadata):
    try:
        request_response = requests.post(
            f"{base_url}/library-track-metadata",
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


async def apromptlayer_track_metadata(api_key: str, base_url: str, request_id: str, metadata: Dict[str, Any]) -> bool:
    url = f"{base_url}/library-track-metadata"
    payload = {
        "request_id": request_id,
        "metadata": metadata,
        "api_key": api_key,
    }
    try:
        async with _make_httpx_client() as client:
            response = await client.post(url, json=payload)

        if RAISE_FOR_STATUS:
            response.raise_for_status()
        elif response.status_code != 200:
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


def promptlayer_track_score(api_key: str, base_url: str, request_id, score, score_name):
    try:
        data = {"request_id": request_id, "score": score, "api_key": api_key}
        if score_name is not None:
            data["name"] = score_name
        request_response = requests.post(
            f"{base_url}/library-track-score",
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
    api_key: str,
    base_url: str,
    request_id: str,
    score: float,
    score_name: Optional[str],
) -> bool:
    url = f"{base_url}/library-track-score"
    data = {
        "request_id": request_id,
        "score": score,
        "api_key": api_key,
    }
    if score_name is not None:
        data["name"] = score_name
    try:
        async with _make_httpx_client() as client:
            response = await client.post(url, json=data)

        if RAISE_FOR_STATUS:
            response.raise_for_status()
        elif response.status_code != 200:
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


def build_anthropic_content_blocks(events):
    content_blocks = []
    current_block = None
    current_signature = ""
    current_thinking = ""
    current_text = ""
    current_tool_input_json = ""
    usage = None
    stop_reason = None

    for event in events:
        if event.type == "content_block_start":
            current_block = deepcopy(event.content_block)
            if current_block.type == "thinking":
                current_signature = ""
                current_thinking = ""
            elif current_block.type == "text":
                current_text = ""
            elif current_block.type == "tool_use":
                current_tool_input_json = ""
        elif event.type == "content_block_delta" and current_block is not None:
            if current_block.type == "thinking":
                if hasattr(event.delta, "signature"):
                    current_signature = event.delta.signature
                if hasattr(event.delta, "thinking"):
                    current_thinking += event.delta.thinking
            elif current_block.type == "text":
                if hasattr(event.delta, "text"):
                    current_text += event.delta.text
            elif current_block.type == "tool_use":
                if hasattr(event.delta, "partial_json"):
                    current_tool_input_json += event.delta.partial_json
        elif event.type == "content_block_stop" and current_block is not None:
            if current_block.type == "thinking":
                current_block.signature = current_signature
                current_block.thinking = current_thinking
            elif current_block.type == "text":
                current_block.text = current_text
            elif current_block.type == "tool_use":
                try:
                    current_block.input = json.loads(current_tool_input_json)
                except json.JSONDecodeError:
                    current_block.input = {}
            content_blocks.append(current_block)
            current_block = None
            current_signature = ""
            current_thinking = ""
            current_text = ""
            current_tool_input_json = ""
        elif event.type == "message_delta":
            if hasattr(event, "usage"):
                usage = event.usage
            if hasattr(event.delta, "stop_reason"):
                stop_reason = event.delta.stop_reason
    return content_blocks, usage, stop_reason


class GeneratorProxy:
    def __init__(self, generator, api_request_arguments, api_key, base_url):
        self.generator = generator
        self.results = []
        self.api_request_arugments = api_request_arguments
        self.api_key = api_key
        self.base_url = base_url

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
                self.base_url,
            )

    def __enter__(self):
        api_request_arguments = self.api_request_arugments
        if hasattr(self.generator, "_MessageStreamManager__api_request"):
            stream = self.generator.__enter__()
            return GeneratorProxy(
                stream,
                api_request_arguments,
                self.api_key,
                self.base_url,
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
            return GeneratorProxy(self.generator.text_stream, self.api_request_arugments, self.api_key, self.base_url)
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
            result.choices[0].finish_reason == "stop" or result.choices[0].finish_reason == "length"
        )

        if end_anthropic or end_openai:
            request_id = promptlayer_api_request(
                base_url=self.base_url,
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
                llm_request_span_id=self.api_request_arugments.get("llm_request_span_id"),
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
                    response += result.completion
                elif hasattr(result, "message") and isinstance(result.message, str):
                    response += result.message
                elif (
                    hasattr(result, "content_block")
                    and hasattr(result.content_block, "text")
                    and getattr(result, "type", None) != "message_stop"
                ):
                    response += result.content_block.text
                elif hasattr(result, "delta"):
                    if hasattr(result.delta, "thinking"):
                        response += result.delta.thinking
                    elif hasattr(result.delta, "text"):
                        response += result.delta.text

            # 2) If this is a “stream” (ended by message_stop), reconstruct both ThinkingBlock & TextBlock
            last_event = self.results[-1]
            if getattr(last_event, "type", None) == "message_stop":
                final_result = deepcopy(self.results[0].message)

                content_blocks, usage, stop_reason = build_anthropic_content_blocks(self.results)
            final_result.content = content_blocks
            if usage:
                final_result.usage.output_tokens = usage.output_tokens
            if stop_reason:
                final_result.stop_reason = stop_reason
            return final_result
        else:
            return deepcopy(self.results[-1])
        if hasattr(self.results[0].choices[0], "text"):  # this is regular completion
            response = ""
            for result in self.results:
                response = f"{response}{result.choices[0].text}"
            final_result = deepcopy(self.results[-1])
            final_result.choices[0].text = response
            return final_result
        elif hasattr(self.results[0].choices[0], "delta"):  # this is completion with delta
            response = {"role": "", "content": ""}
            for result in self.results:
                if hasattr(result.choices[0].delta, "role") and result.choices[0].delta.role is not None:
                    response["role"] = result.choices[0].delta.role
                if hasattr(result.choices[0].delta, "content") and result.choices[0].delta.content is not None:
                    response["content"] = response["content"] = (
                        f"{response['content']}{result.choices[0].delta.content}"
                    )
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
    api_key: str,
    base_url: str,
    coroutine_obj,
    return_pl_id,
    request_start_time,
    function_name,
    provider_type,
    tags,
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
            api_key,
            base_url,
            function_name,
            provider_type,
            args,
            kwargs,
            tags,
            response,
            request_start_time,
            request_end_time,
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


def promptlayer_create_group(api_key: str, base_url: str):
    try:
        request_response = requests.post(
            f"{base_url}/create-group",
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
        raise Exception(f"PromptLayer had the following error while creating your group: {e}")
    return request_response.json()["id"]


async def apromptlayer_create_group(api_key: str, base_url: str):
    try:
        async with _make_httpx_client() as client:
            response = await client.post(
                f"{base_url}/create-group",
                json={
                    "api_key": api_key,
                },
            )

        if RAISE_FOR_STATUS:
            response.raise_for_status()
        elif response.status_code != 200:
            warn_on_bad_response(
                response,
                "WARNING: While creating your group, PromptLayer had the following error",
            )
            return False
        return response.json()["id"]
    except httpx.RequestError as e:
        raise Exception(f"PromptLayer had the following error while creating your group: {str(e)}") from e


def promptlayer_track_group(api_key: str, base_url: str, request_id, group_id):
    try:
        request_response = requests.post(
            f"{base_url}/track-group",
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
        raise Exception(f"PromptLayer had the following error while tracking your group: {e}")
    return True


async def apromptlayer_track_group(api_key: str, base_url: str, request_id, group_id):
    try:
        payload = {
            "api_key": api_key,
            "request_id": request_id,
            "group_id": group_id,
        }
        async with _make_httpx_client() as client:
            response = await client.post(
                f"{base_url}/track-group",
                headers={"X-API-KEY": api_key},
                json=payload,
            )

        if RAISE_FOR_STATUS:
            response.raise_for_status()
        elif response.status_code != 200:
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
    api_key: str, base_url: str, prompt_name: str, params: Union[GetPromptTemplate, None] = None
) -> GetPromptTemplateResponse:
    try:
        json_body = {"api_key": api_key}
        if params:
            json_body = {**json_body, **params}
        response = requests.post(
            f"{base_url}/prompt-templates/{prompt_name}",
            headers={"X-API-KEY": api_key},
            json=json_body,
        )
        if response.status_code != 200:
            raise Exception(f"PromptLayer had the following error while getting your prompt template: {response.text}")

        warning = response.json().get("warning", None)
        if warning is not None:
            warn_on_bad_response(
                warning,
                "WARNING: While getting your prompt template",
            )
        return response.json()
    except requests.exceptions.RequestException as e:
        raise Exception(f"PromptLayer had the following error while getting your prompt template: {e}")


async def aget_prompt_template(
    api_key: str,
    base_url: str,
    prompt_name: str,
    params: Union[GetPromptTemplate, None] = None,
) -> GetPromptTemplateResponse:
    try:
        json_body = {"api_key": api_key}
        if params:
            json_body.update(params)
        async with _make_httpx_client() as client:
            response = await client.post(
                f"{base_url}/prompt-templates/{prompt_name}",
                headers={"X-API-KEY": api_key},
                json=json_body,
            )

            if RAISE_FOR_STATUS:
                response.raise_for_status()
            elif response.status_code != 200:
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
        raise Exception(f"PromptLayer had the following error while getting your prompt template: {str(e)}") from e


def publish_prompt_template(
    api_key: str,
    base_url: str,
    body: PublishPromptTemplate,
) -> PublishPromptTemplateResponse:
    try:
        response = requests.post(
            f"{base_url}/rest/prompt-templates",
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
        raise Exception(f"PromptLayer had the following error while publishing your prompt template: {e}")


async def apublish_prompt_template(
    api_key: str,
    base_url: str,
    body: PublishPromptTemplate,
) -> PublishPromptTemplateResponse:
    try:
        async with _make_httpx_client() as client:
            response = await client.post(
                f"{base_url}/rest/prompt-templates",
                headers={"X-API-KEY": api_key},
                json={
                    "prompt_template": {**body},
                    "prompt_version": {**body},
                    "release_labels": body.get("release_labels"),
                },
            )

        if RAISE_FOR_STATUS:
            response.raise_for_status()
        elif response.status_code == 400:
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
        raise Exception(f"PromptLayer had the following error while publishing your prompt template: {str(e)}") from e


def get_all_prompt_templates(
    api_key: str, base_url: str, page: int = 1, per_page: int = 30, label: str = None
) -> List[ListPromptTemplateResponse]:
    try:
        params = {"page": page, "per_page": per_page}
        if label:
            params["label"] = label
        response = requests.get(
            f"{base_url}/prompt-templates",
            headers={"X-API-KEY": api_key},
            params=params,
        )
        if response.status_code != 200:
            raise Exception(
                f"PromptLayer had the following error while getting all your prompt templates: {response.text}"
            )
        items = response.json().get("items", [])
        return items
    except requests.exceptions.RequestException as e:
        raise Exception(f"PromptLayer had the following error while getting all your prompt templates: {e}")


async def aget_all_prompt_templates(
    api_key: str, base_url: str, page: int = 1, per_page: int = 30, label: str = None
) -> List[ListPromptTemplateResponse]:
    try:
        params = {"page": page, "per_page": per_page}
        if label:
            params["label"] = label
        async with _make_httpx_client() as client:
            response = await client.get(
                f"{base_url}/prompt-templates",
                headers={"X-API-KEY": api_key},
                params=params,
            )

        if RAISE_FOR_STATUS:
            response.raise_for_status()
        elif response.status_code != 200:
            raise_on_bad_response(
                response,
                "PromptLayer had the following error while getting all your prompt templates",
            )
        items = response.json().get("items", [])
        return items
    except httpx.RequestError as e:
        raise Exception(f"PromptLayer had the following error while getting all your prompt templates: {str(e)}") from e


def openai_chat_request(client, **kwargs):
    return client.chat.completions.create(**kwargs)


def openai_completions_request(client, **kwargs):
    return client.completions.create(**kwargs)


MAP_TYPE_TO_OPENAI_FUNCTION = {
    "chat": openai_chat_request,
    "completion": openai_completions_request,
}


def openai_request(prompt_blueprint: GetPromptTemplateResponse, client_kwargs: dict, function_kwargs: dict):
    from openai import OpenAI

    client = OpenAI(**client_kwargs)
    api_type = prompt_blueprint["metadata"]["model"].get("api_type", "chat-completions")

    if api_type == "chat-completions":
        request_to_make = MAP_TYPE_TO_OPENAI_FUNCTION[prompt_blueprint["prompt_template"]["type"]]
        return request_to_make(client, **function_kwargs)
    else:
        return client.responses.create(**function_kwargs)


async def aopenai_chat_request(client, **kwargs):
    return await client.chat.completions.create(**kwargs)


async def aopenai_completions_request(client, **kwargs):
    return await client.completions.create(**kwargs)


AMAP_TYPE_TO_OPENAI_FUNCTION = {
    "chat": aopenai_chat_request,
    "completion": aopenai_completions_request,
}


async def aopenai_request(prompt_blueprint: GetPromptTemplateResponse, client_kwargs: dict, function_kwargs: dict):
    from openai import AsyncOpenAI

    client = AsyncOpenAI(**client_kwargs)
    api_type = prompt_blueprint["metadata"]["model"].get("api_type", "chat-completions")

    if api_type == "chat-completions":
        request_to_make = AMAP_TYPE_TO_OPENAI_FUNCTION[prompt_blueprint["prompt_template"]["type"]]
        return await request_to_make(client, **function_kwargs)
    else:
        return await client.responses.create(**function_kwargs)


def azure_openai_request(prompt_blueprint: GetPromptTemplateResponse, client_kwargs: dict, function_kwargs: dict):
    from openai import AzureOpenAI

    client = AzureOpenAI(azure_endpoint=client_kwargs.pop("base_url", None))
    api_type = prompt_blueprint["metadata"]["model"].get("api_type", "chat-completions")

    if api_type == "chat-completions":
        request_to_make = MAP_TYPE_TO_OPENAI_FUNCTION[prompt_blueprint["prompt_template"]["type"]]
        return request_to_make(client, **function_kwargs)
    else:
        return client.responses.create(**function_kwargs)


async def aazure_openai_request(
    prompt_blueprint: GetPromptTemplateResponse, client_kwargs: dict, function_kwargs: dict
):
    from openai import AsyncAzureOpenAI

    client = AsyncAzureOpenAI(azure_endpoint=client_kwargs.pop("base_url", None))
    api_type = prompt_blueprint["metadata"]["model"].get("api_type", "chat-completions")

    if api_type == "chat-completions":
        request_to_make = AMAP_TYPE_TO_OPENAI_FUNCTION[prompt_blueprint["prompt_template"]["type"]]
        return await request_to_make(client, **function_kwargs)
    else:
        return await client.responses.create(**function_kwargs)


def anthropic_chat_request(client, **kwargs):
    return client.messages.create(**kwargs)


def anthropic_completions_request(client, **kwargs):
    return client.completions.create(**kwargs)


MAP_TYPE_TO_ANTHROPIC_FUNCTION = {
    "chat": anthropic_chat_request,
    "completion": anthropic_completions_request,
}


def anthropic_request(prompt_blueprint: GetPromptTemplateResponse, client_kwargs: dict, function_kwargs: dict):
    from anthropic import Anthropic

    client = Anthropic(**client_kwargs)
    request_to_make = MAP_TYPE_TO_ANTHROPIC_FUNCTION[prompt_blueprint["prompt_template"]["type"]]
    return request_to_make(client, **function_kwargs)


async def aanthropic_chat_request(client, **kwargs):
    return await client.messages.create(**kwargs)


async def aanthropic_completions_request(client, **kwargs):
    return await client.completions.create(**kwargs)


AMAP_TYPE_TO_ANTHROPIC_FUNCTION = {
    "chat": aanthropic_chat_request,
    "completion": aanthropic_completions_request,
}


async def aanthropic_request(prompt_blueprint: GetPromptTemplateResponse, client_kwargs: dict, function_kwargs: dict):
    from anthropic import AsyncAnthropic

    client = AsyncAnthropic(**client_kwargs)
    request_to_make = AMAP_TYPE_TO_ANTHROPIC_FUNCTION[prompt_blueprint["prompt_template"]["type"]]
    return await request_to_make(client, **function_kwargs)


# do not remove! This is used in the langchain integration.
def get_api_key():
    # raise an error if the api key is not set
    api_key = os.environ.get("PROMPTLAYER_API_KEY")
    if not api_key:
        raise Exception(
            "Please set your PROMPTLAYER_API_KEY environment variable or set API KEY in code using 'promptlayer.api_key = <your_api_key>' "
        )
    return api_key


def util_log_request(api_key: str, base_url: str, **kwargs) -> Union[RequestLog, None]:
    try:
        response = requests.post(
            f"{base_url}/log-request",
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


async def autil_log_request(api_key: str, base_url: str, **kwargs) -> Union[RequestLog, None]:
    try:
        async with _make_httpx_client() as client:
            response = await client.post(
                f"{base_url}/log-request",
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


def mistral_request(prompt_blueprint: GetPromptTemplateResponse, client_kwargs: dict, function_kwargs: dict):
    from mistralai import Mistral

    client = Mistral(api_key=os.environ.get("MISTRAL_API_KEY"), client=_make_simple_httpx_client())
    if "stream" in function_kwargs and function_kwargs["stream"]:
        function_kwargs.pop("stream")
        return client.chat.stream(**function_kwargs)
    if "stream" in function_kwargs:
        function_kwargs.pop("stream")
    return client.chat.complete(**function_kwargs)


async def amistral_request(
    prompt_blueprint: GetPromptTemplateResponse,
    _: dict,
    function_kwargs: dict,
):
    from mistralai import Mistral

    client = Mistral(api_key=os.environ.get("MISTRAL_API_KEY"), async_client=_make_httpx_client())
    if "stream" in function_kwargs and function_kwargs["stream"]:
        return await client.chat.stream_async(**function_kwargs)
    return await client.chat.complete_async(**function_kwargs)


def google_chat_request(client, **kwargs):
    from google.genai.chats import Content

    stream = kwargs.pop("stream", False)
    model = kwargs.get("model", "gemini-2.0-flash")
    history = [Content(**item) for item in kwargs.get("history", [])]
    generation_config = kwargs.get("generation_config", {})
    chat = client.chats.create(model=model, history=history, config=generation_config)
    last_message = history[-1].parts if history else ""
    if stream:
        return chat.send_message_stream(message=last_message)
    return chat.send_message(message=last_message)


def google_completions_request(client, **kwargs):
    config = kwargs.pop("generation_config", {})
    model = kwargs.get("model", "gemini-2.0-flash")
    contents = kwargs.get("contents", [])
    stream = kwargs.pop("stream", False)
    if stream:
        return client.models.generate_content_stream(model=model, contents=contents, config=config)
    return client.models.generate_content(model=model, contents=contents, config=config)


MAP_TYPE_TO_GOOGLE_FUNCTION = {
    "chat": google_chat_request,
    "completion": google_completions_request,
}


def google_request(prompt_blueprint: GetPromptTemplateResponse, client_kwargs: dict, function_kwargs: dict):
    from google import genai

    if os.environ.get("GOOGLE_GENAI_USE_VERTEXAI") == "true":
        client = genai.Client(
            vertexai=True,
            project=os.environ.get("GOOGLE_CLOUD_PROJECT"),
            location=os.environ.get("GOOGLE_CLOUD_LOCATION"),
        )
    else:
        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
    request_to_make = MAP_TYPE_TO_GOOGLE_FUNCTION[prompt_blueprint["prompt_template"]["type"]]
    return request_to_make(client, **function_kwargs)


async def agoogle_chat_request(client, **kwargs):
    from google.genai.chats import Content

    stream = kwargs.pop("stream", False)
    model = kwargs.get("model", "gemini-2.0-flash")
    history = [Content(**item) for item in kwargs.get("history", [])]
    generation_config = kwargs.get("generation_config", {})
    chat = client.aio.chats.create(model=model, history=history, config=generation_config)
    last_message = history[-1].parts[0] if history else ""
    if stream:
        return await chat.send_message_stream(message=last_message)
    return await chat.send_message(message=last_message)


async def agoogle_completions_request(client, **kwargs):
    config = kwargs.pop("generation_config", {})
    model = kwargs.get("model", "gemini-2.0-flash")
    contents = kwargs.get("contents", [])
    stream = kwargs.pop("stream", False)
    if stream:
        return await client.aio.models.generate_content_stream(model=model, contents=contents, config=config)
        return await client.aio.models.generate_content(model=model, contents=contents, config=config)


AMAP_TYPE_TO_GOOGLE_FUNCTION = {
    "chat": agoogle_chat_request,
    "completion": agoogle_completions_request,
}


async def agoogle_request(prompt_blueprint: GetPromptTemplateResponse, client_kwargs: dict, function_kwargs: dict):
    from google import genai

    if os.environ.get("GOOGLE_GENAI_USE_VERTEXAI") == "true":
        client = genai.Client(
            vertexai=True,
            project=os.environ.get("GOOGLE_CLOUD_PROJECT"),
            location=os.environ.get("GOOGLE_CLOUD_LOCATION"),
        )
    else:
        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
    request_to_make = AMAP_TYPE_TO_GOOGLE_FUNCTION[prompt_blueprint["prompt_template"]["type"]]
    return await request_to_make(client, **function_kwargs)


def vertexai_request(prompt_blueprint: GetPromptTemplateResponse, client_kwargs: dict, function_kwargs: dict):
    if "gemini" in prompt_blueprint["metadata"]["model"]["name"]:
        return google_request(
            prompt_blueprint=prompt_blueprint,
            client_kwargs=client_kwargs,
            function_kwargs=function_kwargs,
        )

    if "claude" in prompt_blueprint["metadata"]["model"]["name"]:
        from anthropic import AnthropicVertex

        client = AnthropicVertex(**client_kwargs)
        if prompt_blueprint["prompt_template"]["type"] == "chat":
            return anthropic_chat_request(client=client, **function_kwargs)
        raise NotImplementedError(
            f"Unsupported prompt template type {prompt_blueprint['prompt_template']['type']}' for Anthropic Vertex AI"
        )

    raise NotImplementedError(
        f"Vertex AI request for model {prompt_blueprint['metadata']['model']['name']} is not implemented yet."
    )


async def avertexai_request(prompt_blueprint: GetPromptTemplateResponse, client_kwargs: dict, function_kwargs: dict):
    if "gemini" in prompt_blueprint["metadata"]["model"]["name"]:
        return await agoogle_request(
            prompt_blueprint=prompt_blueprint,
            client_kwargs=client_kwargs,
            function_kwargs=function_kwargs,
        )

    if "claude" in prompt_blueprint["metadata"]["model"]["name"]:
        from anthropic import AsyncAnthropicVertex

        client = AsyncAnthropicVertex(**client_kwargs)
        if prompt_blueprint["prompt_template"]["type"] == "chat":
            return await aanthropic_chat_request(client=client, **function_kwargs)
        raise NotImplementedError(
            f"Unsupported prompt template type {prompt_blueprint['prompt_template']['type']}' for Anthropic Vertex AI"
        )

    raise NotImplementedError(
        f"Vertex AI request for model {prompt_blueprint['metadata']['model']['name']} is not implemented yet."
    )


def amazon_bedrock_request(prompt_blueprint: GetPromptTemplateResponse, client_kwargs: dict, function_kwargs: dict):
    import boto3

    bedrock_client = boto3.client(
        "bedrock-runtime",
        aws_access_key_id=function_kwargs.pop("aws_access_key", None),
        aws_secret_access_key=function_kwargs.pop("aws_secret_key", None),
        region_name=function_kwargs.pop("aws_region", "us-east-1"),
    )

    stream = function_kwargs.pop("stream", False)

    if stream:
        return bedrock_client.converse_stream(**function_kwargs)
    else:
        return bedrock_client.converse(**function_kwargs)


async def aamazon_bedrock_request(
    prompt_blueprint: GetPromptTemplateResponse, client_kwargs: dict, function_kwargs: dict
):
    import aioboto3

    aws_access_key = function_kwargs.pop("aws_access_key", None)
    aws_secret_key = function_kwargs.pop("aws_secret_key", None)
    aws_region = function_kwargs.pop("aws_region", "us-east-1")

    session_kwargs = {}
    if aws_access_key:
        session_kwargs["aws_access_key_id"] = aws_access_key
    if aws_secret_key:
        session_kwargs["aws_secret_access_key"] = aws_secret_key
    if aws_region:
        session_kwargs["region_name"] = aws_region

    stream = function_kwargs.pop("stream", False)
    session = aioboto3.Session()

    async with session.client("bedrock-runtime", **session_kwargs) as client:
        if stream:
            return await client.converse_stream(**function_kwargs)
        else:
            return await client.converse(**function_kwargs)


def anthropic_bedrock_request(prompt_blueprint: GetPromptTemplateResponse, client_kwargs: dict, function_kwargs: dict):
    from anthropic import AnthropicBedrock

    client = AnthropicBedrock(
        aws_access_key=function_kwargs.pop("aws_access_key", None),
        aws_secret_key=function_kwargs.pop("aws_secret_key", None),
        aws_region=function_kwargs.pop("aws_region", None),
        aws_session_token=function_kwargs.pop("aws_session_token", None),
        base_url=function_kwargs.pop("base_url", None),
        **client_kwargs,
    )
    if prompt_blueprint["prompt_template"]["type"] == "chat":
        return anthropic_chat_request(client=client, **function_kwargs)
    elif prompt_blueprint["prompt_template"]["type"] == "completion":
        return anthropic_completions_request(client=client, **function_kwargs)
    raise NotImplementedError(
        f"Unsupported prompt template type {prompt_blueprint['prompt_template']['type']}' for Anthropic Bedrock"
    )


async def aanthropic_bedrock_request(
    prompt_blueprint: GetPromptTemplateResponse, client_kwargs: dict, function_kwargs: dict
):
    from anthropic import AsyncAnthropicBedrock

    client = AsyncAnthropicBedrock(
        aws_access_key=function_kwargs.pop("aws_access_key", None),
        aws_secret_key=function_kwargs.pop("aws_secret_key", None),
        aws_region=function_kwargs.pop("aws_region", None),
        aws_session_token=function_kwargs.pop("aws_session_token", None),
        base_url=function_kwargs.pop("base_url", None),
        **client_kwargs,
    )
    if prompt_blueprint["prompt_template"]["type"] == "chat":
        return await aanthropic_chat_request(client=client, **function_kwargs)
    elif prompt_blueprint["prompt_template"]["type"] == "completion":
        return await aanthropic_completions_request(client=client, **function_kwargs)
    raise NotImplementedError(
        f"Unsupported prompt template type {prompt_blueprint['prompt_template']['type']}' for Anthropic Bedrock"
    )
