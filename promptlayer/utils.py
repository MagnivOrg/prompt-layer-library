import asyncio
import contextvars
import datetime
import functools
import json
import logging
import os
import sys
import threading
import types
from contextlib import asynccontextmanager
from copy import deepcopy
from enum import Enum
from importlib.metadata import version as get_package_version
from typing import Any, Callable, Coroutine, Dict, List, Optional, Union
from urllib.parse import quote
from uuid import uuid4

import httpx
import requests
import urllib3
import urllib3.util
from ably import AblyRealtime
from ably.types.message import Message
from cachetools import LRUCache
from centrifuge import (
    Client,
    PublicationContext,
    SubscriptionEventHandler,
    SubscriptionState,
)
from opentelemetry import context, trace
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from promptlayer import exceptions as _exceptions
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

# SDK version and HTTP headers
SDK_VERSION = get_package_version("promptlayer")
_PYTHON_VERSION = f"{sys.version_info.major}.{sys.version_info.minor}"
_PROMPTLAYER_USER_AGENT = f"promptlayer-python/{SDK_VERSION} (python {_PYTHON_VERSION})"

WORKFLOW_RUN_URL_TEMPLATE = "{base_url}/workflows/{workflow_id}/run"
WORKFLOW_RUN_CHANNEL_NAME_TEMPLATE = "workflows:{workflow_id}:run:{channel_name_suffix}"
SET_WORKFLOW_COMPLETE_MESSAGE = "SET_WORKFLOW_COMPLETE"
WS_TOKEN_REQUEST_LIBRARY_URL = (
    f"{os.getenv('PROMPTLAYER_BASE_URL', 'https://api.promptlayer.com')}/ws-token-request-library"
)


logger = logging.getLogger(__name__)

# Module-level session for connection pooling (thread-safe initialization)
_requests_session: Optional[requests.Session] = None
_requests_session_lock = threading.Lock()


def _get_requests_session() -> requests.Session:
    """Get or create a module-level requests.Session for connection pooling.

    Using a shared session prevents connection pool exhaustion under high concurrency
    by reusing TCP connections instead of creating new ones for each request.

    Thread-safe: uses double-checked locking pattern.

    The session is configured with connection pool limits that can be customized
    via environment variables for high-traffic production workloads:
    - PROMPTLAYER_POOL_CONNECTIONS: Number of connection pools to cache (default: 100)
    - PROMPTLAYER_POOL_MAXSIZE: Max connections per pool (default: 100)
    """
    global _requests_session
    if _requests_session is None:
        with _requests_session_lock:
            # Double-check after acquiring lock
            if _requests_session is None:
                session = requests.Session()
                # Set User-Agent and SDK version headers for debugging
                default_ua = session.headers.get("User-Agent", "")
                session.headers["User-Agent"] = f"{_PROMPTLAYER_USER_AGENT} {default_ua}".strip()
                session.headers["X-SDK-Version"] = SDK_VERSION
                # Connection pool configuration - can be tuned via environment variables
                pool_connections = int(os.getenv("PROMPTLAYER_POOL_CONNECTIONS", 100))
                pool_maxsize = int(os.getenv("PROMPTLAYER_POOL_MAXSIZE", 100))
                adapter = requests.adapters.HTTPAdapter(
                    pool_connections=pool_connections,
                    pool_maxsize=pool_maxsize,
                    max_retries=0,  # We handle retries at application level via should_retry_error()
                )
                session.mount("https://", adapter)
                session.mount("http://", adapter)
                _requests_session = session
    return _requests_session


# Generic LLM client cache - prevents connection pool exhaustion under high concurrency
# Uses LRUCache to limit memory growth in long-running systems
_llm_client_cache: LRUCache = LRUCache(maxsize=100)
_llm_client_cache_lock = threading.Lock()


def _get_cached_client(cache_key: str, factory: Callable[[], Any]) -> Any:
    """Generic client caching with thread-safe double-checked locking."""
    if cache_key not in _llm_client_cache:
        with _llm_client_cache_lock:
            if cache_key not in _llm_client_cache:
                _llm_client_cache[cache_key] = factory()
    return _llm_client_cache[cache_key]


class FinalOutputCode(Enum):
    OK = "OK"
    EXCEEDS_SIZE_LIMIT = "EXCEEDS_SIZE_LIMIT"


def should_retry_error(exception):
    """Check if an exception should trigger a retry.

    Retries on:
    - Server errors (5xx) and rate limits (429)
    - Connection errors (connection pool exhaustion, network issues)
    """
    # Check for connection errors that should be retried
    if isinstance(
        exception,
        (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            urllib3.exceptions.NewConnectionError,
            urllib3.exceptions.MaxRetryError,
        ),
    ):
        return True

    if hasattr(exception, "response"):
        response = exception.response
        if hasattr(response, "status_code"):
            status_code = response.status_code
            if status_code >= 500 or status_code == 429:
                return True

    if isinstance(
        exception,
        (
            _exceptions.PromptLayerInternalServerError,
            _exceptions.PromptLayerRateLimitError,
        ),
    ):
        return True

    return False


def retry_on_api_error(func):
    return retry(
        retry=retry_if_exception(should_retry_error),
        stop=stop_after_attempt(4),  # 4 total attempts (1 initial + 3 retries)
        wait=wait_exponential(multiplier=2, max=15),  # 2s, 4s, 8s
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )(func)


def _get_http_timeout():
    try:
        return float(os.getenv("PROMPTLAYER_HTTP_TIMEOUT", DEFAULT_HTTP_TIMEOUT))
    except (ValueError, TypeError):
        return DEFAULT_HTTP_TIMEOUT


def _make_httpx_client():
    client = httpx.AsyncClient(timeout=_get_http_timeout())
    default_ua = client.headers.get("user-agent", "")
    client.headers["user-agent"] = f"{_PROMPTLAYER_USER_AGENT} {default_ua}".strip()
    client.headers["X-SDK-Version"] = SDK_VERSION
    return client


def _make_simple_httpx_client():
    client = httpx.Client(timeout=_get_http_timeout())
    default_ua = client.headers.get("user-agent", "")
    client.headers["user-agent"] = f"{_PROMPTLAYER_USER_AGENT} {default_ua}".strip()
    client.headers["X-SDK-Version"] = SDK_VERSION
    return client


def _get_workflow_workflow_id_or_name(workflow_id_or_name, workflow_name):
    # This is backward compatibility code
    if (workflow_id_or_name := (workflow_name if workflow_id_or_name is None else workflow_id_or_name)) is None:
        raise ValueError('Either "workflow_id_or_name" or "workflow_name" must be provided')

    return workflow_id_or_name


async def _get_final_output(
    base_url: str,
    execution_id: int,
    return_all_outputs: bool,
    *,
    headers: Dict[str, str],
) -> Dict[str, Any]:
    async with _make_httpx_client() as client:
        response = await client.get(
            f"{base_url}/workflow-version-execution-results",
            headers=headers,
            params={
                "workflow_version_execution_id": execution_id,
                "return_all_outputs": return_all_outputs,
            },
        )
        if response.status_code != 200:
            raise_on_bad_response(
                response,
                "PromptLayer had the following error while getting workflow results",
            )
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
        if response.status_code != 200:
            raise_on_bad_response(response, "PromptLayer had the following error while resolving workflow")

        return response.json()["workflow"]["id"]


async def _get_ably_token(base_url: str, channel_name, authentication_headers):
    try:
        async with _make_httpx_client() as client:
            response = await client.post(
                f"{base_url}/ws-token-request-library",
                headers=authentication_headers,
                params={"capability": channel_name},
            )
            if response.status_code != 201:
                raise_on_bad_response(
                    response,
                    "PromptLayer had the following error while getting WebSocket token",
                )
            return response.json()
    except Exception as ex:
        error_message = f"Failed to get WebSocket token: {ex}"
        logger.exception(error_message)
        if RERAISE_ORIGINAL_EXCEPTION:
            raise
        else:
            raise _exceptions.PromptLayerAPIError(error_message, response=None, body=None) from ex


def _make_message_listener(base_url: str, results_future, execution_id_future, return_all_outputs, headers):
    # We need this function to be mocked by unittests
    async def message_listener(message: Message):
        if results_future.cancelled() or message.name != SET_WORKFLOW_COMPLETE_MESSAGE:
            return  # TODO(dmu) LOW: Do we really need this check?

        execution_id = await asyncio.wait_for(execution_id_future, _get_http_timeout() * 1.1)
        message_data = json.loads(message.data)
        if message_data["workflow_version_execution_id"] != execution_id:
            return

        if (result_code := message_data.get("result_code")) in (
            FinalOutputCode.OK.value,
            None,
        ):
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
            if response.status_code != 201:
                raise_on_bad_response(
                    response,
                    "PromptLayer had the following error while running your workflow",
                )

            result = response.json()
            if warning := result.get("warning"):
                logger.warning(f"{warning}")
    except Exception as ex:
        error_message = f"Failed to run workflow: {str(ex)}"
        logger.exception(error_message)
        if RERAISE_ORIGINAL_EXCEPTION:
            raise
        else:
            raise _exceptions.PromptLayerAPIError(error_message, response=None, body=None) from ex

    return result.get("workflow_version_execution_id")


async def _wait_for_workflow_completion(channel, results_future, message_listener, timeout):
    # We need this function for mocking in unittests
    try:
        return await asyncio.wait_for(results_future, timeout)
    except asyncio.TimeoutError:
        raise _exceptions.PromptLayerAPITimeoutError(
            "Workflow execution did not complete properly", response=None, body=None
        )
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


@retry_on_api_error
async def arun_workflow_request(
    *,
    api_key: str,
    base_url: str,
    throw_on_error: bool,
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
        base_url,
        _get_workflow_workflow_id_or_name(workflow_id_or_name, workflow_name),
        headers,
    )
    channel_name_suffix = _make_channel_name_suffix()
    channel_name = WORKFLOW_RUN_CHANNEL_NAME_TEMPLATE.format(
        workflow_id=workflow_id, channel_name_suffix=channel_name_suffix
    )
    ably_token = await _get_ably_token(base_url, channel_name, headers)
    token = ably_token["token_details"]["token"]

    execution_id_future = asyncio.Future[int]()

    if ably_token.get("messaging_backend") == "centrifugo":
        ws_scheme = "wss" if urllib3.util.parse_url(base_url).scheme == "https" else "ws"
        address = urllib3.util.parse_url(base_url)._replace(scheme=ws_scheme, path="/connection/websocket").url
        async with centrifugo_client(address, token) as client:
            results_future = asyncio.Future[dict[str, Any]]()
            async with centrifugo_subscription(
                client,
                channel_name,
                _make_message_listener(
                    base_url,
                    results_future,
                    execution_id_future,
                    return_all_outputs,
                    headers,
                ),
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
        request_response = _get_requests_session().post(
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
                request_response,
                "WARNING: While logging your request PromptLayer had the following issue",
            )
        elif request_response.status_code != 200:
            warn_on_bad_response(
                request_response,
                "WARNING: While logging your request PromptLayer had the following error",
            )
    except Exception as e:
        logger.warning(f"While logging your request PromptLayer had the following error: {e}")
    if request_response is not None and return_pl_id:
        return request_response.json().get("request_id")


@retry_on_api_error
def track_request(base_url: str, throw_on_error: bool, **body):
    try:
        response = _get_requests_session().post(
            f"{base_url}/track-request",
            json=body,
        )
        if response.status_code != 200:
            if throw_on_error:
                raise_on_bad_response(
                    response,
                    "PromptLayer had the following error while tracking your request",
                )
            else:
                warn_on_bad_response(
                    response,
                    f"PromptLayer had the following error while tracking your request: {response.text}",
                )
        return response.json()
    except requests.exceptions.RequestException as e:
        if throw_on_error:
            raise _exceptions.PromptLayerAPIConnectionError(
                f"PromptLayer had the following error while tracking your request: {e}",
                response=None,
                body=None,
            ) from e
        logger.warning(f"While logging your request PromptLayer had the following error: {e}")
        return {}


@retry_on_api_error
async def atrack_request(base_url: str, throw_on_error: bool, **body: Any) -> Dict[str, Any]:
    try:
        async with _make_httpx_client() as client:
            response = await client.post(
                f"{base_url}/track-request",
                json=body,
            )
            if response.status_code != 200:
                if throw_on_error:
                    raise_on_bad_response(
                        response,
                        "PromptLayer had the following error while tracking your request",
                    )
                else:
                    warn_on_bad_response(
                        response,
                        f"PromptLayer had the following error while tracking your request: {response.text}",
                    )
        return response.json()
    except httpx.RequestError as e:
        if throw_on_error:
            raise _exceptions.PromptLayerAPIConnectionError(
                f"PromptLayer had the following error while tracking your request: {e}",
                response=None,
                body=None,
            ) from e
        logger.warning(f"While logging your request PromptLayer had the following error: {e}")
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


@retry_on_api_error
def promptlayer_get_prompt(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    prompt_name,
    version: int = None,
    label: str = None,
):
    """
    Get a prompt from the PromptLayer library
    version: version of the prompt to get, None for latest
    label: The specific label of a prompt you want to get. Setting this will supercede version
    """
    try:
        request_response = _get_requests_session().get(
            f"{base_url}/library-get-prompt-template",
            headers={"X-API-KEY": api_key},
            params={"prompt_name": prompt_name, "version": version, "label": label},
        )
    except Exception as e:
        if throw_on_error:
            raise _exceptions.PromptLayerAPIError(
                f"PromptLayer had the following error while getting your prompt: {e}",
                response=None,
                body=None,
            ) from e
        logger.warning(f"PromptLayer had the following error while getting your prompt: {e}")
        return None
    if request_response.status_code != 200:
        if throw_on_error:
            raise_on_bad_response(
                request_response,
                "PromptLayer had the following error while getting your prompt",
            )
        else:
            warn_on_bad_response(
                request_response,
                "WARNING: PromptLayer had the following error while getting your prompt",
            )
            return None

    return request_response.json()


@retry_on_api_error
def promptlayer_publish_prompt(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    prompt_name,
    prompt_template,
    commit_message,
    tags,
    metadata=None,
):
    try:
        request_response = _get_requests_session().post(
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
        if throw_on_error:
            raise _exceptions.PromptLayerAPIError(
                f"PromptLayer had the following error while publishing your prompt: {e}",
                response=None,
                body=None,
            ) from e
        logger.warning(f"PromptLayer had the following error while publishing your prompt: {e}")
        return False
    if request_response.status_code != 200:
        if throw_on_error:
            raise_on_bad_response(
                request_response,
                "PromptLayer had the following error while publishing your prompt",
            )
        else:
            warn_on_bad_response(
                request_response,
                "WARNING: PromptLayer had the following error while publishing your prompt",
            )
            return False
    return True


@retry_on_api_error
def promptlayer_track_prompt(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    request_id,
    prompt_name,
    input_variables,
    version,
    label,
):
    try:
        request_response = _get_requests_session().post(
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
            if throw_on_error:
                raise_on_bad_response(
                    request_response,
                    "While tracking your prompt PromptLayer had the following error",
                )
            else:
                warn_on_bad_response(
                    request_response,
                    "WARNING: While tracking your prompt PromptLayer had the following error",
                )
                return False
    except Exception as e:
        if throw_on_error:
            raise _exceptions.PromptLayerAPIError(
                f"While tracking your prompt PromptLayer had the following error: {e}",
                response=None,
                body=None,
            ) from e
        logger.warning(f"While tracking your prompt PromptLayer had the following error: {e}")
        return False
    return True


@retry_on_api_error
async def apromptlayer_track_prompt(
    api_key: str,
    base_url: str,
    request_id: str,
    prompt_name: str,
    input_variables: Dict[str, Any],
    version: Optional[int] = None,
    label: Optional[str] = None,
    throw_on_error: bool = True,
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

        if response.status_code != 200:
            if throw_on_error:
                raise_on_bad_response(
                    response,
                    "While tracking your prompt, PromptLayer had the following error",
                )
            else:
                warn_on_bad_response(
                    response,
                    "WARNING: While tracking your prompt, PromptLayer had the following error",
                )
                return False
    except httpx.RequestError as e:
        if throw_on_error:
            raise _exceptions.PromptLayerAPIConnectionError(
                f"While tracking your prompt PromptLayer had the following error: {e}",
                response=None,
                body=None,
            ) from e
        logger.warning(f"While tracking your prompt PromptLayer had the following error: {e}")
        return False

    return True


@retry_on_api_error
def promptlayer_track_metadata(api_key: str, base_url: str, throw_on_error: bool, request_id, metadata):
    try:
        request_response = _get_requests_session().post(
            f"{base_url}/library-track-metadata",
            json={
                "request_id": request_id,
                "metadata": metadata,
                "api_key": api_key,
            },
        )
        if request_response.status_code != 200:
            if throw_on_error:
                raise_on_bad_response(
                    request_response,
                    "While tracking your metadata PromptLayer had the following error",
                )
            else:
                warn_on_bad_response(
                    request_response,
                    "WARNING: While tracking your metadata PromptLayer had the following error",
                )
                return False
    except Exception as e:
        if throw_on_error:
            raise _exceptions.PromptLayerAPIError(
                f"While tracking your metadata PromptLayer had the following error: {e}",
                response=None,
                body=None,
            ) from e
        logger.warning(f"While tracking your metadata PromptLayer had the following error: {e}")
        return False
    return True


@retry_on_api_error
async def apromptlayer_track_metadata(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    request_id: str,
    metadata: Dict[str, Any],
) -> bool:
    url = f"{base_url}/library-track-metadata"
    payload = {
        "request_id": request_id,
        "metadata": metadata,
        "api_key": api_key,
    }
    try:
        async with _make_httpx_client() as client:
            response = await client.post(url, json=payload)

        if response.status_code != 200:
            if throw_on_error:
                raise_on_bad_response(
                    response,
                    "While tracking your metadata, PromptLayer had the following error",
                )
            else:
                warn_on_bad_response(
                    response,
                    "WARNING: While tracking your metadata, PromptLayer had the following error",
                )
                return False
    except httpx.RequestError as e:
        if throw_on_error:
            raise _exceptions.PromptLayerAPIConnectionError(
                f"While tracking your metadata PromptLayer had the following error: {e}",
                response=None,
                body=None,
            ) from e
        logger.warning(f"While tracking your metadata PromptLayer had the following error: {e}")
        return False

    return True


@retry_on_api_error
def promptlayer_track_score(api_key: str, base_url: str, throw_on_error: bool, request_id, score, score_name):
    try:
        data = {"request_id": request_id, "score": score, "api_key": api_key}
        if score_name is not None:
            data["name"] = score_name
        request_response = _get_requests_session().post(
            f"{base_url}/library-track-score",
            json=data,
        )
        if request_response.status_code != 200:
            if throw_on_error:
                raise_on_bad_response(
                    request_response,
                    "While tracking your score PromptLayer had the following error",
                )
            else:
                warn_on_bad_response(
                    request_response,
                    "WARNING: While tracking your score PromptLayer had the following error",
                )
                return False
    except Exception as e:
        if throw_on_error:
            raise _exceptions.PromptLayerAPIError(
                f"While tracking your score PromptLayer had the following error: {e}",
                response=None,
                body=None,
            ) from e
        logger.warning(f"While tracking your score PromptLayer had the following error: {e}")
        return False
    return True


@retry_on_api_error
async def apromptlayer_track_score(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
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

        if response.status_code != 200:
            if throw_on_error:
                raise_on_bad_response(
                    response,
                    "While tracking your score, PromptLayer had the following error",
                )
            else:
                warn_on_bad_response(
                    response,
                    "WARNING: While tracking your score, PromptLayer had the following error",
                )
                return False
    except httpx.RequestError as e:
        if throw_on_error:
            raise _exceptions.PromptLayerAPIConnectionError(
                f"PromptLayer had the following error while tracking your score: {str(e)}",
                response=None,
                body=None,
            ) from e
        logger.warning(f"While tracking your score PromptLayer had the following error: {str(e)}")
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
            return GeneratorProxy(
                self.generator.text_stream,
                self.api_request_arugments,
                self.api_key,
                self.base_url,
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
            logger.warning(f"{main_message}: {request_response.json().get('message')}")
        except json.JSONDecodeError:
            logger.warning(f"{main_message}: {request_response}")
    else:
        logger.warning(f"{main_message}: {request_response}")


def raise_on_bad_response(request_response, main_message):
    """Raise an appropriate exception based on the HTTP status code."""
    status_code = getattr(request_response, "status_code", None)

    body = None
    error_detail = None
    if hasattr(request_response, "json"):
        try:
            body = request_response.json()
            error_detail = body.get("message") or body.get("error") or body.get("detail")
        except (json.JSONDecodeError, AttributeError):
            body = getattr(request_response, "text", str(request_response))
            error_detail = body
    else:
        body = str(request_response)
        error_detail = body

    if error_detail:
        err_msg = f"{main_message}: {error_detail}"
    else:
        err_msg = main_message

    if status_code == 400:
        raise _exceptions.PromptLayerBadRequestError(err_msg, response=request_response, body=body)

    if status_code == 401:
        raise _exceptions.PromptLayerAuthenticationError(err_msg, response=request_response, body=body)

    if status_code == 403:
        raise _exceptions.PromptLayerPermissionDeniedError(err_msg, response=request_response, body=body)

    if status_code == 404:
        raise _exceptions.PromptLayerNotFoundError(err_msg, response=request_response, body=body)

    if status_code == 409:
        raise _exceptions.PromptLayerConflictError(err_msg, response=request_response, body=body)

    if status_code == 422:
        raise _exceptions.PromptLayerUnprocessableEntityError(err_msg, response=request_response, body=body)

    if status_code == 429:
        raise _exceptions.PromptLayerRateLimitError(err_msg, response=request_response, body=body)

    if status_code and status_code >= 500:
        raise _exceptions.PromptLayerInternalServerError(err_msg, response=request_response, body=body)

    raise _exceptions.PromptLayerAPIStatusError(err_msg, response=request_response, body=body)


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


@retry_on_api_error
def promptlayer_create_group(api_key: str, base_url: str, throw_on_error: bool):
    try:
        request_response = _get_requests_session().post(
            f"{base_url}/create-group",
            json={
                "api_key": api_key,
            },
        )
        if request_response.status_code != 200:
            if throw_on_error:
                raise_on_bad_response(
                    request_response,
                    "While creating your group PromptLayer had the following error",
                )
            else:
                warn_on_bad_response(
                    request_response,
                    "WARNING: While creating your group PromptLayer had the following error",
                )
                return False
    except requests.exceptions.RequestException as e:
        if throw_on_error:
            raise _exceptions.PromptLayerAPIConnectionError(
                f"PromptLayer had the following error while creating your group: {e}",
                response=None,
                body=None,
            ) from e
        logger.warning(f"While creating your group PromptLayer had the following error: {e}")
        return False
    return request_response.json()["id"]


@retry_on_api_error
async def apromptlayer_create_group(api_key: str, base_url: str, throw_on_error: bool):
    try:
        async with _make_httpx_client() as client:
            response = await client.post(
                f"{base_url}/create-group",
                json={
                    "api_key": api_key,
                },
            )

        if response.status_code != 200:
            if throw_on_error:
                raise_on_bad_response(
                    response,
                    "While creating your group, PromptLayer had the following error",
                )
            else:
                warn_on_bad_response(
                    response,
                    "WARNING: While creating your group, PromptLayer had the following error",
                )
                return False
        return response.json()["id"]
    except httpx.RequestError as e:
        if throw_on_error:
            raise _exceptions.PromptLayerAPIConnectionError(
                f"PromptLayer had the following error while creating your group: {str(e)}",
                response=None,
                body=None,
            ) from e
        logger.warning(f"While creating your group PromptLayer had the following error: {e}")
        return False


@retry_on_api_error
def promptlayer_track_group(api_key: str, base_url: str, throw_on_error: bool, request_id, group_id):
    try:
        request_response = _get_requests_session().post(
            f"{base_url}/track-group",
            json={
                "api_key": api_key,
                "request_id": request_id,
                "group_id": group_id,
            },
        )
        if request_response.status_code != 200:
            if throw_on_error:
                raise_on_bad_response(
                    request_response,
                    "While tracking your group PromptLayer had the following error",
                )
            else:
                warn_on_bad_response(
                    request_response,
                    "WARNING: While tracking your group PromptLayer had the following error",
                )
                return False
    except requests.exceptions.RequestException as e:
        if throw_on_error:
            raise _exceptions.PromptLayerAPIConnectionError(
                f"PromptLayer had the following error while tracking your group: {e}",
                response=None,
                body=None,
            ) from e
        logger.warning(f"While tracking your group PromptLayer had the following error: {e}")
        return False
    return True


@retry_on_api_error
async def apromptlayer_track_group(api_key: str, base_url: str, throw_on_error: bool, request_id, group_id):
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

        if response.status_code != 200:
            if throw_on_error:
                raise_on_bad_response(
                    response,
                    "While tracking your group, PromptLayer had the following error",
                )
            else:
                warn_on_bad_response(
                    response,
                    "WARNING: While tracking your group, PromptLayer had the following error",
                )
                return False
    except httpx.RequestError as e:
        if throw_on_error:
            raise _exceptions.PromptLayerAPIConnectionError(
                f"PromptLayer had the following error while tracking your group: {str(e)}",
                response=None,
                body=None,
            ) from e
        logger.warning(f"While tracking your group PromptLayer had the following error: {e}")
        return False

    return True


@retry_on_api_error
def get_prompt_template(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    prompt_name: str,
    params: Union[GetPromptTemplate, None] = None,
) -> GetPromptTemplateResponse:
    try:
        json_body = {"api_key": api_key}
        if params:
            json_body = {**json_body, **params}
        response = _get_requests_session().post(
            f"{base_url}/prompt-templates/{quote(prompt_name, safe='')}",
            headers={"X-API-KEY": api_key},
            json=json_body,
        )
        if response.status_code != 200:
            if throw_on_error:
                raise_on_bad_response(
                    response,
                    "PromptLayer had the following error while getting your prompt template",
                )
            else:
                warn_on_bad_response(
                    response,
                    "WARNING: PromptLayer had the following error while getting your prompt template",
                )
                return None

        return response.json()
    except requests.exceptions.ConnectionError as e:
        err_msg = f"PromptLayer had the following error while getting your prompt template: {e}"
        if throw_on_error:
            raise _exceptions.PromptLayerAPIConnectionError(err_msg, response=None, body=None) from e
        logger.warning(err_msg)
        return None
    except requests.exceptions.Timeout as e:
        err_msg = f"PromptLayer had the following error while getting your prompt template: {e}"
        if throw_on_error:
            raise _exceptions.PromptLayerAPITimeoutError(err_msg, response=None, body=None) from e
        logger.warning(err_msg)
        return None
    except requests.exceptions.RequestException as e:
        err_msg = f"PromptLayer had the following error while getting your prompt template: {e}"
        if throw_on_error:
            raise _exceptions.PromptLayerError(err_msg, response=None, body=None) from e
        logger.warning(err_msg)
        return None


@retry_on_api_error
async def aget_prompt_template(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    prompt_name: str,
    params: Union[GetPromptTemplate, None] = None,
) -> GetPromptTemplateResponse:
    try:
        json_body = {"api_key": api_key}
        if params:
            json_body.update(params)
        async with _make_httpx_client() as client:
            response = await client.post(
                f"{base_url}/prompt-templates/{quote(prompt_name, safe='')}",
                headers={"X-API-KEY": api_key},
                json=json_body,
            )
            if response.status_code != 200:
                if throw_on_error:
                    raise_on_bad_response(
                        response,
                        "PromptLayer had the following error while getting your prompt template",
                    )
                else:
                    warn_on_bad_response(
                        response,
                        "WARNING: While getting your prompt template PromptLayer had the following error",
                    )
                    return None
        return response.json()
    except (httpx.ConnectError, httpx.NetworkError) as e:
        err_msg = f"PromptLayer had the following error while getting your prompt template: {str(e)}"
        if throw_on_error:
            raise _exceptions.PromptLayerAPIConnectionError(err_msg, response=None, body=None) from e
        logger.warning(err_msg)
        return None
    except httpx.TimeoutException as e:
        err_msg = f"PromptLayer had the following error while getting your prompt template: {str(e)}"
        if throw_on_error:
            raise _exceptions.PromptLayerAPITimeoutError(err_msg, response=None, body=None) from e
        logger.warning(err_msg)
        return None
    except httpx.RequestError as e:
        err_msg = f"PromptLayer had the following error while getting your prompt template: {str(e)}"
        if throw_on_error:
            raise _exceptions.PromptLayerAPIConnectionError(err_msg, response=None, body=None) from e
        logger.warning(err_msg)
        return None


@retry_on_api_error
def publish_prompt_template(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    body: PublishPromptTemplate,
) -> PublishPromptTemplateResponse:
    try:
        response = _get_requests_session().post(
            f"{base_url}/rest/prompt-templates",
            headers={"X-API-KEY": api_key},
            json={
                "prompt_template": {**body},
                "prompt_version": {**body},
                "release_labels": body.get("release_labels"),
            },
        )
        if response.status_code == 400:
            if throw_on_error:
                raise_on_bad_response(
                    response,
                    "PromptLayer had the following error while publishing your prompt template",
                )
            else:
                warn_on_bad_response(
                    response,
                    "WARNING: PromptLayer had the following error while publishing your prompt template",
                )
                return None
        return response.json()
    except requests.exceptions.RequestException as e:
        if throw_on_error:
            raise _exceptions.PromptLayerAPIConnectionError(
                f"PromptLayer had the following error while publishing your prompt template: {e}",
                response=None,
                body=None,
            ) from e
        logger.warning(f"PromptLayer had the following error while publishing your prompt template: {e}")
        return None


@retry_on_api_error
async def apublish_prompt_template(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
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

        if response.status_code == 400 or response.status_code != 201:
            if throw_on_error:
                raise_on_bad_response(
                    response,
                    "PromptLayer had the following error while publishing your prompt template",
                )
            else:
                warn_on_bad_response(
                    response,
                    "WARNING: PromptLayer had the following error while publishing your prompt template",
                )
                return None
        return response.json()
    except httpx.RequestError as e:
        if throw_on_error:
            raise _exceptions.PromptLayerAPIConnectionError(
                f"PromptLayer had the following error while publishing your prompt template: {str(e)}",
                response=None,
                body=None,
            ) from e
        logger.warning(f"PromptLayer had the following error while publishing your prompt template: {e}")
        return None


@retry_on_api_error
def get_all_prompt_templates(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    page: int = 1,
    per_page: int = 30,
    label: str = None,
) -> List[ListPromptTemplateResponse]:
    try:
        params = {"page": page, "per_page": per_page}
        if label:
            params["label"] = label
        response = _get_requests_session().get(
            f"{base_url}/prompt-templates",
            headers={"X-API-KEY": api_key},
            params=params,
        )
        if response.status_code != 200:
            if throw_on_error:
                raise_on_bad_response(
                    response,
                    "PromptLayer had the following error while getting all your prompt templates",
                )
            else:
                warn_on_bad_response(
                    response,
                    "WARNING: PromptLayer had the following error while getting all your prompt templates",
                )
                return []
        items = response.json().get("items", [])
        return items
    except requests.exceptions.RequestException as e:
        if throw_on_error:
            raise _exceptions.PromptLayerAPIConnectionError(
                f"PromptLayer had the following error while getting all your prompt templates: {e}",
                response=None,
                body=None,
            ) from e
        logger.warning(f"PromptLayer had the following error while getting all your prompt templates: {e}")
        return []


@retry_on_api_error
async def aget_all_prompt_templates(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    page: int = 1,
    per_page: int = 30,
    label: str = None,
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

        if response.status_code != 200:
            if throw_on_error:
                raise_on_bad_response(
                    response,
                    "PromptLayer had the following error while getting all your prompt templates",
                )
            else:
                warn_on_bad_response(
                    response,
                    "WARNING: PromptLayer had the following error while getting all your prompt templates",
                )
                return []
        items = response.json().get("items", [])
        return items
    except httpx.RequestError as e:
        if throw_on_error:
            raise _exceptions.PromptLayerAPIConnectionError(
                f"PromptLayer had the following error while getting all your prompt templates: {str(e)}",
                response=None,
                body=None,
            ) from e
        logger.warning(f"PromptLayer had the following error while getting all your prompt templates: {e}")
        return []


def openai_chat_request(client, **kwargs):
    return client.chat.completions.create(**kwargs)


def openai_completions_request(client, **kwargs):
    return client.completions.create(**kwargs)


MAP_TYPE_TO_OPENAI_FUNCTION = {
    "chat": openai_chat_request,
    "completion": openai_completions_request,
}


def openai_request(
    prompt_blueprint: GetPromptTemplateResponse,
    client_kwargs: dict,
    function_kwargs: dict,
):
    from openai import OpenAI

    cache_key = f"openai:{client_kwargs.get('api_key', '')}:{client_kwargs.get('base_url', '')}"
    client = _get_cached_client(cache_key, lambda: OpenAI(**client_kwargs))
    api_type = prompt_blueprint["metadata"]["model"].get("api_type", "chat-completions")

    if api_type is None:
        api_type = "chat-completions"

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


async def aopenai_request(
    prompt_blueprint: GetPromptTemplateResponse,
    client_kwargs: dict,
    function_kwargs: dict,
):
    from openai import AsyncOpenAI

    cache_key = f"async_openai:{client_kwargs.get('api_key', '')}:{client_kwargs.get('base_url', '')}"
    client = _get_cached_client(cache_key, lambda: AsyncOpenAI(**client_kwargs))
    api_type = prompt_blueprint["metadata"]["model"].get("api_type", "chat-completions")

    if api_type == "chat-completions":
        request_to_make = AMAP_TYPE_TO_OPENAI_FUNCTION[prompt_blueprint["prompt_template"]["type"]]
        return await request_to_make(client, **function_kwargs)
    else:
        return await client.responses.create(**function_kwargs)


def azure_openai_request(
    prompt_blueprint: GetPromptTemplateResponse,
    client_kwargs: dict,
    function_kwargs: dict,
):
    from openai import AzureOpenAI

    azure_endpoint = client_kwargs.pop("base_url", None)
    cache_key = f"azure_openai:{client_kwargs.get('api_key', '')}:{azure_endpoint or ''}"
    client = _get_cached_client(cache_key, lambda: AzureOpenAI(azure_endpoint=azure_endpoint, **client_kwargs))
    api_type = prompt_blueprint["metadata"]["model"].get("api_type", "chat-completions")

    if api_type == "chat-completions":
        request_to_make = MAP_TYPE_TO_OPENAI_FUNCTION[prompt_blueprint["prompt_template"]["type"]]
        return request_to_make(client, **function_kwargs)
    else:
        return client.responses.create(**function_kwargs)


async def aazure_openai_request(
    prompt_blueprint: GetPromptTemplateResponse,
    client_kwargs: dict,
    function_kwargs: dict,
):
    from openai import AsyncAzureOpenAI

    azure_endpoint = client_kwargs.pop("base_url", None)
    cache_key = f"async_azure_openai:{client_kwargs.get('api_key', '')}:{azure_endpoint or ''}"
    client = _get_cached_client(
        cache_key,
        lambda: AsyncAzureOpenAI(azure_endpoint=azure_endpoint, **client_kwargs),
    )
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


def anthropic_request(
    prompt_blueprint: GetPromptTemplateResponse,
    client_kwargs: dict,
    function_kwargs: dict,
):
    from anthropic import Anthropic

    cache_key = f"anthropic:{client_kwargs.get('api_key', '')}:{client_kwargs.get('base_url', '')}"
    client = _get_cached_client(cache_key, lambda: Anthropic(**client_kwargs))
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


async def aanthropic_request(
    prompt_blueprint: GetPromptTemplateResponse,
    client_kwargs: dict,
    function_kwargs: dict,
):
    from anthropic import AsyncAnthropic

    cache_key = f"async_anthropic:{client_kwargs.get('api_key', '')}:{client_kwargs.get('base_url', '')}"
    client = _get_cached_client(cache_key, lambda: AsyncAnthropic(**client_kwargs))
    request_to_make = AMAP_TYPE_TO_ANTHROPIC_FUNCTION[prompt_blueprint["prompt_template"]["type"]]
    return await request_to_make(client, **function_kwargs)


# do not remove! This is used in the langchain integration.
def get_api_key():
    # raise an error if the api key is not set
    api_key = os.environ.get("PROMPTLAYER_API_KEY")
    if not api_key:
        raise _exceptions.PromptLayerAuthenticationError(
            "Please set your PROMPTLAYER_API_KEY environment variable or set API KEY in code using 'promptlayer.api_key = <your_api_key>'",
            response=None,
            body=None,
        )
    return api_key


@retry_on_api_error
def util_log_request(api_key: str, base_url: str, throw_on_error: bool, **kwargs) -> Union[RequestLog, None]:
    try:
        response = _get_requests_session().post(
            f"{base_url}/log-request",
            headers={"X-API-KEY": api_key},
            json=kwargs,
        )
        if response.status_code != 201:
            if throw_on_error:
                raise_on_bad_response(
                    response,
                    "PromptLayer had the following error while logging your request",
                )
            else:
                warn_on_bad_response(
                    response,
                    "WARNING: While logging your request PromptLayer had the following error",
                )
                return None
        return response.json()
    except Exception as e:
        if throw_on_error:
            raise _exceptions.PromptLayerAPIError(
                f"While logging your request PromptLayer had the following error: {e}",
                response=None,
                body=None,
            ) from e
        logger.warning(f"While tracking your prompt PromptLayer had the following error: {e}")
        return None


@retry_on_api_error
async def autil_log_request(api_key: str, base_url: str, throw_on_error: bool, **kwargs) -> Union[RequestLog, None]:
    try:
        async with _make_httpx_client() as client:
            response = await client.post(
                f"{base_url}/log-request",
                headers={"X-API-KEY": api_key},
                json=kwargs,
            )
        if response.status_code != 201:
            if throw_on_error:
                raise_on_bad_response(
                    response,
                    "PromptLayer had the following error while logging your request",
                )
            else:
                warn_on_bad_response(
                    response,
                    "WARNING: While logging your request PromptLayer had the following error",
                )
                return None
        return response.json()
    except Exception as e:
        if throw_on_error:
            raise _exceptions.PromptLayerAPIError(
                f"While logging your request PromptLayer had the following error: {e}",
                response=None,
                body=None,
            ) from e
        logger.warning(f"While tracking your prompt PromptLayer had the following error: {e}")
        return None


def mistral_request(
    prompt_blueprint: GetPromptTemplateResponse,
    client_kwargs: dict,
    function_kwargs: dict,
):
    from mistralai import Mistral

    api_key = os.environ.get("MISTRAL_API_KEY")
    cache_key = f"mistral:{api_key or ''}"
    client = _get_cached_client(cache_key, lambda: Mistral(api_key=api_key, client=_make_simple_httpx_client()))
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

    api_key = os.environ.get("MISTRAL_API_KEY")
    cache_key = f"async_mistral:{api_key or ''}"
    client = _get_cached_client(cache_key, lambda: Mistral(api_key=api_key, async_client=_make_httpx_client()))
    if "stream" in function_kwargs and function_kwargs["stream"]:
        return await client.chat.stream_async(**function_kwargs)
    return await client.chat.complete_async(**function_kwargs)


class _GoogleStreamWrapper:
    """Wrapper to keep Google client alive during streaming."""

    def __init__(self, stream_generator, client):
        self._stream = stream_generator
        self._client = client  # Keep client alive

    def __iter__(self):
        return self._stream.__iter__()

    def __next__(self):
        return next(self._stream)

    def __aiter__(self):
        return self._stream.__aiter__()

    async def __anext__(self):
        return await self._stream.__anext__()


def google_chat_request(client, **kwargs):
    from google.genai.chats import Content

    stream = kwargs.pop("stream", False)
    model = kwargs.get("model", "gemini-2.0-flash")
    history = [Content(**item) for item in kwargs.get("history", [])]
    generation_config = kwargs.get("generation_config", {})
    chat = client.chats.create(model=model, history=history, config=generation_config)
    last_message = history[-1].parts if history else ""
    if stream:
        stream_gen = chat.send_message_stream(message=last_message)
        return _GoogleStreamWrapper(stream_gen, client)
    return chat.send_message(message=last_message)


def google_completions_request(client, **kwargs):
    config = kwargs.pop("generation_config", {})
    model = kwargs.get("model", "gemini-2.0-flash")
    contents = kwargs.get("contents", [])
    stream = kwargs.pop("stream", False)
    if stream:
        stream_gen = client.models.generate_content_stream(model=model, contents=contents, config=config)
        return _GoogleStreamWrapper(stream_gen, client)
    return client.models.generate_content(model=model, contents=contents, config=config)


MAP_TYPE_TO_GOOGLE_FUNCTION = {
    "chat": google_chat_request,
    "completion": google_completions_request,
}


def google_request(
    prompt_blueprint: GetPromptTemplateResponse,
    client_kwargs: dict,
    function_kwargs: dict,
):
    from google import genai

    use_vertexai = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI") == "true"
    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    location = os.environ.get("GOOGLE_CLOUD_LOCATION")
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

    cache_key = f"google_genai:{api_key or ''}:{project or ''}:{location or ''}"

    def create_client():
        if use_vertexai:
            return genai.Client(vertexai=True, project=project, location=location)
        return genai.Client(api_key=api_key)

    client = _get_cached_client(cache_key, create_client)
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
        stream_gen = await chat.send_message_stream(message=last_message)
        return _GoogleStreamWrapper(stream_gen, client)
    return await chat.send_message(message=last_message)


async def agoogle_completions_request(client, **kwargs):
    config = kwargs.pop("generation_config", {})
    model = kwargs.get("model", "gemini-2.0-flash")
    contents = kwargs.get("contents", [])
    stream = kwargs.pop("stream", False)
    if stream:
        stream_gen = await client.aio.models.generate_content_stream(model=model, contents=contents, config=config)
        return _GoogleStreamWrapper(stream_gen, client)
    return await client.aio.models.generate_content(model=model, contents=contents, config=config)


AMAP_TYPE_TO_GOOGLE_FUNCTION = {
    "chat": agoogle_chat_request,
    "completion": agoogle_completions_request,
}


async def agoogle_request(
    prompt_blueprint: GetPromptTemplateResponse,
    client_kwargs: dict,
    function_kwargs: dict,
):
    from google import genai

    use_vertexai = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI") == "true"
    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    location = os.environ.get("GOOGLE_CLOUD_LOCATION")
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

    cache_key = f"async_google_genai:{api_key or ''}:{project or ''}:{location or ''}"

    def create_client():
        if use_vertexai:
            return genai.Client(vertexai=True, project=project, location=location)
        return genai.Client(api_key=api_key)

    client = _get_cached_client(cache_key, create_client)
    request_to_make = AMAP_TYPE_TO_GOOGLE_FUNCTION[prompt_blueprint["prompt_template"]["type"]]
    return await request_to_make(client, **function_kwargs)


def vertexai_request(
    prompt_blueprint: GetPromptTemplateResponse,
    client_kwargs: dict,
    function_kwargs: dict,
):
    if "gemini" in prompt_blueprint["metadata"]["model"]["name"]:
        return google_request(
            prompt_blueprint=prompt_blueprint,
            client_kwargs=client_kwargs,
            function_kwargs=function_kwargs,
        )

    if "claude" in prompt_blueprint["metadata"]["model"]["name"]:
        from anthropic import AnthropicVertex

        cache_key = f"anthropic_vertex:{client_kwargs.get('project_id', '')}:{client_kwargs.get('region', '')}"
        client = _get_cached_client(cache_key, lambda: AnthropicVertex(**client_kwargs))
        if prompt_blueprint["prompt_template"]["type"] == "chat":
            return anthropic_chat_request(client=client, **function_kwargs)
        raise NotImplementedError(
            f"Unsupported prompt template type {prompt_blueprint['prompt_template']['type']}' for Anthropic Vertex AI"
        )

    raise NotImplementedError(
        f"Vertex AI request for model {prompt_blueprint['metadata']['model']['name']} is not implemented yet."
    )


async def avertexai_request(
    prompt_blueprint: GetPromptTemplateResponse,
    client_kwargs: dict,
    function_kwargs: dict,
):
    if "gemini" in prompt_blueprint["metadata"]["model"]["name"]:
        return await agoogle_request(
            prompt_blueprint=prompt_blueprint,
            client_kwargs=client_kwargs,
            function_kwargs=function_kwargs,
        )

    if "claude" in prompt_blueprint["metadata"]["model"]["name"]:
        from anthropic import AsyncAnthropicVertex

        cache_key = f"async_anthropic_vertex:{client_kwargs.get('project_id', '')}:{client_kwargs.get('region', '')}"
        client = _get_cached_client(cache_key, lambda: AsyncAnthropicVertex(**client_kwargs))
        if prompt_blueprint["prompt_template"]["type"] == "chat":
            return await aanthropic_chat_request(client=client, **function_kwargs)
        raise NotImplementedError(
            f"Unsupported prompt template type {prompt_blueprint['prompt_template']['type']}' for Anthropic Vertex AI"
        )

    raise NotImplementedError(
        f"Vertex AI request for model {prompt_blueprint['metadata']['model']['name']} is not implemented yet."
    )


def amazon_bedrock_request(
    prompt_blueprint: GetPromptTemplateResponse,
    client_kwargs: dict,
    function_kwargs: dict,
):
    import boto3

    aws_access_key_id = function_kwargs.pop("aws_access_key", None)
    aws_secret_access_key = function_kwargs.pop("aws_secret_key", None)
    region_name = function_kwargs.pop("aws_region", "us-east-1")

    cache_key = f"boto3_bedrock:{aws_access_key_id or ''}:{region_name}"
    bedrock_client = _get_cached_client(
        cache_key,
        lambda: boto3.client(
            "bedrock-runtime",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name,
        ),
    )

    stream = function_kwargs.pop("stream", False)

    if stream:
        return bedrock_client.converse_stream(**function_kwargs)
    else:
        return bedrock_client.converse(**function_kwargs)


async def aamazon_bedrock_request(
    prompt_blueprint: GetPromptTemplateResponse,
    client_kwargs: dict,
    function_kwargs: dict,
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


def anthropic_bedrock_request(
    prompt_blueprint: GetPromptTemplateResponse,
    client_kwargs: dict,
    function_kwargs: dict,
):
    from anthropic import AnthropicBedrock

    aws_access_key = function_kwargs.pop("aws_access_key", None)
    aws_secret_key = function_kwargs.pop("aws_secret_key", None)
    aws_region = function_kwargs.pop("aws_region", None)
    aws_session_token = function_kwargs.pop("aws_session_token", None)
    base_url = function_kwargs.pop("base_url", None)

    cache_key = f"anthropic_bedrock:{aws_access_key or ''}:{aws_region or ''}:{base_url or ''}"
    client = _get_cached_client(
        cache_key,
        lambda: AnthropicBedrock(
            aws_access_key=aws_access_key,
            aws_secret_key=aws_secret_key,
            aws_region=aws_region,
            aws_session_token=aws_session_token,
            base_url=base_url,
            **client_kwargs,
        ),
    )
    if prompt_blueprint["prompt_template"]["type"] == "chat":
        return anthropic_chat_request(client=client, **function_kwargs)
    elif prompt_blueprint["prompt_template"]["type"] == "completion":
        return anthropic_completions_request(client=client, **function_kwargs)
    raise NotImplementedError(
        f"Unsupported prompt template type {prompt_blueprint['prompt_template']['type']}' for Anthropic Bedrock"
    )


async def aanthropic_bedrock_request(
    prompt_blueprint: GetPromptTemplateResponse,
    client_kwargs: dict,
    function_kwargs: dict,
):
    from anthropic import AsyncAnthropicBedrock

    aws_access_key = function_kwargs.pop("aws_access_key", None)
    aws_secret_key = function_kwargs.pop("aws_secret_key", None)
    aws_region = function_kwargs.pop("aws_region", None)
    aws_session_token = function_kwargs.pop("aws_session_token", None)
    base_url = function_kwargs.pop("base_url", None)

    cache_key = f"async_anthropic_bedrock:{aws_access_key or ''}:{aws_region or ''}:{base_url or ''}"
    client = _get_cached_client(
        cache_key,
        lambda: AsyncAnthropicBedrock(
            aws_access_key=aws_access_key,
            aws_secret_key=aws_secret_key,
            aws_region=aws_region,
            aws_session_token=aws_session_token,
            base_url=base_url,
            **client_kwargs,
        ),
    )
    if prompt_blueprint["prompt_template"]["type"] == "chat":
        return await aanthropic_chat_request(client=client, **function_kwargs)
    elif prompt_blueprint["prompt_template"]["type"] == "completion":
        return await aanthropic_completions_request(client=client, **function_kwargs)
    raise NotImplementedError(
        f"Unsupported prompt template type {prompt_blueprint['prompt_template']['type']}' for Anthropic Bedrock"
    )
