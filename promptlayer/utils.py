import asyncio
import contextvars
import datetime
import functools
import json
import os
import sys
import types
from copy import deepcopy

import requests

import promptlayer

URL_API_PROMPTLAYER = os.environ.setdefault(
    "URL_API_PROMPTLAYER", "https://api.promptlayer.com"
)


def get_api_key():
    # raise an error if the api key is not set
    if promptlayer.api_key is None:
        raise Exception(
            "Please set your PROMPTLAYER_API_KEY environment variable or set API KEY in code using 'promptlayer.api_key = <your_api_key>' "
        )
    else:
        return promptlayer.api_key


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
):
    if (
        isinstance(response, types.GeneratorType)
        or isinstance(response, types.AsyncGeneratorType)
        or type(response).__name__ in ["Stream", "AsyncStream"]
    ):
        return GeneratorProxy(
            response,
            {
                "function_name": function_name,
                "provider_type": provider_type,
                "args": args,
                "kwargs": kwargs,
                "tags": tags,
                "request_start_time": request_start_time,
                "request_end_time": request_end_time,
                "return_pl_id": return_pl_id,
            },
        )
    else:
        request_id = promptlayer_api_request(
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
        get_api_key(),
        return_pl_id=return_pl_id,
    )


def promptlayer_api_request(
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
                "kwargs": {
                    k: v for k, v in kwargs.items() if _check_if_json_serializable(v)
                },
                "tags": tags,
                "request_response": response,
                "request_start_time": request_start_time,
                "request_end_time": request_end_time,
                "metadata": metadata,
                "api_key": api_key,
            },
        )
        if request_response.status_code != 200:
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


class GeneratorProxy:
    def __init__(self, generator, api_request_arguments):
        self.generator = generator
        self.results = []
        self.api_request_arugments = api_request_arguments

    def __iter__(self):
        return self

    def __aiter__(self):
        return self

    async def __anext__(self):
        result = await self.generator.__anext__()
        return self._abstracted_next(result)

    def __next__(self):
        result = next(self.generator)
        return self._abstracted_next(result)

    def _abstracted_next(self, result):
        self.results.append(result)
        provider_type = self.api_request_arugments["provider_type"]
        end_anthropic = provider_type == "anthropic" and result.stop_reason
        end_openai = provider_type == "openai" and (
            result.choices[0].finish_reason == "stop"
            or result.choices[0].finish_reason == "length"
        )
        if end_anthropic or end_openai:
            request_id = promptlayer_api_request(
                self.api_request_arugments["function_name"],
                self.api_request_arugments["provider_type"],
                self.api_request_arugments["args"],
                self.api_request_arugments["kwargs"],
                self.api_request_arugments["tags"],
                self.cleaned_result(),
                self.api_request_arugments["request_start_time"],
                self.api_request_arugments["request_end_time"],
                get_api_key(),
                return_pl_id=self.api_request_arugments["return_pl_id"],
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
                response = f"{response}{result.completion}"
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
                    response[
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
            raise Exception(f"{main_message}: {request_response.json().get('message')}")
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
    *args,
    **kwargs,
):
    response = await coroutine_obj
    request_end_time = datetime.datetime.now().timestamp()
    return await promptlayer_api_handler_async(
        function_name,
        provider_type,
        args,
        kwargs,
        tags,
        response,
        request_start_time,
        request_end_time,
        get_api_key(),
        return_pl_id=return_pl_id,
    )


def _check_if_json_serializable(value):
    try:
        json.dumps(value)
        return True
    except Exception:
        return False


def promptlayer_create_group():
    try:
        request_response = requests.post(
            f"{URL_API_PROMPTLAYER}/create-group",
            json={
                "api_key": get_api_key(),
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


def promptlayer_track_group(request_id, group_id):
    try:
        request_response = requests.post(
            f"{URL_API_PROMPTLAYER}/track-group",
            json={
                "api_key": get_api_key(),
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
