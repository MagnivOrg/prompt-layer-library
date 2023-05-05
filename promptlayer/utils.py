import asyncio
import contextvars
import functools
import os
import sys
import types
from copy import deepcopy
from promptlayer.config import API_KEY, API_SECRET, URL_API_PROMPTLAYER

import requests

import promptlayer

frappe_access_key = f'token {API_KEY}:{API_SECRET}'


headers = {
  'Authorization': frappe_access_key,
  'Content-Type': 'application/json'
}


def promptlayer_api_handler(
    function_name,
    provider_type,
    args,
    kwargs,
    tags,
    response,
    request_start_time,
    request_end_time,
    return_pl_id=False,
):
    if isinstance(response, types.GeneratorType) or isinstance(
        response, types.AsyncGeneratorType
    ):
        return OpenAIGeneratorProxy(
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
    return_pl_id=False,
):
    if type(response) != dict and hasattr(response, "to_dict_recursive"):
        response = response.to_dict_recursive()
    try:
        request_response = requests.post(
            f"{URL_API_PROMPTLAYER}.track",
            headers=headers,
            json={
                "function_name": function_name,
                "provider_type": provider_type,
                "args": args,
                "kwargs": kwargs,
                "tags": tags,
                "request_response": response,
                "request_start_time": request_start_time,
                "request_end_time": request_end_time,
            },
        )
        if request_response.status_code != 200:
            if hasattr(request_response, "json"):
                print(
                    f"WARNING: While logging your request PromptLayer had the following error: {request_response.json().get('message')}",
                    file=sys.stderr,
                )
            else:
                print(
                    f"WARNING: While logging your request PromptLayer had the following error: {request_response}",
                    file=sys.stderr,
                )
    except Exception as e:
        print(
            f"WARNING: While logging your request PromptLayer had the following error: {e}",
            file=sys.stderr,
        )
    if return_pl_id:
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
        return_pl_id=return_pl_id,
    )


def promptlayer_get_prompt(prompt_name, version=None):
    """
    Get a prompt from the PromptLayer library
    version: version of the prompt to get, None for latest
    """
    try:
        request_response = requests.post(
            f"{URL_API_PROMPTLAYER}.library_get_prompt_template",
            headers=headers,
            json={"prompt_name": prompt_name,
                'version': version},
        )
        if request_response.status_code != 200:
            if hasattr(request_response, "json"):
                raise Exception(
                    f"PromptLayer had the following error while getting your prompt: {request_response.json().get('message')}"
                )
            else:
                raise Exception(
                    f"PromptLayer had the following error while getting your prompt: {request_response}"
                )
    except Exception as e:
        raise Exception(
            f"PromptLayer had the following error while getting your prompt: {e}"
        )
    return request_response.json()


def promptlayer_publish_prompt(prompt_name, prompt_template, tags=None):
    try:
        request_response = requests.post(
            f"{URL_API_PROMPTLAYER}.library_publish_prompt_template",
            headers=headers,
            json={
                "prompt_name": prompt_name,
                "prompt_template": prompt_template,
                "tags": tags,
            },
        )
        if request_response.status_code != 200:
            if hasattr(request_response, "json"):
                raise Exception(
                    f"PromptLayer had the following error while publishing your prompt: {request_response.json().get('message')}"
                )
            else:
                raise Exception(
                    f"PromptLayer had the following error while publishing your prompt: {request_response}"
                )
    except Exception as e:
        raise Exception(
            f"PromptLayer had the following error while publishing your prompt: {e}"
        )
    return True


def promptlayer_track_prompt(request_id, prompt_name, input_variables, version):
    try:
        request_response = requests.post(
            f"{URL_API_PROMPTLAYER}.library_track_prompt",
            json={
                "request_id": request_id,
                "prompt_name": prompt_name,
                "prompt_input_variables": input_variables,
                "version": version,
            },
        )
        if request_response.status_code != 200:
            if hasattr(request_response, "json"):
                print(
                    f"WARNING: While tracking your prompt PromptLayer had the following error: {request_response.json().get('message')}",
                    file=sys.stderr,
                )
                return False
            else:
                print(
                    f"WARNING: While tracking your prompt PromptLayer had the following error: {request_response}",
                    file=sys.stderr,
                )
                return False
    except Exception as e:
        print(
            f"WARNING: While tracking your prompt PromptLayer had the following error: {e}",
            file=sys.stderr,
        )
        return False
    return True


def promptlayer_track_metadata(request_id, metadata):
    try:
        request_response = requests.post(
            f"{URL_API_PROMPTLAYER}.library_track_metadata",
            json={"request_id": request_id,
                  "metadata": metadata},
        )
        if request_response.status_code != 200:
            if hasattr(request_response, "json"):
                print(
                    f"WARNING: While tracking your metadata PromptLayer had the following error: {request_response.json().get('message')}",
                    file=sys.stderr,
                )
                return False
            else:
                print(
                    f"WARNING: While tracking your metadata PromptLayer had the following error: {request_response}",
                    file=sys.stderr,
                )
                return False
    except Exception as e:
        print(
            f"WARNING: While tracking your metadata PromptLayer had the following error: {e}",
            file=sys.stderr,
        )
        return False
    return True


def promptlayer_track_score(request_id, score):
    try:
        request_response = requests.post(
            f"{URL_API_PROMPTLAYER}.library_track_score",
            json={"request_id": request_id,
                  "score": score},
        )
        if request_response.status_code != 200:
            if hasattr(request_response, "json"):
                print(
                    f"WARNING: While tracking your score PromptLayer had the following error: {request_response.json().get('message')}",
                    file=sys.stderr,
                )
                return False
            else:
                print(
                    f"WARNING: While tracking your score PromptLayer had the following error: {request_response}",
                    file=sys.stderr,
                )
                return False
    except Exception as e:
        print(
            f"WARNING: While tracking your score PromptLayer had the following error: {e}",
            file=sys.stderr,
        )
        return False
    return True


class OpenAIGeneratorProxy:
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
        if (
            result.choices[0].finish_reason == "stop"
            or result.choices[0].finish_reason == "length"
        ):
            request_id = promptlayer_api_request(
                self.api_request_arugments["function_name"],
                self.api_request_arugments["provider_type"],
                self.api_request_arugments["args"],
                self.api_request_arugments["kwargs"],
                self.api_request_arugments["tags"],
                self.cleaned_result(),
                self.api_request_arugments["request_start_time"],
                self.api_request_arugments["request_end_time"],
                return_pl_id=self.api_request_arugments["return_pl_id"],
            )
            if self.api_request_arugments["return_pl_id"]:
                return result, request_id
        if self.api_request_arugments["return_pl_id"]:
            return result, None
        return result

    def cleaned_result(self):
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
            response = {"message": {"role": "", "content": ""}}
            for result in self.results:
                if hasattr(result.choices[0].delta, "role"):
                    response["message"]["role"] = result.choices[0].delta.role
                if hasattr(result.choices[0].delta, "content"):
                    response["message"][
                        "content"
                    ] = f"{response['message']['content']}{result.choices[0].delta.content}"
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
