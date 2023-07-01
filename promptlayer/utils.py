import asyncio
import contextvars
import functools
import os
import sys
import types
from copy import deepcopy
import requests
import time
import openai
import anthropic
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

def run_prompt_registry(prompt, obj):

    current_time = time.time()
    print(obj)
    # Update the request_start_time and request_end_time with the current time
    request_start_time = current_time
    request_end_time = current_time
    if (obj['model'] == 'openai'):
        if "messages" in prompt:
            # get the system message
            template = prompt['messages'][0]['prompt']['template']
            # get user message 
            message = prompt['messages'][1]['prompt']['template']

            # generate new ChatCompletionPrompt
            completion = openai.ChatCompletion.create(
                model=obj['engine'],
                messages=[
                    {"role": "system", "content": template},
                    {"role": "user", "content": message}
                ]
            )

            # print(completion['choices'][0]['message']['content'])
            request_response = requests.post(
            "https://api.promptlayer.com/rest/track-request",
                json={
                    "function_name": "openai.ChatCompletion.create",
                    "kwargs": {"engine": obj['engine'], "messages": [
                    {
                        "content": template,
                        "role": "system"
                    },
                    {
                        "content": message,
                        "role": "user"
                    }
                    ]},
                    "tags": obj['tags'],
                    "request_response": completion,
                    "request_start_time": request_start_time,
                    "request_end_time": request_end_time,
                    # "prompt_id": prompt_name,
                    "prompt_input_variables": "",
                    "prompt_version": obj['version'],
                    "api_key": promptlayer.api_key,
                },
            )
            # get the id of the request
            requestID = request_response.json()['request_id']
        else:
            # this will load chat completion
            # get user message 
            message = prompt['template']

            #generate a new Completion prompt
            completion = openai.Completion.create(
                model=obj['engine'],
                prompt=message,
                max_tokens=7,
                temperature=0
            )
            #generate a new track with the answer of the completion
            request_response = requests.post(
            "https://api.promptlayer.com/rest/track-request",
                json={
                    "function_name": "openai.Completion.create",
                    "kwargs": {"engine": obj['engine'], "prompt": message},
                    "tags": obj['tags'],
                    "request_response": completion,
                    "request_start_time": request_start_time,
                    "request_end_time": request_end_time,
                    # "prompt_id": prompt_name,
                    "prompt_input_variables": "",
                    "prompt_version": obj['version'],
                    "api_key": promptlayer.api_key,
                },
            )
            # get the id of the track
            requestID = request_response.json()['request_id']
    else:
        if "messages" in prompt:
            message = prompt['messages'][1]['prompt']['template']
        else:
            message = prompt['template']

        anthropic = promptlayer.anthropic
        promptlayer.api_key = os.environ.get("PROMPTLAYER_API_KEY")
        anthropic_client = anthropic.Client(
            os.environ.get("ANTHROPIC_API_KEY")   
        )

        response = anthropic_client.completion(
            prompt=f"{anthropic.HUMAN_PROMPT} {message}{anthropic.AI_PROMPT}",
            stop_sequences=[anthropic.HUMAN_PROMPT],
            model="claude-v1-100k",
            max_tokens_to_sample=100,
            pl_tags=obj['tags'],
            return_pl_id=True,
        )
        return

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
    if isinstance(response, types.GeneratorType) or isinstance(
        response, types.AsyncGeneratorType
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
    if type(response) != dict and hasattr(response, "to_dict_recursive"):
        response = response.to_dict_recursive()
    try:
        request_response = requests.post(
            f"{URL_API_PROMPTLAYER}/track-request",
            json={
                "function_name": function_name,
                "provider_type": provider_type,
                "args": args,
                "kwargs": kwargs,
                "tags": tags,
                "request_response": response,
                "request_start_time": request_start_time,
                "request_end_time": request_end_time,
                "metadata": metadata,
                "api_key": api_key,
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


def promptlayer_get_prompt(prompt_name, api_key, version=None):
    """
    Get a prompt from the PromptLayer library
    version: version of the prompt to get, None for latest
    """
    try:
        request_response = requests.get(
            f"{URL_API_PROMPTLAYER}/library-get-prompt-template",
            headers={"X-API-KEY": api_key},
            params={"prompt_name": prompt_name, "version": version},
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


def promptlayer_publish_prompt(prompt_name, prompt_template, tags, api_key):
    try:
        request_response = requests.post(
            f"{URL_API_PROMPTLAYER}/library-publish-prompt-template",
            json={
                "prompt_name": prompt_name,
                "prompt_template": prompt_template,
                "tags": tags,
                "api_key": api_key,
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


def promptlayer_track_prompt(
    request_id, prompt_name, input_variables, api_key, version
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


def promptlayer_track_score(request_id, score, api_key):
    try:
        request_response = requests.post(
            f"{URL_API_PROMPTLAYER}/library-track-score",
            json={
                "request_id": request_id,
                "score": score,
                "api_key": api_key,
            },
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
        end_anthropic = provider_type == "anthropic" and result.get("stop")
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
            final_result = deepcopy(self.results[-1])
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
                if hasattr(result.choices[0].delta, "role"):
                    response["role"] = result.choices[0].delta.role
                if hasattr(result.choices[0].delta, "content"):
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
