from copy import deepcopy
import promptlayer
import requests
import sys
import types


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
            },
        )
    else:
        promptlayer_api_request(
            function_name,
            provider_type,
            args,
            kwargs,
            tags,
            response,
            request_start_time,
            request_end_time,
            api_key,
        )
        return response


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
):
    try:
        request_response = requests.post(
            "https://api.promptlayer.com/track-request",
            json={
                "function_name": function_name,
                "provider_type": provider_type,
                "args": args,
                "kwargs": kwargs,
                "tags": tags,
                "request_response": response,
                "request_start_time": request_start_time,
                "request_end_time": request_end_time,
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


def promptlayer_get_prompt(prompt_name, api_key):
    request_response = requests.post(
        "https://api.promptlayer.com/library-get-prompt-template",
        json={"prompt_name": prompt_name, "api_key": api_key,},
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
    return request_response.json()


def promptlayer_publish_prompt(prompt_name, prompt_template, tags, api_key):
    request_response = requests.post(
        "https://api.promptlayer.com/library-publish-prompt-template",
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
    return request_response.json()


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
            promptlayer_api_request(
                self.api_request_arugments["function_name"],
                self.api_request_arugments["provider_type"],
                self.api_request_arugments["args"],
                self.api_request_arugments["kwargs"],
                self.api_request_arugments["tags"],
                self.cleaned_result(),
                self.api_request_arugments["request_start_time"],
                self.api_request_arugments["request_end_time"],
                get_api_key(),
            )
        return result

    def cleaned_result(self):
        response = ""
        for result in self.results:
            response = f"{response}{result.choices[0].text}"
        final_result = deepcopy(self.results[-1])
        final_result.choices[0].text = response
        return final_result
