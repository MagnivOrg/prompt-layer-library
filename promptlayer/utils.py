import promptlayer
import requests

def get_api_key():
    # raise an error if the api key is not set
    if promptlayer.api_key is None:
        raise Exception(
            "Please set your PROMPTLAYER_API_KEY environment variable or set API KEY in code using 'promptlayer.api_key = <your_api_key>' "
        )
    else:
        return promptlayer.api_key

def promptlayer_api_request(function_name, provider_type, args, kwargs, tags, response, request_start_time, request_end_time, api_key):
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
        raise Exception(f"Error while tracking request: {request_response.json().get('message')}")

class OpenAIGeneratorProxy:
    def __init__(self, generator, api_request_arguments):
        self.generator = generator
        self.results = []
        self.api_request_arugments = api_request_arguments
    def __iter__(self):
        return self
    
    def __next__(self):
        result = next(self.generator)
        self.results.append(result)
        if result.choices[0].finish_reason == "stop":
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
            response = f'{response}{result.choices[0].text}'
        final_result = self.results[-1]
        final_result.choices[0].text = response
        return final_result
