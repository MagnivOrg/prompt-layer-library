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

