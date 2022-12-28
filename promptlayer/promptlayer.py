import requests


class PromptLayer(object):
    __slots__ = ["_obj", "__weakref__", "_function_name"]

    def __init__(self, obj, function_name=""):
        object.__setattr__(self, "_obj", obj)
        object.__setattr__(self, "_function_name", function_name)

    def __getattr__(self, name):
        return PromptLayer(
            getattr(object.__getattribute__(self, "_obj"), name),
            function_name=f'{object.__getattribute__(self, "_function_name")}.{name}',
        )

    def __delattr__(self, name):
        delattr(object.__getattribute__(self, "_obj"), name)

    def __setattr__(self, name, value):
        setattr(object.__getattribute__(self, "_obj"), name, value)

    def __call__(self, *args, **kwargs):
        from promptlayer.utils import get_api_key

        tag = kwargs.pop("pl_tag", None)
        response = object.__getattribute__(self, "_obj")(*args, **kwargs)
        requests.post(
            "https://api.promptlayer.com/track",
            headers={"Authorization": f"Bearer {get_api_key()}"},
            data={
                "function_name": object.__getattribute__(self, "_function_name"),
                "args": args,
                "kwargs": kwargs,
                "tag": tag,
                "response": response,
            },
        )
        return response
