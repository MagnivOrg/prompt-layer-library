import datetime
import inspect
import re

from promptlayer.utils import async_wrapper, get_api_key, promptlayer_api_handler


class PromptLayerBase(object):
    __slots__ = ["_obj", "__weakref__", "_function_name", "_provider_type"]

    def __init__(self, obj, function_name="", provider_type="openai"):
        object.__setattr__(self, "_obj", obj)
        object.__setattr__(self, "_function_name", function_name)
        object.__setattr__(self, "_provider_type", provider_type)

    def __getattr__(self, name):
        attr = getattr(object.__getattribute__(self, "_obj"), name)
        if (
            name != "count_tokens"  # fix for anthropic count_tokens
            and not re.match(
                "<class 'anthropic\..*Error'>", str(attr)
            )  # fix for anthropic errors
            and not re.match(
                "<class 'openai\..*Error'>", str(attr)
            )  # fix for openai errors
            and (
                inspect.isclass(attr)
                or inspect.isfunction(attr)
                or inspect.ismethod(attr)
                or str(type(attr))
                == "<class 'anthropic.resources.completions.Completions'>"
                or str(type(attr))
                == "<class 'anthropic.resources.completions.AsyncCompletions'>"
                or re.match("<class 'openai\.resources.*'>", str(type(attr)))
            )
        ):
            return PromptLayerBase(
                attr,
                function_name=f'{object.__getattribute__(self, "_function_name")}.{name}',
                provider_type=object.__getattribute__(self, "_provider_type"),
            )
        return attr

    def __delattr__(self, name):
        delattr(object.__getattribute__(self, "_obj"), name)

    def __setattr__(self, name, value):
        setattr(object.__getattribute__(self, "_obj"), name, value)

    def __call__(self, *args, **kwargs):
        tags = kwargs.pop("pl_tags", None)
        if tags is not None and not isinstance(tags, list):
            raise Exception("pl_tags must be a list of strings.")
        return_pl_id = kwargs.pop("return_pl_id", False)
        request_start_time = datetime.datetime.now().timestamp()
        function_object = object.__getattribute__(self, "_obj")
        if inspect.isclass(function_object):
            return PromptLayerBase(
                function_object(*args, **kwargs),
                function_name=object.__getattribute__(self, "_function_name"),
                provider_type=object.__getattribute__(self, "_provider_type"),
            )
        function_response = function_object(*args, **kwargs)
        if inspect.iscoroutinefunction(function_object) or inspect.iscoroutine(
            function_response
        ):
            return async_wrapper(
                function_response,
                return_pl_id,
                request_start_time,
                object.__getattribute__(self, "_function_name"),
                object.__getattribute__(self, "_provider_type"),
                tags,
                *args,
                **kwargs,
            )
        request_end_time = datetime.datetime.now().timestamp()
        return promptlayer_api_handler(
            object.__getattribute__(self, "_function_name"),
            object.__getattribute__(self, "_provider_type"),
            args,
            kwargs,
            tags,
            function_response,
            request_start_time,
            request_end_time,
            get_api_key(),
            return_pl_id=return_pl_id,
        )
