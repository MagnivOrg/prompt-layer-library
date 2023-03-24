import datetime
import inspect
from promptlayer.utils import get_api_key, promptlayer_api_handler, run_in_thread_async


class PromptLayerBase(object):
    __slots__ = ["_obj", "__weakref__", "_function_name", "_provider_type"]

    def __init__(self, obj, function_name="", provider_type="openai"):
        object.__setattr__(self, "_obj", obj)
        object.__setattr__(self, "_function_name", function_name)
        object.__setattr__(self, "_provider_type", provider_type)

    def __getattr__(self, name):
        return PromptLayerBase(
            getattr(object.__getattribute__(self, "_obj"), name),
            function_name=f'{object.__getattribute__(self, "_function_name")}.{name}',
            provider_type=object.__getattribute__(self, "_provider_type"),
        )

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
        if inspect.iscoroutinefunction(function_object):

            async def async_wrapper(*args, **kwargs):
                response = await function_object(*args, **kwargs)
                request_end_time = datetime.datetime.now().timestamp()
                return await run_in_thread_async(
                    None,
                    promptlayer_api_handler,
                    object.__getattribute__(self, "_function_name"),
                    object.__getattribute__(self, "_provider_type"),
                    args,
                    kwargs,
                    tags,
                    response,
                    request_start_time,
                    request_end_time,
                    get_api_key(),
                    return_pl_id=return_pl_id,
                )

            return async_wrapper(*args, **kwargs)
        response = function_object(*args, **kwargs)
        request_end_time = datetime.datetime.now().timestamp()
        return promptlayer_api_handler(
            object.__getattribute__(self, "_function_name"),
            object.__getattribute__(self, "_provider_type"),
            args,
            kwargs,
            tags,
            response,
            request_start_time,
            request_end_time,
            get_api_key(),
            return_pl_id=return_pl_id,
        )
