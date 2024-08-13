import datetime
import inspect
import re

from promptlayer.utils import async_wrapper, promptlayer_api_handler


class PromptLayerBase(object):
    __slots__ = [
        "_obj",
        "__weakref__",
        "_function_name",
        "_provider_type",
        "_api_key",
        "_tracer",
    ]

    def __init__(
        self, obj, function_name="", provider_type="openai", api_key=None, tracer=None
    ):
        object.__setattr__(self, "_obj", obj)
        object.__setattr__(self, "_function_name", function_name)
        object.__setattr__(self, "_provider_type", provider_type)
        object.__setattr__(self, "_api_key", api_key)
        object.__setattr__(self, "_tracer", tracer)

    def __getattr__(self, name):
        attr = getattr(object.__getattribute__(self, "_obj"), name)

        if (
            name != "count_tokens"  # fix for anthropic count_tokens
            and not re.match(
                r"<class 'anthropic\..*Error'>", str(attr)
            )  # fix for anthropic errors
            and not re.match(
                r"<class 'openai\..*Error'>", str(attr)
            )  # fix for openai errors
            and (
                inspect.isclass(attr)
                or inspect.isfunction(attr)
                or inspect.ismethod(attr)
                or str(type(attr))
                == "<class 'anthropic.resources.completions.Completions'>"
                or str(type(attr))
                == "<class 'anthropic.resources.completions.AsyncCompletions'>"
                or str(type(attr)) == "<class 'anthropic.resources.messages.Messages'>"
                or str(type(attr))
                == "<class 'anthropic.resources.messages.AsyncMessages'>"
                or re.match(r"<class 'openai\.resources.*'>", str(type(attr)))
            )
        ):
            return PromptLayerBase(
                attr,
                function_name=f'{object.__getattribute__(self, "_function_name")}.{name}',
                provider_type=object.__getattribute__(self, "_provider_type"),
                api_key=object.__getattribute__(self, "_api_key"),
                tracer=object.__getattribute__(self, "_tracer"),
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
        tracer = object.__getattribute__(self, "_tracer")
        function_name = object.__getattribute__(self, "_function_name")

        if tracer:
            with tracer.start_as_current_span(function_name) as llm_request_span:
                llm_request_span_id = hex(llm_request_span.context.span_id)[2:].zfill(
                    16
                )
                llm_request_span.set_attribute(
                    "provider", object.__getattribute__(self, "_provider_type")
                )
                llm_request_span.set_attribute("function_name", function_name)
                llm_request_span.set_attribute(
                    "function_input", str({"args": args, "kwargs": kwargs})
                )

                if inspect.isclass(function_object):
                    result = PromptLayerBase(
                        function_object(*args, **kwargs),
                        function_name=function_name,
                        provider_type=object.__getattribute__(self, "_provider_type"),
                        api_key=object.__getattribute__(self, "_api_key"),
                        tracer=tracer,
                    )
                    llm_request_span.set_attribute("function_output", str(result))
                    return result

                function_response = function_object(*args, **kwargs)

                if inspect.iscoroutinefunction(function_object) or inspect.iscoroutine(
                    function_response
                ):
                    return async_wrapper(
                        function_response,
                        return_pl_id,
                        request_start_time,
                        function_name,
                        object.__getattribute__(self, "_provider_type"),
                        tags,
                        api_key=object.__getattribute__(self, "_api_key"),
                        llm_request_span_id=llm_request_span_id,
                        tracer=tracer,  # Pass the tracer to async_wrapper
                        *args,
                        **kwargs,
                    )

                request_end_time = datetime.datetime.now().timestamp()
                result = promptlayer_api_handler(
                    function_name,
                    object.__getattribute__(self, "_provider_type"),
                    args,
                    kwargs,
                    tags,
                    function_response,
                    request_start_time,
                    request_end_time,
                    object.__getattribute__(self, "_api_key"),
                    return_pl_id=return_pl_id,
                    llm_request_span_id=llm_request_span_id,
                )
                llm_request_span.set_attribute("function_output", str(result))
                return result
        else:
            # Without tracing
            if inspect.isclass(function_object):
                return PromptLayerBase(
                    function_object(*args, **kwargs),
                    function_name=function_name,
                    provider_type=object.__getattribute__(self, "_provider_type"),
                    api_key=object.__getattribute__(self, "_api_key"),
                )

            function_response = function_object(*args, **kwargs)

            if inspect.iscoroutinefunction(function_object) or inspect.iscoroutine(
                function_response
            ):
                return async_wrapper(
                    function_response,
                    return_pl_id,
                    request_start_time,
                    function_name,
                    object.__getattribute__(self, "_provider_type"),
                    tags,
                    api_key=object.__getattribute__(self, "_api_key"),
                    *args,
                    **kwargs,
                )

            request_end_time = datetime.datetime.now().timestamp()
            return promptlayer_api_handler(
                function_name,
                object.__getattribute__(self, "_provider_type"),
                args,
                kwargs,
                tags,
                function_response,
                request_start_time,
                request_end_time,
                object.__getattribute__(self, "_api_key"),
                return_pl_id=return_pl_id,
            )
