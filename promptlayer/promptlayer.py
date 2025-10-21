import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Literal, Optional, Union

import nest_asyncio

from promptlayer.groups import AsyncGroupManager, GroupManager
from promptlayer.promptlayer_base import PromptLayerBase
from promptlayer.promptlayer_mixins import PromptLayerMixin
from promptlayer.streaming import astream_response, stream_response
from promptlayer.templates import AsyncTemplateManager, TemplateManager
from promptlayer.track import AsyncTrackManager, TrackManager
from promptlayer.types.prompt_template import PromptTemplate
from promptlayer.utils import (
    RERAISE_ORIGINAL_EXCEPTION,
    _get_workflow_workflow_id_or_name,
    arun_workflow_request,
    atrack_request,
    autil_log_request,
    track_request,
    util_log_request,
)

logger = logging.getLogger(__name__)


def get_base_url(base_url: Union[str, None]):
    return base_url or os.environ.get("PROMPTLAYER_BASE_URL", "https://api.promptlayer.com")


def is_workflow_results_dict(obj: Any) -> bool:
    if not isinstance(obj, dict):
        return False

    required_keys = {
        "status",
        "value",
        "error_message",
        "raw_error_message",
        "is_output_node",
    }

    for val in obj.values():
        if not isinstance(val, dict):
            return False
        if not required_keys.issubset(val.keys()):
            return False

    return True


class PromptLayer(PromptLayerMixin):
    def __init__(
        self, api_key: Union[str, None] = None, enable_tracing: bool = False, base_url: Union[str, None] = None
    ):
        if api_key is None:
            api_key = os.environ.get("PROMPTLAYER_API_KEY")

        if api_key is None:
            raise ValueError(
                "PromptLayer API key not provided. "
                "Please set the PROMPTLAYER_API_KEY environment variable or pass the api_key parameter."
            )

        self.base_url = get_base_url(base_url)
        self.api_key = api_key
        self.templates = TemplateManager(api_key, self.base_url)
        self.group = GroupManager(api_key, self.base_url)
        self.tracer_provider, self.tracer = self._initialize_tracer(api_key, self.base_url, enable_tracing)
        self.track = TrackManager(api_key, self.base_url)

    def __getattr__(
        self,
        name: Union[Literal["openai"], Literal["anthropic"], Literal["prompts"]],
    ):
        if name == "openai":
            import openai as openai_module

            return PromptLayerBase(
                self.api_key, self.base_url, openai_module, function_name="openai", tracer=self.tracer
            )
        elif name == "anthropic":
            import anthropic as anthropic_module

            return PromptLayerBase(
                self.api_key,
                self.base_url,
                anthropic_module,
                function_name="anthropic",
                provider_type="anthropic",
                tracer=self.tracer,
            )
        else:
            raise AttributeError(f"module {__name__} has no attribute {name}")

    def _create_track_request_callable(
        self,
        *,
        request_params,
        tags,
        input_variables,
        group_id,
        pl_run_span_id: Union[str, None] = None,
    ):
        def _track_request(**body):
            track_request_kwargs = self._prepare_track_request_kwargs(
                self.api_key,
                request_params,
                tags,
                input_variables,
                group_id,
                pl_run_span_id,
                **body,
            )
            return track_request(**track_request_kwargs)

        return _track_request

    def _run_internal(
        self,
        *,
        prompt_name: str,
        prompt_version: Union[int, None] = None,
        prompt_release_label: Union[str, None] = None,
        input_variables: Union[Dict[str, Any], None] = None,
        model_parameter_overrides: Union[Dict[str, Any], None] = None,
        tags: Union[List[str], None] = None,
        metadata: Union[Dict[str, str], None] = None,
        group_id: Union[int, None] = None,
        stream: bool = False,
        pl_run_span_id: Union[str, None] = None,
        provider: Union[str, None] = None,
        model: Union[str, None] = None,
    ) -> Dict[str, Any]:
        get_prompt_template_params = self._prepare_get_prompt_template_params(
            prompt_version=prompt_version,
            prompt_release_label=prompt_release_label,
            input_variables=input_variables,
            metadata=metadata,
            provider=provider,
            model=model,
            model_parameter_overrides=model_parameter_overrides,
        )
        prompt_blueprint = self.templates.get(prompt_name, get_prompt_template_params)
        prompt_blueprint_model = self._validate_and_extract_model_from_prompt_blueprint(
            prompt_blueprint=prompt_blueprint, prompt_name=prompt_name
        )
        llm_data = self._prepare_llm_data(
            prompt_blueprint=prompt_blueprint,
            prompt_template=prompt_blueprint["prompt_template"],
            prompt_blueprint_model=prompt_blueprint_model,
            stream=stream,
        )

        # response is just whatever the LLM call returns
        # streaming=False > Pydantic model instance
        # streaming=True > generator that yields ChatCompletionChunk pieces as they arrive
        response = llm_data["request_function"](
            prompt_blueprint=llm_data["prompt_blueprint"],
            client_kwargs=llm_data["client_kwargs"],
            function_kwargs=llm_data["function_kwargs"],
        )

        if stream:
            return stream_response(
                generator=response,
                after_stream=self._create_track_request_callable(
                    request_params=llm_data,
                    tags=tags,
                    input_variables=input_variables,
                    group_id=group_id,
                    pl_run_span_id=pl_run_span_id,
                ),
                map_results=llm_data["stream_function"],
                metadata=llm_data["prompt_blueprint"]["metadata"],
            )

        if isinstance(response, dict):
            request_response = response
        else:
            request_response = response.model_dump(mode="json")

        request_log = self._track_request_log(
            llm_data,
            tags,
            input_variables,
            group_id,
            pl_run_span_id,
            metadata=metadata,
            request_response=request_response,
        )

        return {
            "request_id": request_log.get("request_id", None),
            "raw_response": response,
            "prompt_blueprint": request_log.get("prompt_blueprint", None),
        }

    def _track_request_log(
        self,
        request_params,
        tags,
        input_variables,
        group_id,
        pl_run_span_id: Union[str, None] = None,
        metadata: Union[Dict[str, str], None] = None,
        **body,
    ):
        track_request_kwargs = self._prepare_track_request_kwargs(
            self.api_key,
            request_params,
            tags,
            input_variables,
            group_id,
            pl_run_span_id,
            metadata=metadata,
            **body,
        )
        return track_request(self.base_url, **track_request_kwargs)

    def run(
        self,
        prompt_name: str,
        prompt_version: Union[int, None] = None,
        prompt_release_label: Union[str, None] = None,
        input_variables: Union[Dict[str, Any], None] = None,
        model_parameter_overrides: Union[Dict[str, Any], None] = None,
        tags: Union[List[str], None] = None,
        metadata: Union[Dict[str, str], None] = None,
        group_id: Union[int, None] = None,
        stream: bool = False,
        provider: Union[str, None] = None,
        model: Union[str, None] = None,
    ) -> Dict[str, Any]:
        _run_internal_kwargs = {
            "prompt_name": prompt_name,
            "prompt_version": prompt_version,
            "prompt_release_label": prompt_release_label,
            "input_variables": input_variables or {},
            "model_parameter_overrides": model_parameter_overrides,
            "tags": tags,
            "metadata": metadata,
            "group_id": group_id,
            "stream": stream,
            "provider": provider,
            "model": model,
        }

        if self.tracer:
            with self.tracer.start_as_current_span("PromptLayer Run") as span:
                span.set_attribute("prompt_name", prompt_name)
                span.set_attribute("function_input", str(_run_internal_kwargs))
                pl_run_span_id = hex(span.context.span_id)[2:].zfill(16)
                result = self._run_internal(**_run_internal_kwargs, pl_run_span_id=pl_run_span_id)
                span.set_attribute("function_output", str(result))
                return result
        else:
            return self._run_internal(**_run_internal_kwargs)

    def run_workflow(
        self,
        workflow_id_or_name: Optional[Union[int, str]] = None,
        input_variables: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, str]] = None,
        workflow_label_name: Optional[str] = None,
        workflow_version: Optional[int] = None,
        return_all_outputs: Optional[bool] = False,
        # `workflow_name` deprecated, kept for backward compatibility only.
        # Allows `workflow_name` to be passed both as keyword and positional argument
        # (virtually identical to `workflow_id_or_name`)
        workflow_name: Optional[str] = None,
    ) -> Union[Dict[str, Any], Any]:
        try:
            try:
                loop = asyncio.get_running_loop()  # Check if we're inside a running event loop
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                nest_asyncio.apply()

            results = asyncio.run(
                arun_workflow_request(
                    api_key=self.api_key,
                    base_url=self.base_url,
                    workflow_id_or_name=_get_workflow_workflow_id_or_name(workflow_id_or_name, workflow_name),
                    input_variables=input_variables or {},
                    metadata=metadata,
                    workflow_label_name=workflow_label_name,
                    workflow_version_number=workflow_version,
                    return_all_outputs=return_all_outputs,
                )
            )

            if not return_all_outputs and is_workflow_results_dict(results):
                output_nodes = [node_data for node_data in results.values() if node_data.get("is_output_node")]
                if not output_nodes:
                    raise Exception("Output nodes not found: %S", json.dumps(results, indent=4))

                if not any(node.get("status") == "SUCCESS" for node in output_nodes):
                    raise Exception("None of the output nodes have succeeded", json.dumps(results, indent=4))

            return results
        except Exception as ex:
            logger.exception("Error running workflow")
            if RERAISE_ORIGINAL_EXCEPTION:
                raise
            else:
                raise Exception(f"Error running workflow: {str(ex)}") from ex

    def log_request(
        self,
        *,
        provider: str,
        model: str,
        input: PromptTemplate,
        output: PromptTemplate,
        request_start_time: float,
        request_end_time: float,
        # TODO(dmu) MEDIUM: Avoid using mutable defaults
        # TODO(dmu) MEDIUM: Deprecate and remove this wrapper function?
        parameters: Dict[str, Any] = {},
        tags: List[str] = [],
        metadata: Dict[str, str] = {},
        prompt_name: Union[str, None] = None,
        prompt_version_number: Union[int, None] = None,
        prompt_input_variables: Dict[str, Any] = {},
        input_tokens: int = 0,
        output_tokens: int = 0,
        price: float = 0.0,
        function_name: str = "",
        score: int = 0,
        prompt_id: Union[int, None] = None,
        score_name: Union[str, None] = None,
    ):
        return util_log_request(
            self.api_key,
            self.base_url,
            provider=provider,
            model=model,
            input=input,
            output=output,
            request_start_time=request_start_time,
            request_end_time=request_end_time,
            parameters=parameters,
            tags=tags,
            metadata=metadata,
            prompt_name=prompt_name,
            prompt_version_number=prompt_version_number,
            prompt_input_variables=prompt_input_variables,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            price=price,
            function_name=function_name,
            score=score,
            prompt_id=prompt_id,
            score_name=score_name,
        )


class AsyncPromptLayer(PromptLayerMixin):
    def __init__(
        self, api_key: Union[str, None] = None, enable_tracing: bool = False, base_url: Union[str, None] = None
    ):
        if api_key is None:
            api_key = os.environ.get("PROMPTLAYER_API_KEY")

        if api_key is None:
            raise ValueError(
                "PromptLayer API key not provided. "
                "Please set the PROMPTLAYER_API_KEY environment variable or pass the api_key parameter."
            )

        self.base_url = get_base_url(base_url)
        self.api_key = api_key
        self.templates = AsyncTemplateManager(api_key, self.base_url)
        self.group = AsyncGroupManager(api_key, self.base_url)
        self.tracer_provider, self.tracer = self._initialize_tracer(api_key, self.base_url, enable_tracing)
        self.track = AsyncTrackManager(api_key, self.base_url)

    def __getattr__(self, name: Union[Literal["openai"], Literal["anthropic"], Literal["prompts"]]):
        if name == "openai":
            import openai as openai_module

            openai = PromptLayerBase(
                self.api_key, self.base_url, openai_module, function_name="openai", tracer=self.tracer
            )
            return openai
        elif name == "anthropic":
            import anthropic as anthropic_module

            anthropic = PromptLayerBase(
                self.api_key,
                self.base_url,
                anthropic_module,
                function_name="anthropic",
                provider_type="anthropic",
                tracer=self.tracer,
            )
            return anthropic
        else:
            raise AttributeError(f"module {__name__} has no attribute {name}")

    async def run_workflow(
        self,
        workflow_id_or_name: Optional[Union[int, str]] = None,
        input_variables: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, str]] = None,
        workflow_label_name: Optional[str] = None,
        workflow_version: Optional[int] = None,  # This is the version number, not the version ID
        return_all_outputs: Optional[bool] = False,
        # `workflow_name` deprecated, kept for backward compatibility only.
        # Allows `workflow_name` to be passed both as keyword and positional argument
        # (virtually identical to `workflow_id_or_name`)
        workflow_name: Optional[str] = None,
    ) -> Union[Dict[str, Any], Any]:
        try:
            return await arun_workflow_request(
                api_key=self.api_key,
                base_url=self.base_url,
                workflow_id_or_name=_get_workflow_workflow_id_or_name(workflow_id_or_name, workflow_name),
                input_variables=input_variables or {},
                metadata=metadata,
                workflow_label_name=workflow_label_name,
                workflow_version_number=workflow_version,
                return_all_outputs=return_all_outputs,
            )
        except Exception as ex:
            logger.exception("Error running workflow")
            if RERAISE_ORIGINAL_EXCEPTION:
                raise
            else:
                raise Exception(f"Error running workflow: {str(ex)}")

    async def run(
        self,
        prompt_name: str,
        prompt_version: Union[int, None] = None,
        prompt_release_label: Union[str, None] = None,
        input_variables: Union[Dict[str, Any], None] = None,
        model_parameter_overrides: Union[Dict[str, Any], None] = None,
        tags: Union[List[str], None] = None,
        metadata: Union[Dict[str, str], None] = None,
        group_id: Union[int, None] = None,
        stream: bool = False,
        provider: Union[str, None] = None,
        model: Union[str, None] = None,
    ) -> Dict[str, Any]:
        _run_internal_kwargs = {
            "prompt_name": prompt_name,
            "prompt_version": prompt_version,
            "prompt_release_label": prompt_release_label,
            "input_variables": input_variables,
            "model_parameter_overrides": model_parameter_overrides,
            "tags": tags,
            "metadata": metadata,
            "group_id": group_id,
            "stream": stream,
            "provider": provider,
            "model": model,
        }

        if self.tracer:
            with self.tracer.start_as_current_span("PromptLayer Run") as span:
                span.set_attribute("prompt_name", prompt_name)
                span.set_attribute("function_input", str(_run_internal_kwargs))
                pl_run_span_id = hex(span.context.span_id)[2:].zfill(16)
                result = await self._run_internal(**_run_internal_kwargs, pl_run_span_id=pl_run_span_id)
                span.set_attribute("function_output", str(result))
                return result
        else:
            return await self._run_internal(**_run_internal_kwargs)

    async def log_request(
        self,
        *,
        provider: str,
        model: str,
        input: PromptTemplate,
        output: PromptTemplate,
        request_start_time: float,
        request_end_time: float,
        parameters: Dict[str, Any] = {},
        tags: List[str] = [],
        metadata: Dict[str, str] = {},
        prompt_name: Union[str, None] = None,
        prompt_version_number: Union[int, None] = None,
        prompt_input_variables: Dict[str, Any] = {},
        input_tokens: int = 0,
        output_tokens: int = 0,
        price: float = 0.0,
        function_name: str = "",
        score: int = 0,
        prompt_id: Union[int, None] = None,
    ):
        return await autil_log_request(
            self.api_key,
            self.base_url,
            provider=provider,
            model=model,
            input=input,
            output=output,
            request_start_time=request_start_time,
            request_end_time=request_end_time,
            parameters=parameters,
            tags=tags,
            metadata=metadata,
            prompt_name=prompt_name,
            prompt_version_number=prompt_version_number,
            prompt_input_variables=prompt_input_variables,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            price=price,
            function_name=function_name,
            score=score,
            prompt_id=prompt_id,
        )

    async def _create_track_request_callable(
        self,
        *,
        request_params,
        tags,
        input_variables,
        group_id,
        pl_run_span_id: Union[str, None] = None,
    ):
        async def _track_request(**body):
            track_request_kwargs = self._prepare_track_request_kwargs(
                self.api_key,
                request_params,
                tags,
                input_variables,
                group_id,
                pl_run_span_id,
                **body,
            )
            return await atrack_request(self.base_url, **track_request_kwargs)

        return _track_request

    async def _track_request_log(
        self,
        request_params,
        tags,
        input_variables,
        group_id,
        pl_run_span_id: Union[str, None] = None,
        metadata: Union[Dict[str, str], None] = None,
        **body,
    ):
        track_request_kwargs = self._prepare_track_request_kwargs(
            self.api_key,
            request_params,
            tags,
            input_variables,
            group_id,
            pl_run_span_id,
            metadata=metadata,
            **body,
        )
        return await atrack_request(self.base_url, **track_request_kwargs)

    async def _run_internal(
        self,
        *,
        prompt_name: str,
        prompt_version: Union[int, None] = None,
        prompt_release_label: Union[str, None] = None,
        input_variables: Union[Dict[str, Any], None] = None,
        model_parameter_overrides: Union[Dict[str, Any], None] = None,
        tags: Union[List[str], None] = None,
        metadata: Union[Dict[str, str], None] = None,
        group_id: Union[int, None] = None,
        stream: bool = False,
        pl_run_span_id: Union[str, None] = None,
        provider: Union[str, None] = None,
        model: Union[str, None] = None,
    ) -> Dict[str, Any]:
        get_prompt_template_params = self._prepare_get_prompt_template_params(
            prompt_version=prompt_version,
            prompt_release_label=prompt_release_label,
            input_variables=input_variables,
            metadata=metadata,
            provider=provider,
            model=model,
            model_parameter_overrides=model_parameter_overrides,
        )
        prompt_blueprint = await self.templates.get(prompt_name, get_prompt_template_params)
        prompt_blueprint_model = self._validate_and_extract_model_from_prompt_blueprint(
            prompt_blueprint=prompt_blueprint, prompt_name=prompt_name
        )
        llm_data = self._prepare_llm_data(
            prompt_blueprint=prompt_blueprint,
            prompt_template=prompt_blueprint["prompt_template"],
            prompt_blueprint_model=prompt_blueprint_model,
            stream=stream,
            is_async=True,
        )

        response = await llm_data["request_function"](
            prompt_blueprint=llm_data["prompt_blueprint"],
            client_kwargs=llm_data["client_kwargs"],
            function_kwargs=llm_data["function_kwargs"],
        )

        if hasattr(response, "model_dump"):
            request_response = response.model_dump(mode="json")
        else:
            request_response = response

        if stream:
            track_request_callable = await self._create_track_request_callable(
                request_params=llm_data,
                tags=tags,
                input_variables=input_variables,
                group_id=group_id,
                pl_run_span_id=pl_run_span_id,
            )
            return astream_response(
                request_response,
                track_request_callable,
                llm_data["stream_function"],
                llm_data["prompt_blueprint"]["metadata"],
            )

        request_log = await self._track_request_log(
            llm_data,
            tags,
            input_variables,
            group_id,
            pl_run_span_id,
            metadata=metadata,
            request_response=request_response,
        )

        return {
            "request_id": request_log.get("request_id", None),
            "raw_response": response,
            "prompt_blueprint": request_log.get("prompt_blueprint", None),
        }
