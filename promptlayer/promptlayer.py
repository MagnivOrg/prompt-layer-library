import asyncio
import os
from typing import Any, Dict, List, Literal, Optional, Union

import nest_asyncio

from promptlayer.groups import AsyncGroupManager, GroupManager
from promptlayer.promptlayer_base import PromptLayerBase
from promptlayer.promptlayer_mixins import PromptLayerMixin
from promptlayer.templates import AsyncTemplateManager, TemplateManager
from promptlayer.track import AsyncTrackManager, TrackManager
from promptlayer.types.prompt_template import PromptTemplate
from promptlayer.utils import (
    arun_workflow_request,
    astream_response,
    atrack_request,
    autil_log_request,
    stream_response,
    track_request,
    util_log_request,
)


class PromptLayer(PromptLayerMixin):
    def __init__(
        self,
        api_key: str = None,
        enable_tracing: bool = False,
    ):
        if api_key is None:
            api_key = os.environ.get("PROMPTLAYER_API_KEY")

        if api_key is None:
            raise ValueError(
                "PromptLayer API key not provided. "
                "Please set the PROMPTLAYER_API_KEY environment variable or pass the api_key parameter."
            )

        self.api_key = api_key
        self.templates = TemplateManager(api_key)
        self.group = GroupManager(api_key)
        self.tracer_provider, self.tracer = self._initialize_tracer(
            api_key, enable_tracing
        )
        self.track = TrackManager(api_key)

    def __getattr__(
        self,
        name: Union[Literal["openai"], Literal["anthropic"], Literal["prompts"]],
    ):
        if name == "openai":
            import openai as openai_module

            openai = PromptLayerBase(
                openai_module,
                function_name="openai",
                api_key=self.api_key,
                tracer=self.tracer,
            )
            return openai
        elif name == "anthropic":
            import anthropic as anthropic_module

            anthropic = PromptLayerBase(
                anthropic_module,
                function_name="anthropic",
                provider_type="anthropic",
                api_key=self.api_key,
                tracer=self.tracer,
            )
            return anthropic
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
    ) -> Dict[str, Any]:
        get_prompt_template_params = self._prepare_get_prompt_template_params(
            prompt_version=prompt_version,
            prompt_release_label=prompt_release_label,
            input_variables=input_variables,
            metadata=metadata,
        )
        prompt_blueprint = self.templates.get(prompt_name, get_prompt_template_params)
        prompt_blueprint_model = self._validate_and_extract_model_from_prompt_blueprint(
            prompt_blueprint=prompt_blueprint, prompt_name=prompt_name
        )
        llm_request_params = self._prepare_llm_request_params(
            prompt_blueprint=prompt_blueprint,
            prompt_template=prompt_blueprint["prompt_template"],
            prompt_blueprint_model=prompt_blueprint_model,
            model_parameter_overrides=model_parameter_overrides,
            stream=stream,
        )

        response = llm_request_params["request_function"](
            llm_request_params["prompt_blueprint"], **llm_request_params["kwargs"]
        )

        if stream:
            return stream_response(
                response,
                self._create_track_request_callable(
                    request_params=llm_request_params,
                    tags=tags,
                    input_variables=input_variables,
                    group_id=group_id,
                    pl_run_span_id=pl_run_span_id,
                ),
                llm_request_params["stream_function"],
            )

        request_log = self._track_request_log(
            llm_request_params,
            tags,
            input_variables,
            group_id,
            pl_run_span_id,
            metadata=metadata,
            request_response=response.model_dump(),
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
        return track_request(**track_request_kwargs)

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
        }

        if self.tracer:
            with self.tracer.start_as_current_span("PromptLayer Run") as span:
                span.set_attribute("prompt_name", prompt_name)
                span.set_attribute("function_input", str(_run_internal_kwargs))
                pl_run_span_id = hex(span.context.span_id)[2:].zfill(16)
                result = self._run_internal(
                    **_run_internal_kwargs, pl_run_span_id=pl_run_span_id
                )
                span.set_attribute("function_output", str(result))
                return result
        else:
            return self._run_internal(**_run_internal_kwargs)

    def run_workflow(
        self,
        workflow_name: str,
        input_variables: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, str]] = None,
        workflow_label_name: Optional[str] = None,
        workflow_version: Optional[
            int
        ] = None,  # This is the version number, not the version ID
        return_all_outputs: Optional[bool] = False,
    ) -> Dict[str, Any]:
        try:
            try:
                # Check if we're inside a running event loop
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                nest_asyncio.apply()
                # If there's an active event loop, use `await` directly
                return asyncio.run(
                    arun_workflow_request(
                        workflow_name=workflow_name,
                        input_variables=input_variables or {},
                        metadata=metadata,
                        workflow_label_name=workflow_label_name,
                        workflow_version_number=workflow_version,
                        api_key=self.api_key,
                        return_all_outputs=return_all_outputs,
                    )
                )
            else:
                # If there's no active event loop, use `asyncio.run()`
                return asyncio.run(
                    arun_workflow_request(
                        workflow_name=workflow_name,
                        input_variables=input_variables or {},
                        metadata=metadata,
                        workflow_label_name=workflow_label_name,
                        workflow_version_number=workflow_version,
                        api_key=self.api_key,
                        return_all_outputs=return_all_outputs,
                    )
                )
        except Exception as e:
            raise Exception(f"Error running workflow: {str(e)}")

    def log_request(
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
    ):
        return util_log_request(
            self.api_key,
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
        )


class AsyncPromptLayer(PromptLayerMixin):
    def __init__(
        self,
        api_key: str = None,
        enable_tracing: bool = False,
    ):
        if api_key is None:
            api_key = os.environ.get("PROMPTLAYER_API_KEY")

        if api_key is None:
            raise ValueError(
                "PromptLayer API key not provided. "
                "Please set the PROMPTLAYER_API_KEY environment variable or pass the api_key parameter."
            )

        self.api_key = api_key
        self.templates = AsyncTemplateManager(api_key)
        self.group = AsyncGroupManager(api_key)
        self.tracer_provider, self.tracer = self._initialize_tracer(
            api_key, enable_tracing
        )
        self.track = AsyncTrackManager(api_key)

    def __getattr__(
        self, name: Union[Literal["openai"], Literal["anthropic"], Literal["prompts"]]
    ):
        if name == "openai":
            import openai as openai_module

            openai = PromptLayerBase(
                openai_module,
                function_name="openai",
                api_key=self.api_key,
            )
            return openai
        elif name == "anthropic":
            import anthropic as anthropic_module

            anthropic = PromptLayerBase(
                anthropic_module,
                function_name="anthropic",
                provider_type="anthropic",
                api_key=self.api_key,
            )
            return anthropic
        else:
            raise AttributeError(f"module {__name__} has no attribute {name}")

    async def run_workflow(
        self,
        workflow_name: str,
        input_variables: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, str]] = None,
        workflow_label_name: Optional[str] = None,
        workflow_version: Optional[
            int
        ] = None,  # This is the version number, not the version ID
        return_all_outputs: Optional[bool] = False,
    ) -> Dict[str, Any]:
        try:
            result = await arun_workflow_request(
                workflow_name=workflow_name,
                input_variables=input_variables or {},
                metadata=metadata,
                workflow_label_name=workflow_label_name,
                workflow_version_number=workflow_version,
                api_key=self.api_key,
                return_all_outputs=return_all_outputs,
            )
            return result
        except Exception as e:
            raise Exception(f"Error running workflow: {str(e)}")

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
        }

        if self.tracer:
            with self.tracer.start_as_current_span("PromptLayer Run") as span:
                span.set_attribute("prompt_name", prompt_name)
                span.set_attribute("function_input", str(_run_internal_kwargs))
                pl_run_span_id = hex(span.context.span_id)[2:].zfill(16)
                result = await self._run_internal(
                    **_run_internal_kwargs, pl_run_span_id=pl_run_span_id
                )
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
    ):
        return await autil_log_request(
            self.api_key,
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
            return await atrack_request(**track_request_kwargs)

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
        return await atrack_request(**track_request_kwargs)

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
    ) -> Dict[str, Any]:
        get_prompt_template_params = self._prepare_get_prompt_template_params(
            prompt_version=prompt_version,
            prompt_release_label=prompt_release_label,
            input_variables=input_variables,
            metadata=metadata,
        )
        prompt_blueprint = await self.templates.get(
            prompt_name, get_prompt_template_params
        )
        prompt_blueprint_model = self._validate_and_extract_model_from_prompt_blueprint(
            prompt_blueprint=prompt_blueprint, prompt_name=prompt_name
        )
        llm_request_params = self._prepare_llm_request_params(
            prompt_blueprint=prompt_blueprint,
            prompt_template=prompt_blueprint["prompt_template"],
            prompt_blueprint_model=prompt_blueprint_model,
            model_parameter_overrides=model_parameter_overrides,
            stream=stream,
            is_async=True,
        )

        response = await llm_request_params["request_function"](
            llm_request_params["prompt_blueprint"], **llm_request_params["kwargs"]
        )

        if stream:
            track_request_callable = await self._create_track_request_callable(
                request_params=llm_request_params,
                tags=tags,
                input_variables=input_variables,
                group_id=group_id,
                pl_run_span_id=pl_run_span_id,
            )
            return astream_response(
                response,
                track_request_callable,
                llm_request_params["stream_function"],
            )

        request_log = await self._track_request_log(
            llm_request_params,
            tags,
            input_variables,
            group_id,
            pl_run_span_id,
            metadata=metadata,
            request_response=response.model_dump(),
        )

        return {
            "request_id": request_log.get("request_id", None),
            "raw_response": response,
            "prompt_blueprint": request_log.get("prompt_blueprint", None),
        }
