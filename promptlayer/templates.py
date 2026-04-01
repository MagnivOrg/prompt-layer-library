import logging
from typing import Union

from promptlayer import exceptions as _exceptions
from promptlayer.span_exporter import set_prompt_span_attributes
from promptlayer.template_cache import (
    PromptTemplateCache,
    is_locally_renderable,
    make_cache_params,
    render_response,
    should_skip_cache,
)
from promptlayer.types.prompt_template import GetPromptTemplate, PublishPromptTemplate
from promptlayer.utils import (
    aget_all_prompt_templates,
    aget_prompt_template,
    get_all_prompt_templates,
    get_prompt_template,
    publish_prompt_template,
)

logger = logging.getLogger(__name__)

_TRANSIENT_ERRORS = (
    _exceptions.PromptLayerInternalServerError,
    _exceptions.PromptLayerAPIConnectionError,
    _exceptions.PromptLayerAPITimeoutError,
)


def _extract_label(params):
    if isinstance(params, dict):
        return params.get("label")
    return getattr(params, "label", None)


class TemplateManager:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        throw_on_error: bool,
        cache: Union[PromptTemplateCache, None] = None,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.throw_on_error = throw_on_error
        self._cache = cache

    def get(self, prompt_name: str, params: Union[GetPromptTemplate, None] = None):
        if self._cache and not should_skip_cache(params):
            return self._get_with_cache(prompt_name, params)
        return self._fetch_normal(prompt_name, params)

    def _fetch_normal(self, prompt_name, params):
        result = get_prompt_template(self.api_key, self.base_url, self.throw_on_error, prompt_name, params)
        if result:
            set_prompt_span_attributes(result, prompt_name, label=_extract_label(params))
        return result

    def _get_with_cache(self, prompt_name, params):
        cache_key = self._cache.make_key(prompt_name, params)
        input_variables = params.get("input_variables") if params else None
        label = _extract_label(params)

        if self._cache.is_non_renderable(cache_key):
            return self._fetch_normal(prompt_name, params)

        cached, is_fresh = self._cache.get(cache_key)

        if cached is not None and is_fresh:
            result = render_response(cached, input_variables)
            set_prompt_span_attributes(result, prompt_name, label=label)
            return result

        stale = cached

        cache_params = make_cache_params(params)
        try:
            api_result = get_prompt_template(self.api_key, self.base_url, True, prompt_name, cache_params)
        except _TRANSIENT_ERRORS:
            if stale is not None:
                logger.debug("Transient API error, serving stale cache for '%s'", prompt_name)
                result = render_response(stale, input_variables)
                set_prompt_span_attributes(result, prompt_name, label=label)
                return result
            if not self.throw_on_error:
                return None
            raise
        except _exceptions.PromptLayerError:
            if not self.throw_on_error:
                return None
            raise

        if api_result is None:
            return None

        if not is_locally_renderable(api_result):
            self._cache.mark_non_renderable(cache_key)
            return self._fetch_normal(prompt_name, params)

        self._cache.put(cache_key, api_result)
        result = render_response(api_result, input_variables)
        set_prompt_span_attributes(result, prompt_name, label=label)
        return result

    def publish(self, body: PublishPromptTemplate):
        result = publish_prompt_template(self.api_key, self.base_url, self.throw_on_error, body)
        if self._cache and result:
            prompt_name = body.get("prompt_name") if isinstance(body, dict) else getattr(body, "prompt_name", None)
            if prompt_name:
                self._cache.invalidate(prompt_name)
        return result

    def all(self, page: int = 1, per_page: int = 30, label: str = None):
        return get_all_prompt_templates(self.api_key, self.base_url, self.throw_on_error, page, per_page, label)


class AsyncTemplateManager:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        throw_on_error: bool,
        cache: Union[PromptTemplateCache, None] = None,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.throw_on_error = throw_on_error
        self._cache = cache

    async def get(self, prompt_name: str, params: Union[GetPromptTemplate, None] = None):
        if self._cache and not should_skip_cache(params):
            return await self._aget_with_cache(prompt_name, params)
        return await self._afetch_normal(prompt_name, params)

    async def _afetch_normal(self, prompt_name, params):
        result = await aget_prompt_template(self.api_key, self.base_url, self.throw_on_error, prompt_name, params)
        if result:
            set_prompt_span_attributes(result, prompt_name, label=_extract_label(params))
        return result

    async def _aget_with_cache(self, prompt_name, params):
        cache_key = self._cache.make_key(prompt_name, params)
        input_variables = params.get("input_variables") if params else None
        label = _extract_label(params)

        if self._cache.is_non_renderable(cache_key):
            return await self._afetch_normal(prompt_name, params)

        cached, is_fresh = self._cache.get(cache_key)

        if cached is not None and is_fresh:
            result = render_response(cached, input_variables)
            set_prompt_span_attributes(result, prompt_name, label=label)
            return result

        stale = cached

        cache_params = make_cache_params(params)
        try:
            api_result = await aget_prompt_template(self.api_key, self.base_url, True, prompt_name, cache_params)
        except _TRANSIENT_ERRORS:
            if stale is not None:
                logger.debug("Transient API error, serving stale cache for '%s'", prompt_name)
                result = render_response(stale, input_variables)
                set_prompt_span_attributes(result, prompt_name, label=label)
                return result
            if not self.throw_on_error:
                return None
            raise
        except _exceptions.PromptLayerError:
            if not self.throw_on_error:
                return None
            raise

        if api_result is None:
            return None

        if not is_locally_renderable(api_result):
            self._cache.mark_non_renderable(cache_key)
            return await self._afetch_normal(prompt_name, params)

        self._cache.put(cache_key, api_result)
        result = render_response(api_result, input_variables)
        set_prompt_span_attributes(result, prompt_name, label=label)
        return result

    async def all(self, page: int = 1, per_page: int = 30, label: str = None):
        return await aget_all_prompt_templates(self.api_key, self.base_url, self.throw_on_error, page, per_page, label)
