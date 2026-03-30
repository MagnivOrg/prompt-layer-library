from typing import Union

from promptlayer.span_exporter import set_prompt_span_attributes
from promptlayer.types.prompt_template import GetPromptTemplate, PublishPromptTemplate
from promptlayer.utils import (
    aget_all_prompt_templates,
    aget_prompt_template,
    get_all_prompt_templates,
    get_prompt_template,
    publish_prompt_template,
)


class TemplateManager:
    def __init__(self, api_key: str, base_url: str, throw_on_error: bool):
        self.api_key = api_key
        self.base_url = base_url
        self.throw_on_error = throw_on_error

    def get(
        self,
        prompt_name: str,
        params: Union[GetPromptTemplate, None] = None,
        cache_ttl_seconds: Union[int, None] = None,
    ):
        """
        Get a prompt template from PromptLayer.

        Args:
            prompt_name: Name or numeric ID of the prompt template
            params: Optional parameters for getting the template
            cache_ttl_seconds: Cache TTL in seconds. If 0 or None, caching is disabled.
                If > 0, the template will be cached for the specified duration.
                Expired cache entries are served as fallback when API is unreachable.
                Cache is automatically bypassed when metadata_filters or provider is specified.

        Returns:
            Prompt template response dict

        Example:
            # Enable 15-minute cache
            template = client.templates.get("my-prompt", cache_ttl_seconds=900)

            # Disable cache (default behavior)
            template = client.templates.get("my-prompt")
        """
        result = get_prompt_template(
            self.api_key, self.base_url, self.throw_on_error, prompt_name, params,
            cache_ttl_seconds=cache_ttl_seconds,
        )
        if result:
            label = params.get("label") if isinstance(params, dict) else getattr(params, "label", None)
            set_prompt_span_attributes(result, prompt_name, label=label)
        return result

    def publish(self, body: PublishPromptTemplate):
        return publish_prompt_template(self.api_key, self.base_url, self.throw_on_error, body)

    def all(self, page: int = 1, per_page: int = 30, label: str = None):
        return get_all_prompt_templates(self.api_key, self.base_url, self.throw_on_error, page, per_page, label)


class AsyncTemplateManager:
    def __init__(self, api_key: str, base_url: str, throw_on_error: bool):
        self.api_key = api_key
        self.base_url = base_url
        self.throw_on_error = throw_on_error

    async def get(
        self,
        prompt_name: str,
        params: Union[GetPromptTemplate, None] = None,
        cache_ttl_seconds: Union[int, None] = None,
    ):
        """
        Get a prompt template from PromptLayer (async version).

        Args:
            prompt_name: Name or numeric ID of the prompt template
            params: Optional parameters for getting the template
            cache_ttl_seconds: Cache TTL in seconds. If 0 or None, caching is disabled.
                If > 0, the template will be cached for the specified duration.
                Expired cache entries are served as fallback when API is unreachable.
                Cache is automatically bypassed when metadata_filters or provider is specified.

        Returns:
            Prompt template response dict

        Example:
            # Enable 15-minute cache
            template = await client.templates.get("my-prompt", cache_ttl_seconds=900)

            # Disable cache (default behavior)
            template = await client.templates.get("my-prompt")
        """
        result = await aget_prompt_template(
            self.api_key, self.base_url, self.throw_on_error, prompt_name, params,
            cache_ttl_seconds=cache_ttl_seconds,
        )
        if result:
            label = params.get("label") if isinstance(params, dict) else getattr(params, "label", None)
            set_prompt_span_attributes(result, prompt_name, label=label)
        return result

    async def all(self, page: int = 1, per_page: int = 30, label: str = None):
        return await aget_all_prompt_templates(self.api_key, self.base_url, self.throw_on_error, page, per_page, label)
