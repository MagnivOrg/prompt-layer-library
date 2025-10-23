from typing import Union

from promptlayer.types.prompt_template import GetPromptTemplate, PublishPromptTemplate
from promptlayer.utils import (
    aget_all_prompt_templates,
    aget_prompt_template,
    get_all_prompt_templates,
    get_prompt_template,
    publish_prompt_template,
)


class TemplateManager:
    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url

    def get(self, prompt_name: str, params: Union[GetPromptTemplate, None] = None):
        return get_prompt_template(self.api_key, self.base_url, prompt_name, params)

    def publish(self, body: PublishPromptTemplate):
        return publish_prompt_template(self.api_key, self.base_url, body)

    def all(self, page: int = 1, per_page: int = 30, label: str = None):
        return get_all_prompt_templates(self.api_key, self.base_url, page, per_page, label)


class AsyncTemplateManager:
    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url

    async def get(self, prompt_name: str, params: Union[GetPromptTemplate, None] = None):
        return await aget_prompt_template(self.api_key, self.base_url, prompt_name, params)

    async def all(self, page: int = 1, per_page: int = 30, label: str = None):
        return await aget_all_prompt_templates(self.api_key, self.base_url, page, per_page, label)
