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
    def __init__(self, api_key: str):
        self.api_key = api_key

    def get(self, prompt_name: str, params: Union[GetPromptTemplate, None] = None):
        return get_prompt_template(prompt_name, params, self.api_key)

    def publish(self, body: PublishPromptTemplate):
        return publish_prompt_template(body, self.api_key)

    def all(self, page: int = 1, per_page: int = 30):
        return get_all_prompt_templates(page, per_page, self.api_key)


class AsyncTemplateManager:
    def __init__(self, api_key: str):
        self.api_key = api_key

    async def get(
        self, prompt_name: str, params: Union[GetPromptTemplate, None] = None
    ):
        return await aget_prompt_template(prompt_name, params, self.api_key)

    async def all(self, page: int = 1, per_page: int = 30):
        return await aget_all_prompt_templates(page, per_page, self.api_key)
