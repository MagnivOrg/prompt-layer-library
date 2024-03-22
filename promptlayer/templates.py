from typing import Union

from promptlayer.types.prompt_template import GetPromptTemplate, PublishPromptTemplate
from promptlayer.utils import (
    get_all_prompt_templates,
    get_prompt_template,
    publish_prompt_template,
)


def get(prompt_name: str, params: Union[GetPromptTemplate, None] = None):
    return get_prompt_template(prompt_name, params)


def publish(body: PublishPromptTemplate):
    return publish_prompt_template(body)


def all(page: int = 1, per_page: int = 30):
    return get_all_prompt_templates(page, per_page)
