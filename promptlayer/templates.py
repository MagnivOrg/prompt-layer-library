from typing import Union

from promptlayer.types.prompt_template import GetPromptTemplate
from promptlayer.utils import get_prompt_template


def get(prompt_name: str, params: Union[GetPromptTemplate, None] = None):
    return get_prompt_template(prompt_name, params)
