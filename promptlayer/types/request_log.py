from typing import TypedDict, Union

from .prompt_template import PromptBlueprint


class RequestLog(TypedDict):
    id: int
    prompt_version: Union[PromptBlueprint, None]
