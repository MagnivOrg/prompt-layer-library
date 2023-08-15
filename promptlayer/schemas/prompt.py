from typing import Dict, Literal, Union, List
from typing_extensions import Annotated

from langchain.prompts import ChatMessagePromptTemplate
from langchain.prompts import ChatPromptTemplate as BaseChatPromptTemplate
from langchain.prompts import PromptTemplate as BasePromptTemplate
from pydantic import BaseModel, constr, root_validator, validator

from promptlayer.schemas import openai

# PromptTemplate from langchain calls root_validator which doesn't skip on failure
class SafePromptTemplate(BasePromptTemplate):
    @root_validator(skip_on_failure=True)
    def template_is_valid(cls, values: Dict) -> Dict:
        return super().template_is_valid(values)


class BaseChatMessagePromptTemplate(ChatMessagePromptTemplate):
    prompt: BasePromptTemplate


class AIMessagePromptTemplate(BaseChatMessagePromptTemplate):
    role = "assistant"


class HumanMessagePromptTemplate(BaseChatMessagePromptTemplate):
    role = "user"


class SystemMessagePromptTemplate(BaseChatMessagePromptTemplate):
    role = "system"


Message = Union[BaseChatMessagePromptTemplate, openai.Message]


class ChatPromptTemplate(BaseChatPromptTemplate):
    messages: List[Message]
    functions: Union[List[openai.Function], None] = None
    function_call: Union[Literal["none", "auto"], Dict[Literal["name"], str]] = "none"

    @property
    def _prompt_type(self) -> str:
        return "chat_promptlayer_langchain"

    @root_validator(pre=True)
    def validate_input_variables(cls, values: Dict) -> Dict:
        if "messages" not in values:
            return values
        messages = values["messages"]
        input_vars = set()
        for message in messages:
            try:
                message = BaseChatMessagePromptTemplate.parse_obj(message)
            except ValueError:
                openai.Message.parse_obj(message)
                continue
            input_vars.update(message.prompt.input_variables)
        if "partial_variables" in values:
            input_vars = input_vars - set(values["partial_variables"])
        if "input_variables" in values:
            if input_vars != set(values["input_variables"]):
                raise ValueError(
                    "Got mismatched input_variables. "
                    f"Expected: {input_vars}. "
                    f"Got: {values['input_variables']}"
                )
        else:
            values["input_variables"] = list(input_vars)
        return values

    @validator("function_call", pre=True)
    def validate_function_call(cls, value, values: dict):
        functions = values.get("functions", [])
        if isinstance(value, dict):
            if "name" not in value:
                raise ValueError("function_call must have a name")
            if value["name"] not in [f.name for f in functions]:
                raise ValueError(
                    f"function_call name {value['name']} not found in functions"
                )
            return value
        if value == "auto":
            if functions is None or len(functions) == 0:
                raise ValueError("Cannot set function_call to auto with no functions")
            return value
        if value == "none":
            return value
        raise ValueError(
            f"function_call must be one of 'none', 'auto', or a dict with a name. Got {value}"
        )


PromptTemplate = Union[ChatPromptTemplate, SafePromptTemplate]


class Base(BaseModel):
    prompt_name: Annotated[
        str, constr(min_length=1, max_length=128, regex="^[a-zA-Z0-9_-]*$")
    ]
    prompt_template: PromptTemplate
    tags: List[str] = []
