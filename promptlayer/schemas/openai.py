from typing import Literal, Union
from typing_extensions import Annotated

from pydantic import BaseModel, constr, validator


class FunctionCall(BaseModel):
    name: str
    arguments: str


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "function"]
    function_call: Union[FunctionCall, None] = None
    content: str = ""
    name: Union[
        Annotated[str, constr(max_length=64, regex="^[a-zA-Z0-9_-]{1,64}$")], None
    ] = None

    @validator("content")
    def validate_content(cls, value, values):
        is_role_assistant = values["role"] == "assistant"
        is_function_call = (
            "function_call" in values and values["function_call"] is not None
        )
        is_assistant_function_call = is_role_assistant and is_function_call
        if is_assistant_function_call and value is None:
            return value
        if value is None:
            raise ValueError("Message content must be specified")
        return value

    @validator("name")
    def validate_name(cls, value, values):
        is_role_function = values["role"] == "function"
        if is_role_function and not value:
            raise ValueError("Function messages must have a name")
        return value


class Function(BaseModel):
    name: str
    description: str = ""
    parameters: dict = {"type": "object", "properties": {}}
