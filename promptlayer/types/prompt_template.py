from typing import Dict, Literal, Sequence, TypedDict, Union


class GetPromptTemplate(TypedDict, total=False):
    version: int
    label: str
    provider: str
    input_variables: Dict[str, str]


TemplateFormat = Literal["f-string", "jinja2"]


class ImageUrl(TypedDict, total=False):
    url: str


class TextContent(TypedDict, total=False):
    type: Literal["text"]
    text: str


class ImageContent(TypedDict, total=False):
    type: Literal["image_url"]
    image_url: ImageUrl


Content = Union[TextContent, ImageContent]


class Function(TypedDict, total=False):
    name: str
    description: str
    parameters: dict


class FunctionCall(TypedDict, total=False):
    name: str
    arguments: str


class SystemMessage(TypedDict, total=False):
    role: Literal["system"]
    template_format: Union[TemplateFormat, None]
    content: Sequence[Content]
    name: Union[str, None]


class UserMessage(TypedDict, total=False):
    role: Literal["user"]
    template_format: Union[TemplateFormat, None]
    content: Sequence[Content]
    name: Union[str, None]


class AssistantMessage(TypedDict, total=False):
    role: Literal["assistant"]
    template_format: Union[TemplateFormat, None]
    content: Union[Sequence[Content], None]
    function_call: Union[FunctionCall, None]
    name: Union[str, None]


class FunctionMessage(TypedDict, total=False):
    role: Literal["function"]
    template_format: Union[TemplateFormat, None]
    content: Union[Sequence[Content], None]
    name: str


class ChatFunctionCall(TypedDict, total=False):
    name: str


Message = Union[SystemMessage, UserMessage, AssistantMessage, FunctionMessage]


class CompletionPromptTemplate(TypedDict, total=False):
    type: Literal["completion"]
    template_format: Union[TemplateFormat, None]
    content: Sequence[Content]


class ChatPromptTemplate(TypedDict, total=False):
    type: Literal["chat"]
    messages: Sequence[Message]
    functions: Union[Sequence[Function], None]
    function_call: Union[Literal["auto", "none"], ChatFunctionCall, None]


PromptTemplate = Union[CompletionPromptTemplate, ChatPromptTemplate]


class PublishPromptTemplate(TypedDict, total=False):
    prompt_name: str
    prompt_template: PromptTemplate


class PublishPromptTemplateResponse(PublishPromptTemplate):
    id: int
