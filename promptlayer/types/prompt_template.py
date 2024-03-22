from typing import Dict, List, Literal, Sequence, TypedDict, Union

from typing_extensions import NotRequired


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


class Tool(TypedDict, total=False):
    type: Literal["function"]
    function: Function


class FunctionCall(TypedDict, total=False):
    name: str
    arguments: str


class SystemMessage(TypedDict, total=False):
    role: Literal["system"]
    input_variables: NotRequired[List[str]]
    template_format: NotRequired[TemplateFormat]
    content: Sequence[Content]
    name: NotRequired[str]


class UserMessage(TypedDict, total=False):
    role: Literal["user"]
    input_variables: NotRequired[List[str]]
    template_format: NotRequired[TemplateFormat]
    content: Sequence[Content]
    name: NotRequired[str]


class ToolCall(TypedDict, total=False):
    id: str
    type: Literal["function"]
    function: FunctionCall


class AssistantMessage(TypedDict, total=False):
    role: Literal["assistant"]
    input_variables: NotRequired[List[str]]
    template_format: NotRequired[TemplateFormat]
    content: NotRequired[Sequence[Content]]
    function_call: NotRequired[FunctionCall]
    name: NotRequired[str]
    tool_calls: NotRequired[List[ToolCall]]


class FunctionMessage(TypedDict, total=False):
    role: Literal["function"]
    input_variables: NotRequired[List[str]]
    template_format: NotRequired[TemplateFormat]
    content: NotRequired[Sequence[Content]]
    name: str


class ToolMessage(TypedDict, total=False):
    role: Literal["tool"]
    input_variables: NotRequired[List[str]]
    template_format: NotRequired[TemplateFormat]
    content: Sequence[Content]
    tool_call_id: str
    name: NotRequired[str]


class ChatFunctionCall(TypedDict, total=False):
    name: str


class ChatToolChoice(TypedDict, total=False):
    type: Literal["function"]
    function: ChatFunctionCall


ToolChoice = Union[str, ChatToolChoice]

Message = Union[
    SystemMessage, UserMessage, AssistantMessage, FunctionMessage, ToolMessage
]


class CompletionPromptTemplate(TypedDict, total=False):
    type: Literal["completion"]
    template_format: NotRequired[TemplateFormat]
    content: Sequence[Content]
    input_variables: NotRequired[List[str]]


class ChatPromptTemplate(TypedDict, total=False):
    type: Literal["chat"]
    messages: Sequence[Message]
    functions: NotRequired[Sequence[Function]]
    function_call: NotRequired[Union[Literal["auto", "none"], ChatFunctionCall]]
    input_variables: NotRequired[List[str]]
    tools: NotRequired[Sequence[Tool]]
    tool_choice: NotRequired[ToolChoice]


PromptTemplate = Union[CompletionPromptTemplate, ChatPromptTemplate]


class Model(TypedDict, total=False):
    provider: str
    name: str
    parameters: Dict[str, object]


class Metadata(TypedDict, total=False):
    model: NotRequired[Model]


class BasePromptTemplate(TypedDict, total=False):
    prompt_name: str
    tags: NotRequired[List[str]]


class PromptVersion(TypedDict, total=False):
    prompt_template: PromptTemplate
    commit_message: NotRequired[str]
    metadata: NotRequired[Metadata]


class PublishPromptTemplate(BasePromptTemplate, PromptVersion):
    pass


class BasePromptTemplateResponse(TypedDict, total=False):
    id: int
    prompt_name: str
    tags: List[str]
    prompt_template: PromptTemplate
    commit_message: NotRequired[str]
    metadata: NotRequired[Metadata]


class PublishPromptTemplateResponse(TypedDict, total=False):
    pass


class GetPromptTemplateResponse(BasePromptTemplateResponse):
    llm_kwargs: NotRequired[Dict[str, object]]
    version: int


class ListPromptTemplateResponse(BasePromptTemplateResponse, total=False):
    version: int
