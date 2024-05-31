from typing import Dict, List, Literal, Optional, Sequence, TypedDict, Union

from typing_extensions import Required


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
    input_variables: List[str]
    template_format: TemplateFormat
    content: Sequence[Content]
    name: str


class UserMessage(TypedDict, total=False):
    role: Literal["user"]
    input_variables: List[str]
    template_format: TemplateFormat
    content: Sequence[Content]
    name: str


class ToolCall(TypedDict, total=False):
    id: str
    type: Literal["function"]
    function: FunctionCall


class AssistantMessage(TypedDict, total=False):
    role: Literal["assistant"]
    input_variables: List[str]
    template_format: TemplateFormat
    content: Sequence[Content]
    function_call: FunctionCall
    name: str
    tool_calls: List[ToolCall]


class FunctionMessage(TypedDict, total=False):
    role: Literal["function"]
    input_variables: List[str]
    template_format: TemplateFormat
    content: Sequence[Content]
    name: str


class ToolMessage(TypedDict, total=False):
    role: Literal["tool"]
    input_variables: List[str]
    template_format: TemplateFormat
    content: Sequence[Content]
    tool_call_id: str
    name: str


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
    type: Required[Literal["completion"]]
    template_format: TemplateFormat
    content: Sequence[Content]
    input_variables: List[str]


class ChatPromptTemplate(TypedDict, total=False):
    type: Required[Literal["chat"]]
    messages: Required[Sequence[Message]]
    functions: Sequence[Function]
    function_call: Union[Literal["auto", "none"], ChatFunctionCall]
    input_variables: List[str]
    tools: Sequence[Tool]
    tool_choice: ToolChoice


PromptTemplate = Union[CompletionPromptTemplate, ChatPromptTemplate]


class Model(TypedDict, total=False):
    provider: Required[str]
    name: Required[str]
    parameters: Required[Dict[str, object]]


class Metadata(TypedDict, total=False):
    model: Model


class BasePromptTemplate(TypedDict, total=False):
    prompt_name: str
    tags: List[str]


class PromptVersion(TypedDict, total=False):
    prompt_template: PromptTemplate
    commit_message: str
    metadata: Metadata


class PublishPromptTemplate(BasePromptTemplate, PromptVersion, total=False):
    release_labels: Optional[List[str]] = None


class BaseProviderBaseURL(TypedDict):
    name: Required[str]
    provider: Required[str]
    url: Required[str]


class ProviderBaseURL(BaseProviderBaseURL):
    id: Required[int]


class BasePromptTemplateResponse(TypedDict, total=False):
    id: Required[int]
    prompt_name: Required[str]
    tags: List[str]
    prompt_template: Required[PromptTemplate]
    commit_message: str
    metadata: Metadata
    provider_base_url: ProviderBaseURL


a: BasePromptTemplateResponse = {"provider_base_url": {"url": ""}}


class PublishPromptTemplateResponse(BasePromptTemplateResponse):
    pass


class GetPromptTemplateResponse(BasePromptTemplateResponse):
    llm_kwargs: Union[Dict[str, object], None]
    version: int


class ListPromptTemplateResponse(BasePromptTemplateResponse, total=False):
    version: int
