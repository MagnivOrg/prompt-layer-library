from typing import Any, Dict, List, Literal, Optional, Sequence, TypedDict, Union

from typing_extensions import Required


class GetPromptTemplate(TypedDict, total=False):
    version: int
    label: str
    provider: str
    input_variables: Dict[str, Any]
    metadata_filters: Dict[str, str]
    skip_input_variable_rendering: bool


TemplateFormat = Literal["f-string", "jinja2"]


class ImageUrl(TypedDict, total=False):
    url: str


class WebAnnotation(TypedDict, total=False):
    type: Literal["url_citation"]
    title: str
    url: str
    start_index: int
    end_index: int


class FileAnnotation(TypedDict, total=False):
    type: Literal["file_citation"]
    index: int
    file_id: str
    filename: str


class MapAnnotation(TypedDict, total=False):
    type: Literal["map_citation"]
    title: str
    url: str
    place_id: Union[str, None]
    start_index: int
    end_index: int


class ContainerFileAnnotation(TypedDict, total=False):
    type: Literal["container_file_citation"]
    container_id: str
    start_index: Union[int, None]
    end_index: Union[int, None]
    filename: Union[str, None]
    file_id: Union[str, None]


Annotation = Union[WebAnnotation, FileAnnotation, MapAnnotation, ContainerFileAnnotation]


class TextContent(TypedDict, total=False):
    type: Literal["text"]
    text: str
    id: Union[str, None]
    annotations: Union[List[Annotation], None]


class CodeContent(TypedDict, total=False):
    type: Literal["code"]
    code: str
    id: Union[str, None]
    container_id: Union[str, None]
    language: Union[str, None]


class ThinkingContent(TypedDict, total=False):
    signature: Union[str, None]
    type: Literal["thinking"]
    thinking: str
    id: Union[str, None]


class ImageContent(TypedDict, total=False):
    type: Literal["image_url"]
    image_url: ImageUrl


class Media(TypedDict, total=False):
    title: str
    type: str
    url: str


class MediaContnt(TypedDict, total=False):
    type: Literal["media"]
    media: Media


class MediaVariable(TypedDict, total=False):
    type: Literal["media_variable"]
    name: str


class OutputMediaContent(TypedDict, total=False):
    type: Literal["output_media"]
    id: Union[str, None]
    url: str
    mime_type: str
    media_type: str
    provider_metadata: Union[Dict[str, Any], None]


class ServerToolUseContent(TypedDict, total=False):
    type: Literal["server_tool_use"]
    id: str
    name: str
    input: dict


class WebSearchResult(TypedDict, total=False):
    type: Literal["web_search_result"]
    url: str
    title: str
    encrypted_content: str
    page_age: Union[str, None]


class WebSearchToolResultContent(TypedDict, total=False):
    type: Literal["web_search_tool_result"]
    tool_use_id: str
    content: List[Dict[str, Any]]


class BashCodeExecutionToolResultContent(TypedDict, total=False):
    type: Literal["bash_code_execution_tool_result"]
    tool_use_id: str
    content: Dict[str, Any]


class TextEditorCodeExecutionToolResultContent(TypedDict, total=False):
    type: Literal["text_editor_code_execution_tool_result"]
    tool_use_id: str
    content: Dict[str, Any]


class CodeExecutionResultContent(TypedDict, total=False):
    type: Literal["code_execution_result"]
    output: str
    outcome: str


class ShellCallContent(TypedDict, total=False):
    type: Literal["shell_call"]
    id: Union[str, None]
    call_id: Union[str, None]
    action: Dict[str, Any]
    status: Union[str, None]


class ShellCallOutputContent(TypedDict, total=False):
    type: Literal["shell_call_output"]
    id: Union[str, None]
    call_id: Union[str, None]
    output: List[Dict[str, Any]]
    status: Union[str, None]


class ApplyPatchCallContent(TypedDict, total=False):
    type: Literal["apply_patch_call"]
    id: Union[str, None]
    call_id: Union[str, None]
    operation: Dict[str, Any]
    status: Union[str, None]


class ApplyPatchCallOutputContent(TypedDict, total=False):
    type: Literal["apply_patch_call_output"]
    id: Union[str, None]
    call_id: Union[str, None]
    output: Union[str, None]
    status: Union[str, None]


class McpListToolsContent(TypedDict, total=False):
    type: Literal["mcp_list_tools"]
    id: Union[str, None]
    server_label: str
    tools: List[Dict[str, Any]]
    error: Union[str, Dict[str, Any], None]


class McpCallContent(TypedDict, total=False):
    type: Literal["mcp_call"]
    id: Union[str, None]
    call_id: Union[str, None]
    name: str
    server_label: str
    arguments: str
    output: Union[str, None]
    error: Union[str, Dict[str, Any], None]
    approval_request_id: Union[str, None]
    status: Union[str, None]


class McpApprovalRequestContent(TypedDict, total=False):
    type: Literal["mcp_approval_request"]
    id: Union[str, None]
    name: str
    arguments: str
    server_label: str


class McpApprovalResponseContent(TypedDict, total=False):
    type: Literal["mcp_approval_response"]
    approval_request_id: str
    approve: bool


Content = Union[
    TextContent,
    ThinkingContent,
    CodeContent,
    ImageContent,
    MediaContnt,
    MediaVariable,
    OutputMediaContent,
    ServerToolUseContent,
    WebSearchToolResultContent,
    CodeExecutionResultContent,
    McpListToolsContent,
    McpCallContent,
    McpApprovalRequestContent,
    McpApprovalResponseContent,
    BashCodeExecutionToolResultContent,
    TextEditorCodeExecutionToolResultContent,
    ShellCallContent,
    ShellCallOutputContent,
    ApplyPatchCallContent,
    ApplyPatchCallOutputContent,
]


class Function(TypedDict, total=False):
    name: str
    description: str
    parameters: dict


class Tool(TypedDict, total=False):
    type: Literal["function"]
    function: Function


class BuiltInTool(TypedDict, total=False):
    type: str
    name: str


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
    tool_id: Union[str, None]
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


class PlaceholderMessage(TypedDict, total=False):
    role: Literal["placeholder"]
    name: str


class DeveloperMessage(TypedDict, total=False):
    role: Literal["developer"]
    input_variables: List[str]
    template_format: TemplateFormat
    content: Sequence[Content]


class ChatFunctionCall(TypedDict, total=False):
    name: str


class ChatToolChoice(TypedDict, total=False):
    type: Literal["function"]
    function: ChatFunctionCall


ToolChoice = Union[str, ChatToolChoice]

Message = Union[
    SystemMessage,
    UserMessage,
    AssistantMessage,
    FunctionMessage,
    ToolMessage,
    PlaceholderMessage,
    DeveloperMessage,
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
    tools: Sequence[Union[Tool, BuiltInTool]]
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


class PromptBlueprint(TypedDict, total=False):
    prompt_template: PromptTemplate
    commit_message: str
    metadata: Metadata


class PublishPromptTemplate(BasePromptTemplate, PromptBlueprint, total=False):
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
