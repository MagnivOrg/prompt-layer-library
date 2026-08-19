from typing import Any, Dict, List, Literal, Optional, Sequence, TypedDict, Union

from typing_extensions import Required


class GetPromptTemplate(TypedDict, total=False):
    version: int
    label: str
    provider: str
    model: str
    input_variables: Dict[str, Any]
    metadata_filters: Dict[str, str]
    model_parameter_overrides: Dict[str, Any]
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


class OpenAIWebSearchToolConfig(TypedDict, total=False):
    type: Literal["web_search", "web_search_2025_08_26"]
    filters: Dict[str, Any]
    search_context_size: Literal["low", "medium", "high"]
    user_location: Dict[str, Any]


class FileSearchToolConfig(TypedDict, total=False):
    type: Literal["file_search"]
    vector_store_ids: List[str]
    filters: Dict[str, Any]
    max_num_results: int
    ranking_options: Dict[str, Any]


class CodeInterpreterToolConfig(TypedDict, total=False):
    type: Literal["code_interpreter"]
    container: Dict[str, Any]


class ImageGenerationToolConfig(TypedDict, total=False):
    type: Literal["image_generation"]
    action: Literal["generate", "edit", "auto"]
    background: Literal["transparent", "opaque", "auto"]
    input_fidelity: Optional[Literal["high", "low"]]
    input_image_mask: Dict[str, str]
    model: str
    moderation: Literal["auto", "low"]
    output_compression: int
    output_format: Literal["png", "webp", "jpeg"]
    partial_images: int
    quality: Literal["low", "medium", "high", "auto"]
    size: Literal["1024x1024", "1024x1536", "1536x1024", "auto"]


class McpToolApprovalFilter(TypedDict, total=False):
    tool_names: List[str]


class McpToolApproval(TypedDict, total=False):
    never: McpToolApprovalFilter
    always: McpToolApprovalFilter


class OpenAINativeMcpToolConfig(TypedDict, total=False):
    type: Literal["mcp"]
    server_label: str
    server_url: str
    server_description: str
    connector_id: str
    authorization: str
    headers: Dict[str, str]
    allowed_tools: List[str]
    require_approval: Union[Literal["always", "never"], McpToolApproval]


class ShellToolConfig(TypedDict, total=False):
    type: Literal["shell"]
    environment: Dict[str, Any]


class ApplyPatchToolConfig(TypedDict, total=False):
    type: Literal["apply_patch"]


class OpenRouterWebSearchToolConfig(TypedDict, total=False):
    id: Literal["web"]
    engine: Literal["native", "exa", "firecrawl", "parallel", "perplexity"]
    max_results: int
    search_prompt: str
    include_domains: List[str]
    exclude_domains: List[str]


class OpenRouterServerToolConfig(TypedDict, total=False):
    openrouter_server_tool: str
    parameters: Dict[str, Any]


BuiltInToolConfig = Union[
    OpenAIWebSearchToolConfig,
    FileSearchToolConfig,
    CodeInterpreterToolConfig,
    ImageGenerationToolConfig,
    OpenAINativeMcpToolConfig,
    ShellToolConfig,
    ApplyPatchToolConfig,
    OpenRouterWebSearchToolConfig,
    OpenRouterServerToolConfig,
    Dict[str, Any],
]


class BuiltInTool(TypedDict, total=False):
    id: str
    type: str
    name: str
    description: str
    provider: str
    config: BuiltInToolConfig


class McpTool(TypedDict, total=False):
    type: Literal["mcp"]
    mcp_server_id: int


class LegacyOpenAINativeMcpToolConfig(OpenAINativeMcpToolConfig, total=False):
    """Legacy OpenAI-native MCP config accepted and normalized by the backend."""

    execution_mode: Literal["provider"]


class LegacyOpenAINativeMcpTool(TypedDict, total=False):
    """Deprecated legacy shape. Use BuiltInTool with type ``openai_mcp``."""

    id: str
    name: str
    description: str
    provider: Literal["openai", "openai.azure"]
    type: Literal["mcp"]
    config: LegacyOpenAINativeMcpToolConfig


class ToolVariable(TypedDict, total=False):
    type: Literal["variable"]
    name: str


class RegistryTool(TypedDict, total=False):
    type: Literal["registry"]
    tool_registry_id: int
    label: Optional[str]
    version_number: Optional[int]


PromptTool = Union[Tool, McpTool, BuiltInTool, LegacyOpenAINativeMcpTool, ToolVariable, RegistryTool]


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
    provider_metadata: Union[Dict[str, Any], None]


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
    tools: Sequence[PromptTool]
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
    parent_version_id: int


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


class PublishPromptTemplateResponse(BasePromptTemplateResponse):
    pass


class GetPromptTemplateResponse(BasePromptTemplateResponse):
    llm_kwargs: Union[Dict[str, object], None]
    version: int


class ListPromptTemplateResponse(BasePromptTemplateResponse, total=False):
    version: int
