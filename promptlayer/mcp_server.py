"""
PromptLayer MCP Server

Exposes PromptLayer functionality as an MCP (Model Context Protocol) server,
allowing AI agents to interact with prompt templates, workflows, request logging,
and tracking features.

MCP Tools:
    - get_prompt_template: Retrieve a prompt template by name
    - list_prompt_templates: List all available prompt templates
    - publish_prompt_template: Create or update a prompt template
    - run_prompt: Execute a prompt template through its configured LLM
    - run_workflow: Execute a PromptLayer workflow
    - log_request: Log an LLM request to PromptLayer
    - track_prompt: Associate a prompt template with a logged request
    - track_metadata: Attach metadata to a logged request
    - track_score: Score a logged request (0-100)
    - track_group: Associate a request with a group
    - create_group: Create a new request group

MCP Resources:
    - promptlayer://templates: List of all prompt templates
    - promptlayer://templates/{name}: Individual prompt template details
"""

import json
import logging
import os
from typing import Any

from mcp.server.fastmcp import FastMCP

from promptlayer import AsyncPromptLayer
from promptlayer.utils import apublish_prompt_template

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Server instance
# ---------------------------------------------------------------------------
mcp = FastMCP(
    "PromptLayer",
    instructions=(
        "MCP server for the PromptLayer platform. "
        "Manage prompt templates, run prompts and workflows, "
        "log LLM requests, and track metadata/scores."
    ),
)

# ---------------------------------------------------------------------------
# Lazy client helper
# ---------------------------------------------------------------------------
_client: AsyncPromptLayer | None = None


def _get_client() -> AsyncPromptLayer:
    """Return a shared AsyncPromptLayer client, creating it on first use.

    Requires the ``PROMPTLAYER_API_KEY`` environment variable to be set.
    An optional ``PROMPTLAYER_BASE_URL`` can override the default API endpoint.
    """
    global _client
    if _client is None:
        api_key = os.environ.get("PROMPTLAYER_API_KEY")
        if not api_key:
            raise ValueError(
                "PROMPTLAYER_API_KEY environment variable is required. "
                "Set it before starting the MCP server."
            )
        base_url = os.environ.get("PROMPTLAYER_BASE_URL")
        _client = AsyncPromptLayer(api_key=api_key, base_url=base_url)
    return _client


def _json_dump(obj: Any) -> str:
    """Compact JSON serialisation helper."""
    return json.dumps(obj, indent=2, default=str)


# ============================================================================
# MCP Tools – Prompt Templates
# ============================================================================


@mcp.tool()
async def get_prompt_template(
    prompt_name: str,
    version: int | None = None,
    label: str | None = None,
    input_variables: dict[str, Any] | None = None,
    metadata_filters: dict[str, str] | None = None,
    provider: str | None = None,
    model: str | None = None,
) -> str:
    """Retrieve a prompt template from PromptLayer by name.

    Args:
        prompt_name: The name of the prompt template to retrieve.
        version: Specific version number to fetch. Omit for the latest version.
        label: Release label to fetch (e.g. "production"). Overrides version.
        input_variables: Dict of variables to fill into the template.
        metadata_filters: Filter by metadata key-value pairs.
        provider: Filter by LLM provider (e.g. "openai", "anthropic").
        model: Filter by model name.

    Returns:
        JSON string containing the prompt template, its metadata, and LLM kwargs.
    """
    client = _get_client()
    params: dict[str, Any] = {}
    if version is not None:
        params["version"] = version
    if label is not None:
        params["label"] = label
    if input_variables is not None:
        params["input_variables"] = input_variables
    if metadata_filters is not None:
        params["metadata_filters"] = metadata_filters
    if provider is not None:
        params["provider"] = provider
    if model is not None:
        params["model"] = model

    result = await client.templates.get(prompt_name, params or None)
    if result is None:
        return json.dumps({"error": f"Prompt template '{prompt_name}' not found."})
    return _json_dump(result)


@mcp.tool()
async def list_prompt_templates(
    page: int = 1,
    per_page: int = 30,
    label: str | None = None,
) -> str:
    """List all prompt templates in the PromptLayer workspace.

    Args:
        page: Page number for pagination (default 1).
        per_page: Number of templates per page (default 30, max varies by plan).
        label: Optional release label to filter by.

    Returns:
        JSON array of prompt template summaries.
    """
    client = _get_client()
    result = await client.templates.all(page=page, per_page=per_page, label=label)
    return _json_dump(result)


@mcp.tool()
async def publish_prompt_template(
    prompt_name: str,
    prompt_template: dict[str, Any],
    commit_message: str = "",
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    release_labels: list[str] | None = None,
) -> str:
    """Publish (create or update) a prompt template in the PromptLayer registry.

    Args:
        prompt_name: Name for the prompt template.
        prompt_template: The template body. For chat templates use
            ``{"type": "chat", "messages": [...]}``. For completion templates use
            ``{"type": "completion", "content": [...]}``.
        commit_message: Human-readable description of this version.
        tags: Optional list of tags for organisation.
        metadata: Optional metadata dict (e.g. model configuration).
        release_labels: Optional list of labels to assign (e.g. ["production"]).

    Returns:
        JSON with the published template details including its new version number.
    """
    client = _get_client()
    body: dict[str, Any] = {
        "prompt_name": prompt_name,
        "prompt_template": prompt_template,
        "commit_message": commit_message,
    }
    if tags is not None:
        body["tags"] = tags
    if metadata is not None:
        body["metadata"] = metadata
    if release_labels is not None:
        body["release_labels"] = release_labels

    result = await apublish_prompt_template(
        client.api_key, client.base_url, client.throw_on_error, body
    )
    if result is None:
        return json.dumps({"error": "Failed to publish prompt template."})
    return _json_dump(result)


# ============================================================================
# MCP Tools – Run Prompts & Workflows
# ============================================================================


@mcp.tool()
async def run_prompt(
    prompt_name: str,
    input_variables: dict[str, Any] | None = None,
    prompt_version: int | None = None,
    prompt_release_label: str | None = None,
    model_parameter_overrides: dict[str, Any] | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, str] | None = None,
    group_id: int | None = None,
    provider: str | None = None,
    model: str | None = None,
) -> str:
    """Run a prompt template through its configured LLM and return the response.

    This fetches the named prompt template from the PromptLayer registry,
    fills in input variables, sends it to the configured LLM provider,
    logs the request, and returns the result.

    Args:
        prompt_name: Name of the prompt template to run.
        input_variables: Dict of variables to fill into the template placeholders.
        prompt_version: Specific version to run. Omit for latest.
        prompt_release_label: Release label to use (e.g. "production").
        model_parameter_overrides: Override model parameters (temperature, max_tokens, etc.).
        tags: Tags to associate with this request for filtering in the dashboard.
        metadata: Key-value metadata to attach to the logged request.
        group_id: Group ID to associate this request with.
        provider: Override the LLM provider.
        model: Override the model name.

    Returns:
        JSON containing request_id, raw_response from the LLM, and prompt_blueprint used.
    """
    client = _get_client()
    result = await client.run(
        prompt_name=prompt_name,
        input_variables=input_variables or {},
        prompt_version=prompt_version,
        prompt_release_label=prompt_release_label,
        model_parameter_overrides=model_parameter_overrides,
        tags=tags,
        metadata=metadata,
        group_id=group_id,
        stream=False,
        provider=provider,
        model=model,
    )
    # The raw_response may be a Pydantic model; convert for JSON serialisation
    serialisable = {
        "request_id": result.get("request_id"),
        "prompt_blueprint": result.get("prompt_blueprint"),
    }
    raw = result.get("raw_response")
    if hasattr(raw, "model_dump"):
        serialisable["raw_response"] = raw.model_dump(mode="json")
    elif isinstance(raw, dict):
        serialisable["raw_response"] = raw
    else:
        serialisable["raw_response"] = str(raw)

    return _json_dump(serialisable)


@mcp.tool()
async def run_workflow(
    workflow_id_or_name: int | str,
    input_variables: dict[str, Any] | None = None,
    metadata: dict[str, str] | None = None,
    workflow_label_name: str | None = None,
    workflow_version: int | None = None,
    return_all_outputs: bool = False,
    timeout: float | None = None,
) -> str:
    """Execute a PromptLayer workflow and return its output.

    Workflows are multi-step prompt pipelines configured in the PromptLayer dashboard.

    Args:
        workflow_id_or_name: The workflow ID (int) or name (str) to execute.
        input_variables: Input variables to pass to the workflow.
        metadata: Key-value metadata to attach to the workflow execution.
        workflow_label_name: Specific workflow label to run (e.g. "production").
        workflow_version: Specific workflow version number.
        return_all_outputs: If True, return outputs from all nodes, not just the output node.
        timeout: Maximum seconds to wait for workflow completion.

    Returns:
        JSON with the workflow execution results.
    """
    client = _get_client()
    result = await client.run_workflow(
        workflow_id_or_name=workflow_id_or_name,
        input_variables=input_variables or {},
        metadata=metadata,
        workflow_label_name=workflow_label_name,
        workflow_version=workflow_version,
        return_all_outputs=return_all_outputs,
        timeout=timeout,
    )
    return _json_dump(result)


# ============================================================================
# MCP Tools – Request Logging
# ============================================================================


@mcp.tool()
async def log_request(
    provider: str,
    model: str,
    input: dict[str, Any],
    output: dict[str, Any],
    request_start_time: float,
    request_end_time: float,
    parameters: dict[str, Any] | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, str] | None = None,
    prompt_name: str | None = None,
    prompt_version_number: int | None = None,
    prompt_input_variables: dict[str, Any] | None = None,
    input_tokens: int = 0,
    output_tokens: int = 0,
    price: float = 0.0,
) -> str:
    """Log an LLM request to PromptLayer for tracking and analytics.

    Use this to log requests made outside of PromptLayer's run() method.

    Args:
        provider: LLM provider name (e.g. "openai", "anthropic").
        model: Model name (e.g. "gpt-4", "claude-3-opus").
        input: The input prompt template (chat messages or completion content).
        output: The output/response from the LLM.
        request_start_time: Unix timestamp when the request started.
        request_end_time: Unix timestamp when the request completed.
        parameters: Model parameters used (temperature, max_tokens, etc.).
        tags: Tags to associate with the request.
        metadata: Key-value metadata for filtering/searching.
        prompt_name: Name of a prompt template to associate.
        prompt_version_number: Version of the prompt template.
        prompt_input_variables: Variables used to fill the prompt.
        input_tokens: Number of input tokens consumed.
        output_tokens: Number of output tokens generated.
        price: Estimated cost of the request.

    Returns:
        JSON with the created request log including its request_id.
    """
    client = _get_client()
    result = await client.log_request(
        provider=provider,
        model=model,
        input=input,
        output=output,
        request_start_time=request_start_time,
        request_end_time=request_end_time,
        parameters=parameters or {},
        tags=tags or [],
        metadata=metadata or {},
        prompt_name=prompt_name,
        prompt_version_number=prompt_version_number,
        prompt_input_variables=prompt_input_variables or {},
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        price=price,
    )
    if result is None:
        return json.dumps({"error": "Failed to log request."})
    return _json_dump(result)


# ============================================================================
# MCP Tools – Tracking
# ============================================================================


@mcp.tool()
async def track_prompt(
    request_id: int,
    prompt_name: str,
    prompt_input_variables: dict[str, Any],
    version: int | None = None,
    label: str | None = None,
) -> str:
    """Associate a prompt template with a previously logged request.

    This links a specific prompt template (and its input variables) to
    a request that was already logged, enabling prompt-level analytics.

    Args:
        request_id: The ID of the logged request to associate.
        prompt_name: Name of the prompt template.
        prompt_input_variables: Dict of variables that were used in the template.
        version: Specific prompt version (omit for latest).
        label: Release label of the prompt version.

    Returns:
        JSON indicating success or failure.
    """
    client = _get_client()
    result = await client.track.prompt(
        request_id=request_id,
        prompt_name=prompt_name,
        prompt_input_variables=prompt_input_variables,
        version=version,
        label=label,
    )
    return json.dumps({"success": bool(result)})


@mcp.tool()
async def track_metadata(
    request_id: int,
    metadata: dict[str, str],
) -> str:
    """Attach metadata key-value pairs to a previously logged request.

    Metadata can be used for filtering and searching requests in the
    PromptLayer dashboard.

    Args:
        request_id: The ID of the logged request.
        metadata: Dict of string key-value pairs to attach.

    Returns:
        JSON indicating success or failure.
    """
    client = _get_client()
    result = await client.track.metadata(
        request_id=request_id,
        metadata=metadata,
    )
    return json.dumps({"success": bool(result)})


@mcp.tool()
async def track_score(
    request_id: int,
    score: int,
    score_name: str | None = None,
) -> str:
    """Score a previously logged request.

    Scores enable quality tracking and evaluation of LLM responses.

    Args:
        request_id: The ID of the logged request to score.
        score: Integer score between 0 and 100.
        score_name: Optional name for the score dimension (e.g. "relevance", "accuracy").

    Returns:
        JSON indicating success or failure.
    """
    client = _get_client()
    result = await client.track.score(
        request_id=request_id,
        score=score,
        score_name=score_name,
    )
    return json.dumps({"success": bool(result)})


@mcp.tool()
async def track_group(
    request_id: int,
    group_id: int,
) -> str:
    """Associate a previously logged request with a group.

    Groups allow you to bundle related requests together for analysis.

    Args:
        request_id: The ID of the logged request.
        group_id: The group ID to associate (created via create_group).

    Returns:
        JSON indicating success or failure.
    """
    client = _get_client()
    result = await client.track.group(
        request_id=request_id,
        group_id=group_id,
    )
    return json.dumps({"success": bool(result)})


# ============================================================================
# MCP Tools – Groups
# ============================================================================


@mcp.tool()
async def create_group() -> str:
    """Create a new request group in PromptLayer.

    Groups allow you to bundle related LLM requests together. After creating
    a group, use track_group to associate individual requests with it.

    Returns:
        JSON with the new group's ID, or an error.
    """
    client = _get_client()
    result = await client.group.create()
    if result is False:
        return json.dumps({"error": "Failed to create group."})
    return json.dumps({"group_id": result})


# ============================================================================
# MCP Resources
# ============================================================================


@mcp.resource("promptlayer://templates")
async def resource_list_templates() -> str:
    """Browse all prompt templates in the PromptLayer workspace."""
    client = _get_client()
    result = await client.templates.all(page=1, per_page=100)
    return _json_dump(result)


@mcp.resource("promptlayer://templates/{name}")
async def resource_get_template(name: str) -> str:
    """Retrieve a specific prompt template by name."""
    client = _get_client()
    result = await client.templates.get(name)
    if result is None:
        return json.dumps({"error": f"Prompt template '{name}' not found."})
    return _json_dump(result)


# ============================================================================
# Entry point
# ============================================================================


def main():
    """Run the PromptLayer MCP server (stdio transport)."""
    mcp.run()


if __name__ == "__main__":
    main()
