from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence

from promptlayer.cli.setup.agents import SetupAgentId, resolve_setup_agents
from promptlayer.cli.setup.constants import DOCS_MCP_URL, DOCS_URL, SDK_EVALS_SKILLS_ZIP_URL
from promptlayer.cli.setup.mcp import McpInstallResult, describe_mcp_install, install_docs_mcp_for_agents
from promptlayer.cli.setup.skills import (
    FetchBinary,
    FetchText,
    SkillInstallResult,
    default_fetch_binary,
    default_fetch_text,
    describe_skill_install,
    fetch_promptlayer_skill,
    fetch_sdk_evals_skill_entries,
    install_skill_entries_for_agents,
    install_skill_for_agents,
)

SetupTarget = str  # "all" | "skills" | "mcp"
WriteFn = Callable[[str], None]


@dataclass
class SetupResult:
    agents: List[SetupAgentId]
    skills: List[SkillInstallResult]
    mcp: List[McpInstallResult]


def run_setup_command(
    *,
    cwd: Optional[Path] = None,
    agents: Optional[Sequence[str]] = None,
    force: bool = False,
    target: SetupTarget = "all",
    fetch_text: Optional[FetchText] = None,
    fetch_binary: Optional[FetchBinary] = None,
    evals_url: str = SDK_EVALS_SKILLS_ZIP_URL,
    write: Optional[WriteFn] = None,
) -> SetupResult:
    root = Path.cwd() if cwd is None else Path(cwd)
    emit = write or (lambda message: print(message))
    resolved_agents = resolve_setup_agents(agents)
    text_fetcher = fetch_text or default_fetch_text
    binary_fetcher = fetch_binary or default_fetch_binary

    emit(f"Configuring PromptLayer for: {', '.join(agent.label for agent in resolved_agents)}")

    skill_results: List[SkillInstallResult] = []
    mcp_results: List[McpInstallResult] = []

    if target in {"all", "skills"}:
        emit(f"Fetching PromptLayer skill from {DOCS_URL}...")
        skill_content = fetch_promptlayer_skill(text_fetcher)
        promptlayer_results = install_skill_for_agents(
            cwd=root,
            agents=resolved_agents,
            skill_content=skill_content,
            force=force,
        )
        skill_results.extend(promptlayer_results)
        for result in promptlayer_results:
            emit(describe_skill_install(result))

        emit(f"Fetching SDK evals skills from {evals_url}...")
        eval_entries = fetch_sdk_evals_skill_entries(binary_fetcher, url=evals_url)
        eval_results = install_skill_entries_for_agents(
            cwd=root,
            agents=resolved_agents,
            files=eval_entries,
            force=force,
        )
        skill_results.extend(eval_results)
        for result in eval_results:
            emit(describe_skill_install(result))

    if target in {"all", "mcp"}:
        emit(f"Installing Docs MCP ({DOCS_MCP_URL})...")
        mcp_results = install_docs_mcp_for_agents(cwd=root, agents=resolved_agents, force=force)
        for result in mcp_results:
            emit(describe_mcp_install(result))

    emit("Done. Restart or reload your coding agent to pick up the new config.")
    return SetupResult(
        agents=[agent.id for agent in resolved_agents],
        skills=skill_results,
        mcp=mcp_results,
    )
