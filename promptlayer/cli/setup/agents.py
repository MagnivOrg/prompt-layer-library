from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

SetupAgentId = str  # "cursor" | "claude" | "codex"


@dataclass(frozen=True)
class SetupAgent:
    id: SetupAgentId
    aliases: Sequence[str]
    label: str
    skills_dir: str
    mcp_path: Optional[str]
    mcp_format: str  # "cursor" | "claude" | "codex" | "none"


SETUP_AGENTS: List[SetupAgent] = [
    SetupAgent(
        id="cursor",
        aliases=("cursor", "cursor-cli"),
        label="Cursor",
        skills_dir=".agents/skills",
        mcp_path=".cursor/mcp.json",
        mcp_format="cursor",
    ),
    SetupAgent(
        id="claude",
        aliases=("claude", "claude-code", "claudecode"),
        label="Claude Code",
        skills_dir=".claude/skills",
        mcp_path=".mcp.json",
        mcp_format="claude",
    ),
    SetupAgent(
        id="codex",
        aliases=("codex",),
        label="Codex",
        skills_dir=".agents/skills",
        mcp_path=".codex/config.toml",
        mcp_format="codex",
    ),
]

DEFAULT_SETUP_AGENTS: Sequence[SetupAgentId] = ("cursor", "claude")


def skill_package_path(agent: SetupAgent, skill_name: str) -> str:
    return f"{agent.skills_dir}/{skill_name}"


def resolve_setup_agents(requested: Optional[Sequence[str]]) -> List[SetupAgent]:
    if not requested:
        return [agent for agent in SETUP_AGENTS if agent.id in DEFAULT_SETUP_AGENTS]

    selected: Dict[SetupAgentId, SetupAgent] = {}
    for raw in requested:
        normalized = raw.strip().lower()
        if normalized in {"*", "all"}:
            for agent in SETUP_AGENTS:
                selected[agent.id] = agent
            continue
        match = next((agent for agent in SETUP_AGENTS if normalized in agent.aliases), None)
        if match is None:
            supported = ", ".join(agent.id for agent in SETUP_AGENTS)
            raise ValueError(f'Unknown agent "{raw}". Supported agents: {supported}, or "*".')
        selected[match.id] = match
    return list(selected.values())
