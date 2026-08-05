from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

from promptlayer.cli.setup.agents import SetupAgent
from promptlayer.cli.setup.constants import DOCS_MCP_SERVER_NAME, DOCS_MCP_URL


@dataclass
class McpInstallResult:
    agent: SetupAgent
    path: str
    status: str  # written | updated | skipped | unsupported


_CURSOR_DOCS_SERVER = {"url": DOCS_MCP_URL}
_CLAUDE_DOCS_SERVER = {"type": "http", "url": DOCS_MCP_URL}


def install_docs_mcp_for_agents(
    *,
    cwd: Path,
    agents: Sequence[SetupAgent],
    force: bool = False,
) -> List[McpInstallResult]:
    results: List[McpInstallResult] = []
    written: Dict[str, str] = {}

    for agent in agents:
        if not agent.mcp_path or agent.mcp_format == "none":
            results.append(McpInstallResult(agent=agent, path=agent.mcp_path or "", status="unsupported"))
            continue

        absolute_path = (cwd / agent.mcp_path).resolve()
        absolute_key = str(absolute_path)
        if absolute_key in written:
            results.append(
                McpInstallResult(agent=agent, path=agent.mcp_path, status=written[absolute_key])
            )
            continue

        existing_raw = absolute_path.read_text(encoding="utf-8") if absolute_path.exists() else None
        if agent.mcp_format == "cursor":
            merged = _merge_json_mcp_config(
                existing_raw=existing_raw,
                file_path=agent.mcp_path,
                server_entry=_CURSOR_DOCS_SERVER,
                force=force,
            )
        elif agent.mcp_format == "claude":
            merged = _merge_json_mcp_config(
                existing_raw=existing_raw,
                file_path=agent.mcp_path,
                server_entry=_CLAUDE_DOCS_SERVER,
                force=force,
            )
        else:
            merged = _merge_codex_mcp_config(existing_raw=existing_raw, force=force)

        if merged["status"] != "skipped":
            absolute_path.parent.mkdir(parents=True, exist_ok=True)
            absolute_path.write_text(merged["next"], encoding="utf-8")

        written[absolute_key] = merged["status"]
        results.append(McpInstallResult(agent=agent, path=agent.mcp_path, status=merged["status"]))

    return results


def describe_mcp_install(result: McpInstallResult) -> str:
    label = f"{result.agent.label} Docs MCP ({DOCS_MCP_SERVER_NAME})"
    if result.status == "unsupported":
        return f"Skipped {label}: no known MCP config path for this agent"
    if result.status == "skipped":
        return f"Skipped {label}: {result.path} already configured (use --force to overwrite)"
    if result.status == "updated":
        return f"Updated {label}: {result.path}"
    return f"Installed {label}: {result.path}"


def _merge_json_mcp_config(
    *,
    existing_raw: Optional[str],
    file_path: str,
    server_entry: Mapping[str, object],
    force: bool,
) -> Dict[str, str]:
    existing = _parse_json_object(existing_raw, file_path) if existing_raw else {}
    servers_value = existing.get("mcpServers")
    servers = (
        dict(servers_value)
        if isinstance(servers_value, dict)
        else {}
    )
    current = servers.get(DOCS_MCP_SERVER_NAME)
    already_matches = (
        isinstance(current, dict) and json.dumps(current, sort_keys=True) == json.dumps(dict(server_entry), sort_keys=True)
    )
    if already_matches and not force:
        return {"next": json.dumps(existing, indent=2) + "\n", "status": "skipped"}
    if current is not None and not force and not already_matches:
        return {"next": json.dumps(existing, indent=2) + "\n", "status": "skipped"}

    servers[DOCS_MCP_SERVER_NAME] = dict(server_entry)
    next_object = {**existing, "mcpServers": servers}
    return {
        "next": json.dumps(next_object, indent=2) + "\n",
        "status": "updated" if existing_raw else "written",
    }


def _merge_codex_mcp_config(*, existing_raw: Optional[str], force: bool) -> Dict[str, str]:
    section_header = f"[mcp_servers.{DOCS_MCP_SERVER_NAME}]"
    section_body = f'{section_header}\nurl = "{DOCS_MCP_URL}"\n'
    existing = existing_raw or ""

    if section_header in existing:
        if not force:
            return {"next": existing if existing.endswith("\n") else f"{existing}\n", "status": "skipped"}
        next_text = re.sub(
            rf"\[mcp_servers\.{re.escape(DOCS_MCP_SERVER_NAME)}\][\s\S]*?(?=\n\[|$)",
            section_body.rstrip(),
            existing,
            count=1,
        )
        return {"next": next_text if next_text.endswith("\n") else f"{next_text}\n", "status": "updated"}

    if not existing.strip():
        return {"next": f"{section_body}\n", "status": "written"}

    prefix = existing if existing.endswith("\n") else f"{existing}\n"
    return {"next": f"{prefix}\n{section_body}\n", "status": "updated"}


def _parse_json_object(raw: str, file_path: str) -> Dict[str, object]:
    if not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Could not parse JSON in {file_path}. Fix or remove it, then retry.") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError(f"Expected a JSON object in {file_path}.")
    return parsed
