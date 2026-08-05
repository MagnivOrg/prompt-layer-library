"""Tests for ``promptlayer setup``, matching the JS SDK setup CLI behavior."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pytest

from promptlayer.cli import main
from promptlayer.cli.setup.agents import resolve_setup_agents
from promptlayer.cli.setup.constants import (
    DOCS_MCP_SERVER_NAME,
    DOCS_MCP_URL,
    SDK_EVALS_SKILLS_ZIP_URL,
)
from promptlayer.cli.setup.mcp import install_docs_mcp_for_agents
from promptlayer.cli.setup.run import run_setup_command
from promptlayer.cli.setup.skills import (
    ensure_python_sdk_appendix,
    install_skill_entries_for_agents,
    install_skill_for_agents,
    read_zip_entries,
)


SKILL_FIXTURE = """---
name: Promptlayer
description: Test skill
---

# PromptLayer Skill

Base content.
"""

EVAL_SKILL_FIXTURE = """---
name: sdk-eval-builder
description: Test eval skill
---

# SDK Eval Builder
"""


def _evals_zip_bytes() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        archive.writestr("sdk-eval-builder/SKILL.md", EVAL_SKILL_FIXTURE)
        archive.writestr(
            "sdk-eval-builder/references/model-comparison-eval.md",
            "# Model comparison\n",
        )
    return buf.getvalue()


def test_resolve_setup_agents_defaults_to_cursor_and_claude():
    assert [agent.id for agent in resolve_setup_agents(None)] == ["cursor", "claude"]


def test_resolve_setup_agents_aliases_and_dedupes():
    assert [agent.id for agent in resolve_setup_agents(["claude-code", "cursor", "claude"])] == [
        "claude",
        "cursor",
    ]


def test_resolve_setup_agents_rejects_unknown():
    with pytest.raises(ValueError, match="Unknown agent"):
        resolve_setup_agents(["windsurf"])


def test_ensure_python_sdk_appendix_once():
    once = ensure_python_sdk_appendix(SKILL_FIXTURE)
    twice = ensure_python_sdk_appendix(once)
    assert "Python SDK guidance" in once
    assert twice == (once if once.endswith("\n") else f"{once}\n")


def test_read_zip_entries():
    entries = read_zip_entries(_evals_zip_bytes())
    assert sorted(entries) == [
        "sdk-eval-builder/SKILL.md",
        "sdk-eval-builder/references/model-comparison-eval.md",
    ]
    assert entries["sdk-eval-builder/SKILL.md"].decode("utf-8") == EVAL_SKILL_FIXTURE


def test_install_skill_for_agents_writes_and_skips(tmp_path: Path):
    agents = resolve_setup_agents(["cursor", "claude"])
    content = ensure_python_sdk_appendix(SKILL_FIXTURE)

    first = install_skill_for_agents(cwd=tmp_path, agents=agents, skill_content=content)
    assert all(result.status == "written" for result in first)

    second = install_skill_for_agents(cwd=tmp_path, agents=agents, skill_content=content)
    assert all(result.status == "skipped" for result in second)

    forced = install_skill_for_agents(
        cwd=tmp_path,
        agents=agents,
        skill_content=f"{content}\n# changed\n",
        force=True,
    )
    assert all(result.status == "updated" for result in forced)
    assert "# changed" in (tmp_path / ".agents/skills/promptlayer/SKILL.md").read_text(encoding="utf-8")


def test_install_zip_skill_trees(tmp_path: Path):
    agents = resolve_setup_agents(["cursor", "claude"])
    results = install_skill_entries_for_agents(
        cwd=tmp_path,
        agents=agents,
        files=read_zip_entries(_evals_zip_bytes()),
    )
    assert any(result.status == "written" for result in results)
    assert (tmp_path / ".agents/skills/sdk-eval-builder/SKILL.md").is_file()
    assert (
        tmp_path / ".agents/skills/sdk-eval-builder/references/model-comparison-eval.md"
    ).is_file()
    assert (tmp_path / ".claude/skills/sdk-eval-builder/SKILL.md").is_file()


def test_install_docs_mcp_for_agents(tmp_path: Path):
    agents = resolve_setup_agents(["cursor", "claude", "codex"])
    (tmp_path / ".cursor").mkdir()
    (tmp_path / ".cursor" / "mcp.json").write_text(
        json.dumps({"mcpServers": {"other": {"url": "https://example.com"}}}),
        encoding="utf-8",
    )

    first = install_docs_mcp_for_agents(cwd=tmp_path, agents=agents)
    assert {result.status for result in first} == {"written", "updated"}

    cursor = json.loads((tmp_path / ".cursor/mcp.json").read_text(encoding="utf-8"))
    assert cursor["mcpServers"]["other"]["url"] == "https://example.com"
    assert cursor["mcpServers"][DOCS_MCP_SERVER_NAME]["url"] == DOCS_MCP_URL

    claude = json.loads((tmp_path / ".mcp.json").read_text(encoding="utf-8"))
    assert claude["mcpServers"][DOCS_MCP_SERVER_NAME]["type"] == "http"

    codex = (tmp_path / ".codex/config.toml").read_text(encoding="utf-8")
    assert f"[mcp_servers.{DOCS_MCP_SERVER_NAME}]" in codex
    assert DOCS_MCP_URL in codex

    second = install_docs_mcp_for_agents(cwd=tmp_path, agents=agents)
    assert all(result.status == "skipped" for result in second)


def test_run_setup_command_installs_everything(tmp_path: Path):
    messages: list[str] = []
    zip_bytes = _evals_zip_bytes()

    result = run_setup_command(
        cwd=tmp_path,
        agents=["cursor", "claude"],
        fetch_text=lambda url: SKILL_FIXTURE,
        fetch_binary=lambda url: zip_bytes,
        evals_url=SDK_EVALS_SKILLS_ZIP_URL,
        write=messages.append,
    )

    assert result.agents == ["cursor", "claude"]
    assert (tmp_path / ".agents/skills/promptlayer/SKILL.md").is_file()
    assert (tmp_path / ".claude/skills/promptlayer/SKILL.md").is_file()
    assert (tmp_path / ".agents/skills/sdk-eval-builder/SKILL.md").is_file()
    assert (tmp_path / ".claude/skills/sdk-eval-builder/SKILL.md").is_file()
    assert (tmp_path / ".cursor/mcp.json").is_file()
    assert (tmp_path / ".mcp.json").is_file()

    skill_text = (tmp_path / ".agents/skills/promptlayer/SKILL.md").read_text(encoding="utf-8")
    assert "Python SDK guidance" in skill_text
    assert "Configuring PromptLayer for: Cursor, Claude Code" in messages
    assert any("Fetching PromptLayer skill from" in message for message in messages)
    assert any("Fetching SDK evals skills from" in message for message in messages)
    assert any("Installing Docs MCP" in message for message in messages)
    assert messages[-1] == "Done. Restart or reload your coding agent to pick up the new config."


def test_cli_setup_skills_only(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    zip_path = tmp_path / "evals.zip"
    zip_path.write_bytes(_evals_zip_bytes())

    # Patch network fetchers used by CLI path through run_setup_command defaults by
    # invoking run helpers via monkeypatch on module functions.
    from promptlayer.cli import setup_cmd

    def fake_run(**kwargs):
        return run_setup_command(
            cwd=tmp_path,
            agents=kwargs.get("agents"),
            force=kwargs.get("force", False),
            target=kwargs["target"],
            fetch_text=lambda url: SKILL_FIXTURE,
            fetch_binary=lambda url: _evals_zip_bytes(),
            write=lambda message: print(message),
        )

    monkeypatch.setattr(setup_cmd, "run_setup_command", fake_run)
    assert main(["setup", "skills", "--agent", "cursor"]) == 0
    out = capsys.readouterr().out
    assert "Installed Cursor skill (promptlayer)" in out
    assert "Docs MCP" not in out


def test_cli_setup_all_default_agents(tmp_path: Path, monkeypatch, capsys):
    from promptlayer.cli import setup_cmd

    def fake_run(**kwargs):
        return run_setup_command(
            cwd=tmp_path,
            agents=kwargs.get("agents"),
            force=kwargs.get("force", False),
            target=kwargs["target"],
            fetch_text=lambda url: SKILL_FIXTURE,
            fetch_binary=lambda url: _evals_zip_bytes(),
            write=lambda message: print(message),
        )

    monkeypatch.setattr(setup_cmd, "run_setup_command", fake_run)
    assert main(["setup"]) == 0
    out = capsys.readouterr().out
    assert "Configuring PromptLayer for: Cursor, Claude Code" in out
    assert (tmp_path / ".agents/skills/promptlayer/SKILL.md").is_file()
    assert (tmp_path / ".claude/skills/promptlayer/SKILL.md").is_file()
    assert (tmp_path / ".cursor/mcp.json").is_file()
    assert (tmp_path / ".mcp.json").is_file()
