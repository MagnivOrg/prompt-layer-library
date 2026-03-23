from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Literal, TypedDict


class PromptLayerClaudeAgentsPlugin(TypedDict):
    type: Literal["local"]
    path: str


class PromptLayerClaudeAgentsEnv(TypedDict, total=False):
    TRACE_TO_PROMPTLAYER: Literal["true"]
    PROMPTLAYER_API_KEY: str
    PROMPTLAYER_TRACEPARENT: str


@dataclass(frozen=True)
class PromptLayerClaudeAgentsConfig:
    plugin: PromptLayerClaudeAgentsPlugin
    env: PromptLayerClaudeAgentsEnv


_VENDOR_ROOT_FALLBACK = Path(__file__).resolve().parent / "vendor" / "trace"
_REQUIRED_VENDOR_FILES = (
    ".claude-plugin/plugin.json",
    "setup.sh",
    "hooks/hooks.json",
    "hooks/lib.sh",
    "hooks/session_start.sh",
    "hooks/user_prompt_submit.sh",
    "hooks/post_tool_use.sh",
    "hooks/stop_hook.sh",
    "hooks/session_end.sh",
    "hooks/py/__init__.py",
    "hooks/py/cli.py",
    "hooks/py/context.py",
    "hooks/py/handlers.py",
    "hooks/py/otlp.py",
    "hooks/py/settings.py",
    "hooks/py/state.py",
    "hooks/py/stop_parser.py",
    "hooks/py/traceparent.py",
)


def get_claude_config(*, api_key: str | None = None, traceparent: str | None = None) -> PromptLayerClaudeAgentsConfig:
    if sys.platform == "win32":
        raise RuntimeError("PromptLayer Claude Agents integration is not supported on Windows. Use Linux or macOS.")

    resolved_api_key = api_key or os.environ.get("PROMPTLAYER_API_KEY")
    if not resolved_api_key:
        raise ValueError(
            "PromptLayer API key not provided. "
            "Please set the PROMPTLAYER_API_KEY environment variable or pass the api_key parameter."
        )

    plugin_root = _resolve_vendor_root()
    _assert_required_vendor_files(plugin_root)
    plugin = PromptLayerClaudeAgentsPlugin(type="local", path=str(plugin_root))

    env = PromptLayerClaudeAgentsEnv(
        TRACE_TO_PROMPTLAYER="true",
        PROMPTLAYER_API_KEY=resolved_api_key,
    )
    if traceparent is not None:
        env["PROMPTLAYER_TRACEPARENT"] = traceparent

    return PromptLayerClaudeAgentsConfig(plugin=plugin, env=env)


def _resolve_vendor_root() -> Path:
    traversable = files("promptlayer.integrations.claude_agents").joinpath("vendor").joinpath("trace")
    if isinstance(traversable, Path):
        return traversable
    if _VENDOR_ROOT_FALLBACK.exists():
        return _VENDOR_ROOT_FALLBACK
    raise RuntimeError(
        "PromptLayer Claude Code vendored plugin path is not available from package resources. "
        "Reinstall the package from a standard wheel or source distribution."
    )


def _assert_required_vendor_files(plugin_root: Path) -> None:
    missing = [relative for relative in _REQUIRED_VENDOR_FILES if not (plugin_root / relative).exists()]
    if missing:
        raise RuntimeError(
            "PromptLayer Claude Code vendored plugin is incomplete. Missing files: " + ", ".join(sorted(missing))
        )
