import json
from pathlib import Path

import pytest

from promptlayer.integrations.claude_agents import (
    PromptLayerClaudeAgentsConfig,
    config as claude_agents_config,
    get_claude_config,
)


def test_get_claude_config_uses_explicit_api_key(monkeypatch):
    monkeypatch.delenv("PROMPTLAYER_API_KEY", raising=False)

    config = get_claude_config(api_key="pl_explicit")

    assert config.env["PROMPTLAYER_API_KEY"] == "pl_explicit"
    assert config.env["TRACE_TO_PROMPTLAYER"] == "true"


def test_get_claude_config_uses_environment_api_key(monkeypatch):
    monkeypatch.setenv("PROMPTLAYER_API_KEY", "pl_env")

    config = get_claude_config()

    assert config.env["PROMPTLAYER_API_KEY"] == "pl_env"


def test_get_claude_config_prefers_explicit_api_key(monkeypatch):
    monkeypatch.setenv("PROMPTLAYER_API_KEY", "pl_env")

    config = get_claude_config(api_key="pl_explicit")

    assert config.env["PROMPTLAYER_API_KEY"] == "pl_explicit"


def test_get_claude_config_raises_when_api_key_missing(monkeypatch):
    monkeypatch.delenv("PROMPTLAYER_API_KEY", raising=False)

    with pytest.raises(ValueError, match="PROMPTLAYER_API_KEY"):
        get_claude_config()


def test_get_claude_config_raises_on_windows(monkeypatch):
    monkeypatch.setenv("PROMPTLAYER_API_KEY", "pl_env")
    monkeypatch.setattr(claude_agents_config.sys, "platform", "win32")

    with pytest.raises(RuntimeError, match="not supported on Windows"):
        get_claude_config()


def test_get_claude_config_omits_traceparent_by_default(monkeypatch):
    monkeypatch.setenv("PROMPTLAYER_API_KEY", "pl_env")

    config = get_claude_config()

    assert "PROMPTLAYER_TRACEPARENT" not in config.env
    assert "PROMPTLAYER_CC_DEBUG" not in config.env
    assert "PROMPTLAYER_OTLP_ENDPOINT" not in config.env


def test_get_claude_config_includes_traceparent_when_provided(monkeypatch):
    monkeypatch.setenv("PROMPTLAYER_API_KEY", "pl_env")
    traceparent = "00-11111111111111111111111111111111-2222222222222222-01"

    config = get_claude_config(traceparent=traceparent)

    assert config.env["PROMPTLAYER_TRACEPARENT"] == traceparent


def test_get_claude_config_returns_typed_config(monkeypatch):
    monkeypatch.setenv("PROMPTLAYER_API_KEY", "pl_env")

    config = get_claude_config()

    assert isinstance(config, PromptLayerClaudeAgentsConfig)
    assert config.plugin["type"] == "local"
    assert isinstance(config.plugin["path"], str)


def test_get_claude_config_points_at_complete_vendored_plugin(monkeypatch):
    monkeypatch.setenv("PROMPTLAYER_API_KEY", "pl_env")

    config = get_claude_config()
    plugin_root = Path(config.plugin["path"])

    assert plugin_root.is_dir()
    for relative in (
        ".claude-plugin/plugin.json",
        "hooks/hooks.json",
        "hooks/lib.sh",
        "hooks/session_start.sh",
        "hooks/user_prompt_submit.sh",
        "hooks/post_tool_use.sh",
        "hooks/stop_hook.sh",
        "hooks/session_end.sh",
        "hooks/hook_utils.py",
        "hooks/parse_stop_transcript.py",
    ):
        assert (plugin_root / relative).exists(), relative


def test_vendor_metadata_contains_required_fields():
    metadata_path = Path(claude_agents_config.__file__).resolve().parent / "vendor" / "vendor_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert metadata["repository"]
    assert metadata["commit_sha"]
    assert metadata["timestamp"]


def test_vendor_metadata_is_valid_json_file():
    metadata_path = Path(claude_agents_config.__file__).resolve().parent / "vendor" / "vendor_metadata.json"
    parsed = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert parsed["repository"]
    assert parsed["commit_sha"]
    assert parsed["timestamp"]
