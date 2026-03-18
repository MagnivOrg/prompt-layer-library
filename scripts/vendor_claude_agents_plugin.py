#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEST_VENDOR_ROOT = REPO_ROOT / "promptlayer" / "integrations" / "claude_agents" / "vendor"
DEST_PLUGIN_ROOT = DEST_VENDOR_ROOT / "trace"

SOURCE_PLUGIN_FILES = (
    "plugins/trace/.claude-plugin/plugin.json",
    "plugins/trace/hooks/hooks.json",
    "plugins/trace/hooks/lib.sh",
    "plugins/trace/hooks/session_start.sh",
    "plugins/trace/hooks/user_prompt_submit.sh",
    "plugins/trace/hooks/post_tool_use.sh",
    "plugins/trace/hooks/stop_hook.sh",
    "plugins/trace/hooks/session_end.sh",
    "plugins/trace/hooks/hook_utils.py",
    "plugins/trace/hooks/parse_stop_transcript.py",
)


def _git_output(source: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(source), *args],
        stderr=subprocess.DEVNULL,
        text=True,
    ).strip()


def _repository_identifier(source: Path) -> str:
    try:
        return _git_output(source, "remote", "get-url", "origin")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return str(source.resolve())


def _commit_sha(source: Path) -> str:
    try:
        return _git_output(source, "rev-parse", "HEAD")
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise SystemExit(f"Could not determine git commit SHA for {source}") from exc


def _validate_source(source: Path) -> None:
    missing = [relative for relative in SOURCE_PLUGIN_FILES if not (source / relative).exists()]
    if missing:
        raise SystemExit("Missing required plugin source files: " + ", ".join(missing))


def _copy_plugin_files(source: Path) -> None:
    if DEST_PLUGIN_ROOT.exists():
        shutil.rmtree(DEST_PLUGIN_ROOT)

    for relative in SOURCE_PLUGIN_FILES:
        source_path = source / relative
        destination_relative = Path(relative).relative_to("plugins/trace")
        destination_path = DEST_PLUGIN_ROOT / destination_relative
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination_path)


def _write_metadata(source: Path) -> None:
    DEST_VENDOR_ROOT.mkdir(parents=True, exist_ok=True)
    metadata = {
        "repository": _repository_identifier(source),
        "commit_sha": _commit_sha(source),
        "timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    }
    metadata_path = DEST_VENDOR_ROOT / "vendor_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def vendor(source: Path) -> None:
    _validate_source(source)
    _copy_plugin_files(source)
    _write_metadata(source)


def main() -> int:
    parser = argparse.ArgumentParser(description="Vendor the PromptLayer Claude Code plugin assets.")
    parser.add_argument("--source", required=True, help="Path to a local promptlayer-claude-plugins checkout.")
    args = parser.parse_args()

    source = Path(args.source).expanduser().resolve()
    if not source.exists():
        raise SystemExit(f"Source path does not exist: {source}")
    vendor(source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
