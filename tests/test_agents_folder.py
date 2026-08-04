"""Tests for the repository .agents feature documentation prompt."""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
AGENTS_DIR = REPO_ROOT / ".agents"
AGENTS_MD = AGENTS_DIR / "AGENTS.md"

# Each SDK surface area exposed by prompt-layer-library and its canonical docs URL.
SDK_FEATURE_DOCS: dict[str, str] = {
    "Python SDK": "https://docs.promptlayer.com/sdks/python.md",
    "Prompt templates": "https://docs.promptlayer.com/features/prompt-registry/overview.md",
    "Run prompts": "https://docs.promptlayer.com/sdks/python.md",
    "Run workflows": "https://docs.promptlayer.com/why-promptlayer/workflows.md",
    "Request logging": "https://docs.promptlayer.com/features/observability/request-logs/custom-logging.md",
    "Track metadata, prompts, and scores": "https://docs.promptlayer.com/features/observability/request-logs/metadata.md",
    "Request groups": "https://docs.promptlayer.com/features/observability/request-logs/tags.md",
    "Manual tracing": "https://docs.promptlayer.com/features/observability/traces/manual-tracing.md",
    "Tracing overview": "https://docs.promptlayer.com/features/observability/traces/overview.md",
    "Provider auto-instrumentation": "https://docs.promptlayer.com/features/observability/traces/auto-instrumentation/overview.md",
    "Skill collections": "https://docs.promptlayer.com/features/skill-collections/overview.md",
    "Tables": "https://docs.promptlayer.com/features/tables/overview.md",
    "Table scorecards": "https://docs.promptlayer.com/features/tables/scorecards.md",
    "Provider proxies": "https://docs.promptlayer.com/sdks/python.md",
    "SDK evals": "https://docs.promptlayer.com/sdks/evals/overview.md",
    "Eval CLI and CI": "https://docs.promptlayer.com/sdks/evals/cli-and-ci.md",
    "OpenAI Agents integration": "https://docs.promptlayer.com/sdks/evals/agent-tracing.md",
    "Claude Agents integration": "https://docs.promptlayer.com/agents/overview.md",
}

DOCS_INDEX_URL = "https://docs.promptlayer.com/llms.txt"


def test_agents_directory_exists():
    assert AGENTS_DIR.is_dir(), f"Expected {AGENTS_DIR} to exist"


def test_agents_md_exists_and_is_non_empty():
    assert AGENTS_MD.is_file(), f"Expected {AGENTS_MD} to exist"
    assert AGENTS_MD.read_text(encoding="utf-8").strip(), "AGENTS.md must not be empty"


def test_agents_md_points_to_docs_index():
    content = AGENTS_MD.read_text(encoding="utf-8")
    assert DOCS_INDEX_URL in content


def test_agents_md_documents_each_sdk_feature():
    content = AGENTS_MD.read_text(encoding="utf-8")
    missing = [feature for feature, url in SDK_FEATURE_DOCS.items() if url not in content]
    assert not missing, f"AGENTS.md is missing documentation links for: {', '.join(missing)}"


def test_agents_md_uses_promptlayer_docs_urls_only():
    content = AGENTS_MD.read_text(encoding="utf-8")
    urls = re.findall(r"https://[^\s)>\]]+", content)
    assert urls, "AGENTS.md should include documentation URLs"
    assert all(url.startswith("https://docs.promptlayer.com/") for url in urls)
