from dataclasses import dataclass
import os
import subprocess


PLUGIN_VERSION = "1.0.0"


def env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return default


def detect_claude_version() -> str:
    try:
        result = subprocess.run(
            ["claude", "--version"],
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "unknown"
    return result.stdout.strip() or "unknown"


@dataclass(frozen=True)
class HookContext:
    log_file: str
    queue_file: str
    session_state_dir: str
    lock_dir: str
    debug: str
    api_key: str
    otlp_endpoint: str
    queue_drain_limit: int
    otlp_connect_timeout: int
    otlp_max_time: int
    plugin_version: str
    cc_version: str
    user_agent: str


def load_context() -> HookContext:
    cc_version = detect_claude_version()
    plugin_version = PLUGIN_VERSION
    return HookContext(
        log_file=os.path.expanduser("~/.claude/state/promptlayer_hook.log"),
        queue_file=os.path.expanduser("~/.claude/state/promptlayer_otlp_queue.ndjson"),
        session_state_dir=os.path.expanduser("~/.claude/state/promptlayer_sessions"),
        lock_dir=os.path.expanduser("~/.claude/state/promptlayer_locks"),
        debug=os.environ.get("PROMPTLAYER_CC_DEBUG", "false"),
        api_key=os.environ.get("PROMPTLAYER_API_KEY", ""),
        otlp_endpoint=os.environ.get("PROMPTLAYER_OTLP_ENDPOINT", "https://api.promptlayer.com/v1/traces"),
        queue_drain_limit=env_int("PROMPTLAYER_QUEUE_DRAIN_LIMIT", 10),
        otlp_connect_timeout=env_int("PROMPTLAYER_OTLP_CONNECT_TIMEOUT", 5),
        otlp_max_time=env_int("PROMPTLAYER_OTLP_MAX_TIME", 12),
        plugin_version=plugin_version,
        cc_version=cc_version,
        user_agent=f"promptlayer-claude-plugin/{plugin_version} claude-code/{cc_version}",
    )
