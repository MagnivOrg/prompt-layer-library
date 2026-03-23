#!/bin/bash
# Shared runtime helpers for trace plugin hooks.

set -euo pipefail
umask 077

export PL_LOG_FILE="$HOME/.claude/state/promptlayer_hook.log"
export PL_QUEUE_FILE="$HOME/.claude/state/promptlayer_otlp_queue.ndjson"
export PL_SESSION_STATE_DIR="$HOME/.claude/state/promptlayer_sessions"
export PL_LOCK_DIR="$HOME/.claude/state/promptlayer_locks"

export PL_DEBUG="${PROMPTLAYER_CC_DEBUG:-false}"
export PL_API_KEY="${PROMPTLAYER_API_KEY:-}"
export PL_OTLP_ENDPOINT="${PROMPTLAYER_OTLP_ENDPOINT:-https://api.promptlayer.com/v1/traces}"
export PL_QUEUE_DRAIN_LIMIT="${PROMPTLAYER_QUEUE_DRAIN_LIMIT:-10}"
export PL_OTLP_CONNECT_TIMEOUT="${PROMPTLAYER_OTLP_CONNECT_TIMEOUT:-5}"
export PL_OTLP_MAX_TIME="${PROMPTLAYER_OTLP_MAX_TIME:-12}"
export PL_PLUGIN_VERSION="1.0.0"
PL_CC_VERSION="$(claude --version 2>/dev/null || echo 'unknown')"
export PL_CC_VERSION
export PL_USER_AGENT="promptlayer-claude-plugin/${PL_PLUGIN_VERSION} claude-code/${PL_CC_VERSION}"
PL_HOOKS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PL_HOOKS_DIR

mkdir -p "$(dirname "$PL_LOG_FILE")"
mkdir -p "$PL_SESSION_STATE_DIR"
mkdir -p "$PL_LOCK_DIR"
chmod 700 "$(dirname "$PL_LOG_FILE")" "$PL_SESSION_STATE_DIR" 2>/dev/null || true
chmod 700 "$PL_LOCK_DIR" 2>/dev/null || true

log() {
	printf '%s [%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$1" "$2" >>"$PL_LOG_FILE"
}

debug() {
	local v
	v="$(echo "$PL_DEBUG" | tr '[:upper:]' '[:lower:]')"
	if [[ "$v" == "1" || "$v" == "true" || "$v" == "yes" || "$v" == "on" ]]; then
		log "DEBUG" "$1"
	fi
}

tracing_enabled() {
	local v
	v="$(echo "${TRACE_TO_PROMPTLAYER:-true}" | tr '[:upper:]' '[:lower:]')"
	[[ "$v" == "1" || "$v" == "true" || "$v" == "yes" || "$v" == "on" ]]
}

check_requirements() {
	if ! command -v python3 >/dev/null 2>&1; then
		log "ERROR" "Missing required command: python3"
		return 1
	fi
	if [[ -z "$PL_API_KEY" ]]; then
		log "ERROR" "PROMPTLAYER_API_KEY is not set"
		return 1
	fi
	return 0
}
