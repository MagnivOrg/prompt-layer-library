#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=plugins/trace/hooks/lib.sh
source "$SCRIPT_DIR/lib.sh"

tracing_enabled || exit 0
check_requirements || exit 0

input="$(cat)"
result="$(printf '%s' "$input" | python3 "$SCRIPT_DIR/py/cli.py" post-tool-use)"
IFS=$'\t' read -r session_id tool_name <<<"$result"
[[ -z "$session_id" || -z "$tool_name" ]] && exit 0

log "INFO" "PostToolUse captured session_id=$session_id tool=$tool_name"
