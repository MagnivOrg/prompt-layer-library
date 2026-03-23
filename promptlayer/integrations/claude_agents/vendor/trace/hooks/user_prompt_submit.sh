#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=plugins/trace/hooks/lib.sh
source "$SCRIPT_DIR/lib.sh"

tracing_enabled || exit 0
check_requirements || exit 0

input="$(cat)"
session_id="$(printf '%s' "$input" | python3 "$SCRIPT_DIR/py/cli.py" user-prompt-submit)"
[[ -z "$session_id" ]] && exit 0

log "INFO" "UserPromptSubmit captured session_id=$session_id"
