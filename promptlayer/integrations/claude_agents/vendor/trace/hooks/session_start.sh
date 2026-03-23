#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=plugins/trace/hooks/lib.sh
source "$SCRIPT_DIR/lib.sh"

tracing_enabled || exit 0
check_requirements || exit 0

input="$(cat)"
result="$(printf '%s' "$input" | python3 "$SCRIPT_DIR/py/cli.py" session-start)"
IFS=$'\t' read -r session_id trace_id status <<<"$result"
[[ -z "$session_id" ]] && exit 0

if [[ "$status" == "existing" ]]; then
	log "INFO" "SessionStart ignored existing state session_id=$session_id trace_id=$trace_id"
	exit 0
fi

log "INFO" "SessionStart captured session_id=$session_id trace_id=$trace_id"
