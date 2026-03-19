#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=plugins/trace/hooks/lib.sh
source "$SCRIPT_DIR/lib.sh"

tracing_enabled || exit 0
check_requirements || exit 0

input="$(cat)"
session_id="$(echo "$input" | jq -r '.session_id // empty')"
[[ -z "$session_id" ]] && exit 0

ensure_session_initialized "$session_id"

trace_id="$(get_session_state "$session_id" trace_id)"
session_span_id="$(get_session_state "$session_id" session_span_id)"
[[ -z "$trace_id" || -z "$session_span_id" ]] && exit 0
start_ns="$(now_ns)"

set_session_state "$session_id" current_turn_start_ns "$start_ns"
set_session_state "$session_id" pending_tool_calls "[]"

log "INFO" "UserPromptSubmit captured session_id=$session_id"
