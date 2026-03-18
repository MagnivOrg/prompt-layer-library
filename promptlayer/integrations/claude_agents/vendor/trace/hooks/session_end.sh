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

acquire_session_lock "$session_id" || exit 0
trap 'release_session_lock' EXIT

trace_id="$(get_session_state "$session_id" trace_id)"
session_span_id="$(get_session_state "$session_id" session_span_id)"
session_parent_span_id="$(get_session_state "$session_id" session_parent_span_id)"
session_start_ns="$(get_session_state "$session_id" session_start_ns)"
[[ -z "$trace_id" || -z "$session_span_id" ]] && exit 0
[[ -z "$session_start_ns" ]] && session_start_ns="$(now_ns)"

release_session_lock
trap - EXIT

# Always emit/re-emit root span with final end time. The server upserts on
# span_id conflict, so this safely updates the end time and lifecycle attribute.
end_ns="$(now_ns)"
attrs='{"source":"claude-code","hook":"SessionEnd","node_type":"WORKFLOW","session.lifecycle":"complete"}'
emit_span "$trace_id" "$session_span_id" "$session_parent_span_id" "Claude Code session" "1" "$session_start_ns" "$end_ns" "$attrs" || true

acquire_session_lock "$session_id" || exit 0
trap 'release_session_lock' EXIT
rm -f "$PL_SESSION_STATE_DIR/$session_id.json"
log "INFO" "SessionEnd finalized session_id=$session_id"
