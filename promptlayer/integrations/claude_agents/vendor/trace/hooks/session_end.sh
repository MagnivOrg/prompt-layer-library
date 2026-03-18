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
stop_in_flight="$(get_session_state "$session_id" stop_in_flight)"
current_turn_start_ns="$(get_session_state "$session_id" current_turn_start_ns)"
[[ -z "$trace_id" || -z "$session_span_id" ]] && exit 0
[[ -z "$session_start_ns" ]] && session_start_ns="$(now_ns)"
[[ -z "$stop_in_flight" ]] && stop_in_flight="false"

if [[ -n "$current_turn_start_ns" || "$stop_in_flight" == "true" ]]; then
	set_session_state "$session_id" session_end_requested "true"
	log "INFO" "SessionEnd deferred until Stop session_id=$session_id"
	exit 0
fi

release_session_lock
trap - EXIT

# Always emit/re-emit root span with final end time. The server upserts on
# span_id conflict, so this safely updates the end time and lifecycle attribute.
end_ns="$(now_ns)"
attrs='{"source":"claude-code","hook":"SessionEnd","node_type":"WORKFLOW","session.lifecycle":"complete"}'
emit_span "$trace_id" "$session_span_id" "$session_parent_span_id" "Claude Code session" "1" "$session_start_ns" "$end_ns" "$attrs" || true

acquire_session_lock "$session_id" || exit 0
trap 'release_session_lock' EXIT

stop_in_flight="$(get_session_state "$session_id" stop_in_flight)"
current_turn_start_ns="$(get_session_state "$session_id" current_turn_start_ns)"
[[ -z "$stop_in_flight" ]] && stop_in_flight="false"
if [[ -n "$current_turn_start_ns" || "$stop_in_flight" == "true" ]]; then
	set_session_state "$session_id" session_end_requested "true"
	log "INFO" "SessionEnd deferred until Stop session_id=$session_id"
	exit 0
fi

set_session_state "$session_id" session_root_emitted "true"
rm -f "$PL_SESSION_STATE_DIR/$session_id.json"
log "INFO" "SessionEnd finalized session_id=$session_id"
