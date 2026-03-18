#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=plugins/trace/hooks/lib.sh
source "$SCRIPT_DIR/lib.sh"

tracing_enabled || exit 0
check_requirements || exit 0

input="$(cat)"
session_id="$(echo "$input" | jq -r '.session_id // empty')"
[[ -z "$session_id" ]] && session_id="$(uuidgen | tr '[:upper:]' '[:lower:]')"

existing_trace_id="$(get_session_state "$session_id" trace_id)"
existing_session_span_id="$(get_session_state "$session_id" session_span_id)"
if [[ -n "$existing_trace_id" && -n "$existing_session_span_id" ]]; then
	if [[ -z "$(get_session_state "$session_id" session_start_ns)" ]]; then
		set_session_state "$session_id" session_start_ns "$(now_ns)"
	fi
	if [[ -z "$(get_session_state "$session_id" pending_tool_calls)" ]]; then
		set_session_state "$session_id" pending_tool_calls "[]"
	fi
	if [[ -z "$(get_session_state "$session_id" session_parent_span_id)" ]]; then
		set_session_state "$session_id" session_parent_span_id ""
	fi
	if [[ -z "$(get_session_state "$session_id" session_traceparent_version)" ]]; then
		set_session_state "$session_id" session_traceparent_version ""
	fi
	if [[ -z "$(get_session_state "$session_id" session_trace_flags)" ]]; then
		set_session_state "$session_id" session_trace_flags ""
	fi
	if [[ -z "$(get_session_state "$session_id" trace_context_source)" ]]; then
		set_session_state "$session_id" trace_context_source "generated"
	fi
	if [[ -z "$(get_session_state "$session_id" session_end_requested)" ]]; then
		set_session_state "$session_id" session_end_requested "false"
	fi
	if [[ -z "$(get_session_state "$session_id" stop_in_flight)" ]]; then
		set_session_state "$session_id" stop_in_flight "false"
	fi
	log "INFO" "SessionStart ignored existing state session_id=$session_id trace_id=$existing_trace_id"
	exit 0
fi

load_initial_trace_context || true
trace_id="${PL_INITIAL_TRACE_ID:-}"
[[ -z "$trace_id" ]] && trace_id="$(generate_trace_id)"
span_id="$(generate_span_id)"
start_ns="$(now_ns)"

set_session_state "$session_id" trace_id "$trace_id"
set_session_state "$session_id" session_span_id "$span_id"
set_session_state "$session_id" session_parent_span_id "${PL_INITIAL_PARENT_SPAN_ID:-}"
set_session_state "$session_id" session_start_ns "$start_ns"
set_session_state "$session_id" current_turn_start_ns ""
set_session_state "$session_id" pending_tool_calls "[]"
set_session_state "$session_id" session_init_source "session_start_hook"
set_session_state "$session_id" session_traceparent_version "${PL_INITIAL_TRACEPARENT_VERSION:-}"
set_session_state "$session_id" session_trace_flags "${PL_INITIAL_TRACE_FLAGS:-}"
set_session_state "$session_id" trace_context_source "${PL_INITIAL_TRACE_CONTEXT_SOURCE:-generated}"
set_session_state "$session_id" session_root_emitted "false"
set_session_state "$session_id" session_end_requested "false"
set_session_state "$session_id" stop_in_flight "false"

log "INFO" "SessionStart captured session_id=$session_id trace_id=$trace_id"
