#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=plugins/trace/hooks/lib.sh
source "$SCRIPT_DIR/lib.sh"

tracing_enabled || exit 0
check_requirements || exit 0

input="$(cat)"
session_id="$(echo "$input" | jq -r '.session_id // empty')"
transcript_path="$(echo "$input" | jq -r '.transcript_path // empty')"

if [[ -z "$session_id" && -n "$transcript_path" ]]; then
	session_id="$(basename "$transcript_path" .jsonl)"
fi
[[ -z "$session_id" ]] && exit 0
spans_file="$(mktemp "${TMPDIR:-/tmp}/pl-stop-spans.XXXXXX")"
cleanup() {
	rm -f "$spans_file"
	release_session_lock
}
trap cleanup EXIT

add_span_to_batch() {
	local trace="$1"
	local span="$2"
	local parent="$3"
	local span_name="$4"
	local span_kind="$5"
	local start="$6"
	local end="$7"
	local attrs="$8"

	local span_json
	span_json="$(build_span_json "$trace" "$span" "$parent" "$span_name" "$span_kind" "$start" "$end" "$attrs")" || return 1
	printf '%s\n' "$span_json" >>"$spans_file"
}

acquire_session_lock "$session_id" || exit 0
ensure_session_initialized "$session_id"

trace_id="$(get_session_state "$session_id" trace_id)"
session_span_id="$(get_session_state "$session_id" session_span_id)"
session_parent_span_id="$(get_session_state "$session_id" session_parent_span_id)"
turn_start_ns="$(get_session_state "$session_id" current_turn_start_ns)"
pending_tool_calls="$(get_session_state "$session_id" pending_tool_calls)"
session_init_source="$(get_session_state "$session_id" session_init_source)"
session_start_ns="$(get_session_state "$session_id" session_start_ns)"

[[ -z "$trace_id" || -z "$session_span_id" ]] && exit 0
[[ -z "$pending_tool_calls" ]] && pending_tool_calls='[]'
[[ -z "$session_start_ns" ]] && session_start_ns="$(now_ns)"

[[ -z "$turn_start_ns" ]] && turn_start_ns="$(now_ns)"

# Keep lock scope short: snapshot + clear turn-specific mutable state.
set_session_state "$session_id" current_turn_start_ns ""
set_session_state "$session_id" pending_tool_calls "[]"

release_session_lock

parse_transcript_with_retry() {
	local attempts=0
	local parsed llm_count
	while true; do
		parsed="$(PL_PENDING_TOOL_CALLS="$pending_tool_calls" python3 "$SCRIPT_DIR/parse_stop_transcript.py" "$transcript_path" "$turn_start_ns" "$session_id")"
		llm_count="$(echo "$parsed" | jq -r '.llms | length')"
		if [[ "$llm_count" -gt 0 || $attempts -ge 10 ]]; then
			echo "$parsed"
			return 0
		fi
		attempts=$((attempts + 1))
		sleep 0.2
	done
}

if [[ -z "$transcript_path" || ! -f "$transcript_path" ]]; then
	log "WARN" "Stop missing transcript session_id=$session_id"
else
	parsed="$(parse_transcript_with_retry)"

	turn_start_ns="$(echo "$parsed" | jq -r '.turn.start_ns')"
	turn_end_ns="$(echo "$parsed" | jq -r '.turn.end_ns')"

	# Emit (or re-emit) the root session span eagerly so the trace is visible
	# in the UI before the session ends. The server upserts on span_id conflict,
	# so re-emitting with an updated end time is safe.
	if [[ "$session_init_source" == "lazy_init" ]]; then
		session_hook_attr="StopFallback"
		session_lifecycle_attr="stop_fallback"
	else
		session_hook_attr="Stop"
		session_lifecycle_attr="in_progress"
	fi
	session_attrs="{\"source\":\"claude-code\",\"hook\":\"$session_hook_attr\",\"node_type\":\"WORKFLOW\",\"session.lifecycle\":\"$session_lifecycle_attr\"}"
	add_span_to_batch "$trace_id" "$session_span_id" "$session_parent_span_id" "Claude Code session" "1" "$session_start_ns" "$turn_end_ns" "$session_attrs" || true

	while IFS= read -r tool; do
		[[ -z "$tool" ]] && continue
		span_id="$(generate_span_id)"
		name="$(echo "$tool" | jq -r '.name')"
		start_ns="$(echo "$tool" | jq -r '.start_ns')"
		end_ns="$(echo "$tool" | jq -r '.end_ns')"
		attrs="$(echo "$tool" | jq -c '.attributes')"
		add_span_to_batch "$trace_id" "$span_id" "$session_span_id" "$name" "3" "$start_ns" "$end_ns" "$attrs" || true
	done < <(echo "$parsed" | jq -c '.tools[]?')

	while IFS= read -r llm; do
		[[ -z "$llm" ]] && continue
		span_id="$(generate_span_id)"
		name="$(echo "$llm" | jq -r '.name')"
		start_ns="$(echo "$llm" | jq -r '.start_ns')"
		end_ns="$(echo "$llm" | jq -r '.end_ns')"
		attrs="$(echo "$llm" | jq -c '.attributes')"
		add_span_to_batch "$trace_id" "$span_id" "$session_span_id" "$name" "3" "$start_ns" "$end_ns" "$attrs" || true
	done < <(echo "$parsed" | jq -c '.llms[]?')
fi

emit_spans_batch_file "$spans_file" || true

log "INFO" "Stop finalized session_id=$session_id"
