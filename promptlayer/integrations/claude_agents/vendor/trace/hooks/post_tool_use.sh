#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=plugins/trace/hooks/lib.sh
source "$SCRIPT_DIR/lib.sh"

tracing_enabled || exit 0
check_requirements || exit 0

input="$(cat)"
session_id="$(echo "$input" | jq -r '.session_id // empty')"
tool_name="$(echo "$input" | jq -r '.tool_name // empty')"
tool_input="$(echo "$input" | jq -c '.tool_input // {}')"
tool_output="$(echo "$input" | jq -c '.tool_response // .output // {}')"
[[ -z "$session_id" || -z "$tool_name" ]] && exit 0

ensure_session_initialized "$session_id"

trace_id="$(get_session_state "$session_id" trace_id)"
[[ -z "$trace_id" ]] && exit 0
turn_start_ns="$(get_session_state "$session_id" current_turn_start_ns)"
if [[ -z "$turn_start_ns" ]]; then
	set_session_state "$session_id" current_turn_start_ns "$(now_ns)"
fi

attrs="$(jq -nc \
	--arg source claude-code \
	--arg hook PostToolUse \
	--arg tool_name "$tool_name" \
	--argjson function_input "$tool_input" \
	--argjson function_output "$tool_output" \
	'{source:$source,hook:$hook,tool_name:$tool_name,node_type:"CODE_EXECUTION",function_input:$function_input,function_output:$function_output}')"

pending_tool_calls="$(get_session_state "$session_id" pending_tool_calls)"
[[ -z "$pending_tool_calls" ]] && pending_tool_calls='[]'

pending_tool_calls="$(echo "$pending_tool_calls" | jq -c --argjson attrs "$attrs" '. + [$attrs]')"
set_session_state "$session_id" pending_tool_calls "$pending_tool_calls"

log "INFO" "PostToolUse captured session_id=$session_id tool=$tool_name"
