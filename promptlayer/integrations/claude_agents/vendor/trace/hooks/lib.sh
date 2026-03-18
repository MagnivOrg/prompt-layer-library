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
	local cmd
	for cmd in jq curl uuidgen python3; do
		if ! command -v "$cmd" >/dev/null 2>&1; then
			log "ERROR" "Missing required command: $cmd"
			return 1
		fi
	done
	if [[ -z "$PL_API_KEY" ]]; then
		log "ERROR" "PROMPTLAYER_API_KEY is not set"
		return 1
	fi
	return 0
}

generate_trace_id() {
	uuidgen | tr -d '-' | tr '[:upper:]' '[:lower:]'
}

generate_span_id() {
	uuidgen | tr -d '-' | tr '[:upper:]' '[:lower:]' | cut -c1-16
}

parse_traceparent() {
	local raw="${1:-}"
	[[ -z "$raw" ]] && return 1

	raw="$(printf '%s' "$raw" | tr '[:upper:]' '[:lower:]')"

	if [[ ! "$raw" =~ ^([0-9a-f]{2})-([0-9a-f]{32})-([0-9a-f]{16})-([0-9a-f]{2})(-.+)?$ ]]; then
		return 1
	fi

	local version="${BASH_REMATCH[1]}"
	local trace_id="${BASH_REMATCH[2]}"
	local parent_span_id="${BASH_REMATCH[3]}"
	local trace_flags="${BASH_REMATCH[4]}"
	local suffix="${BASH_REMATCH[5]:-}"

	if [[ "$version" == "ff" ]]; then
		return 1
	fi
	if [[ "$version" == "00" && -n "$suffix" ]]; then
		return 1
	fi
	if [[ "$trace_id" == "00000000000000000000000000000000" ]]; then
		return 1
	fi
	if [[ "$parent_span_id" == "0000000000000000" ]]; then
		return 1
	fi

	printf '%s %s %s %s\n' "$version" "$trace_id" "$parent_span_id" "$trace_flags"
}

load_initial_trace_context() {
	PL_INITIAL_TRACEPARENT_VERSION=""
	PL_INITIAL_TRACE_ID=""
	PL_INITIAL_PARENT_SPAN_ID=""
	PL_INITIAL_TRACE_FLAGS=""
	PL_INITIAL_TRACE_CONTEXT_SOURCE="generated"

	local raw="${PROMPTLAYER_TRACEPARENT:-}"
	if [[ -z "$raw" ]]; then
		return 0
	fi

	local parsed
	if ! parsed="$(parse_traceparent "$raw")"; then
		log "WARN" "Ignoring invalid PROMPTLAYER_TRACEPARENT"
		return 1
	fi

	read -r PL_INITIAL_TRACEPARENT_VERSION PL_INITIAL_TRACE_ID PL_INITIAL_PARENT_SPAN_ID PL_INITIAL_TRACE_FLAGS <<<"$parsed"
	PL_INITIAL_TRACE_CONTEXT_SOURCE="external_traceparent"
	return 0
}

normalize_hex_id() {
	local raw="$1"
	local expected_len="$2"
	local fallback="$3"
	local label="$4"

	local clean
	clean="$(echo "$raw" | tr -cd '[:xdigit:]' | tr '[:upper:]' '[:lower:]')"
	if [[ -z "$clean" ]]; then
		clean="$fallback"
	fi

	if ((${#clean} > expected_len)); then
		clean="${clean:0:expected_len}"
	elif ((${#clean} < expected_len)); then
		clean="$(printf "%-${expected_len}s" "$clean" | tr ' ' '0')"
	fi

	if [[ "$clean" != "$raw" ]]; then
		log "WARN" "Normalized $label from '$raw' to '$clean'"
	fi

	echo "$clean"
}

hex_to_base64() {
	local hex="$1"
	python3 "$PL_HOOKS_DIR/hook_utils.py" hex_to_base64 "$hex"
}

now_ns() {
	python3 "$PL_HOOKS_DIR/hook_utils.py" now_ns
}

session_state_file() {
	echo "$PL_SESSION_STATE_DIR/$1.json"
}

acquire_session_lock() {
	local sid="$1"
	[[ -z "$sid" ]] && return 1

	local lock_dir="$PL_LOCK_DIR/$sid.lock"
	local attempts=0
	while ! mkdir "$lock_dir" 2>/dev/null; do
		attempts=$((attempts + 1))
		if ((attempts >= 250)); then
			log "ERROR" "Timed out waiting for session lock session_id=$sid"
			return 1
		fi
		sleep 0.02
	done

	export PL_HELD_SESSION_LOCK="$lock_dir"
	return 0
}

release_session_lock() {
	local lock_dir="${PL_HELD_SESSION_LOCK:-}"
	if [[ -n "$lock_dir" ]]; then
		rmdir "$lock_dir" 2>/dev/null || rm -rf "$lock_dir"
		unset PL_HELD_SESSION_LOCK
	fi
	return 0
}

acquire_queue_lock() {
	local lock_dir="$PL_LOCK_DIR/queue.lock"
	local attempts=0
	while ! mkdir "$lock_dir" 2>/dev/null; do
		attempts=$((attempts + 1))
		if ((attempts >= 250)); then
			log "ERROR" "Timed out waiting for queue lock"
			return 1
		fi
		sleep 0.02
	done

	export PL_HELD_QUEUE_LOCK="$lock_dir"
	return 0
}

release_queue_lock() {
	local lock_dir="${PL_HELD_QUEUE_LOCK:-}"
	if [[ -n "$lock_dir" ]]; then
		rmdir "$lock_dir" 2>/dev/null || rm -rf "$lock_dir"
		unset PL_HELD_QUEUE_LOCK
	fi
	return 0
}

get_session_state() {
	local sid="$1"
	local key="$2"
	local f
	f="$(session_state_file "$sid")"
	if [[ -f "$f" ]]; then
		jq -r ".${key} // empty" "$f" 2>/dev/null || true
	fi
}

set_session_state() {
	local sid="$1"
	local key="$2"
	local val="$3"
	local f
	f="$(session_state_file "$sid")"
	local current
	current='{}'
	if [[ -f "$f" ]]; then
		current="$(cat "$f")"
	fi
	echo "$current" | jq --arg k "$key" --arg v "$val" '.[$k] = $v' >"$f"
}

ensure_session_initialized() {
	local sid="$1"
	local requested_start_ns="${2:-}"
	[[ -z "$sid" ]] && return 1

	local trace_id session_span_id session_parent_span_id session_start_ns init_source pending_tool_calls
	trace_id="$(get_session_state "$sid" trace_id)"
	session_span_id="$(get_session_state "$sid" session_span_id)"
	session_parent_span_id="$(get_session_state "$sid" session_parent_span_id)"
	session_start_ns="$(get_session_state "$sid" session_start_ns)"
	init_source="$(get_session_state "$sid" session_init_source)"
	pending_tool_calls="$(get_session_state "$sid" pending_tool_calls)"

	# Normal path: SessionStart already created state.
	if [[ -n "$trace_id" && -n "$session_span_id" ]]; then
		if [[ -z "$session_start_ns" ]]; then
			[[ -z "$requested_start_ns" ]] && requested_start_ns="$(now_ns)"
			set_session_state "$sid" session_start_ns "$requested_start_ns"
		fi
		if [[ -z "$init_source" ]]; then
			set_session_state "$sid" session_init_source "unknown"
		fi
		if [[ -z "$pending_tool_calls" ]]; then
			set_session_state "$sid" pending_tool_calls "[]"
		fi
		if [[ -z "$session_parent_span_id" ]]; then
			set_session_state "$sid" session_parent_span_id ""
		fi
		if [[ -z "$(get_session_state "$sid" session_traceparent_version)" ]]; then
			set_session_state "$sid" session_traceparent_version ""
		fi
		if [[ -z "$(get_session_state "$sid" session_trace_flags)" ]]; then
			set_session_state "$sid" session_trace_flags ""
		fi
		if [[ -z "$(get_session_state "$sid" trace_context_source)" ]]; then
			set_session_state "$sid" trace_context_source "generated"
		fi
		return 0
	fi

	# Fallback path for SDK environments that do not surface SessionStart.
	[[ -z "$requested_start_ns" ]] && requested_start_ns="$(now_ns)"
	load_initial_trace_context || true
	[[ -z "$trace_id" ]] && trace_id="${PL_INITIAL_TRACE_ID:-}"
	[[ -z "$trace_id" ]] && trace_id="$(generate_trace_id)"
	[[ -z "$session_span_id" ]] && session_span_id="$(generate_span_id)"

	set_session_state "$sid" trace_id "$trace_id"
	set_session_state "$sid" session_span_id "$session_span_id"
	set_session_state "$sid" session_parent_span_id "${PL_INITIAL_PARENT_SPAN_ID:-}"
	set_session_state "$sid" session_start_ns "$requested_start_ns"
	set_session_state "$sid" current_turn_start_ns ""
	set_session_state "$sid" pending_tool_calls "[]"
	set_session_state "$sid" session_init_source "lazy_init"
	set_session_state "$sid" session_traceparent_version "${PL_INITIAL_TRACEPARENT_VERSION:-}"
	set_session_state "$sid" session_trace_flags "${PL_INITIAL_TRACE_FLAGS:-}"
	set_session_state "$sid" trace_context_source "${PL_INITIAL_TRACE_CONTEXT_SOURCE:-generated}"

	log "INFO" "Session initialized lazily session_id=$sid trace_id=$trace_id"
}

post_otlp_payload_file() {
	local payload_file="$1"
	local status response_file
	response_file="$(mktemp "${TMPDIR:-/tmp}/pl-otlp-response.XXXXXX")"
	status="$(curl -sS -o "$response_file" -w "%{http_code}" -X POST \
		-H "Content-Type: application/json" \
		-H "X-Api-Key: $PL_API_KEY" \
		-H "User-Agent: $PL_USER_AGENT" \
		--connect-timeout "$PL_OTLP_CONNECT_TIMEOUT" \
		--max-time "$PL_OTLP_MAX_TIME" \
		"$PL_OTLP_ENDPOINT" \
		-d @"$payload_file" || true)"

	if [[ "$status" != "200" ]]; then
		log "ERROR" "Failed OTLP export status=$status"
		rm -f "$response_file"
		return 1
	fi

	# OTLP JSON can return 200 with partialSuccess and rejected spans.
	# Treat this as non-retryable (same payload is likely to be rejected again).
	if command -v jq >/dev/null 2>&1; then
		local rejected
		rejected="$(jq -r '.partialSuccess.rejectedSpans // 0' "$response_file" 2>/dev/null || echo 0)"
		if [[ "$rejected" != "0" ]]; then
			local message
			message="$(jq -r '.partialSuccess.errorMessage // "Unknown rejection"' "$response_file" 2>/dev/null || true)"
			log "ERROR" "OTLP partial success: rejected_spans=$rejected message=$message"
			rm -f "$response_file"
			return 2
		fi
	fi

	rm -f "$response_file"
	return 0
}

append_to_otlp_queue_file() {
	local payload_file="$1"
	acquire_queue_lock || return 1
	# Compact to single line to preserve ndjson format
	jq -c '.' "$payload_file" >>"$PL_QUEUE_FILE"
	chmod 600 "$PL_QUEUE_FILE" 2>/dev/null || true
	release_queue_lock
	return 0
}

drain_otlp_queue() {
	if [[ ! -s "$PL_QUEUE_FILE" ]]; then
		return 0
	fi

	local drain_limit="$PL_QUEUE_DRAIN_LIMIT"
	if [[ ! "$drain_limit" =~ ^[0-9]+$ ]]; then
		drain_limit=10
	fi
	if ((drain_limit <= 0)); then
		return 0
	fi

	acquire_queue_lock || return 0

	local -a queue_payloads
	mapfile -t queue_payloads <"$PL_QUEUE_FILE" || true

	local total max_attempts i rc replayed dropped retryable_fail fail_index
	total="${#queue_payloads[@]}"
	if ((total == 0)); then
		release_queue_lock
		return 0
	fi

	max_attempts=$((total < drain_limit ? total : drain_limit))
	replayed=0
	dropped=0
	retryable_fail=0
	fail_index=-1

	local queued_tmp
	queued_tmp="$(mktemp "${TMPDIR:-/tmp}/pl-otlp-queued.XXXXXX")"

	for ((i = 0; i < max_attempts; i++)); do
		if [[ -z "${queue_payloads[$i]}" ]]; then
			continue
		fi

		printf '%s' "${queue_payloads[$i]}" >"$queued_tmp"

		if post_otlp_payload_file "$queued_tmp"; then
			replayed=$((replayed + 1))
			continue
		else
			rc=$?
		fi

		if [[ "$rc" == "2" ]]; then
			dropped=$((dropped + 1))
			continue
		fi

		retryable_fail=1
		fail_index="$i"
		break
	done

	rm -f "$queued_tmp"

	local tmp remaining_start
	tmp="$(mktemp "${TMPDIR:-/tmp}/pl-otlp-queue.XXXXXX")"
	if ((retryable_fail == 1)); then
		remaining_start="$fail_index"
	else
		remaining_start="$max_attempts"
	fi

	for ((i = remaining_start; i < total; i++)); do
		printf '%s\n' "${queue_payloads[$i]}" >>"$tmp"
	done

	mv "$tmp" "$PL_QUEUE_FILE"
	chmod 600 "$PL_QUEUE_FILE" 2>/dev/null || true
	release_queue_lock

	if ((replayed > 0)); then
		log "INFO" "Drained queued OTLP payloads count=$replayed"
	fi
	if ((dropped > 0)); then
		log "WARN" "Dropped non-retryable queued OTLP payloads count=$dropped"
	fi

	return 0
}

send_otlp_payload_file() {
	local payload_file="$1"
	local rc

	drain_otlp_queue || true

	if post_otlp_payload_file "$payload_file"; then
		return 0
	else
		rc=$?
	fi

	if [[ "$rc" == "1" ]]; then
		append_to_otlp_queue_file "$payload_file" || log "ERROR" "Failed to append OTLP payload to queue"
	fi
	return 1
}

kind_int_to_string() {
	case "$1" in
	0) echo "SPAN_KIND_UNSPECIFIED" ;;
	1) echo "SPAN_KIND_INTERNAL" ;;
	2) echo "SPAN_KIND_SERVER" ;;
	3) echo "SPAN_KIND_CLIENT" ;;
	4) echo "SPAN_KIND_PRODUCER" ;;
	5) echo "SPAN_KIND_CONSUMER" ;;
	*) echo "SPAN_KIND_UNSPECIFIED" ;;
	esac
}

build_span_json() {
	local trace_id="$1"
	local span_id="$2"
	local parent_span_id="$3"
	local name="$4"
	local kind="$5"
	local start_ns="$6"
	local end_ns="$7"
	local attrs_json="$8"

	# Convert integer kind to protobuf JSON enum string if needed
	if [[ "$kind" =~ ^[0-9]+$ ]]; then
		kind="$(kind_int_to_string "$kind")"
	fi

	trace_id="$(normalize_hex_id "$trace_id" 32 "$(generate_trace_id)" "trace_id")"
	span_id="$(normalize_hex_id "$span_id" 16 "$(generate_span_id)" "span_id")"
	if [[ -n "$parent_span_id" ]]; then
		parent_span_id="$(normalize_hex_id "$parent_span_id" 16 "$(generate_span_id)" "parent_span_id")"
	fi

	local trace_id_b64 span_id_b64 parent_span_id_b64
	trace_id_b64="$(hex_to_base64 "$trace_id")"
	span_id_b64="$(hex_to_base64 "$span_id")"
	parent_span_id_b64=""
	if [[ -n "$parent_span_id" ]]; then
		parent_span_id_b64="$(hex_to_base64 "$parent_span_id")"
	fi

	local span_json
	span_json="$(jq -cn \
		--arg trace_id "$trace_id_b64" \
		--arg span_id "$span_id_b64" \
		--arg parent_span_id "$parent_span_id_b64" \
		--arg name "$name" \
		--arg kind "$kind" \
		--arg start "$start_ns" \
		--arg end "$end_ns" \
		--argjson attributes "$attrs_json" \
		'{
      traceId: $trace_id,
      spanId: $span_id,
      parentSpanId: (if $parent_span_id == "" then null else $parent_span_id end),
      name: $name,
      kind: $kind,
      startTimeUnixNano: $start,
      endTimeUnixNano: $end,
      attributes: (
        $attributes
        | to_entries
        | map(select(.value != null))
        | map(
            . as $kv
            | {
                key: $kv.key,
                value: (
                  if ($kv.value | type) == "string" then
                    {stringValue: $kv.value}
                  elif ($kv.value | type) == "boolean" then
                    {boolValue: $kv.value}
                  elif ($kv.value | type) == "number" then
                    if ($kv.value | floor) == $kv.value then
                      {intValue: ($kv.value | tostring)}
                    else
                      {doubleValue: $kv.value}
                    end
                  else
                    {stringValue: ($kv.value | tojson)}
                  end
                )
              }
          )
      )
    }')"

	echo "$span_json"
}

emit_spans_batch_file() {
	local spans_file="$1"
	if [[ ! -s "$spans_file" ]]; then
		return 0
	fi

	local payload_file
	payload_file="$(mktemp "${TMPDIR:-/tmp}/pl-otlp-batch.XXXXXX")"
	jq -cs '{resourceSpans:[{resource:{attributes:[{key:"service.name",value:{stringValue:"claude-code"}}]},scopeSpans:[{spans:.}]}]}' "$spans_file" >"$payload_file"
	send_otlp_payload_file "$payload_file"
	local rc=$?
	rm -f "$payload_file"
	return $rc
}

emit_span() {
	local trace_id="$1"
	local span_id="$2"
	local parent_span_id="$3"
	local name="$4"
	local kind="$5"
	local start_ns="$6"
	local end_ns="$7"
	local attrs_json="$8"

	local span_json spans_file
	span_json="$(build_span_json "$trace_id" "$span_id" "$parent_span_id" "$name" "$kind" "$start_ns" "$end_ns" "$attrs_json")" || return 1
	spans_file="$(mktemp "${TMPDIR:-/tmp}/pl-otlp-span.XXXXXX")"
	printf '%s\n' "$span_json" >"$spans_file"
	emit_spans_batch_file "$spans_file"
	local rc=$?
	rm -f "$spans_file"
	return $rc
}
