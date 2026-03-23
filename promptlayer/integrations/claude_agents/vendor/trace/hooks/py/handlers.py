import json
import os
import time
from typing import Optional

from otlp import (
    SpanSpec,
    build_payload,
    build_span,
    generate_session_id,
    generate_span_id,
    generate_trace_id,
    send_payload_with_queueing,
)
from state import (
    acquire_lock,
    ensure_session_initialized,
    load_session_state,
    parse_pending_tool_calls,
    release_lock,
    save_session_state,
    session_lock_path,
)
from stop_parser import build_stop_hook_span_specs, parse_transcript
from traceparent import parse_traceparent


def read_stdin_json(raw: str):
    if not raw.strip():
        return {}
    try:
        data = json.loads(raw)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def handle_session_start(ctx, raw_input: str) -> str:
    input_data = read_stdin_json(raw_input)
    session_id = input_data.get("session_id")
    session_id = str(session_id) if session_id else generate_session_id()

    state, path = load_session_state(ctx.session_state_dir, session_id)
    existing = bool(state.trace_id and state.session_span_id)
    if existing:
        state, _ = ensure_session_initialized(
            state,
            traceparent_raw=os.environ.get("PROMPTLAYER_TRACEPARENT", ""),
            generate_trace_id=generate_trace_id,
            generate_span_id=generate_span_id,
        )
        status = "existing"
    else:
        trace_context = parse_traceparent(os.environ.get("PROMPTLAYER_TRACEPARENT", ""))
        state.trace_id = trace_context["trace_id"] if trace_context else generate_trace_id()
        state.session_span_id = generate_span_id()
        state.session_parent_span_id = trace_context["parent_span_id"] if trace_context else ""
        state.session_start_ns = str(time.time_ns())
        state.current_turn_start_ns = ""
        state.pending_tool_calls = "[]"
        state.session_init_source = "session_start_hook"
        state.session_traceparent_version = trace_context["version"] if trace_context else ""
        state.session_trace_flags = trace_context["trace_flags"] if trace_context else ""
        state.trace_context_source = trace_context["source"] if trace_context else "generated"
        status = "captured"

    save_session_state(path, state)
    return f"{session_id}\t{state.trace_id}\t{status}"


def handle_user_prompt_submit(ctx, raw_input: str) -> str:
    input_data = read_stdin_json(raw_input)
    session_id = input_data.get("session_id")
    if not session_id:
        return ""

    state, path = load_session_state(ctx.session_state_dir, str(session_id))
    state, _ = ensure_session_initialized(
        state,
        traceparent_raw=os.environ.get("PROMPTLAYER_TRACEPARENT", ""),
        generate_trace_id=generate_trace_id,
        generate_span_id=generate_span_id,
    )
    if not state.trace_id or not state.session_span_id:
        return ""

    state.current_turn_start_ns = str(time.time_ns())
    state.pending_tool_calls = "[]"
    save_session_state(path, state)
    return str(session_id)


def handle_post_tool_use(ctx, raw_input: str) -> str:
    input_data = read_stdin_json(raw_input)
    session_id = input_data.get("session_id")
    tool_name = input_data.get("tool_name")
    if not session_id or not tool_name:
        return ""

    tool_input = input_data.get("tool_input", {})
    tool_output = input_data.get("tool_response", input_data.get("output", {}))

    state, path = load_session_state(ctx.session_state_dir, str(session_id))
    state, _ = ensure_session_initialized(
        state,
        traceparent_raw=os.environ.get("PROMPTLAYER_TRACEPARENT", ""),
        generate_trace_id=generate_trace_id,
        generate_span_id=generate_span_id,
    )
    if not state.trace_id:
        return ""
    if not state.current_turn_start_ns:
        state.current_turn_start_ns = str(time.time_ns())

    pending_tool_calls = parse_pending_tool_calls(state.pending_tool_calls)
    pending_tool_calls.append(
        {
            "source": "claude-code",
            "hook": "PostToolUse",
            "tool_name": str(tool_name),
            "node_type": "CODE_EXECUTION",
            "function_input": tool_input,
            "function_output": tool_output,
        }
    )
    state.pending_tool_calls = json.dumps(pending_tool_calls, ensure_ascii=False, separators=(",", ":"))
    save_session_state(path, state)
    return f"{session_id}\t{tool_name}"


def handle_session_end(ctx, raw_input: str) -> str:
    input_data = read_stdin_json(raw_input)
    session_id = input_data.get("session_id")
    if not session_id:
        return ""

    lock_path = session_lock_path(ctx.lock_dir, str(session_id))
    if not acquire_lock(lock_path):
        return ""

    try:
        state, path = load_session_state(ctx.session_state_dir, str(session_id))
        if not state.trace_id or not state.session_span_id:
            return ""

        spec = build_span(
            SpanSpec(
                trace_id=state.trace_id,
                span_id=state.session_span_id,
                parent_span_id=state.session_parent_span_id,
                name="Claude Code session",
                kind="1",
                start_ns=state.session_start_ns or str(time.time_ns()),
                end_ns=str(time.time_ns()),
                attrs={
                    "source": "claude-code",
                    "hook": "SessionEnd",
                    "node_type": "WORKFLOW",
                    "session.lifecycle": "complete",
                },
            )
        )
        send_payload_with_queueing(ctx, build_payload([spec]))
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
        return str(session_id)
    finally:
        release_lock(lock_path)


def resolve_stop_session_id(input_data):
    session_id = input_data.get("session_id")
    transcript_path = input_data.get("transcript_path")
    if not session_id and transcript_path:
        session_id = os.path.basename(str(transcript_path))
        if session_id.endswith(".jsonl"):
            session_id = session_id[: -len(".jsonl")]
    return session_id


def handle_stop_hook(ctx, raw_input: str) -> str:
    input_data = read_stdin_json(raw_input)
    session_id = resolve_stop_session_id(input_data)
    transcript_path = input_data.get("transcript_path")
    if not session_id:
        return ""

    lock_path = session_lock_path(ctx.lock_dir, str(session_id))
    if not acquire_lock(lock_path):
        return ""

    try:
        state, path = load_session_state(ctx.session_state_dir, str(session_id))
        state, _ = ensure_session_initialized(
            state,
            traceparent_raw=os.environ.get("PROMPTLAYER_TRACEPARENT", ""),
            generate_trace_id=generate_trace_id,
            generate_span_id=generate_span_id,
        )
        if not state.trace_id or not state.session_span_id:
            return ""

        turn_start_ns = state.current_turn_start_ns or str(time.time_ns())
        pending_tool_calls = state.pending_tool_calls or "[]"
        state.current_turn_start_ns = ""
        state.pending_tool_calls = "[]"
        save_session_state(path, state)
    finally:
        release_lock(lock_path)

    if not transcript_path or not os.path.exists(str(transcript_path)):
        return f"{session_id}\tmissing_transcript"

    pending_payloads = parse_pending_tool_calls(pending_tool_calls)
    attempts = 0
    while True:
        parsed = parse_transcript(str(transcript_path), int(turn_start_ns), pending_payloads, str(session_id))
        if parsed.get("llms") or attempts >= 10:
            break
        attempts += 1
        time.sleep(0.2)

    span_specs = build_stop_hook_span_specs(
        parsed=parsed,
        trace_id=state.trace_id,
        session_span_id=state.session_span_id,
        session_parent_span_id=state.session_parent_span_id,
        session_start_ns=state.session_start_ns or str(time.time_ns()),
        session_init_source=state.session_init_source,
        generate_span_id=generate_span_id,
    )
    spans = [build_span(span_spec) for span_spec in span_specs]
    if spans:
        send_payload_with_queueing(ctx, build_payload(spans))
    return f"{session_id}\tok"


def handle_parse_stop_transcript(transcript_path: str, turn_start_ns: str, expected_session_id: Optional[str]) -> str:
    pending_raw = os.environ.get("PL_PENDING_TOOL_CALLS", "[]")
    pending_payloads = parse_pending_tool_calls(pending_raw)
    parsed = parse_transcript(transcript_path, int(turn_start_ns) or None, pending_payloads, expected_session_id)
    return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
