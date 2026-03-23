from dataclasses import dataclass
import json
import os
import shutil
import time

from traceparent import parse_traceparent


@dataclass
class SessionState:
    trace_id: str = ""
    session_span_id: str = ""
    session_parent_span_id: str = ""
    session_start_ns: str = ""
    current_turn_start_ns: str = ""
    pending_tool_calls: str = ""
    session_init_source: str = ""
    session_traceparent_version: str = ""
    session_trace_flags: str = ""
    trace_context_source: str = ""

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict):
            return cls()
        return cls(**{field: str(data.get(field, "")) for field in cls.__dataclass_fields__})

    def to_dict(self):
        return {field: getattr(self, field, "") for field in self.__dataclass_fields__}


def compact_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def session_state_path(session_state_dir: str, session_id: str) -> str:
    return os.path.join(session_state_dir, f"{session_id}.json")


def load_session_state(session_state_dir: str, session_id: str):
    path = session_state_path(session_state_dir, session_id)
    if not os.path.exists(path):
        return SessionState(), path
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return SessionState(), path
    return SessionState.from_dict(data), path


def save_session_state(path: str, state: SessionState) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state.to_dict(), f, ensure_ascii=False, separators=(",", ":"))


def parse_pending_tool_calls(raw: str):
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except Exception:
        return []
    return data if isinstance(data, list) else []


def session_lock_path(lock_dir: str, session_id: str) -> str:
    return os.path.join(lock_dir, f"{session_id}.lock")


def queue_lock_path(lock_dir: str) -> str:
    return os.path.join(lock_dir, "queue.lock")


def acquire_lock(path: str, attempts: int = 250, sleep_seconds: float = 0.02) -> bool:
    for _ in range(attempts):
        try:
            os.mkdir(path)
            return True
        except FileExistsError:
            time.sleep(sleep_seconds)
        except Exception:
            return False
    return False


def release_lock(path: str) -> None:
    try:
        os.rmdir(path)
    except Exception:
        shutil.rmtree(path, ignore_errors=True)


def ensure_session_initialized(
    state: SessionState,
    *,
    traceparent_raw: str,
    generate_trace_id,
    generate_span_id,
    requested_start_ns=None,
):
    if state.trace_id and state.session_span_id:
        if not state.session_start_ns:
            state.session_start_ns = str(requested_start_ns or time.time_ns())
        if not state.session_init_source:
            state.session_init_source = "unknown"
        if not state.pending_tool_calls:
            state.pending_tool_calls = "[]"
        if not state.session_parent_span_id:
            state.session_parent_span_id = ""
        if not state.session_traceparent_version:
            state.session_traceparent_version = ""
        if not state.session_trace_flags:
            state.session_trace_flags = ""
        if not state.trace_context_source:
            state.trace_context_source = "generated"
        return state, False

    trace_context = parse_traceparent(traceparent_raw)
    if requested_start_ns is None:
        requested_start_ns = time.time_ns()

    state.trace_id = trace_context["trace_id"] if trace_context else generate_trace_id()
    state.session_span_id = generate_span_id()
    state.session_parent_span_id = trace_context["parent_span_id"] if trace_context else ""
    state.session_start_ns = str(requested_start_ns)
    state.current_turn_start_ns = ""
    state.pending_tool_calls = "[]"
    state.session_init_source = "lazy_init"
    state.session_traceparent_version = trace_context["version"] if trace_context else ""
    state.session_trace_flags = trace_context["trace_flags"] if trace_context else ""
    state.trace_context_source = trace_context["source"] if trace_context else "generated"
    return state, True
