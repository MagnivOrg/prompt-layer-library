import hashlib
import re

_TRACE_HEX_RE = re.compile(r"^[0-9a-fA-F]{32}$")


def map_trace_id(original_trace_id: str) -> str:
    suffix = original_trace_id[len("trace_") :] if original_trace_id.startswith("trace_") else original_trace_id
    if _TRACE_HEX_RE.fullmatch(suffix):
        return suffix.lower()
    return hashlib.sha256(original_trace_id.encode("utf-8")).hexdigest()[:32]


def map_span_id(original_span_id: str) -> str:
    return hashlib.sha256(original_span_id.encode("utf-8")).hexdigest()[:16]


def synthetic_root_span_id(original_trace_id: str) -> str:
    return hashlib.sha256(f"{original_trace_id}:root".encode("utf-8")).hexdigest()[:16]


def hex_id_to_int(value: str) -> int:
    return int(value, 16)
