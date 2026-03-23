def parse_traceparent(raw: str):
    if not raw:
        return None

    parts = raw.lower().split("-")
    if len(parts) < 4:
        return None

    version, trace_id, parent_span_id, trace_flags = parts[:4]
    suffix = parts[4:]

    if len(version) != 2 or len(trace_id) != 32 or len(parent_span_id) != 16 or len(trace_flags) != 2:
        return None

    hexdigits = set("0123456789abcdef")
    if any(ch not in hexdigits for ch in version + trace_id + parent_span_id + trace_flags):
        return None
    if version == "ff":
        return None
    if version == "00" and suffix:
        return None
    if trace_id == "0" * 32 or parent_span_id == "0" * 16:
        return None

    return {
        "version": version,
        "trace_id": trace_id,
        "parent_span_id": parent_span_id,
        "trace_flags": trace_flags,
        "source": "external_traceparent",
    }
