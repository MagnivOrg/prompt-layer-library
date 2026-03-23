from dataclasses import dataclass
import base64
import binascii
import json
import os
import secrets
import uuid
from urllib import error, request

from state import acquire_lock, queue_lock_path, release_lock


@dataclass
class SpanSpec:
    trace_id: str
    span_id: str
    parent_span_id: str
    name: str
    kind: str
    start_ns: str
    end_ns: str
    attrs: dict


def compact_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def generate_trace_id() -> str:
    return secrets.token_hex(16)


def generate_span_id() -> str:
    return secrets.token_hex(8)


def generate_session_id() -> str:
    return str(uuid.uuid4())


def normalize_hex_id(raw: str, expected_len: int, fallback: str) -> str:
    clean = "".join(ch for ch in str(raw).lower() if ch in "0123456789abcdef")
    if not clean:
        clean = fallback
    if len(clean) > expected_len:
        clean = clean[:expected_len]
    if len(clean) < expected_len:
        clean = clean.ljust(expected_len, "0")
    return clean


def hex_to_base64(hex_value: str) -> str:
    raw = binascii.unhexlify(hex_value)
    return base64.b64encode(raw).decode("ascii")


def kind_int_to_string(kind) -> str:
    value = str(kind)
    return {
        "0": "SPAN_KIND_UNSPECIFIED",
        "1": "SPAN_KIND_INTERNAL",
        "2": "SPAN_KIND_SERVER",
        "3": "SPAN_KIND_CLIENT",
        "4": "SPAN_KIND_PRODUCER",
        "5": "SPAN_KIND_CONSUMER",
    }.get(value, "SPAN_KIND_UNSPECIFIED")


def otlp_attribute_value(value):
    if isinstance(value, str):
        return {"stringValue": value}
    if isinstance(value, bool):
        return {"boolValue": value}
    if isinstance(value, int):
        return {"intValue": str(value)}
    if isinstance(value, float):
        if value.is_integer():
            return {"intValue": str(int(value))}
        return {"doubleValue": value}
    return {"stringValue": compact_json(value)}


def build_span(spec: SpanSpec):
    trace_id = normalize_hex_id(spec.trace_id, 32, generate_trace_id())
    span_id = normalize_hex_id(spec.span_id, 16, generate_span_id())
    parent_span = ""
    if spec.parent_span_id:
        parent_span = normalize_hex_id(spec.parent_span_id, 16, generate_span_id())

    attributes = []
    for key, value in (spec.attrs or {}).items():
        if value is None:
            continue
        attributes.append({"key": key, "value": otlp_attribute_value(value)})

    span = {
        "traceId": hex_to_base64(trace_id),
        "spanId": hex_to_base64(span_id),
        "name": spec.name,
        "kind": kind_int_to_string(spec.kind),
        "startTimeUnixNano": str(spec.start_ns),
        "endTimeUnixNano": str(spec.end_ns),
        "attributes": attributes,
    }
    if parent_span:
        span["parentSpanId"] = hex_to_base64(parent_span)
    return span


def build_payload(spans):
    return {
        "resourceSpans": [
            {
                "resource": {
                    "attributes": [
                        {"key": "service.name", "value": {"stringValue": "claude-code"}}
                    ]
                },
                "scopeSpans": [{"spans": spans}],
            }
        ]
    }


def http_post_json(endpoint: str, payload, api_key: str = "", user_agent: str = "", timeout: int = 12):
    body = compact_json(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-Api-Key"] = api_key
    if user_agent:
        headers["User-Agent"] = user_agent

    req = request.Request(endpoint, data=body, headers=headers, method="POST")
    try:
        with request.urlopen(req, timeout=max(timeout, 1)) as response:
            return response.getcode(), response.read().decode("utf-8")
    except error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", errors="replace")
    except Exception:
        return 0, ""


def parse_partial_success(response_text: str):
    if not response_text:
        return 0, ""
    try:
        parsed = json.loads(response_text)
    except Exception:
        return 0, ""
    partial = parsed.get("partialSuccess", {})
    if not isinstance(partial, dict):
        return 0, ""
    rejected = partial.get("rejectedSpans", 0)
    try:
        rejected_int = int(rejected)
    except Exception:
        rejected_int = 0
    message = partial.get("errorMessage", "")
    return rejected_int, str(message) if message else ""


def post_otlp_payload(ctx, payload):
    return http_post_json(
        ctx.otlp_endpoint,
        payload,
        api_key=ctx.api_key,
        user_agent=ctx.user_agent,
        timeout=ctx.otlp_max_time,
    )


def append_queue_payload(ctx, payload):
    if not ctx.queue_file or not ctx.lock_dir:
        return False

    os.makedirs(os.path.dirname(ctx.queue_file), exist_ok=True)
    lock_path = queue_lock_path(ctx.lock_dir)
    if not acquire_lock(lock_path):
        return False
    try:
        with open(ctx.queue_file, "a", encoding="utf-8") as f:
            f.write(compact_json(payload))
            f.write("\n")
        try:
            os.chmod(ctx.queue_file, 0o600)
        except Exception:
            pass
        return True
    finally:
        release_lock(lock_path)


def read_queue_payloads(queue_file: str):
    if not os.path.exists(queue_file) or os.path.getsize(queue_file) == 0:
        return []

    payloads = []
    with open(queue_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                payloads.append(json.loads(line))
            except Exception:
                continue
    return payloads


def write_queue_payloads(queue_file: str, payloads) -> None:
    os.makedirs(os.path.dirname(queue_file), exist_ok=True)
    with open(queue_file, "w", encoding="utf-8") as f:
        for payload in payloads:
            f.write(compact_json(payload))
            f.write("\n")
    try:
        os.chmod(queue_file, 0o600)
    except Exception:
        pass


def post_payload_result(ctx, payload):
    status, response_text = post_otlp_payload(ctx, payload)
    if status != 200:
        return 1
    rejected, _ = parse_partial_success(response_text)
    if rejected:
        return 2
    return 0


def drain_queue(ctx):
    if not ctx.queue_file or not ctx.lock_dir or not os.path.exists(ctx.queue_file):
        return
    if ctx.queue_drain_limit <= 0:
        return

    lock_path = queue_lock_path(ctx.lock_dir)
    if not acquire_lock(lock_path):
        return
    try:
        payloads = read_queue_payloads(ctx.queue_file)
        if not payloads:
            return

        max_attempts = min(len(payloads), ctx.queue_drain_limit)
        remaining_start = max_attempts
        for idx in range(max_attempts):
            result = post_payload_result(ctx, payloads[idx])
            if result == 0:
                continue
            if result == 2:
                continue
            remaining_start = idx
            break

        if max_attempts < len(payloads):
            remaining = payloads[remaining_start:]
        elif remaining_start < max_attempts:
            remaining = payloads[remaining_start:]
        else:
            remaining = payloads[max_attempts:]
        write_queue_payloads(ctx.queue_file, remaining)
    finally:
        release_lock(lock_path)


def send_payload_with_queueing(ctx, payload):
    drain_queue(ctx)
    result = post_payload_result(ctx, payload)
    if result == 1:
        append_queue_payload(ctx, payload)
    return result


def probe_endpoint(endpoint: str, api_key: str) -> str:
    status, _ = http_post_json(endpoint, {"resourceSpans": []}, api_key=api_key, timeout=12)
    return f"{status:03d}" if status else "000"
