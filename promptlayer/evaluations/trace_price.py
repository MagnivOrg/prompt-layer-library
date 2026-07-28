import asyncio
import time
from typing import Any, Callable, Optional, Sequence

from promptlayer.tables import api as tables_api

_TRACE_PRICE_MAX_WAIT_SECONDS = 5.0
_TRACE_PRICE_DELAYS_SECONDS: tuple[float, ...] = (1.0, 2.0, 2.0)


def _is_numeric_price(value: Any) -> bool:
    if isinstance(value, bool) or value is None:
        return False
    return isinstance(value, (int, float))


def _request_log_ids_from_trace_payload(payload: Any) -> list[int]:
    if not isinstance(payload, dict):
        return []
    spans = payload.get("spans")
    if not isinstance(spans, list):
        return []
    request_log_ids: list[int] = []
    for span in spans:
        if not isinstance(span, dict):
            continue
        request_log_id = span.get("request_log_id")
        if isinstance(request_log_id, bool) or request_log_id is None:
            continue
        try:
            parsed = int(request_log_id)
        except (TypeError, ValueError):
            continue
        if parsed not in request_log_ids:
            request_log_ids.append(parsed)
    return request_log_ids


def _trace_has_request_price(api_key: str, base_url: str, trace_id: str) -> bool:
    if not trace_id:
        return False
    trace_payload = tables_api.get_trace(
        api_key,
        base_url,
        False,
        trace_id,
    )
    request_log_ids = _request_log_ids_from_trace_payload(trace_payload)
    if not request_log_ids:
        return False
    for request_log_id in request_log_ids:
        request_payload = tables_api.get_request(
            api_key,
            base_url,
            False,
            request_log_id,
        )
        if isinstance(request_payload, dict) and _is_numeric_price(request_payload.get("price")):
            return True
    return False


async def _atrace_has_request_price(api_key: str, base_url: str, trace_id: str) -> bool:
    if not trace_id:
        return False
    trace_payload = await tables_api.aget_trace(
        api_key,
        base_url,
        False,
        trace_id,
    )
    request_log_ids = _request_log_ids_from_trace_payload(trace_payload)
    if not request_log_ids:
        return False
    for request_log_id in request_log_ids:
        request_payload = await tables_api.aget_request(
            api_key,
            base_url,
            False,
            request_log_id,
        )
        if isinstance(request_payload, dict) and _is_numeric_price(request_payload.get("price")):
            return True
    return False


def wait_for_trace_request_price(
    api_key: str,
    base_url: str,
    trace_id: str,
    *,
    max_wait_seconds: float = _TRACE_PRICE_MAX_WAIT_SECONDS,
    delays_seconds: Sequence[float] = _TRACE_PRICE_DELAYS_SECONDS,
    sleep: Optional[Callable[[float], Any]] = None,
) -> None:
    if sleep is None:
        sleep = time.sleep
    deadline = time.monotonic() + max_wait_seconds
    for delay in delays_seconds:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        sleep(min(float(delay), remaining))
        if _trace_has_request_price(api_key, base_url, trace_id):
            return


async def await_for_trace_request_price(
    api_key: str,
    base_url: str,
    trace_id: str,
    *,
    max_wait_seconds: float = _TRACE_PRICE_MAX_WAIT_SECONDS,
    delays_seconds: Sequence[float] = _TRACE_PRICE_DELAYS_SECONDS,
) -> None:
    deadline = time.monotonic() + max_wait_seconds
    for delay in delays_seconds:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        await asyncio.sleep(min(float(delay), remaining))
        if await _atrace_has_request_price(api_key, base_url, trace_id):
            return
