"""Live Smart Table execution progress via Centrifugo websockets."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set

import urllib3

from promptlayer.utils import (
    _get_websocket_token,
    centrifugo_client,
    centrifugo_subscription,
)

logger = logging.getLogger(__name__)

SMART_TABLE_EXECUTION_STATUS_UPDATE = "SMART_TABLE_EXECUTION_STATUS_UPDATE"
SMART_SHEET_CHANNEL_TEMPLATE = "smart_sheets:smart_sheet#{sheet_id}"
_TERMINAL_EXECUTION_STATUSES = frozenset({"completed", "failed", "cancelled"})
_DEFAULT_SAFETY_POLL_INTERVAL_SECONDS = 5.0

ExecutionUpdateCallback = Callable[[Dict[str, Any]], None]
SafetyPollCallback = Callable[[], Awaitable[Optional[Dict[str, Dict[str, Any]]]]]


def smart_sheet_channel(sheet_id: Any) -> str:
    return SMART_SHEET_CHANNEL_TEMPLATE.format(sheet_id=sheet_id)


def execution_update_to_operation_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Map WS execution status fields onto the public operation status shape.

    Only copy counters that are present on the event. Status-only updates must not
    invent ``completed/failed/total = 0``, or count-based terminal detection can
    treat ``0 + 0 >= 0`` as done while the operation is still ``running``.
    """
    execution_id = str(payload.get("execution_id") or "")
    mapped: Dict[str, Any] = {
        "operation_id": execution_id,
        "execution_id": execution_id,
        "status": payload.get("status"),
    }
    if "completed" in payload:
        mapped["completed_count"] = payload["completed"]
    if "failed" in payload:
        mapped["failed_count"] = payload["failed"]
    if "total" in payload:
        mapped["cell_count"] = payload["total"]
    # Derive pending only when all three counters are present so count-based
    # terminal detection matches REST safety-poll payloads.
    if not all(key in mapped for key in ("completed_count", "failed_count", "cell_count")):
        return mapped
    try:
        completed_n = int(mapped["completed_count"])
        failed_n = int(mapped["failed_count"])
        total_n = int(mapped["cell_count"])
    except (TypeError, ValueError):
        return mapped
    if completed_n >= 0 and failed_n >= 0 and total_n >= 0:
        mapped["pending_count"] = max(total_n - completed_n - failed_n, 0)
    return mapped


def _parse_execution_update(data: Any) -> Optional[Dict[str, Any]]:
    if isinstance(data, str):
        try:
            payload = json.loads(data)
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
    elif isinstance(data, dict):
        payload = data
    else:
        return None
    if not isinstance(payload, dict):
        return None
    execution_id = payload.get("execution_id")
    status = payload.get("status")
    if execution_id is None or not str(execution_id).strip():
        return None
    if not isinstance(status, str) or not status.strip():
        return None
    return payload


def _non_negative_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _normalize_operation_status_payload(payload: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Unwrap GET /operations/:id which returns ``{"success": true, "operation": {...}}``."""
    if not isinstance(payload, dict):
        return payload
    nested = payload.get("operation")
    if isinstance(nested, dict):
        return nested
    return payload


def operation_payload_is_terminal(payload: Optional[Dict[str, Any]]) -> bool:
    """True when status is terminal, or when cell counts show the operation finished.

    Matches REST polling: Redis status can lag behind finished cells, so treat
    ``pending_count == 0`` and ``completed + failed >= cell_count`` as done when
    ``cell_count > 0``. A zero/missing total alone must not end the wait (status-only
    WS events used to invent zeros and falsely trip this path).
    """
    payload = _normalize_operation_status_payload(payload)
    if not isinstance(payload, dict):
        return False
    status = payload.get("status")
    if isinstance(status, str) and status.strip().lower() in _TERMINAL_EXECUTION_STATUSES:
        return True
    pending = _non_negative_int(payload.get("pending_count"))
    completed = _non_negative_int(payload.get("completed_count"))
    failed = _non_negative_int(payload.get("failed_count"))
    cell_count = _non_negative_int(payload.get("cell_count"))
    if pending is None or completed is None or failed is None or cell_count is None:
        return False
    if pending > 0 or cell_count == 0:
        return False
    return completed + failed >= cell_count


async def await_sheet_execution_progress(
    *,
    api_key: str,
    base_url: str,
    sheet_id: Any,
    execution_ids: List[str],
    on_execution_update: ExecutionUpdateCallback,
    safety_poll: Optional[SafetyPollCallback] = None,
    safety_poll_interval_seconds: float = _DEFAULT_SAFETY_POLL_INTERVAL_SECONDS,
    timeout_seconds: float,
) -> Dict[str, Dict[str, Any]]:
    """Subscribe to sheet execution status updates until all tracked IDs are terminal.

    Raises on websocket token/connect/subscribe failure so callers can fall back to REST.
    """
    tracked: Set[str] = {str(item) for item in execution_ids if item is not None and str(item).strip()}
    if not tracked:
        return {}

    channel_name = smart_sheet_channel(sheet_id)
    headers = {"X-API-KEY": api_key}
    websocket_token = await _get_websocket_token(base_url, channel_name, headers)
    token = websocket_token["token_details"]["token"]

    ws_scheme = "wss" if urllib3.util.parse_url(base_url).scheme == "https" else "ws"
    address = urllib3.util.parse_url(base_url)._replace(scheme=ws_scheme, path="/connection/websocket").url

    states: Dict[str, Dict[str, Any]] = {execution_id: {} for execution_id in tracked}
    done = asyncio.Event()

    def _apply_update(execution_id: str, payload: Dict[str, Any]) -> None:
        states[execution_id] = payload
        on_execution_update(payload)
        # Use the same terminal rules as REST polling (status *or* finished counts)
        # so a safety-poll payload that lags on Redis status still completes the wait.
        if all(operation_payload_is_terminal(states[item]) for item in tracked):
            done.set()

    async def message_listener(message_name: str, data: str) -> None:
        if message_name != SMART_TABLE_EXECUTION_STATUS_UPDATE or done.is_set():
            return
        payload = _parse_execution_update(data)
        if payload is None:
            return
        execution_id = str(payload["execution_id"])
        if execution_id not in tracked:
            return
        _apply_update(execution_id, execution_update_to_operation_payload(payload))

    async def safety_poll_loop() -> None:
        if safety_poll is None:
            await done.wait()
            return
        while not done.is_set():
            try:
                await asyncio.wait_for(done.wait(), timeout=safety_poll_interval_seconds)
                return
            except asyncio.TimeoutError:
                pass
            try:
                polled = await safety_poll()
            except Exception:
                logger.debug("Safety poll for sheet execution progress failed", exc_info=True)
                continue
            if not polled:
                continue
            for execution_id, payload in polled.items():
                if execution_id not in tracked or not isinstance(payload, dict):
                    continue
                _apply_update(execution_id, payload)

    async with centrifugo_client(address, token) as client:
        async with centrifugo_subscription(client, channel_name, message_listener):
            safety_task = asyncio.create_task(safety_poll_loop())
            try:
                await asyncio.wait_for(done.wait(), timeout=timeout_seconds)
            finally:
                done.set()
                safety_task.cancel()
                try:
                    await safety_task
                except asyncio.CancelledError:
                    pass

    return states
