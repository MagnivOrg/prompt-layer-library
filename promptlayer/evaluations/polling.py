import asyncio
import time
from typing import Any, Callable, Dict, List, Optional

from promptlayer.evaluations.terminal import get_terminal
from promptlayer.evaluations.utils import (
    _DEFAULT_CELL_WAIT_TIMEOUT_SECONDS,
    _DEFAULT_POLL_INTERVAL_SECONDS,
    find_column_by_title,
    serialize_cell_value,
)
from promptlayer.evaluations.validation import api_error, timeout_error
from promptlayer.tables import api as tables_api
from promptlayer.types.table import Column, ResourceId

_TERMINAL_OPERATION_STATUSES = frozenset({"completed", "failed", "cancelled"})


def _iter_cell_updates(
    row: Dict[str, Any],
    columns_by_title_map: Dict[str, Column],
    values_by_title: Dict[str, Any],
):
    cells = row.get("cells") or {}
    for title, value in values_by_title.items():
        column = find_column_by_title(columns_by_title_map, title)
        if not column:
            continue
        cell = cells.get(str(column["id"]))
        cell_id = cell.get("id") if isinstance(cell, dict) else None
        if cell_id is None:
            continue
        serialized = serialize_cell_value(value if value is not None else "")
        yield cell_id, serialized


def fill_row_cells(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: ResourceId,
    sheet_id: ResourceId,
    row: Dict[str, Any],
    columns_by_title_map: Dict[str, Column],
    values_by_title: Dict[str, Any],
) -> None:
    for cell_id, serialized in _iter_cell_updates(row, columns_by_title_map, values_by_title):
        tables_api.update_sheet_cell(
            api_key,
            base_url,
            throw_on_error,
            table_id,
            sheet_id,
            cell_id,
            {"value": serialized, "display_value": str(serialized)},
        )


async def afill_row_cells(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: ResourceId,
    sheet_id: ResourceId,
    row: Dict[str, Any],
    columns_by_title_map: Dict[str, Column],
    values_by_title: Dict[str, Any],
) -> None:
    for cell_id, serialized in _iter_cell_updates(row, columns_by_title_map, values_by_title):
        await tables_api.aupdate_sheet_cell(
            api_key,
            base_url,
            throw_on_error,
            table_id,
            sheet_id,
            cell_id,
            {"value": serialized, "display_value": str(serialized)},
        )


def _poll_until(
    *,
    fetch: Callable[[], Any],
    is_done: Callable[[Any], bool],
    timeout_seconds: float,
    poll_interval_seconds: float,
    timeout_message: str,
    sleep: Optional[Callable[[float], Any]] = None,
    backoff: bool = False,
    on_update: Optional[Callable[[Any], None]] = None,
) -> Any:
    if sleep is None:
        sleep = time.sleep
    deadline = time.monotonic() + timeout_seconds
    delay = poll_interval_seconds
    last = None
    while True:
        last = fetch()
        if on_update is not None:
            on_update(last)
        if is_done(last):
            return last
        if time.monotonic() >= deadline:
            raise timeout_error(timeout_message)
        sleep(delay)
        if backoff:
            delay = min(delay * 1.5, 2.0)


async def _apoll_until(
    *,
    fetch,
    is_done: Callable[[Any], bool],
    timeout_seconds: float,
    poll_interval_seconds: float,
    timeout_message: str,
    backoff: bool = False,
    on_update: Optional[Callable[[Any], None]] = None,
) -> Any:
    deadline = time.monotonic() + timeout_seconds
    delay = poll_interval_seconds
    last = None
    while True:
        last = await fetch()
        if on_update is not None:
            on_update(last)
        if is_done(last):
            return last
        if time.monotonic() >= deadline:
            raise timeout_error(timeout_message)
        await asyncio.sleep(delay)
        if backoff:
            delay = min(delay * 1.5, 2.0)


_ACTIVE_CELL_STATUSES = ("STALE", "QUEUED", "DISPATCHED", "RUNNING")


def _status_count(counts: Dict[str, Any], status: str) -> Optional[int]:
    value = counts.get(status, counts.get(status.lower(), 0))
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value or 0)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _status_counts(payload: Optional[Dict[str, Any]]) -> Optional[tuple]:
    if not isinstance(payload, dict):
        return None
    total = payload.get("total_cells")
    counts = payload.get("status_counts")
    if counts is None:
        counts = payload.get("counts")
    if not isinstance(total, int) or total < 0 or not isinstance(counts, dict):
        return None
    return total, counts


def _status_counts_are_terminal(payload: Optional[Dict[str, Any]]) -> bool:
    status_counts = _status_counts(payload)
    if status_counts is None:
        return False
    total, counts = status_counts
    if total == 0:
        return True
    parsed = {status: _status_count(counts, status) for status in (*_ACTIVE_CELL_STATUSES, "COMPLETED", "FAILED")}
    if any(value is None for value in parsed.values()):
        return False
    if any(parsed[status] > 0 for status in _ACTIVE_CELL_STATUSES):  # type: ignore[operator]
        return False
    return parsed["COMPLETED"] + parsed["FAILED"] == total  # type: ignore[operator]


def _report_cell_progress(
    payload: Optional[Dict[str, Any]],
    callback: Optional[Callable[[int, int, int], None]],
) -> None:
    if callback is None:
        return
    status_counts = _status_counts(payload)
    if status_counts is None:
        return
    total, counts = status_counts
    completed = _status_count(counts, "COMPLETED")
    failed = _status_count(counts, "FAILED")
    if completed is None or failed is None:
        return
    callback(min(completed + failed, total), total, failed)


def _wait_for_sheet_cells(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: ResourceId,
    sheet_id: ResourceId,
    *,
    timeout_seconds: float = _DEFAULT_CELL_WAIT_TIMEOUT_SECONDS,
    poll_interval_seconds: float = _DEFAULT_POLL_INTERVAL_SECONDS,
    on_progress: Optional[Callable[[int, int, int], None]] = None,
) -> Dict[str, Any]:
    return (
        _poll_until(
            fetch=lambda: tables_api.get_sheet_status_counts(api_key, base_url, throw_on_error, table_id, sheet_id),
            is_done=_status_counts_are_terminal,
            timeout_seconds=timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
            timeout_message="Timed out waiting for eval cells on the experiment sheet to finish.",
            backoff=True,
            on_update=lambda payload: _report_cell_progress(payload, on_progress),
        )
        or {}
    )


async def _await_for_sheet_cells(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: ResourceId,
    sheet_id: ResourceId,
    *,
    timeout_seconds: float = _DEFAULT_CELL_WAIT_TIMEOUT_SECONDS,
    poll_interval_seconds: float = _DEFAULT_POLL_INTERVAL_SECONDS,
    on_progress: Optional[Callable[[int, int, int], None]] = None,
) -> Dict[str, Any]:
    return (
        await _apoll_until(
            fetch=lambda: tables_api.aget_sheet_status_counts(api_key, base_url, throw_on_error, table_id, sheet_id),
            is_done=_status_counts_are_terminal,
            timeout_seconds=timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
            timeout_message="Timed out waiting for eval cells on the experiment sheet to finish.",
            backoff=True,
            on_update=lambda payload: _report_cell_progress(payload, on_progress),
        )
        or {}
    )


def _non_negative_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _normalize_operation_status_payload(payload: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Unwrap GET /operations/:id which returns ``{"success": true, "operation": {...}}``.

    CREATE responses keep a string ``operation`` field (``"recalculate"``), so leave those
    payloads unchanged.
    """
    if not isinstance(payload, dict):
        return payload
    nested = payload.get("operation")
    if isinstance(nested, dict):
        return nested
    return payload


def _operation_is_terminal(payload: Optional[Dict[str, Any]]) -> bool:
    payload = _normalize_operation_status_payload(payload)
    if not isinstance(payload, dict):
        return False
    status = payload.get("status")
    if isinstance(status, str) and status.lower() in _TERMINAL_OPERATION_STATUSES:
        return True
    # Fallback when Redis status lags behind finished cells.
    pending = _non_negative_int(payload.get("pending_count"))
    completed = _non_negative_int(payload.get("completed_count"))
    failed = _non_negative_int(payload.get("failed_count"))
    cell_count = _non_negative_int(payload.get("cell_count"))
    if pending is None or completed is None or failed is None or cell_count is None:
        return False
    if pending > 0:
        return False
    return completed + failed >= cell_count


def _operation_ids_from_create_response(payload: Optional[Dict[str, Any]]) -> List[str]:
    if not isinstance(payload, dict):
        return []
    ids: List[str] = []
    execution_ids = payload.get("execution_ids")
    if isinstance(execution_ids, list):
        for item in execution_ids:
            if item is not None and str(item).strip():
                ids.append(str(item))
    for key in ("operation_id", "execution_id"):
        value = payload.get(key)
        if value is not None and str(value).strip():
            ids.append(str(value))
    return list(dict.fromkeys(ids))


def _report_operation_cell_progress(
    payload: Optional[Dict[str, Any]],
    on_progress: Optional[Callable[[int, int, int, Optional[str]], None]] = None,
) -> None:
    if on_progress is None:
        return
    payload = _normalize_operation_status_payload(payload)
    if not isinstance(payload, dict):
        return
    counts = payload.get("status_counts")
    if counts is None:
        counts = payload.get("counts")
    completed = _non_negative_int(payload.get("completed_count"))
    failed = _non_negative_int(payload.get("failed_count"))
    total = _non_negative_int(payload.get("cell_count"))
    if isinstance(counts, dict):
        completed_from_counts = _status_count(counts, "COMPLETED")
        failed_from_counts = _status_count(counts, "FAILED")
        if completed is None and completed_from_counts is not None:
            completed = completed_from_counts
        if failed is None and failed_from_counts is not None:
            failed = failed_from_counts
        if total is None:
            values = [_non_negative_int(value) for value in counts.values()]
            present = [value for value in values if value is not None]
            if present:
                total = sum(present)
    pending = _non_negative_int(payload.get("pending_count"))
    if total is None and completed is not None and failed is not None and pending is not None:
        total = completed + failed + pending
    status = payload.get("status") if isinstance(payload.get("status"), str) and payload.get("status").strip() else None
    if total is None and completed is None and not status:
        return
    on_progress((completed or 0) + (failed or 0), total or 0, failed or 0, status)


def _report_terminal_operation_progress(
    completed: int,
    total: int,
    failed: int = 0,
    status: Optional[str] = None,
) -> None:
    get_terminal().cell_progress(completed, total, failed, status=status)


def wait_for_sheet_operations(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: ResourceId,
    sheet_id: ResourceId,
    *,
    column_ids: List[str],
    row_ids: Optional[List[int]] = None,
    timeout_seconds: float = _DEFAULT_CELL_WAIT_TIMEOUT_SECONDS,
    poll_interval_seconds: float = _DEFAULT_POLL_INTERVAL_SECONDS,
) -> Optional[Dict[str, Any]]:
    """Start a scoped recalculate operation and poll until each operation finishes."""
    if not column_ids:
        return None
    create_response = tables_api.create_sheet_operation(
        api_key,
        base_url,
        throw_on_error,
        table_id,
        sheet_id,
        {
            "operation": "recalculate",
            "column_ids": column_ids,
            "row_ids": row_ids,
        },
    )
    _report_operation_cell_progress(create_response, _report_terminal_operation_progress)
    operation_ids = _operation_ids_from_create_response(create_response if isinstance(create_response, dict) else None)
    if not operation_ids:
        if isinstance(create_response, dict):
            cell_count = _non_negative_int(create_response.get("cell_count")) or 0
            get_terminal().cell_progress(cell_count, cell_count, 0, status="completed")
        return create_response if isinstance(create_response, dict) else None

    last: Optional[Dict[str, Any]] = None
    for operation_id in operation_ids:
        last = _poll_until(
            fetch=lambda operation_id=operation_id: _normalize_operation_status_payload(
                tables_api.get_sheet_operation(api_key, base_url, throw_on_error, table_id, sheet_id, operation_id)
            ),
            is_done=_operation_is_terminal,
            timeout_seconds=timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
            timeout_message="Timed out waiting for supporting column computation to finish.",
            backoff=True,
            on_update=lambda payload: _report_operation_cell_progress(payload, _report_terminal_operation_progress),
        )
        status = str((last or {}).get("status") or "").lower()
        if status == "failed":
            raise api_error(f"Supporting column operation {operation_id} failed while computing preprocessing columns.")
        if status == "cancelled":
            raise api_error(
                f"Supporting column operation {operation_id} was cancelled while computing preprocessing columns."
            )
    return last


async def await_for_sheet_operations(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: ResourceId,
    sheet_id: ResourceId,
    *,
    column_ids: List[str],
    row_ids: Optional[List[int]] = None,
    timeout_seconds: float = _DEFAULT_CELL_WAIT_TIMEOUT_SECONDS,
    poll_interval_seconds: float = _DEFAULT_POLL_INTERVAL_SECONDS,
) -> Optional[Dict[str, Any]]:
    if not column_ids:
        return None
    create_response = await tables_api.acreate_sheet_operation(
        api_key,
        base_url,
        throw_on_error,
        table_id,
        sheet_id,
        {
            "operation": "recalculate",
            "column_ids": column_ids,
            "row_ids": row_ids,
        },
    )
    _report_operation_cell_progress(create_response, _report_terminal_operation_progress)
    operation_ids = _operation_ids_from_create_response(create_response if isinstance(create_response, dict) else None)
    if not operation_ids:
        if isinstance(create_response, dict):
            cell_count = _non_negative_int(create_response.get("cell_count")) or 0
            get_terminal().cell_progress(cell_count, cell_count, 0, status="completed")
        return create_response if isinstance(create_response, dict) else None

    last: Optional[Dict[str, Any]] = None
    for operation_id in operation_ids:

        async def _fetch(operation_id: str = operation_id):
            return _normalize_operation_status_payload(
                await tables_api.aget_sheet_operation(
                    api_key, base_url, throw_on_error, table_id, sheet_id, operation_id
                )
            )

        last = await _apoll_until(
            fetch=_fetch,
            is_done=_operation_is_terminal,
            timeout_seconds=timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
            timeout_message="Timed out waiting for supporting column computation to finish.",
            backoff=True,
            on_update=lambda payload: _report_operation_cell_progress(payload, _report_terminal_operation_progress),
        )
        status = str((last or {}).get("status") or "").lower()
        if status == "failed":
            raise api_error(f"Supporting column operation {operation_id} failed while computing preprocessing columns.")
        if status == "cancelled":
            raise api_error(
                f"Supporting column operation {operation_id} was cancelled while computing preprocessing columns."
            )
    return last
