from contextlib import asynccontextmanager
from contextvars import ContextVar
from functools import wraps
from typing import Any, AsyncIterator, Dict, Optional, Sequence, Union
from urllib.parse import quote

import httpx
import requests

from promptlayer import exceptions as _exceptions
from promptlayer.tables.helpers import (
    build_add_rows_body,
    build_add_trace_body,
    build_batch_recalculate_body,
    build_configure_score_body,
    build_create_column_body,
    build_create_sheet_body,
    build_create_table_body,
    build_create_version_body,
    build_delete_rows_body,
    build_list_tables_params,
    build_update_cell_body,
    build_update_column_body,
    build_update_sheet_body,
    build_update_table_body,
    extract_list,
    extract_sheets,
    extract_tables,
    with_default_empty_sheet_source,
)
from promptlayer.types.table import (
    AddTableRows,
    AddTraceImport,
    BatchRecalculateCells,
    ConfigureSheetScore,
    CreateColumn,
    CreateSheet,
    CreateSheetOperation,
    CreateSheetVersion,
    CreateTable,
    DeleteSheetRows,
    ListTablesParams,
    ResourceId,
    CellResponse,
    ColumnListResponse,
    ColumnResponse,
    SheetListResponse,
    SheetResponse,
    SheetStatusCountsResponse,
    SheetVersionListResponse,
    SheetVersionResponse,
    TableListResponse,
    TableResponse,
    TableScoreResponse,
    UpdateCell,
    UpdateColumn,
    UpdateSheet,
    UpdateTable,
)
from promptlayer.utils import (
    _get_requests_session,
    _make_httpx_client,
    logger,
    raise_on_bad_response,
    retry_on_api_error,
    warn_on_bad_response,
)

_active_async_client: ContextVar[Optional[httpx.AsyncClient]] = ContextVar(
    "promptlayer_tables_async_client",
    default=None,
)


@asynccontextmanager
async def async_client_session() -> AsyncIterator[httpx.AsyncClient]:
    existing = _active_async_client.get()
    if existing is not None:
        yield existing
        return
    async with _make_httpx_client() as client:
        token = _active_async_client.set(client)
        try:
            yield client
        finally:
            _active_async_client.reset(token)


def reuse_async_client(func):
    @wraps(func)
    async def wrapped(*args, **kwargs):
        async with async_client_session():
            return await func(*args, **kwargs)

    return wrapped


def _tables_endpoint(base_url: str) -> str:
    return f"{base_url}/api/public/v2/tables"


def _table_endpoint(base_url: str, table_id: ResourceId) -> str:
    return f"{_tables_endpoint(base_url)}/{quote(str(table_id), safe='')}"


def _table_sheets_endpoint(base_url: str, table_id: ResourceId) -> str:
    return f"{_table_endpoint(base_url, table_id)}/sheets"


def _table_sheet_endpoint(base_url: str, table_id: ResourceId, sheet_id: ResourceId) -> str:
    return f"{_table_sheets_endpoint(base_url, table_id)}/{quote(str(sheet_id), safe='')}"


def _table_sheet_rows_endpoint(base_url: str, table_id: ResourceId, sheet_id: ResourceId) -> str:
    return f"{_table_sheet_endpoint(base_url, table_id, sheet_id)}/rows"


def _table_sheet_columns_endpoint(base_url: str, table_id: ResourceId, sheet_id: ResourceId) -> str:
    return f"{_table_sheet_endpoint(base_url, table_id, sheet_id)}/columns"


def _table_sheet_column_endpoint(
    base_url: str, table_id: ResourceId, sheet_id: ResourceId, column_id: ResourceId
) -> str:
    return f"{_table_sheet_columns_endpoint(base_url, table_id, sheet_id)}/{quote(str(column_id), safe='')}"


def _table_sheet_cell_endpoint(base_url: str, table_id: ResourceId, sheet_id: ResourceId, cell_id: ResourceId) -> str:
    return f"{_table_sheet_endpoint(base_url, table_id, sheet_id)}/cells/{quote(str(cell_id), safe='')}"


def _table_sheet_cells_recalculations_endpoint(base_url: str, table_id: ResourceId, sheet_id: ResourceId) -> str:
    return f"{_table_sheet_endpoint(base_url, table_id, sheet_id)}/cells/recalculations"


def _table_sheet_cell_recalculation_endpoint(
    base_url: str, table_id: ResourceId, sheet_id: ResourceId, cell_id: ResourceId
) -> str:
    return f"{_table_sheet_cell_endpoint(base_url, table_id, sheet_id, cell_id)}/recalculations"


def _table_sheet_versions_endpoint(base_url: str, table_id: ResourceId, sheet_id: ResourceId) -> str:
    return f"{_table_sheet_endpoint(base_url, table_id, sheet_id)}/versions"


def _table_sheet_version_endpoint(
    base_url: str, table_id: ResourceId, sheet_id: ResourceId, version_id: ResourceId
) -> str:
    return f"{_table_sheet_versions_endpoint(base_url, table_id, sheet_id)}/{quote(str(version_id), safe='')}"


def _table_sheet_score_history_endpoint(base_url: str, table_id: ResourceId, sheet_id: ResourceId) -> str:
    return f"{_table_sheet_versions_endpoint(base_url, table_id, sheet_id)}/score-history"


def _table_sheet_score_endpoint(base_url: str, table_id: ResourceId, sheet_id: ResourceId) -> str:
    return f"{_table_sheet_endpoint(base_url, table_id, sheet_id)}/score"


def _table_sheet_scorecard_endpoint(
    base_url: str,
    table_id: ResourceId,
    sheet_id: ResourceId,
    *parts: Any,
) -> str:
    base = f"{_table_sheet_endpoint(base_url, table_id, sheet_id)}/scorecard"
    if not parts:
        return base
    suffix = "/".join(quote(str(part), safe="") for part in parts if part is not None)
    return f"{base}/{suffix}"


def _table_sheet_status_counts_endpoint(base_url: str, table_id: ResourceId, sheet_id: ResourceId) -> str:
    return f"{_table_sheet_endpoint(base_url, table_id, sheet_id)}/status-counts"


def _table_sheet_operations_endpoint(base_url: str, table_id: ResourceId, sheet_id: ResourceId) -> str:
    return f"{_table_sheet_endpoint(base_url, table_id, sheet_id)}/operations"


def _table_sheet_operation_endpoint(
    base_url: str, table_id: ResourceId, sheet_id: ResourceId, operation_id: ResourceId
) -> str:
    return f"{_table_sheet_operations_endpoint(base_url, table_id, sheet_id)}/{quote(str(operation_id), safe='')}"


def _add_trace_endpoint(base_url: str) -> str:
    return f"{base_url}/api/public/v2/dataset-versions/add-trace"


def _headers(api_key: str) -> Dict[str, str]:
    return {"X-API-KEY": api_key}


def _json_headers(api_key: str) -> Dict[str, str]:
    return {"X-API-KEY": api_key, "Content-Type": "application/json"}


def _handle_request_exception(
    *,
    throw_on_error: bool,
    message: str,
    exception: Exception,
) -> None:
    if throw_on_error:
        raise _exceptions.PromptLayerAPIConnectionError(message, response=None, body=None) from exception
    logger.warning(message)


def _handle_bad_response(
    *,
    response: Any,
    throw_on_error: bool,
    message: str,
    warning_prefix: str,
) -> None:
    if throw_on_error:
        raise_on_bad_response(response, message)
    warn_on_bad_response(response, warning_prefix)


def _error_messages(action: str) -> tuple:
    message = f"PromptLayer had the following error while {action}"
    return message, f"WARNING: {message}"


def _parse_response(
    response: Any,
    *,
    throw_on_error: bool,
    expected_statuses: Sequence[int],
    action: str,
    empty_ok: bool = False,
    empty_value: Any = None,
) -> Any:
    if response.status_code not in expected_statuses:
        message, warning_prefix = _error_messages(action)
        _handle_bad_response(
            response=response,
            throw_on_error=throw_on_error,
            message=message,
            warning_prefix=warning_prefix,
        )
        return empty_value
    if empty_ok:
        # Boolean deletes always succeed as True; row deletes may return JSON or 204.
        if empty_value is True:
            return True
        if response.status_code == 204 or not getattr(response, "content", None):
            return empty_value if empty_value is not None else {"success": True}
    return response.json()


def _request(
    *,
    method: str,
    url: str,
    api_key: str,
    throw_on_error: bool,
    action: str,
    expected_statuses: Sequence[int] = (200,),
    params: Optional[Dict[str, Any]] = None,
    json_body: Optional[Dict[str, Any]] = None,
    use_json_headers: bool = False,
    empty_ok: bool = False,
    empty_value: Any = None,
) -> Any:
    headers = _json_headers(api_key) if use_json_headers else _headers(api_key)
    try:
        session = _get_requests_session()
        kwargs: Dict[str, Any] = {"headers": headers}
        if params is not None:
            kwargs["params"] = params
        if json_body is not None:
            kwargs["json"] = json_body
        # Prefer method-specific calls (get/post/...) so existing test mocks keep working.
        request_fn = getattr(session, method.lower(), None)
        if request_fn is not None:
            response = request_fn(url, **kwargs)
        else:
            response = session.request(method, url, **kwargs)
        return _parse_response(
            response,
            throw_on_error=throw_on_error,
            expected_statuses=expected_statuses,
            action=action,
            empty_ok=empty_ok,
            empty_value=empty_value,
        )
    except requests.exceptions.RequestException as e:
        _handle_request_exception(
            throw_on_error=throw_on_error,
            message=f"PromptLayer had the following error while {action}: {e}",
            exception=e,
        )
        return empty_value


async def _arequest(
    *,
    method: str,
    url: str,
    api_key: str,
    throw_on_error: bool,
    action: str,
    expected_statuses: Sequence[int] = (200,),
    params: Optional[Dict[str, Any]] = None,
    json_body: Optional[Dict[str, Any]] = None,
    use_json_headers: bool = False,
    empty_ok: bool = False,
    empty_value: Any = None,
) -> Any:
    headers = _json_headers(api_key) if use_json_headers else _headers(api_key)
    try:
        async with async_client_session() as client:
            kwargs: Dict[str, Any] = {"headers": headers}
            if params is not None:
                kwargs["params"] = params
            if json_body is not None:
                kwargs["json"] = json_body
            request_fn = getattr(client, method.lower(), None)
            if request_fn is not None:
                response = await request_fn(url, **kwargs)
            else:
                response = await client.request(method, url, **kwargs)
        return _parse_response(
            response,
            throw_on_error=throw_on_error,
            expected_statuses=expected_statuses,
            action=action,
            empty_ok=empty_ok,
            empty_value=empty_value,
        )
    except httpx.RequestError as e:
        _handle_request_exception(
            throw_on_error=throw_on_error,
            message=f"PromptLayer had the following error while {action}: {e}",
            exception=e,
        )
        return empty_value


@retry_on_api_error
def list_tables(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    params: Optional[ListTablesParams] = None,
) -> Union[TableListResponse, None]:
    return _request(
        method="GET",
        url=_tables_endpoint(base_url),
        api_key=api_key,
        throw_on_error=throw_on_error,
        action="listing your tables",
        params=build_list_tables_params(params),
    )


@retry_on_api_error
async def alist_tables(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    params: Optional[ListTablesParams] = None,
) -> Union[TableListResponse, None]:
    return await _arequest(
        method="GET",
        url=_tables_endpoint(base_url),
        api_key=api_key,
        throw_on_error=throw_on_error,
        action="listing your tables",
        params=build_list_tables_params(params),
    )


@retry_on_api_error
def create_table(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    body: CreateTable,
) -> Union[TableResponse, None]:
    return _request(
        method="POST",
        url=_tables_endpoint(base_url),
        api_key=api_key,
        throw_on_error=throw_on_error,
        action="creating your table",
        expected_statuses=(200, 201),
        json_body=build_create_table_body(body),
        use_json_headers=True,
    )


@retry_on_api_error
async def acreate_table(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    body: CreateTable,
) -> Union[TableResponse, None]:
    return await _arequest(
        method="POST",
        url=_tables_endpoint(base_url),
        api_key=api_key,
        throw_on_error=throw_on_error,
        action="creating your table",
        expected_statuses=(200, 201),
        json_body=build_create_table_body(body),
        use_json_headers=True,
    )


@retry_on_api_error
def get_table(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: ResourceId,
) -> Union[TableResponse, None]:
    return _request(
        method="GET",
        url=_table_endpoint(base_url, table_id),
        api_key=api_key,
        throw_on_error=throw_on_error,
        action="getting your table",
    )


@retry_on_api_error
async def aget_table(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: ResourceId,
) -> Union[TableResponse, None]:
    return await _arequest(
        method="GET",
        url=_table_endpoint(base_url, table_id),
        api_key=api_key,
        throw_on_error=throw_on_error,
        action="getting your table",
    )


@retry_on_api_error
def update_table(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: ResourceId,
    body: UpdateTable,
) -> Union[TableResponse, None]:
    return _request(
        method="PATCH",
        url=_table_endpoint(base_url, table_id),
        api_key=api_key,
        throw_on_error=throw_on_error,
        action="updating your table",
        json_body=build_update_table_body(body),
        use_json_headers=True,
    )


@retry_on_api_error
async def aupdate_table(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: ResourceId,
    body: UpdateTable,
) -> Union[TableResponse, None]:
    return await _arequest(
        method="PATCH",
        url=_table_endpoint(base_url, table_id),
        api_key=api_key,
        throw_on_error=throw_on_error,
        action="updating your table",
        json_body=build_update_table_body(body),
        use_json_headers=True,
    )


@retry_on_api_error
def delete_table(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: ResourceId,
) -> bool:
    result = _request(
        method="DELETE",
        url=_table_endpoint(base_url, table_id),
        api_key=api_key,
        throw_on_error=throw_on_error,
        action="deleting your table",
        expected_statuses=(200, 204),
        empty_ok=True,
        empty_value=True,
    )
    return bool(result)


@retry_on_api_error
async def adelete_table(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: ResourceId,
) -> bool:
    result = await _arequest(
        method="DELETE",
        url=_table_endpoint(base_url, table_id),
        api_key=api_key,
        throw_on_error=throw_on_error,
        action="deleting your table",
        expected_statuses=(200, 204),
        empty_ok=True,
        empty_value=True,
    )
    return bool(result)


def upsert_table_by_title(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    title: str,
    folder_id: Optional[int] = None,
):
    list_response = list_tables(
        api_key,
        base_url,
        throw_on_error,
        {"name": title, "folder_id": folder_id, "limit": 100},
    )
    if not list_response:
        return None

    for table in extract_tables(list_response):
        if table.get("title") == title and not table.get("deleted_at"):
            return table

    create_response = create_table(
        api_key,
        base_url,
        throw_on_error,
        {"title": title, "folder_id": folder_id},
    )
    if not create_response or not create_response.get("table"):
        return None
    return create_response["table"]


async def aupsert_table_by_title(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    title: str,
    folder_id: Optional[int] = None,
):
    list_response = await alist_tables(
        api_key,
        base_url,
        throw_on_error,
        {"name": title, "folder_id": folder_id, "limit": 100},
    )
    if not list_response:
        return None

    for table in extract_tables(list_response):
        if table.get("title") == title and not table.get("deleted_at"):
            return table

    create_response = await acreate_table(
        api_key,
        base_url,
        throw_on_error,
        {"title": title, "folder_id": folder_id},
    )
    if not create_response or not create_response.get("table"):
        return None
    return create_response["table"]


@retry_on_api_error
def list_sheets(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: ResourceId,
) -> Union[SheetListResponse, None]:
    return _request(
        method="GET",
        url=_table_sheets_endpoint(base_url, table_id),
        api_key=api_key,
        throw_on_error=throw_on_error,
        action="listing your sheets",
    )


@retry_on_api_error
async def alist_sheets(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: ResourceId,
) -> Union[SheetListResponse, None]:
    return await _arequest(
        method="GET",
        url=_table_sheets_endpoint(base_url, table_id),
        api_key=api_key,
        throw_on_error=throw_on_error,
        action="listing your sheets",
    )


def ensure_default_sheet(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: ResourceId,
):
    list_response = list_sheets(api_key, base_url, throw_on_error, table_id)
    if not list_response:
        return None

    sheets = extract_sheets(list_response)
    if sheets:
        return sheets[0]

    create_response = create_sheet(
        api_key,
        base_url,
        throw_on_error,
        table_id,
        with_default_empty_sheet_source({"title": "Sheet 1"}),
    )
    if not create_response or not create_response.get("sheet"):
        return None
    return create_response["sheet"]


async def aensure_default_sheet(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: ResourceId,
):
    list_response = await alist_sheets(api_key, base_url, throw_on_error, table_id)
    if not list_response:
        return None

    sheets = extract_sheets(list_response)
    if sheets:
        return sheets[0]

    create_response = await acreate_sheet(
        api_key,
        base_url,
        throw_on_error,
        table_id,
        with_default_empty_sheet_source({"title": "Sheet 1"}),
    )
    if not create_response or not create_response.get("sheet"):
        return None
    return create_response["sheet"]


@retry_on_api_error
def create_sheet(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: ResourceId,
    body: CreateSheet,
) -> Union[SheetResponse, None]:
    return _request(
        method="POST",
        url=_table_sheets_endpoint(base_url, table_id),
        api_key=api_key,
        throw_on_error=throw_on_error,
        action="creating your sheet",
        # Create-with-import returns 202 with the new sheet + operation.
        expected_statuses=(200, 201, 202),
        json_body=build_create_sheet_body(body),
        use_json_headers=True,
    )


@retry_on_api_error
async def acreate_sheet(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: ResourceId,
    body: CreateSheet,
) -> Union[SheetResponse, None]:
    return await _arequest(
        method="POST",
        url=_table_sheets_endpoint(base_url, table_id),
        api_key=api_key,
        throw_on_error=throw_on_error,
        action="creating your sheet",
        expected_statuses=(200, 201, 202),
        json_body=build_create_sheet_body(body),
        use_json_headers=True,
    )


@retry_on_api_error
def get_sheet(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: ResourceId,
    sheet_id: ResourceId,
) -> Union[SheetResponse, None]:
    return _request(
        method="GET",
        url=_table_sheet_endpoint(base_url, table_id, sheet_id),
        api_key=api_key,
        throw_on_error=throw_on_error,
        action="getting your sheet",
    )


@retry_on_api_error
async def aget_sheet(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: ResourceId,
    sheet_id: ResourceId,
) -> Union[SheetResponse, None]:
    return await _arequest(
        method="GET",
        url=_table_sheet_endpoint(base_url, table_id, sheet_id),
        api_key=api_key,
        throw_on_error=throw_on_error,
        action="getting your sheet",
    )


@retry_on_api_error
def update_sheet(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: ResourceId,
    sheet_id: ResourceId,
    body: UpdateSheet,
) -> Union[SheetResponse, None]:
    return _request(
        method="PATCH",
        url=_table_sheet_endpoint(base_url, table_id, sheet_id),
        api_key=api_key,
        throw_on_error=throw_on_error,
        action="updating your sheet",
        json_body=build_update_sheet_body(body),
        use_json_headers=True,
    )


@retry_on_api_error
async def aupdate_sheet(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: ResourceId,
    sheet_id: ResourceId,
    body: UpdateSheet,
) -> Union[SheetResponse, None]:
    return await _arequest(
        method="PATCH",
        url=_table_sheet_endpoint(base_url, table_id, sheet_id),
        api_key=api_key,
        throw_on_error=throw_on_error,
        action="updating your sheet",
        json_body=build_update_sheet_body(body),
        use_json_headers=True,
    )


@retry_on_api_error
def delete_sheet(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: ResourceId,
    sheet_id: ResourceId,
) -> bool:
    result = _request(
        method="DELETE",
        url=_table_sheet_endpoint(base_url, table_id, sheet_id),
        api_key=api_key,
        throw_on_error=throw_on_error,
        action="deleting your sheet",
        expected_statuses=(200, 204),
        empty_ok=True,
        empty_value=True,
    )
    return bool(result)


@retry_on_api_error
async def adelete_sheet(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: ResourceId,
    sheet_id: ResourceId,
) -> bool:
    result = await _arequest(
        method="DELETE",
        url=_table_sheet_endpoint(base_url, table_id, sheet_id),
        api_key=api_key,
        throw_on_error=throw_on_error,
        action="deleting your sheet",
        expected_statuses=(200, 204),
        empty_ok=True,
        empty_value=True,
    )
    return bool(result)


@retry_on_api_error
def list_smart_sheet_rows(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: ResourceId,
    sheet_id: ResourceId,
    params: Optional[Dict[str, Any]] = None,
) -> Union[Dict[str, Any], None]:
    return _request(
        method="GET",
        url=_table_sheet_rows_endpoint(base_url, table_id, sheet_id),
        api_key=api_key,
        throw_on_error=throw_on_error,
        action="listing your sheet rows",
        params=params or {},
    )


@retry_on_api_error
async def alist_smart_sheet_rows(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: ResourceId,
    sheet_id: ResourceId,
    params: Optional[Dict[str, Any]] = None,
) -> Union[Dict[str, Any], None]:
    return await _arequest(
        method="GET",
        url=_table_sheet_rows_endpoint(base_url, table_id, sheet_id),
        api_key=api_key,
        throw_on_error=throw_on_error,
        action="listing your sheet rows",
        params=params or {},
    )


def _rows_next_cursor(payload: Optional[Dict[str, Any]]) -> Optional[str]:
    if not payload:
        return None
    pagination = payload.get("pagination")
    if not isinstance(pagination, dict):
        return None
    cursor = pagination.get("next_cursor")
    return str(cursor) if cursor else None


def _rows_from_payload(payload: Optional[Dict[str, Any]]) -> list:
    return extract_list(payload, "rows")


def _row_list_query(params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    query = dict(params or {})
    query["limit"] = max(1, min(int(query.get("limit") or 100), 100))
    query.pop("cursor", None)
    return query


def _handle_repeated_cursor(cursor: str, seen_cursors: set, *, throw_on_error: bool) -> bool:
    """Return True when pagination should stop due to a repeated cursor."""
    if cursor not in seen_cursors:
        seen_cursors.add(cursor)
        return False
    message = "Table row pagination returned a repeated cursor."
    if throw_on_error:
        raise _exceptions.PromptLayerAPIError(message, response=None, body=None)
    logger.warning(message)
    return True


def list_all_smart_sheet_rows(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: ResourceId,
    sheet_id: ResourceId,
    params: Optional[Dict[str, Any]] = None,
) -> Union[Dict[str, Any], None]:
    query = _row_list_query(params)
    first = list_smart_sheet_rows(api_key, base_url, throw_on_error, table_id, sheet_id, params=query)
    if first is None:
        return None

    merged = dict(first)
    rows = _rows_from_payload(first)
    cursor = _rows_next_cursor(first)
    seen_cursors = set()
    while cursor:
        if _handle_repeated_cursor(cursor, seen_cursors, throw_on_error=throw_on_error):
            break
        page_query = dict(query)
        page_query["cursor"] = cursor
        page_query["include_columns"] = False
        page = list_smart_sheet_rows(api_key, base_url, throw_on_error, table_id, sheet_id, params=page_query)
        if page is None:
            break
        rows.extend(_rows_from_payload(page))
        merged["pagination"] = page.get("pagination")
        cursor = _rows_next_cursor(page)
    merged["data"] = rows
    return merged


async def alist_all_smart_sheet_rows(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: ResourceId,
    sheet_id: ResourceId,
    params: Optional[Dict[str, Any]] = None,
) -> Union[Dict[str, Any], None]:
    query = _row_list_query(params)
    first = await alist_smart_sheet_rows(api_key, base_url, throw_on_error, table_id, sheet_id, params=query)
    if first is None:
        return None

    merged = dict(first)
    rows = _rows_from_payload(first)
    cursor = _rows_next_cursor(first)
    seen_cursors = set()
    while cursor:
        if _handle_repeated_cursor(cursor, seen_cursors, throw_on_error=throw_on_error):
            break
        page_query = dict(query)
        page_query["cursor"] = cursor
        page_query["include_columns"] = False
        page = await alist_smart_sheet_rows(
            api_key,
            base_url,
            throw_on_error,
            table_id,
            sheet_id,
            params=page_query,
        )
        if page is None:
            break
        rows.extend(_rows_from_payload(page))
        merged["pagination"] = page.get("pagination")
        cursor = _rows_next_cursor(page)
    merged["data"] = rows
    return merged


@retry_on_api_error
def add_smart_sheet_rows(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: ResourceId,
    sheet_id: ResourceId,
    body: AddTableRows,
) -> Union[Dict[str, Any], None]:
    return _request(
        method="POST",
        url=_table_sheet_rows_endpoint(base_url, table_id, sheet_id),
        api_key=api_key,
        throw_on_error=throw_on_error,
        action="adding your sheet rows",
        expected_statuses=(200, 201),
        json_body=build_add_rows_body(body),
        use_json_headers=True,
    )


@retry_on_api_error
async def aadd_smart_sheet_rows(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: ResourceId,
    sheet_id: ResourceId,
    body: AddTableRows,
) -> Union[Dict[str, Any], None]:
    return await _arequest(
        method="POST",
        url=_table_sheet_rows_endpoint(base_url, table_id, sheet_id),
        api_key=api_key,
        throw_on_error=throw_on_error,
        action="adding your sheet rows",
        expected_statuses=(200, 201),
        json_body=build_add_rows_body(body),
        use_json_headers=True,
    )


@retry_on_api_error
def delete_sheet_rows(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: ResourceId,
    sheet_id: ResourceId,
    body: DeleteSheetRows,
) -> Union[Dict[str, Any], None]:
    return _request(
        method="DELETE",
        url=_table_sheet_rows_endpoint(base_url, table_id, sheet_id),
        api_key=api_key,
        throw_on_error=throw_on_error,
        action="deleting your sheet rows",
        expected_statuses=(200, 204),
        json_body=build_delete_rows_body(body),
        use_json_headers=True,
        empty_ok=True,
        empty_value={"success": True},
    )


@retry_on_api_error
async def adelete_sheet_rows(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: ResourceId,
    sheet_id: ResourceId,
    body: DeleteSheetRows,
) -> Union[Dict[str, Any], None]:
    return await _arequest(
        method="DELETE",
        url=_table_sheet_rows_endpoint(base_url, table_id, sheet_id),
        api_key=api_key,
        throw_on_error=throw_on_error,
        action="deleting your sheet rows",
        expected_statuses=(200, 204),
        json_body=build_delete_rows_body(body),
        use_json_headers=True,
        empty_ok=True,
        empty_value={"success": True},
    )


@retry_on_api_error
def list_smart_sheet_columns(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: ResourceId,
    sheet_id: ResourceId,
) -> Union[ColumnListResponse, None]:
    return _request(
        method="GET",
        url=_table_sheet_columns_endpoint(base_url, table_id, sheet_id),
        api_key=api_key,
        throw_on_error=throw_on_error,
        action="listing your sheet columns",
    )


@retry_on_api_error
async def alist_smart_sheet_columns(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: ResourceId,
    sheet_id: ResourceId,
) -> Union[ColumnListResponse, None]:
    return await _arequest(
        method="GET",
        url=_table_sheet_columns_endpoint(base_url, table_id, sheet_id),
        api_key=api_key,
        throw_on_error=throw_on_error,
        action="listing your sheet columns",
    )


@retry_on_api_error
def create_sheet_column(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: ResourceId,
    sheet_id: ResourceId,
    body: CreateColumn,
) -> Union[ColumnResponse, None]:
    return _request(
        method="POST",
        url=_table_sheet_columns_endpoint(base_url, table_id, sheet_id),
        api_key=api_key,
        throw_on_error=throw_on_error,
        action="creating your sheet column",
        expected_statuses=(200, 201),
        json_body=build_create_column_body(body),
        use_json_headers=True,
    )


@retry_on_api_error
async def acreate_sheet_column(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: ResourceId,
    sheet_id: ResourceId,
    body: CreateColumn,
) -> Union[ColumnResponse, None]:
    return await _arequest(
        method="POST",
        url=_table_sheet_columns_endpoint(base_url, table_id, sheet_id),
        api_key=api_key,
        throw_on_error=throw_on_error,
        action="creating your sheet column",
        expected_statuses=(200, 201),
        json_body=build_create_column_body(body),
        use_json_headers=True,
    )


@retry_on_api_error
def update_sheet_column(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: ResourceId,
    sheet_id: ResourceId,
    column_id: ResourceId,
    body: UpdateColumn,
) -> Union[ColumnResponse, None]:
    return _request(
        method="PATCH",
        url=_table_sheet_column_endpoint(base_url, table_id, sheet_id, column_id),
        api_key=api_key,
        throw_on_error=throw_on_error,
        action="updating your sheet column",
        json_body=build_update_column_body(body),
        use_json_headers=True,
    )


@retry_on_api_error
async def aupdate_sheet_column(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: ResourceId,
    sheet_id: ResourceId,
    column_id: ResourceId,
    body: UpdateColumn,
) -> Union[ColumnResponse, None]:
    return await _arequest(
        method="PATCH",
        url=_table_sheet_column_endpoint(base_url, table_id, sheet_id, column_id),
        api_key=api_key,
        throw_on_error=throw_on_error,
        action="updating your sheet column",
        json_body=build_update_column_body(body),
        use_json_headers=True,
    )


@retry_on_api_error
def delete_sheet_column(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: ResourceId,
    sheet_id: ResourceId,
    column_id: ResourceId,
) -> bool:
    result = _request(
        method="DELETE",
        url=_table_sheet_column_endpoint(base_url, table_id, sheet_id, column_id),
        api_key=api_key,
        throw_on_error=throw_on_error,
        action="deleting your sheet column",
        expected_statuses=(200, 204),
        empty_ok=True,
        empty_value=True,
    )
    return bool(result)


@retry_on_api_error
async def adelete_sheet_column(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: ResourceId,
    sheet_id: ResourceId,
    column_id: ResourceId,
) -> bool:
    result = await _arequest(
        method="DELETE",
        url=_table_sheet_column_endpoint(base_url, table_id, sheet_id, column_id),
        api_key=api_key,
        throw_on_error=throw_on_error,
        action="deleting your sheet column",
        expected_statuses=(200, 204),
        empty_ok=True,
        empty_value=True,
    )
    return bool(result)


@retry_on_api_error
def get_sheet_cell(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: ResourceId,
    sheet_id: ResourceId,
    cell_id: ResourceId,
) -> Union[CellResponse, None]:
    return _request(
        method="GET",
        url=_table_sheet_cell_endpoint(base_url, table_id, sheet_id, cell_id),
        api_key=api_key,
        throw_on_error=throw_on_error,
        action="getting your sheet cell",
    )


@retry_on_api_error
async def aget_sheet_cell(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: ResourceId,
    sheet_id: ResourceId,
    cell_id: ResourceId,
) -> Union[CellResponse, None]:
    return await _arequest(
        method="GET",
        url=_table_sheet_cell_endpoint(base_url, table_id, sheet_id, cell_id),
        api_key=api_key,
        throw_on_error=throw_on_error,
        action="getting your sheet cell",
    )


@retry_on_api_error
def update_sheet_cell(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: ResourceId,
    sheet_id: ResourceId,
    cell_id: ResourceId,
    body: UpdateCell,
) -> Union[CellResponse, None]:
    return _request(
        method="PATCH",
        url=_table_sheet_cell_endpoint(base_url, table_id, sheet_id, cell_id),
        api_key=api_key,
        throw_on_error=throw_on_error,
        action="updating your sheet cell",
        json_body=build_update_cell_body(body),
        use_json_headers=True,
    )


@retry_on_api_error
async def aupdate_sheet_cell(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: ResourceId,
    sheet_id: ResourceId,
    cell_id: ResourceId,
    body: UpdateCell,
) -> Union[CellResponse, None]:
    return await _arequest(
        method="PATCH",
        url=_table_sheet_cell_endpoint(base_url, table_id, sheet_id, cell_id),
        api_key=api_key,
        throw_on_error=throw_on_error,
        action="updating your sheet cell",
        json_body=build_update_cell_body(body),
        use_json_headers=True,
    )


@retry_on_api_error
def recalculate_smart_sheet_cell(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: ResourceId,
    sheet_id: ResourceId,
    cell_id: ResourceId,
) -> Union[CellResponse, None]:
    return _request(
        method="POST",
        url=_table_sheet_cell_recalculation_endpoint(base_url, table_id, sheet_id, cell_id),
        api_key=api_key,
        throw_on_error=throw_on_error,
        action="recalculating your sheet cell",
        expected_statuses=(200, 202),
        json_body={},
        use_json_headers=True,
    )


@retry_on_api_error
async def arecalculate_smart_sheet_cell(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: ResourceId,
    sheet_id: ResourceId,
    cell_id: ResourceId,
) -> Union[CellResponse, None]:
    return await _arequest(
        method="POST",
        url=_table_sheet_cell_recalculation_endpoint(base_url, table_id, sheet_id, cell_id),
        api_key=api_key,
        throw_on_error=throw_on_error,
        action="recalculating your sheet cell",
        expected_statuses=(200, 202),
        json_body={},
        use_json_headers=True,
    )


@retry_on_api_error
def batch_recalculate_smart_sheet_cells(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: ResourceId,
    sheet_id: ResourceId,
    body: BatchRecalculateCells,
) -> Union[Dict[str, Any], None]:
    return _request(
        method="POST",
        url=_table_sheet_cells_recalculations_endpoint(base_url, table_id, sheet_id),
        api_key=api_key,
        throw_on_error=throw_on_error,
        action="batch recalculating your sheet cells",
        expected_statuses=(200, 202),
        json_body=build_batch_recalculate_body(body),
        use_json_headers=True,
    )


@retry_on_api_error
async def abatch_recalculate_smart_sheet_cells(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: ResourceId,
    sheet_id: ResourceId,
    body: BatchRecalculateCells,
) -> Union[Dict[str, Any], None]:
    return await _arequest(
        method="POST",
        url=_table_sheet_cells_recalculations_endpoint(base_url, table_id, sheet_id),
        api_key=api_key,
        throw_on_error=throw_on_error,
        action="batch recalculating your sheet cells",
        expected_statuses=(200, 202),
        json_body=build_batch_recalculate_body(body),
        use_json_headers=True,
    )


@retry_on_api_error
def list_sheet_versions(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: ResourceId,
    sheet_id: ResourceId,
) -> Union[SheetVersionListResponse, None]:
    return _request(
        method="GET",
        url=_table_sheet_versions_endpoint(base_url, table_id, sheet_id),
        api_key=api_key,
        throw_on_error=throw_on_error,
        action="listing your sheet versions",
    )


@retry_on_api_error
async def alist_sheet_versions(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: ResourceId,
    sheet_id: ResourceId,
) -> Union[SheetVersionListResponse, None]:
    return await _arequest(
        method="GET",
        url=_table_sheet_versions_endpoint(base_url, table_id, sheet_id),
        api_key=api_key,
        throw_on_error=throw_on_error,
        action="listing your sheet versions",
    )


@retry_on_api_error
def create_sheet_version(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: ResourceId,
    sheet_id: ResourceId,
    body: CreateSheetVersion,
) -> Union[SheetVersionResponse, None]:
    return _request(
        method="POST",
        url=_table_sheet_versions_endpoint(base_url, table_id, sheet_id),
        api_key=api_key,
        throw_on_error=throw_on_error,
        action="creating your sheet version",
        expected_statuses=(200, 201),
        json_body=build_create_version_body(body),
        use_json_headers=True,
    )


@retry_on_api_error
async def acreate_sheet_version(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: ResourceId,
    sheet_id: ResourceId,
    body: CreateSheetVersion,
) -> Union[SheetVersionResponse, None]:
    return await _arequest(
        method="POST",
        url=_table_sheet_versions_endpoint(base_url, table_id, sheet_id),
        api_key=api_key,
        throw_on_error=throw_on_error,
        action="creating your sheet version",
        expected_statuses=(200, 201),
        json_body=build_create_version_body(body),
        use_json_headers=True,
    )


@retry_on_api_error
def get_sheet_version(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: ResourceId,
    sheet_id: ResourceId,
    version_id: ResourceId,
) -> Union[SheetVersionResponse, None]:
    return _request(
        method="GET",
        url=_table_sheet_version_endpoint(base_url, table_id, sheet_id, version_id),
        api_key=api_key,
        throw_on_error=throw_on_error,
        action="getting your sheet version",
    )


@retry_on_api_error
async def aget_sheet_version(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: ResourceId,
    sheet_id: ResourceId,
    version_id: ResourceId,
) -> Union[SheetVersionResponse, None]:
    return await _arequest(
        method="GET",
        url=_table_sheet_version_endpoint(base_url, table_id, sheet_id, version_id),
        api_key=api_key,
        throw_on_error=throw_on_error,
        action="getting your sheet version",
    )


@retry_on_api_error
def get_sheet_score_history(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: ResourceId,
    sheet_id: ResourceId,
) -> Union[Dict[str, Any], None]:
    return _request(
        method="GET",
        url=_table_sheet_score_history_endpoint(base_url, table_id, sheet_id),
        api_key=api_key,
        throw_on_error=throw_on_error,
        action="getting your sheet score history",
    )


@retry_on_api_error
async def aget_sheet_score_history(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: ResourceId,
    sheet_id: ResourceId,
) -> Union[Dict[str, Any], None]:
    return await _arequest(
        method="GET",
        url=_table_sheet_score_history_endpoint(base_url, table_id, sheet_id),
        api_key=api_key,
        throw_on_error=throw_on_error,
        action="getting your sheet score history",
    )


@retry_on_api_error
def get_sheet_status_counts(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: ResourceId,
    sheet_id: ResourceId,
) -> Union[SheetStatusCountsResponse, None]:
    return _request(
        method="GET",
        url=_table_sheet_status_counts_endpoint(base_url, table_id, sheet_id),
        api_key=api_key,
        throw_on_error=throw_on_error,
        action="getting your sheet status counts",
    )


@retry_on_api_error
async def aget_sheet_status_counts(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: ResourceId,
    sheet_id: ResourceId,
) -> Union[SheetStatusCountsResponse, None]:
    return await _arequest(
        method="GET",
        url=_table_sheet_status_counts_endpoint(base_url, table_id, sheet_id),
        api_key=api_key,
        throw_on_error=throw_on_error,
        action="getting your sheet status counts",
    )


def _build_create_operation_body(body: CreateSheetOperation) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"operation": body.get("operation") or "recalculate"}
    if body.get("column_ids") is not None:
        payload["column_ids"] = [str(column_id) for column_id in body["column_ids"]]
    if body.get("row_ids") is not None:
        payload["row_ids"] = list(body["row_ids"])
    if "statuses" in body:
        payload["statuses"] = body.get("statuses")
    return payload


@retry_on_api_error
def create_sheet_operation(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: ResourceId,
    sheet_id: ResourceId,
    body: CreateSheetOperation,
) -> Union[Dict[str, Any], None]:
    return _request(
        method="POST",
        url=_table_sheet_operations_endpoint(base_url, table_id, sheet_id),
        api_key=api_key,
        throw_on_error=throw_on_error,
        action="starting a sheet operation",
        expected_statuses=(200, 202),
        json_body=_build_create_operation_body(body),
        use_json_headers=True,
    )


@retry_on_api_error
async def acreate_sheet_operation(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: ResourceId,
    sheet_id: ResourceId,
    body: CreateSheetOperation,
) -> Union[Dict[str, Any], None]:
    return await _arequest(
        method="POST",
        url=_table_sheet_operations_endpoint(base_url, table_id, sheet_id),
        api_key=api_key,
        throw_on_error=throw_on_error,
        action="starting a sheet operation",
        expected_statuses=(200, 202),
        json_body=_build_create_operation_body(body),
        use_json_headers=True,
    )


@retry_on_api_error
def get_sheet_operation(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: ResourceId,
    sheet_id: ResourceId,
    operation_id: ResourceId,
) -> Union[Dict[str, Any], None]:
    return _request(
        method="GET",
        url=_table_sheet_operation_endpoint(base_url, table_id, sheet_id, operation_id),
        api_key=api_key,
        throw_on_error=throw_on_error,
        action="getting your sheet operation status",
    )


@retry_on_api_error
async def aget_sheet_operation(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: ResourceId,
    sheet_id: ResourceId,
    operation_id: ResourceId,
) -> Union[Dict[str, Any], None]:
    return await _arequest(
        method="GET",
        url=_table_sheet_operation_endpoint(base_url, table_id, sheet_id, operation_id),
        api_key=api_key,
        throw_on_error=throw_on_error,
        action="getting your sheet operation status",
    )


@retry_on_api_error
def get_sheet_score(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: ResourceId,
    sheet_id: ResourceId,
) -> Union[TableScoreResponse, None]:
    return _request(
        method="GET",
        url=_table_sheet_score_endpoint(base_url, table_id, sheet_id),
        api_key=api_key,
        throw_on_error=throw_on_error,
        action="getting your sheet score",
    )


@retry_on_api_error
async def aget_sheet_score(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: ResourceId,
    sheet_id: ResourceId,
) -> Union[TableScoreResponse, None]:
    return await _arequest(
        method="GET",
        url=_table_sheet_score_endpoint(base_url, table_id, sheet_id),
        api_key=api_key,
        throw_on_error=throw_on_error,
        action="getting your sheet score",
    )


@retry_on_api_error
def configure_sheet_score(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: ResourceId,
    sheet_id: ResourceId,
    body: ConfigureSheetScore,
) -> Union[TableScoreResponse, None]:
    return _request(
        method="PATCH",
        url=_table_sheet_score_endpoint(base_url, table_id, sheet_id),
        api_key=api_key,
        throw_on_error=throw_on_error,
        action="configuring your sheet score",
        json_body=build_configure_score_body(body),
        use_json_headers=True,
    )


@retry_on_api_error
async def aconfigure_sheet_score(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: ResourceId,
    sheet_id: ResourceId,
    body: ConfigureSheetScore,
) -> Union[TableScoreResponse, None]:
    return await _arequest(
        method="PATCH",
        url=_table_sheet_score_endpoint(base_url, table_id, sheet_id),
        api_key=api_key,
        throw_on_error=throw_on_error,
        action="configuring your sheet score",
        json_body=build_configure_score_body(body),
        use_json_headers=True,
    )


@retry_on_api_error
def recalculate_smart_sheet_score(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: ResourceId,
    sheet_id: ResourceId,
) -> Union[TableScoreResponse, None]:
    return _request(
        method="POST",
        url=_table_sheet_score_endpoint(base_url, table_id, sheet_id),
        api_key=api_key,
        throw_on_error=throw_on_error,
        action="recalculating your sheet score",
        expected_statuses=(200, 202),
        json_body={},
        use_json_headers=True,
    )


@retry_on_api_error
async def arecalculate_smart_sheet_score(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: ResourceId,
    sheet_id: ResourceId,
) -> Union[TableScoreResponse, None]:
    return await _arequest(
        method="POST",
        url=_table_sheet_score_endpoint(base_url, table_id, sheet_id),
        api_key=api_key,
        throw_on_error=throw_on_error,
        action="recalculating your sheet score",
        expected_statuses=(200, 202),
        json_body={},
        use_json_headers=True,
    )


@retry_on_api_error
def get_sheet_scorecard(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: ResourceId,
    sheet_id: ResourceId,
) -> Union[Dict[str, Any], None]:
    return _request(
        method="GET",
        url=_table_sheet_scorecard_endpoint(base_url, table_id, sheet_id),
        api_key=api_key,
        throw_on_error=throw_on_error,
        action="getting your sheet scorecard",
    )


@retry_on_api_error
async def aget_sheet_scorecard(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: ResourceId,
    sheet_id: ResourceId,
) -> Union[Dict[str, Any], None]:
    return await _arequest(
        method="GET",
        url=_table_sheet_scorecard_endpoint(base_url, table_id, sheet_id),
        api_key=api_key,
        throw_on_error=throw_on_error,
        action="getting your sheet scorecard",
    )


@retry_on_api_error
def configure_sheet_scorecard(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: ResourceId,
    sheet_id: ResourceId,
    body: Dict[str, Any],
) -> Union[Dict[str, Any], None]:
    return _request(
        method="PATCH",
        url=_table_sheet_scorecard_endpoint(base_url, table_id, sheet_id),
        api_key=api_key,
        throw_on_error=throw_on_error,
        action="configuring your sheet scorecard",
        json_body=body,
        use_json_headers=True,
    )


@retry_on_api_error
async def aconfigure_sheet_scorecard(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: ResourceId,
    sheet_id: ResourceId,
    body: Dict[str, Any],
) -> Union[Dict[str, Any], None]:
    return await _arequest(
        method="PATCH",
        url=_table_sheet_scorecard_endpoint(base_url, table_id, sheet_id),
        api_key=api_key,
        throw_on_error=throw_on_error,
        action="configuring your sheet scorecard",
        json_body=body,
        use_json_headers=True,
    )


@retry_on_api_error
def recalculate_smart_sheet_scorecard(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: ResourceId,
    sheet_id: ResourceId,
    body: Optional[Dict[str, Any]] = None,
) -> Union[Dict[str, Any], None]:
    return _request(
        method="POST",
        url=_table_sheet_scorecard_endpoint(base_url, table_id, sheet_id, "recalculate"),
        api_key=api_key,
        throw_on_error=throw_on_error,
        action="recalculating your sheet scorecard",
        expected_statuses=(200, 202),
        json_body=body or {},
        use_json_headers=True,
    )


@retry_on_api_error
async def arecalculate_smart_sheet_scorecard(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: ResourceId,
    sheet_id: ResourceId,
    body: Optional[Dict[str, Any]] = None,
) -> Union[Dict[str, Any], None]:
    return await _arequest(
        method="POST",
        url=_table_sheet_scorecard_endpoint(base_url, table_id, sheet_id, "recalculate"),
        api_key=api_key,
        throw_on_error=throw_on_error,
        action="recalculating your sheet scorecard",
        expected_statuses=(200, 202),
        json_body=body or {},
        use_json_headers=True,
    )


@retry_on_api_error
def get_sheet_scorecard_row(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: ResourceId,
    sheet_id: ResourceId,
    row_index: int,
    params: Optional[Dict[str, Any]] = None,
) -> Union[Dict[str, Any], None]:
    return _request(
        method="GET",
        url=_table_sheet_scorecard_endpoint(base_url, table_id, sheet_id, "rows", row_index),
        api_key=api_key,
        throw_on_error=throw_on_error,
        action="getting your sheet scorecard row",
        params=params,
    )


@retry_on_api_error
async def aget_sheet_scorecard_row(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: ResourceId,
    sheet_id: ResourceId,
    row_index: int,
    params: Optional[Dict[str, Any]] = None,
) -> Union[Dict[str, Any], None]:
    return await _arequest(
        method="GET",
        url=_table_sheet_scorecard_endpoint(base_url, table_id, sheet_id, "rows", row_index),
        api_key=api_key,
        throw_on_error=throw_on_error,
        action="getting your sheet scorecard row",
        params=params,
    )


@retry_on_api_error
def add_trace_import(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    body: AddTraceImport,
) -> Union[Dict[str, Any], None]:
    return _request(
        method="POST",
        url=_add_trace_endpoint(base_url),
        api_key=api_key,
        throw_on_error=throw_on_error,
        action="adding your trace import",
        expected_statuses=(200, 201),
        json_body=build_add_trace_body(body),
        use_json_headers=True,
    )


@retry_on_api_error
async def aadd_trace_import(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    body: AddTraceImport,
) -> Union[Dict[str, Any], None]:
    return await _arequest(
        method="POST",
        url=_add_trace_endpoint(base_url),
        api_key=api_key,
        throw_on_error=throw_on_error,
        action="adding your trace import",
        expected_statuses=(200, 201),
        json_body=build_add_trace_body(body),
        use_json_headers=True,
    )
