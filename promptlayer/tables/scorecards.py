from typing import Any, Dict, Union
from urllib.parse import quote

import httpx
import requests

from promptlayer import exceptions as _exceptions
from promptlayer.types.scorecard import (
    CancelScorecardRequest,
    ConfigureScorecardRequest,
    GetScorecardRowOptions,
    ListScorecardRowsOptions,
    MigrateLegacyScorecardRequest,
    RecalculateScorecardRequest,
    ScorecardActionResponse,
    ScorecardCalculationResponse,
    ScorecardResponse,
    ScorecardRowResponse,
    ScorecardRowsResponse,
)
from promptlayer.utils import (
    _get_requests_session,
    _make_httpx_client,
    logger,
    raise_on_bad_response,
    retry_on_api_error,
    warn_on_bad_response,
)

JsonResponse = Union[
    ScorecardResponse,
    ScorecardActionResponse,
    ScorecardCalculationResponse,
    ScorecardRowsResponse,
    ScorecardRowResponse,
    Dict[str, Any],
]


def _scorecard_endpoint(base_url: str, table_id: Union[str, int], sheet_id: Union[str, int], *parts: Any) -> str:
    encoded_table_id = quote(str(table_id), safe="")
    encoded_sheet_id = quote(str(sheet_id), safe="")
    suffix = "/".join(quote(str(part), safe="") for part in parts if part is not None)
    base = f"{base_url}/api/public/v2/tables/{encoded_table_id}/sheets/{encoded_sheet_id}/scorecard"
    return f"{base}/{suffix}" if suffix else base


def _handle_response(response, throw_on_error: bool, success_status: int, error_context: str) -> Any:
    if response.status_code != success_status:
        if throw_on_error:
            raise_on_bad_response(response, error_context)
        warn_on_bad_response(response, f"WARNING: {error_context}")
        return None
    if response.status_code == 204:
        return {"success": True}
    return response.json()


@retry_on_api_error
def _sync_request(
    method: str,
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: Union[str, int],
    sheet_id: Union[str, int],
    *,
    parts=(),
    params: Any = None,
    json: Any = None,
    success_status: int = 200,
    error_context: str,
) -> Any:
    try:
        request_method = getattr(_get_requests_session(), method)
        response = request_method(
            _scorecard_endpoint(base_url, table_id, sheet_id, *parts),
            headers={"X-API-KEY": api_key},
            params=params,
            json=json,
        )
        return _handle_response(response, throw_on_error, success_status, error_context)
    except requests.exceptions.RequestException as e:
        if throw_on_error:
            raise _exceptions.PromptLayerAPIConnectionError(
                f"{error_context}: {e}",
                response=None,
                body=None,
            ) from e
        logger.warning(f"{error_context}: {e}")
        return None


@retry_on_api_error
async def _async_request(
    method: str,
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: Union[str, int],
    sheet_id: Union[str, int],
    *,
    parts=(),
    params: Any = None,
    json: Any = None,
    success_status: int = 200,
    error_context: str,
) -> Any:
    try:
        async with _make_httpx_client() as client:
            request_method = getattr(client, method)
            response = await request_method(
                _scorecard_endpoint(base_url, table_id, sheet_id, *parts),
                headers={"X-API-KEY": api_key},
                params=params,
                json=json,
            )
        return _handle_response(response, throw_on_error, success_status, error_context)
    except httpx.RequestError as e:
        if throw_on_error:
            raise _exceptions.PromptLayerAPIConnectionError(
                f"{error_context}: {str(e)}",
                response=None,
                body=None,
            ) from e
        logger.warning(f"{error_context}: {e}")
        return None


def get_scorecard(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: Union[str, int],
    sheet_id: Union[str, int],
) -> Union[ScorecardResponse, None]:
    return _sync_request(
        "get",
        api_key,
        base_url,
        throw_on_error,
        table_id,
        sheet_id,
        error_context="PromptLayer had the following error while fetching your scorecard",
    )


async def aget_scorecard(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: Union[str, int],
    sheet_id: Union[str, int],
) -> Union[ScorecardResponse, None]:
    return await _async_request(
        "get",
        api_key,
        base_url,
        throw_on_error,
        table_id,
        sheet_id,
        error_context="PromptLayer had the following error while fetching your scorecard",
    )


def configure_scorecard(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: Union[str, int],
    sheet_id: Union[str, int],
    body: ConfigureScorecardRequest,
) -> Union[ScorecardResponse, None]:
    return _sync_request(
        "put",
        api_key,
        base_url,
        throw_on_error,
        table_id,
        sheet_id,
        json=body,
        error_context="PromptLayer had the following error while configuring your scorecard",
    )


async def aconfigure_scorecard(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: Union[str, int],
    sheet_id: Union[str, int],
    body: ConfigureScorecardRequest,
) -> Union[ScorecardResponse, None]:
    return await _async_request(
        "put",
        api_key,
        base_url,
        throw_on_error,
        table_id,
        sheet_id,
        json=body,
        error_context="PromptLayer had the following error while configuring your scorecard",
    )


def delete_scorecard(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: Union[str, int],
    sheet_id: Union[str, int],
) -> Union[Dict[str, Any], None]:
    return _sync_request(
        "delete",
        api_key,
        base_url,
        throw_on_error,
        table_id,
        sheet_id,
        error_context="PromptLayer had the following error while deleting your scorecard",
    )


async def adelete_scorecard(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: Union[str, int],
    sheet_id: Union[str, int],
) -> Union[Dict[str, Any], None]:
    return await _async_request(
        "delete",
        api_key,
        base_url,
        throw_on_error,
        table_id,
        sheet_id,
        error_context="PromptLayer had the following error while deleting your scorecard",
    )


def migrate_legacy_score(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: Union[str, int],
    sheet_id: Union[str, int],
    body: Union[MigrateLegacyScorecardRequest, None] = None,
) -> Union[ScorecardActionResponse, None]:
    return _sync_request(
        "post",
        api_key,
        base_url,
        throw_on_error,
        table_id,
        sheet_id,
        parts=("migrate-legacy-score",),
        json=body or {},
        error_context="PromptLayer had the following error while migrating your legacy score",
    )


async def amigrate_legacy_score(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: Union[str, int],
    sheet_id: Union[str, int],
    body: Union[MigrateLegacyScorecardRequest, None] = None,
) -> Union[ScorecardActionResponse, None]:
    return await _async_request(
        "post",
        api_key,
        base_url,
        throw_on_error,
        table_id,
        sheet_id,
        parts=("migrate-legacy-score",),
        json=body or {},
        error_context="PromptLayer had the following error while migrating your legacy score",
    )


def recalculate_scorecard(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: Union[str, int],
    sheet_id: Union[str, int],
    body: Union[RecalculateScorecardRequest, None] = None,
) -> Union[ScorecardActionResponse, None]:
    return _sync_request(
        "post",
        api_key,
        base_url,
        throw_on_error,
        table_id,
        sheet_id,
        parts=("recalculate",),
        json=body or {},
        error_context="PromptLayer had the following error while recalculating your scorecard",
    )


async def arecalculate_scorecard(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: Union[str, int],
    sheet_id: Union[str, int],
    body: Union[RecalculateScorecardRequest, None] = None,
) -> Union[ScorecardActionResponse, None]:
    return await _async_request(
        "post",
        api_key,
        base_url,
        throw_on_error,
        table_id,
        sheet_id,
        parts=("recalculate",),
        json=body or {},
        error_context="PromptLayer had the following error while recalculating your scorecard",
    )


def cancel_scorecard_calculation(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: Union[str, int],
    sheet_id: Union[str, int],
    body: Union[CancelScorecardRequest, None] = None,
) -> Union[ScorecardActionResponse, None]:
    return _sync_request(
        "post",
        api_key,
        base_url,
        throw_on_error,
        table_id,
        sheet_id,
        parts=("cancel",),
        json=body or {},
        error_context="PromptLayer had the following error while cancelling your scorecard calculation",
    )


async def acancel_scorecard_calculation(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: Union[str, int],
    sheet_id: Union[str, int],
    body: Union[CancelScorecardRequest, None] = None,
) -> Union[ScorecardActionResponse, None]:
    return await _async_request(
        "post",
        api_key,
        base_url,
        throw_on_error,
        table_id,
        sheet_id,
        parts=("cancel",),
        json=body or {},
        error_context="PromptLayer had the following error while cancelling your scorecard calculation",
    )


def get_scorecard_calculation(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: Union[str, int],
    sheet_id: Union[str, int],
    calculation_id: str,
) -> Union[ScorecardCalculationResponse, None]:
    return _sync_request(
        "get",
        api_key,
        base_url,
        throw_on_error,
        table_id,
        sheet_id,
        parts=("calculations", calculation_id),
        error_context="PromptLayer had the following error while fetching your scorecard calculation",
    )


async def aget_scorecard_calculation(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: Union[str, int],
    sheet_id: Union[str, int],
    calculation_id: str,
) -> Union[ScorecardCalculationResponse, None]:
    return await _async_request(
        "get",
        api_key,
        base_url,
        throw_on_error,
        table_id,
        sheet_id,
        parts=("calculations", calculation_id),
        error_context="PromptLayer had the following error while fetching your scorecard calculation",
    )


def list_scorecard_rows(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: Union[str, int],
    sheet_id: Union[str, int],
    options: Union[ListScorecardRowsOptions, None] = None,
) -> Union[ScorecardRowsResponse, None]:
    return _sync_request(
        "get",
        api_key,
        base_url,
        throw_on_error,
        table_id,
        sheet_id,
        parts=("rows",),
        params=options,
        error_context="PromptLayer had the following error while listing your scorecard rows",
    )


async def alist_scorecard_rows(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: Union[str, int],
    sheet_id: Union[str, int],
    options: Union[ListScorecardRowsOptions, None] = None,
) -> Union[ScorecardRowsResponse, None]:
    return await _async_request(
        "get",
        api_key,
        base_url,
        throw_on_error,
        table_id,
        sheet_id,
        parts=("rows",),
        params=options,
        error_context="PromptLayer had the following error while listing your scorecard rows",
    )


def get_scorecard_row(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: Union[str, int],
    sheet_id: Union[str, int],
    row_index: int,
    options: Union[GetScorecardRowOptions, None] = None,
) -> Union[ScorecardRowResponse, None]:
    return _sync_request(
        "get",
        api_key,
        base_url,
        throw_on_error,
        table_id,
        sheet_id,
        parts=("rows", row_index),
        params=options,
        error_context="PromptLayer had the following error while fetching your scorecard row",
    )


async def aget_scorecard_row(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: Union[str, int],
    sheet_id: Union[str, int],
    row_index: int,
    options: Union[GetScorecardRowOptions, None] = None,
) -> Union[ScorecardRowResponse, None]:
    return await _async_request(
        "get",
        api_key,
        base_url,
        throw_on_error,
        table_id,
        sheet_id,
        parts=("rows", row_index),
        params=options,
        error_context="PromptLayer had the following error while fetching your scorecard row",
    )


class ScorecardManager:
    def __init__(self, api_key: str, base_url: str, throw_on_error: bool):
        self.api_key = api_key
        self.base_url = base_url
        self.throw_on_error = throw_on_error

    def get(self, table_id: Union[str, int], sheet_id: Union[str, int]) -> Union[ScorecardResponse, None]:
        return get_scorecard(self.api_key, self.base_url, self.throw_on_error, table_id, sheet_id)

    def configure(
        self, table_id: Union[str, int], sheet_id: Union[str, int], body: ConfigureScorecardRequest
    ) -> Union[ScorecardResponse, None]:
        return configure_scorecard(self.api_key, self.base_url, self.throw_on_error, table_id, sheet_id, body)

    def delete(self, table_id: Union[str, int], sheet_id: Union[str, int]) -> Union[Dict[str, Any], None]:
        return delete_scorecard(self.api_key, self.base_url, self.throw_on_error, table_id, sheet_id)

    def migrate_legacy_score(
        self,
        table_id: Union[str, int],
        sheet_id: Union[str, int],
        options: Union[MigrateLegacyScorecardRequest, None] = None,
    ) -> Union[ScorecardActionResponse, None]:
        return migrate_legacy_score(self.api_key, self.base_url, self.throw_on_error, table_id, sheet_id, options)

    def recalculate(
        self,
        table_id: Union[str, int],
        sheet_id: Union[str, int],
        options: Union[RecalculateScorecardRequest, None] = None,
    ) -> Union[ScorecardActionResponse, None]:
        return recalculate_scorecard(self.api_key, self.base_url, self.throw_on_error, table_id, sheet_id, options)

    def cancel(
        self,
        table_id: Union[str, int],
        sheet_id: Union[str, int],
        options: Union[CancelScorecardRequest, None] = None,
    ) -> Union[ScorecardActionResponse, None]:
        return cancel_scorecard_calculation(
            self.api_key, self.base_url, self.throw_on_error, table_id, sheet_id, options
        )

    def get_calculation(
        self, table_id: Union[str, int], sheet_id: Union[str, int], calculation_id: str
    ) -> Union[ScorecardCalculationResponse, None]:
        return get_scorecard_calculation(
            self.api_key, self.base_url, self.throw_on_error, table_id, sheet_id, calculation_id
        )

    def list_rows(
        self,
        table_id: Union[str, int],
        sheet_id: Union[str, int],
        options: Union[ListScorecardRowsOptions, None] = None,
    ) -> Union[ScorecardRowsResponse, None]:
        return list_scorecard_rows(self.api_key, self.base_url, self.throw_on_error, table_id, sheet_id, options)

    def get_row(
        self,
        table_id: Union[str, int],
        sheet_id: Union[str, int],
        row_index: int,
        options: Union[GetScorecardRowOptions, None] = None,
    ) -> Union[ScorecardRowResponse, None]:
        return get_scorecard_row(
            self.api_key, self.base_url, self.throw_on_error, table_id, sheet_id, row_index, options
        )


class AsyncScorecardManager:
    def __init__(self, api_key: str, base_url: str, throw_on_error: bool):
        self.api_key = api_key
        self.base_url = base_url
        self.throw_on_error = throw_on_error

    async def get(self, table_id: Union[str, int], sheet_id: Union[str, int]) -> Union[ScorecardResponse, None]:
        return await aget_scorecard(self.api_key, self.base_url, self.throw_on_error, table_id, sheet_id)

    async def configure(
        self, table_id: Union[str, int], sheet_id: Union[str, int], body: ConfigureScorecardRequest
    ) -> Union[ScorecardResponse, None]:
        return await aconfigure_scorecard(self.api_key, self.base_url, self.throw_on_error, table_id, sheet_id, body)

    async def delete(self, table_id: Union[str, int], sheet_id: Union[str, int]) -> Union[Dict[str, Any], None]:
        return await adelete_scorecard(self.api_key, self.base_url, self.throw_on_error, table_id, sheet_id)

    async def migrate_legacy_score(
        self,
        table_id: Union[str, int],
        sheet_id: Union[str, int],
        options: Union[MigrateLegacyScorecardRequest, None] = None,
    ) -> Union[ScorecardActionResponse, None]:
        return await amigrate_legacy_score(
            self.api_key, self.base_url, self.throw_on_error, table_id, sheet_id, options
        )

    async def recalculate(
        self,
        table_id: Union[str, int],
        sheet_id: Union[str, int],
        options: Union[RecalculateScorecardRequest, None] = None,
    ) -> Union[ScorecardActionResponse, None]:
        return await arecalculate_scorecard(
            self.api_key, self.base_url, self.throw_on_error, table_id, sheet_id, options
        )

    async def cancel(
        self,
        table_id: Union[str, int],
        sheet_id: Union[str, int],
        options: Union[CancelScorecardRequest, None] = None,
    ) -> Union[ScorecardActionResponse, None]:
        return await acancel_scorecard_calculation(
            self.api_key, self.base_url, self.throw_on_error, table_id, sheet_id, options
        )

    async def get_calculation(
        self, table_id: Union[str, int], sheet_id: Union[str, int], calculation_id: str
    ) -> Union[ScorecardCalculationResponse, None]:
        return await aget_scorecard_calculation(
            self.api_key, self.base_url, self.throw_on_error, table_id, sheet_id, calculation_id
        )

    async def list_rows(
        self,
        table_id: Union[str, int],
        sheet_id: Union[str, int],
        options: Union[ListScorecardRowsOptions, None] = None,
    ) -> Union[ScorecardRowsResponse, None]:
        return await alist_scorecard_rows(self.api_key, self.base_url, self.throw_on_error, table_id, sheet_id, options)

    async def get_row(
        self,
        table_id: Union[str, int],
        sheet_id: Union[str, int],
        row_index: int,
        options: Union[GetScorecardRowOptions, None] = None,
    ) -> Union[ScorecardRowResponse, None]:
        return await aget_scorecard_row(
            self.api_key, self.base_url, self.throw_on_error, table_id, sheet_id, row_index, options
        )
