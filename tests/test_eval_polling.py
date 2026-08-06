from contextlib import asynccontextmanager
from unittest.mock import MagicMock, patch

import pytest

from promptlayer import PromptLayerValidationError
from promptlayer.evaluations.polling import (
    _status_counts_are_terminal,
    _wait_for_sheet_cells,
    await_for_sheet_operations,
    wait_for_sheet_operations,
)
from promptlayer.evaluations.runner import _map_batch_row_indices
from promptlayer.evaluations.scorecard import recalculate_and_wait_scorecard
from promptlayer.evaluations.tracing import flush_traces, maybe_await
from promptlayer.tables import api as tables_api


def _counts(**overrides):
    counts = {
        "STALE": 0,
        "QUEUED": 0,
        "DISPATCHED": 0,
        "RUNNING": 0,
        "COMPLETED": 1,
        "FAILED": 0,
    }
    counts.update(overrides)
    return {"total_cells": sum(counts.values()), "status_counts": counts}


def test_status_counts_terminal_contract():
    assert _status_counts_are_terminal({"total_cells": 0, "status_counts": {}})
    assert _status_counts_are_terminal(_counts(COMPLETED=2, FAILED=1))
    assert not _status_counts_are_terminal(_counts(COMPLETED=0, RUNNING=1))
    assert not _status_counts_are_terminal(_counts(COMPLETED=0, STALE=1))
    assert not _status_counts_are_terminal({"total_cells": 2, "status_counts": {"COMPLETED": 1}})


def test_status_count_polling_uses_one_request_per_iteration():
    responses = [_counts(COMPLETED=0, RUNNING=1), _counts()]
    progress = []
    with (
        patch(
            "promptlayer.tables.api.get_sheet_status_counts",
            side_effect=responses,
        ) as get_counts,
        patch("promptlayer.evaluations.polling.time.sleep") as sleep,
    ):
        result = _wait_for_sheet_cells(
            "key",
            "url",
            True,
            "table",
            "sheet",
            on_progress=lambda completed, total, failed: progress.append((completed, total, failed)),
        )

    assert result == responses[-1]
    assert progress == [(0, 1, 0), (1, 1, 0)]
    assert get_counts.call_count == 2
    sleep.assert_called_once_with(0.5)


def test_status_count_polling_times_out():
    with (
        patch(
            "promptlayer.tables.api.get_sheet_status_counts",
            return_value=_counts(COMPLETED=0, RUNNING=1),
        ),
        patch("promptlayer.evaluations.polling.time.monotonic", side_effect=[0.0, 2.0]),
    ):
        with pytest.raises(Exception, match="Timed out waiting for eval cells"):
            _wait_for_sheet_cells(
                "key",
                "url",
                True,
                "table",
                "sheet",
                timeout_seconds=1.0,
            )


def test_preprocessing_operation_polls_operation_status():
    terminal = {
        "operation_id": "operation-1",
        "status": "completed",
        "completed_count": 4,
        "failed_count": 0,
        "pending_count": 0,
        "cell_count": 4,
    }
    with (
        patch(
            "promptlayer.tables.api.create_sheet_operation",
            return_value={"cell_count": 4, "operation_id": "operation-1", "operation": "recalculate"},
        ),
        patch(
            "promptlayer.tables.api.get_sheet_operation",
            # Public API nests status under "operation".
            return_value={"success": True, "operation": terminal},
        ) as get_operation,
        patch("promptlayer.tables.api.get_sheet_status_counts") as get_counts,
    ):
        result = wait_for_sheet_operations(
            "key",
            "url",
            True,
            "table",
            "sheet",
            column_ids=["column"],
        )

    assert result == terminal
    get_operation.assert_called_once()
    get_counts.assert_not_called()


@pytest.mark.asyncio
async def test_async_preprocessing_operation_polls_operation_status():
    terminal = {
        "operation_id": "operation-1",
        "status": "completed",
        "completed_count": 4,
        "failed_count": 0,
        "pending_count": 0,
        "cell_count": 4,
    }
    with (
        patch(
            "promptlayer.tables.api.acreate_sheet_operation",
            return_value={"cell_count": 4, "operation_id": "operation-1", "operation": "recalculate"},
        ),
        patch(
            "promptlayer.tables.api.aget_sheet_operation",
            return_value={"success": True, "operation": terminal},
        ) as get_operation,
        patch("promptlayer.tables.api.aget_sheet_status_counts") as get_counts,
    ):
        result = await await_for_sheet_operations(
            "key",
            "url",
            True,
            "table",
            "sheet",
            column_ids=["column"],
        )

    assert result == terminal
    get_operation.assert_awaited_once()
    get_counts.assert_not_called()


def test_operation_is_terminal_when_cell_counts_finish_without_status():
    from promptlayer.evaluations.polling import _operation_is_terminal

    assert _operation_is_terminal(
        {
            "success": True,
            "operation": {
                "pending_count": 0,
                "completed_count": 8,
                "failed_count": 0,
                "cell_count": 8,
            },
        }
    )
    assert not _operation_is_terminal(
        {
            "success": True,
            "operation": {
                "status": "running",
                "pending_count": 2,
                "completed_count": 6,
                "failed_count": 0,
                "cell_count": 8,
            },
        }
    )


def test_scorecard_polling_waits_for_terminal_calculation():
    responses = [
        {
            "latest_calculation": {"id": "calc-old", "status": "completed"},
            "scorecard": {"status": "completed"},
        },
        {
            "latest_calculation": {"id": "calc-1", "status": "running"},
            "progress": {"scored_rows": 1, "total_rows": 2},
        },
        {
            "latest_calculation": {"id": "calc-1", "status": "completed"},
            "progress": {"scored_rows": 2, "total_rows": 2},
            "scorecard": {"status": "completed"},
        },
    ]
    with (
        patch(
            "promptlayer.tables.api.recalculate_smart_sheet_scorecard",
            return_value={"calculation_id": "calc-1"},
        ),
        patch(
            "promptlayer.tables.api.get_sheet_scorecard",
            side_effect=responses,
        ) as get_scorecard,
        patch("promptlayer.evaluations.polling.time.sleep"),
        patch("promptlayer.evaluations.scorecard.get_terminal") as terminal,
    ):
        result = recalculate_and_wait_scorecard("key", "url", True, "table", "sheet")

    assert result == responses[-1]
    assert get_scorecard.call_count == 3
    terminal.return_value.scoring_progress.assert_any_call(1, 2, 0)
    terminal.return_value.scoring_progress.assert_any_call(2, 2, 0)


def test_scorecard_polling_times_out():
    with (
        patch(
            "promptlayer.tables.api.recalculate_smart_sheet_scorecard",
            return_value={"calculation_id": "calc-1"},
        ),
        patch(
            "promptlayer.tables.api.get_sheet_scorecard",
            return_value={"latest_calculation": {"id": "calc-1", "status": "running"}},
        ),
        patch("promptlayer.evaluations.polling.time.monotonic", side_effect=[0.0, 2.0]),
        patch("promptlayer.evaluations.polling.time.sleep"),
    ):
        with pytest.raises(Exception, match="Timed out waiting for scorecard"):
            recalculate_and_wait_scorecard(
                "key",
                "url",
                True,
                "table",
                "sheet",
                timeout_seconds=1.0,
            )


def test_list_all_rows_follows_cursor_and_preserves_first_page_metadata():
    pages = [
        {
            "data": [{"row_index": 0}],
            "columns": [{"id": "input"}],
            "row_count": 2,
            "pagination": {"next_cursor": "next"},
        },
        {
            "data": [{"row_index": 1}],
            "pagination": {"next_cursor": None},
        },
    ]
    with patch("promptlayer.tables.api.list_smart_sheet_rows", side_effect=pages) as list_rows:
        result = tables_api.list_all_smart_sheet_rows("key", "url", True, "table", "sheet")

    assert result["data"] == [{"row_index": 0}, {"row_index": 1}]
    assert result["columns"] == [{"id": "input"}]
    assert result["row_count"] == 2
    assert list_rows.call_args_list[1].kwargs["params"]["cursor"] == "next"
    assert list_rows.call_args_list[1].kwargs["params"]["include_columns"] is False


def test_status_counts_api_uses_public_sheet_endpoint():
    response = _counts()
    with patch("promptlayer.tables.api._request", return_value=response) as request:
        result = tables_api.get_sheet_status_counts(
            "key",
            "https://api.example.com",
            True,
            "table/id",
            "sheet/id",
        )

    assert result == response
    assert request.call_args.kwargs["url"] == (
        "https://api.example.com/api/public/v2/tables/table%2Fid/sheets/sheet%2Fid/status-counts"
    )


def test_batch_row_indices_must_match_case_count():
    with pytest.raises(PromptLayerValidationError, match="1 row indices for 3 eval cases"):
        _map_batch_row_indices({"row_indices": [5]}, 3)


@pytest.mark.asyncio
async def test_nested_async_client_session_reuses_one_client():
    client = object()
    entered = 0

    @asynccontextmanager
    async def client_factory():
        nonlocal entered
        entered += 1
        yield client

    with patch("promptlayer.tables.api._make_httpx_client", side_effect=client_factory):
        async with tables_api.async_client_session() as first:
            async with tables_api.async_client_session() as second:
                assert first is second is client

    assert entered == 1


def test_trace_flush_failure_warns_or_raises(caplog):
    provider = MagicMock()
    provider.force_flush.side_effect = RuntimeError("flush failed")

    flush_traces(provider)
    assert "Failed to flush eval traces" in caplog.text

    with pytest.raises(RuntimeError, match="flush failed"):
        flush_traces(provider, throw_on_error=True)


def test_eval_runner_rejects_stream_output():
    def stream():
        yield "chunk"

    with pytest.raises(PromptLayerValidationError, match="returned a stream"):
        maybe_await(stream())
