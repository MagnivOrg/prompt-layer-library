from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from promptlayer.tables import (
    aget_scorecard,
    configure_scorecard,
    delete_scorecard,
    get_scorecard,
    get_scorecard_calculation,
    get_scorecard_row,
    list_scorecard_rows,
    migrate_legacy_score,
    recalculate_scorecard,
)
from promptlayer.tables.scorecards import cancel_scorecard_calculation
from promptlayer.types.scorecard import (
    CancelScorecardRequest,
    ConfigureScorecardRequest,
    GetScoreScorecardCompatibilityResponse,
    GetScorecardRowOptions,
    ListScorecardRowsOptions,
    MigrateLegacyScorecardRequest,
    RecalculateScoreResponse,
    RecalculateScorecardRequest,
)

TABLE_ID = "table/1"
SHEET_ID = "sheet 2"
ENCODED_BASE = "/api/public/v2/tables/table%2F1/sheets/sheet%202/scorecard"


def _mock_response(payload, status_code=200):
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = payload
    return response


def _assert_request(call_args, base_url, api_key, suffix="", *, json=None, params=None):
    assert call_args[0][0] == f"{base_url}{ENCODED_BASE}{suffix}"
    assert call_args[1]["headers"] == {"X-API-KEY": api_key}
    assert call_args[1]["json"] == json
    assert call_args[1]["params"] == params


def test_get_scorecard_sends_expected_request(promptlayer_api_key, base_url):
    payload = {"success": True, "scorecard": {"id": "sc_1", "name": "Quality"}}

    with patch("promptlayer.tables.scorecards._get_requests_session") as mock_session:
        mock_session.return_value.get.return_value = _mock_response(payload)

        response = get_scorecard(promptlayer_api_key, base_url, True, TABLE_ID, SHEET_ID)

        _assert_request(mock_session.return_value.get.call_args, base_url, promptlayer_api_key)
        assert response == payload


def test_configure_scorecard_puts_body(promptlayer_api_key, base_url):
    body: ConfigureScorecardRequest = {
        "name": "Quality Scorecard",
        "evaluated_column_ids": [],
        "aggregation": {
            "method": "weighted_mean",
            "required_step_failure_behavior": "fail",
            "pass_threshold": 0.8,
            "warn_threshold": 0.6,
        },
        "steps": [],
    }
    payload = {"success": True, "scorecard": {"id": "sc_1", **body}}

    with patch("promptlayer.tables.scorecards._get_requests_session") as mock_session:
        mock_session.return_value.put.return_value = _mock_response(payload)

        response = configure_scorecard(promptlayer_api_key, base_url, True, TABLE_ID, SHEET_ID, body)

        _assert_request(mock_session.return_value.put.call_args, base_url, promptlayer_api_key, json=body)
        assert response == payload


def test_migrate_legacy_score_posts_options(promptlayer_api_key, base_url):
    body: MigrateLegacyScorecardRequest = {"delete_legacy_score": False}
    payload = {"success": True, "scorecard": {"id": "sc_1"}}

    with patch("promptlayer.tables.scorecards._get_requests_session") as mock_session:
        mock_session.return_value.post.return_value = _mock_response(payload)

        response = migrate_legacy_score(promptlayer_api_key, base_url, True, TABLE_ID, SHEET_ID, body)

        _assert_request(
            mock_session.return_value.post.call_args,
            base_url,
            promptlayer_api_key,
            "/migrate-legacy-score",
            json=body,
        )
        assert response == payload


def test_recalculate_scorecard_posts_options(promptlayer_api_key, base_url):
    body: RecalculateScorecardRequest = {"row_indices": [0, 1], "force": True}
    payload = {"success": True, "calculation_id": "calc_1", "status": "queued", "version": 3}

    with patch("promptlayer.tables.scorecards._get_requests_session") as mock_session:
        mock_session.return_value.post.return_value = _mock_response(payload)

        response = recalculate_scorecard(promptlayer_api_key, base_url, True, TABLE_ID, SHEET_ID, body)

        _assert_request(
            mock_session.return_value.post.call_args,
            base_url,
            promptlayer_api_key,
            "/recalculate",
            json=body,
        )
        assert response == payload


def test_cancel_scorecard_posts_options(promptlayer_api_key, base_url):
    body: CancelScorecardRequest = {"calculation_id": "calc_1"}
    payload = {"success": True, "calculation_id": "calc_1", "status": "cancelled"}

    with patch("promptlayer.tables.scorecards._get_requests_session") as mock_session:
        mock_session.return_value.post.return_value = _mock_response(payload)

        response = cancel_scorecard_calculation(promptlayer_api_key, base_url, True, TABLE_ID, SHEET_ID, body)

        _assert_request(mock_session.return_value.post.call_args, base_url, promptlayer_api_key, "/cancel", json=body)
        assert response == payload


def test_get_scorecard_calculation_sends_expected_request(promptlayer_api_key, base_url):
    payload = {"success": True, "calculation": {"id": "calc_1", "status": "completed"}}

    with patch("promptlayer.tables.scorecards._get_requests_session") as mock_session:
        mock_session.return_value.get.return_value = _mock_response(payload)

        response = get_scorecard_calculation(promptlayer_api_key, base_url, True, TABLE_ID, SHEET_ID, "calc_1")

        _assert_request(mock_session.return_value.get.call_args, base_url, promptlayer_api_key, "/calculations/calc_1")
        assert response == payload


def test_list_scorecard_rows_forwards_query_params(promptlayer_api_key, base_url):
    options: ListScorecardRowsOptions = {"calculation_id": "calc_1", "verdict": "fail", "limit": 25, "offset": 50}
    payload = {"success": True, "rows": [{"row_index": 0, "verdict": "fail"}], "next_offset": None}

    with patch("promptlayer.tables.scorecards._get_requests_session") as mock_session:
        mock_session.return_value.get.return_value = _mock_response(payload)

        response = list_scorecard_rows(promptlayer_api_key, base_url, True, TABLE_ID, SHEET_ID, options)

        _assert_request(mock_session.return_value.get.call_args, base_url, promptlayer_api_key, "/rows", params=options)
        assert response == payload


def test_get_scorecard_row_forwards_query_params(promptlayer_api_key, base_url):
    options: GetScorecardRowOptions = {"calculation_id": "calc_1"}
    payload = {"success": True, "row": {"row_index": 0, "verdict": "fail", "criteria": []}}

    with patch("promptlayer.tables.scorecards._get_requests_session") as mock_session:
        mock_session.return_value.get.return_value = _mock_response(payload)

        response = get_scorecard_row(promptlayer_api_key, base_url, True, TABLE_ID, SHEET_ID, 0, options)

        _assert_request(
            mock_session.return_value.get.call_args,
            base_url,
            promptlayer_api_key,
            "/rows/0",
            params=options,
        )
        assert response == payload


def test_delete_scorecard_sends_expected_request(promptlayer_api_key, base_url):
    payload = {"success": True}

    with patch("promptlayer.tables.scorecards._get_requests_session") as mock_session:
        mock_session.return_value.delete.return_value = _mock_response(payload)

        response = delete_scorecard(promptlayer_api_key, base_url, True, TABLE_ID, SHEET_ID)

        _assert_request(mock_session.return_value.delete.call_args, base_url, promptlayer_api_key)
        assert response == payload


def test_scorecard_manager_is_available_on_client(promptlayer_client):
    assert hasattr(promptlayer_client.tables.sheets.scorecards, "get")
    assert hasattr(promptlayer_client.tables.sheets.scorecards, "configure")
    assert hasattr(promptlayer_client.tables.sheets.scorecards, "migrate_legacy_score")


@pytest.mark.asyncio
async def test_async_get_scorecard_sends_expected_request(promptlayer_api_key, base_url):
    payload = {"success": True, "scorecard": {"id": "sc_1", "name": "Quality"}}

    with patch("promptlayer.tables.scorecards._make_httpx_client") as mock_client_factory:
        mock_client = AsyncMock()
        mock_client.get.return_value = _mock_response(payload)
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client_factory.return_value = mock_client

        response = await aget_scorecard(promptlayer_api_key, base_url, True, TABLE_ID, SHEET_ID)

        _assert_request(mock_client.get.call_args, base_url, promptlayer_api_key)
        assert response == payload


def test_get_score_scorecard_compatibility_shape_is_supported_by_types():
    payload = cast(
        GetScoreScorecardCompatibilityResponse,
        cast(
            object,
            {
                "score_type": "scorecard",
                "scoring_type": "scorecard",
                "score_configuration": None,
                "details": {
                    "aggregate_result": {
                        "scorecard_id": "sc_1",
                        "scorecard_calculation_id": "calc_1",
                    }
                },
            },
        ),
    )

    assert payload["score_configuration"] is None
    assert payload["details"]["aggregate_result"]["scorecard_calculation_id"] == "calc_1"


def test_recalculate_score_scorecard_piggyback_shape_is_supported_by_types():
    payload = cast(
        RecalculateScoreResponse,
        cast(
            object,
            {
                "success": True,
                "score_source": "scorecard",
                "calculation_id": "calc_1",
                "status": "queued",
                "version": 2,
            },
        ),
    )

    assert payload["success"] is True
    assert payload["status"] == "queued"
