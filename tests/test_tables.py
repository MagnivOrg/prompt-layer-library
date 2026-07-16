from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from promptlayer import PromptLayer
from promptlayer.tables.api import (
    add_trace_import,
    create_sheet_column,
    create_table,
    list_tables,
)


def test_list_tables_uses_public_api(promptlayer_api_key, base_url):
    payload = {
        "success": True,
        "data": [{"id": "1", "title": "Eval Table"}],
    }

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = payload

    with patch("promptlayer.tables.api._get_requests_session") as mock_session:
        mock_session.return_value.get.return_value = mock_response

        response = list_tables(
            api_key=promptlayer_api_key,
            base_url=base_url,
            throw_on_error=True,
            params={"limit": 20},
        )

        call_args = mock_session.return_value.get.call_args
        assert call_args[0][0] == f"{base_url}/api/public/v2/tables"
        assert call_args[1]["params"] == {"limit": 20}
        assert response == payload


def test_create_table_maps_request_body(promptlayer_api_key, base_url):
    body = {"title": "Support Agent Eval", "folder_id": 3}
    payload = {
        "success": True,
        "table": {"id": "10", "title": "Support Agent Eval"},
    }

    mock_response = MagicMock()
    mock_response.status_code = 201
    mock_response.json.return_value = payload

    with patch("promptlayer.tables.api._get_requests_session") as mock_session:
        mock_session.return_value.post.return_value = mock_response

        response = create_table(
            api_key=promptlayer_api_key,
            base_url=base_url,
            throw_on_error=True,
            body=body,
        )

        call_args = mock_session.return_value.post.call_args
        assert call_args[0][0] == f"{base_url}/api/public/v2/tables"
        assert call_args[1]["json"] == body
        assert response == payload


def test_create_sheet_column_maps_snake_case_body(promptlayer_api_key, base_url):
    payload = {
        "success": True,
        "column": {"id": "7", "title": "Correctness", "type": "LLM_ASSERTION"},
    }

    mock_response = MagicMock()
    mock_response.status_code = 201
    mock_response.json.return_value = payload

    with patch("promptlayer.tables.api._get_requests_session") as mock_session:
        mock_session.return_value.post.return_value = mock_response

        response = create_sheet_column(
            api_key=promptlayer_api_key,
            base_url=base_url,
            throw_on_error=True,
            table_id="10",
            sheet_id="5",
            body={
                "title": "Correctness",
                "type": "LLM_ASSERTION",
                "is_output_column": True,
            },
        )

        call_args = mock_session.return_value.post.call_args
        assert call_args[0][0] == f"{base_url}/api/public/v2/tables/10/sheets/5/columns"
        assert call_args[1]["json"] == {
            "title": "Correctness",
            "type": "LLM_ASSERTION",
            "is_output_column": True,
        }
        assert response == payload


def test_add_trace_import_uses_legacy_endpoint(promptlayer_api_key, base_url):
    payload = {"success": True, "rows_added": 1}

    mock_response = MagicMock()
    mock_response.status_code = 201
    mock_response.json.return_value = payload

    with patch("promptlayer.tables.api._get_requests_session") as mock_session:
        mock_session.return_value.post.return_value = mock_response

        response = add_trace_import(
            api_key=promptlayer_api_key,
            base_url=base_url,
            throw_on_error=True,
            body={
                "trace_id": "abc123trace",
                "sheet_id": "5",
                "smart_table_id": "10",
                "metadata": {"source": "eval"},
            },
        )

        call_args = mock_session.return_value.post.call_args
        assert call_args[0][0] == f"{base_url}/api/public/v2/dataset-versions/add-trace"
        assert call_args[1]["json"] == {
            "trace_id": "abc123trace",
            "sheet_id": "5",
            "smart_table_id": "10",
            "metadata": {"source": "eval"},
        }
        assert response == payload


def test_promptlayer_client_exposes_tables_manager(promptlayer_api_key, base_url):
    client = PromptLayer(api_key=promptlayer_api_key, base_url=base_url)

    assert client.tables is not None
    assert hasattr(client.tables, "list")
    assert hasattr(client.tables, "create")
    assert hasattr(client.tables.imports, "add_trace")


def test_nested_sheet_resources_are_addressable(promptlayer_api_key, base_url):
    client = PromptLayer(api_key=promptlayer_api_key, base_url=base_url)
    sheet = client.tables.sheets("10").for_sheet("5")

    assert hasattr(sheet.rows, "list")
    assert hasattr(sheet.columns, "create")
    assert hasattr(sheet.score, "recalculate")


@pytest.mark.asyncio
async def test_async_list_tables(promptlayer_api_key, base_url):
    payload = {"success": True, "data": [{"id": "1", "title": "Eval Table"}]}

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = payload

    with patch("promptlayer.tables.api._make_httpx_client") as mock_client_factory:
        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client_factory.return_value = mock_client

        from promptlayer.tables.api import alist_tables

        response = await alist_tables(
            api_key=promptlayer_api_key,
            base_url=base_url,
            throw_on_error=True,
        )

        call_args = mock_client.get.call_args
        assert call_args[0][0] == f"{base_url}/api/public/v2/tables"
        assert response == payload
