import asyncio
from contextlib import asynccontextmanager, nullcontext
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pytest_parametrize_cases import Case, parametrize_cases

from promptlayer.utils import arun_workflow_request
from tests.utils.mocks import Any
from tests.utils.vcr import assert_played, is_cassette_recording


@parametrize_cases(
    Case("Regular call", kwargs={"workflow_id_or_name": "analyze_1", "input_variables": {"var1": "value1"}}),
    Case("Legacy call", kwargs={"workflow_name": "analyze_1", "input_variables": {"var1": "value1"}}),
)
@pytest.mark.asyncio
async def test_arun_workflow_request(base_url: str, throw_on_error: bool, promptlayer_api_key, kwargs):
    is_recording = is_cassette_recording()
    client = MagicMock()

    @asynccontextmanager
    async def centrifugo_client_context():
        yield client

    @asynccontextmanager
    async def centrifugo_subscription_context():
        yield

    async def resolve_results(results_future, timeout):
        results_future.set_result({"Node 2": "False", "Node 3": "AAA"})
        return results_future.result()

    with (
        assert_played("test_arun_workflow_request.yaml") as cassette,
        patch(
            "promptlayer.utils._make_channel_name_suffix", return_value="8dd7e4d404754c60a50e78f70f74aade"
        ) as _make_channel_name_suffix_mock,
        nullcontext()
        if is_recording
        else patch(
            "promptlayer.utils.centrifugo_client", return_value=centrifugo_client_context()
        ) as centrifugo_client_mock,
        nullcontext()
        if is_recording
        else patch(
            "promptlayer.utils.centrifugo_subscription", return_value=centrifugo_subscription_context()
        ) as centrifugo_subscription_mock,
        nullcontext()
        if is_recording
        else patch(
            "promptlayer.utils.asyncio.wait_for", new_callable=AsyncMock, side_effect=resolve_results
        ) as wait_for_mock,
    ):
        assert await arun_workflow_request(
            api_key=promptlayer_api_key, base_url=base_url, throw_on_error=throw_on_error, **kwargs
        ) == {
            "Node 2": "False",
            "Node 3": "AAA",
        }
        assert [(request.method, request.uri) for request in cassette.requests] == [
            ("GET", "http://localhost:8000/workflows/analyze_1"),
            (
                "POST",
                (
                    "http://localhost:8000/ws-token-request-library?"
                    "capability=workflows%3A3%3Arun%3A8dd7e4d404754c60a50e78f70f74aade"
                ),
            ),
            ("POST", "http://localhost:8000/workflows/3/run"),
        ]

    _make_channel_name_suffix_mock.assert_called_once()
    if not is_recording:
        centrifugo_client_mock.assert_called_once_with("ws://localhost:8000/connection/websocket", Any(type_=str))
        centrifugo_subscription_mock.assert_called_once_with(
            client,
            "workflows:3:run:8dd7e4d404754c60a50e78f70f74aade",
            Any(),
        )
        wait_for_mock.assert_awaited_once_with(Any(type_=asyncio.Future), 3600)
