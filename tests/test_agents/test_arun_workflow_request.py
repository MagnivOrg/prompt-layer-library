import asyncio
from contextlib import nullcontext
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from ably.realtime.realtime_channel import RealtimeChannel
from pytest_parametrize_cases import Case, parametrize_cases

from promptlayer.utils import arun_workflow_request
from tests.utils.mocks import Any
from tests.utils.vcr import assert_played, is_cassette_recording


@patch("promptlayer.utils.WS_TOKEN_REQUEST_LIBRARY_URL", "http://localhost:8000/ws-token-request-library")
@parametrize_cases(
    Case("Regular call", kwargs={"workflow_id_or_name": "analyze_1", "input_variables": {"var1": "value1"}}),
    Case("Legacy call", kwargs={"workflow_name": "analyze_1", "input_variables": {"var1": "value1"}}),
)
@pytest.mark.asyncio
async def test_arun_workflow_request(base_url: str, promptlayer_api_key, kwargs):
    is_recording = is_cassette_recording()
    results_future = MagicMock()
    message_listener = MagicMock()
    with (
        assert_played("test_arun_workflow_request.yaml") as cassette,
        patch(
            "promptlayer.utils._make_channel_name_suffix", return_value="8dd7e4d404754c60a50e78f70f74aade"
        ) as _make_channel_name_suffix_mock,
        nullcontext()
        if is_recording
        else patch(
            "promptlayer.utils._subscribe_to_workflow_completion_channel",
            return_value=(results_future, message_listener),
        ) as _subscribe_to_workflow_completion_channel_mock,
        nullcontext()
        if is_recording
        else patch(
            "promptlayer.utils._wait_for_workflow_completion",
            new_callable=AsyncMock,
            return_value={"Node 2": "False", "Node 3": "AAA"},
        ) as _wait_for_workflow_completion_mock,
    ):
        assert await arun_workflow_request(api_key=promptlayer_api_key, base_url=base_url, **kwargs) == {
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
        _subscribe_to_workflow_completion_channel_mock.assert_awaited_once_with(
            base_url, Any(type_=RealtimeChannel), Any(type_=asyncio.Future), False, {"X-API-KEY": promptlayer_api_key}
        )
        _wait_for_workflow_completion_mock.assert_awaited_once_with(
            Any(type_=RealtimeChannel), results_future, message_listener, 3600
        )
