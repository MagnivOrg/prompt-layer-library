import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from ably.types.message import Message

from promptlayer.utils import _get_final_output, _make_message_listener, _wait_for_workflow_completion
from tests.utils.vcr import assert_played


@pytest.mark.asyncio
async def test_get_final_output(headers):
    with assert_played("test_get_final_output_1.yaml"):
        assert (await _get_final_output(717, True, headers=headers)) == {
            "Node 1": {
                "status": "SUCCESS",
                "value": "AAA",
                "error_message": None,
                "raw_error_message": None,
                "is_output_node": True,
            }
        }

    with assert_played("test_get_final_output_2.yaml"):
        assert (await _get_final_output(717, False, headers=headers)) == "AAA"


@pytest.mark.asyncio
async def test_make_message_listener(
    headers, workflow_update_data_no_result_code, workflow_update_data_ok, workflow_update_data_exceeds_size_limit
):
    future = asyncio.Future()
    message_listener = _make_message_listener(future, 717, True, headers)
    await message_listener(Message(name="INVALID"))
    assert not future.done()

    # Final output is in the message
    for message_data in (workflow_update_data_no_result_code, workflow_update_data_ok):
        future = asyncio.Future()
        message_listener = _make_message_listener(future, 717, True, headers)
        await message_listener(Message(name="SET_WORKFLOW_COMPLETE", data=json.dumps(message_data)))
        assert future.done()
        assert (await asyncio.wait_for(future, 0.1)) == message_data["final_output"]

    # Final output is not in the message (return all outputs)
    with assert_played("test_make_message_listener_1.yaml"):
        future = asyncio.Future()
        message_listener = _make_message_listener(future, 717, True, headers)
        await message_listener(
            Message(name="SET_WORKFLOW_COMPLETE", data=json.dumps(workflow_update_data_exceeds_size_limit))
        )
        assert future.done()
        assert (await asyncio.wait_for(future, 0.1)) == {
            "Node 1": {
                "status": "SUCCESS",
                "value": "AAA",
                "error_message": None,
                "raw_error_message": None,
                "is_output_node": True,
            }
        }

    # Final output is not in the message (return final output)
    with assert_played("test_make_message_listener_2.yaml"):
        future = asyncio.Future()
        message_listener = _make_message_listener(future, 717, False, headers)
        await message_listener(
            Message(name="SET_WORKFLOW_COMPLETE", data=json.dumps(workflow_update_data_exceeds_size_limit))
        )
        assert future.done()
        assert (await asyncio.wait_for(future, 0.1)) == "AAA"


@pytest.mark.asyncio
async def test_wait_for_workflow_completion(headers, workflow_update_data_ok):
    with patch("promptlayer.utils.AblyRealtime") as MockAbly:
        MockAbly.return_value = mock_client = MagicMock()
        mock_client.channels.get.return_value = mock_channel = MagicMock()
        mock_client.close = AsyncMock()

        captured_listener = None

        async def async_subscribe_mock(event_name, listener):
            nonlocal captured_listener
            captured_listener = listener

        mock_channel.subscribe.side_effect = async_subscribe_mock

        corotine = _wait_for_workflow_completion(
            token="uV0vXQ.sanitized",
            channel_name="workflow_updates:717",
            execution_id=717,
            return_all_outputs=True,
            headers=headers,
            timeout=0.1,
        )
        task = asyncio.create_task(corotine)
        await asyncio.sleep(0.1)
        await captured_listener(Message(name="SET_WORKFLOW_COMPLETE", data=json.dumps(workflow_update_data_ok)))
        assert (await task) == workflow_update_data_ok["final_output"]

        mock_channel.unsubscribe.assert_called_once()
        mock_client.close.assert_awaited_once()
