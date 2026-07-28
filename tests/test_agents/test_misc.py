import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from promptlayer.utils import WorkflowSubscriptionEventHandler, _get_final_output, _make_message_listener
from tests.utils.vcr import assert_played


@pytest.mark.asyncio
async def test_get_final_output(base_url: str, headers):
    with assert_played("test_get_final_output_1.yaml"):
        assert (await _get_final_output(base_url, 717, True, headers=headers)) == {
            "Node 1": {
                "status": "SUCCESS",
                "value": "AAA",
                "error_message": None,
                "raw_error_message": None,
                "is_output_node": True,
            }
        }

    with assert_played("test_get_final_output_2.yaml"):
        assert (await _get_final_output(base_url, 717, False, headers=headers)) == "AAA"


@pytest.mark.asyncio
async def test_make_message_listener(
    base_url: str,
    headers,
    workflow_update_data_no_result_code,
    workflow_update_data_ok,
    workflow_update_data_exceeds_size_limit,
):
    results_future = asyncio.Future()
    execution_id_future = asyncio.Future()
    message_listener = _make_message_listener(base_url, results_future, execution_id_future, True, headers)
    execution_id_future.set_result(717)
    await message_listener("INVALID", "")
    assert not results_future.done()
    assert execution_id_future.done()

    # Final output is in the message
    for message_data in (workflow_update_data_no_result_code, workflow_update_data_ok):
        results_future = asyncio.Future()
        execution_id_future = asyncio.Future()
        execution_id_future.set_result(717)
        message_listener = _make_message_listener(base_url, results_future, execution_id_future, True, headers)
        await message_listener("SET_WORKFLOW_COMPLETE", json.dumps(message_data))
        assert results_future.done()
        assert (await asyncio.wait_for(results_future, 0.1)) == message_data["final_output"]

    # Final output is not in the message (return all outputs)
    with assert_played("test_make_message_listener_1.yaml"):
        results_future = asyncio.Future()
        execution_id_future = asyncio.Future()
        execution_id_future.set_result(717)
        message_listener = _make_message_listener(base_url, results_future, execution_id_future, True, headers)
        await message_listener("SET_WORKFLOW_COMPLETE", json.dumps(workflow_update_data_exceeds_size_limit))
        assert results_future.done()
        assert (await asyncio.wait_for(results_future, 0.1)) == {
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
        results_future = asyncio.Future()
        execution_id_future = asyncio.Future()
        execution_id_future.set_result(717)
        message_listener = _make_message_listener(base_url, results_future, execution_id_future, False, headers)
        await message_listener("SET_WORKFLOW_COMPLETE", json.dumps(workflow_update_data_exceeds_size_limit))
        assert results_future.done()
        assert (await asyncio.wait_for(results_future, 0.1)) == "AAA"


@pytest.mark.asyncio
async def test_workflow_subscription_event_handler():
    callback = AsyncMock()
    handler = WorkflowSubscriptionEventHandler(callback)

    await handler.on_publication(
        SimpleNamespace(pub=SimpleNamespace(data={"message_name": "SET_WORKFLOW_COMPLETE", "data": "payload"}))
    )

    callback.assert_awaited_once_with("SET_WORKFLOW_COMPLETE", "payload")
