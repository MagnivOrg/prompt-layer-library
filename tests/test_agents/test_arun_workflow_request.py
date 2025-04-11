from unittest.mock import AsyncMock, patch

import pytest

from promptlayer.utils import arun_workflow_request
from tests.utils.vcr import assert_played


@patch("promptlayer.utils.WORKFLOWS_RUN_URL", "http://localhost:8000/workflows/{}/run")
@patch("promptlayer.utils.WS_TOKEN_REQUEST_LIBRARY_URL", "http://localhost:8000/ws-token-request-library")
@pytest.mark.asyncio
async def test_arun_workflow_request(promptlayer_api_key):
    with (
        assert_played("test_arun_workflow_request.yaml"),
        patch(
            "promptlayer.utils._wait_for_workflow_completion", new_callable=AsyncMock, return_value="AAA"
        ) as _wait_for_workflow_completion_mock,
    ):
        assert (
            await arun_workflow_request(
                workflow_name="analyze_1", input_variables={"var1": "value1"}, api_key=promptlayer_api_key
            )
        ) == "AAA"

    execution_id = 722
    _wait_for_workflow_completion_mock.assert_called_once_with(
        token="uV0vXQ.sanitized",
        channel_name=f"workflow_updates:{execution_id}",
        execution_id=execution_id,
        return_all_outputs=False,
        headers={"X-API-KEY": promptlayer_api_key},
        timeout=3600,
    )
