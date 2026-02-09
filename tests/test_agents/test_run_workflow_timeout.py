"""Tests for run_workflow timeout parameter and timeout error handling."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from promptlayer import AsyncPromptLayer, PromptLayer
from promptlayer.exceptions import PromptLayerAPITimeoutError


async def _slow_workflow(*args, **kwargs):
    """Simulates a workflow that takes longer than the timeout."""
    await asyncio.sleep(10)


@pytest.mark.asyncio
async def test_async_run_workflow_timeout_raises_promptlayer_timeout_error(
    promptlayer_async_client: AsyncPromptLayer,
    base_url: str,
    promptlayer_api_key: str,
):
    """When workflow exceeds timeout, PromptLayerAPITimeoutError is raised."""
    with patch(
        "promptlayer.promptlayer.arun_workflow_request",
        new_callable=AsyncMock,
        side_effect=_slow_workflow,
    ):
        with pytest.raises(PromptLayerAPITimeoutError) as exc_info:
            await promptlayer_async_client.run_workflow(
                workflow_id_or_name="test_workflow",
                input_variables={},
                timeout=0.01,  # 10ms - will timeout before sleep completes
            )

        assert "timed out" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_async_run_workflow_timeout_applied_at_outer_level(
    promptlayer_async_client: AsyncPromptLayer,
    base_url: str,
    promptlayer_api_key: str,
):
    """Timeout is applied via outer asyncio.wait_for, not passed to arun_workflow_request."""
    with patch(
        "promptlayer.promptlayer.arun_workflow_request",
        new_callable=AsyncMock,
        return_value={"output": "result"},
    ) as mock_arun:
        await promptlayer_async_client.run_workflow(
            workflow_id_or_name="test_workflow",
            input_variables={},
            timeout=120,
        )

        mock_arun.assert_awaited_once()
        # timeout is NOT passed to arun_workflow_request - it has its own internal timeout (3600s)
        # User timeout is enforced by outer asyncio.wait_for wrapper
        call_kwargs = mock_arun.call_args.kwargs
        assert "timeout" not in call_kwargs


@pytest.mark.asyncio
async def test_async_run_workflow_no_timeout_param_uses_default(
    promptlayer_async_client: AsyncPromptLayer,
    base_url: str,
    promptlayer_api_key: str,
):
    """When timeout is not passed, it is not included in arun_workflow_request kwargs."""
    with patch(
        "promptlayer.promptlayer.arun_workflow_request",
        new_callable=AsyncMock,
        return_value={"output": "result"},
    ) as mock_arun:
        await promptlayer_async_client.run_workflow(
            workflow_id_or_name="test_workflow",
            input_variables={},
        )

        mock_arun.assert_awaited_once()
        # timeout should not be in kwargs when not passed (arun_workflow_request uses default)
        call_kwargs = mock_arun.call_args.kwargs
        assert "timeout" not in call_kwargs


def test_sync_run_workflow_timeout_raises_promptlayer_timeout_error(
    promptlayer_client: PromptLayer,
    base_url: str,
    promptlayer_api_key: str,
):
    """Sync run_workflow: when workflow exceeds timeout, PromptLayerAPITimeoutError is raised."""

    async def slow_workflow(*args, **kwargs):
        await asyncio.sleep(10)

    with patch(
        "promptlayer.promptlayer.arun_workflow_request",
        new_callable=AsyncMock,
        side_effect=slow_workflow,
    ):
        with pytest.raises(PromptLayerAPITimeoutError) as exc_info:
            promptlayer_client.run_workflow(
                workflow_id_or_name="test_workflow",
                input_variables={},
                timeout=0.01,  # 10ms - will timeout before sleep completes
            )

        assert "timed out" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_async_run_workflow_inner_timeout_propagates(
    promptlayer_async_client: AsyncPromptLayer,
    base_url: str,
    promptlayer_api_key: str,
):
    """PromptLayerAPITimeoutError from inner arun_workflow_request propagates unchanged."""
    with patch(
        "promptlayer.promptlayer.arun_workflow_request",
        new_callable=AsyncMock,
        side_effect=PromptLayerAPITimeoutError(
            "Workflow execution did not complete properly", response=None, body=None
        ),
    ):
        with pytest.raises(PromptLayerAPITimeoutError) as exc_info:
            await promptlayer_async_client.run_workflow(
                workflow_id_or_name="test_workflow",
                input_variables={},
                timeout=120,
            )

        assert "did not complete properly" in str(exc_info.value)


def test_sync_run_workflow_timeout_applied_at_outer_level(
    promptlayer_client: PromptLayer,
    base_url: str,
    promptlayer_api_key: str,
):
    """Sync run_workflow: timeout applied via outer asyncio.wait_for, not passed to arun_workflow_request."""
    with patch(
        "promptlayer.promptlayer.arun_workflow_request",
        new_callable=AsyncMock,
        return_value={"output": "result"},
    ) as mock_arun:
        promptlayer_client.run_workflow(
            workflow_id_or_name="test_workflow",
            input_variables={},
            timeout=60,
        )

        mock_arun.assert_awaited_once()
        call_kwargs = mock_arun.call_args.kwargs
        assert "timeout" not in call_kwargs
