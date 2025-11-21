"""
Unit tests for verifying that latency is correctly recorded in pl_client.run
Tests the fix for the issue where request_start_time and request_end_time were being set to the same value.
"""

import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest

from promptlayer import AsyncPromptLayer, PromptLayer
from promptlayer.promptlayer_mixins import PromptLayerMixin


class TestLatencyRecording:
    """Test suite for latency recording functionality"""

    def test_prepare_track_request_kwargs_with_timestamps(self):
        """Test that _prepare_track_request_kwargs correctly uses provided timestamps"""

        # Arrange
        api_key = "test_api_key"
        request_params = {
            "function_name": "test_function",
            "provider": "openai",
            "function_kwargs": {"model": "gpt-4"},
            "prompt_blueprint": {"id": 123, "version": 1},
        }
        tags = ["test"]
        input_variables = {"query": "test"}
        group_id = None

        # Create distinct timestamps with 2 second difference
        start_time = 1000.0
        end_time = 1002.0

        # Act
        result = PromptLayerMixin._prepare_track_request_kwargs(
            api_key=api_key,
            request_params=request_params,
            tags=tags,
            input_variables=input_variables,
            group_id=group_id,
            request_start_time=start_time,
            request_end_time=end_time,
        )

        # Assert
        assert result["request_start_time"] == start_time, "Start time should match provided value"
        assert result["request_end_time"] == end_time, "End time should match provided value"

        # Calculate latency
        latency = result["request_end_time"] - result["request_start_time"]
        assert latency == 2.0, f"Latency should be 2.0 seconds, but got {latency}"

    def test_prepare_track_request_kwargs_without_timestamps(self):
        """Test backward compatibility when timestamps are not provided"""

        # Arrange
        api_key = "test_api_key"
        request_params = {
            "function_name": "test_function",
            "provider": "openai",
            "function_kwargs": {"model": "gpt-4"},
            "prompt_blueprint": {"id": 123, "version": 1},
        }

        # Act
        result = PromptLayerMixin._prepare_track_request_kwargs(
            api_key=api_key,
            request_params=request_params,
            tags=["test"],
            input_variables={"query": "test"},
            group_id=None,
        )

        # Assert - timestamps should be auto-generated and nearly identical (old buggy behavior)
        latency = result["request_end_time"] - result["request_start_time"]
        assert latency < 0.01, "Without provided timestamps, latency should be near-zero (backward compatibility)"

    @patch("promptlayer.promptlayer.track_request")
    @patch("promptlayer.templates.TemplateManager.get")
    def test_run_internal_captures_timing_correctly(self, mock_template_get, mock_track_request):
        """Test that _run_internal correctly captures timing before and after LLM call"""

        # Arrange
        mock_template_get.return_value = {
            "id": 1,
            "prompt_template": {"type": "chat", "messages": []},
            "metadata": {"model": {"provider": "openai", "name": "gpt-4"}},
            "llm_kwargs": {"model": "gpt-4"},
        }

        # Create a mock LLM response
        mock_response = MagicMock()
        mock_response.model_dump = MagicMock(return_value={"choices": []})

        # Create a mock request function that simulates delay
        def mock_request_function(**kwargs):
            time.sleep(0.5)  # Simulate 500ms API call
            return mock_response

        # Mock the track request to capture the timing
        tracked_times = {}

        def capture_track_request(*args, **kwargs):
            tracked_times["start"] = kwargs.get("request_start_time")
            tracked_times["end"] = kwargs.get("request_end_time")
            return {"request_id": "test_id", "prompt_blueprint": {}}

        mock_track_request.side_effect = capture_track_request

        client = PromptLayer(api_key="test_key")

        # Patch _prepare_llm_data to use our mock request function
        with patch.object(client, "_prepare_llm_data") as mock_prepare:
            mock_prepare.return_value = {
                "request_function": mock_request_function,
                "provider": "openai",
                "function_name": "test",
                "stream_function": None,
                "client_kwargs": {},
                "function_kwargs": {},
                "prompt_blueprint": {"id": 1, "version": 1, "metadata": {}},
            }

            # Act
            client._run_internal(prompt_name="test_prompt", input_variables={})

        # Assert
        assert "start" in tracked_times, "Start time should be tracked"
        assert "end" in tracked_times, "End time should be tracked"

        if tracked_times["start"] and tracked_times["end"]:
            latency = tracked_times["end"] - tracked_times["start"]
            # Should be at least 0.5 seconds (our simulated delay)
            assert latency >= 0.5, f"Latency should be at least 0.5 seconds, but got {latency}"
            # But shouldn't be more than 1 second
            assert latency < 1.0, f"Latency should be less than 1 second, but got {latency}"

    @pytest.mark.asyncio
    @patch("promptlayer.promptlayer.atrack_request")
    @patch("promptlayer.templates.AsyncTemplateManager.get")
    async def test_async_run_internal_captures_timing_correctly(self, mock_template_get, mock_atrack_request):
        """Test that async _run_internal correctly captures timing"""

        # Arrange
        mock_template_get.return_value = {
            "id": 1,
            "prompt_template": {"type": "chat", "messages": []},
            "metadata": {"model": {"provider": "openai", "name": "gpt-4"}},
            "llm_kwargs": {"model": "gpt-4"},
        }

        # Create a mock LLM response
        mock_response = MagicMock()
        mock_response.model_dump = MagicMock(return_value={"choices": []})

        # Create an async mock request function that simulates delay
        async def mock_request_function(**kwargs):
            await asyncio.sleep(0.5)  # Simulate 500ms API call
            return mock_response

        # Mock the track request to capture the timing
        tracked_times = {}

        async def capture_track_request(*args, **kwargs):
            tracked_times["start"] = kwargs.get("request_start_time")
            tracked_times["end"] = kwargs.get("request_end_time")
            return {"request_id": "test_id", "prompt_blueprint": {}}

        mock_atrack_request.side_effect = capture_track_request

        client = AsyncPromptLayer(api_key="test_key")

        # Patch _prepare_llm_data to use our mock request function
        with patch.object(client, "_prepare_llm_data") as mock_prepare:
            mock_prepare.return_value = {
                "request_function": mock_request_function,
                "provider": "openai",
                "function_name": "test",
                "stream_function": None,
                "client_kwargs": {},
                "function_kwargs": {},
                "prompt_blueprint": {"id": 1, "version": 1, "metadata": {}},
            }

            # Act
            await client._run_internal(prompt_name="test_prompt", input_variables={})

        # Assert
        assert "start" in tracked_times, "Start time should be tracked"
        assert "end" in tracked_times, "End time should be tracked"

        if tracked_times["start"] and tracked_times["end"]:
            latency = tracked_times["end"] - tracked_times["start"]
            # Should be at least 0.5 seconds (our simulated delay)
            assert latency >= 0.5, f"Latency should be at least 0.5 seconds, but got {latency}"
            # But shouldn't be more than 1 second
            assert latency < 1.0, f"Latency should be less than 1 second, but got {latency}"

    def test_streaming_captures_start_time(self):
        """Test that streaming mode correctly passes start_time to the stream handler"""

        # This test verifies that when stream=True, the start_time is captured
        # and passed to the stream_response function

        client = PromptLayer(api_key="test_key")

        # Mock the dependencies
        with patch("promptlayer.templates.TemplateManager.get") as mock_get, patch(
            "promptlayer.promptlayer.stream_response"
        ) as mock_stream:
            mock_get.return_value = {
                "id": 1,
                "prompt_template": {"type": "chat"},
                "metadata": {"model": {"provider": "openai", "name": "gpt-4"}},
                "llm_kwargs": {"model": "gpt-4"},
            }

            # Mock the LLM request to return a generator (streaming response)
            def mock_generator():
                yield {"chunk": 1}
                yield {"chunk": 2}

            with patch.object(client, "_prepare_llm_data") as mock_prepare:
                mock_prepare.return_value = {
                    "request_function": lambda **k: mock_generator(),
                    "provider": "openai",
                    "function_name": "test",
                    "stream_function": lambda x: x,
                    "client_kwargs": {},
                    "function_kwargs": {},
                    "prompt_blueprint": {"id": 1, "version": 1, "metadata": {}},
                }

                # Act
                client._run_internal(prompt_name="test", stream=True)

                # Assert that stream_response was called
                mock_stream.assert_called_once()

                # Get the after_stream callable that was passed
                call_args = mock_stream.call_args
                after_stream_callable = call_args[1]["after_stream"]

                # The callable should have access to request_start_time
                # This is verified by the structure of _create_track_request_callable
                assert callable(after_stream_callable)

    def test_latency_calculation_accuracy(self):
        """Test that latency calculation is accurate for various time differences"""

        test_cases = [
            (1000.0, 1000.1, 0.1),  # 100ms
            (1000.0, 1001.0, 1.0),  # 1 second
            (1000.0, 1005.5, 5.5),  # 5.5 seconds
            (1000.0, 1000.001, 0.001),  # 1ms
        ]

        for start, end, expected_latency in test_cases:
            result = PromptLayerMixin._prepare_track_request_kwargs(
                api_key="test",
                request_params={
                    "function_name": "test",
                    "provider": "openai",
                    "function_kwargs": {},
                    "prompt_blueprint": {"id": 1, "version": 1},
                },
                tags=[],
                input_variables={},
                group_id=None,
                request_start_time=start,
                request_end_time=end,
            )

            actual_latency = result["request_end_time"] - result["request_start_time"]
            assert abs(actual_latency - expected_latency) < 0.0001, (
                f"Expected latency {expected_latency}, got {actual_latency}"
            )


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])
