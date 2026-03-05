"""Tests for set_prompt_span_attributes in promptlayer.span_exporter."""

from unittest.mock import MagicMock, patch

from promptlayer.span_exporter import set_prompt_span_attributes


class TestSetPromptSpanAttributes:
    def test_sets_all_attributes_on_recording_span(self):
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True

        blueprint = {"id": 42, "version": 3}

        with patch("promptlayer.span_exporter.trace") as mock_trace:
            mock_trace.get_current_span.return_value = mock_span
            set_prompt_span_attributes(blueprint, "my-chatbot", label="production")

        mock_span.set_attribute.assert_any_call("promptlayer.prompt.name", "my-chatbot")
        mock_span.set_attribute.assert_any_call("promptlayer.prompt.id", 42)
        mock_span.set_attribute.assert_any_call("promptlayer.prompt.version", 3)
        mock_span.set_attribute.assert_any_call("promptlayer.prompt.label", "production")

    def test_skips_label_when_none(self):
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True

        blueprint = {"id": 1, "version": 1}

        with patch("promptlayer.span_exporter.trace") as mock_trace:
            mock_trace.get_current_span.return_value = mock_span
            set_prompt_span_attributes(blueprint, "tpl")

        calls = [c[0] for c in mock_span.set_attribute.call_args_list]
        assert not any(key == "promptlayer.prompt.label" for key, _ in calls)

    def test_noop_when_span_not_recording(self):
        mock_span = MagicMock()
        mock_span.is_recording.return_value = False

        with patch("promptlayer.span_exporter.trace") as mock_trace:
            mock_trace.get_current_span.return_value = mock_span
            set_prompt_span_attributes({"id": 1, "version": 1}, "tpl")

        mock_span.set_attribute.assert_not_called()

    def test_handles_missing_id_and_version(self):
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True

        with patch("promptlayer.span_exporter.trace") as mock_trace:
            mock_trace.get_current_span.return_value = mock_span
            set_prompt_span_attributes({}, "my-prompt")

        mock_span.set_attribute.assert_called_once_with("promptlayer.prompt.name", "my-prompt")
