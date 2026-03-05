"""Tests for set_prompt_span_attributes in promptlayer.span_exporter."""

from unittest.mock import MagicMock, patch

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider

from promptlayer.span_exporter import set_prompt_span_attributes


class TestSetPromptSpanAttributes:
    def test_sets_all_attributes_on_recording_span(self):
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True

        blueprint = {"id": 42, "version": 3}

        with patch("promptlayer.span_exporter.trace") as mock_trace, patch(
            "promptlayer.span_exporter.baggage"
        ) as mock_baggage, patch("promptlayer.span_exporter.context") as mock_context:
            mock_trace.get_current_span.return_value = mock_span
            mock_context.get_current.return_value = {}
            mock_baggage.set_baggage.return_value = {}
            mock_baggage.remove_baggage.return_value = {}
            set_prompt_span_attributes(blueprint, "my-chatbot", label="production")

        mock_span.set_attribute.assert_any_call("promptlayer.prompt.name", "my-chatbot")
        mock_span.set_attribute.assert_any_call("promptlayer.prompt.id", "42")
        mock_span.set_attribute.assert_any_call("promptlayer.prompt.version", "3")
        mock_span.set_attribute.assert_any_call("promptlayer.prompt.label", "production")

    def test_skips_label_when_none(self):
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True

        blueprint = {"id": 1, "version": 1}

        with patch("promptlayer.span_exporter.trace") as mock_trace, patch(
            "promptlayer.span_exporter.baggage"
        ) as mock_baggage, patch("promptlayer.span_exporter.context") as mock_context:
            mock_trace.get_current_span.return_value = mock_span
            mock_context.get_current.return_value = {}
            mock_baggage.set_baggage.return_value = {}
            mock_baggage.remove_baggage.return_value = {}
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

        with patch("promptlayer.span_exporter.trace") as mock_trace, patch(
            "promptlayer.span_exporter.baggage"
        ) as mock_baggage, patch("promptlayer.span_exporter.context") as mock_context:
            mock_trace.get_current_span.return_value = mock_span
            mock_context.get_current.return_value = {}
            mock_baggage.set_baggage.return_value = {}
            mock_baggage.remove_baggage.return_value = {}
            set_prompt_span_attributes({}, "my-prompt")

        mock_span.set_attribute.assert_called_once_with("promptlayer.prompt.name", "my-prompt")

    def test_baggage_propagates_to_child_spans(self):
        """Integration test: baggage set by set_prompt_span_attributes
        is copied onto child spans via BaggageSpanProcessor."""
        from opentelemetry.processor.baggage import ALLOW_ALL_BAGGAGE_KEYS, BaggageSpanProcessor

        provider = TracerProvider()
        provider.add_span_processor(BaggageSpanProcessor(ALLOW_ALL_BAGGAGE_KEYS))
        trace.set_tracer_provider(provider)
        tracer = trace.get_tracer("test")

        try:
            with tracer.start_as_current_span("parent"):
                set_prompt_span_attributes({"id": 7, "version": 2}, "my-prompt", label="staging")
                with tracer.start_as_current_span("child") as child_span:
                    pass

            attrs = dict(child_span.attributes)
            assert attrs["promptlayer.prompt.name"] == "my-prompt"
            assert attrs["promptlayer.prompt.id"] == "7"
            assert attrs["promptlayer.prompt.version"] == "2"
            assert attrs["promptlayer.prompt.label"] == "staging"
        finally:
            provider.shutdown()

    def test_baggage_clears_stale_keys(self):
        """When a second prompt has no label, the stale label baggage is removed."""
        from opentelemetry.processor.baggage import ALLOW_ALL_BAGGAGE_KEYS, BaggageSpanProcessor

        provider = TracerProvider()
        provider.add_span_processor(BaggageSpanProcessor(ALLOW_ALL_BAGGAGE_KEYS))
        trace.set_tracer_provider(provider)
        tracer = trace.get_tracer("test")

        try:
            with tracer.start_as_current_span("parent"):
                # First prompt with label
                set_prompt_span_attributes({"id": 1, "version": 1}, "prompt-a", label="prod")
                # Second prompt without label — should clear stale label
                set_prompt_span_attributes({"id": 2, "version": 3}, "prompt-b")
                with tracer.start_as_current_span("child") as child_span:
                    pass

            attrs = dict(child_span.attributes)
            assert attrs["promptlayer.prompt.name"] == "prompt-b"
            assert attrs["promptlayer.prompt.id"] == "2"
            assert attrs["promptlayer.prompt.version"] == "3"
            assert "promptlayer.prompt.label" not in attrs
        finally:
            provider.shutdown()
