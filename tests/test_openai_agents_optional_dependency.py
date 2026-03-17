import builtins

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from promptlayer.integrations.openai_agents import instrument_openai_agents


def test_instrument_openai_agents_raises_clear_error_when_agents_dependency_missing(monkeypatch):
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(InMemorySpanExporter()))

    original_import = builtins.__import__

    def raising_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "agents" or name.startswith("agents."):
            raise ModuleNotFoundError("No module named 'agents'")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", raising_import)

    with pytest.raises(ImportError, match="openai-agents is required"):
        instrument_openai_agents(tracer_provider=provider)

    provider.shutdown()
