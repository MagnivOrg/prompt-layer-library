from __future__ import annotations

import os
from typing import Optional, Tuple

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.semconv.resource import ResourceAttributes
from opentelemetry.trace import Tracer

from promptlayer.utils import _PROMPTLAYER_USER_AGENT, SDK_VERSION

DEFAULT_SERVICE_NAME = "prompt-layer-library"


def resolve_otlp_traces_endpoint(*, endpoint: Optional[str] = None, base_url: Optional[str] = None) -> str:
    if endpoint:
        return endpoint

    env_endpoint = os.environ.get("PROMPTLAYER_OTLP_TRACES_ENDPOINT")
    if env_endpoint:
        return env_endpoint

    normalized_base_url = (base_url or os.environ.get("PROMPTLAYER_BASE_URL") or "https://api.promptlayer.com").rstrip(
        "/"
    )
    return f"{normalized_base_url}/v1/traces"


def create_otlp_span_exporter(
    *, api_key: str, endpoint: Optional[str] = None, base_url: Optional[str] = None
) -> OTLPSpanExporter:
    return OTLPSpanExporter(
        endpoint=resolve_otlp_traces_endpoint(endpoint=endpoint, base_url=base_url),
        headers={
            "X-Api-Key": api_key,
            "User-Agent": _PROMPTLAYER_USER_AGENT,
            "X-SDK-Version": SDK_VERSION,
        },
    )


def create_promptlayer_tracer_provider(
    *,
    api_key: str,
    endpoint: Optional[str] = None,
    base_url: Optional[str] = None,
    service_name: str = DEFAULT_SERVICE_NAME,
) -> TracerProvider:
    exporter = create_otlp_span_exporter(api_key=api_key, endpoint=endpoint, base_url=base_url)
    provider = TracerProvider(resource=Resource(attributes={ResourceAttributes.SERVICE_NAME: service_name}))
    provider.add_span_processor(BatchSpanProcessor(exporter))
    return provider


def initialize_promptlayer_tracer(
    *,
    api_key: str,
    base_url: str,
    enable_tracing: bool = False,
    service_name: str = DEFAULT_SERVICE_NAME,
) -> Tuple[Optional[TracerProvider], Optional[Tracer]]:
    if not enable_tracing:
        return None, None

    provider = create_promptlayer_tracer_provider(
        api_key=api_key,
        base_url=base_url,
        service_name=service_name,
    )
    return provider, provider.get_tracer(__name__)
