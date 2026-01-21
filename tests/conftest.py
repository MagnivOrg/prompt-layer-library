"""Pytest configuration and fixtures for promptlayer tests."""

import pytest

# Import all fixtures from the fixtures module
from tests.fixtures import (
    anthropic_api_key,
    anthropic_async_client,
    anthropic_client,
    autouse_disable_network,
    base_url,
    headers,
    openai_api_key,
    openai_async_client,
    openai_client,
    promptlayer_api_key,
    promptlayer_async_client,
    promptlayer_client,
    sample_template_content,
    sample_template_name,
    throw_on_error,
    workflow_update_data_exceeds_size_limit,
    workflow_update_data_no_result_code,
    workflow_update_data_ok,
)

# Re-export all fixtures so pytest can discover them
__all__ = [
    "anthropic_api_key",
    "anthropic_async_client",
    "anthropic_client",
    "autouse_disable_network",
    "base_url",
    "headers",
    "openai_api_key",
    "openai_async_client",
    "openai_client",
    "promptlayer_api_key",
    "promptlayer_async_client",
    "promptlayer_client",
    "sample_template_content",
    "sample_template_name",
    "throw_on_error",
    "workflow_update_data_exceeds_size_limit",
    "workflow_update_data_no_result_code",
    "workflow_update_data_ok",
]


# Provide the disable_network fixture if pytest-network is not available
# This fixture is used by autouse_disable_network
@pytest.fixture
def disable_network():
    """Dummy disable_network fixture when pytest-network is not available.

    In CI with VCR cassettes, actual network calls are blocked by VCR.
    This fixture ensures tests can run without the pytest-network dependency.
    """
    yield
