"""
Integration test for error tracking in PromptLayer.run().

Verifies that when an LLM call fails, the SDK logs the error to PromptLayer
via /track-request with status=ERROR and correct error_type before re-raising.

Uses VCR cassettes to record/replay the HTTP interactions:
  1. POST /prompt-templates/<name> — fetch the template (success)
  2. POST https://api.openai.com/v1/chat/completions — LLM call (401 failure)
  3. POST /track-request — error logged to PromptLayer (success)
"""

import os
from unittest.mock import patch

import openai
import pytest

from promptlayer import PromptLayer
from tests.utils.vcr import assert_played


def test_run_logs_error_on_llm_auth_failure(promptlayer_api_key, base_url):
    """When OpenAI rejects the API key, the SDK should:
    1. Log the error to PromptLayer with status=ERROR, error_type=PROVIDER_AUTH_ERROR
    2. Re-raise the original AuthenticationError
    """
    client = PromptLayer(api_key=promptlayer_api_key, base_url=base_url)

    with (
        patch.dict(os.environ, {"OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", "sk-invalid-for-test")}),
        assert_played("test_run_logs_error_on_llm_auth_failure.yaml"),
    ):
        with pytest.raises(openai.AuthenticationError):
            client.run(prompt_name="error_tracking_test", input_variables={})
