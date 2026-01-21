from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from promptlayer.utils import aget_prompt_template, get_prompt_template
from tests.utils.vcr import assert_played


class TestUrlEncodingInGetPromptTemplate:
    """Tests for URL encoding of prompt names in get_prompt_template and aget_prompt_template."""

    def test_sync_get_prompt_template_encodes_slashes(self, promptlayer_api_key, base_url):
        """Prompt names with slashes should be URL-encoded."""
        prompt_name = "feature1/resolve_problem_2"
        expected_encoded = "feature1%2Fresolve_problem_2"

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"id": 1, "prompt_name": prompt_name}

        with patch("promptlayer.utils._get_requests_session") as mock_session:
            mock_session.return_value.post.return_value = mock_response

            get_prompt_template(
                api_key=promptlayer_api_key,
                base_url=base_url,
                throw_on_error=True,
                prompt_name=prompt_name,
            )

            # Verify the URL was called with the encoded prompt name
            call_args = mock_session.return_value.post.call_args
            actual_url = call_args[0][0]
            assert expected_encoded in actual_url, f"Expected {expected_encoded} in URL, got {actual_url}"
            assert "/" + prompt_name not in actual_url, "Unencoded slash found in URL"

    def test_sync_get_prompt_template_encodes_colons(self, promptlayer_api_key, base_url):
        """Prompt names with colons should be URL-encoded."""
        prompt_name = "namespace:template:v1"
        expected_encoded = "namespace%3Atemplate%3Av1"

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"id": 1, "prompt_name": prompt_name}

        with patch("promptlayer.utils._get_requests_session") as mock_session:
            mock_session.return_value.post.return_value = mock_response

            get_prompt_template(
                api_key=promptlayer_api_key,
                base_url=base_url,
                throw_on_error=True,
                prompt_name=prompt_name,
            )

            call_args = mock_session.return_value.post.call_args
            actual_url = call_args[0][0]
            assert expected_encoded in actual_url, f"Expected {expected_encoded} in URL, got {actual_url}"

    def test_sync_get_prompt_template_encodes_spaces(self, promptlayer_api_key, base_url):
        """Prompt names with spaces should be URL-encoded."""
        prompt_name = "my prompt template"
        expected_encoded = "my%20prompt%20template"

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"id": 1, "prompt_name": prompt_name}

        with patch("promptlayer.utils._get_requests_session") as mock_session:
            mock_session.return_value.post.return_value = mock_response

            get_prompt_template(
                api_key=promptlayer_api_key,
                base_url=base_url,
                throw_on_error=True,
                prompt_name=prompt_name,
            )

            call_args = mock_session.return_value.post.call_args
            actual_url = call_args[0][0]
            assert expected_encoded in actual_url, f"Expected {expected_encoded} in URL, got {actual_url}"

    def test_sync_get_prompt_template_encodes_special_chars(self, promptlayer_api_key, base_url):
        """Prompt names with various special characters should be URL-encoded."""
        prompt_name = "test/prompt:name@v1#latest"
        # URL encoding: / -> %2F, : -> %3A, @ -> %40, # -> %23

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"id": 1, "prompt_name": prompt_name}

        with patch("promptlayer.utils._get_requests_session") as mock_session:
            mock_session.return_value.post.return_value = mock_response

            get_prompt_template(
                api_key=promptlayer_api_key,
                base_url=base_url,
                throw_on_error=True,
                prompt_name=prompt_name,
            )

            call_args = mock_session.return_value.post.call_args
            actual_url = call_args[0][0]
            # Verify special characters are encoded
            assert "%2F" in actual_url, "Slash not encoded"
            assert "%3A" in actual_url, "Colon not encoded"
            assert "%40" in actual_url, "At sign not encoded"
            assert "%23" in actual_url, "Hash not encoded"

    @pytest.mark.asyncio
    async def test_async_get_prompt_template_encodes_slashes(self, promptlayer_api_key, base_url):
        """Async: Prompt names with slashes should be URL-encoded."""
        prompt_name = "feature1/resolve_problem_2"
        expected_encoded = "feature1%2Fresolve_problem_2"

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"id": 1, "prompt_name": prompt_name}

        with patch("promptlayer.utils._make_httpx_client") as mock_client_factory:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_factory.return_value = mock_client

            await aget_prompt_template(
                api_key=promptlayer_api_key,
                base_url=base_url,
                throw_on_error=True,
                prompt_name=prompt_name,
            )

            call_args = mock_client.post.call_args
            actual_url = call_args[0][0]
            assert expected_encoded in actual_url, f"Expected {expected_encoded} in URL, got {actual_url}"
            assert "/" + prompt_name not in actual_url, "Unencoded slash found in URL"

    @pytest.mark.asyncio
    async def test_async_get_prompt_template_encodes_colons(self, promptlayer_api_key, base_url):
        """Async: Prompt names with colons should be URL-encoded."""
        prompt_name = "namespace:template:v1"
        expected_encoded = "namespace%3Atemplate%3Av1"

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"id": 1, "prompt_name": prompt_name}

        with patch("promptlayer.utils._make_httpx_client") as mock_client_factory:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_factory.return_value = mock_client

            await aget_prompt_template(
                api_key=promptlayer_api_key,
                base_url=base_url,
                throw_on_error=True,
                prompt_name=prompt_name,
            )

            call_args = mock_client.post.call_args
            actual_url = call_args[0][0]
            assert expected_encoded in actual_url, f"Expected {expected_encoded} in URL, got {actual_url}"

    @pytest.mark.asyncio
    async def test_async_get_prompt_template_encodes_special_chars(self, promptlayer_api_key, base_url):
        """Async: Prompt names with various special characters should be URL-encoded."""
        prompt_name = "test/prompt:name@v1#latest"

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"id": 1, "prompt_name": prompt_name}

        with patch("promptlayer.utils._make_httpx_client") as mock_client_factory:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_factory.return_value = mock_client

            await aget_prompt_template(
                api_key=promptlayer_api_key,
                base_url=base_url,
                throw_on_error=True,
                prompt_name=prompt_name,
            )

            call_args = mock_client.post.call_args
            actual_url = call_args[0][0]
            # Verify special characters are encoded
            assert "%2F" in actual_url, "Slash not encoded"
            assert "%3A" in actual_url, "Colon not encoded"
            assert "%40" in actual_url, "At sign not encoded"
            assert "%23" in actual_url, "Hash not encoded"


def test_get_prompt_template_provider_base_url_name(capsys, promptlayer_client):
    # TODO(dmu) HIGH: Improve assertions for this test
    provider_base_url_name = "does_not_exist"
    prompt_template = {
        "type": "chat",
        "provider_base_url_name": provider_base_url_name,
        "messages": [
            {
                "content": [{"text": "You are an AI.", "type": "text"}],
                "input_variables": [],
                "name": None,
                "raw_request_display_role": "",
                "role": "system",
                "template_format": "f-string",
            },
            {
                "content": [{"text": "What is the capital of Japan?", "type": "text"}],
                "input_variables": [],
                "name": None,
                "raw_request_display_role": "",
                "role": "user",
                "template_format": "f-string",
            },
        ],
    }

    prompt_registry_name = "test_template:test"
    with assert_played("test_get_prompt_template_provider_base_url_name.yaml"):
        promptlayer_client.templates.publish(
            {
                "provider_base_url_name": provider_base_url_name,
                "prompt_name": prompt_registry_name,
                "prompt_template": prompt_template,
            }
        )
        response = promptlayer_client.templates.get(
            prompt_registry_name, {"provider": "openai", "model": "gpt-3.5-turbo"}
        )
        assert response["provider_base_url"] is None
