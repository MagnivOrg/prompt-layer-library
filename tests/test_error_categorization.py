import httpx
import openai
import anthropic
import pytest

from promptlayer.track.error_tracking import categorize_error


def _make_response(status_code):
    return httpx.Response(status_code, request=httpx.Request("POST", "https://example.com"))


def _make_request():
    return httpx.Request("POST", "https://example.com")


class QuotaLimitError(Exception):
    """Simple exception with status_code for testing quota limit detection."""

    status_code = 402


@pytest.mark.parametrize(
    "exception,expected",
    [
        # OpenAI
        (openai.RateLimitError(message="rate limited", response=_make_response(429), body=None), "PROVIDER_RATE_LIMIT"),
        (
            openai.AuthenticationError(message="bad key", response=_make_response(401), body=None),
            "PROVIDER_AUTH_ERROR",
        ),
        (openai.APITimeoutError(request=_make_request()), "PROVIDER_TIMEOUT"),
        (openai.BadRequestError(message="bad", response=_make_response(400), body=None), "PROVIDER_ERROR"),
        # Anthropic
        (
            anthropic.RateLimitError(message="rate limited", response=_make_response(429), body=None),
            "PROVIDER_RATE_LIMIT",
        ),
        (
            anthropic.AuthenticationError(message="bad key", response=_make_response(401), body=None),
            "PROVIDER_AUTH_ERROR",
        ),
        (anthropic.APITimeoutError(request=_make_request()), "PROVIDER_TIMEOUT"),
        # Status code based
        (QuotaLimitError("payment required"), "PROVIDER_QUOTA_LIMIT"),
        # Generic
        (ValueError("something broke"), "UNKNOWN_ERROR"),
        (RuntimeError("quota exceeded"), "UNKNOWN_ERROR"),
        (TimeoutError("connection timed out"), "PROVIDER_TIMEOUT"),
    ],
    ids=[
        "openai-rate-limit",
        "openai-auth",
        "openai-timeout",
        "openai-bad-request",
        "anthropic-rate-limit",
        "anthropic-auth",
        "anthropic-timeout",
        "status-code-402-quota",
        "generic-unknown",
        "generic-quota-not-provider",
        "generic-timeout",
    ],
)
def test_categorize_error(exception, expected):
    assert categorize_error(exception) == expected
