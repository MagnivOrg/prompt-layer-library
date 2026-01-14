"""Tests for promptlayer.utils module - session management and retry logic."""

import threading

import requests
import urllib3

from promptlayer import utils
from promptlayer.utils import _get_requests_session, should_retry_error


class TestGetRequestsSession:
    """Tests for _get_requests_session function."""

    def setup_method(self):
        """Reset the module-level session before each test."""
        utils._requests_session = None

    def test_creates_session_on_first_call(self):
        """Session should be created on first call."""
        assert utils._requests_session is None
        session = _get_requests_session()
        assert session is not None
        assert isinstance(session, requests.Session)

    def test_returns_same_session_on_subsequent_calls(self):
        """Same session should be returned on subsequent calls."""
        session1 = _get_requests_session()
        session2 = _get_requests_session()
        assert session1 is session2

    def test_session_is_stored_in_module(self):
        """Session should be stored in module-level variable."""
        session = _get_requests_session()
        assert utils._requests_session is session

    def test_session_has_default_pool_size(self):
        """Session should have HTTPAdapter with default pool size of 100."""
        session = _get_requests_session()
        # Get the adapter for https
        adapter = session.get_adapter("https://api.promptlayer.com")
        assert isinstance(adapter, requests.adapters.HTTPAdapter)
        # Verify default pool configuration
        assert adapter._pool_connections == 100
        assert adapter._pool_maxsize == 100

    def test_session_pool_size_configurable_via_env(self, monkeypatch):
        """Pool size should be configurable via environment variables."""
        utils._requests_session = None
        monkeypatch.setenv("PROMPTLAYER_POOL_CONNECTIONS", "50")
        monkeypatch.setenv("PROMPTLAYER_POOL_MAXSIZE", "75")

        session = _get_requests_session()
        adapter = session.get_adapter("https://api.promptlayer.com")
        assert adapter._pool_connections == 50
        assert adapter._pool_maxsize == 75

    def test_thread_safety_single_session_created(self):
        """Only one session should be created even with concurrent access."""
        utils._requests_session = None
        sessions = []
        errors = []

        def get_session():
            try:
                session = _get_requests_session()
                sessions.append(session)
            except Exception as e:
                errors.append(e)

        # Create 50 threads all trying to get the session at once
        threads = [threading.Thread(target=get_session) for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(sessions) == 50
        # All sessions should be the same instance
        assert all(s is sessions[0] for s in sessions)


class TestShouldRetryError:
    """Tests for should_retry_error function."""

    def test_retries_on_connection_error(self):
        """Should retry on requests.exceptions.ConnectionError."""
        error = requests.exceptions.ConnectionError("Connection refused")
        assert should_retry_error(error) is True

    def test_retries_on_timeout(self):
        """Should retry on requests.exceptions.Timeout."""
        error = requests.exceptions.Timeout("Request timed out")
        assert should_retry_error(error) is True

    def test_retries_on_new_connection_error(self):
        """Should retry on urllib3.exceptions.NewConnectionError."""
        error = urllib3.exceptions.NewConnectionError(None, "Failed to establish connection")
        assert should_retry_error(error) is True

    def test_retries_on_max_retry_error(self):
        """Should retry on urllib3.exceptions.MaxRetryError."""
        error = urllib3.exceptions.MaxRetryError(None, "http://test.com", "Max retries exceeded")
        assert should_retry_error(error) is True

    def test_retries_on_500_status(self):
        """Should retry on HTTP 500 errors."""

        class MockResponse:
            status_code = 500

        class MockException(Exception):
            response = MockResponse()

        assert should_retry_error(MockException()) is True

    def test_retries_on_502_status(self):
        """Should retry on HTTP 502 errors."""

        class MockResponse:
            status_code = 502

        class MockException(Exception):
            response = MockResponse()

        assert should_retry_error(MockException()) is True

    def test_retries_on_503_status(self):
        """Should retry on HTTP 503 errors."""

        class MockResponse:
            status_code = 503

        class MockException(Exception):
            response = MockResponse()

        assert should_retry_error(MockException()) is True

    def test_retries_on_429_rate_limit(self):
        """Should retry on HTTP 429 rate limit errors."""

        class MockResponse:
            status_code = 429

        class MockException(Exception):
            response = MockResponse()

        assert should_retry_error(MockException()) is True

    def test_no_retry_on_400_client_error(self):
        """Should NOT retry on HTTP 400 client errors."""

        class MockResponse:
            status_code = 400

        class MockException(Exception):
            response = MockResponse()

        assert should_retry_error(MockException()) is False

    def test_no_retry_on_404_not_found(self):
        """Should NOT retry on HTTP 404 not found errors."""

        class MockResponse:
            status_code = 404

        class MockException(Exception):
            response = MockResponse()

        assert should_retry_error(MockException()) is False

    def test_no_retry_on_generic_exception(self):
        """Should NOT retry on generic exceptions."""
        error = Exception("Something went wrong")
        assert should_retry_error(error) is False

    def test_no_retry_on_value_error(self):
        """Should NOT retry on ValueError."""
        error = ValueError("Invalid value")
        assert should_retry_error(error) is False
