"""Keep provider-client reuse safe when requests compete with LRU eviction."""

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from cachetools import LRUCache

from promptlayer import utils

# --- Cache isolation and deterministic scheduling helpers ---


class _PauseAfterRelease:
    """Expose a valid eviction interleaving without relying on thread timing."""

    def __init__(self):
        self._lock = threading.Lock()
        self.pause_thread = None
        self.released = threading.Event()
        self.resume = threading.Event()

    def __enter__(self):
        self._lock.acquire()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._lock.release()
        if threading.get_ident() == self.pause_thread:
            self.released.set()
            assert self.resume.wait(timeout=10), "Evicting request did not complete"


class _ObserveMissDuringConstruction(LRUCache):
    """Expose a competing miss before the first factory can finish."""

    def __init__(self, factory_started, competing_miss):
        super().__init__(maxsize=100)
        self.factory_started = factory_started
        self.competing_miss = competing_miss

    def __contains__(self, key):
        present = super().__contains__(key)
        if key == "shared" and not present and self.factory_started.is_set():
            self.competing_miss.set()
        return present


@pytest.fixture
def client_cache(monkeypatch):
    cache = LRUCache(maxsize=100)
    monkeypatch.setattr(utils, "_llm_client_cache", cache)
    monkeypatch.setattr(utils, "_llm_client_cache_lock", threading.Lock())
    return cache


# --- End helpers ---


@pytest.mark.parametrize("capacity", [1, 100])
def test_created_client_survives_concurrent_eviction(monkeypatch, capacity):
    cache = LRUCache(maxsize=capacity)
    lock = _PauseAfterRelease()
    monkeypatch.setattr(utils, "_llm_client_cache", cache)
    monkeypatch.setattr(utils, "_llm_client_cache_lock", lock)
    client = object()

    def first_request():
        lock.pause_thread = threading.get_ident()
        return utils._get_cached_client("first", lambda: client)

    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(first_request)
        try:
            assert lock.released.wait(timeout=10), "Creating request did not release its lock"
            for index in range(capacity):
                utils._get_cached_client(f"other-{index}", object)
            assert "first" not in cache
            assert len(cache) == capacity
        finally:
            lock.resume.set()

        assert pending.result(timeout=10) is client


def test_concurrent_requests_share_one_client(monkeypatch):
    factory_started = threading.Event()
    finish_factory = threading.Event()
    competing_miss = threading.Event()
    cache = _ObserveMissDuringConstruction(factory_started, competing_miss)
    monkeypatch.setattr(utils, "_llm_client_cache", cache)
    monkeypatch.setattr(utils, "_llm_client_cache_lock", threading.Lock())
    created = []

    def factory():
        client = object()
        created.append(client)
        factory_started.set()
        assert finish_factory.wait(timeout=10), "Factory was not released"
        return client

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(utils._get_cached_client, "shared", factory)
        try:
            assert factory_started.wait(timeout=10), "Factory did not start"
            second = executor.submit(utils._get_cached_client, "shared", factory)
            assert competing_miss.wait(timeout=10), "Second request did not observe a miss"
        finally:
            finish_factory.set()
        clients = [first.result(timeout=10), second.result(timeout=10)]

    assert len(created) == 1
    assert all(client is created[0] for client in clients)


def test_cached_client_does_not_wait_for_an_unrelated_factory(client_cache):
    cached = utils._get_cached_client("cached", object)
    factory_started = threading.Event()
    finish_factory = threading.Event()

    def slow_factory():
        factory_started.set()
        assert finish_factory.wait(timeout=10), "Factory was not released"
        return object()

    def unexpected_factory():
        pytest.fail("A cache hit must reuse the existing client")

    with ThreadPoolExecutor(max_workers=2) as executor:
        creating = executor.submit(utils._get_cached_client, "new", slow_factory)
        try:
            assert factory_started.wait(timeout=10), "Factory did not start"
            hit = executor.submit(utils._get_cached_client, "cached", unexpected_factory)
            assert hit.result(timeout=5) is cached
        finally:
            finish_factory.set()
        assert creating.result(timeout=10) is not None


def test_cache_keeps_recent_clients_and_stays_bounded(monkeypatch):
    cache = LRUCache(maxsize=2)
    monkeypatch.setattr(utils, "_llm_client_cache", cache)
    monkeypatch.setattr(utils, "_llm_client_cache_lock", threading.Lock())
    first = utils._get_cached_client("first", object)
    utils._get_cached_client("second", object)

    def unexpected_factory():
        pytest.fail("A cache hit must reuse the existing client")

    assert utils._get_cached_client("first", unexpected_factory) is first
    utils._get_cached_client("third", object)

    assert set(cache) == {"first", "third"}
    assert utils._get_cached_client("first", unexpected_factory) is first


def test_failed_factory_can_be_retried(client_cache):
    error = RuntimeError("Client configuration failed")

    def failing_factory():
        raise error

    with pytest.raises(RuntimeError) as caught:
        utils._get_cached_client("client", failing_factory)

    assert caught.value is error
    assert "client" not in client_cache
    client = object()
    assert utils._get_cached_client("client", lambda: client) is client
