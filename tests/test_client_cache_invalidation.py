from promptlayer import AsyncPromptLayer, PromptLayer


def test_promptlayer_client_invalidate_prompt_name(promptlayer_api_key, base_url):
    client = PromptLayer(api_key=promptlayer_api_key, base_url=base_url, cache_ttl_seconds=60)
    cache = client.templates._cache

    alpha_key = cache.make_key("alpha")
    beta_key = cache.make_key("beta")
    cache.put(alpha_key, {"prompt_template": {"type": "chat", "messages": []}})
    cache.put(beta_key, {"prompt_template": {"type": "chat", "messages": []}})

    client.invalidate("alpha")

    alpha_cached, _ = cache.get(alpha_key)
    beta_cached, _ = cache.get(beta_key)
    assert alpha_cached is None
    assert beta_cached is not None


def test_promptlayer_client_invalidate_all(promptlayer_api_key, base_url):
    client = PromptLayer(api_key=promptlayer_api_key, base_url=base_url, cache_ttl_seconds=60)
    cache = client.templates._cache

    first_key = cache.make_key("first")
    second_key = cache.make_key("second")
    cache.put(first_key, {"prompt_template": {"type": "chat", "messages": []}})
    cache.put(second_key, {"prompt_template": {"type": "chat", "messages": []}})

    client.invalidate()

    first_cached, _ = cache.get(first_key)
    second_cached, _ = cache.get(second_key)
    assert first_cached is None
    assert second_cached is None


def test_async_promptlayer_client_invalidate(promptlayer_api_key, base_url):
    client = AsyncPromptLayer(api_key=promptlayer_api_key, base_url=base_url, cache_ttl_seconds=60)
    cache = client.templates._cache

    key = cache.make_key("async-template")
    cache.put(key, {"prompt_template": {"type": "chat", "messages": []}})

    client.invalidate("async-template")

    cached, _ = cache.get(key)
    assert cached is None
