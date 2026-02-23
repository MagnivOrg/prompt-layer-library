KNOWN_PROVIDER_MODULES = ("openai", "anthropic", "google", "mistralai", "boto")


def _is_provider_exception(exception: Exception) -> bool:
    module = getattr(type(exception), "__module__", "")
    return any(module.startswith(provider) for provider in KNOWN_PROVIDER_MODULES)


def categorize_error(exception: Exception) -> str:
    class_name = type(exception).__name__
    status_code = getattr(exception, "status_code", None)

    if "RateLimitError" in class_name or status_code == 429:
        return "PROVIDER_RATE_LIMIT"

    if status_code == 402:
        return "PROVIDER_QUOTA_LIMIT"

    if "TimeoutError" in class_name:
        return "PROVIDER_TIMEOUT"

    if status_code == 401 or "AuthenticationError" in class_name:
        return "PROVIDER_AUTH_ERROR"

    if _is_provider_exception(exception):
        message = str(exception).lower()
        if "quota" in message:
            return "PROVIDER_QUOTA_LIMIT"
        if "timeout" in message:
            return "PROVIDER_TIMEOUT"
        return "PROVIDER_ERROR"

    return "UNKNOWN_ERROR"
