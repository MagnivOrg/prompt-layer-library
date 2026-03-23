import json
import os


def write_settings_env(settings_file: str, api_key: str, endpoint: str, debug: str) -> str:
    env_values = {
        "TRACE_TO_PROMPTLAYER": "true",
        "PROMPTLAYER_API_KEY": api_key,
        "PROMPTLAYER_OTLP_ENDPOINT": endpoint,
        "PROMPTLAYER_CC_DEBUG": debug,
    }

    settings = {}
    if os.path.exists(settings_file):
        try:
            with open(settings_file, encoding="utf-8") as f:
                settings = json.load(f)
        except Exception as exc:
            raise SystemExit(f"invalid settings json: {exc}")
        if not isinstance(settings, dict):
            raise SystemExit("invalid settings json: root must be an object")

    current_env = settings.get("env", {})
    if not isinstance(current_env, dict):
        current_env = {}
    current_env.update(env_values)
    settings["env"] = current_env

    os.makedirs(os.path.dirname(settings_file), exist_ok=True)
    with open(settings_file, "w", encoding="utf-8") as f:
        json.dump(settings, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return settings_file
