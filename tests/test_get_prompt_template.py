import datetime
import os

from promptlayer import PromptLayer


def test_get_prompt_template_provider_base_url_name(capsys):
    promptlayer = PromptLayer(api_key=os.environ.get("PROMPTLAYER_API_KEY"))

    prompt_registry_name = f"test_template:{datetime.datetime.now()}"
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

    promptlayer.templates.publish({
        "provider_base_url_name": provider_base_url_name,
        "prompt_name": prompt_registry_name,
        "prompt_template": prompt_template,
    })

    get_response = promptlayer.templates.get(
        prompt_registry_name, {"provider": "openai", "model": "gpt-3.5-turbo"}
    )

    assert get_response['provider_base_url'] is None
