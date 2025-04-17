from tests.utils.vcr import assert_played


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
