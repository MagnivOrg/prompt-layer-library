from tests.utils.vcr import assert_played


def test_track_and_templates(sample_template_name, promptlayer_client, openai_client):
    # TODO(dmu) HIGH: Improve asserts in this test
    with assert_played("test_track_and_templates.yaml"):
        response = promptlayer_client.templates.get(
            sample_template_name, {"provider": "openai", "model": "gpt-3.5-turbo"}
        )
        assert response == {
            "id": 4,
            "prompt_name": "sample_template",
            "tags": ["test"],
            "workspace_id": 1,
            "commit_message": "test",
            "metadata": {
                "model": {
                    "provider": "openai",
                    "name": "gpt-4o-mini",
                    "parameters": {
                        "frequency_penalty": 0,
                        "max_tokens": 256,
                        "messages": [{"content": "Hello", "role": "system"}],
                        "model": "gpt-4o",
                        "presence_penalty": 0,
                        "seed": 0,
                        "temperature": 1,
                        "top_p": 1,
                    },
                }
            },
            "prompt_template": {
                "messages": [
                    {
                        "input_variables": [],
                        "template_format": "f-string",
                        "content": [{"type": "text", "text": ""}],
                        "raw_request_display_role": "",
                        "dataset_examples": [],
                        "role": "system",
                        "name": None,
                    },
                    {
                        "input_variables": [],
                        "template_format": "f-string",
                        "content": [{"type": "text", "text": "What is the capital of Japan?"}],
                        "raw_request_display_role": "",
                        "dataset_examples": [],
                        "role": "user",
                        "name": None,
                    },
                ],
                "functions": [],
                "tools": None,
                "function_call": "none",
                "tool_choice": None,
                "type": "chat",
                "input_variables": [],
                "dataset_examples": [],
            },
            "llm_kwargs": {
                "messages": [{"content": "Hello", "role": "system"}],
                "model": "gpt-4o",
                "frequency_penalty": 0,
                "max_tokens": 256,
                "presence_penalty": 0,
                "seed": 0,
                "temperature": 1,
                "top_p": 1,
            },
            "provider_base_url": None,
            "version": 1,
            "snippets": [],
            "warning": None,
        }

        llm_kwargs = response["llm_kwargs"].copy()
        llm_kwargs.pop("model", None)
        _, pl_id = openai_client.chat.completions.create(return_pl_id=True, model="gpt-3.5-turbo", **llm_kwargs)
        assert promptlayer_client.track.score(request_id=pl_id, score_name="accuracy", score=10) is not None
        assert promptlayer_client.track.metadata(request_id=pl_id, metadata={"test": "test"})

        group_id = promptlayer_client.group.create()
        assert isinstance(group_id, int)
        assert promptlayer_client.track.group(request_id=pl_id, group_id=group_id)


def test_get_all_templates(promptlayer_client):
    with assert_played("test_get_all_templates.yaml"):
        all_templates = promptlayer_client.templates.all()
    assert isinstance(all_templates, list)
    assert len(all_templates) > 0
