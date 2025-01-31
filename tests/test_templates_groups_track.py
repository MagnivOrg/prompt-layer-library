import datetime
import os

from promptlayer import PromptLayer


def test_track_and_templates():
    promptlayer = PromptLayer(api_key=os.environ.get("PROMPTLAYER_API_KEY"))
    OpenAI = promptlayer.openai.OpenAI
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    template_name = f"test_template:{datetime.datetime.now()}"
    template_content = {
        "dataset_examples": [],
        "function_call": "none",
        "functions": [],
        "input_variables": [],
        "messages": [
            {
                "content": [{"text": "", "type": "text"}],
                "dataset_examples": [],
                "input_variables": [],
                "name": None,
                "raw_request_display_role": "",
                "role": "system",
                "template_format": "f-string",
            },
            {
                "content": [{"text": "what is the capital of Japan?", "type": "text"}],
                "dataset_examples": [],
                "input_variables": [],
                "name": None,
                "raw_request_display_role": "",
                "role": "user",
                "template_format": "f-string",
            },
        ],
        "tool_choice": None,
        "tools": None,
        "type": "chat",
    }
    promptlayer.templates.publish({"prompt_name": template_name, "prompt_template": template_content})
    get_response = promptlayer.templates.get(template_name, {"provider": "openai", "model": "gpt-3.5-turbo"})

    assert get_response["prompt_name"] == template_name
    assert get_response["prompt_template"] == template_content
    completion, pl_id = client.chat.completions.create(
        return_pl_id=True, model="gpt-3.5-turbo", **get_response["llm_kwargs"]
    )
    score = promptlayer.track.score(request_id=pl_id, score_name="accuracy", score=10)
    metadata = promptlayer.track.metadata(request_id=pl_id, metadata={"test": "test"})
    assert score is not None
    assert metadata is not None

    group_id = promptlayer.group.create()
    assert isinstance(group_id, int)

    track_group = promptlayer.track.group(request_id=pl_id, group_id=group_id)
    assert track_group is not None


def test_get_all_templates():
    promptlayer = PromptLayer(api_key=os.getenv("PROMPTLAYER_API_KEY"))
    all_templates = promptlayer.templates.all()

    assert isinstance(all_templates, list)
    assert len(all_templates) > 0
