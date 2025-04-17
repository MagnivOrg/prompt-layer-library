import pytest


@pytest.fixture
def sample_template_name():
    return "sample_template"


@pytest.fixture
def sample_template_content():
    return {
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
                "content": [{"text": "What is the capital of Japan?", "type": "text"}],
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
