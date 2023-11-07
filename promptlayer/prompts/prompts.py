from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.prompts.loading import load_prompt_from_config

from promptlayer.prompts.chat import CHAT_PROMPTLAYER_LANGCHAIN, to_dict, to_prompt
from promptlayer.resources.prompt import Prompt
from promptlayer.utils import (
    get_api_key,
    promptlayer_get_prompt,
    promptlayer_publish_prompt,
)


def get_prompt(
    prompt_name,
    langchain=False,
    version: int = None,
    label: str = None,
    include_metadata: bool = False,
):
    """
    Get a prompt template from PromptLayer.
    prompt_name: the prompt name
    langchain: Enable this for langchain compatible prompt
    version: The version of the prompt to get. If not specified, the latest version will be returned.
    label: The specific label of a prompt you want to get. Setting this will supercede version
    include_metadata: Whether or not to include the metadata of the prompt in the response.
    """
    api_key = get_api_key()
    prompt = promptlayer_get_prompt(prompt_name, api_key, version, label)
    if langchain:
        if "_type" not in prompt["prompt_template"]:
            prompt["prompt_template"]["_type"] = "prompt"
        if prompt["prompt_template"]["_type"] == CHAT_PROMPTLAYER_LANGCHAIN:
            prompt_template = to_prompt(prompt["prompt_template"])
        else:
            prompt_template = load_prompt_from_config(prompt["prompt_template"])
    else:
        prompt_template = prompt["prompt_template"]
    if include_metadata:
        return prompt_template, prompt["metadata"]
    return prompt_template


def publish_prompt(
    prompt_name, tags=[], commit_message=None, prompt_template=None, metadata=None
):
    api_key = get_api_key()
    if commit_message is not None and len(commit_message) > 72:
        raise Exception("Commit message must be less than 72 characters.")
    if isinstance(prompt_template, ChatPromptTemplate):
        prompt_template = to_dict(prompt_template)
    elif isinstance(prompt_template, PromptTemplate):
        prompt_template = prompt_template.dict()
    elif not isinstance(prompt_template, dict):
        raise Exception(
            "Please provide either a JSON prompt template or a langchain prompt template."
        )
    promptlayer_publish_prompt(
        prompt_name, prompt_template, commit_message, tags, api_key, metadata
    )


def all(page: int = 1, per_page: int = 30):
    """
    List all prompts on PromptLayer.

    Parameters:
    ----------
    page: int
        The page of prompts to get.
    per_page: int
        The number of prompts to get per page.

    Returns:
    -------
    list of prompts
    """
    response = Prompt.list({"page": page, "per_page": per_page})
    if not response.get("success", True):
        raise Exception(
            f"Failed to get prompts from PromptLayer. {response.get('message')}"
        )
    return response["items"]
