from langchain import PromptTemplate, prompts
from langchain.prompts.loading import load_prompt_from_config

from promptlayer.prompts.chat import CHAT_PROMPTLAYER_LANGCHAIN, to_dict, to_prompt
from promptlayer.resources.prompt import Prompt
from promptlayer.utils import (
    get_api_key,
    promptlayer_get_prompt,
    promptlayer_publish_prompt,
)


def get_prompt(prompt_name, langchain=False, version: int = None, label: str = None):
    """
    Get a prompt template from PromptLayer.
    prompt_name: the prompt name
    langchain: Enable this for langchain compatible prompt
    version: The version of the prompt to get. If not specified, the latest version will be returned.
    label: The specific label of a prompt you want to get. Setting this will supercede version
    """
    api_key = get_api_key()
    prompt = promptlayer_get_prompt(prompt_name, api_key, version, label)
    if langchain:
        if "_type" not in prompt["prompt_template"]:
            prompt["prompt_template"]["_type"] = "prompt"
        elif prompt["prompt_template"]["_type"] == CHAT_PROMPTLAYER_LANGCHAIN:
            return to_prompt(prompt["prompt_template"])
        return load_prompt_from_config(prompt["prompt_template"])
    else:
        return prompt["prompt_template"]


def publish_prompt(prompt_name, tags=[], commit_message=None, prompt_template=None):
    api_key = get_api_key()
    if len(commit_message) > 72:
        raise Exception("Commit message must be less than 72 characters.")
    if type(prompt_template) == dict:
        promptlayer_publish_prompt(
            prompt_name, prompt_template, commit_message, tags, api_key
        )
    elif isinstance(prompt_template, prompts.ChatPromptTemplate):
        prompt_template_dict = to_dict(prompt_template)
        promptlayer_publish_prompt(
            prompt_name, prompt_template_dict, commit_message, tags, api_key
        )
    elif isinstance(prompt_template, PromptTemplate):
        promptlayer_publish_prompt(
            prompt_name, prompt_template.dict(), commit_message, tags, api_key
        )
    else:
        raise Exception(
            "Please provide either a JSON prompt template or a langchain prompt template."
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
    try:
        response = Prompt.list({"page": page, "per_page": per_page})
        return response["items"]
    except Exception as e:
        print(e)
        return []
