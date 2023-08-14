from langchain import PromptTemplate, prompts
from langchain.prompts.loading import load_prompt_from_config

from promptlayer.prompts.chat import CHAT_PROMPTLAYER_LANGCHAIN, to_dict, to_prompt
from promptlayer.resources.prompt import Prompt
from promptlayer.utils import (
    get_api_key,
    promptlayer_get_prompt,
    promptlayer_publish_prompt,
)


def get_prompt(prompt_name, langchain=False, version=None):
    """
    Get a prompt template from PromptLayer.
    version: The version of the prompt to get. If not specified, the latest version will be returned.
    """
    api_key = get_api_key()
    prompt = promptlayer_get_prompt(prompt_name, api_key, version)
    if langchain:
        if "_type" not in prompt["prompt_template"]:
            prompt["prompt_template"]["_type"] = "prompt"
        elif prompt["prompt_template"]["_type"] == CHAT_PROMPTLAYER_LANGCHAIN:
            return to_prompt(prompt["prompt_template"])
        return load_prompt_from_config(prompt["prompt_template"])
    else:
        return prompt["prompt_template"]


def publish_prompt(prompt_name, tags=[], prompt_template=None):
    api_key = get_api_key()
    if type(prompt_template) == dict:
        promptlayer_publish_prompt(prompt_name, prompt_template, tags, api_key)
    elif isinstance(prompt_template, prompts.ChatPromptTemplate):
        prompt_template_dict = to_dict(prompt_template)
        promptlayer_publish_prompt(prompt_name, prompt_template_dict, tags, api_key)
    elif isinstance(prompt_template, PromptTemplate):
        promptlayer_publish_prompt(prompt_name, prompt_template.dict(), tags, api_key)
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
        # TODO: When the API is updated, this should be changed to return a list of PromptTemplate objects.
        return response["items"]
    except Exception as e:
        print(e)
        return []
