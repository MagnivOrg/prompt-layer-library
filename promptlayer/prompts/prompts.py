from langchain import PromptTemplate, prompts
from langchain.prompts.loading import load_prompt_from_config
from promptlayer.prompts.chat import CHAT_PROMPTLAYER_LANGCHAIN, to_dict, to_prompt
from promptlayer.utils import (
    get_api_key,
    promptlayer_get_prompt,
    promptlayer_publish_prompt,
    run_prompt_registry,
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


def run_prompt(prompt_name, variables=[], tags=[], version=None, engine="", model=""):
    """
    Get a prompt template from PromptLayer and run it to see their results.
    version: The version of the prompt to get. If not specified, the latest version will be returned.
    """
    api_key = get_api_key()
    if engine == "" or engine is None or model == "" or model is None:
        return "Error: Engine and model values are required."
    # Get the current time in seconds since the epoch
    run_prompt_registry(prompt_name, version, tags, variables, engine, model, api_key)


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
