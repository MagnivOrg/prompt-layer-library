from langchain import PromptTemplate
from langchain.prompts.loading import load_prompt_from_config

from promptlayer.utils import (promptlayer_get_prompt,
                               promptlayer_publish_prompt)


def get_prompt(prompt_name, langchain=False, version=None):
    """
    Get a prompt template from PromptLayer.
    version: The version of the prompt to get. If not specified, the latest version will be returned.
    """
    prompt = promptlayer_get_prompt(prompt_name, version)
    if langchain:
        if "_type" not in prompt["prompt_template"]:
            prompt["prompt_template"]["_type"] = "prompt"
        return load_prompt_from_config(prompt["prompt_template"])
    else:
        return prompt['message']["prompt_template"]


def publish_prompt(prompt_name, prompt_template=None):
    if type(prompt_name) == None:
        raise Exception("Please provide a prompt name")
    else:
        promptlayer_publish_prompt(prompt_name, prompt_template)