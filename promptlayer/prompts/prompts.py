from promptlayer.utils import get_api_key, promptlayer_get_prompt, promptlayer_publish_prompt
from langchain.prompts.loading import load_prompt_from_config

def get_prompt(prompt_name, langchain=False):
    api_key = get_api_key()
    prompt = promptlayer_get_prompt(prompt_name, api_key)
    if langchain:
        return load_prompt_from_config(prompt["prompt_template"])
    else:
        return prompt["prompt_template"]
    
def publish_prompt(prompt_name, tags=[], json=None, langchain=None):
    api_key = get_api_key()
    if json:
        promptlayer_publish_prompt(prompt_name, json, tags, api_key)
    elif langchain:
        promptlayer_publish_prompt(prompt_name, langchain.dict(), tags, api_key)
    else:
        raise Exception("Please provide either a JSON prompt template or a langchain prompt template.")