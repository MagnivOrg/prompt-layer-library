import json

from langchain import prompts
import copy

CHAT_PROMPTLAYER_LANGCHAIN = "chat_promptlayer_langchain"
ROLE_SYSTEM = "system"
ROLE_ASSISTANT = "assistant"
ROLE_USER = "user"


def to_dict(prompt_template: prompts.ChatPromptTemplate):
    prompt_dict = json.loads(prompt_template.json())
    prompt_dict["_type"] = CHAT_PROMPTLAYER_LANGCHAIN
    for index, message in enumerate(prompt_template.messages):
        if isinstance(message, prompts.SystemMessagePromptTemplate):
            prompt_dict["messages"][index]["role"] = ROLE_SYSTEM
        elif isinstance(message, prompts.AIMessagePromptTemplate):
            prompt_dict["messages"][index]["role"] = ROLE_ASSISTANT
        elif isinstance(message, prompts.HumanMessagePromptTemplate):
            prompt_dict["messages"][index]["role"] = ROLE_USER

    return prompt_dict


def to_prompt(prompt_dict: dict):
    prompt_dict_copy = copy.deepcopy(prompt_dict)
    try:
        messages = []
        prompt_dict_copy.pop("_type")
        for message in prompt_dict_copy.pop("messages"):
            role = message.pop("role")
            prompt = message.pop("prompt")
            prompt.pop("_type")
            prompt = prompts.PromptTemplate(**prompt)
            if role == ROLE_SYSTEM:
                message = prompts.SystemMessagePromptTemplate(prompt=prompt, **message)
            elif role == ROLE_ASSISTANT:
                message = prompts.AIMessagePromptTemplate(prompt=prompt, **message)
            elif role == ROLE_USER:
                message = prompts.HumanMessagePromptTemplate(prompt=prompt, **message)
            messages.append(message)
        prompt_template = prompts.ChatPromptTemplate(
            messages=messages, **prompt_dict_copy
        )
        return prompt_template
    except Exception as e:
        print(e)
        return None
