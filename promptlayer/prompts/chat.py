import json

from langchain import prompts

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
    try:
        messages = []
        for message in prompt_dict.get("messages", []):
            role = message.get("role")
            prompt = message.get("prompt", {})
            if not prompt or "_type" not in prompt:
                continue
            prompt_template = prompts.PromptTemplate(
                **{k: v for k, v in prompt.items() if k != "_type"}
            )
            message_fields = {
                k: v for k, v in message.items() if k != "role" and k != "prompt"
            }
            if role == ROLE_SYSTEM:
                message = prompts.SystemMessagePromptTemplate(
                    prompt=prompt_template, **message_fields
                )
            elif role == ROLE_ASSISTANT:
                message = prompts.AIMessagePromptTemplate(
                    prompt=prompt_template, **message_fields
                )
            elif role == ROLE_USER:
                message = prompts.HumanMessagePromptTemplate(
                    prompt=prompt_template, **message_fields
                )
            messages.append(message)
        prompt_template = prompts.ChatPromptTemplate(
            messages=messages,
            input_variables=prompt_dict.get("input_variables", []),
            output_parser=prompt_dict.get("output_parser", None),
            partial_variables=prompt_dict.get("partial_variables", {}),
        )
        return prompt_template
    except Exception as e:
        print("Unknown error occurred. ", e)
        return None
