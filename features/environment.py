import os

import openai
from langchain.chat_models import PromptLayerChatOpenAI
from langchain.llms import PromptLayerOpenAIChat

import promptlayer

promptlayer.api_key = os.environ.get("PROMPTLAYER_API_KEY")
openai = promptlayer.openai
openai.api_key = os.environ.get("OPENAI_API_KEY")


def before_all(context):
    context.openai = openai
    context.langchain_chat_openai = PromptLayerChatOpenAI()
    context.langchain_openai_chat = PromptLayerOpenAIChat()
