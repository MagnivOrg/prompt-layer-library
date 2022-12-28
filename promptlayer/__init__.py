from promptlayer.promptlayer import PromptLayer
import openai
import os

api_key = os.environ.get("PROMPTLAYER_API_KEY")
openai = PromptLayer(openai, function_name="openai")


__all__ = [
    "api_key",
    "openai",
]
