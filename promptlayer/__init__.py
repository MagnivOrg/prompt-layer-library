import os
import sys

import promptlayer.langchain as langchain
import promptlayer.prompts as prompts
import promptlayer.track as track
from promptlayer.promptlayer import PromptLayerBase

api_key = os.environ.get("PROMPTLAYER_API_KEY")

openai = None
try:
    import openai as openai_module

    openai = PromptLayerBase(openai_module, function_name="openai")
except ImportError:
    print(
        "OpenAI module not found. Install with `pip install openai`.", file=sys.stderr
    )
    pass

anthropic = None
try:
    import anthropic as anthropic_module

    anthropic = PromptLayerBase(
        anthropic_module,
        function_name="anthropic",
        provider_type="anthropic",
    )
except ImportError:
    print(
        "Anthropic module not found. Install with `pip install anthropic`.",
        file=sys.stderr,
    )
    pass

__all__ = ["api_key", "openai", "anthropic"]
