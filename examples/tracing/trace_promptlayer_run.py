"""Trace a PromptLayer prompt-template run."""

import os

from promptlayer import PromptLayer

promptlayer = PromptLayer(enable_tracing=True)

try:
    result = promptlayer.run(
        prompt_name=os.environ["PROMPTLAYER_PROMPT_NAME"],
        input_variables={"question": "Explain distributed tracing in one sentence."},
        metadata={"example": "promptlayer.run"},
    )
    print(result["raw_response"])
finally:
    if promptlayer.tracer_provider is not None:
        promptlayer.tracer_provider.force_flush()
