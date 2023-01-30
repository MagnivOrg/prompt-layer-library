from promptlayer.promptlayer import PromptLayerBase

class PromptLayer(PromptLayerBase):
    def __init__(self, *args, provider_type="langchain", function_name="langchain", **kwargs):
        super().__init__(*args, provider_type=provider_type, function_name=function_name, **kwargs)