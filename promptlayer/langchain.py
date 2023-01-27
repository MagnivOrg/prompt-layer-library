from promptlayer.promptlayer import PromptLayerBase

class PromptLayer(PromptLayerBase):
    def __init__(self, *args, provider_type="langchain", **kwargs):
        super().__init__(*args, provider_type=provider_type, **kwargs)