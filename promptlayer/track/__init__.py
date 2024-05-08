from promptlayer.track.track import group, metadata, prompt


class TrackManager:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def group(self, group_id: str, input_text: str, output_text: str, model: str):
        return group(group_id, input_text, output_text, model, self.api_key)

    def metadata(self, key: str, value: str):
        return metadata(key, value, self.api_key)

    def prompt(
        self, request_id, prompt_name, prompt_input_variables, version=None, label=None
    ):
        return prompt(
            request_id,
            prompt_name,
            prompt_input_variables,
            version,
            label,
            self.api_key,
        )

    def score(self, request_id, score, score_name=None):
        return score(request_id, score, score_name, self.api_key)


__all__ = ["TrackManager"]
