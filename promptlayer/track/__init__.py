from promptlayer.track.track import group
from promptlayer.track.track import metadata as metadata_
from promptlayer.track.track import prompt
from promptlayer.track.track import score as score_


class TrackManager:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def group(self, request_id, group_id):
        return group(request_id, group_id, self.api_key)

    def metadata(self, request_id, metadata):
        return metadata_(request_id, metadata, self.api_key)

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
        return score_(request_id, score, score_name, self.api_key)


__all__ = ["TrackManager"]
