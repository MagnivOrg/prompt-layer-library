from promptlayer.track.track import (
    agroup,
    ametadata,
    aprompt,
    ascore,
    group,
    metadata as metadata_,
    prompt,
    score as score_,
)

# TODO(dmu) LOW: Move this code to another file


class TrackManager:
    def __init__(self, api_key: str, base_url: str, throw_on_error: bool):
        self.api_key = api_key
        self.base_url = base_url
        self.throw_on_error = throw_on_error

    def group(self, request_id, group_id):
        return group(self.api_key, self.base_url, self.throw_on_error, request_id, group_id)

    def metadata(self, request_id, metadata):
        return metadata_(self.api_key, self.base_url, self.throw_on_error, request_id, metadata)

    def prompt(self, request_id, prompt_name, prompt_input_variables, version=None, label=None):
        return prompt(
            self.api_key,
            self.base_url,
            self.throw_on_error,
            request_id,
            prompt_name,
            prompt_input_variables,
            version,
            label,
        )

    def score(self, request_id, score, score_name=None):
        return score_(self.api_key, self.base_url, self.throw_on_error, request_id, score, score_name)


class AsyncTrackManager:
    def __init__(self, api_key: str, base_url: str, throw_on_error: bool):
        self.api_key = api_key
        self.base_url = base_url
        self.throw_on_error = throw_on_error

    async def group(self, request_id, group_id):
        return await agroup(self.api_key, self.base_url, self.throw_on_error, request_id, group_id)

    async def metadata(self, request_id, metadata):
        return await ametadata(self.api_key, self.base_url, self.throw_on_error, request_id, metadata)

    async def prompt(self, request_id, prompt_name, prompt_input_variables, version=None, label=None):
        return await aprompt(
            self.api_key,
            self.base_url,
            self.throw_on_error,
            request_id,
            prompt_name,
            prompt_input_variables,
            version,
            label,
        )

    async def score(self, request_id, score, score_name=None):
        return await ascore(self.api_key, self.base_url, self.throw_on_error, request_id, score, score_name)


__all__ = ["TrackManager"]
