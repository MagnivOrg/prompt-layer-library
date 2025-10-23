from promptlayer.utils import (
    apromptlayer_track_group,
    apromptlayer_track_metadata,
    apromptlayer_track_prompt,
    apromptlayer_track_score,
    promptlayer_track_group,
    promptlayer_track_metadata,
    promptlayer_track_prompt,
    promptlayer_track_score,
)


def prompt(
    api_key: str,
    base_url: str,
    request_id,
    prompt_name,
    prompt_input_variables,
    version=None,
    label=None,
):
    if not isinstance(prompt_input_variables, dict):
        raise Exception("Please provide a dictionary of input variables.")
    return promptlayer_track_prompt(
        api_key, base_url, request_id, prompt_name, prompt_input_variables, api_key, version, label
    )


def metadata(api_key: str, base_url: str, request_id, metadata):
    if not isinstance(metadata, dict):
        raise Exception("Please provide a dictionary of metadata.")
    for key, value in metadata.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise Exception("Please provide a dictionary of metadata with key value pair of strings.")
    return promptlayer_track_metadata(api_key, base_url, request_id, metadata)


def score(api_key: str, base_url: str, request_id, score, score_name=None):
    if not isinstance(score, int):
        raise Exception("Please provide a int score.")
    if not isinstance(score_name, str) and score_name is not None:
        raise Exception("Please provide a string as score name.")
    if score < 0 or score > 100:
        raise Exception("Please provide a score between 0 and 100.")
    return promptlayer_track_score(api_key, base_url, request_id, score, score_name)


def group(api_key: str, base_url: str, request_id, group_id):
    return promptlayer_track_group(api_key, base_url, request_id, group_id)


async def aprompt(
    api_key: str,
    base_url: str,
    request_id,
    prompt_name,
    prompt_input_variables,
    version=None,
    label=None,
):
    if not isinstance(prompt_input_variables, dict):
        raise Exception("Please provide a dictionary of input variables.")
    return await apromptlayer_track_prompt(
        api_key, base_url, request_id, prompt_name, prompt_input_variables, version, label
    )


async def ametadata(api_key: str, base_url: str, request_id, metadata):
    if not isinstance(metadata, dict):
        raise Exception("Please provide a dictionary of metadata.")
    for key, value in metadata.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise Exception("Please provide a dictionary of metadata with key-value pairs of strings.")
    return await apromptlayer_track_metadata(api_key, base_url, request_id, metadata)


async def ascore(api_key: str, base_url: str, request_id, score, score_name=None):
    if not isinstance(score, int):
        raise Exception("Please provide an integer score.")
    if not isinstance(score_name, str) and score_name is not None:
        raise Exception("Please provide a string as score name.")
    if score < 0 or score > 100:
        raise Exception("Please provide a score between 0 and 100.")
    return await apromptlayer_track_score(api_key, base_url, request_id, score, score_name)


async def agroup(api_key: str, base_url: str, request_id, group_id):
    return await apromptlayer_track_group(api_key, base_url, request_id, group_id)
