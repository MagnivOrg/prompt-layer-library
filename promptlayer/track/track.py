from promptlayer.utils import get_api_key, promptlayer_track_prompt, promptlayer_track_metadata

def prompt(request_id, prompt_name, prompt_input_variables):
    if not isinstance(prompt_input_variables, dict):
        raise Exception("Please provide a dictionary of input variables.")
    return promptlayer_track_prompt(request_id, prompt_name, prompt_input_variables, get_api_key())


def metadata(request_id, metadata):
    if not isinstance(metadata, dict):
        raise Exception("Please provide a dictionary of metadata.")
    return promptlayer_track_metadata(request_id, metadata, get_api_key())