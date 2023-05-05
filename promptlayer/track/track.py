from promptlayer.utils import promptlayer_track_prompt, promptlayer_track_metadata, promptlayer_track_score

def prompt(request_id, prompt_name, prompt_input_variables, version=None):
    if not isinstance(prompt_input_variables, dict):
        raise Exception("Please provide a dictionary of input variables.")
    return promptlayer_track_prompt(request_id, prompt_name, prompt_input_variables, version)


def metadata(request_id, metadata):
    if not isinstance(metadata, dict):
        raise Exception("Please provide a dictionary of metadata.")
    return promptlayer_track_metadata(request_id, metadata)

def score(request_id, score):
    if not isinstance(score, int):
        raise Exception("Please provide a int score.")
    if score < 0 or score > 100:
        raise Exception("Please provide a score between 0 and 100.")
    return promptlayer_track_score(request_id, score)