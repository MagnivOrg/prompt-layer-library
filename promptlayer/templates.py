from typing import Dict, Union
from promptlayer.utils import get_prompt_template


def get(
    *,
    prompt_name: str,
    provider: Union[str, None] = None,
    input_variables: Dict[str, str] = {},
):
    return get_prompt_template(
        prompt_name=prompt_name, provider=provider, input_variables=input_variables
    )
