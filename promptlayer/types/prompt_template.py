from typing import Dict, TypedDict, Union


class GetPromptTemplate(TypedDict, total=False):
    version: Union[int, None]
    label: Union[str, None]
    provider: Union[str, None]
    input_variables: Union[Dict[str, str], None]
