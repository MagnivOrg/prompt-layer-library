from typing import Dict, TypedDict


class GetPromptTemplate(TypedDict, total=False):
    version: int
    label: str
    provider: str
    input_variables: Dict[str, str]
