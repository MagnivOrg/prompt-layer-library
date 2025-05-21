from typing import Any, Dict, List, Union


class RunInternal:
    def __init__(
        self,
        *,
        prompt_name: str,
        prompt_version: Union[int, None] = None,
        prompt_release_label: Union[str, None] = None,
        input_variables: Union[Dict[str, Any], None] = None,
        model_parameter_overrides: Union[Dict[str, Any], None] = None,
        tags: Union[List[str], None] = None,
        metadata: Union[Dict[str, str], None] = None,
        group_id: Union[int, None] = None,
        stream: bool = False,
        pl_run_span_id: Union[str, None] = None,
    ):
        self.prompt_name = prompt_name
        self.prompt_version_number = prompt_version
        self.prompt_release_label = prompt_release_label
        self.input_variables = input_variables
        self.model_parameter_overrides = model_parameter_overrides
        self.tags = tags
        self.metadata = metadata
        self.group_id = group_id
        self.stream = stream
        self.pl_run_span_id = pl_run_span_id
