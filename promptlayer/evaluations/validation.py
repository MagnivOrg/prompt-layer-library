from typing import Any, Dict, Iterator, List, Optional, Tuple

from promptlayer import exceptions as _exceptions
from promptlayer.types.table import (
    EvalDataset,
    EvalProcessingColumn,
    EvalResult,
    EvalScorerColumn,
    ResourceId,
    Column,
)

from promptlayer.evaluations.scores import (  # noqa: F401 - re-exported for callers/tests
    case_has_failed_scorer,
    collect_failed_cell_row_indices,
    collect_failing_row_indices,
    extract_overall_score,
    scorer_pass_rates,
)
from promptlayer.evaluations.utils import (
    BASE_TEXT_COLUMNS,
    COLUMN_TITLE_ALIASES,
    EXPECTED_TRACE_COLUMN,
    TRACE_RESERVED_COLUMN_TITLES,
    TRACE_TEXT_COLUMNS,
    find_column_by_title,
)

_RESERVED_EVAL_COLUMN_TITLES = frozenset(
    BASE_TEXT_COLUMNS
    + TRACE_RESERVED_COLUMN_TITLES
    + (EXPECTED_TRACE_COLUMN,)
    + tuple(COLUMN_TITLE_ALIASES.keys())
)

# Config keys that bind a single source column by title (must match backend NAMED_SOURCE_KEYS).
_NAMED_SOURCE_KEYS = (
    "source",
    "chat_history_source",
    "user_persona_source",
    "conversation_completed_prompt_source",
    "prompt_source",
    "iterator_source",
    "value_source",
    "content_source",
    "diff_source",
    "trace_source",
    "expected_source",
)
_MAPPING_SOURCE_KEYS = (
    "prompt_template_variable_mappings",
    "variable_mappings",
    "input_variables",
)


def validation_error(message: str) -> _exceptions.PromptLayerValidationError:
    return _exceptions.PromptLayerValidationError(message, response=None, body=None)


def api_error(message: str) -> _exceptions.PromptLayerAPIError:
    return _exceptions.PromptLayerAPIError(message, response=None, body=None)


def not_found_error(message: str) -> _exceptions.PromptLayerNotFoundError:
    return _exceptions.PromptLayerNotFoundError(message, response=None, body=None)


def timeout_error(message: str) -> _exceptions.PromptLayerAPITimeoutError:
    return _exceptions.PromptLayerAPITimeoutError(message, response=None, body=None)


def _normalize_column_dict(
    column: Any,
    *,
    label: str,
    allow_callable: bool,
    forbid_text: bool,
    require_object_config: bool,
) -> Dict[str, Any]:
    if callable(column) and not isinstance(column, type):
        if not allow_callable:
            raise validation_error(
                "Eval columns must be explicit column definitions (e.g. column(...)); "
                "callables are only supported in scorers."
            )
        from promptlayer.evaluations.columns import scorer_from_function

        return scorer_from_function(column)
    if not isinstance(column, dict):
        if allow_callable:
            raise validation_error(
                "Eval scorers must be column dicts (e.g. llm_assertion_scorer(...)) or named Python functions."
            )
        raise validation_error("Eval columns must be column dicts (e.g. column(...)).")
    title = column.get("title")
    column_type = column.get("type")
    if not isinstance(title, str) or not title.strip():
        raise validation_error(f"Eval {label} title must be a non-empty string.")
    if not isinstance(column_type, str) or not column_type.strip():
        raise validation_error(f"Eval {label} type must be a non-empty string.")
    if forbid_text and column_type.upper() == "TEXT":
        raise validation_error(
            "Eval columns cannot be TEXT; use dataset fields or built-in input/expected/output columns."
        )
    normalized: Dict[str, Any] = {
        "title": title.strip() if forbid_text else title,
        "type": column_type,
    }
    if column.get("config") is not None:
        if require_object_config and not isinstance(column["config"], dict):
            raise validation_error(f"Eval {label} '{title}' config must be a dict.")
        normalized["config"] = column["config"]
    if column.get("weight") is not None:
        normalized["weight"] = float(column["weight"])
    if "required" in column:
        normalized["required"] = bool(column["required"])
    thresholds = column.get("thresholds")
    if isinstance(thresholds, dict):
        normalized["thresholds"] = dict(thresholds)
    return normalized


def normalize_scorer(scorer: Any) -> EvalScorerColumn:
    return _normalize_column_dict(  # type: ignore[return-value]
        scorer,
        label="scorer",
        allow_callable=True,
        forbid_text=False,
        require_object_config=False,
    )


def normalize_processing_column(column: Any) -> EvalProcessingColumn:
    return _normalize_column_dict(  # type: ignore[return-value]
        column,
        label="column",
        allow_callable=False,
        forbid_text=True,
        require_object_config=True,
    )


def validate_eval_target(
    *,
    table_id: Optional[ResourceId],
    sheet_id: Optional[ResourceId],
    folder_id: Optional[int],
) -> None:
    if table_id is not None and folder_id is not None:
        raise validation_error("Eval folder_id cannot be used together with table_id.")
    if folder_id is not None and not isinstance(folder_id, int):
        raise validation_error("Eval folder_id must be an integer.")
    if sheet_id is not None:
        raise validation_error("Eval sheet_id is not supported. Evals require a dedicated experiment sheet.")


def _assert_unique_column_titles(
    processing_columns: List[EvalProcessingColumn],
    scorers: List[EvalScorerColumn],
) -> None:
    seen: Dict[str, str] = {}
    for column in processing_columns:
        title = column["title"]
        if title in _RESERVED_EVAL_COLUMN_TITLES:
            raise validation_error(
                f"Eval column title {title!r} is reserved for built-in eval columns."
            )
        if title in seen:
            raise validation_error(f"Duplicate eval column title {title!r}.")
        seen[title] = "columns"
    for scorer in scorers:
        title = scorer["title"]
        if title in _RESERVED_EVAL_COLUMN_TITLES:
            raise validation_error(
                f"Eval scorer title {title!r} is reserved for built-in eval columns."
            )
        if title in seen:
            raise validation_error(
                f"Eval scorer title {title!r} conflicts with a supporting column title."
            )
        seen[title] = "scorers"


def assert_eval_args(
    name: str,
    dataset: EvalDataset,
    runner: Any,
    scorers: List[Any],
    *,
    columns: Optional[List[Any]] = None,
    table_id: Optional[ResourceId] = None,
    sheet_id: Optional[ResourceId] = None,
    folder_id: Optional[int] = None,
    experiment_name: Optional[str] = None,
    max_concurrency: int = 1,
    passing_score: Optional[float] = None,
) -> tuple[List[EvalScorerColumn], List[EvalProcessingColumn]]:
    if not isinstance(name, str) or not name.strip():
        raise validation_error("Eval name must be a non-empty string.")
    if not callable(runner):
        raise validation_error("Eval runner must be a function.")
    if not isinstance(scorers, list) or not scorers:
        raise validation_error("Eval scorers must be a non-empty list of column definitions or functions.")
    if columns is not None and not isinstance(columns, list):
        raise validation_error("Eval columns must be a list of column definitions.")
    normalized_scorers = [normalize_scorer(scorer) for scorer in scorers]
    normalized_columns = [normalize_processing_column(column) for column in (columns or [])]
    _assert_unique_column_titles(normalized_columns, normalized_scorers)

    validate_eval_target(table_id=table_id, sheet_id=sheet_id, folder_id=folder_id)
    if experiment_name is not None and (not isinstance(experiment_name, str) or not experiment_name.strip()):
        raise validation_error("Eval experiment_name must be a non-empty string.")
    if not isinstance(max_concurrency, int) or isinstance(max_concurrency, bool) or max_concurrency < 1:
        raise validation_error("Eval max_concurrency must be a positive integer.")
    if passing_score is not None and (not isinstance(passing_score, (int, float)) or isinstance(passing_score, bool)):
        raise validation_error("Eval passing_score must be a number.")

    if isinstance(dataset, list):
        if not dataset:
            raise validation_error("Eval dataset list must not be empty.")
        for case in dataset:
            if not isinstance(case, dict) or "input" not in case:
                raise validation_error("Each inline eval case must be a dict with an 'input' key.")
        return normalized_scorers, normalized_columns
    if isinstance(dataset, dict):
        if dataset.get("table_id") is None:
            raise validation_error("Eval dataset table reference requires table_id.")
        if "table" in dataset:
            raise validation_error("Eval dataset no longer accepts 'table' titles; use table_id.")
        return normalized_scorers, normalized_columns
    raise validation_error("Eval dataset must be a list of cases or a dict with table_id (and optional sheet_id).")


def assert_passing_score(
    score: Any,
    passing_score: Optional[float],
    *,
    result: Optional[EvalResult] = None,
    failing_row_indices: Optional[List[int]] = None,
) -> None:
    """Raise when a scorecard evaluator fails or the overall score misses the threshold."""
    eval_name = None
    sheet_url = None
    failed_cell_rows: List[int] = []
    if isinstance(result, dict):
        eval_name = result.get("name")
        sheet_url = result.get("url")
        failed_cell_rows = collect_failed_cell_row_indices(result.get("results") or [])

    if failed_cell_rows:
        header = (
            f"Evaluation {eval_name!r} failed: one or more scorecard evaluators failed to execute"
            if eval_name
            else "Evaluation failed: one or more scorecard evaluators failed to execute"
        )
        failure_lines = [header]
        if sheet_url:
            failure_lines.extend(["", f"Inspect the sheet: {sheet_url}"])
        raise _exceptions.EvaluationFailedError(
            "\n".join(failure_lines),
            score=score,
            passing_score=passing_score,
            result=result,
            failing_row_indices=failed_cell_rows,
        )

    if passing_score is None:
        return

    threshold = float(passing_score)
    overall = extract_overall_score(score)
    if overall is not None and overall >= threshold:
        return

    if overall is None:
        header = (
            f"Evaluation {eval_name!r} failed: overall score is missing (passing score {threshold})"
            if eval_name
            else f"Evaluation failed: overall score is missing (passing score {threshold})"
        )
    else:
        header = (
            f"Evaluation {eval_name!r} failed: overall score {overall} is below passing score {threshold}"
            if eval_name
            else f"Evaluation failed: overall score {overall} is below passing score {threshold}"
        )

    failure_lines = [header]
    if sheet_url:
        failure_lines.extend(["", f"Inspect the sheet: {sheet_url}"])

    raise _exceptions.EvaluationFailedError(
        "\n".join(failure_lines),
        score=score,
        passing_score=threshold,
        result=result,
        failing_row_indices=failing_row_indices,
    )


def dependency_item(
    column_id: Any,
    *,
    config_key: str,
    config_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    item: Dict[str, Any] = {
        "column_id": str(column_id),
        "reference_type": "value",
        "config_key": config_key,
    }
    if config_meta is not None:
        item["config_meta"] = config_meta
    return item


def iter_scorer_sources(config: Optional[Dict[str, Any]]) -> Iterator[Tuple[Any, str, Optional[Dict[str, Any]]]]:
    if not isinstance(config, dict):
        return
    for key in _NAMED_SOURCE_KEYS:
        yield config.get(key), key, None
    sources = config.get("sources")
    if isinstance(sources, list):
        for position, title in enumerate(sources):
            yield title, "sources", {"position": position}
    for key in _MAPPING_SOURCE_KEYS:
        mapping = config.get(key)
        if isinstance(mapping, dict):
            for variable_name, title in mapping.items():
                yield title, key, {"variable_name": str(variable_name)}


def scorer_dependencies_from_config(
    config: Optional[Dict[str, Any]],
    columns_by_title: Dict[str, Column],
    *,
    label: str = "scorer",
) -> List[Dict[str, Any]]:
    """Build config-driven dependency edges from title-based column config."""
    if not isinstance(config, dict):
        return []

    dependencies: List[Dict[str, Any]] = []
    missing: List[str] = []

    def _require_column(title: Any) -> Optional[Column]:
        if not isinstance(title, str) or not title.strip():
            return None
        column = find_column_by_title(columns_by_title, title)
        if column is None:
            missing.append(title)
            return None
        return column

    for source_title, key, config_meta in iter_scorer_sources(config):
        column = _require_column(source_title)
        if column is not None:
            dependencies.append(dependency_item(column["id"], config_key=key, config_meta=config_meta))

    if missing:
        unique_missing = ", ".join(sorted(set(missing)))
        raise validation_error(
            f"Eval {label} source column(s) not found: {unique_missing}. "
            "Use exact column titles (e.g. 'Output' or 'Trace'), or declare "
            "supporting columns before they are referenced."
        )
    return dependencies


def resolve_config_sources_to_column_ids(
    config: Optional[Dict[str, Any]],
    columns_by_title: Dict[str, Column],
) -> Dict[str, Any]:
    """Return a copy of config with source column titles rewritten to column IDs.

    Authoring APIs keep human-readable titles (e.g. ``source="Output"``). The
    scorecard UI and backing-column dependency wiring expect UUIDs in
    ``primitive_config``, so rewrite titles once columns exist.
    """
    if not isinstance(config, dict):
        return {}

    by_id = {str(column["id"]): column for column in columns_by_title.values()}

    def _resolve_ref(reference: Any) -> Any:
        if not isinstance(reference, str) or not reference.strip():
            return reference
        column = find_column_by_title(columns_by_title, reference) or by_id.get(reference)
        if column is None:
            return reference
        return str(column["id"])

    resolved = dict(config)
    for key in _NAMED_SOURCE_KEYS:
        if key in resolved:
            resolved[key] = _resolve_ref(resolved[key])

    # Used by COLUMN_AGGREGATE; not in dependency iteration but still UI-bound.
    if "label_source" in resolved:
        resolved["label_source"] = _resolve_ref(resolved["label_source"])

    sources = resolved.get("sources")
    if isinstance(sources, list):
        resolved["sources"] = [_resolve_ref(item) for item in sources]

    for key in _MAPPING_SOURCE_KEYS:
        mapping = resolved.get(key)
        if isinstance(mapping, dict):
            resolved[key] = {
                variable_name: _resolve_ref(source_ref)
                for variable_name, source_ref in mapping.items()
            }
    return resolved


def _config_references_trace(config: Optional[Dict[str, Any]]) -> bool:
    from promptlayer.evaluations.code_refs import config_references_titles

    return config_references_titles(config, TRACE_TEXT_COLUMNS)


def scorers_reference_trace(scorers: List[EvalScorerColumn]) -> bool:
    """True when any scorer config references the Trace column by title."""
    return any(_config_references_trace(scorer.get("config") if isinstance(scorer, dict) else None) for scorer in scorers)


def columns_reference_trace(columns: List[EvalProcessingColumn]) -> bool:
    """True when any supporting column config references the Trace column by title."""
    return any(
        _config_references_trace(column.get("config") if isinstance(column, dict) else None) for column in columns
    )
