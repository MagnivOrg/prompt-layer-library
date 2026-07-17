from typing import Any, Dict, List, Optional, Sequence

from promptlayer.tables import api as tables_api
from promptlayer.types.table import EvalScorerColumn, ResourceId, Column

from promptlayer.evaluations.polling import _apoll_until, _poll_until
from promptlayer.evaluations.terminal import get_terminal
from promptlayer.evaluations.utils import (
    _DEFAULT_POLL_INTERVAL_SECONDS,
    _DEFAULT_SCORE_WAIT_TIMEOUT_SECONDS,
    columns_by_title,
)
from promptlayer.evaluations.validation import (
    api_error,
    resolve_config_sources_to_column_ids,
    scorer_dependencies_from_config,
)

_DEFAULT_AGGREGATION = {
    "method": "weighted_mean",
    "required_step_failure_behavior": "fail",
    "pass_threshold": 0.8,
    "warn_threshold": 0.6,
}
_ACTIVE_CALCULATION_STATUSES = frozenset({"queued", "running"})
_TERMINAL_CALCULATION_STATUSES = frozenset({"completed", "failed", "cancelled"})


def _strip_sdk_only_config(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not config:
        return {}
    return {
        key: value
        for key, value in config.items()
        if key not in {"_sdkDiagnosis", "_sdk_diagnosis"}
    }


def _infer_code_execution_source_ids(
    code: Any,
    columns_by_title_map: Dict[str, Column],
) -> List[str]:
    from promptlayer.evaluations.code_refs import infer_referenced_column_titles

    if not isinstance(code, str) or not code.strip():
        return [str(column["id"]) for column in columns_by_title_map.values()]
    referenced = infer_referenced_column_titles(code, columns_by_title_map.keys())
    ids = [str(columns_by_title_map[title]["id"]) for title in columns_by_title_map if title in referenced]
    if ids:
        return ids
    return [str(column["id"]) for column in columns_by_title_map.values()]


def build_scorecard_steps_from_scorers(
    scorers: Sequence[EvalScorerColumn],
    columns: Sequence[Column],
) -> List[Dict[str, Any]]:
    by_title = columns_by_title(list(columns))
    steps: List[Dict[str, Any]] = []
    for index, scorer in enumerate(scorers):
        primitive_type = str(scorer["type"]).upper()
        author_config = _strip_sdk_only_config(scorer.get("config"))
        dependencies = scorer_dependencies_from_config(scorer.get("config"), by_title)
        source_column_ids = [str(dependency["column_id"]) for dependency in dependencies]
        if primitive_type == "CODE_EXECUTION" and not source_column_ids:
            source_column_ids = _infer_code_execution_source_ids(
                author_config.get("code"),
                by_title,
            )
        # Persist column IDs (not titles) so the Scorecard UI can resolve sources.
        primitive_config = resolve_config_sources_to_column_ids(author_config, by_title)
        step: Dict[str, Any] = {
            "title": scorer["title"],
            "primitive_type": primitive_type,
            "primitive_config": primitive_config,
            "order_index": index,
            "weight": 1,
            "required": False,
        }
        if source_column_ids:
            step["source_column_ids"] = source_column_ids
        steps.append(step)
    return steps


def configure_scorecard_from_scorers(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: ResourceId,
    sheet_id: ResourceId,
    columns: Sequence[Column],
    scorers: Sequence[EvalScorerColumn],
    name: str,
) -> Dict[str, Any]:
    steps = build_scorecard_steps_from_scorers(scorers, columns)
    response = tables_api.configure_sheet_scorecard(
        api_key,
        base_url,
        throw_on_error,
        table_id,
        sheet_id,
        {
            "name": name or "Evaluation",
            "evaluated_column_ids": [],
            "aggregation": dict(_DEFAULT_AGGREGATION),
            "steps": steps,
        },
    )
    if not response or not response.get("scorecard"):
        raise api_error("Failed to configure scorecard evaluators for this eval.")
    return response


async def aconfigure_scorecard_from_scorers(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: ResourceId,
    sheet_id: ResourceId,
    columns: Sequence[Column],
    scorers: Sequence[EvalScorerColumn],
    name: str,
) -> Dict[str, Any]:
    steps = build_scorecard_steps_from_scorers(scorers, columns)
    response = await tables_api.aconfigure_sheet_scorecard(
        api_key,
        base_url,
        throw_on_error,
        table_id,
        sheet_id,
        {
            "name": name or "Evaluation",
            "evaluated_column_ids": [],
            "aggregation": dict(_DEFAULT_AGGREGATION),
            "steps": steps,
        },
    )
    if not response or not response.get("scorecard"):
        raise api_error("Failed to configure scorecard evaluators for this eval.")
    return response


def _is_terminal_scorecard_response(payload: Optional[Dict[str, Any]]) -> bool:
    if not payload:
        return False
    latest = payload.get("latest_calculation") or {}
    latest_status = str(latest.get("status") or "").lower() or None
    if latest_status in _ACTIVE_CALCULATION_STATUSES:
        return False
    if latest_status in _TERMINAL_CALCULATION_STATUSES:
        return True
    scorecard = payload.get("scorecard") or {}
    status = str(scorecard.get("status") or "").lower()
    return status in {"completed", "failed", "stale", "ready"}


def _report_scorecard_progress(payload: Optional[Dict[str, Any]]) -> None:
    progress = (payload or {}).get("progress") or {}
    if isinstance(progress.get("scored_rows"), int) and isinstance(progress.get("total_rows"), int):
        get_terminal().scoring_progress(progress["scored_rows"], progress["total_rows"], 0)


def _scorecard_is_done(payload: Any, calculation_id: str) -> bool:
    if not isinstance(payload, dict):
        return False
    latest = payload.get("latest_calculation") or {}
    latest_id = latest.get("id")
    # Ignore stale payloads from a previous calculation until the new one appears
    # or the response is already terminal.
    if latest_id is not None and str(latest_id) != calculation_id and not _is_terminal_scorecard_response(payload):
        return False
    return _is_terminal_scorecard_response(payload)


def recalculate_and_wait_scorecard(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: ResourceId,
    sheet_id: ResourceId,
    *,
    timeout_seconds: float = _DEFAULT_SCORE_WAIT_TIMEOUT_SECONDS,
    poll_interval_seconds: float = _DEFAULT_POLL_INTERVAL_SECONDS,
) -> Dict[str, Any]:
    recalculate = tables_api.recalculate_smart_sheet_scorecard(
        api_key, base_url, throw_on_error, table_id, sheet_id, {}
    )
    if not recalculate or not recalculate.get("calculation_id"):
        raise api_error("Failed to start scorecard recalculation for this eval.")
    calculation_id = str(recalculate["calculation_id"])

    payload = _poll_until(
        fetch=lambda: tables_api.get_sheet_scorecard(
            api_key, base_url, throw_on_error, table_id, sheet_id
        ),
        is_done=lambda response: _scorecard_is_done(response, calculation_id),
        timeout_seconds=timeout_seconds,
        poll_interval_seconds=poll_interval_seconds,
        timeout_message="Timed out waiting for scorecard calculation to finish.",
        on_update=_report_scorecard_progress,
    )
    if not payload:
        raise api_error("Scorecard calculation returned an empty response.")
    return payload


async def arecalculate_and_wait_scorecard(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: ResourceId,
    sheet_id: ResourceId,
    *,
    timeout_seconds: float = _DEFAULT_SCORE_WAIT_TIMEOUT_SECONDS,
    poll_interval_seconds: float = _DEFAULT_POLL_INTERVAL_SECONDS,
) -> Dict[str, Any]:
    recalculate = await tables_api.arecalculate_smart_sheet_scorecard(
        api_key, base_url, throw_on_error, table_id, sheet_id, {}
    )
    if not recalculate or not recalculate.get("calculation_id"):
        raise api_error("Failed to start scorecard recalculation for this eval.")
    calculation_id = str(recalculate["calculation_id"])

    payload = await _apoll_until(
        fetch=lambda: tables_api.aget_sheet_scorecard(
            api_key, base_url, throw_on_error, table_id, sheet_id
        ),
        is_done=lambda response: _scorecard_is_done(response, calculation_id),
        timeout_seconds=timeout_seconds,
        poll_interval_seconds=poll_interval_seconds,
        timeout_message="Timed out waiting for scorecard calculation to finish.",
        on_update=_report_scorecard_progress,
    )
    if not payload:
        raise api_error("Scorecard calculation returned an empty response.")
    return payload


def _map_step_result_to_score_value(result: Optional[Dict[str, Any]]) -> Any:
    if not result:
        return None
    verdict = str(result.get("verdict") or "").lower() or None
    if verdict == "error":
        return {
            "status": "FAILED",
            "error": result.get("error_message")
            or result.get("evidence")
            or result.get("raw_value")
            or result,
        }
    if result.get("raw_value") is not None:
        return result["raw_value"]
    if isinstance(result.get("score"), (int, float)):
        return result["score"]
    if verdict == "pass":
        return True
    if verdict in {"fail", "warn"}:
        return False
    return result


def extract_scorecard_scorer_outputs(
    row_payload: Optional[Dict[str, Any]],
    steps: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    outputs: Dict[str, Any] = {}
    step_results = (row_payload or {}).get("step_results") or {}
    if not isinstance(step_results, dict):
        step_results = {}

    for step in steps:
        step_id = step.get("id")
        title = step.get("title")
        if step_id is None or title is None:
            continue
        outputs[str(title)] = _map_step_result_to_score_value(step_results.get(str(step_id)))

    for step in steps:
        title = step.get("title")
        if title is None or str(title) in outputs:
            continue
        if str(title) in step_results:
            outputs[str(title)] = _map_step_result_to_score_value(step_results.get(str(title)))
    return outputs


def fetch_scorecard_row_scores(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: ResourceId,
    sheet_id: ResourceId,
    row_indices: Sequence[Optional[int]],
    scorecard_payload: Dict[str, Any],
) -> Dict[int, Dict[str, Any]]:
    scorecard = scorecard_payload.get("scorecard") or {}
    steps = scorecard.get("steps") if isinstance(scorecard.get("steps"), list) else []
    latest = scorecard_payload.get("latest_calculation") or {}
    calculation_id = str(latest["id"]) if latest.get("id") is not None else None
    params = {"calculation_id": calculation_id} if calculation_id else None

    scores_by_row: Dict[int, Dict[str, Any]] = {}
    for row_index in row_indices:
        if row_index is None:
            continue
        row_payload = tables_api.get_sheet_scorecard_row(
            api_key,
            base_url,
            throw_on_error,
            table_id,
            sheet_id,
            int(row_index),
            params=params,
        )
        scores_by_row[int(row_index)] = extract_scorecard_scorer_outputs(row_payload, steps)
    return scores_by_row


async def afetch_scorecard_row_scores(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: ResourceId,
    sheet_id: ResourceId,
    row_indices: Sequence[Optional[int]],
    scorecard_payload: Dict[str, Any],
) -> Dict[int, Dict[str, Any]]:
    scorecard = scorecard_payload.get("scorecard") or {}
    steps = scorecard.get("steps") if isinstance(scorecard.get("steps"), list) else []
    latest = scorecard_payload.get("latest_calculation") or {}
    calculation_id = str(latest["id"]) if latest.get("id") is not None else None
    params = {"calculation_id": calculation_id} if calculation_id else None

    scores_by_row: Dict[int, Dict[str, Any]] = {}
    for row_index in row_indices:
        if row_index is None:
            continue
        row_payload = await tables_api.aget_sheet_scorecard_row(
            api_key,
            base_url,
            throw_on_error,
            table_id,
            sheet_id,
            int(row_index),
            params=params,
        )
        scores_by_row[int(row_index)] = extract_scorecard_scorer_outputs(row_payload, steps)
    return scores_by_row


def extract_scorecard_overall_score(scorecard_payload: Optional[Dict[str, Any]]) -> Optional[float]:
    if not scorecard_payload:
        return None
    latest = scorecard_payload.get("latest_calculation") or {}
    aggregate = latest.get("aggregate_score")
    if isinstance(aggregate, (int, float)):
        return float(aggregate)
    progress = scorecard_payload.get("progress") or {}
    partial = progress.get("partial_score")
    if isinstance(partial, (int, float)):
        return float(partial)
    return None
