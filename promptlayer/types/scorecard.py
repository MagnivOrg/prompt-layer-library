from typing import Any, Dict, List, Literal, Optional, TypedDict, Union

from typing_extensions import NotRequired, Required

ScoreVerdict = Literal["pass", "warn", "fail", "error", "skipped", "unknown"]
ScorecardStatus = Literal["needs_setup", "ready", "queued", "running", "completed", "stale", "failed"]
ScorecardCalculationStatus = Literal["queued", "running", "completed", "failed", "cancelled"]
ScorecardPrimitiveType = Literal[
    "BOOLEAN",
    "NUMBER",
    "CATEGORICAL",
    "TEXT",
    "JSON",
]
ScorecardStaleState = Literal["fresh", "stale", "missing", "unknown"]


class ScorecardThresholds(TypedDict, total=False):
    pass_threshold: Optional[float]
    warn_threshold: Optional[float]


class ScorecardAggregationConfig(ScorecardThresholds, total=False):
    method: Required[str]
    required_step_failure_behavior: Optional[str]
    weights: Dict[str, float]


class EvaluatorResult(TypedDict, total=False):
    evaluator_id: Optional[str]
    evaluator_name: Optional[str]
    primitive_type: Optional[ScorecardPrimitiveType]
    score: Optional[float]
    value: Any
    verdict: Optional[ScoreVerdict]
    status: Optional[str]
    error_message: Optional[str]
    metadata: Dict[str, Any]


class CriterionSummary(TypedDict, total=False):
    criterion_id: Required[str]
    name: Required[str]
    score: Optional[float]
    verdict: Optional[ScoreVerdict]
    weight: Optional[float]
    required: Optional[bool]
    evaluator_result: Optional[EvaluatorResult]


class ScorecardStep(TypedDict, total=False):
    id: NotRequired[str]
    title: Required[str]
    description: NotRequired[str]
    evaluator_id: NotRequired[str]
    primitive_type: NotRequired[ScorecardPrimitiveType]
    required: NotRequired[bool]
    weight: NotRequired[float]
    thresholds: NotRequired[ScorecardThresholds]
    primitive_config: NotRequired[Dict[str, Any]]
    score_adapter: NotRequired[Dict[str, Any]]
    config: NotRequired[Dict[str, Any]]


class ScorecardDriftSummary(TypedDict, total=False):
    stale_state: Optional[ScorecardStaleState]
    stale_row_count: Optional[int]
    total_row_count: Optional[int]
    last_calculation_id: Optional[str]
    last_calculated_at: Optional[str]


class Scorecard(TypedDict, total=False):
    id: Required[str]
    table_id: Required[Union[str, int]]
    sheet_id: Required[Union[str, int]]
    name: Required[str]
    status: Optional[ScorecardStatus]
    evaluated_column_ids: List[Union[str, int]]
    aggregation: Required[ScorecardAggregationConfig]
    steps: Required[List[ScorecardStep]]
    version: Optional[int]
    thresholds: Optional[ScorecardThresholds]
    drift_summary: Optional[ScorecardDriftSummary]
    created_at: Optional[str]
    updated_at: Optional[str]


class ScorecardCalculation(TypedDict, total=False):
    id: Required[str]
    scorecard_id: Required[str]
    table_id: Required[Union[str, int]]
    sheet_id: Required[Union[str, int]]
    status: Required[ScorecardCalculationStatus]
    version: NotRequired[int]
    aggregate_score: NotRequired[float]
    aggregate_verdict: NotRequired[ScoreVerdict]
    criterion_summaries: NotRequired[List[CriterionSummary]]
    drift_summary: NotRequired[ScorecardDriftSummary]
    row_count: NotRequired[int]
    completed_row_count: NotRequired[int]
    failed_row_count: NotRequired[int]
    error_summary: NotRequired[Dict[str, Any]]
    started_at: NotRequired[str]
    completed_at: NotRequired[str]
    created_at: NotRequired[str]
    updated_at: NotRequired[str]


class ScorecardRowSummary(TypedDict, total=False):
    row_index: Required[int]
    calculation_id: NotRequired[str]
    aggregate_score: NotRequired[float]
    aggregate_verdict: NotRequired[ScoreVerdict]
    step_results: NotRequired[Dict[str, Any]]
    drift_summary: NotRequired[ScorecardDriftSummary]
    stale_state: NotRequired[ScorecardStaleState]
    error_summary: NotRequired[Dict[str, Any]]


class ScorecardRowBreakdown(ScorecardRowSummary, total=False):
    criterion_summaries: NotRequired[List[CriterionSummary]]
    row_data: NotRequired[Dict[str, Any]]


class ConfigureScorecardRequest(TypedDict, total=False):
    name: Required[str]
    evaluated_column_ids: Required[List[Union[str, int]]]
    aggregation: Required[ScorecardAggregationConfig]
    steps: Required[List[ScorecardStep]]


class MigrateLegacyScorecardRequest(TypedDict, total=False):
    # Defaults to False so migrations do not remove legacy score configuration unless requested.
    delete_legacy_score: bool


class RecalculateScorecardRequest(TypedDict, total=False):
    row_indices: List[int]
    step_ids: List[str]


class CancelScorecardRequest(TypedDict, total=False):
    row_indices: List[int]
    step_ids: List[str]


class ListScorecardRowsOptions(TypedDict, total=False):
    calculation_id: str
    verdict: ScoreVerdict
    limit: int
    cursor: str


class GetScorecardRowOptions(TypedDict, total=False):
    calculation_id: str


class ScorecardResponse(TypedDict, total=False):
    success: Required[bool]
    scorecard: Required[Scorecard]


class ScorecardCalculationResponse(TypedDict, total=False):
    success: Required[bool]
    calculation: Required[ScorecardCalculation]


class ScorecardActionResponse(TypedDict, total=False):
    success: Required[bool]
    message: NotRequired[str]
    scorecard: NotRequired[Scorecard]


class ScorecardRecalculateResponse(TypedDict, total=False):
    success: Required[bool]
    calculation_id: Required[str]
    status: Required[Literal["queued"]]
    version: Required[int]


class ScorecardCancelResponse(TypedDict, total=False):
    success: Required[bool]
    message: NotRequired[str]
    scorecard: NotRequired[Scorecard]
    cancelled_count: NotRequired[int]
    execution_ids: NotRequired[List[str]]
    calculation_id: NotRequired[str]


class ScorecardRowsResponse(TypedDict, total=False):
    success: Required[bool]
    calculation_id: NotRequired[str]
    rows: Required[List[ScorecardRowSummary]]
    next_cursor: NotRequired[Optional[str]]
    verdict_counts: NotRequired[Dict[str, int]]


class ScorecardRowResponse(ScorecardRowBreakdown, total=False):
    success: Required[bool]


class LegacyScorecardAggregateResult(TypedDict, total=False):
    scorecard_id: Required[str]
    scorecard_calculation_id: Required[str]
    aggregate_score: NotRequired[float]
    aggregate_verdict: NotRequired[ScoreVerdict]


class LegacyScorecardDetails(TypedDict, total=False):
    aggregate_result: Required[LegacyScorecardAggregateResult]
    aggregate: NotRequired[Dict[str, Any]]
    per_column: NotRequired[Dict[str, Any]]


class GetScoreScorecardCompatibilityResponse(TypedDict, total=False):
    score_type: Required[Literal["scorecard"]]
    scoring_type: Required[Literal["scorecard"]]
    score_configuration: Required[None]
    details: Required[LegacyScorecardDetails]
    overall_score: NotRequired[float]
    aggregate_score: NotRequired[float]
    aggregate: NotRequired[Dict[str, Any]]
    per_column: NotRequired[Dict[str, Any]]
    status: NotRequired[str]


class RecalculateScoreLegacyResponse(TypedDict, total=False):
    success: Required[bool]
    score_configuration_id: Required[str]
    status: Required[str]


class RecalculateScoreScorecardResponse(TypedDict, total=False):
    success: Required[bool]
    score_source: Required[Literal["scorecard"]]
    calculation_id: Required[str]
    status: Required[Literal["queued"]]
    version: Required[int]


RecalculateScoreResponse = Union[RecalculateScoreLegacyResponse, RecalculateScoreScorecardResponse]
