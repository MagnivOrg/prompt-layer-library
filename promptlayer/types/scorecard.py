from typing import Any, Dict, List, Literal, Optional, TypedDict, Union

from typing_extensions import Required

ScoreVerdict = Literal["pass", "warn", "fail", "error", "skipped", "unknown"]
ScorecardStatus = Literal["active", "draft", "disabled", "deleted"]
ScorecardCalculationStatus = Literal["queued", "running", "completed", "failed", "cancelled"]
ScorecardPrimitiveType = Literal["boolean", "number", "categorical", "text", "json"]
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
    id: Optional[str]
    name: Required[str]
    description: Optional[str]
    evaluator_id: Optional[str]
    primitive_type: Optional[ScorecardPrimitiveType]
    required: Optional[bool]
    weight: Optional[float]
    thresholds: Optional[ScorecardThresholds]
    config: Dict[str, Any]


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
    version: Optional[int]
    score: Optional[float]
    verdict: Optional[ScoreVerdict]
    row_count: Optional[int]
    completed_row_count: Optional[int]
    failed_row_count: Optional[int]
    error_message: Optional[str]
    started_at: Optional[str]
    completed_at: Optional[str]
    created_at: Optional[str]
    updated_at: Optional[str]


class ScorecardRowSummary(TypedDict, total=False):
    row_index: Required[int]
    calculation_id: Optional[str]
    score: Optional[float]
    verdict: Optional[ScoreVerdict]
    stale_state: Optional[ScorecardStaleState]
    criterion_summaries: List[CriterionSummary]
    error_message: Optional[str]


class ScorecardRowBreakdown(ScorecardRowSummary, total=False):
    criteria: List[CriterionSummary]
    evaluator_results: List[EvaluatorResult]
    aggregate_result: Dict[str, Any]
    row_data: Dict[str, Any]


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
    force: bool


class CancelScorecardRequest(TypedDict, total=False):
    calculation_id: str


class ListScorecardRowsOptions(TypedDict, total=False):
    calculation_id: str
    verdict: ScoreVerdict
    stale_state: ScorecardStaleState
    limit: int
    offset: int


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
    calculation_id: Optional[str]
    status: Optional[ScorecardCalculationStatus]
    version: Optional[int]
    scorecard: Optional[Scorecard]
    calculation: Optional[ScorecardCalculation]


class ScorecardRowsResponse(TypedDict, total=False):
    success: Required[bool]
    rows: Required[List[ScorecardRowSummary]]
    next_offset: Optional[int]


class ScorecardRowResponse(TypedDict, total=False):
    success: Required[bool]
    row: Required[ScorecardRowBreakdown]


class LegacyScorecardAggregateResult(TypedDict):
    scorecard_id: Required[str]
    scorecard_calculation_id: Required[str]


class LegacyScorecardDetails(TypedDict):
    aggregate_result: Required[LegacyScorecardAggregateResult]


class GetScoreScorecardCompatibilityResponse(TypedDict):
    score_type: Required[Literal["scorecard"]]
    scoring_type: Required[Literal["scorecard"]]
    score_configuration: Required[None]
    details: Required[LegacyScorecardDetails]


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
