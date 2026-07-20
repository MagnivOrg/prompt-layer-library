from enum import Enum
from typing import Any, Dict, List, Literal, Optional, TypedDict, Union

from typing_extensions import Required

ResourceId = Union[str, int]


class ColumnType(str, Enum):
    TEXT = "TEXT"
    NUMBER = "NUMBER"
    BOOLEAN = "BOOLEAN"
    PROMPT_TEMPLATE = "PROMPT_TEMPLATE"
    LLM_ASSERTION = "LLM_ASSERTION"
    CODE_EXECUTION = "CODE_EXECUTION"
    COMPARE = "COMPARE"
    CONTAINS = "CONTAINS"
    COALESCE = "COALESCE"
    FOR_LOOP = "FOR_LOOP"
    WHILE_LOOP = "WHILE_LOOP"
    JSON_PATH = "JSON_PATH"
    COUNT = "COUNT"
    REGEX = "REGEX"
    TRAJECTORY = "TRAJECTORY"
    COSINE_SIMILARITY = "COSINE_SIMILARITY"
    ASSERT_VALID = "ASSERT_VALID"
    AI_DATA_EXTRACTION = "AI_DATA_EXTRACTION"
    COMPOSITION = "COMPOSITION"


# Literal form kept for TypedDict / backwards-compatible string literals.
ColumnTypeValue = Literal[
    "TEXT",
    "NUMBER",
    "BOOLEAN",
    "PROMPT_TEMPLATE",
    "LLM_ASSERTION",
    "CODE_EXECUTION",
    "COMPARE",
    "CONTAINS",
    "COALESCE",
    "FOR_LOOP",
    "WHILE_LOOP",
    "JSON_PATH",
    "COUNT",
    "REGEX",
    "TRAJECTORY",
    "COSINE_SIMILARITY",
    "ASSERT_VALID",
    "AI_DATA_EXTRACTION",
    "COMPOSITION",
]

CellStatus = Literal[
    "PENDING",
    "QUEUED",
    "DISPATCHED",
    "RUNNING",
    "COMPLETED",
    "FAILED",
    "STALE",
    "SKIPPED",
]


class Table(TypedDict, total=False):
    id: Required[Union[str, int]]
    title: Required[str]
    workspace_id: Optional[int]
    folder_id: Optional[int]
    sheet_count: Optional[int]
    created_at: Optional[str]
    updated_at: Optional[str]
    deleted_at: Optional[str]


class Sheet(TypedDict, total=False):
    id: Required[Union[str, int]]
    table_id: Optional[Union[str, int]]
    title: Optional[str]
    index: Optional[int]
    row_count: Optional[int]
    version_count: Optional[int]
    created_at: Optional[str]
    updated_at: Optional[str]


class Column(TypedDict, total=False):
    id: Required[Union[str, int]]
    sheet_id: Optional[Union[str, int]]
    title: Required[str]
    type: Required[ColumnTypeValue]
    position_rank: Optional[float]
    config: Optional[Dict[str, Any]]
    is_output_column: Optional[bool]
    created_at: Optional[str]
    updated_at: Optional[str]


class Cell(TypedDict, total=False):
    id: Required[Union[str, int]]
    sheet_id: Optional[Union[str, int]]
    column_id: Optional[Union[str, int]]
    row_index: Optional[int]
    status: Optional[CellStatus]
    value: Any
    display_value: Optional[str]


class SheetVersion(TypedDict, total=False):
    id: Required[Union[str, int]]
    sheet_id: Optional[Union[str, int]]
    version_number: Optional[int]
    snapshot: Optional[Dict[str, Any]]
    created_by: Optional[Union[str, int]]
    created_at: Optional[str]


class TableScore(TypedDict, total=False):
    score: Optional[float]
    score_configuration: Optional[Dict[str, Any]]


class CreateTable(TypedDict, total=False):
    title: Required[str]
    folder_id: Optional[int]


class UpdateTable(TypedDict, total=False):
    title: Optional[str]
    folder_id: Optional[int]


class CreateSheetFileSource(TypedDict, total=False):
    type: Required[Literal["file"]]
    file_name: Required[str]
    file_content_base64: Required[str]


class CreateSheetRequestLogsSource(TypedDict, total=False):
    type: Required[Literal["request_logs"]]
    request_log_ids: Optional[List[int]]
    prompt_id: Optional[int]
    prompt_version_id: Optional[int]
    prompt_label_id: Optional[int]
    start_time: Optional[str]
    end_time: Optional[str]


CreateSheetSource = Union[CreateSheetFileSource, CreateSheetRequestLogsSource]


class CreateSheet(TypedDict, total=False):
    title: Optional[str]
    index: Optional[int]
    operation_id: Optional[str]
    source: Required[CreateSheetSource]


class UpdateSheet(TypedDict, total=False):
    title: Optional[str]


class ColumnDependency(TypedDict, total=False):
    column_id: Required[str]
    reference_type: Optional[str]
    config_key: Optional[str]
    config_meta: Optional[Dict[str, Any]]


class CreateColumn(TypedDict, total=False):
    title: Required[str]
    type: Required[ColumnTypeValue]
    config: Optional[Dict[str, Any]]
    dependencies: Optional[List[ColumnDependency]]
    is_output_column: Optional[bool]


class UpdateColumn(TypedDict, total=False):
    title: Optional[str]
    type: Optional[ColumnTypeValue]
    config: Optional[Dict[str, Any]]
    dependencies: Optional[List[ColumnDependency]]
    is_output_column: Optional[bool]


class AddTableRows(TypedDict, total=False):
    count: Optional[int]
    values: Optional[List[Dict[str, Any]]]


class DeleteSheetRows(TypedDict, total=False):
    row_indices: Required[List[int]]


class UpdateCell(TypedDict, total=False):
    display_value: Optional[str]
    value: Any


class CreateSheetVersion(TypedDict, total=False):
    name: Optional[str]


class ConfigureSheetScore(TypedDict, total=False):
    score_type: Optional[str]
    score_config: Optional[Dict[str, Any]]
    column_ids: Optional[List[str]]
    column_names: Optional[List[str]]
    code: Optional[str]
    code_language: Optional[Literal["PYTHON", "JAVASCRIPT"]]
    true_values: Optional[List[str]]
    false_values: Optional[List[str]]
    assertion_aggregation: Optional[Literal["all", "any", "mean"]]


class AddTraceImport(TypedDict, total=False):
    trace_id: Required[str]
    sheet_id: Required[Union[str, int]]
    smart_table_id: Optional[Union[str, int]]
    table_id: Optional[Union[str, int]]
    span_id: Optional[str]
    metadata: Optional[Dict[str, Any]]


class ListTablesParams(TypedDict, total=False):
    cursor: Optional[str]
    limit: Optional[int]
    name: Optional[str]
    folder_id: Optional[int]
    page: Optional[int]
    per_page: Optional[int]


class TableListResponse(TypedDict, total=False):
    success: Optional[bool]
    data: Optional[List[Table]]
    tables: Optional[List[Table]]
    items: Optional[List[Table]]


class TableResponse(TypedDict, total=False):
    success: Optional[bool]
    table: Optional[Table]


class SheetListResponse(TypedDict, total=False):
    success: Optional[bool]
    data: Optional[List[Sheet]]
    sheets: Optional[List[Sheet]]
    items: Optional[List[Sheet]]


class SheetResponse(TypedDict, total=False):
    success: Optional[bool]
    sheet: Optional[Sheet]


class ColumnListResponse(TypedDict, total=False):
    success: Optional[bool]
    data: Optional[List[Column]]
    columns: Optional[List[Column]]
    items: Optional[List[Column]]


class ColumnResponse(TypedDict, total=False):
    success: Optional[bool]
    column: Optional[Column]


class CellResponse(TypedDict, total=False):
    success: Optional[bool]
    cell: Optional[Cell]


class SheetVersionListResponse(TypedDict, total=False):
    success: Optional[bool]
    data: Optional[List[SheetVersion]]
    versions: Optional[List[SheetVersion]]
    items: Optional[List[SheetVersion]]


class SheetVersionResponse(TypedDict, total=False):
    success: Optional[bool]
    version: Optional[SheetVersion]


class TableScoreResponse(TypedDict, total=False):
    success: Optional[bool]
    score: Optional[TableScore]


class SheetStatusCounts(TypedDict, total=False):
    STALE: int
    QUEUED: int
    DISPATCHED: int
    RUNNING: int
    COMPLETED: int
    FAILED: int


class SheetStatusCountsResponse(TypedDict, total=False):
    success: Optional[bool]
    total_cells: Required[int]
    status_counts: Required[SheetStatusCounts]


class BatchRecalculateCells(TypedDict, total=False):
    cell_ids: Optional[List[Union[str, int]]]
    column_ids: Optional[List[Union[str, int]]]
    row_indices: Optional[List[int]]


class CreateSheetOperation(TypedDict, total=False):
    operation: str
    column_ids: Optional[List[Union[str, int]]]
    row_ids: Optional[List[int]]
    statuses: Optional[List[str]]


class EvalCase(TypedDict, total=False):
    input: Required[Any]
    expected: Optional[Any]
    expected_trace: Optional[Any]


class EvalDatasetRef(TypedDict, total=False):
    table_id: Required[Union[str, int]]
    sheet_id: Optional[Union[str, int]]


EvalDataset = Union[List[EvalCase], EvalDatasetRef]


class EvalScorerColumn(TypedDict, total=False):
    title: Required[str]
    type: Required[ColumnTypeValue]
    config: Optional[Dict[str, Any]]
    weight: Optional[float]
    required: Optional[bool]
    thresholds: Optional[Dict[str, float]]


# Supporting/preprocessing columns share the same wire shape as scorer defs.
EvalProcessingColumn = EvalScorerColumn


class EvalDefinition(TypedDict, total=False):
    name: Required[str]
    dataset: Required[EvalDataset]
    runner: Required[Any]
    scorers: Required[List[Any]]
    columns: Optional[List[Any]]
    table_id: Optional[Union[str, int]]
    sheet_id: Optional[Union[str, int]]
    folder_id: Optional[int]
    experiment_name: Optional[str]
    max_concurrency: Optional[int]
    passing_score: Optional[float]
    include_failure_examples: Optional[bool]
    base_url: Optional[str]
    api_key: Optional[str]


class EvalCaseResult(TypedDict, total=False):
    input: Any
    expected: Any
    output: Any
    scores: Dict[str, Any]
    trace_id: Optional[str]
    span_id: Optional[str]
    row_index: Optional[int]


class EvalScoreCard(TypedDict):
    scorer: str
    passed: int
    total: int
    pass_rate: float


class EvalResult(TypedDict, total=False):
    name: str
    table_id: Union[str, int]
    sheet_id: Union[str, int]
    failed_row_indices: List[int]
    score_cards: List[EvalScoreCard]
    total_rows: int
    results: List[EvalCaseResult]
    url: Optional[str]
