import json
import os
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse, urlunparse

from promptlayer.tables.helpers import extract_list
from promptlayer.types.table import Column

DATASET_TEXT_COLUMNS = ("input", "expected")
BASE_TEXT_COLUMNS = DATASET_TEXT_COLUMNS + ("Output",)
EXPECTED_TRACE_COLUMN = "expected_trace"
TRACE_TEXT_COLUMNS = ("Trace",)
TRACE_RESERVED_COLUMN_TITLES = (
    "Trace",
    "Trace.price",
    "Trace.latency",
    "Trace link",
    "total_trace_time",
    "total_price",
)
LEGACY_COLUMN_TITLES = {
    "input": "Input",
    "expected": "Expected",
    EXPECTED_TRACE_COLUMN: "Expected Trace",
}
GENERATED_COLUMN_TITLE_ALIASES = {
    "output": "Output",
    "trace": "Trace",
}
LEGACY_TO_CURRENT_COLUMN_TITLES = {legacy: current for current, legacy in LEGACY_COLUMN_TITLES.items()}
RESERVED_EVAL_COLUMN_TITLES = frozenset(
    BASE_TEXT_COLUMNS
    + TRACE_RESERVED_COLUMN_TITLES
    + (EXPECTED_TRACE_COLUMN,)
    + tuple(LEGACY_COLUMN_TITLES.values())
    + tuple(GENERATED_COLUMN_TITLE_ALIASES.keys())
)
EVAL_CASE_BUILTIN_KEYS = frozenset({"input", "expected", "expected_trace"})


def is_reserved_eval_column_title(title: str) -> bool:
    return title in RESERVED_EVAL_COLUMN_TITLES


def custom_eval_field_titles(cases: List[Dict[str, Any]]) -> List[str]:
    """Return custom case keys in stable first-seen order."""
    seen = set()
    titles: List[str] = []
    for case in cases:
        for title in case:
            if title in EVAL_CASE_BUILTIN_KEYS or title in seen:
                continue
            seen.add(title)
            titles.append(title)
    return titles


def custom_eval_field_values(case: Dict[str, Any]) -> Dict[str, Any]:
    return {title: value for title, value in case.items() if title not in EVAL_CASE_BUILTIN_KEYS}


def resolve_column_title(title: str) -> str:
    return GENERATED_COLUMN_TITLE_ALIASES.get(title, LEGACY_TO_CURRENT_COLUMN_TITLES.get(title, title))


def find_column_by_title(columns_by_title_map: dict, title: str):
    if not title:
        return None
    column = columns_by_title_map.get(title)
    if column is not None:
        return column
    current = resolve_column_title(title)
    if current != title:
        column = columns_by_title_map.get(current)
        if column is not None:
            return column
    legacy = LEGACY_COLUMN_TITLES.get(current)
    if legacy is not None:
        return columns_by_title_map.get(legacy)
    return None


_DEFAULT_POLL_INTERVAL_SECONDS = 0.5
_DEFAULT_CELL_WAIT_TIMEOUT_SECONDS = 300.0
# LLM scorecard steps (esp. multi-row) routinely need longer than cell wait.
_DEFAULT_SCORE_WAIT_TIMEOUT_SECONDS = 600.0

_DEFAULT_DASHBOARD_BASE_URL = "https://dashboard.promptlayer.com"
_API_HOST_TO_DASHBOARD_HOST = {
    "api.promptlayer.com": "dashboard.promptlayer.com",
    "api.eu.promptlayer.com": "dashboard.eu.promptlayer.com",
    "api.dev.gcp.promptlayer.com": "dashboard.dev.gcp.promptlayer.com",
}


def serialize_cell_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, default=str)


def parse_cell_value(cell: Optional[Dict[str, Any]]) -> Any:
    if not cell:
        return None
    value = cell.get("value")
    if isinstance(value, dict) and "value" in value and len(value) == 1:
        value = value["value"]
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            return value
    if value is not None:
        return value
    display = cell.get("display_value")
    if isinstance(display, str):
        try:
            return json.loads(display)
        except (TypeError, ValueError, json.JSONDecodeError):
            return display
    return display


def extract_rows(payload: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return extract_list(payload, "rows")


def columns_by_title(columns: List[Column]) -> Dict[str, Column]:
    return {column["title"]: column for column in columns if column.get("title")}


def find_last_row(rows_payload: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    rows = extract_rows(rows_payload)
    rows_with_index = [row for row in rows if row.get("row_index") is not None]
    if not rows_with_index:
        return None
    return max(rows_with_index, key=lambda row: int(row["row_index"]))


def extract_row_indices(row_response: Optional[Dict[str, Any]]) -> List[int]:
    if not row_response:
        return []
    if row_response.get("row_index") is not None:
        return [int(row_response["row_index"])]
    rows = row_response.get("rows") or row_response.get("data") or []
    indices = [int(row["row_index"]) for row in rows if isinstance(row, dict) and row.get("row_index") is not None]
    if indices:
        return indices
    added = row_response.get("added_rows") or row_response.get("row_indices")
    if isinstance(added, list) and added:
        return [int(value) for value in added]
    return []


def cell_is_blank(cell: Any) -> bool:
    if not isinstance(cell, dict):
        return True
    display = cell.get("display_value")
    if isinstance(display, str) and display.strip():
        return False
    value = cell.get("value")
    if isinstance(value, dict) and "value" in value:
        value = value["value"]
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, dict)):
        return len(value) == 0
    return False


def row_is_blank_scaffold(row: Dict[str, Any]) -> bool:
    cells = row.get("cells") or {}
    if not isinstance(cells, dict) or not cells:
        return True
    return all(cell_is_blank(cell) for cell in cells.values())


def blank_row_indices(rows_payload: Optional[Dict[str, Any]]) -> List[int]:
    return [
        int(row["row_index"])
        for row in extract_rows(rows_payload)
        if row.get("row_index") is not None and row_is_blank_scaffold(row)
    ]


def find_scaffold_column(existing: List[Column]) -> Optional[Column]:
    scaffold = columns_by_title(existing).get("Column A")
    if not scaffold or scaffold.get("type") not in (None, "TEXT"):
        return None
    if scaffold.get("id") is None:
        return None
    return scaffold


def merge_column(columns: List[Column], column: Column) -> List[Column]:
    column_id = column.get("id")
    merged = [column if c.get("id") == column_id else c for c in columns]
    if not any(c.get("id") == column_id for c in merged):
        merged.append(column)
    return merged


def build_scorer_column_body(
    normalized: Dict[str, Any],
    dependencies: List[Dict[str, Any]],
) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "title": normalized["title"],
        "type": normalized["type"],
    }
    if normalized.get("config") is not None:
        body["config"] = normalized["config"]
    if dependencies:
        body["dependencies"] = dependencies
    return body


def resolve_dashboard_base_url(api_base_url: str) -> str:
    """Map an API base URL to the PromptLayer dashboard origin."""
    override = os.environ.get("PROMPTLAYER_DASHBOARD_URL")
    if override and override.strip():
        return override.strip().rstrip("/")

    parsed = urlparse((api_base_url or "").strip() or "https://api.promptlayer.com")
    host = (parsed.hostname or "").lower()
    if host in _API_HOST_TO_DASHBOARD_HOST:
        dashboard_host = _API_HOST_TO_DASHBOARD_HOST[host]
        return urlunparse((parsed.scheme or "https", dashboard_host, "", "", "", "")).rstrip("/")
    if host in {"localhost", "127.0.0.1"}:
        return "http://localhost:3000"
    if host.startswith("api."):
        dashboard_host = f"dashboard.{host[len('api.') :]}"
        return urlunparse((parsed.scheme or "https", dashboard_host, "", "", "", "")).rstrip("/")
    return _DEFAULT_DASHBOARD_BASE_URL


def build_table_dashboard_url(
    *,
    api_base_url: str,
    workspace_id: Any,
    table_id: Any,
    sheet_id: Any = None,
) -> Optional[str]:
    """Build a dashboard deep link to a Table (optionally a sheet)."""
    if workspace_id is None or table_id is None:
        return None
    base = resolve_dashboard_base_url(api_base_url)
    url = f"{base}/workspace/{workspace_id}/smart-tables/{table_id}"
    if sheet_id is not None:
        url = f"{url}?sheet={sheet_id}"
    return url


def build_row_values(
    columns_by_title_map: Dict[str, Column],
    *,
    input_value: Any,
    expected_value: Any,
    expected_trace_value: Any,
    output_value: Any,
    custom_values: Optional[Dict[str, Any]] = None,
    custom_titles: Optional[List[str]] = None,
) -> Dict[str, Any]:
    values: Dict[str, Any] = {}
    for title, value in (
        ("input", input_value),
        ("expected", expected_value),
        (EXPECTED_TRACE_COLUMN, expected_trace_value),
        ("Output", output_value),
    ):
        column = find_column_by_title(columns_by_title_map, title)
        if not column:
            continue
        values[str(column["id"])] = serialize_cell_value(value if value is not None else "")
    for title in custom_titles or []:
        column = columns_by_title_map.get(title)
        if not column:
            continue
        value = (custom_values or {}).get(title, "")
        values[str(column["id"])] = serialize_cell_value(value if value is not None else "")
    return values


def build_trace_import_body(
    *,
    trace_id: str,
    sheet_id: Any,
    table_id: Any,
    eval_name: str,
) -> Dict[str, Any]:
    from promptlayer.tables.helpers import build_add_trace_body

    return build_add_trace_body(
        {
            "trace_id": trace_id,
            "sheet_id": sheet_id,
            "smart_table_id": table_id,
            "metadata": {"eval_name": eval_name},
        }
    )


def build_case_result(
    *,
    input_value: Any,
    expected_value: Any,
    output_value: Any,
    scores: Dict[str, Any],
    price: Optional[float] = None,
    latency: Optional[float] = None,
    trace_id: str,
    span_id: str,
    row_index: Optional[int],
) -> Dict[str, Any]:
    return {
        "input": input_value,
        "expected": expected_value,
        "output": output_value,
        "scores": scores,
        "price": price,
        "latency": latency,
        "trace_id": trace_id or None,
        "span_id": span_id or None,
        "row_index": row_index,
    }


def cases_from_rows(
    rows_payload: Optional[Dict[str, Any]],
    columns: List[Column],
) -> List[Dict[str, Any]]:
    by_title = columns_by_title(columns)
    input_col = find_column_by_title(by_title, "input")
    expected_col = find_column_by_title(by_title, "expected")
    expected_trace_col = find_column_by_title(by_title, EXPECTED_TRACE_COLUMN)
    custom_columns = [
        column
        for column in columns
        if column.get("type") == "TEXT"
        and isinstance(column.get("title"), str)
        and column["title"].strip()
        and not is_reserved_eval_column_title(column["title"])
    ]
    cases: List[Dict[str, Any]] = []
    for row in extract_rows(rows_payload):
        cells = row.get("cells") or {}
        input_value = None
        expected_value = None
        expected_trace_value = None
        if input_col:
            input_value = parse_cell_value(cells.get(str(input_col["id"])))
        if expected_col:
            expected_value = parse_cell_value(cells.get(str(expected_col["id"])))
        if expected_trace_col:
            expected_trace_value = parse_cell_value(cells.get(str(expected_trace_col["id"])))
        if input_value is None and "input" in row:
            input_value = row.get("input")
        if expected_value is None and "expected" in row:
            expected_value = row.get("expected")
        if expected_trace_value is None and "expected_trace" in row:
            expected_trace_value = row.get("expected_trace")
        if input_value is None:
            continue
        case: Dict[str, Any] = {"input": input_value}
        if expected_value is not None:
            case["expected"] = expected_value
        if expected_trace_value is not None:
            case["expected_trace"] = expected_trace_value
        for column in custom_columns:
            case[column["title"]] = parse_cell_value(cells.get(str(column["id"])))
        cases.append(case)
    return cases
