import base64
from typing import Any, Dict, List, Optional

from promptlayer.types.table import (
    AddTableRows,
    AddTraceImport,
    BatchRecalculateCells,
    ConfigureSheetScore,
    CreateColumn,
    CreateSheet,
    CreateSheetVersion,
    CreateTable,
    DeleteSheetRows,
    ListTablesParams,
    Column,
    Sheet,
    Table,
    UpdateCell,
    UpdateColumn,
    UpdateSheet,
    UpdateTable,
)


def omit_undefined(data: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in data.items() if value is not None}


def _defined_fields(data: Dict[str, Any], *fields: str) -> Dict[str, Any]:
    return omit_undefined({field: data.get(field) for field in fields})


def extract_list(data: Optional[Dict[str, Any]], *entity_keys: str) -> List[Any]:
    if not data:
        return []
    for key in ("data", *entity_keys, "items"):
        value = data.get(key)
        if isinstance(value, list):
            return value
    return []


def extract_tables(data: Dict[str, Any]) -> List[Table]:
    return extract_list(data, "tables")


def extract_sheets(data: Dict[str, Any]) -> List[Sheet]:
    return extract_list(data, "sheets")


def extract_columns(data: Dict[str, Any]) -> List[Column]:
    return extract_list(data, "columns")


def build_list_tables_params(params: Optional[ListTablesParams]) -> Dict[str, Any]:
    return _defined_fields(
        params or {},
        "cursor",
        "limit",
        "name",
        "folder_id",
        "page",
        "per_page",
    )


def build_create_table_body(body: CreateTable) -> Dict[str, Any]:
    return _defined_fields(body, "title", "folder_id")


def build_update_table_body(body: UpdateTable) -> Dict[str, Any]:
    return _defined_fields(body, "title", "folder_id")


def build_create_sheet_body(body: CreateSheet) -> Dict[str, Any]:
    return _defined_fields(body, "title", "index", "operation_id", "source")


def empty_csv_sheet_source(file_name: str = "empty.csv") -> Dict[str, str]:
    """Minimal file source used to create an empty experiment sheet."""
    # Header-only CSV so the import succeeds without seeding eval rows.
    content = base64.b64encode(b"input\n").decode("ascii")
    safe_name = file_name if file_name.endswith((".csv", ".json")) else f"{file_name}.csv"
    return {
        "type": "file",
        "file_name": safe_name,
        "file_content_base64": content,
    }


def with_default_empty_sheet_source(body: Optional[CreateSheet] = None) -> CreateSheet:
    """Ensure a create-sheet body has a source (required by the public API)."""
    payload: CreateSheet = dict(body or {})  # type: ignore[assignment]
    if payload.get("source") is None:
        title = payload.get("title") or "Sheet 1"
        payload["source"] = empty_csv_sheet_source(f"{title}.csv")  # type: ignore[typeddict-item]
    return payload


def empty_sheet_create_body(title: str) -> CreateSheet:
    return with_default_empty_sheet_source({"title": title})


def build_update_sheet_body(body: UpdateSheet) -> Dict[str, Any]:
    return _defined_fields(body, "title")


def build_add_rows_body(body: AddTableRows) -> Dict[str, Any]:
    return _defined_fields(body, "count", "values")


def build_delete_rows_body(body: DeleteSheetRows) -> Dict[str, Any]:
    return _defined_fields(body, "row_indices")


def build_create_column_body(body: CreateColumn) -> Dict[str, Any]:
    return _defined_fields(body, "title", "type", "config", "dependencies", "is_output_column")


def build_update_column_body(body: UpdateColumn) -> Dict[str, Any]:
    return _defined_fields(body, "title", "type", "config", "dependencies", "is_output_column")


def build_update_cell_body(body: UpdateCell) -> Dict[str, Any]:
    return _defined_fields(body, "display_value", "value")


def build_create_version_body(body: CreateSheetVersion) -> Dict[str, Any]:
    return _defined_fields(body, "name")


def build_batch_recalculate_body(body: BatchRecalculateCells) -> Dict[str, Any]:
    return _defined_fields(body, "cell_ids", "column_ids", "row_indices")


def build_add_trace_body(body: AddTraceImport) -> Dict[str, Any]:
    table_id = body.get("smart_table_id", body.get("table_id"))
    return omit_undefined(
        {
            "trace_id": body.get("trace_id"),
            "sheet_id": body.get("sheet_id"),
            "smart_table_id": table_id,
            "span_id": body.get("span_id"),
            "metadata": body.get("metadata"),
        }
    )


def build_configure_score_body(body: ConfigureSheetScore) -> Dict[str, Any]:
    return omit_undefined(dict(body))
