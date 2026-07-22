import re
from typing import Any, Dict, List, Optional, Set

from promptlayer.evaluations.utils import (
    BASE_TEXT_COLUMNS,
    EXPECTED_TRACE_COLUMN,
    LEGACY_COLUMN_TITLES,
    TRACE_TEXT_COLUMNS,
    blank_row_indices,
    build_scorer_column_body,
    cases_from_rows,
    columns_by_title,
    find_scaffold_column,
    merge_column,
)
from promptlayer.evaluations.validation import (
    api_error,
    not_found_error,
    scorer_dependencies_from_config,
)
from promptlayer.tables import api as tables_api
from promptlayer.tables.helpers import extract_columns, extract_sheets
from promptlayer.types.table import (
    Column,
    EvalCase,
    EvalDataset,
    EvalProcessingColumn,
    ResourceId,
    Sheet,
    Table,
)


def clear_blank_scaffold_rows(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: ResourceId,
    sheet_id: ResourceId,
) -> None:
    """Delete empty starter rows left by new sheets (usually row 0)."""
    rows_payload = tables_api.list_smart_sheet_rows(
        api_key,
        base_url,
        throw_on_error,
        table_id,
        sheet_id,
        params={"order": "asc", "limit": 20, "include_columns": False},
    )
    blank_indices = blank_row_indices(rows_payload)
    if not blank_indices:
        return
    tables_api.delete_sheet_rows(
        api_key,
        base_url,
        throw_on_error,
        table_id,
        sheet_id,
        {"row_indices": blank_indices},
    )


async def aclear_blank_scaffold_rows(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: ResourceId,
    sheet_id: ResourceId,
) -> None:
    rows_payload = await tables_api.alist_smart_sheet_rows(
        api_key,
        base_url,
        throw_on_error,
        table_id,
        sheet_id,
        params={"order": "asc", "limit": 20, "include_columns": False},
    )
    blank_indices = blank_row_indices(rows_payload)
    if not blank_indices:
        return
    await tables_api.adelete_sheet_rows(
        api_key,
        base_url,
        throw_on_error,
        table_id,
        sheet_id,
        {"row_indices": blank_indices},
    )


def _repurpose_scaffold_column(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: ResourceId,
    sheet_id: ResourceId,
    existing: List[Column],
    desired_title: str,
) -> Optional[Column]:
    """Rename leftover 'Column A' into the first missing eval TEXT column."""
    by_title = columns_by_title(existing)
    if desired_title in by_title:
        return by_title[desired_title]
    scaffold = find_scaffold_column(existing)
    if scaffold is None:
        return None
    update_response = tables_api.update_sheet_column(
        api_key,
        base_url,
        throw_on_error,
        table_id,
        sheet_id,
        scaffold["id"],
        {"title": desired_title},
    )
    updated = (update_response or {}).get("column") or scaffold
    return {**updated, "title": desired_title}


async def _arepurpose_scaffold_column(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: ResourceId,
    sheet_id: ResourceId,
    existing: List[Column],
    desired_title: str,
) -> Optional[Column]:
    by_title = columns_by_title(existing)
    if desired_title in by_title:
        return by_title[desired_title]
    scaffold = find_scaffold_column(existing)
    if scaffold is None:
        return None
    update_response = await tables_api.aupdate_sheet_column(
        api_key,
        base_url,
        throw_on_error,
        table_id,
        sheet_id,
        scaffold["id"],
        {"title": desired_title},
    )
    updated = (update_response or {}).get("column") or scaffold
    return {**updated, "title": desired_title}


def ensure_named_text_columns(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: ResourceId,
    sheet_id: ResourceId,
    existing: List[Column],
    titles: tuple,
) -> List[Column]:
    by_title = columns_by_title(existing)
    columns = list(existing)
    for title in titles:
        if title in by_title:
            continue
        legacy_title = LEGACY_COLUMN_TITLES.get(title)
        if legacy_title and legacy_title in by_title:
            by_title[title] = by_title[legacy_title]
            continue
        repurposed = _repurpose_scaffold_column(api_key, base_url, throw_on_error, table_id, sheet_id, columns, title)
        if repurposed is not None:
            columns = merge_column(columns, repurposed)
            by_title = columns_by_title(columns)
            continue
        create_response = tables_api.create_sheet_column(
            api_key,
            base_url,
            throw_on_error,
            table_id,
            sheet_id,
            {
                "title": title,
                "type": "TEXT",
            },
        )
        if not create_response or not create_response.get("column"):
            raise api_error(f"Failed to create eval column '{title}'.")
        columns.append(create_response["column"])
        by_title[title] = create_response["column"]
    return columns


async def aensure_named_text_columns(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: ResourceId,
    sheet_id: ResourceId,
    existing: List[Column],
    titles: tuple,
) -> List[Column]:
    by_title = columns_by_title(existing)
    columns = list(existing)
    for title in titles:
        if title in by_title:
            continue
        legacy_title = LEGACY_COLUMN_TITLES.get(title)
        if legacy_title and legacy_title in by_title:
            by_title[title] = by_title[legacy_title]
            continue
        repurposed = await _arepurpose_scaffold_column(
            api_key, base_url, throw_on_error, table_id, sheet_id, columns, title
        )
        if repurposed is not None:
            columns = merge_column(columns, repurposed)
            by_title = columns_by_title(columns)
            continue
        create_response = await tables_api.acreate_sheet_column(
            api_key,
            base_url,
            throw_on_error,
            table_id,
            sheet_id,
            {
                "title": title,
                "type": "TEXT",
            },
        )
        if not create_response or not create_response.get("column"):
            raise api_error(f"Failed to create eval column '{title}'.")
        columns.append(create_response["column"])
        by_title[title] = create_response["column"]
    return columns


def ensure_text_columns(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: ResourceId,
    sheet_id: ResourceId,
    existing: List[Column],
    *,
    include_trace_columns: bool = False,
) -> List[Column]:
    titles = BASE_TEXT_COLUMNS + (TRACE_TEXT_COLUMNS if include_trace_columns else ())
    return ensure_named_text_columns(api_key, base_url, throw_on_error, table_id, sheet_id, existing, titles)


async def aensure_text_columns(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: ResourceId,
    sheet_id: ResourceId,
    existing: List[Column],
    *,
    include_trace_columns: bool = False,
) -> List[Column]:
    titles = BASE_TEXT_COLUMNS + (TRACE_TEXT_COLUMNS if include_trace_columns else ())
    return await aensure_named_text_columns(api_key, base_url, throw_on_error, table_id, sheet_id, existing, titles)


def _next_processing_column_body(
    definition: EvalProcessingColumn,
    by_title: Dict[str, Column],
) -> Optional[Dict[str, Any]]:
    title = definition["title"]
    if title in by_title:
        return None
    dependencies = scorer_dependencies_from_config(
        definition.get("config"),
        by_title,
        label="column",
    )
    return build_scorer_column_body(definition, dependencies)


def ensure_processing_columns(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: ResourceId,
    sheet_id: ResourceId,
    existing: List[Column],
    processing_columns: List[EvalProcessingColumn],
) -> List[Column]:
    """Create supporting Table columns in declaration order.

    Later definitions may reference earlier ones by title via config sources.
    """
    by_title = columns_by_title(existing)
    columns = list(existing)
    for definition in processing_columns:
        body = _next_processing_column_body(definition, by_title)
        if body is None:
            continue
        create_response = tables_api.create_sheet_column(
            api_key,
            base_url,
            throw_on_error,
            table_id,
            sheet_id,
            body,  # type: ignore[arg-type]
        )
        if not create_response or not create_response.get("column"):
            raise api_error(f"Failed to create supporting column '{definition['title']}'.")
        columns.append(create_response["column"])
        by_title[definition["title"]] = create_response["column"]
    return columns


async def aensure_processing_columns(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: ResourceId,
    sheet_id: ResourceId,
    existing: List[Column],
    processing_columns: List[EvalProcessingColumn],
) -> List[Column]:
    by_title = columns_by_title(existing)
    columns = list(existing)
    for definition in processing_columns:
        body = _next_processing_column_body(definition, by_title)
        if body is None:
            continue
        create_response = await tables_api.acreate_sheet_column(
            api_key,
            base_url,
            throw_on_error,
            table_id,
            sheet_id,
            body,  # type: ignore[arg-type]
        )
        if not create_response or not create_response.get("column"):
            raise api_error(f"Failed to create supporting column '{definition['title']}'.")
        columns.append(create_response["column"])
        by_title[definition["title"]] = create_response["column"]
    return columns


def ensure_eval_scaffold_columns(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: ResourceId,
    sheet_id: ResourceId,
    existing: List[Column],
    *,
    include_trace_columns: bool = False,
    include_expected_trace: bool = False,
    processing_columns: Optional[List[EvalProcessingColumn]] = None,
) -> List[Column]:
    """Create the full eval column scaffold in declaration order.

    Order: Input/Expected/Output (+ Trace columns) → Expected Trace → supporting columns.
    """
    columns = ensure_text_columns(
        api_key,
        base_url,
        throw_on_error,
        table_id,
        sheet_id,
        existing,
        include_trace_columns=include_trace_columns,
    )
    if include_expected_trace:
        columns = ensure_named_text_columns(
            api_key,
            base_url,
            throw_on_error,
            table_id,
            sheet_id,
            columns,
            (EXPECTED_TRACE_COLUMN,),
        )
    if processing_columns:
        columns = ensure_processing_columns(
            api_key,
            base_url,
            throw_on_error,
            table_id,
            sheet_id,
            columns,
            processing_columns,
        )
    return columns


async def aensure_eval_scaffold_columns(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: ResourceId,
    sheet_id: ResourceId,
    existing: List[Column],
    *,
    include_trace_columns: bool = False,
    include_expected_trace: bool = False,
    processing_columns: Optional[List[EvalProcessingColumn]] = None,
) -> List[Column]:
    columns = await aensure_text_columns(
        api_key,
        base_url,
        throw_on_error,
        table_id,
        sheet_id,
        existing,
        include_trace_columns=include_trace_columns,
    )
    if include_expected_trace:
        columns = await aensure_named_text_columns(
            api_key,
            base_url,
            throw_on_error,
            table_id,
            sheet_id,
            columns,
            (EXPECTED_TRACE_COLUMN,),
        )
    if processing_columns:
        columns = await aensure_processing_columns(
            api_key,
            base_url,
            throw_on_error,
            table_id,
            sheet_id,
            columns,
            processing_columns,
        )
    return columns


def resolve_table(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    *,
    name: str,
    table_id: Optional[ResourceId],
    folder_id: Optional[int],
) -> Table:
    if table_id is not None:
        response = tables_api.get_table(api_key, base_url, throw_on_error, table_id)
        if response and response.get("table"):
            return response["table"]
        raise not_found_error(f"Table '{table_id}' was not found.")
    table = tables_api.upsert_table_by_title(api_key, base_url, throw_on_error, name, folder_id)
    if not table:
        raise api_error(f"Failed to upsert table '{name}'.")
    return table


async def aresolve_table(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    *,
    name: str,
    table_id: Optional[ResourceId],
    folder_id: Optional[int],
) -> Table:
    if table_id is not None:
        response = await tables_api.aget_table(api_key, base_url, throw_on_error, table_id)
        if response and response.get("table"):
            return response["table"]
        raise not_found_error(f"Table '{table_id}' was not found.")
    table = await tables_api.aupsert_table_by_title(api_key, base_url, throw_on_error, name, folder_id)
    if not table:
        raise api_error(f"Failed to upsert table '{name}'.")
    return table


_EXPERIMENT_NUMBER_RE = re.compile(r"^Experiment #(\d+)$")


def _sheet_titles(sheets: List[Sheet]) -> Set[str]:
    return {str(sheet["title"]) for sheet in sheets if sheet.get("title")}


def _default_scaffold_sheet(sheets: List[Sheet]) -> Optional[Sheet]:
    if len(sheets) == 1 and sheets[0].get("title") == "Sheet 1":
        return sheets[0]
    return None


def next_unique_sheet_title(existing_titles: Set[str], base_title: str) -> str:
    """Return base_title, or the next unused '{base} #N' suffix."""
    if base_title not in existing_titles:
        return base_title
    suffix = 2
    while f"{base_title} #{suffix}" in existing_titles:
        suffix += 1
    return f"{base_title} #{suffix}"


def next_experiment_number_title(existing_titles: Set[str], sheet_count_hint: int = 0) -> str:
    """Return the next unused 'Experiment #N' title."""
    used_numbers = set()
    for title in existing_titles:
        match = _EXPERIMENT_NUMBER_RE.match(title)
        if match:
            used_numbers.add(int(match.group(1)))
    candidate = max(sheet_count_hint + 1, 1)
    while candidate in used_numbers or f"Experiment #{candidate}" in existing_titles:
        candidate += 1
    return f"Experiment #{candidate}"


def _create_sheet(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: ResourceId,
    title: str,
) -> Sheet:
    from promptlayer.tables.helpers import empty_sheet_create_body

    create_response = tables_api.create_sheet(
        api_key,
        base_url,
        throw_on_error,
        table_id,
        empty_sheet_create_body(title),
    )
    if not create_response or not create_response.get("sheet"):
        raise api_error(f"Failed to create experiment sheet '{title}'.")
    return create_response["sheet"]


async def _acreate_sheet(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: ResourceId,
    title: str,
) -> Sheet:
    from promptlayer.tables.helpers import empty_sheet_create_body

    create_response = await tables_api.acreate_sheet(
        api_key,
        base_url,
        throw_on_error,
        table_id,
        empty_sheet_create_body(title),
    )
    if not create_response or not create_response.get("sheet"):
        raise api_error(f"Failed to create experiment sheet '{title}'.")
    return create_response["sheet"]


def resolve_sheet(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: ResourceId,
    *,
    sheet_id: Optional[ResourceId],
    experiment_name: Optional[str],
    reuse_default_sheet: bool = False,
) -> Sheet:
    list_response = tables_api.list_sheets(api_key, base_url, throw_on_error, table_id)
    sheets = extract_sheets(list_response or {})

    if sheet_id is not None:
        for sheet in sheets:
            if str(sheet.get("id")) == str(sheet_id):
                return sheet
        raise not_found_error(f"Sheet '{sheet_id}' was not found.")

    scaffold = _default_scaffold_sheet(sheets) if reuse_default_sheet else None
    titles = _sheet_titles([] if scaffold else sheets)
    if experiment_name:
        title = next_unique_sheet_title(titles, experiment_name.strip())
    else:
        title = next_experiment_number_title(titles, sheet_count_hint=len(titles))

    if scaffold is not None:
        update_response = tables_api.update_sheet(
            api_key,
            base_url,
            throw_on_error,
            table_id,
            scaffold["id"],
            {"title": title},
        )
        updated = (update_response or {}).get("sheet")
        if not updated:
            raise api_error(f"Failed to prepare experiment sheet '{title}'.")
        return updated
    return _create_sheet(api_key, base_url, throw_on_error, table_id, title)


async def aresolve_sheet(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: ResourceId,
    *,
    sheet_id: Optional[ResourceId],
    experiment_name: Optional[str],
    reuse_default_sheet: bool = False,
) -> Sheet:
    list_response = await tables_api.alist_sheets(api_key, base_url, throw_on_error, table_id)
    sheets = extract_sheets(list_response or {})

    if sheet_id is not None:
        for sheet in sheets:
            if str(sheet.get("id")) == str(sheet_id):
                return sheet
        raise not_found_error(f"Sheet '{sheet_id}' was not found.")

    scaffold = _default_scaffold_sheet(sheets) if reuse_default_sheet else None
    titles = _sheet_titles([] if scaffold else sheets)
    if experiment_name:
        title = next_unique_sheet_title(titles, experiment_name.strip())
    else:
        title = next_experiment_number_title(titles, sheet_count_hint=len(titles))

    if scaffold is not None:
        update_response = await tables_api.aupdate_sheet(
            api_key,
            base_url,
            throw_on_error,
            table_id,
            scaffold["id"],
            {"title": title},
        )
        updated = (update_response or {}).get("sheet")
        if not updated:
            raise api_error(f"Failed to prepare experiment sheet '{title}'.")
        return updated
    return await _acreate_sheet(api_key, base_url, throw_on_error, table_id, title)


def resolve_cases(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    dataset: EvalDataset,
) -> List[EvalCase]:
    if isinstance(dataset, list):
        return list(dataset)

    table_id = dataset.get("table_id")
    if table_id is None:
        raise not_found_error("Eval dataset table reference requires table_id.")
    sheet_id = dataset.get("sheet_id")
    if sheet_id is None:
        sheet = tables_api.ensure_default_sheet(api_key, base_url, throw_on_error, table_id)
        if not sheet:
            raise api_error("Failed to resolve sheet for dataset table reference.")
        sheet_id = sheet["id"]

    columns_response = tables_api.list_smart_sheet_columns(api_key, base_url, throw_on_error, table_id, sheet_id)
    source_columns = extract_columns(columns_response or {})
    rows_payload = tables_api.list_all_smart_sheet_rows(api_key, base_url, throw_on_error, table_id, sheet_id)
    return cases_from_rows(rows_payload, source_columns)  # type: ignore[return-value]


async def aresolve_cases(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    dataset: EvalDataset,
) -> List[EvalCase]:
    if isinstance(dataset, list):
        return list(dataset)

    table_id = dataset.get("table_id")
    if table_id is None:
        raise not_found_error("Eval dataset table reference requires table_id.")
    sheet_id = dataset.get("sheet_id")
    if sheet_id is None:
        sheet = await tables_api.aensure_default_sheet(api_key, base_url, throw_on_error, table_id)
        if not sheet:
            raise api_error("Failed to resolve sheet for dataset table reference.")
        sheet_id = sheet["id"]

    columns_response = await tables_api.alist_smart_sheet_columns(api_key, base_url, throw_on_error, table_id, sheet_id)
    source_columns = extract_columns(columns_response or {})
    rows_payload = await tables_api.alist_all_smart_sheet_rows(api_key, base_url, throw_on_error, table_id, sheet_id)
    return cases_from_rows(rows_payload, source_columns)  # type: ignore[return-value]
