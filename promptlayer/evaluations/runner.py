import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from opentelemetry.sdk.trace import TracerProvider

from promptlayer.evaluations.polling import (
    afill_row_cells,
    await_for_sheet_operations,
    fill_row_cells,
    wait_for_sheet_operations,
)
from promptlayer.evaluations.scorecard import (
    aconfigure_scorecard_from_scorers,
    afetch_scorecard_row_scores,
    arecalculate_and_wait_scorecard,
    configure_scorecard_from_scorers,
    extract_scorecard_overall_score,
    fetch_scorecard_row_scores,
    recalculate_and_wait_scorecard,
)
from promptlayer.evaluations.scores import (
    case_has_failed_scorer,
    collect_failing_row_indices,
    extract_overall_score,
    scorer_pass_rates,
)
from promptlayer.evaluations.setup import (
    aclear_blank_scaffold_rows,
    aensure_eval_scaffold_columns,
    aresolve_cases,
    aresolve_sheet,
    aresolve_table,
    clear_blank_scaffold_rows,
    ensure_eval_scaffold_columns,
    resolve_cases,
    resolve_sheet,
    resolve_table,
)
from promptlayer.evaluations.terminal import format_score_value, get_terminal
from promptlayer.evaluations.trace_output import resolve_output_from_trace_row
from promptlayer.evaluations.trace_price import (
    await_for_trace_request_price,
    wait_for_trace_request_price,
)
from promptlayer.evaluations.tracing import (
    arun_case_in_span,
    flush_traces,
    maybe_await,
    maybe_await_async,
    run_case_in_span,
)
from promptlayer.evaluations.utils import (
    build_case_result,
    build_row_values,
    build_table_dashboard_url,
    build_trace_import_body,
    columns_by_title,
    custom_eval_field_titles,
    custom_eval_field_values,
    extract_row_indices,
    extract_rows,
    find_last_row,
)
from promptlayer.evaluations.validation import (
    assert_eval_args,
    assert_passing_score,
    validation_error,
)
from promptlayer.tables import api as tables_api
from promptlayer.tables.helpers import extract_columns
from promptlayer.types.table import (
    Column,
    EvalCase,
    EvalCaseResult,
    EvalDataset,
    EvalProcessingColumn,
    EvalResult,
    EvalScorerColumn,
    ResourceId,
)


@dataclass
class CaseExecution:
    input: Any
    expected: Any
    expected_trace: Any
    dataset_fields: Dict[str, Any]
    output: Any
    trace_id: str
    span_id: str


def _emit_status(message: str) -> None:
    get_terminal().step(message)


def _emit_runners_start(total: int) -> None:
    get_terminal().runners_start(total)


def _emit_runner_progress(completed: int, total: int) -> None:
    get_terminal().progress(completed, total)


def _emit_score(score: Any, passing_score: Optional[float]) -> None:
    passed = None
    if passing_score is not None:
        overall = extract_overall_score(score)
        passed = overall is not None and overall >= float(passing_score)
    get_terminal().score(format_score_value(score), passed=passed)


def _emit_dashboard_url(url: Optional[str]) -> None:
    if url:
        get_terminal().link(url)


def _emit_evaluation_summary(
    *,
    case_results: List[EvalCaseResult],
    include_failure_examples: bool,
) -> None:
    terminal = get_terminal()
    rates = scorer_pass_rates(case_results)
    if rates:
        terminal.evaluation_results(rates)

    if include_failure_examples:
        failed_cases = [case for case in case_results if case_has_failed_scorer(case)]
        scorer_titles = [row["scorer"] for row in rates]
        if not scorer_titles:
            for case in case_results:
                for title in case.get("scores") or {}:
                    if title not in scorer_titles:
                        scorer_titles.append(title)
        terminal.failure_examples(failed_cases, scorer_titles=scorer_titles)


def _build_eval_result(
    *,
    name: str,
    table: Dict[str, Any],
    sheet: Dict[str, Any],
    results: List[EvalCaseResult],
    failed_row_indices: List[int],
    score_cards: List[Dict[str, Any]],
    api_base_url: str,
) -> EvalResult:
    url = build_table_dashboard_url(
        api_base_url=api_base_url,
        workspace_id=table.get("workspace_id"),
        table_id=table.get("id"),
        sheet_id=sheet.get("id"),
    )
    result: EvalResult = {
        "name": name,
        "table_id": table["id"],
        "sheet_id": sheet["id"],
        "failed_row_indices": failed_row_indices,
        "score_cards": score_cards,  # type: ignore[typeddict-item]
        "total_rows": len(results),
        "results": results,
    }
    if url:
        result["url"] = url
    return result


def _map_batch_row_indices(
    row_response: Optional[Dict[str, Any]],
    case_count: int,
) -> List[Optional[int]]:
    indices = extract_row_indices(row_response)
    if len(indices) >= case_count:
        return indices[:case_count]
    rows = extract_rows(row_response)
    if len(rows) >= case_count:
        mapped: List[Optional[int]] = []
        for row in rows[:case_count]:
            if isinstance(row, dict) and row.get("row_index") is not None:
                mapped.append(int(row["row_index"]))
            else:
                mapped.append(None)
        return mapped
    raise validation_error(f"Table row creation returned {len(indices)} row indices for {case_count} eval cases.")


def _execute_cases_sync(
    *,
    name: str,
    cases: List[EvalCase],
    runner: Any,
    tracer_provider: Optional[TracerProvider],
    max_concurrency: int,
    table_id: Any = None,
    sheet_id: Any = None,
) -> List[CaseExecution]:
    total = len(cases)
    results: List[Optional[CaseExecution]] = [None] * total

    def _run_one(index: int, case: EvalCase) -> Tuple[int, CaseExecution]:
        input_value = case["input"]
        expected_value = case.get("expected")
        expected_trace_value = case.get("expected_trace")
        if tracer_provider is not None:
            output_value, trace_id, span_id = run_case_in_span(
                name,
                runner,
                input_value,
                tracer_provider,
                table_id=table_id,
                sheet_id=sheet_id,
            )
        else:
            output_value = maybe_await(runner(input_value))
            trace_id = ""
            span_id = ""
        return index, CaseExecution(
            input=input_value,
            expected=expected_value,
            expected_trace=expected_trace_value,
            dataset_fields=custom_eval_field_values(case),
            output=output_value,
            trace_id=trace_id,
            span_id=span_id,
        )

    workers = max(1, min(max_concurrency, total or 1))
    completed = 0
    _emit_runners_start(total)
    if workers == 1:
        for index, case in enumerate(cases):
            result_index, executed = _run_one(index, case)
            results[result_index] = executed
            completed += 1
            _emit_runner_progress(completed, total)
        return [item for item in results if item is not None]

    executor = ThreadPoolExecutor(max_workers=workers)
    futures = [executor.submit(_run_one, index, case) for index, case in enumerate(cases)]
    interrupted = False
    try:
        for future in as_completed(futures):
            index, executed = future.result()
            results[index] = executed
            completed += 1
            _emit_runner_progress(completed, total)
    except KeyboardInterrupt:
        interrupted = True
        for future in futures:
            future.cancel()
        raise
    finally:
        executor.shutdown(wait=not interrupted, cancel_futures=True)

    return [item for item in results if item is not None]


async def _execute_cases_async(
    *,
    name: str,
    cases: List[EvalCase],
    runner: Any,
    tracer_provider: Optional[TracerProvider],
    max_concurrency: int,
    table_id: Any = None,
    sheet_id: Any = None,
) -> List[CaseExecution]:
    total = len(cases)
    results: List[Optional[CaseExecution]] = [None] * total
    semaphore = asyncio.Semaphore(max(1, max_concurrency))

    async def _run_one(index: int, case: EvalCase) -> Tuple[int, CaseExecution]:
        input_value = case["input"]
        expected_value = case.get("expected")
        expected_trace_value = case.get("expected_trace")
        async with semaphore:
            if tracer_provider is not None:
                output_value, trace_id, span_id = await arun_case_in_span(
                    name,
                    runner,
                    input_value,
                    tracer_provider,
                    table_id=table_id,
                    sheet_id=sheet_id,
                )
            else:
                output_value = await maybe_await_async(runner(input_value))
                trace_id = ""
                span_id = ""
        return index, CaseExecution(
            input=input_value,
            expected=expected_value,
            expected_trace=expected_trace_value,
            dataset_fields=custom_eval_field_values(case),
            output=output_value,
            trace_id=trace_id,
            span_id=span_id,
        )

    completed = 0
    _emit_runners_start(total)
    tasks = [asyncio.create_task(_run_one(index, case)) for index, case in enumerate(cases)]
    for task in asyncio.as_completed(tasks):
        index, executed = await task
        results[index] = executed
        completed += 1
        _emit_runner_progress(completed, total)

    return [item for item in results if item is not None]


def _postprocess_trace_import(
    import_response: Any,
    *,
    output_value: Any,
    by_title: Dict[str, Column],
    fallback_row: Optional[Dict[str, Any]],
) -> Tuple[Optional[int], Optional[Dict[str, Any]], Any]:
    trace_row = import_response.get("row") if isinstance(import_response, dict) else None
    row_index = (
        int(import_response["row_index"])
        if isinstance(import_response, dict) and import_response.get("row_index") is not None
        else None
    )
    if trace_row is None:
        trace_row = fallback_row
    resolved_output = output_value
    if trace_row is not None:
        if trace_row.get("row_index") is not None:
            row_index = int(trace_row["row_index"])
        resolved_output = resolve_output_from_trace_row(
            trace_row,
            by_title,
            fallback=output_value,
        )
    return row_index, trace_row, resolved_output


def _persist_trace_rows_sync(
    *,
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: ResourceId,
    sheet_id: ResourceId,
    eval_name: str,
    executed: List[CaseExecution],
    by_title: Dict[str, Column],
    custom_field_titles: List[str],
    tracer_provider: TracerProvider,
) -> Tuple[List[Optional[int]], List[Optional[Dict[str, Any]]], List[CaseExecution]]:
    row_indices: List[Optional[int]] = []
    rows: List[Optional[Dict[str, Any]]] = []
    updated: List[CaseExecution] = []
    for case in executed:
        flush_traces(tracer_provider, throw_on_error=throw_on_error)
        wait_for_trace_request_price(api_key, base_url, case.trace_id)
        import_response = tables_api.add_trace_import(
            api_key,
            base_url,
            throw_on_error,
            build_trace_import_body(
                trace_id=case.trace_id,
                sheet_id=sheet_id,
                table_id=table_id,
                eval_name=eval_name,
            ),
        )
        fallback_row = None
        if not (isinstance(import_response, dict) and import_response.get("row")):
            rows_payload = tables_api.list_smart_sheet_rows(
                api_key,
                base_url,
                throw_on_error,
                table_id,
                sheet_id,
                params={"order": "desc", "limit": 1, "include_columns": False},
            )
            fallback_row = find_last_row(rows_payload)
        row_index, trace_row, resolved_output = _postprocess_trace_import(
            import_response,
            output_value=case.output,
            by_title=by_title,
            fallback_row=fallback_row,
        )
        if trace_row is not None:
            fill_row_cells(
                api_key,
                base_url,
                throw_on_error,
                table_id,
                sheet_id,
                trace_row,
                by_title,
                {
                    "input": case.input,
                    "expected": case.expected,
                    "expected_trace": case.expected_trace,
                    "output": resolved_output,
                    **{title: case.dataset_fields.get(title, "") for title in custom_field_titles},
                },
            )
        row_indices.append(row_index)
        rows.append(trace_row)
        updated.append(
            CaseExecution(
                input=case.input,
                expected=case.expected,
                expected_trace=case.expected_trace,
                dataset_fields=case.dataset_fields,
                output=resolved_output,
                trace_id=case.trace_id,
                span_id=case.span_id,
            )
        )
    return row_indices, rows, updated


async def _persist_trace_rows_async(
    *,
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: ResourceId,
    sheet_id: ResourceId,
    eval_name: str,
    executed: List[CaseExecution],
    by_title: Dict[str, Column],
    custom_field_titles: List[str],
    tracer_provider: TracerProvider,
) -> Tuple[List[Optional[int]], List[Optional[Dict[str, Any]]], List[CaseExecution]]:
    row_indices: List[Optional[int]] = []
    rows: List[Optional[Dict[str, Any]]] = []
    updated: List[CaseExecution] = []
    for case in executed:
        flush_traces(tracer_provider, throw_on_error=throw_on_error)
        await await_for_trace_request_price(api_key, base_url, case.trace_id)
        import_response = await tables_api.aadd_trace_import(
            api_key,
            base_url,
            throw_on_error,
            build_trace_import_body(
                trace_id=case.trace_id,
                sheet_id=sheet_id,
                table_id=table_id,
                eval_name=eval_name,
            ),
        )
        fallback_row = None
        if not (isinstance(import_response, dict) and import_response.get("row")):
            rows_payload = await tables_api.alist_smart_sheet_rows(
                api_key,
                base_url,
                throw_on_error,
                table_id,
                sheet_id,
                params={"order": "desc", "limit": 1, "include_columns": False},
            )
            fallback_row = find_last_row(rows_payload)
        row_index, trace_row, resolved_output = _postprocess_trace_import(
            import_response,
            output_value=case.output,
            by_title=by_title,
            fallback_row=fallback_row,
        )
        if trace_row is not None:
            await afill_row_cells(
                api_key,
                base_url,
                throw_on_error,
                table_id,
                sheet_id,
                trace_row,
                by_title,
                {
                    "input": case.input,
                    "expected": case.expected,
                    "expected_trace": case.expected_trace,
                    "output": resolved_output,
                    **{title: case.dataset_fields.get(title, "") for title in custom_field_titles},
                },
            )
        row_indices.append(row_index)
        rows.append(trace_row)
        updated.append(
            CaseExecution(
                input=case.input,
                expected=case.expected,
                expected_trace=case.expected_trace,
                dataset_fields=case.dataset_fields,
                output=resolved_output,
                trace_id=case.trace_id,
                span_id=case.span_id,
            )
        )
    return row_indices, rows, updated


def _rows_from_batch_response(
    row_response: Any,
    case_count: int,
) -> Tuple[List[Optional[int]], List[Optional[Dict[str, Any]]]]:
    payload = row_response if isinstance(row_response, dict) else None
    row_indices = _map_batch_row_indices(payload, case_count)
    response_rows = extract_rows(payload)
    response_by_index = {
        int(row["row_index"]): row
        for row in response_rows
        if isinstance(row, dict) and row.get("row_index") is not None
    }
    rows: List[Optional[Dict[str, Any]]] = []
    for row_index in row_indices:
        if row_index is None:
            rows.append(None)
            continue
        rows.append(response_by_index.get(row_index))
    return row_indices, rows


def _persist_batch_rows_sync(
    *,
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: ResourceId,
    sheet_id: ResourceId,
    executed: List[CaseExecution],
    by_title: Dict[str, Column],
    custom_field_titles: List[str],
) -> Tuple[List[Optional[int]], List[Optional[Dict[str, Any]]]]:
    values = [
        build_row_values(
            by_title,
            input_value=case.input,
            expected_value=case.expected,
            expected_trace_value=case.expected_trace,
            output_value=case.output,
            custom_values=case.dataset_fields,
            custom_titles=custom_field_titles,
        )
        for case in executed
    ]
    row_response = tables_api.add_smart_sheet_rows(
        api_key,
        base_url,
        throw_on_error,
        table_id,
        sheet_id,
        {"count": len(values), "values": values},
    )
    return _rows_from_batch_response(row_response, len(executed))


async def _persist_batch_rows_async(
    *,
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    table_id: ResourceId,
    sheet_id: ResourceId,
    executed: List[CaseExecution],
    by_title: Dict[str, Column],
    custom_field_titles: List[str],
) -> Tuple[List[Optional[int]], List[Optional[Dict[str, Any]]]]:
    values = [
        build_row_values(
            by_title,
            input_value=case.input,
            expected_value=case.expected,
            expected_trace_value=case.expected_trace,
            output_value=case.output,
            custom_values=case.dataset_fields,
            custom_titles=custom_field_titles,
        )
        for case in executed
    ]
    row_response = await tables_api.aadd_smart_sheet_rows(
        api_key,
        base_url,
        throw_on_error,
        table_id,
        sheet_id,
        {"count": len(values), "values": values},
    )
    return _rows_from_batch_response(row_response, len(executed))


def _build_results(
    executed: List[CaseExecution],
    row_indices: List[Optional[int]],
    scores_by_row: Dict[int, Dict[str, Any]],
) -> List[EvalCaseResult]:
    results: List[EvalCaseResult] = []
    for case, row_index in zip(executed, row_indices):
        results.append(
            build_case_result(  # type: ignore[arg-type]
                input_value=case.input,
                expected_value=case.expected,
                output_value=case.output,
                scores=scores_by_row.get(row_index, {}) if row_index is not None else {},
                trace_id=case.trace_id,
                span_id=case.span_id,
                row_index=row_index,
            )
        )
    return results


@dataclass(frozen=True)
class _EvalRunContext:
    name: str
    dataset: EvalDataset
    runner: Any
    scorers: List[EvalScorerColumn]
    columns: List[EvalProcessingColumn]
    api_key: str
    base_url: str
    throw_on_error: bool
    tracer_provider: Optional[TracerProvider]
    table_id: Optional[ResourceId]
    folder_id: Optional[int]
    experiment_name: Optional[str]
    max_concurrency: int
    passing_score: Optional[float]
    include_failure_examples: bool


@dataclass
class _PreparedEval:
    table: Dict[str, Any]
    sheet: Dict[str, Any]
    columns: List[Column]
    cases: List[EvalCase]
    custom_field_titles: List[str]


def _validate_dataset_field_conflicts(
    custom_field_titles: List[str],
    processing_columns: List[EvalProcessingColumn],
) -> None:
    processing_titles = {column["title"] for column in processing_columns}
    conflicts = [title for title in custom_field_titles if title in processing_titles]
    if conflicts:
        raise validation_error(
            "Eval dataset field(s) conflict with supporting column title(s): "
            + ", ".join(repr(title) for title in conflicts)
        )


def _processing_column_ids(
    sheet_columns: List[Column],
    processing_columns: List[EvalProcessingColumn],
) -> List[str]:
    by_title = columns_by_title(sheet_columns)
    ids: List[str] = []
    for definition in processing_columns:
        column = by_title.get(definition["title"])
        if column is not None and column.get("id") is not None:
            ids.append(str(column["id"]))
    return ids


def _make_eval_context(
    *,
    name: str,
    dataset: EvalDataset,
    runner: Any,
    scorers: List[Any],
    columns: Optional[List[Any]] = None,
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    tracer_provider: Optional[TracerProvider],
    table_id: Optional[ResourceId],
    sheet_id: Optional[ResourceId],
    folder_id: Optional[int],
    experiment_name: Optional[str],
    max_concurrency: int,
    passing_score: Optional[float],
    include_failure_examples: bool = False,
) -> _EvalRunContext:
    normalized_scorers, normalized_columns = assert_eval_args(
        name,
        dataset,
        runner,
        scorers,
        columns=columns,
        table_id=table_id,
        sheet_id=sheet_id,
        folder_id=folder_id,
        experiment_name=experiment_name,
        max_concurrency=max_concurrency,
        passing_score=passing_score,
    )
    return _EvalRunContext(
        name=name,
        dataset=dataset,
        runner=runner,
        scorers=normalized_scorers,
        columns=normalized_columns,
        api_key=api_key,
        base_url=base_url,
        throw_on_error=throw_on_error,
        tracer_provider=tracer_provider,
        table_id=table_id,
        folder_id=folder_id,
        experiment_name=experiment_name,
        max_concurrency=max_concurrency,
        passing_score=passing_score,
        include_failure_examples=bool(include_failure_examples),
    )


def _prepare_eval_sync(context: _EvalRunContext) -> _PreparedEval:
    _emit_status("Resolving Table")
    table = resolve_table(
        context.api_key,
        context.base_url,
        context.throw_on_error,
        name=context.name,
        table_id=context.table_id,
        folder_id=context.folder_id,
    )
    _emit_status("Preparing experiment sheet")
    sheet = resolve_sheet(
        context.api_key,
        context.base_url,
        context.throw_on_error,
        table["id"],
        sheet_id=None,
        experiment_name=context.experiment_name,
        reuse_default_sheet=context.table_id is None,
    )
    _emit_status("Loading dataset")
    cases = resolve_cases(context.api_key, context.base_url, context.throw_on_error, context.dataset)
    if not cases:
        raise validation_error("Eval dataset resolved to zero cases.")
    custom_field_titles = custom_eval_field_titles(cases)
    _validate_dataset_field_conflicts(custom_field_titles, context.columns)

    _emit_status("Setting up columns")
    columns_response = tables_api.list_smart_sheet_columns(
        context.api_key, context.base_url, context.throw_on_error, table["id"], sheet["id"]
    )
    columns = extract_columns(columns_response or {})
    columns = ensure_eval_scaffold_columns(
        context.api_key,
        context.base_url,
        context.throw_on_error,
        table["id"],
        sheet["id"],
        columns,
        include_trace_columns=True,
        include_expected_trace=any(case.get("expected_trace") is not None for case in cases),
        custom_field_titles=custom_field_titles,
        processing_columns=context.columns,
    )
    clear_blank_scaffold_rows(context.api_key, context.base_url, context.throw_on_error, table["id"], sheet["id"])

    _emit_status("Setting up scorers")
    configure_scorecard_from_scorers(
        context.api_key,
        context.base_url,
        context.throw_on_error,
        table["id"],
        sheet["id"],
        columns,
        context.scorers,
        context.name,
    )
    return _PreparedEval(
        table=table,
        sheet=sheet,
        columns=columns,
        cases=cases,
        custom_field_titles=custom_field_titles,
    )


async def _prepare_eval_async(context: _EvalRunContext) -> _PreparedEval:
    _emit_status("Resolving Table")
    table = await aresolve_table(
        context.api_key,
        context.base_url,
        context.throw_on_error,
        name=context.name,
        table_id=context.table_id,
        folder_id=context.folder_id,
    )
    _emit_status("Preparing experiment sheet")
    sheet = await aresolve_sheet(
        context.api_key,
        context.base_url,
        context.throw_on_error,
        table["id"],
        sheet_id=None,
        experiment_name=context.experiment_name,
        reuse_default_sheet=context.table_id is None,
    )
    _emit_status("Loading dataset")
    cases = await aresolve_cases(context.api_key, context.base_url, context.throw_on_error, context.dataset)
    if not cases:
        raise validation_error("Eval dataset resolved to zero cases.")
    custom_field_titles = custom_eval_field_titles(cases)
    _validate_dataset_field_conflicts(custom_field_titles, context.columns)

    _emit_status("Setting up columns")
    columns_response = await tables_api.alist_smart_sheet_columns(
        context.api_key, context.base_url, context.throw_on_error, table["id"], sheet["id"]
    )
    columns = extract_columns(columns_response or {})
    columns = await aensure_eval_scaffold_columns(
        context.api_key,
        context.base_url,
        context.throw_on_error,
        table["id"],
        sheet["id"],
        columns,
        include_trace_columns=True,
        include_expected_trace=any(case.get("expected_trace") is not None for case in cases),
        custom_field_titles=custom_field_titles,
        processing_columns=context.columns,
    )
    await aclear_blank_scaffold_rows(
        context.api_key, context.base_url, context.throw_on_error, table["id"], sheet["id"]
    )

    _emit_status("Setting up scorers")
    await aconfigure_scorecard_from_scorers(
        context.api_key,
        context.base_url,
        context.throw_on_error,
        table["id"],
        sheet["id"],
        columns,
        context.scorers,
        context.name,
    )
    return _PreparedEval(
        table=table,
        sheet=sheet,
        columns=columns,
        cases=cases,
        custom_field_titles=custom_field_titles,
    )


def _score_payload_from_scorecard(scorecard_payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "aggregate_score": extract_scorecard_overall_score(scorecard_payload),
        "scorecard": scorecard_payload.get("scorecard"),
        "latest_calculation": scorecard_payload.get("latest_calculation"),
        "progress": scorecard_payload.get("progress"),
    }


def _finalize_eval(
    context: _EvalRunContext,
    prepared: _PreparedEval,
    executed: List[CaseExecution],
    row_indices: List[Optional[int]],
    scores_by_row: Dict[int, Dict[str, Any]],
    score: Any,
) -> EvalResult:
    case_results = _build_results(executed, row_indices, scores_by_row)
    failed_indices = collect_failing_row_indices(case_results)
    score_cards = scorer_pass_rates(case_results)
    _emit_score(score, context.passing_score)
    _emit_evaluation_summary(
        case_results=case_results,
        include_failure_examples=context.include_failure_examples,
    )
    result = _build_eval_result(
        name=context.name,
        table=prepared.table,
        sheet=prepared.sheet,
        results=case_results,
        failed_row_indices=failed_indices,
        score_cards=score_cards,
        api_base_url=context.base_url,
    )
    assert_passing_score(
        score,
        context.passing_score,
        result=result,
        failing_row_indices=failed_indices,
    )
    _emit_dashboard_url(result.get("url"))
    return result


def run_eval(
    *,
    name: str,
    dataset: EvalDataset,
    runner: Any,
    scorers: List[Any],
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    tracer_provider: Optional[TracerProvider],
    columns: Optional[List[Any]] = None,
    table_id: Optional[ResourceId] = None,
    sheet_id: Optional[ResourceId] = None,
    folder_id: Optional[int] = None,
    experiment_name: Optional[str] = None,
    max_concurrency: int = 1,
    passing_score: Optional[float] = None,
    include_failure_examples: bool = False,
) -> EvalResult:
    context = _make_eval_context(
        name=name,
        dataset=dataset,
        runner=runner,
        scorers=scorers,
        columns=columns,
        api_key=api_key,
        base_url=base_url,
        throw_on_error=throw_on_error,
        tracer_provider=tracer_provider,
        table_id=table_id,
        sheet_id=sheet_id,
        folder_id=folder_id,
        experiment_name=experiment_name,
        max_concurrency=max_concurrency,
        passing_score=passing_score,
        include_failure_examples=include_failure_examples,
    )
    prepared = _prepare_eval_sync(context)
    table, sheet, columns, cases = prepared.table, prepared.sheet, prepared.columns, prepared.cases

    _emit_status(f"Running cases ({len(cases)} case{'s' if len(cases) != 1 else ''}, concurrency={max_concurrency})")
    by_title = columns_by_title(columns)
    executed = _execute_cases_sync(
        name=name,
        cases=cases,
        runner=runner,
        tracer_provider=tracer_provider,
        max_concurrency=max_concurrency,
        table_id=table["id"],
        sheet_id=sheet["id"],
    )

    if context.tracer_provider is not None:
        _emit_status("Importing traces and writing rows")
        row_indices, _rows, executed = _persist_trace_rows_sync(
            api_key=api_key,
            base_url=base_url,
            throw_on_error=throw_on_error,
            table_id=table["id"],
            sheet_id=sheet["id"],
            eval_name=name,
            executed=executed,
            by_title=by_title,
            custom_field_titles=prepared.custom_field_titles,
            tracer_provider=context.tracer_provider,
        )
    else:
        _emit_status("Writing rows")
        row_indices, _rows = _persist_batch_rows_sync(
            api_key=api_key,
            base_url=base_url,
            throw_on_error=throw_on_error,
            table_id=table["id"],
            sheet_id=sheet["id"],
            executed=executed,
            by_title=by_title,
            custom_field_titles=prepared.custom_field_titles,
        )

    processing_ids = _processing_column_ids(columns, context.columns)
    if processing_ids:
        _emit_status("Computing preprocessing columns")
        wait_for_sheet_operations(
            api_key,
            base_url,
            throw_on_error,
            table["id"],
            sheet["id"],
            column_ids=processing_ids,
            row_ids=[index for index in row_indices if index is not None],
        )

    _emit_status("Scoring rows")
    scorecard_payload = recalculate_and_wait_scorecard(
        api_key,
        base_url,
        throw_on_error,
        table["id"],
        sheet["id"],
    )
    scores_by_row = fetch_scorecard_row_scores(
        api_key,
        base_url,
        throw_on_error,
        table["id"],
        sheet["id"],
        row_indices,
        scorecard_payload,
    )
    return _finalize_eval(
        context,
        prepared,
        executed,
        row_indices,
        scores_by_row,
        _score_payload_from_scorecard(scorecard_payload),
    )


@tables_api.reuse_async_client
async def arun_eval(
    *,
    name: str,
    dataset: EvalDataset,
    runner: Any,
    scorers: List[Any],
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    tracer_provider: Optional[TracerProvider],
    columns: Optional[List[Any]] = None,
    table_id: Optional[ResourceId] = None,
    sheet_id: Optional[ResourceId] = None,
    folder_id: Optional[int] = None,
    experiment_name: Optional[str] = None,
    max_concurrency: int = 1,
    passing_score: Optional[float] = None,
    include_failure_examples: bool = False,
) -> EvalResult:
    context = _make_eval_context(
        name=name,
        dataset=dataset,
        runner=runner,
        scorers=scorers,
        columns=columns,
        api_key=api_key,
        base_url=base_url,
        throw_on_error=throw_on_error,
        tracer_provider=tracer_provider,
        table_id=table_id,
        sheet_id=sheet_id,
        folder_id=folder_id,
        experiment_name=experiment_name,
        max_concurrency=max_concurrency,
        passing_score=passing_score,
        include_failure_examples=include_failure_examples,
    )
    prepared = await _prepare_eval_async(context)
    table, sheet, columns, cases = prepared.table, prepared.sheet, prepared.columns, prepared.cases

    _emit_status(f"Running cases ({len(cases)} case{'s' if len(cases) != 1 else ''}, concurrency={max_concurrency})")
    by_title = columns_by_title(columns)
    executed = await _execute_cases_async(
        name=name,
        cases=cases,
        runner=runner,
        tracer_provider=tracer_provider,
        max_concurrency=max_concurrency,
        table_id=table["id"],
        sheet_id=sheet["id"],
    )

    if context.tracer_provider is not None:
        _emit_status("Importing traces and writing rows")
        row_indices, _rows, executed = await _persist_trace_rows_async(
            api_key=api_key,
            base_url=base_url,
            throw_on_error=throw_on_error,
            table_id=table["id"],
            sheet_id=sheet["id"],
            eval_name=name,
            executed=executed,
            by_title=by_title,
            custom_field_titles=prepared.custom_field_titles,
            tracer_provider=context.tracer_provider,
        )
    else:
        _emit_status("Writing rows")
        row_indices, _rows = await _persist_batch_rows_async(
            api_key=api_key,
            base_url=base_url,
            throw_on_error=throw_on_error,
            table_id=table["id"],
            sheet_id=sheet["id"],
            executed=executed,
            by_title=by_title,
            custom_field_titles=prepared.custom_field_titles,
        )

    processing_ids = _processing_column_ids(columns, context.columns)
    if processing_ids:
        _emit_status("Computing preprocessing columns")
        await await_for_sheet_operations(
            api_key,
            base_url,
            throw_on_error,
            table["id"],
            sheet["id"],
            column_ids=processing_ids,
            row_ids=[index for index in row_indices if index is not None],
        )

    _emit_status("Scoring rows")
    scorecard_payload = await arecalculate_and_wait_scorecard(
        api_key,
        base_url,
        throw_on_error,
        table["id"],
        sheet["id"],
    )
    scores_by_row = await afetch_scorecard_row_scores(
        api_key,
        base_url,
        throw_on_error,
        table["id"],
        sheet["id"],
        row_indices,
        scorecard_payload,
    )
    return _finalize_eval(
        context,
        prepared,
        executed,
        row_indices,
        scores_by_row,
        _score_payload_from_scorecard(scorecard_payload),
    )


def _create_client(
    *,
    async_: bool = False,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
):
    kwargs: Dict[str, Any] = {"enable_tracing": True}
    if api_key is not None:
        kwargs["api_key"] = api_key
    if base_url is not None:
        kwargs["base_url"] = base_url
    if async_:
        from promptlayer.promptlayer import AsyncPromptLayer

        return AsyncPromptLayer(**kwargs)
    from promptlayer.promptlayer import PromptLayer

    return PromptLayer(**kwargs)


_EVALUATE_DOC = """Run an eval against a PromptLayer Table.

Auth and host default to ``PROMPTLAYER_API_KEY`` / ``PROMPTLAYER_BASE_URL``.
Pass ``base_url`` for on-prem deployments.

Tracing is always enabled so every eval records and imports its runner trace.

Returns an ``EvalResult`` with table/sheet IDs, failed row indices, score
cards, row results, and the dashboard URL.

Raises ``EvaluationFailedError`` if any scorer cell fails to execute. When
``passing_score`` is set, it also raises if the overall sheet score is missing
or below the threshold. The exception includes ``failing_row_indices`` for
programmatic drill-down.

Set ``include_failure_examples=True`` to print up to five failing rows in the
terminal after the Evaluation Results summary.

Boolean aggregate scores are a normalized pass rate in ``[0.0, 1.0]``
(dataset size does not matter), so ``passing_score=0.8`` means "at least 80%".
Numeric or custom scorers may use a different scale; ``passing_score`` is
compared directly to the sheet's overall score.
"""


def _evaluate_definition(
    name: str,
    *,
    dataset: EvalDataset,
    runner: Any,
    scorers: List[Any],
    columns: Optional[List[Any]] = None,
    table_id: Optional[ResourceId] = None,
    sheet_id: Optional[ResourceId] = None,
    folder_id: Optional[int] = None,
    experiment_name: Optional[str] = None,
    max_concurrency: int = 1,
    passing_score: Optional[float] = None,
    include_failure_examples: bool = False,
) -> Dict[str, Any]:
    return {
        "name": name,
        "dataset": dataset,
        "runner": runner,
        "scorers": scorers,
        "columns": columns,
        "table_id": table_id,
        "sheet_id": sheet_id,
        "folder_id": folder_id,
        "experiment_name": experiment_name,
        "max_concurrency": max_concurrency,
        "passing_score": passing_score,
        "include_failure_examples": include_failure_examples,
    }


def evaluate(
    name: str,
    *,
    dataset: EvalDataset,
    runner: Any,
    scorers: List[Any],
    columns: Optional[List[Any]] = None,
    table_id: Optional[ResourceId] = None,
    sheet_id: Optional[ResourceId] = None,
    folder_id: Optional[int] = None,
    experiment_name: Optional[str] = None,
    max_concurrency: int = 1,
    passing_score: Optional[float] = None,
    include_failure_examples: bool = False,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> EvalResult:
    from promptlayer.evaluations.manager import EvalManager

    _emit_status("Initializing PromptLayer client")
    resolved = _create_client(
        api_key=api_key,
        base_url=base_url,
    )
    return EvalManager(
        api_key=resolved.api_key,
        base_url=resolved.base_url,
        throw_on_error=resolved.throw_on_error,
        tracer_provider=getattr(resolved, "tracer_provider", None),
    ).run(
        _evaluate_definition(
            name,
            dataset=dataset,
            runner=runner,
            scorers=scorers,
            columns=columns,
            table_id=table_id,
            sheet_id=sheet_id,
            folder_id=folder_id,
            experiment_name=experiment_name,
            max_concurrency=max_concurrency,
            passing_score=passing_score,
            include_failure_examples=include_failure_examples,
        )
    )


evaluate.__doc__ = _EVALUATE_DOC


async def aevaluate(
    name: str,
    *,
    dataset: EvalDataset,
    runner: Any,
    scorers: List[Any],
    columns: Optional[List[Any]] = None,
    table_id: Optional[ResourceId] = None,
    sheet_id: Optional[ResourceId] = None,
    folder_id: Optional[int] = None,
    experiment_name: Optional[str] = None,
    max_concurrency: int = 1,
    passing_score: Optional[float] = None,
    include_failure_examples: bool = False,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> EvalResult:
    from promptlayer.evaluations.manager import AsyncEvalManager

    _emit_status("Initializing PromptLayer client")
    resolved = _create_client(
        async_=True,
        api_key=api_key,
        base_url=base_url,
    )
    return await AsyncEvalManager(
        api_key=resolved.api_key,
        base_url=resolved.base_url,
        throw_on_error=resolved.throw_on_error,
        tracer_provider=getattr(resolved, "tracer_provider", None),
    ).run(
        _evaluate_definition(
            name,
            dataset=dataset,
            runner=runner,
            scorers=scorers,
            columns=columns,
            table_id=table_id,
            sheet_id=sheet_id,
            folder_id=folder_id,
            experiment_name=experiment_name,
            max_concurrency=max_concurrency,
            passing_score=passing_score,
            include_failure_examples=include_failure_examples,
        )
    )


aevaluate.__doc__ = "Async variant of :func:`evaluate`."
