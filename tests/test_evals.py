import asyncio
import threading
import time
from unittest.mock import AsyncMock, patch

import pytest

from promptlayer import (
    PromptLayer,
    PromptLayerValidationError,
    aevaluate,
    code_execution_column,
    column,
    compare_scorer,
    contains_scorer,
    llm_assertion_scorer,
    regex_scorer,
    evaluate,
)
from promptlayer.evaluations.runner import _execute_cases_sync
from promptlayer.evaluations.setup import (
    next_experiment_number_title,
    next_unique_sheet_title,
    resolve_sheet,
)


@pytest.fixture(autouse=True)
def _terminal_sheet_status_counts():
    terminal = {"total_cells": 0, "status_counts": {}}
    with (
        patch("promptlayer.tables.api.get_sheet_status_counts", return_value=terminal),
        patch("promptlayer.tables.api.aget_sheet_status_counts", return_value=terminal),
        patch("opentelemetry.sdk.trace.export.BatchSpanProcessor.on_end"),
        patch("promptlayer.evaluations.runner.wait_for_trace_request_price"),
        patch(
            "promptlayer.evaluations.runner.await_for_trace_request_price",
            new_callable=AsyncMock,
        ),
    ):
        yield


def test_single_worker_executes_inline_for_interruptibility():
    calling_thread = threading.get_ident()
    runner_threads = []

    results = _execute_cases_sync(
        name="inline",
        cases=[{"input": "one"}, {"input": "two"}],
        runner=lambda value: runner_threads.append(threading.get_ident()) or value,
        tracer_provider=None,
        max_concurrency=1,
    )

    assert runner_threads == [calling_thread, calling_thread]
    assert [result.output for result in results] == ["one", "two"]


def test_expected_trace_round_trips_through_eval_rows():
    from promptlayer.evaluations.utils import build_row_values, cases_from_rows

    columns = [
        {"id": "c1", "title": "Input", "type": "TEXT"},
        {"id": "c2", "title": "Expected", "type": "TEXT"},
        {"id": "c3", "title": "Expected Trace", "type": "TEXT"},
        {"id": "c4", "title": "Output", "type": "TEXT"},
    ]
    expected_trace = {"required_tools": [{"tool": "create_folder"}]}
    values = build_row_values(
        {column["title"]: column for column in columns},
        input_value={"question": "Create a folder"},
        expected_value="The folder is created.",
        expected_trace_value=expected_trace,
        output_value="Done",
    )

    assert values["c2"] == "The folder is created."
    assert values["c3"] == '{"required_tools": [{"tool": "create_folder"}]}'
    cases = cases_from_rows(
        {
            "data": [
                {
                    "cells": {
                        "c1": {"value": values["c1"]},
                        "c2": {"value": values["c2"]},
                        "c3": {"value": values["c3"]},
                    }
                }
            ]
        },
        columns,
    )
    assert cases == [
        {
            "input": {"question": "Create a folder"},
            "expected": "The folder is created.",
            "expected_trace": expected_trace,
        }
    ]


def test_generic_column_helpers_build_backend_configs():
    code = code_execution_column("required_tools", code="result = 1")
    assert code["type"] == "CODE_EXECUTION"
    assert code["config"] == {"code": "result = 1", "language": "PYTHON"}

    generic = column("custom", "JSON_PATH", {"source": "Output", "json_path": "$.a"})
    assert generic["type"] == "JSON_PATH"

    from promptlayer import ColumnType

    via_enum = column("summary", ColumnType.JSON_PATH, {"source": "Output", "json_path": "$.summary"})
    assert via_enum["type"] == "JSON_PATH"
    assert via_enum["type"] == ColumnType.JSON_PATH

    assertion = llm_assertion_scorer(
        title="quality",
        source="Output",
        prompt="Is the answer helpful?",
    )
    assert assertion == {
        "title": "quality",
        "type": "LLM_ASSERTION",
        "config": {"source": "Output", "prompt": "Is the answer helpful?"},
    }

    contains = contains_scorer(title="final_answer", source="Output", value="refund")
    assert contains["type"] == "CONTAINS"
    assert contains["config"] == {"source": "Output", "value": "refund"}

    compare = compare_scorer(title="exact_match")
    assert compare["type"] == "COMPARE"
    assert compare["config"]["sources"] == ["Output", "Expected"]
    assert compare["config"]["comparison_type"] == {"type": "STRING"}

    regex = regex_scorer(title="has_id", source="Output", regex_pattern=r"inv_\d+")
    assert regex["type"] == "REGEX"
    assert regex["config"]["regex_pattern"] == r"inv_\d+"


def test_scorer_dependencies_from_config_resolve_titles():
    from promptlayer.evaluations.validation import (
        resolve_config_sources_to_column_ids as _resolve_config_sources_to_column_ids,
        scorer_dependencies_from_config as _scorer_dependencies_from_config,
        scorers_reference_trace as _scorers_reference_trace,
    )

    columns_by_title = {
        "Output": {"id": "out-1", "title": "Output"},
        "Trace": {"id": "tr-1", "title": "Trace"},
        "Expected": {"id": "exp-1", "title": "Expected"},
        "Input": {"id": "in-1", "title": "Input"},
    }
    deps = _scorer_dependencies_from_config(
        {
            "source": "Trace",
            "variable_mappings": {"ground_truth": "Expected"},
        },
        columns_by_title,
    )
    assert deps == [
        {
            "column_id": "tr-1",
            "reference_type": "value",
            "config_key": "source",
        },
        {
            "column_id": "exp-1",
            "reference_type": "value",
            "config_key": "variable_mappings",
            "config_meta": {"variable_name": "ground_truth"},
        },
    ]
    assert _resolve_config_sources_to_column_ids(
        {
            "source": "Output",
            "prompt": "check {user_request}",
            "variable_mappings": {
                "user_request": "Input",
                "execution_trace": "Trace",
            },
        },
        columns_by_title,
    ) == {
        "source": "out-1",
        "prompt": "check {user_request}",
        "variable_mappings": {
            "user_request": "in-1",
            "execution_trace": "tr-1",
        },
    }
    assert _scorers_reference_trace([llm_assertion_scorer(title="x", source="Trace", prompt="ok")])
    assert not _scorers_reference_trace([llm_assertion_scorer(title="x", source="Output", prompt="ok")])
    with pytest.raises(PromptLayerValidationError, match="not found: missing"):
        _scorer_dependencies_from_config({"source": "missing"}, columns_by_title)


def test_build_scorecard_steps_persist_column_ids():
    from promptlayer.evaluations.scorecard import build_scorecard_steps_from_scorers

    columns = [
        {"id": "out-1", "title": "Output"},
        {"id": "in-1", "title": "Input"},
        {"id": "tr-1", "title": "Trace"},
    ]
    steps = build_scorecard_steps_from_scorers(
        [
            llm_assertion_scorer(
                title="Response grounded",
                source="Output",
                prompt="User: {user_request}\nTrace: {execution_trace}",
                variable_mappings={
                    "user_request": "Input",
                    "execution_trace": "Trace",
                },
            )
        ],
        columns,
    )
    assert len(steps) == 1
    step = steps[0]
    assert step["source_column_ids"] == ["out-1", "in-1", "tr-1"]
    assert step["primitive_config"]["source"] == "out-1"
    assert step["primitive_config"]["variable_mappings"] == {
        "user_request": "in-1",
        "execution_trace": "tr-1",
    }
    assert step["primitive_config"]["prompt"] == "User: {user_request}\nTrace: {execution_trace}"


def test_generic_column_helpers_validate_required_fields():
    with pytest.raises(PromptLayerValidationError):
        llm_assertion_scorer(title="x", source="Output")
    with pytest.raises(PromptLayerValidationError):
        contains_scorer(title="x", source="Output")
    with pytest.raises(PromptLayerValidationError):
        compare_scorer(title="x", sources=["only_one"])
    with pytest.raises(PromptLayerValidationError):
        code_execution_column("x", code=" ")


def exact_match(output, expected):
    return 1 if output == expected else 0


def response_length_under_500(output):
    """Pass when the assistant response is under 500 characters."""
    text = "" if output is None else str(output)
    return 1 if len(text) < 500 else 0


def tool_count_under_5(trace):
    """Pass when the Trace subtree has fewer than 5 tool spans."""

    def _count_tools(node):
        if not isinstance(node, dict):
            return 0
        count = 0
        name = node.get("name") or ""
        if isinstance(name, str) and name.startswith("Tool:"):
            count += 1
        for child in node.get("children") or []:
            count += _count_tools(child)
        return count

    return 1 if _count_tools(trace) < 5 else 0


def test_scorer_from_function_builds_code_execution_column():
    from promptlayer.evaluations.columns import scorer_from_function
    from promptlayer.evaluations.validation import normalize_scorer as _normalize_scorer

    column = scorer_from_function(exact_match)
    assert column["type"] == "CODE_EXECUTION"
    assert column["title"] == "exact match"
    assert column["config"]["language"] == "PYTHON"
    code = column["config"]["code"]
    assert "def exact_match" not in code
    assert 'output = data.get("Output")' in code
    assert 'expected = data.get("Expected")' in code
    assert "return _result" in code
    assert "_result = 1 if output == expected else 0" in code
    assert "\nresult = " not in code and not code.startswith("result = ")

    normalized = _normalize_scorer(exact_match)
    assert normalized["type"] == "CODE_EXECUTION"
    assert "def exact_match" not in normalized["config"]["code"]
    assert 'output = data.get("Output")' in normalized["config"]["code"]
    assert "return _result" in normalized["config"]["code"]

    with pytest.raises(PromptLayerValidationError, match="Lambda"):
        scorer_from_function(lambda output: 1)


def test_scorer_from_function_emits_body_only_with_nested_helpers():
    from promptlayer.evaluations.columns import scorer_from_function

    length_col = scorer_from_function(response_length_under_500)
    length_code = length_col["config"]["code"]
    assert "def response_length_under_500" not in length_code
    assert '"""Pass when the assistant response is under 500 characters."""' not in length_code
    assert 'output = data.get("Output")' in length_code
    assert "if output is None else str(output)" in length_code
    assert "_result = 1 if len(text) < 500 else 0" in length_code
    assert "return _result" in length_code

    tool_col = scorer_from_function(tool_count_under_5)
    tool_code = tool_col["config"]["code"]
    assert "def tool_count_under_5" not in tool_code
    assert 'trace = data.get("Trace")' in tool_code
    assert "def _count_tools(node):" in tool_code
    assert "return count" in tool_code
    assert "_result = 1 if _count_tools(trace) < 5 else 0" in tool_code
    assert "return _result" in tool_code


def test_scorer_from_function_supports_data_param_style():
    from promptlayer.evaluations.columns import scorer_from_function

    def length_from_data(data):
        text = "" if data.get("Output") is None else str(data.get("Output"))
        return 1 if len(text) < 500 else 0

    def tools_from_data(data):
        trace = data.get("trace") or {}

        def _count_tools(node):
            if not isinstance(node, dict):
                return 0
            count = 0
            name = node.get("name") or ""
            if isinstance(name, str) and name.startswith("Tool:"):
                count += 1
            for child in node.get("children") or []:
                count += _count_tools(child)
            return count

        return 1 if _count_tools(trace) < 5 else 0

    length_code = scorer_from_function(length_from_data)["config"]["code"]
    assert "data = data.get" not in length_code
    assert 'data.get("Output")' in length_code or "data.get('Output')" in length_code
    assert "return _result" in length_code

    tool_code = scorer_from_function(tools_from_data)["config"]["code"]
    assert "data = data.get" not in tool_code
    assert 'data.get("Trace")' in tool_code or "data.get('Trace')" in tool_code
    assert 'data.get("trace")' not in tool_code and "data.get('trace')" not in tool_code
    assert "return _result" in tool_code


def test_next_unique_sheet_title_and_experiment_number():
    assert next_unique_sheet_title(set(), "Agent v2") == "Agent v2"
    assert next_unique_sheet_title({"Agent v2"}, "Agent v2") == "Agent v2 #2"
    assert next_unique_sheet_title({"Agent v2", "Agent v2 #2"}, "Agent v2") == "Agent v2 #3"
    assert next_experiment_number_title(set(), sheet_count_hint=0) == "Experiment #1"
    assert next_experiment_number_title({"Experiment #1"}, sheet_count_hint=1) == "Experiment #2"
    assert next_experiment_number_title({"Sheet 1", "Experiment #3"}, sheet_count_hint=2) == "Experiment #4"


def test_resolve_sheet_reuses_default_scaffold_for_new_eval_table(
    promptlayer_api_key,
    base_url,
):
    with (
        patch(
            "promptlayer.tables.api.list_sheets",
            return_value={"data": [{"id": "sheet-1", "title": "Sheet 1"}]},
        ),
        patch(
            "promptlayer.tables.api.update_sheet",
            return_value={"sheet": {"id": "sheet-1", "title": "Experiment #1"}},
        ) as mock_update,
        patch("promptlayer.tables.api.create_sheet") as mock_create,
    ):
        sheet = resolve_sheet(
            promptlayer_api_key,
            base_url,
            True,
            "table-1",
            sheet_id=None,
            experiment_name=None,
            reuse_default_sheet=True,
        )

    assert sheet == {"id": "sheet-1", "title": "Experiment #1"}
    mock_update.assert_called_once_with(
        promptlayer_api_key,
        base_url,
        True,
        "table-1",
        "sheet-1",
        {"title": "Experiment #1"},
    )
    mock_create.assert_not_called()


def _completed_row(row_index, cells):
    return {
        "row_index": row_index,
        "cells": {
            column_id: {
                **cell,
                "status": cell.get("status", "COMPLETED"),
            }
            for column_id, cell in cells.items()
        },
    }


def _base_text_columns():
    return [
        {"id": "c1", "title": "Input", "type": "TEXT"},
        {"id": "c2", "title": "Expected", "type": "TEXT"},
        {"id": "c3", "title": "Output", "type": "TEXT"},
    ]


def _scorer_column():
    return {
        "id": "c4",
        "title": "required_tools",
        "type": "CODE_EXECUTION",
    }



def _stub_scorecard_apis(
    mock_configure_scorecard,
    mock_recalculate_scorecard,
    mock_get_scorecard,
    mock_get_scorecard_row,
    *,
    default_raw_value=1,
    aggregate_score=1.0,
    row_scores=None,
):
    """Configure scorecard API mocks for evaluate() tests.

    ``row_scores`` maps row_index -> raw score value (or title -> value dict via
    default). When omitted, every row uses ``default_raw_value``.
    """
    state = {"steps": []}

    def configure_side_effect(*args, **kwargs):
        body = args[5] if len(args) > 5 else kwargs.get("body")
        steps = []
        for index, step in enumerate((body or {}).get("steps") or []):
            steps.append(
                {
                    "id": f"step_{index}",
                    "title": step["title"],
                    "primitive_type": step.get("primitive_type"),
                }
            )
        state["steps"] = steps
        return {
            "success": True,
            "scorecard": {"id": "sc_1", "status": "ready", "steps": steps},
        }

    def get_scorecard_side_effect(*args, **kwargs):
        return {
            "success": True,
            "scorecard": {
                "id": "sc_1",
                "status": "completed",
                "steps": state["steps"],
            },
            "latest_calculation": {
                "id": "calc_1",
                "status": "completed",
                "aggregate_score": aggregate_score,
            },
            "progress": {
                "scored_rows": 1,
                "total_rows": 1,
                "partial_score": aggregate_score,
            },
        }

    def get_row_side_effect(*args, **kwargs):
        row_index = args[5] if len(args) > 5 else kwargs.get("row_index", 0)
        if row_scores is not None and int(row_index) in row_scores:
            raw = row_scores[int(row_index)]
        else:
            raw = default_raw_value
        step_results = {}
        for step in state["steps"]:
            if isinstance(raw, dict) and raw.get("status") == "FAILED":
                step_results[step["id"]] = {
                    "verdict": "error",
                    "error_message": raw.get("error"),
                    "raw_value": raw,
                }
            else:
                failed = raw is False or raw == 0 or raw == 0.0
                if isinstance(raw, dict) and "status" not in raw:
                    # LLM assertion-style payload
                    failed = False
                step_results[step["id"]] = {
                    "score": raw if not isinstance(raw, dict) else None,
                    "verdict": "fail" if failed else "pass",
                    "raw_value": raw,
                }
        return {
            "success": True,
            "row_index": int(row_index),
            "calculation_id": "calc_1",
            "step_results": step_results,
        }

    mock_configure_scorecard.side_effect = configure_side_effect
    mock_recalculate_scorecard.return_value = {
        "success": True,
        "calculation_id": "calc_1",
        "status": "queued",
    }
    mock_get_scorecard.side_effect = get_scorecard_side_effect
    mock_get_scorecard_row.side_effect = get_row_side_effect


@patch("promptlayer.evaluations.runner.flush_traces")
@patch("promptlayer.tables.api.upsert_table_by_title")
@patch("promptlayer.tables.api.list_sheets")
@patch("promptlayer.tables.api.create_sheet")
@patch("promptlayer.tables.api.list_smart_sheet_columns")
@patch("promptlayer.tables.api.create_sheet_column")
@patch("promptlayer.tables.api.configure_sheet_scorecard")
@patch("promptlayer.tables.api.add_smart_sheet_rows")
@patch("promptlayer.tables.api.list_smart_sheet_rows")
@patch("promptlayer.tables.api.delete_sheet_rows")
@patch("promptlayer.tables.api.recalculate_smart_sheet_scorecard")
@patch("promptlayer.tables.api.get_sheet_scorecard")
@patch("promptlayer.tables.api.get_sheet_scorecard_row")
def test_eval_runs_inline_dataset_and_writes_rows(
    mock_get_scorecard_row,
    mock_get_scorecard,
    mock_recalculate_scorecard,
    mock_delete_rows,
    mock_list_rows,
    mock_add_rows,
    mock_configure_scorecard,
    mock_create_column,
    mock_list_columns,
    mock_create_sheet,
    mock_list_sheets,
    mock_upsert_table,
    mock_flush_traces,
    promptlayer_api_key,
    base_url,
):
    mock_upsert_table.return_value = {"id": "1", "title": "Agent Evals", "workspace_id": 99}
    mock_list_sheets.return_value = {"data": []}
    mock_create_sheet.return_value = {"sheet": {"id": "2", "title": "Experiment #1"}}
    mock_list_columns.return_value = {"data": []}

    created = {
        "Input": {"id": "c1", "title": "Input", "type": "TEXT"},
        "Expected": {"id": "c2", "title": "Expected", "type": "TEXT"},
        "Output": {"id": "c3", "title": "Output", "type": "TEXT"},
        "Expected Trace": {"id": "c5", "title": "Expected Trace", "type": "TEXT"},
        "required_tools": _scorer_column(),
    }

    def create_side_effect(*args, **kwargs):
        body = args[5] if len(args) > 5 else kwargs.get("body")
        return {"column": created[body["title"]]}

    mock_create_column.side_effect = create_side_effect
    mock_add_rows.return_value = {
        "rows": [
            _completed_row(
                0,
                {
                    "c1": {"id": "cell-1"},
                    "c2": {"id": "cell-2"},
                    "c3": {"id": "cell-3"},
                    "c4": {"id": "cell-4", "status": "COMPLETED", "value": {"value": 1}},
                    "c5": {"id": "cell-5"},
                },
            )
        ]
    }
    mock_list_rows.return_value = {
        "data": [
            _completed_row(
                0,
                {
                    "c1": {"id": "cell-1"},
                    "c2": {"id": "cell-2"},
                    "c3": {"id": "cell-3"},
                    "c4": {"id": "cell-4", "status": "COMPLETED", "value": {"value": 1}},
                    "c5": {"id": "cell-5"},
                },
            )
        ]
    }
    mock_delete_rows.return_value = {"success": True}
    call_order = []
    _stub_scorecard_apis(
        mock_configure_scorecard,
        mock_recalculate_scorecard,
        mock_get_scorecard,
        mock_get_scorecard_row,
        default_raw_value=1,
        aggregate_score=1.0,
    )
    configure_impl = mock_configure_scorecard.side_effect

    def configure_tracked(*args, **kwargs):
        call_order.append("configure")
        return configure_impl(*args, **kwargs)

    mock_configure_scorecard.side_effect = configure_tracked
    add_rows_response = mock_add_rows.return_value
    mock_add_rows.side_effect = lambda *args, **kwargs: call_order.append("add_rows") or add_rows_response

    failing = evaluate(
        name="Agent Evals",
        dataset=[
            {
                "input": {"userMessage": "refund status"},
                "expected": "The refund status is returned.",
                "expected_trace": {"required_tools": [{"tool": "lookup_invoice"}]},
            }
        ],
        runner=lambda input_data: {
            "final": "refund ok",
            "toolCalls": [{"name": "lookup_invoice"}],
        },
        scorers=[
            code_execution_column(
                "required_tools",
                code="result = 1",
            )
        ],
        api_key=promptlayer_api_key,
        base_url=base_url,
    )

    assert failing["failed_row_indices"] == []
    assert failing["table_id"] == "1"
    assert failing["sheet_id"] == "2"
    assert "table" not in failing
    assert "sheet" not in failing
    assert failing["total_rows"] == 1
    assert failing["score_cards"] == [
        {
            "scorer": "required_tools",
            "passed": 1,
            "total": 1,
            "pass_rate": 1.0,
        }
    ]
    assert failing["url"] == "http://localhost:3000/workspace/99/smart-tables/1?sheet=2"

    mock_create_sheet.assert_called_once()
    create_sheet_body = mock_create_sheet.call_args[0][4]
    assert create_sheet_body["title"] == "Experiment #1"
    assert mock_create_column.call_count == 4
    create_titles = [call[0][5]["title"] for call in mock_create_column.call_args_list]
    assert create_titles[:3] == ["Input", "Expected", "Output"]
    assert create_titles[3:] == ["Expected Trace"]
    add_rows_body = mock_add_rows.call_args[0][5]
    assert add_rows_body["count"] == 1
    values = add_rows_body["values"][0]
    assert "c1" in values and "c2" in values and "c3" in values and "c5" in values
    assert values["c2"] == "The refund status is returned."
    assert values["c5"] == '{"required_tools": [{"tool": "lookup_invoice"}]}'
    mock_configure_scorecard.assert_called_once()
    score_body = mock_configure_scorecard.call_args[0][5]
    assert score_body["steps"][0]["title"] == "required_tools"
    assert score_body["steps"][0]["primitive_type"] == "CODE_EXECUTION"
    assert "column_ids" not in score_body
    assert call_order == ["configure", "add_rows"]
    mock_recalculate_scorecard.assert_called_once()
    mock_get_scorecard.assert_called()
    mock_get_scorecard_row.assert_called()
    mock_flush_traces.assert_not_called()


@patch("promptlayer.evaluations.runner.flush_traces")
@patch("promptlayer.tables.api.upsert_table_by_title")
@patch("promptlayer.tables.api.list_sheets")
@patch("promptlayer.tables.api.create_sheet")
@patch("promptlayer.tables.api.list_smart_sheet_columns")
@patch("promptlayer.tables.api.create_sheet_column")
@patch("promptlayer.tables.api.create_sheet_operation")
@patch("promptlayer.tables.api.get_sheet_operation")
@patch("promptlayer.tables.api.configure_sheet_scorecard")
@patch("promptlayer.tables.api.add_smart_sheet_rows")
@patch("promptlayer.tables.api.list_smart_sheet_rows")
@patch("promptlayer.tables.api.delete_sheet_rows")
@patch("promptlayer.tables.api.recalculate_smart_sheet_scorecard")
@patch("promptlayer.tables.api.get_sheet_scorecard")
@patch("promptlayer.tables.api.get_sheet_scorecard_row")
def test_eval_creates_supporting_columns_and_runs_operations_before_scorecard(
    mock_get_scorecard_row,
    mock_get_scorecard,
    mock_recalculate_scorecard,
    mock_delete_rows,
    mock_list_rows,
    mock_add_rows,
    mock_configure_scorecard,
    mock_get_operation,
    mock_create_operation,
    mock_create_column,
    mock_list_columns,
    mock_create_sheet,
    mock_list_sheets,
    mock_upsert_table,
    mock_flush_traces,
    promptlayer_api_key,
    base_url,
):
    mock_upsert_table.return_value = {"id": "1", "title": "Extract Evals", "workspace_id": 99}
    mock_list_sheets.return_value = {"data": []}
    mock_create_sheet.return_value = {"sheet": {"id": "2", "title": "Experiment #1"}}
    mock_list_columns.return_value = {"data": []}

    created = {
        "Input": {"id": "c1", "title": "Input", "type": "TEXT"},
        "Expected": {"id": "c2", "title": "Expected", "type": "TEXT"},
        "Output": {"id": "c3", "title": "Output", "type": "TEXT"},
        "Extracted data": {
            "id": "c-extract",
            "title": "Extracted data",
            "type": "JSON_PATH",
        },
    }

    call_order = []

    def create_side_effect(*args, **kwargs):
        body = args[5] if len(args) > 5 else kwargs.get("body")
        if body and body.get("title") == "Extracted data":
            call_order.append("create_extract")
        return {"column": created[body["title"]]}

    mock_create_column.side_effect = create_side_effect
    add_rows_response = {
        "rows": [
            _completed_row(
                0,
                {
                    "c1": {"id": "cell-1"},
                    "c2": {"id": "cell-2"},
                    "c3": {"id": "cell-3"},
                    "c-extract": {"id": "cell-extract", "status": "STALE"},
                },
            )
        ],
        "row_indices": [0],
    }
    mock_add_rows.side_effect = lambda *args, **kwargs: call_order.append("add_rows") or add_rows_response
    mock_list_rows.return_value = {"data": []}
    mock_delete_rows.return_value = {"success": True}
    mock_create_operation.side_effect = lambda *args, **kwargs: call_order.append("operations") or {
        "success": True,
        "operation_id": "op_1",
        "execution_id": "op_1",
        "cell_count": 1,
    }
    mock_get_operation.return_value = {
        "operation_id": "op_1",
        "status": "completed",
        "completed_count": 1,
        "failed_count": 0,
    }
    _stub_scorecard_apis(
        mock_configure_scorecard,
        mock_recalculate_scorecard,
        mock_get_scorecard,
        mock_get_scorecard_row,
        default_raw_value=1,
        aggregate_score=1.0,
    )
    configure_impl = mock_configure_scorecard.side_effect

    def configure_tracked(*args, **kwargs):
        call_order.append("configure")
        return configure_impl(*args, **kwargs)

    mock_configure_scorecard.side_effect = configure_tracked
    recalculate_response = mock_recalculate_scorecard.return_value
    mock_recalculate_scorecard.side_effect = (
        lambda *args, **kwargs: call_order.append("recalculate") or recalculate_response
    )

    result = evaluate(
        name="Extract Evals",
        dataset=[{"input": {"payload": {"name": "ada"}}, "expected": "ada"}],
        runner=lambda input_data: input_data,
        columns=[
            column(
                "Extracted data",
                "JSON_PATH",
                {"source": "Output", "json_path": "$.payload.name"},
            )
        ],
        scorers=[
            contains_scorer(
                title="has_name",
                source="Extracted data",
                value="ada",
            )
        ],
        api_key=promptlayer_api_key,
        base_url=base_url,
    )

    assert result["failed_row_indices"] == []
    create_titles = [call[0][5]["title"] for call in mock_create_column.call_args_list]
    assert create_titles == ["Input", "Expected", "Output", "Extracted data"]
    extract_body = next(
        call[0][5] for call in mock_create_column.call_args_list if call[0][5]["title"] == "Extracted data"
    )
    assert extract_body["type"] == "JSON_PATH"
    assert extract_body["dependencies"] == [
        {
            "column_id": "c3",
            "reference_type": "value",
            "config_key": "source",
        }
    ]
    assert not any(
        call[0][5].get("type") == "CODE_EXECUTION" for call in mock_create_column.call_args_list
    )

    mock_create_operation.assert_called_once()
    operation_body = mock_create_operation.call_args[0][5]
    assert operation_body["column_ids"] == ["c-extract"]
    assert operation_body["row_ids"] == [0]
    mock_get_operation.assert_called()

    scorecard_body = mock_configure_scorecard.call_args[0][5]
    assert scorecard_body["steps"][0]["title"] == "has_name"
    assert scorecard_body["steps"][0]["source_column_ids"] == ["c-extract"]
    mock_recalculate_scorecard.assert_called_once()
    assert call_order == [
        "create_extract",
        "configure",
        "add_rows",
        "operations",
        "recalculate",
    ]


@patch("promptlayer.evaluations.runner.flush_traces")
@patch("promptlayer.tables.api.aupsert_table_by_title")
@patch("promptlayer.tables.api.alist_sheets")
@patch("promptlayer.tables.api.acreate_sheet")
@patch("promptlayer.tables.api.alist_smart_sheet_columns")
@patch("promptlayer.tables.api.acreate_sheet_column")
@patch("promptlayer.tables.api.acreate_sheet_operation")
@patch("promptlayer.tables.api.aget_sheet_operation")
@patch("promptlayer.tables.api.aconfigure_sheet_scorecard")
@patch("promptlayer.tables.api.aadd_smart_sheet_rows")
@patch("promptlayer.tables.api.alist_smart_sheet_rows")
@patch("promptlayer.tables.api.adelete_sheet_rows")
@patch("promptlayer.tables.api.arecalculate_smart_sheet_scorecard")
@patch("promptlayer.tables.api.aget_sheet_scorecard")
@patch("promptlayer.tables.api.aget_sheet_scorecard_row")
def test_aeval_creates_supporting_columns_and_runs_operations_before_scorecard(
    mock_get_scorecard_row,
    mock_get_scorecard,
    mock_recalculate_scorecard,
    mock_delete_rows,
    mock_list_rows,
    mock_add_rows,
    mock_configure_scorecard,
    mock_get_operation,
    mock_create_operation,
    mock_create_column,
    mock_list_columns,
    mock_create_sheet,
    mock_list_sheets,
    mock_upsert_table,
    mock_flush_traces,
    promptlayer_api_key,
    base_url,
):
    async def _run():
        mock_upsert_table.return_value = {"id": "1", "title": "Async Extract", "workspace_id": 99}
        mock_list_sheets.return_value = {"data": []}
        mock_create_sheet.return_value = {"sheet": {"id": "2", "title": "Experiment #1"}}
        mock_list_columns.return_value = {"data": []}

        created = {
            "Input": {"id": "c1", "title": "Input", "type": "TEXT"},
            "Expected": {"id": "c2", "title": "Expected", "type": "TEXT"},
            "Output": {"id": "c3", "title": "Output", "type": "TEXT"},
            "Extracted data": {
                "id": "c-extract",
                "title": "Extracted data",
                "type": "JSON_PATH",
            },
        }

        call_order = []

        def create_side_effect(*args, **kwargs):
            body = args[5] if len(args) > 5 else kwargs.get("body")
            if body and body.get("title") == "Extracted data":
                call_order.append("create_extract")
            return {"column": created[body["title"]]}

        mock_create_column.side_effect = create_side_effect
        add_rows_response = {
            "rows": [
                _completed_row(
                    0,
                    {
                        "c1": {"id": "cell-1"},
                        "c2": {"id": "cell-2"},
                        "c3": {"id": "cell-3"},
                        "c-extract": {"id": "cell-extract", "status": "STALE"},
                    },
                )
            ],
            "row_indices": [0],
        }
        mock_add_rows.side_effect = lambda *args, **kwargs: call_order.append("add_rows") or add_rows_response
        mock_list_rows.return_value = {"data": []}
        mock_delete_rows.return_value = {"success": True}
        mock_create_operation.side_effect = lambda *args, **kwargs: call_order.append("operations") or {
            "success": True,
            "operation_id": "op_1",
            "execution_ids": ["op_1"],
            "cell_count": 1,
        }
        mock_get_operation.return_value = {"operation_id": "op_1", "status": "completed"}
        _stub_scorecard_apis(
            mock_configure_scorecard,
            mock_recalculate_scorecard,
            mock_get_scorecard,
            mock_get_scorecard_row,
            default_raw_value=1,
            aggregate_score=1.0,
        )
        configure_impl = mock_configure_scorecard.side_effect

        def configure_tracked(*args, **kwargs):
            call_order.append("configure")
            return configure_impl(*args, **kwargs)

        mock_configure_scorecard.side_effect = configure_tracked
        recalculate_response = mock_recalculate_scorecard.return_value
        mock_recalculate_scorecard.side_effect = (
            lambda *args, **kwargs: call_order.append("recalculate") or recalculate_response
        )

        result = await aevaluate(
            name="Async Extract",
            dataset=[{"input": {"payload": {"name": "ada"}}}],
            runner=lambda input_data: input_data,
            columns=[
                column(
                    "Extracted data",
                    "JSON_PATH",
                    {"source": "Output", "json_path": "$.payload.name"},
                )
            ],
            scorers=[
                contains_scorer(
                    title="has_name",
                    source="Extracted data",
                    value="ada",
                )
            ],
            api_key=promptlayer_api_key,
            base_url=base_url,
        )
        assert result["failed_row_indices"] == []
        mock_create_operation.assert_awaited_once()
        mock_get_operation.assert_awaited()
        scorecard_body = mock_configure_scorecard.await_args[0][5]
        assert scorecard_body["steps"][0]["source_column_ids"] == ["c-extract"]
        assert call_order == [
            "create_extract",
            "configure",
            "add_rows",
            "operations",
            "recalculate",
        ]

    asyncio.run(_run())


@patch("promptlayer.evaluations.runner.flush_traces")
@patch("promptlayer.tables.api.get_table")
@patch("promptlayer.tables.api.list_sheets")
@patch("promptlayer.tables.api.create_sheet")
@patch("promptlayer.tables.api.ensure_default_sheet")
@patch("promptlayer.tables.api.list_smart_sheet_columns")
@patch("promptlayer.tables.api.create_sheet_column")
@patch("promptlayer.tables.api.configure_sheet_scorecard")
@patch("promptlayer.tables.api.add_smart_sheet_rows")
@patch("promptlayer.tables.api.list_smart_sheet_rows")
@patch("promptlayer.tables.api.delete_sheet_rows")
@patch("promptlayer.tables.api.recalculate_smart_sheet_scorecard")
@patch("promptlayer.tables.api.get_sheet_scorecard")
@patch("promptlayer.tables.api.get_sheet_scorecard_row")
def test_eval_resolves_independent_output_and_dataset_tables(
    mock_get_scorecard_row,
    mock_get_scorecard,
    mock_recalculate_scorecard,
    mock_delete_rows,
    mock_list_rows,
    mock_add_rows,
    mock_configure_scorecard,
    mock_create_column,
    mock_list_columns,
    mock_ensure_sheet,
    mock_create_sheet,
    mock_list_sheets,
    mock_get_table,
    mock_flush_traces,
    promptlayer_api_key,
    base_url,
):
    mock_get_table.return_value = {"table": {"id": "10", "title": "PromptLayer Agent Evals"}}
    mock_list_sheets.return_value = {"data": [{"id": "20", "title": "Sheet 1"}]}
    mock_create_sheet.return_value = {"sheet": {"id": "21", "title": "Experiment #2"}}
    mock_ensure_sheet.return_value = {"id": "30", "title": "Sheet 1"}

    eval_columns = _base_text_columns() + [
        {
            "id": "c4",
            "title": "quality",
            "type": "LLM_ASSERTION",
            "config": {"source": "Output", "prompt": "ok?"},
        },
    ]
    dataset_columns = [
        {"id": "d1", "title": "Input", "type": "TEXT"},
        {"id": "d2", "title": "Expected", "type": "TEXT"},
    ]

    def list_columns_side_effect(api_key, base_url, throw_on_error, table_id, sheet_id):
        if str(table_id) == "10":
            return {"data": eval_columns}
        return {"data": dataset_columns}

    dataset_rows = {
        "data": [
            {
                "row_index": 0,
                "cells": {
                    "d1": {"value": {"value": '{"q": "hello"}'}},
                    "d2": {"value": {"value": '{"a": "world"}'}},
                },
            }
        ]
    }
    eval_rows = {
        "data": [
            _completed_row(
                0,
                {
                    "c1": {"id": "cell-1"},
                    "c2": {"id": "cell-2"},
                    "c3": {"id": "cell-3"},
                    "c4": {"id": "cell-4", "status": "COMPLETED"},
                },
            )
        ]
    }

    def list_rows_side_effect(api_key, base_url, throw_on_error, table_id, sheet_id, params=None):
        if str(table_id) == "10":
            return eval_rows
        return dataset_rows

    mock_list_columns.side_effect = list_columns_side_effect
    mock_list_rows.side_effect = list_rows_side_effect
    mock_add_rows.return_value = {"row_indices": [0]}
    mock_delete_rows.return_value = {"success": True}
    _stub_scorecard_apis(
        mock_configure_scorecard,
        mock_recalculate_scorecard,
        mock_get_scorecard,
        mock_get_scorecard_row,
        default_raw_value=1,
        aggregate_score=0.5,
    )

    pl = PromptLayer(api_key=promptlayer_api_key, base_url=base_url, enable_tracing=False)

    failing = pl.evals.run(
        {
            "name": "PromptLayer Agent Evals",
            "table_id": "10",
            "dataset": {"table_id": "99", "sheet_id": "30"},
            "runner": lambda input_data: {"answer": input_data["q"]},
            "scorers": [
                llm_assertion_scorer(
                    title="quality",
                    source="Output",
                    prompt="ok?",
                )
            ],
        }
    )

    assert failing["failed_row_indices"] == []
    mock_create_column.assert_not_called()
    mock_get_table.assert_called()
    mock_ensure_sheet.assert_not_called()
    create_sheet_body = mock_create_sheet.call_args[0][4]
    assert create_sheet_body["title"] == "Experiment #2"
    mock_configure_scorecard.assert_called_once()
    scorecard_body = mock_configure_scorecard.call_args[0][5]
    assert scorecard_body["steps"][0]["title"] == "quality"
    assert scorecard_body["steps"][0]["primitive_type"] == "LLM_ASSERTION"
    mock_recalculate_scorecard.assert_called_once()


def test_eval_manager_honors_definition_overrides(promptlayer_api_key, base_url):
    from promptlayer.evaluations.manager import definition_to_run_kwargs

    kwargs = definition_to_run_kwargs(
        {
            "name": "override-eval",
            "dataset": [{"input": {"q": "1"}}],
            "runner": lambda x: x,
            "scorers": [code_execution_column("ok", code="return 1")],
            "api_key": "definition-key",
            "base_url": "https://definition.example",
            "max_concurrency": 3,
            "passing_score": 0.9,
        },
        api_key=promptlayer_api_key,
        base_url=base_url,
        throw_on_error=True,
        tracer_provider=object(),  # type: ignore[arg-type]
    )

    assert kwargs["api_key"] == "definition-key"
    assert kwargs["base_url"] == "https://definition.example"
    assert kwargs["tracer_provider"] is not None
    assert kwargs["max_concurrency"] == 3
    assert kwargs["passing_score"] == 0.9
    assert kwargs["columns"] is None

    with_columns = definition_to_run_kwargs(
        {
            "name": "override-eval",
            "dataset": [{"input": {"q": "1"}}],
            "runner": lambda x: x,
            "scorers": [code_execution_column("ok", code="return 1")],
            "columns": [column("Extracted data", "JSON_PATH", {"source": "Output", "json_path": "$.a"})],
            "api_key": "definition-key",
            "base_url": "https://definition.example",
        },
        api_key=promptlayer_api_key,
        base_url=base_url,
        throw_on_error=True,
        tracer_provider=object(),  # type: ignore[arg-type]
    )
    assert with_columns["columns"][0]["title"] == "Extracted data"

    with patch(
        "promptlayer.promptlayer_mixins.PromptLayerMixin._initialize_tracer",
        return_value=("tracer-provider", None),
    ):
        enabled = definition_to_run_kwargs(
            {
                "name": "override-eval",
                "dataset": [{"input": {"q": "1"}}],
                "runner": lambda x: x,
                "scorers": [code_execution_column("ok", code="return 1")],
            },
            api_key=promptlayer_api_key,
            base_url=base_url,
            throw_on_error=True,
            tracer_provider=None,
        )
    assert enabled["tracer_provider"] == "tracer-provider"


@patch("promptlayer.evaluations.runner.flush_traces")
@patch("promptlayer.tables.api.get_table")
@patch("promptlayer.tables.api.list_sheets")
@patch("promptlayer.tables.api.create_sheet")
@patch("promptlayer.tables.api.list_smart_sheet_columns")
@patch("promptlayer.tables.api.create_sheet_column")
@patch("promptlayer.tables.api.configure_sheet_scorecard")
@patch("promptlayer.tables.api.add_smart_sheet_rows")
@patch("promptlayer.tables.api.list_smart_sheet_rows")
@patch("promptlayer.tables.api.delete_sheet_rows")
@patch("promptlayer.tables.api.recalculate_smart_sheet_scorecard")
@patch("promptlayer.tables.api.get_sheet_scorecard")
@patch("promptlayer.tables.api.get_sheet_scorecard_row")
def test_eval_creates_named_experiment_sheet_with_suffix(
    mock_get_scorecard_row,
    mock_get_scorecard,
    mock_recalculate_scorecard,
    mock_delete_rows,
    mock_list_rows,
    mock_add_rows,
    mock_configure_scorecard,
    mock_create_column,
    mock_list_columns,
    mock_create_sheet,
    mock_list_sheets,
    mock_get_table,
    mock_flush_traces,
    promptlayer_api_key,
    base_url,
):
    mock_get_table.return_value = {"table": {"id": "10", "title": "Results"}}
    mock_list_sheets.return_value = {
        "data": [
            {"id": "1", "title": "Agent v2"},
            {"id": "2", "title": "Agent v2 #2"},
        ]
    }
    mock_create_sheet.return_value = {"sheet": {"id": "3", "title": "Agent v2 #3"}}
    mock_list_columns.return_value = {"data": _base_text_columns() + [_scorer_column()]}
    mock_add_rows.return_value = {"row_indices": [0]}
    mock_list_rows.return_value = {
        "data": [
            _completed_row(
                0,
                {
                    "c1": {"id": "cell-1"},
                    "c2": {"id": "cell-2"},
                    "c3": {"id": "cell-3"},
                    "c4": {"id": "cell-4", "status": "COMPLETED", "value": 1},
                },
            )
        ]
    }
    mock_delete_rows.return_value = {"success": True}
    _stub_scorecard_apis(
        mock_configure_scorecard,
        mock_recalculate_scorecard,
        mock_get_scorecard,
        mock_get_scorecard_row,
        default_raw_value=1,
        aggregate_score=1.0,
    )

    failing = evaluate(
        name="Results",
        dataset=[{"input": {"q": "a"}}],
        table_id="10",
        experiment_name="Agent v2",
        runner=lambda x: x,
        scorers=[code_execution_column("required_tools", code="result = 1")],
        api_key=promptlayer_api_key,
        base_url=base_url,
    )

    assert failing["failed_row_indices"] == []
    create_sheet_body = mock_create_sheet.call_args[0][4]
    assert create_sheet_body["title"] == "Agent v2 #3"


@patch("promptlayer.evaluations.runner.flush_traces")
@patch("promptlayer.tables.api.get_table")
@patch("promptlayer.tables.api.list_sheets")
@patch("promptlayer.tables.api.create_sheet")
@patch("promptlayer.tables.api.list_smart_sheet_columns")
@patch("promptlayer.tables.api.configure_sheet_scorecard")
@patch("promptlayer.tables.api.add_smart_sheet_rows")
@patch("promptlayer.tables.api.list_smart_sheet_rows")
@patch("promptlayer.tables.api.delete_sheet_rows")
@patch("promptlayer.tables.api.recalculate_smart_sheet_scorecard")
@patch("promptlayer.tables.api.get_sheet_scorecard")
@patch("promptlayer.tables.api.get_sheet_scorecard_row")
def test_eval_rejects_explicit_sheet_id(
    mock_get_scorecard_row,
    mock_get_scorecard,
    mock_recalculate_scorecard,
    mock_delete_rows,
    mock_list_rows,
    mock_add_rows,
    mock_configure_scorecard,
    mock_list_columns,
    mock_create_sheet,
    mock_list_sheets,
    mock_get_table,
    mock_flush_traces,
    promptlayer_api_key,
    base_url,
):
    mock_get_table.return_value = {"table": {"id": "10", "title": "Results"}}
    mock_list_sheets.return_value = {"data": [{"id": "55", "title": "Existing Run"}]}
    mock_list_columns.return_value = {"data": _base_text_columns() + [_scorer_column()]}
    mock_add_rows.return_value = {"row_indices": [0]}
    mock_list_rows.return_value = {
        "data": [
            _completed_row(
                0,
                {
                    "c1": {"id": "cell-1"},
                    "c2": {"id": "cell-2"},
                    "c3": {"id": "cell-3"},
                    "c4": {"id": "cell-4", "status": "COMPLETED", "value": 1},
                },
            )
        ]
    }
    mock_delete_rows.return_value = {"success": True}
    _stub_scorecard_apis(
        mock_configure_scorecard,
        mock_recalculate_scorecard,
        mock_get_scorecard,
        mock_get_scorecard_row,
        default_raw_value=1,
        aggregate_score=1.0,
    )

    with pytest.raises(PromptLayerValidationError, match="dedicated experiment sheet"):
        evaluate(
            name="Results",
            dataset=[{"input": {"q": "a"}}],
            table_id="10",
            sheet_id="55",
            runner=lambda x: x,
            scorers=[code_execution_column("required_tools", code="result = 1")],
            api_key=promptlayer_api_key,
            base_url=base_url,
        )

    mock_create_sheet.assert_not_called()


class _FakeSpanContext:
    def __init__(self):
        self.trace_id = 0xABCDEF0123456789ABCDEF0123456789
        self.span_id = 0x0123456789ABCDEF
        self.is_valid = True


class _FakeSpan:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def set_attribute(self, *args, **kwargs):
        return None

    def record_exception(self, *args, **kwargs):
        return None

    def set_status(self, *args, **kwargs):
        return None

    def get_span_context(self):
        return _FakeSpanContext()


class _FakeTracer:
    def start_as_current_span(self, *args, **kwargs):
        return _FakeSpan()


@patch("promptlayer.evaluations.runner.flush_traces")
@patch("promptlayer.evaluations.tracing.TracerProvider.get_tracer")
@patch("promptlayer.tables.api.upsert_table_by_title")
@patch("promptlayer.tables.api.list_sheets")
@patch("promptlayer.tables.api.create_sheet")
@patch("promptlayer.tables.api.list_smart_sheet_columns")
@patch("promptlayer.tables.api.create_sheet_column")
@patch("promptlayer.tables.api.configure_sheet_scorecard")
@patch("promptlayer.tables.api.add_smart_sheet_rows")
@patch("promptlayer.tables.api.add_trace_import")
@patch("promptlayer.tables.api.list_smart_sheet_rows")
@patch("promptlayer.tables.api.delete_sheet_rows")
@patch("promptlayer.tables.api.update_sheet_cell")
@patch("promptlayer.tables.api.recalculate_smart_sheet_scorecard")
@patch("promptlayer.tables.api.get_sheet_scorecard")
@patch("promptlayer.tables.api.get_sheet_scorecard_row")
def test_eval_with_tracing_creates_trace_row_and_fills_cells(
    mock_get_scorecard_row,
    mock_get_scorecard,
    mock_recalculate_scorecard,
    mock_update_cell,
    mock_delete_rows,
    mock_list_rows,
    mock_add_trace,
    mock_add_rows,
    mock_configure_scorecard,
    mock_create_column,
    mock_list_columns,
    mock_create_sheet,
    mock_list_sheets,
    mock_upsert_table,
    mock_get_tracer,
    mock_flush_traces,
    promptlayer_api_key,
    base_url,
):
    mock_get_tracer.return_value = _FakeTracer()
    mock_upsert_table.return_value = {"id": "1", "title": "Traced Evals"}
    mock_list_sheets.return_value = {"data": []}
    mock_create_sheet.return_value = {"sheet": {"id": "2", "title": "Experiment #1"}}
    mock_list_columns.return_value = {
        "data": [
            {"id": "c1", "title": "Input", "type": "TEXT"},
            {"id": "c2", "title": "Expected", "type": "TEXT"},
            {"id": "c3", "title": "Output", "type": "TEXT"},
            {"id": "c6", "title": "Trace", "type": "TRACE"},
            {
                "id": "c4",
                "title": "pass",
                "type": "CODE_EXECUTION",
            },
        ]
    }
    mock_add_trace.return_value = {"success": True, "rows_added": 1, "mode": "trace"}
    mock_delete_rows.return_value = {"success": True}
    mock_list_rows.return_value = {
        "data": [
            _completed_row(
                5,
                {
                    "c1": {"id": "cell-1"},
                    "c2": {"id": "cell-2"},
                    "c3": {"id": "cell-3"},
                    "c4": {"id": "cell-4", "status": "COMPLETED"},
                    "c6": {"id": "cell-6"},
                },
            )
        ]
    }
    _stub_scorecard_apis(
        mock_configure_scorecard,
        mock_recalculate_scorecard,
        mock_get_scorecard,
        mock_get_scorecard_row,
        default_raw_value=1,
        aggregate_score=1.0,
    )

    failing = evaluate(
        name="Traced Evals",
        dataset=[{"input": {"q": "hi"}, "expected": {"a": "yo"}}],
        runner=lambda input_data: {"answer": "yo"},
        scorers=[code_execution_column("pass", code='return 1 if data.get("Trace") is not None else 0')],
        api_key=promptlayer_api_key,
        base_url=base_url,
    )

    assert failing["failed_row_indices"] == []
    expected_trace_id = format(0xABCDEF0123456789ABCDEF0123456789, "032x")
    trace_body = mock_add_trace.call_args[0][3]
    assert trace_body["trace_id"] == expected_trace_id
    assert trace_body["sheet_id"] == "2"
    assert trace_body["smart_table_id"] == "1"
    assert "span_id" not in trace_body

    mock_add_rows.assert_not_called()
    mock_flush_traces.assert_called_once()

    assert mock_update_cell.call_count == 3
    updated_cell_ids = {call[0][5] for call in mock_update_cell.call_args_list}
    assert updated_cell_ids == {"cell-1", "cell-2", "cell-3"}

    mock_recalculate_scorecard.assert_called_once()
    mock_get_scorecard.assert_called()


@patch("promptlayer.evaluations.runner.flush_traces")
@patch("promptlayer.evaluations.tracing.TracerProvider.get_tracer")
@patch("promptlayer.tables.api.upsert_table_by_title")
@patch("promptlayer.tables.api.list_sheets")
@patch("promptlayer.tables.api.create_sheet")
@patch("promptlayer.tables.api.list_smart_sheet_columns")
@patch("promptlayer.tables.api.configure_sheet_scorecard")
@patch("promptlayer.tables.api.add_smart_sheet_rows")
@patch("promptlayer.tables.api.add_trace_import")
@patch("promptlayer.tables.api.list_smart_sheet_rows")
@patch("promptlayer.tables.api.delete_sheet_rows")
@patch("promptlayer.tables.api.update_sheet_cell")
@patch("promptlayer.tables.api.recalculate_smart_sheet_scorecard")
@patch("promptlayer.tables.api.get_sheet_scorecard")
@patch("promptlayer.tables.api.get_sheet_scorecard_row")
def test_eval_serializes_trace_imports_even_with_concurrency(
    mock_get_scorecard_row,
    mock_get_scorecard,
    mock_recalculate_scorecard,
    mock_update_cell,
    mock_delete_rows,
    mock_list_rows,
    mock_add_trace,
    mock_add_rows,
    mock_configure_scorecard,
    mock_list_columns,
    mock_create_sheet,
    mock_list_sheets,
    mock_upsert_table,
    mock_get_tracer,
    mock_flush_traces,
    promptlayer_api_key,
    base_url,
):
    mock_get_tracer.return_value = _FakeTracer()
    mock_upsert_table.return_value = {"id": "1", "title": "Traced Evals"}
    mock_list_sheets.return_value = {"data": []}
    mock_create_sheet.return_value = {"sheet": {"id": "2", "title": "Experiment #1"}}
    mock_list_columns.return_value = {
        "data": _base_text_columns()
        + [
            {"id": "c6", "title": "Trace", "type": "TRACE"},
            {
                "id": "c4",
                "title": "pass",
                "type": "CODE_EXECUTION",
            },
        ]
    }
    active_imports = {"count": 0, "max": 0, "lock": threading.Lock()}

    def add_trace_side_effect(*args, **kwargs):
        with active_imports["lock"]:
            active_imports["count"] += 1
            active_imports["max"] = max(active_imports["max"], active_imports["count"])
        time.sleep(0.05)
        with active_imports["lock"]:
            active_imports["count"] -= 1
        return {"success": True, "rows_added": 1}

    imported_rows = []

    def add_trace_and_row(*args, **kwargs):
        add_trace_side_effect(*args, **kwargs)
        idx = len(imported_rows) + 1
        imported_rows.append(
            _completed_row(
                idx,
                {
                    "c1": {"id": f"cell-1-{idx}"},
                    "c2": {"id": f"cell-2-{idx}"},
                    "c3": {"id": f"cell-3-{idx}"},
                    "c4": {"id": f"cell-4-{idx}", "status": "COMPLETED"},
                    "c6": {"id": f"cell-6-{idx}"},
                },
            )
        )
        return {"success": True, "rows_added": 1}

    mock_add_trace.side_effect = add_trace_and_row
    mock_delete_rows.return_value = {"success": True}

    def list_rows_side_effect(*args, **kwargs):
        params = kwargs.get("params") or (args[5] if len(args) > 5 else None) or {}
        if params.get("order") == "desc" and params.get("limit") == 1:
            return {"data": [imported_rows[-1]] if imported_rows else []}
        return {"data": list(imported_rows)}

    mock_list_rows.side_effect = list_rows_side_effect
    _stub_scorecard_apis(
        mock_configure_scorecard,
        mock_recalculate_scorecard,
        mock_get_scorecard,
        mock_get_scorecard_row,
        default_raw_value=1,
        aggregate_score=1.0,
    )

    failing = evaluate(
        name="Traced Evals",
        dataset=[{"input": {"q": "a"}}, {"input": {"q": "b"}}, {"input": {"q": "c"}}],
        runner=lambda input_data: input_data,
        scorers=[code_execution_column("pass", code='return 1 if data.get("Trace") is not None else 0')],
        max_concurrency=3,
        api_key=promptlayer_api_key,
        base_url=base_url,
    )

    assert failing["failed_row_indices"] == []
    assert active_imports["max"] == 1
    assert mock_add_trace.call_count == 3
    mock_add_rows.assert_not_called()


@patch("promptlayer.evaluations.runner.flush_traces")
@patch("promptlayer.tables.api.upsert_table_by_title")
@patch("promptlayer.tables.api.list_sheets")
@patch("promptlayer.tables.api.create_sheet")
@patch("promptlayer.tables.api.list_smart_sheet_columns")
@patch("promptlayer.tables.api.configure_sheet_scorecard")
@patch("promptlayer.tables.api.add_smart_sheet_rows")
@patch("promptlayer.tables.api.list_smart_sheet_rows")
@patch("promptlayer.tables.api.delete_sheet_rows")
@patch("promptlayer.tables.api.recalculate_smart_sheet_scorecard")
@patch("promptlayer.tables.api.get_sheet_scorecard")
@patch("promptlayer.tables.api.get_sheet_scorecard_row")
def test_eval_bounded_concurrency_preserves_order_and_batches_rows(
    mock_get_scorecard_row,
    mock_get_scorecard,
    mock_recalculate_scorecard,
    mock_delete_rows,
    mock_list_rows,
    mock_add_rows,
    mock_configure_scorecard,
    mock_list_columns,
    mock_create_sheet,
    mock_list_sheets,
    mock_upsert_table,
    mock_flush_traces,
    promptlayer_api_key,
    base_url,
    capsys,
):
    mock_upsert_table.return_value = {"id": "1", "title": "Concurrent"}
    mock_list_sheets.return_value = {"data": []}
    mock_create_sheet.return_value = {"sheet": {"id": "2", "title": "Experiment #1"}}
    mock_list_columns.return_value = {"data": _base_text_columns() + [_scorer_column()]}
    mock_delete_rows.return_value = {"success": True}
    mock_add_rows.return_value = {
        "rows": [
            _completed_row(
                0,
                {
                    "c1": {"id": "cell-1a"},
                    "c2": {"id": "cell-2a"},
                    "c3": {"id": "cell-3a"},
                    "c4": {"id": "cell-4a", "status": "COMPLETED", "value": 1},
                },
            ),
            _completed_row(
                1,
                {
                    "c1": {"id": "cell-1b"},
                    "c2": {"id": "cell-2b"},
                    "c3": {"id": "cell-3b"},
                    "c4": {"id": "cell-4b", "status": "COMPLETED", "value": 1},
                },
            ),
            _completed_row(
                2,
                {
                    "c1": {"id": "cell-1c"},
                    "c2": {"id": "cell-2c"},
                    "c3": {"id": "cell-3c"},
                    "c4": {"id": "cell-4c", "status": "COMPLETED", "value": 1},
                },
            ),
        ]
    }
    mock_list_rows.return_value = mock_add_rows.return_value
    _stub_scorecard_apis(
        mock_configure_scorecard,
        mock_recalculate_scorecard,
        mock_get_scorecard,
        mock_get_scorecard_row,
        default_raw_value=1,
        aggregate_score=1.0,
    )

    active = {"count": 0, "max": 0, "lock": threading.Lock()}

    def runner_fn(input_data):
        with active["lock"]:
            active["count"] += 1
            active["max"] = max(active["max"], active["count"])
        time.sleep(0.08)
        with active["lock"]:
            active["count"] -= 1
        return {"answer": input_data["q"]}

    failing = evaluate(
        name="Concurrent",
        dataset=[
            {"input": {"q": "a"}},
            {"input": {"q": "b"}},
            {"input": {"q": "c"}},
        ],
        runner=runner_fn,
        scorers=[code_execution_column("required_tools", code="result = 1")],
        max_concurrency=2,
        api_key=promptlayer_api_key,
        base_url=base_url,
    )

    assert failing["failed_row_indices"] == []
    assert active["max"] == 2
    progress = [line.strip() for line in capsys.readouterr().out.splitlines() if "runners" in line]
    assert progress[0].endswith("runners 0/3")
    assert progress[-1] == "✓ runners 3/3"
    assert any(line.endswith("runners 1/3") for line in progress)
    assert any(line.endswith("runners 2/3") for line in progress)
    mock_add_rows.assert_called_once()
    add_body = mock_add_rows.call_args[0][5]
    assert add_body["count"] == 3
    assert len(add_body["values"]) == 3
    mock_recalculate_scorecard.assert_called_once()


@patch("promptlayer.evaluations.runner.flush_traces")
@patch("promptlayer.tables.api.aupsert_table_by_title")
@patch("promptlayer.tables.api.alist_sheets")
@patch("promptlayer.tables.api.acreate_sheet")
@patch("promptlayer.tables.api.alist_smart_sheet_columns")
@patch("promptlayer.tables.api.aconfigure_sheet_scorecard")
@patch("promptlayer.tables.api.aadd_smart_sheet_rows")
@patch("promptlayer.tables.api.alist_smart_sheet_rows")
@patch("promptlayer.tables.api.adelete_sheet_rows")
@patch("promptlayer.tables.api.arecalculate_smart_sheet_scorecard")
@patch("promptlayer.tables.api.aget_sheet_scorecard")
@patch("promptlayer.tables.api.aget_sheet_scorecard_row")
def test_async_eval_bounded_concurrency_preserves_order(
    mock_get_scorecard_row,
    mock_get_scorecard,
    mock_recalculate_scorecard,
    mock_delete_rows,
    mock_list_rows,
    mock_add_rows,
    mock_configure_scorecard,
    mock_list_columns,
    mock_create_sheet,
    mock_list_sheets,
    mock_upsert_table,
    mock_flush_traces,
    promptlayer_api_key,
    base_url,
):
    async def _run():
        mock_upsert_table.return_value = {"id": "1", "title": "Async Concurrent"}
        mock_list_sheets.return_value = {"data": []}
        mock_create_sheet.return_value = {"sheet": {"id": "2", "title": "Experiment #1"}}
        mock_list_columns.return_value = {"data": _base_text_columns() + [_scorer_column()]}
        mock_delete_rows.return_value = {"success": True}
        mock_add_rows.return_value = {
            "row_indices": [0, 1, 2],
            "rows": [
                _completed_row(
                    idx,
                    {
                        "c1": {"id": f"cell-1-{idx}"},
                        "c2": {"id": f"cell-2-{idx}"},
                        "c3": {"id": f"cell-3-{idx}"},
                        "c4": {"id": f"cell-4-{idx}", "status": "COMPLETED", "value": 1},
                    },
                )
                for idx in range(3)
            ],
        }
        mock_list_rows.return_value = mock_add_rows.return_value
        _stub_scorecard_apis(
            mock_configure_scorecard,
            mock_recalculate_scorecard,
            mock_get_scorecard,
            mock_get_scorecard_row,
            default_raw_value=1,
            aggregate_score=1.0,
        )

        active = {"count": 0, "max": 0}

        async def runner_fn(input_data):
            active["count"] += 1
            active["max"] = max(active["max"], active["count"])
            await asyncio.sleep(0.05)
            active["count"] -= 1
            return {"answer": input_data["q"]}

        failing = await aevaluate(
            name="Async Concurrent",
            dataset=[
                {"input": {"q": "a"}},
                {"input": {"q": "b"}},
                {"input": {"q": "c"}},
            ],
            runner=runner_fn,
            scorers=[code_execution_column("required_tools", code="result = 1")],
            max_concurrency=2,
            api_key=promptlayer_api_key,
            base_url=base_url,
        )
        assert failing["failed_row_indices"] == []
        assert active["max"] == 2
        mock_add_rows.assert_awaited_once()
        add_body = mock_add_rows.await_args[0][5]
        assert add_body["count"] == 3

    asyncio.run(_run())


def test_eval_validates_definition(promptlayer_api_key, base_url):
    with pytest.raises(PromptLayerValidationError):
        evaluate(
            name="",
            dataset=[{"input": {}}],
            runner=lambda x: x,
            scorers=[code_execution_column("x", code="result = 1")],
            api_key=promptlayer_api_key,
            base_url=base_url,
        )
    with pytest.raises(PromptLayerValidationError):
        evaluate(
            name="x",
            dataset=[],
            runner=lambda x: x,
            scorers=[code_execution_column("x", code="result = 1")],
            api_key=promptlayer_api_key,
            base_url=base_url,
        )
    with pytest.raises(PromptLayerValidationError):
        evaluate(
            name="x",
            dataset=[{"input": {}}],
            runner=lambda x: x,
            scorers=[],
            api_key=promptlayer_api_key,
            base_url=base_url,
        )
    with pytest.raises(PromptLayerValidationError, match="dataset"):
        evaluate(
            name="x",
            dataset="My Dataset",  # type: ignore[arg-type]
            runner=lambda x: x,
            scorers=[code_execution_column("x", code="result = 1")],
            api_key=promptlayer_api_key,
            base_url=base_url,
        )
    with pytest.raises(PromptLayerValidationError, match="table"):
        evaluate(
            name="x",
            dataset={"table": "My Dataset"},  # type: ignore[arg-type]
            runner=lambda x: x,
            scorers=[code_execution_column("x", code="result = 1")],
            api_key=promptlayer_api_key,
            base_url=base_url,
        )
    with pytest.raises(PromptLayerValidationError, match="folder_id"):
        evaluate(
            name="x",
            dataset=[{"input": {}}],
            table_id="1",
            folder_id=2,
            runner=lambda x: x,
            scorers=[code_execution_column("x", code="result = 1")],
            api_key=promptlayer_api_key,
            base_url=base_url,
        )
    with pytest.raises(PromptLayerValidationError, match="sheet_id"):
        evaluate(
            name="x",
            dataset=[{"input": {}}],
            sheet_id="1",
            experiment_name="Agent v2",
            runner=lambda x: x,
            scorers=[code_execution_column("x", code="result = 1")],
            api_key=promptlayer_api_key,
            base_url=base_url,
        )
    with pytest.raises(PromptLayerValidationError, match="max_concurrency"):
        evaluate(
            name="x",
            dataset=[{"input": {}}],
            runner=lambda x: x,
            scorers=[code_execution_column("x", code="result = 1")],
            max_concurrency=0,
            api_key=promptlayer_api_key,
            base_url=base_url,
        )
    with pytest.raises(PromptLayerValidationError, match="passing_score"):
        evaluate(
            name="x",
            dataset=[{"input": {}}],
            runner=lambda x: x,
            scorers=[code_execution_column("x", code="result = 1")],
            passing_score="high",  # type: ignore[arg-type]
            api_key=promptlayer_api_key,
            base_url=base_url,
        )
    with pytest.raises(PromptLayerValidationError, match="reserved"):
        evaluate(
            name="x",
            dataset=[{"input": {}}],
            runner=lambda x: x,
            columns=[column("output", "JSON_PATH", {"source": "Input", "json_path": "$"})],
            scorers=[code_execution_column("ok", code="result = 1")],
            api_key=promptlayer_api_key,
            base_url=base_url,
        )
    with pytest.raises(PromptLayerValidationError, match="conflicts with a supporting column"):
        evaluate(
            name="x",
            dataset=[{"input": {}}],
            runner=lambda x: x,
            columns=[column("shared", "JSON_PATH", {"source": "Output", "json_path": "$"})],
            scorers=[code_execution_column("shared", code="result = 1")],
            api_key=promptlayer_api_key,
            base_url=base_url,
        )
    with pytest.raises(PromptLayerValidationError, match="callables are only supported in scorers"):
        evaluate(
            name="x",
            dataset=[{"input": {}}],
            runner=lambda x: x,
            columns=[lambda data: 1],  # type: ignore[list-item]
            scorers=[code_execution_column("ok", code="result = 1")],
            api_key=promptlayer_api_key,
            base_url=base_url,
        )
    with pytest.raises(TypeError):
        evaluate(
            name="x",
            data=[{"input": {}}],  # type: ignore[call-arg]
            runner=lambda x: x,
            scorers=[code_execution_column("x", code="result = 1")],
            api_key=promptlayer_api_key,
            base_url=base_url,
        )


def test_assert_passing_score_raises_when_below_threshold():
    from promptlayer import EvaluationFailedError
    from promptlayer.evaluations.scores import extract_overall_score
    from promptlayer.evaluations.terminal import format_score_value
    from promptlayer.evaluations.utils import build_table_dashboard_url, resolve_dashboard_base_url
    from promptlayer.evaluations.validation import assert_passing_score

    assert extract_overall_score({"aggregate_score": 0.8, "overall_score": 0.8}) == 0.8
    assert extract_overall_score({"aggregate": {"type": "boolean", "value": 0.42}}) == 0.42
    assert extract_overall_score({"aggregate": {"success_count": 8, "total_count": 10}}) == 0.8
    assert extract_overall_score({"columns": [{"score": 1.0}, {"score": 0.0}]}) == 0.5
    assert extract_overall_score({"score": {"score": 1.0}}) == 1.0
    assert (
        format_score_value(
            {
                "status": "completed",
                "aggregate_score": 0.8,
                "aggregate": {"type": "boolean", "value": 0.8, "success_count": 8, "total_count": 10},
            }
        )
        == "0.8 (8/10)"
    )
    assert format_score_value({"status": "completed", "aggregate_score": 1.0}) == "1.0"

    assert resolve_dashboard_base_url("https://api.promptlayer.com") == "https://dashboard.promptlayer.com"
    assert (
        build_table_dashboard_url(
            api_base_url="https://api.promptlayer.com",
            workspace_id=42,
            table_id="tbl-1",
            sheet_id="sheet-2",
        )
        == "https://dashboard.promptlayer.com/workspace/42/smart-tables/tbl-1?sheet=sheet-2"
    )

    assert_passing_score({"score": {"score": 1.0}}, None)
    assert_passing_score({"aggregate_score": 1.0}, 1.0)
    assert_passing_score({"aggregate_score": 0.8}, 0.8)
    assert_passing_score({"aggregate_score": 0.9}, 0.5)

    with pytest.raises(
        EvaluationFailedError,
        match=r"overall score 0\.42 is below passing score 0\.8",
    ) as exc_info:
        assert_passing_score({"aggregate_score": 0.42}, 0.8)
    assert exc_info.value.passing_score == 0.8

    with pytest.raises(
        EvaluationFailedError,
        match=r"overall score 0\.0 is below passing score 1\.0",
    ):
        assert_passing_score({"score": {"score": 0.0}}, 1.0)

    with pytest.raises(EvaluationFailedError, match="overall score is missing"):
        assert_passing_score({"status": "completed"}, 1.0)

    with pytest.raises(EvaluationFailedError, match="overall score is missing"):
        assert_passing_score(None, 0.5)


def test_assert_passing_score_includes_sheet_url_and_failing_indices():
    from promptlayer import EvaluationFailedError
    from promptlayer.evaluations.validation import assert_passing_score

    result = {
        "name": "Score regression",
        "url": "https://dashboard.promptlayer.com/workspace/1/smart-tables/t?sheet=s",
        "results": [
            {
                "row_index": 2,
                "scores": {"trajectory assertions v3": 0},
            }
        ],
    }

    with pytest.raises(EvaluationFailedError) as exc_info:
        assert_passing_score(
            {"aggregate_score": 0.0},
            1.0,
            result=result,
            failing_row_indices=[2],
        )

    message = str(exc_info.value)
    assert "Evaluation 'Score regression' failed" in message
    assert "Row 2" not in message
    assert "Trace mismatch" not in message
    assert "Inspect the sheet:" in message
    assert exc_info.value.result == result
    assert exc_info.value.failing_row_indices == [2]


def test_assert_passing_score_fails_on_failed_cell_without_threshold():
    from promptlayer import EvaluationFailedError
    from promptlayer.evaluations.validation import assert_passing_score

    result = {
        "name": "Cell failure eval",
        "url": "https://dashboard.promptlayer.com/sheet",
        "results": [
            {"row_index": 3, "scores": {"working scorer": 1}},
            {
                "row_index": 4,
                "scores": {
                    "broken scorer": {
                        "status": "FAILED",
                        "error": "Scorer execution failed",
                    }
                },
            },
        ],
    }

    with pytest.raises(EvaluationFailedError, match="scorecard evaluators failed to execute") as exc_info:
        assert_passing_score({"aggregate_score": 1.0}, None, result=result)

    assert exc_info.value.passing_score is None
    assert exc_info.value.failing_row_indices == [4]
    assert "Inspect the sheet:" in str(exc_info.value)


def test_format_scorer_value_shows_llm_assertion_verdict():
    from promptlayer.evaluations.terminal import format_scorer_value

    assert (
        format_scorer_value(
            {
                "Pass when the response is professional": {
                    "value": True,
                    "reasoning": "The response is clear.",
                }
            }
        )
        == "true"
    )
    assert (
        format_scorer_value(
            {
                "Check coherence": {
                    "value": False,
                    "reasoning": "The trace is incomplete.",
                }
            }
        )
        == "false"
    )
    assert format_scorer_value({"custom": "object"}) == '{"custom": "object"}'


def test_assert_passing_score_omits_row_diagnostics():
    from promptlayer import EvaluationFailedError
    from promptlayer.evaluations.validation import assert_passing_score

    result = {
        "name": "LLM assertion eval",
        "url": "https://dashboard.promptlayer.com/workspace/1/smart-tables/t?sheet=s",
        "results": [
            {
                "row_index": 4,
                "scores": {
                    "LLM assertion": {
                        "Is the answer helpful?": {
                            "value": False,
                            "reasoning": "The output never answers the user question.",
                        }
                    },
                    "Exact match": True,
                },
            }
        ],
    }

    with pytest.raises(EvaluationFailedError) as exc_info:
        assert_passing_score(
            {"aggregate_score": 0.0},
            1.0,
            result=result,
            failing_row_indices=[4],
        )

    message = str(exc_info.value)
    assert "overall score 0.0 is below passing score 1.0" in message
    assert "Row 4" not in message
    assert "The output never answers the user question." not in message
    assert exc_info.value.failing_row_indices == [4]


def test_collect_failing_row_indices_and_pass_rates():
    from promptlayer.evaluations.terminal import format_pass_rate
    from promptlayer.evaluations.validation import collect_failing_row_indices, scorer_pass_rates

    cases = [
        {"row_index": 1, "scores": {"Tool check": 1, "Trajectory": 0}},
        {"row_index": 2, "scores": {"Tool check": 1, "Trajectory": 1}},
        {"row_index": 3, "scores": {"Tool check": 0, "Trajectory": 0}},
    ]
    assert collect_failing_row_indices(cases) == [1, 3]
    assert scorer_pass_rates(cases) == [
        {"scorer": "Tool check", "passed": 2, "total": 3, "pass_rate": 2 / 3},
        {"scorer": "Trajectory", "passed": 1, "total": 3, "pass_rate": 1 / 3},
    ]
    assert format_pass_rate(2, 3) == "2/3 (67%)"
    assert format_pass_rate(1, 2) == "1/2 (50%)"


@patch("promptlayer.evaluations.runner.flush_traces")
@patch("promptlayer.tables.api.upsert_table_by_title")
@patch("promptlayer.tables.api.list_sheets")
@patch("promptlayer.tables.api.create_sheet")
@patch("promptlayer.tables.api.list_smart_sheet_columns")
@patch("promptlayer.tables.api.create_sheet_column")
@patch("promptlayer.tables.api.configure_sheet_scorecard")
@patch("promptlayer.tables.api.add_smart_sheet_rows")
@patch("promptlayer.tables.api.add_trace_import")
@patch("promptlayer.tables.api.list_smart_sheet_rows")
@patch("promptlayer.tables.api.delete_sheet_rows")
@patch("promptlayer.tables.api.update_sheet_cell")
@patch("promptlayer.tables.api.recalculate_smart_sheet_scorecard")
@patch("promptlayer.tables.api.get_sheet_scorecard")
@patch("promptlayer.tables.api.get_sheet_scorecard_row")
def test_evaluate_raises_rich_trajectory_failure(
    mock_get_scorecard_row,
    mock_get_scorecard,
    mock_recalculate_scorecard,
    mock_update_cell,
    mock_delete_rows,
    mock_list_rows,
    mock_add_trace,
    mock_add_rows,
    mock_configure_scorecard,
    mock_create_column,
    mock_list_columns,
    mock_create_sheet,
    mock_list_sheets,
    mock_upsert_table,
    mock_flush_traces,
    promptlayer_api_key,
    base_url,
):
    from promptlayer import EvaluationFailedError, trajectory_scorer

    expected = {
        "required_tools": [
            {
                "tool": "create_folder",
                "success": True,
                "output_fields": {"folder.name": "wanted"},
            }
        ],
    }
    trace = {
        "name": "root",
        "children": [
            {
                "name": "Tool: create_folder",
                "output": '{"success": true, "folder": {"name": "wrong"}}',
                "children": [],
            }
        ],
    }

    mock_upsert_table.return_value = {"id": "1", "title": "Trajectory Evals", "workspace_id": 13}
    mock_list_sheets.return_value = {"data": []}
    mock_create_sheet.return_value = {"sheet": {"id": "2", "title": "Experiment #1"}}
    mock_list_columns.return_value = {
        "data": [
            {"id": "c1", "title": "Input", "type": "TEXT"},
            {"id": "c2", "title": "Expected", "type": "TEXT"},
            {"id": "c3", "title": "Output", "type": "TEXT"},
            {"id": "c6", "title": "Trace", "type": "TRACE"},
            {"id": "c4", "title": "trajectory assertions v3", "type": "TRAJECTORY"},
        ]
    }
    mock_delete_rows.return_value = {"success": True}
    mock_add_trace.return_value = {"success": True, "row_index": 7, "rows_added": 1}
    mock_add_rows.return_value = {
        "rows": [
            _completed_row(
                7,
                {
                    "c1": {"id": "cell-1"},
                    "c2": {"id": "cell-2"},
                    "c3": {"id": "cell-3"},
                    "c4": {"id": "cell-4", "status": "COMPLETED", "value": 0},
                    "c6": {"id": "cell-6", "value": trace},
                },
            )
        ]
    }
    mock_list_rows.return_value = {
        "data": [
            _completed_row(
                7,
                {
                    "c1": {"id": "cell-1"},
                    "c2": {"id": "cell-2"},
                    "c3": {"id": "cell-3"},
                    "c4": {"id": "cell-4", "status": "COMPLETED", "value": 0},
                    "c6": {"id": "cell-6", "value": trace},
                },
            )
        ]
    }
    _stub_scorecard_apis(
        mock_configure_scorecard,
        mock_recalculate_scorecard,
        mock_get_scorecard,
        mock_get_scorecard_row,
        default_raw_value=0,
        aggregate_score=0.0,
    )

    with pytest.raises(EvaluationFailedError) as exc_info:
        evaluate(
            name="Trajectory Evals",
            dataset=[{"input": {"question": "create folder"}, "expected": expected}],
            runner=lambda input_data: "ok",
            scorers=[trajectory_scorer(expected_source="expected", title="trajectory assertions v3")],
            api_key=promptlayer_api_key,
            base_url=base_url,
            passing_score=1.0,
        )

    message = str(exc_info.value)
    assert "Evaluation 'Trajectory Evals' failed" in message
    assert "overall score 0.0 is below passing score 1.0" in message
    assert "Row 7" not in message
    assert "Trace mismatch" not in message
    assert "Inspect the sheet:" in message
    assert exc_info.value.passing_score == 1.0
    assert exc_info.value.failing_row_indices == [7]
    assert exc_info.value.result["table_id"] == "1"
    assert exc_info.value.result["sheet_id"] == "2"
    assert exc_info.value.result["failed_row_indices"] == [7]
    updated_cell_ids = {call[0][5] for call in mock_update_cell.call_args_list}
    assert updated_cell_ids == {"cell-1", "cell-2", "cell-3"}


@patch("promptlayer.evaluations.runner.flush_traces")
@patch("promptlayer.tables.api.upsert_table_by_title")
@patch("promptlayer.tables.api.list_sheets")
@patch("promptlayer.tables.api.create_sheet")
@patch("promptlayer.tables.api.list_smart_sheet_columns")
@patch("promptlayer.tables.api.create_sheet_column")
@patch("promptlayer.tables.api.configure_sheet_scorecard")
@patch("promptlayer.tables.api.add_smart_sheet_rows")
@patch("promptlayer.tables.api.list_smart_sheet_rows")
@patch("promptlayer.tables.api.delete_sheet_rows")
@patch("promptlayer.tables.api.update_sheet_cell")
@patch("promptlayer.tables.api.recalculate_smart_sheet_scorecard")
@patch("promptlayer.tables.api.get_sheet_scorecard")
@patch("promptlayer.tables.api.get_sheet_scorecard_row")
def test_evaluate_returns_failing_row_indices_without_passing_score(
    mock_get_scorecard_row,
    mock_get_scorecard,
    mock_recalculate_scorecard,
    mock_update_cell,
    mock_delete_rows,
    mock_list_rows,
    mock_add_rows,
    mock_configure_scorecard,
    mock_create_column,
    mock_list_columns,
    mock_create_sheet,
    mock_list_sheets,
    mock_upsert_table,
    mock_flush_traces,
    promptlayer_api_key,
    base_url,
    capsys,
):
    from promptlayer import llm_assertion_scorer

    mock_upsert_table.return_value = {"id": "1", "title": "Explain Evals", "workspace_id": 7}
    mock_list_sheets.return_value = {"data": []}
    mock_create_sheet.return_value = {"sheet": {"id": "2", "title": "Experiment #1"}}
    mock_list_columns.return_value = {"data": _base_text_columns() + [_scorer_column()]}
    mock_list_columns.return_value["data"][-1] = {
        "id": "c4",
        "title": "LLM assertion",
        "type": "LLM_ASSERTION",
    }
    row = _completed_row(
        3,
        {
            "c1": {"id": "cell-1"},
            "c2": {"id": "cell-2"},
            "c3": {"id": "cell-3"},
            "c4": {
                "id": "cell-4",
                "status": "COMPLETED",
                "value": {
                    "Is helpful?": {
                        "value": False,
                        "reasoning": "The answer ignores the question.",
                    }
                },
            },
        },
    )
    mock_add_rows.return_value = {"rows": [row]}
    mock_list_rows.return_value = {"data": [row]}
    mock_delete_rows.return_value = {"success": True}
    _stub_scorecard_apis(
        mock_configure_scorecard,
        mock_recalculate_scorecard,
        mock_get_scorecard,
        mock_get_scorecard_row,
        default_raw_value={
            "Is helpful?": {
                "value": False,
                "reasoning": "The answer ignores the question.",
            }
        },
        aggregate_score=0.0,
        row_scores={
            3: {
                "Is helpful?": {
                    "value": False,
                    "reasoning": "The answer ignores the question.",
                }
            }
        },
    )

    failing = evaluate(
        name="Explain Evals",
        dataset=[{"input": {"q": "hi"}, "expected": {"a": "yo"}}],
        runner=lambda input_data: "nope",
        scorers=[llm_assertion_scorer(prompt="Is helpful?", title="LLM assertion")],
        api_key=promptlayer_api_key,
        base_url=base_url,
    )

    assert failing["failed_row_indices"] == [3]
    assert mock_update_cell.call_count == 0
    out = capsys.readouterr().out
    assert "Evaluation Results:" in out
    assert "LLM assertion" in out
    assert "0/1 (0%)" in out
    assert "Failure examples:" not in out
    assert "\nPassed\n" not in out
    assert "\nFailed\n" not in out


@patch("promptlayer.evaluations.runner.flush_traces")
@patch("promptlayer.tables.api.upsert_table_by_title")
@patch("promptlayer.tables.api.list_sheets")
@patch("promptlayer.tables.api.create_sheet")
@patch("promptlayer.tables.api.list_smart_sheet_columns")
@patch("promptlayer.tables.api.create_sheet_column")
@patch("promptlayer.tables.api.configure_sheet_scorecard")
@patch("promptlayer.tables.api.add_smart_sheet_rows")
@patch("promptlayer.tables.api.list_smart_sheet_rows")
@patch("promptlayer.tables.api.delete_sheet_rows")
@patch("promptlayer.tables.api.update_sheet_cell")
@patch("promptlayer.tables.api.recalculate_smart_sheet_scorecard")
@patch("promptlayer.tables.api.get_sheet_scorecard")
@patch("promptlayer.tables.api.get_sheet_scorecard_row")
def test_evaluate_include_failure_examples_prints_first_five(
    mock_get_scorecard_row,
    mock_get_scorecard,
    mock_recalculate_scorecard,
    mock_update_cell,
    mock_delete_rows,
    mock_list_rows,
    mock_add_rows,
    mock_configure_scorecard,
    mock_create_column,
    mock_list_columns,
    mock_create_sheet,
    mock_list_sheets,
    mock_upsert_table,
    mock_flush_traces,
    promptlayer_api_key,
    base_url,
    capsys,
):
    mock_upsert_table.return_value = {"id": "1", "title": "Examples Evals", "workspace_id": 7}
    mock_list_sheets.return_value = {"data": []}
    mock_create_sheet.return_value = {"sheet": {"id": "2", "title": "Experiment #1"}}
    mock_list_columns.return_value = {"data": _base_text_columns() + [_scorer_column()]}
    rows = [
        _completed_row(
            index,
            {
                "c1": {"id": f"cell-1-{index}"},
                "c2": {"id": f"cell-2-{index}"},
                "c3": {"id": f"cell-3-{index}"},
                "c4": {"id": f"cell-4-{index}", "status": "COMPLETED", "value": 0 if index < 6 else 1},
            },
        )
        for index in range(7)
    ]
    mock_add_rows.return_value = {"rows": rows}
    mock_list_rows.return_value = {"data": rows}
    mock_delete_rows.return_value = {"success": True}
    _stub_scorecard_apis(
        mock_configure_scorecard,
        mock_recalculate_scorecard,
        mock_get_scorecard,
        mock_get_scorecard_row,
        default_raw_value=1,
        aggregate_score=0.14,
        row_scores={index: (0 if index < 6 else 1) for index in range(7)},
    )

    failing = evaluate(
        name="Examples Evals",
        dataset=[{"input": {"q": f"q{i}"}} for i in range(7)],
        runner=lambda input_data: f"out-{input_data['q']}",
        scorers=[code_execution_column("required_tools", code="result = 0")],
        api_key=promptlayer_api_key,
        base_url=base_url,
        include_failure_examples=True,
    )

    assert failing["failed_row_indices"] == [0, 1, 2, 3, 4, 5]
    out = capsys.readouterr().out
    assert "Failure examples:" in out
    assert "Input" in out
    assert "Output" in out
    assert "required_tools" in out
    assert "├" in out
    # Only first five failing examples are printed.
    assert out.count("out-q") <= 5


@patch("promptlayer.evaluations.runner.flush_traces")
@patch("promptlayer.tables.api.upsert_table_by_title")
@patch("promptlayer.tables.api.list_sheets")
@patch("promptlayer.tables.api.create_sheet")
@patch("promptlayer.tables.api.list_smart_sheet_columns")
@patch("promptlayer.tables.api.create_sheet_column")
@patch("promptlayer.tables.api.configure_sheet_scorecard")
@patch("promptlayer.tables.api.add_smart_sheet_rows")
@patch("promptlayer.tables.api.list_smart_sheet_rows")
@patch("promptlayer.tables.api.delete_sheet_rows")
@patch("promptlayer.tables.api.update_sheet_cell")
@patch("promptlayer.tables.api.recalculate_smart_sheet_scorecard")
@patch("promptlayer.tables.api.get_sheet_scorecard")
@patch("promptlayer.tables.api.get_sheet_scorecard_row")
def test_evaluate_leaves_no_failure_writes_for_passing_rows(
    mock_get_scorecard_row,
    mock_get_scorecard,
    mock_recalculate_scorecard,
    mock_update_cell,
    mock_delete_rows,
    mock_list_rows,
    mock_add_rows,
    mock_configure_scorecard,
    mock_create_column,
    mock_list_columns,
    mock_create_sheet,
    mock_list_sheets,
    mock_upsert_table,
    mock_flush_traces,
    promptlayer_api_key,
    base_url,
):
    mock_upsert_table.return_value = {"id": "1", "title": "Pass Evals", "workspace_id": 7}
    mock_list_sheets.return_value = {"data": []}
    mock_create_sheet.return_value = {"sheet": {"id": "2", "title": "Experiment #1"}}
    mock_list_columns.return_value = {"data": _base_text_columns() + [_scorer_column()]}
    row = _completed_row(
        1,
        {
            "c1": {"id": "cell-1"},
            "c2": {"id": "cell-2"},
            "c3": {"id": "cell-3"},
            "c4": {"id": "cell-4", "status": "COMPLETED", "value": 1},
        },
    )
    mock_add_rows.return_value = {"rows": [row]}
    mock_list_rows.return_value = {"data": [row]}
    mock_delete_rows.return_value = {"success": True}
    _stub_scorecard_apis(
        mock_configure_scorecard,
        mock_recalculate_scorecard,
        mock_get_scorecard,
        mock_get_scorecard_row,
        default_raw_value=1,
        aggregate_score=1.0,
    )

    failing = evaluate(
        name="Pass Evals",
        dataset=[{"input": {"q": "hi"}}],
        runner=lambda input_data: "ok",
        scorers=[code_execution_column("required_tools", code="result = 1")],
        api_key=promptlayer_api_key,
        base_url=base_url,
        passing_score=1.0,
    )

    assert failing["failed_row_indices"] == []
    assert mock_update_cell.call_count == 0


def test_extract_last_assistant_message_string():
    from promptlayer.evaluations.trace_output import extract_last_assistant_message

    trace = {
        "name": "root",
        "start": "2024-01-01T00:00:00Z",
        "children": [
            {
                "name": "llm-1",
                "start": "2024-01-01T00:00:01Z",
                "request_log": {
                    "request_response": {
                        "choices": [
                            {
                                "message": {
                                    "role": "assistant",
                                    "content": "first reply",
                                }
                            }
                        ]
                    }
                },
                "children": [],
            },
            {
                "name": "llm-2",
                "start": "2024-01-01T00:00:02Z",
                "request_log": {
                    "request_response": {
                        "choices": [
                            {
                                "message": {
                                    "role": "assistant",
                                    "content": "final reply",
                                }
                            }
                        ]
                    }
                },
                "children": [],
            },
        ],
    }
    assert extract_last_assistant_message(trace) == "final reply"


def test_extract_last_assistant_message_string_plus_tool():
    from promptlayer.evaluations.trace_output import extract_last_assistant_message

    tool_calls = [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "search", "arguments": '{"q": "x"}'},
        }
    ]
    trace = {
        "name": "root",
        "start": "2024-01-01T00:00:00Z",
        "children": [
            {
                "name": "llm",
                "start": "2024-01-01T00:00:01Z",
                "request_log": {
                    "request_response": {
                        "choices": [
                            {
                                "message": {
                                    "role": "assistant",
                                    "content": "looking that up",
                                    "tool_calls": tool_calls,
                                }
                            }
                        ]
                    }
                },
                "children": [],
            }
        ],
    }
    assert extract_last_assistant_message(trace) == {
        "content": "looking that up",
        "tool_calls": tool_calls,
    }


def test_extract_last_assistant_message_json():
    from promptlayer.evaluations.trace_output import extract_last_assistant_message

    trace = {
        "name": "root",
        "start": "2024-01-01T00:00:00Z",
        "request_log": {
            "request_response": {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": '{"answer": 42, "ok": true}',
                        }
                    }
                ]
            }
        },
        "children": [],
    }
    assert extract_last_assistant_message(trace) == {"answer": 42, "ok": True}


def test_extract_last_assistant_message_anthropic():
    from promptlayer.evaluations.trace_output import extract_last_assistant_message

    trace = {
        "name": "root",
        "start": "2024-01-01T00:00:00Z",
        "request_log": {
            "request_response": {
                "role": "assistant",
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": "hello from anthropic"}],
            }
        },
        "children": [],
    }
    assert extract_last_assistant_message(trace) == "hello from anthropic"


@patch("promptlayer.evaluations.runner.flush_traces")
@patch("promptlayer.evaluations.tracing.TracerProvider.get_tracer")
@patch("promptlayer.tables.api.upsert_table_by_title")
@patch("promptlayer.tables.api.list_sheets")
@patch("promptlayer.tables.api.create_sheet")
@patch("promptlayer.tables.api.list_smart_sheet_columns")
@patch("promptlayer.tables.api.create_sheet_column")
@patch("promptlayer.tables.api.configure_sheet_scorecard")
@patch("promptlayer.tables.api.add_smart_sheet_rows")
@patch("promptlayer.tables.api.add_trace_import")
@patch("promptlayer.tables.api.list_smart_sheet_rows")
@patch("promptlayer.tables.api.delete_sheet_rows")
@patch("promptlayer.tables.api.update_sheet_cell")
@patch("promptlayer.tables.api.recalculate_smart_sheet_scorecard")
@patch("promptlayer.tables.api.get_sheet_scorecard")
@patch("promptlayer.tables.api.get_sheet_scorecard_row")
def test_eval_with_tracing_derives_output_from_trace(
    mock_get_scorecard_row,
    mock_get_scorecard,
    mock_recalculate_scorecard,
    mock_update_cell,
    mock_delete_rows,
    mock_list_rows,
    mock_add_trace,
    mock_add_rows,
    mock_configure_scorecard,
    mock_create_column,
    mock_list_columns,
    mock_create_sheet,
    mock_list_sheets,
    mock_upsert_table,
    mock_get_tracer,
    mock_flush_traces,
    promptlayer_api_key,
    base_url,
):
    mock_get_tracer.return_value = _FakeTracer()
    mock_upsert_table.return_value = {"id": "1", "title": "Traced Evals"}
    mock_list_sheets.return_value = {"data": []}
    mock_create_sheet.return_value = {"sheet": {"id": "2", "title": "Experiment #1"}}
    mock_list_columns.return_value = {
        "data": [
            {"id": "c1", "title": "Input", "type": "TEXT"},
            {"id": "c2", "title": "Expected", "type": "TEXT"},
            {"id": "c3", "title": "Output", "type": "TEXT"},
            {"id": "c6", "title": "Trace", "type": "TRACE"},
            {
                "id": "c4",
                "title": "pass",
                "type": "CODE_EXECUTION",
            },
        ]
    }
    mock_add_trace.return_value = {"success": True, "rows_added": 1, "mode": "trace"}
    mock_delete_rows.return_value = {"success": True}

    trace_payload = {
        "name": "Eval: Traced Evals",
        "start": "2024-01-01T00:00:00Z",
        "children": [
            {
                "name": "llm",
                "start": "2024-01-01T00:00:01Z",
                "request_log": {
                    "request_response": {
                        "choices": [
                            {
                                "message": {
                                    "role": "assistant",
                                    "content": "from-trace",
                                }
                            }
                        ]
                    }
                },
                "children": [],
            }
        ],
    }
    mock_list_rows.return_value = {
        "data": [
            _completed_row(
                5,
                {
                    "c1": {"id": "cell-1"},
                    "c2": {"id": "cell-2"},
                    "c3": {"id": "cell-3"},
                    "c4": {"id": "cell-4", "status": "COMPLETED"},
                    "c6": {
                        "id": "cell-6",
                        "value": trace_payload,
                        "display_value": "trace",
                    },
                },
            )
        ]
    }
    _stub_scorecard_apis(
        mock_configure_scorecard,
        mock_recalculate_scorecard,
        mock_get_scorecard,
        mock_get_scorecard_row,
        default_raw_value=1,
        aggregate_score=1.0,
    )

    failing = evaluate(
        name="Traced Evals",
        dataset=[{"input": {"q": "hi"}, "expected": {"a": "yo"}}],
        runner=lambda input_data: None,  # customer need not extract last message
        scorers=[code_execution_column("pass", code='return 1 if data.get("Trace") is not None else 0')],
        api_key=promptlayer_api_key,
        base_url=base_url,
    )

    assert failing["failed_row_indices"] == []
    output_updates = [call for call in mock_update_cell.call_args_list if call[0][5] == "cell-3"]
    assert output_updates
    assert output_updates[0][0][6]["value"] == "from-trace"
