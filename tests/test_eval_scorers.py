import pytest

from promptlayer import (
    PromptLayerValidationError,
    assert_valid_scorer,
    compare_scorer,
    contains_scorer,
    count_scorer,
    diagnose_trajectory_failure,
    llm_assertion_scorer,
    regex_scorer,
    score_trajectory,
    trajectory_scorer,
)
from promptlayer.evaluations.scorers import (
    TrajectoryMode,
    assert_valid_scorer as package_assert_valid_scorer,
    compare_scorer as package_compare_scorer,
    contains_scorer as package_contains_scorer,
    count_scorer as package_count_scorer,
    trajectory_scorer as package_trajectory_scorer,
)
from promptlayer.evaluations.validation import scorers_reference_trace


def _trace_with_tools(*tool_names: str) -> dict:
    children = []
    for index, tool_name in enumerate(tool_names):
        children.append(
            {
                "name": f"Tool: {tool_name}",
                "start": f"2024-01-01T00:00:0{index}Z",
                "span_id": f"span-{index}",
                "children": [],
            }
        )
    return {
        "name": "root",
        "start": "2024-01-01T00:00:00Z",
        "children": children,
    }


def _tool_trace(*entries: tuple[str, dict]) -> dict:
    import json

    return {
        "name": "root",
        "children": [
            {
                "name": f"Tool: {tool_name}",
                "output": json.dumps(payload),
                "children": [],
            }
            for tool_name, payload in entries
        ],
    }


def test_predefined_scorers_are_publicly_exported():
    assert trajectory_scorer is package_trajectory_scorer
    assert compare_scorer is package_compare_scorer
    assert contains_scorer is package_contains_scorer
    assert count_scorer is package_count_scorer
    assert assert_valid_scorer is package_assert_valid_scorer


def test_predefined_scorer_payloads():
    compare = compare_scorer()
    assert compare == {
        "title": "Compare",
        "type": "COMPARE",
        "config": {
            "sources": ["output", "expected"],
            "comparison_type": {"type": "STRING"},
        },
    }

    contains = contains_scorer(value="refund")
    assert contains == {
        "title": "Contains",
        "type": "CONTAINS",
        "config": {"source": "output", "value": "refund"},
    }

    regex = regex_scorer(regex_pattern=r"inv_\d+")
    assert regex == {
        "title": "Regex",
        "type": "REGEX",
        "config": {"source": "output", "regex_pattern": r"inv_\d+"},
    }

    llm = llm_assertion_scorer(prompt="Is the answer helpful?")
    assert llm == {
        "title": "LLM assertion",
        "type": "LLM_ASSERTION",
        "config": {"source": "output", "prompt": "Is the answer helpful?"},
    }

    count = count_scorer(min_count=1, max_count=500)
    assert count == {
        "title": "Count",
        "type": "COUNT",
        "config": {
            "source": "output",
            "type": "chars",
            "min_count": 1,
            "max_count": 500,
        },
    }

    assert_valid = assert_valid_scorer()
    assert assert_valid == {
        "title": "Assert valid",
        "type": "ASSERT_VALID",
        "config": {"source": "output", "type": "object"},
    }

    assert_valid_custom = assert_valid_scorer(type="email", source="contact")
    assert assert_valid_custom == {
        "title": "Assert valid",
        "type": "ASSERT_VALID",
        "config": {"source": "contact", "type": "email"},
    }

    trajectory = trajectory_scorer(["search", "checkout"], mode="strict")
    assert trajectory == {
        "title": "Trajectory",
        "type": "TRAJECTORY",
        "config": {
            "trace_source": "Trace",
            "expected_tools": ["search", "checkout"],
            "mode": "strict",
        },
    }

    advanced_trajectory = trajectory_scorer(expected_source="expected", title="trajectory assertions v3")
    assert advanced_trajectory == {
        "title": "trajectory assertions v3",
        "type": "TRAJECTORY",
        "config": {"trace_source": "Trace", "expected_source": "expected"},
    }

    expected_trace_trajectory = trajectory_scorer(
        title="expected trace trajectory",
        expected_source="expected_trace",
    )
    assert expected_trace_trajectory["config"] == {
        "trace_source": "Trace",
        "expected_source": "expected_trace",
    }


def test_predefined_scorer_validation():
    with pytest.raises(PromptLayerValidationError):
        trajectory_scorer([])
    with pytest.raises(PromptLayerValidationError):
        trajectory_scorer()
    with pytest.raises(PromptLayerValidationError):
        trajectory_scorer(["search"], expected_source="expected")
    with pytest.raises(PromptLayerValidationError):
        trajectory_scorer(["search"], mode="invalid")  # type: ignore[arg-type]
    with pytest.raises(PromptLayerValidationError):
        contains_scorer()
    with pytest.raises(PromptLayerValidationError):
        regex_scorer(regex_pattern="")
    with pytest.raises(PromptLayerValidationError):
        count_scorer()
    with pytest.raises(PromptLayerValidationError):
        count_scorer(min_count=10, max_count=5)
    with pytest.raises(PromptLayerValidationError):
        compare_scorer(sources=["only_one"])
    with pytest.raises(PromptLayerValidationError):
        llm_assertion_scorer()
    with pytest.raises(PromptLayerValidationError):
        trajectory_scorer(expected_source="expected", title="")


@pytest.mark.parametrize(
    ("mode", "trace_tools", "expected_tools", "expected_score"),
    [
        ("strict", ("search", "checkout"), ["search", "checkout"], True),
        ("strict", ("search",), ["search", "checkout"], False),
        ("strict", ("checkout", "search"), ["search", "checkout"], False),
        ("strict", ("search", "checkout", "email"), ["search", "checkout"], False),
        ("non_strict", ("search", "checkout"), ["search", "checkout"], True),
        ("non_strict", ("search", "lookup", "checkout"), ["search", "checkout"], True),
        ("non_strict", ("search",), ["search", "checkout"], False),
        ("non_strict", ("checkout", "search"), ["search", "checkout"], False),
        ("non_strict", ("search", "search", "checkout"), ["search", "checkout"], True),
    ],
)
def test_trajectory_scorer_modes(
    mode: TrajectoryMode,
    trace_tools: tuple[str, ...],
    expected_tools: list[str],
    expected_score: bool,
):
    assert score_trajectory(_trace_with_tools(*trace_tools), expected_tools, mode) is expected_score


def test_trajectory_tool_prefix_is_normalized():
    trace = {
        "name": "root",
        "children": [
            {"name": "Tool:search", "children": []},
            {"name": "Tool: checkout", "children": []},
        ],
    }

    assert score_trajectory(trace, ["search", "checkout"]) is True
    assert score_trajectory(trace, {"required_tools": ["search", "checkout"]}) is True


def test_count_scorer_settings():
    words = count_scorer(type="words", min_count=2, max_count=4)
    assert words["config"] == {
        "source": "output",
        "type": "words",
        "min_count": 2,
        "max_count": 4,
    }


def test_assert_valid_scorer_validation():
    with pytest.raises(PromptLayerValidationError):
        assert_valid_scorer(type="")
    with pytest.raises(PromptLayerValidationError):
        assert_valid_scorer(source="")


def test_trajectory_scorer_references_trace_for_dependencies():
    scorer = trajectory_scorer(["search"])
    assert scorers_reference_trace([scorer])


def test_trajectory_spec_scorer_passes_and_fails():
    expected = {
        "required_tools": ["create_folder"],
        "tool_checks": [
            {
                "tool": "create_folder",
                "success": True,
                "output_fields": {"folder.name": "my-folder"},
            }
        ],
    }
    matching = _tool_trace(
        ("create_folder", {"success": True, "folder": {"name": "my-folder"}}),
    )
    mismatch = _tool_trace(
        ("create_folder", {"success": True, "folder": {"name": "wrong"}}),
    )
    assert score_trajectory(matching, expected) is True
    assert score_trajectory(mismatch, expected) is False

    scorer = trajectory_scorer(expected_source="expected")
    assert scorer["type"] == "TRAJECTORY"
    assert scorer["config"]["trace_source"] == "Trace"
    assert scorer["config"]["expected_source"] == "expected"


def test_trajectory_spec_scorer_accepts_scenarios_and_any_tool_success():
    expected = {
        "scenarios": [
            {
                "required_tools": ["create_dataset"],
                "tool_checks": [
                    {
                        "tool": "create_dataset",
                        "success": True,
                        "output_fields": {"rows_created_count": 2},
                    }
                ],
            },
            {
                "required_tools": ["create_dataset", "update_dataset"],
                "tool_checks": [
                    {"tool": "create_dataset", "success": True},
                    {"tool": "update_dataset", "success": True},
                ],
            },
        ]
    }
    trace = _tool_trace(
        ("create_dataset", {"success": True}),
        ("update_dataset", {"success": True}),
    )
    assert score_trajectory(trace, expected) is True

    any_success = {
        "required_tools": ["create_table"],
        "any_tool_success": [["add_rows", "update_cell"]],
        "tool_checks": [{"tool": "create_table", "success": True}],
    }
    assert (
        score_trajectory(
            _tool_trace(
                ("create_table", {"success": True}),
                ("update_cell", {"success": True}),
            ),
            any_success,
        )
        is True
    )


def test_diagnose_trajectory_failure_categories():
    assert diagnose_trajectory_failure(None, {"required_tools": ["x"]}) == "trace is missing or not a dict"
    assert diagnose_trajectory_failure({"name": "root", "children": []}, None) == "expected is missing or not a dict"

    order_reason = diagnose_trajectory_failure(
        _tool_trace(("b", {"success": True}), ("a", {"success": True})),
        {"required_tools": ["a", "b"]},
    )
    assert order_reason is not None
    assert "required tool order" in order_reason

    missing_tool = diagnose_trajectory_failure(
        {"name": "root", "children": []},
        {"required_tools": [], "tool_checks": [{"tool": "create_folder", "success": True}]},
    )
    assert missing_tool is not None
    assert "was not called" in missing_tool

    field_reason = diagnose_trajectory_failure(
        _tool_trace(("create_folder", {"success": True, "folder": {"name": "wrong"}})),
        {
            "required_tools": ["create_folder"],
            "tool_checks": [
                {
                    "tool": "create_folder",
                    "success": True,
                    "output_fields": {"folder.name": "expected"},
                }
            ],
        },
    )
    assert field_reason is not None
    assert "folder.name" in field_reason

    list_reason = diagnose_trajectory_failure(
        _tool_trace(("search", {"success": True, "results": [{"name": "other"}]})),
        {
            "tool_checks": [
                {
                    "tool": "search",
                    "success": True,
                    "list_contains": [{"path": "results", "field": "name", "value": "wanted"}],
                }
            ]
        },
    )
    assert list_reason is not None
    assert "missing" in list_reason

    scenario_reason = diagnose_trajectory_failure(
        {"name": "root", "children": []},
        {
            "scenarios": [
                {"required_tools": ["create_dataset"]},
                {"required_tools": ["create_dataset", "update_dataset"]},
            ]
        },
    )
    assert scenario_reason is not None
    assert "scenario 1:" in scenario_reason
    assert "scenario 2:" in scenario_reason

    matching = _tool_trace(("create_folder", {"success": True, "folder": {"name": "ok"}}))
    assert (
        diagnose_trajectory_failure(
            matching,
            {
                "required_tools": ["create_folder"],
                "tool_checks": [
                    {
                        "tool": "create_folder",
                        "success": True,
                        "output_fields": {"folder.name": "ok"},
                    }
                ],
            },
        )
        is None
    )


def test_trajectory_spec_scorer_references_trace():
    assert scorers_reference_trace([trajectory_scorer(expected_source="expected")])


def test_trajectory_tool_order_uses_chronological_not_tree_order():
    # Child span starts earlier than parent → chrono order differs from DFS tree order.
    trace = {
        "name": "root",
        "start": "2024-01-01T00:00:00Z",
        "children": [
            {
                "name": "Tool: checkout",
                "start": "2024-01-02T00:00:00Z",
                "span_id": "parent",
                "output": '{"success": true}',
                "children": [
                    {
                        "name": "Tool: search",
                        "start": "2024-01-01T12:00:00Z",
                        "span_id": "child",
                        "output": '{"success": true}',
                        "children": [],
                    }
                ],
            }
        ],
    }
    assert score_trajectory(trace, ["search", "checkout"]) is True
    assert score_trajectory(trace, {"required_tools": ["search", "checkout"]}) is True
    assert score_trajectory(trace, ["checkout", "search"]) is False
    assert score_trajectory(trace, {"required_tools": ["checkout", "search"]}) is False


def test_diagnose_trajectory_failure_list_expected():
    reason = diagnose_trajectory_failure(_trace_with_tools("search"), ["search", "checkout"])
    assert reason is not None
    assert "checkout" in reason


def test_trajectory_scorer_empty_trace_source_mentions_field():
    with pytest.raises(PromptLayerValidationError, match="trace_source"):
        trajectory_scorer(["search"], trace_source="")


def test_all_predefined_scorers_are_publicly_exported():
    from promptlayer import (
        diagnose_trajectory_failure as top_diagnose,
        llm_assertion_scorer as top_llm,
        regex_scorer as top_regex,
        score_trajectory as top_score,
    )
    from promptlayer.evaluations.scorers import (
        diagnose_trajectory_failure as package_diagnose,
        llm_assertion_scorer as package_llm,
        regex_scorer as package_regex,
        score_trajectory as package_score,
    )

    assert top_regex is package_regex
    assert top_llm is package_llm
    assert top_score is package_score
    assert top_diagnose is package_diagnose


def test_score_value_helpers_match_js_semantics():
    from promptlayer.evaluations.scores import (
        llm_assertion_verdict,
        scorer_value_failed,
        unwrap_nested_value,
    )

    assert unwrap_nested_value({"value": {"value": False}}) is False
    assert scorer_value_failed(False) is True
    assert scorer_value_failed(0) is True
    assert scorer_value_failed({"status": "FAILED"}) is True
    assert scorer_value_failed({"comparison_result": False}) is True
    assert scorer_value_failed({"ok": {"value": True}, "bad": {"value": False}}) is True
    assert llm_assertion_verdict({"ok": {"value": True}, "bad": {"value": False}}) is False
    assert llm_assertion_verdict({"ok": {"value": True}, "also": {"value": True}}) is True
