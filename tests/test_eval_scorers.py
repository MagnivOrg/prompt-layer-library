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
            "sources": ["Output", "expected"],
            "comparison_type": {"type": "STRING"},
        },
    }
    custom_compare = compare_scorer(source_column="actual", expected_column="reference")
    assert custom_compare["config"]["sources"] == ["actual", "reference"]
    literal_compare = compare_scorer(expected=None, comparison_type="JSON")
    assert literal_compare["config"] == {
        "sources": ["Output"],
        "target": None,
        "comparison_type": {"type": "JSON"},
    }

    contains = contains_scorer(expected="refund")
    assert contains == {
        "title": "Contains",
        "type": "CONTAINS",
        "config": {"source": "Output", "value": "refund"},
    }
    contains_column = contains_scorer(expected_column="reference")
    assert contains_column["config"] == {
        "source": "Output",
        "value_source": "reference",
    }

    regex = regex_scorer(regex_pattern=r"inv_\d+")
    assert regex == {
        "title": "Regex",
        "type": "REGEX",
        "config": {"source": "Output", "regex_pattern": r"inv_\d+"},
    }

    llm = llm_assertion_scorer(prompt="Is the answer helpful?")
    assert llm == {
        "title": "LLM assertion",
        "type": "LLM_ASSERTION",
        "config": {"source": "Output", "prompt": "Is the answer helpful?"},
    }

    count = count_scorer(min_count=1, max_count=500)
    assert count == {
        "title": "Count",
        "type": "COUNT",
        "config": {
            "source": "Output",
            "type": "chars",
            "min_count": 1,
            "max_count": 500,
        },
    }

    assert_valid = assert_valid_scorer()
    assert assert_valid == {
        "title": "Assert valid",
        "type": "ASSERT_VALID",
        "config": {"source": "Output", "type": "object"},
    }

    assert_valid_custom = assert_valid_scorer(type="email", source_column="contact")
    assert assert_valid_custom == {
        "title": "Assert valid",
        "type": "ASSERT_VALID",
        "config": {"source": "contact", "type": "email"},
    }

    trajectory = trajectory_scorer(
        expected=[["search", "checkout"]],
        mode="strict",
    )
    assert trajectory == {
        "title": "Trajectory",
        "type": "TRAJECTORY",
        "config": {
            "trace_source": "Trace",
            "accepted_scenarios": [["search", "checkout"]],
            "mode": "strict",
        },
    }

    advanced_trajectory = trajectory_scorer(expected_column="expected", title="trajectory assertions v3")
    assert advanced_trajectory == {
        "title": "trajectory assertions v3",
        "type": "TRAJECTORY",
        "config": {"trace_source": "Trace", "expected_source": "expected", "mode": "strict"},
    }

    expected_trace_trajectory = trajectory_scorer(
        title="expected trace trajectory",
        expected_column="expected_trace",
        mode="non_strict",
    )
    assert expected_trace_trajectory["config"] == {
        "trace_source": "Trace",
        "expected_source": "expected_trace",
        "mode": "non_strict",
    }
    custom_source_trajectory = trajectory_scorer(
        source_column="Agent trace",
        expected=[["search"]],
    )
    assert custom_source_trajectory["config"]["trace_source"] == "Agent trace"
    custom_value_source_trajectory = trajectory_scorer(expected_column="expected trajectory")
    assert custom_value_source_trajectory["config"]["expected_source"] == "expected trajectory"


def test_predefined_scorer_validation():
    with pytest.raises(PromptLayerValidationError):
        trajectory_scorer(expected=[])
    with pytest.raises(PromptLayerValidationError):
        trajectory_scorer()
    with pytest.raises(PromptLayerValidationError):
        trajectory_scorer(
            expected=[["search"]],
            expected_column="expected",
        )
    with pytest.raises(PromptLayerValidationError):
        trajectory_scorer(
            expected=[["search"]],
            mode="invalid",  # type: ignore[arg-type]
        )
    with pytest.raises(PromptLayerValidationError):
        contains_scorer()
    with pytest.raises(PromptLayerValidationError, match="exactly one of expected or expected_column"):
        contains_scorer(expected="refund", expected_column="reference")
    with pytest.raises(PromptLayerValidationError):
        regex_scorer(regex_pattern="")
    with pytest.raises(PromptLayerValidationError):
        count_scorer()
    with pytest.raises(PromptLayerValidationError):
        count_scorer(min_count=10, max_count=5)
    with pytest.raises(PromptLayerValidationError):
        compare_scorer(expected_column="")
    with pytest.raises(PromptLayerValidationError, match="only one of expected or expected_column"):
        compare_scorer(expected="refund", expected_column="reference")
    with pytest.raises(PromptLayerValidationError):
        llm_assertion_scorer()
    with pytest.raises(PromptLayerValidationError):
        trajectory_scorer(expected_column="expected", title="")
    with pytest.raises(PromptLayerValidationError, match="legacy parameter"):
        contains_scorer(value="refund")
    with pytest.raises(PromptLayerValidationError, match="legacy parameter"):
        compare_scorer(source="Output")
    with pytest.raises(PromptLayerValidationError, match="legacy parameter"):
        trajectory_scorer(accepted_scenarios=[["search"]])


def test_compare_scorer_dependencies_support_literal_and_column_targets():
    from promptlayer.evaluations.validation import scorer_dependencies_from_config

    columns_by_title = {
        "Output": {"id": "out-1", "title": "Output"},
        "reference": {"id": "ref-1", "title": "reference"},
    }

    literal = compare_scorer(expected=None)
    assert scorer_dependencies_from_config(literal["config"], columns_by_title) == [
        {
            "column_id": "out-1",
            "reference_type": "value",
            "config_key": "sources",
            "config_meta": {"position": 0},
        }
    ]

    column_target = compare_scorer(expected_column="reference")
    assert scorer_dependencies_from_config(column_target["config"], columns_by_title) == [
        {
            "column_id": "out-1",
            "reference_type": "value",
            "config_key": "sources",
            "config_meta": {"position": 0},
        },
        {
            "column_id": "ref-1",
            "reference_type": "value",
            "config_key": "sources",
            "config_meta": {"position": 1},
        },
    ]


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
    expected = {"accepted_scenarios": [{"required_tools": expected_tools}]}
    assert score_trajectory(_trace_with_tools(*trace_tools), expected, mode) is expected_score


def test_trajectory_tool_prefix_is_normalized():
    trace = {
        "name": "root",
        "children": [
            {"name": "Tool:search", "children": []},
            {"name": "Tool: checkout", "children": []},
        ],
    }

    assert (
        score_trajectory(
            trace,
            {"accepted_scenarios": [{"required_tools": ["search", "checkout"]}]},
        )
        is True
    )


def test_count_scorer_settings():
    words = count_scorer(type="words", min_count=2, max_count=4)
    assert words["config"] == {
        "source": "Output",
        "type": "words",
        "min_count": 2,
        "max_count": 4,
    }


def test_assert_valid_scorer_validation():
    with pytest.raises(PromptLayerValidationError):
        assert_valid_scorer(type="")
    with pytest.raises(PromptLayerValidationError):
        assert_valid_scorer(source_column="")


def test_trajectory_source_parses_accepted_scenarios():
    matching = _tool_trace(("create_folder", {"success": True, "folder": {"name": "my-folder"}}))
    assert (
        score_trajectory(
            matching,
            {"accepted_scenarios": [{"required_tools": ["create_folder"]}]},
        )
        is True
    )
    # Objects / legacy shapes are rejected
    assert score_trajectory(matching, {"required_tools": ["create_folder"]}) is False
    assert score_trajectory(matching, {"accepted_scenarios": [["create_folder"]]}) is False
    assert score_trajectory(matching, ["create_folder"]) is False

    scorer = trajectory_scorer(expected_column="expected")
    assert scorer["type"] == "TRAJECTORY"
    assert scorer["config"]["trace_source"] == "Trace"
    assert scorer["config"]["expected_source"] == "expected"
    assert scorer["config"]["mode"] == "strict"


def test_trajectory_source_matches_any_scenario():
    expected = {
        "accepted_scenarios": [
            {"required_tools": ["get_model_config", "create_prompt"]},
            {"required_tools": ["list_model_configs", "create_prompt"]},
            {"required_tools": ["select_model_config", "create_prompt"]},
        ]
    }
    assert score_trajectory(_tool_trace(("list_model_configs", None), ("create_prompt", None)), expected) is True
    assert score_trajectory(_tool_trace(("create_prompt", None)), expected) is False
    reason = diagnose_trajectory_failure(_tool_trace(("create_prompt", None)), expected)
    assert reason is not None
    assert "scenario 1:" in reason
    assert "scenario 2:" in reason


def test_trajectory_scorer_accepts_config_accepted_scenarios():
    scorer = trajectory_scorer(
        expected=[
            ["get_model_config", "create_prompt"],
            ["list_model_configs", "create_prompt"],
        ],
        title="multi path",
    )
    assert scorer["config"] == {
        "trace_source": "Trace",
        "accepted_scenarios": [
            ["get_model_config", "create_prompt"],
            ["list_model_configs", "create_prompt"],
        ],
        "mode": "strict",
    }
    single = trajectory_scorer(expected=[["search"]], title="one path")
    assert single["config"]["accepted_scenarios"] == [["search"]]


def test_trajectory_scorer_supports_weight_and_failure_threshold():
    scorer = trajectory_scorer(
        expected=[["search"]],
        weight=2.5,
        failure_threshold=0.5,
        pass_threshold=0.9,
        required=True,
    )
    assert scorer["weight"] == 2.5
    assert scorer["required"] is True
    assert scorer["thresholds"] == {"pass": 0.9, "warn": 0.5}
    assert "weight" not in scorer["config"]
    assert "thresholds" not in scorer["config"]

    from promptlayer.evaluations.scorecard import build_scorecard_steps_from_scorers
    from promptlayer.evaluations.validation import normalize_scorer

    normalized = normalize_scorer(scorer)
    assert normalized["weight"] == 2.5
    assert normalized["required"] is True
    assert normalized["thresholds"] == {"pass": 0.9, "warn": 0.5}

    steps = build_scorecard_steps_from_scorers(
        [normalized],
        [{"id": "tr-1", "title": "Trace"}],
    )
    assert steps[0]["weight"] == 2.5
    assert steps[0]["required"] is True
    assert steps[0]["thresholds"] == {"pass": 0.9, "warn": 0.5}


def test_diagnose_trajectory_failure_categories():
    assert (
        diagnose_trajectory_failure(
            None,
            {"accepted_scenarios": [{"required_tools": ["x"]}]},
        )
        == "trace is missing or not a dict"
    )
    assert diagnose_trajectory_failure({"name": "root", "children": []}, None) == "expected is missing or not a dict"
    assert (
        diagnose_trajectory_failure(
            {"name": "root", "children": []},
            {"accepted_scenarios": []},
        )
        == "expected tools could not be parsed from source"
    )

    order_reason = diagnose_trajectory_failure(
        _tool_trace(("b", {"success": True}), ("a", {"success": True})),
        {"accepted_scenarios": [{"required_tools": ["a", "b"]}]},
        mode="non_strict",
    )
    assert order_reason is not None
    assert "required tool order" in order_reason

    missing_tool = diagnose_trajectory_failure(
        {"name": "root", "children": []},
        {"accepted_scenarios": [{"required_tools": ["create_folder"]}]},
    )
    assert missing_tool is not None
    assert "create_folder" in missing_tool

    matching = _tool_trace(("create_folder", {"success": True}))
    assert (
        diagnose_trajectory_failure(
            matching,
            {"accepted_scenarios": [{"required_tools": ["create_folder"]}]},
        )
        is None
    )


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
    assert (
        score_trajectory(
            trace,
            {"accepted_scenarios": [{"required_tools": ["search", "checkout"]}]},
        )
        is True
    )
    assert (
        score_trajectory(
            trace,
            {"accepted_scenarios": [{"required_tools": ["checkout", "search"]}]},
        )
        is False
    )


def test_diagnose_trajectory_failure_list_expected():
    reason = diagnose_trajectory_failure(
        _trace_with_tools("search"),
        {"accepted_scenarios": [{"required_tools": ["search", "checkout"]}]},
    )
    assert reason is not None
    assert "checkout" in reason


def test_trajectory_scorer_empty_source_mentions_field():
    with pytest.raises(PromptLayerValidationError, match="source"):
        trajectory_scorer(expected=[["search"]], source_column="")


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
