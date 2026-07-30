"""Trajectory scorer for agent tool-call sequences."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Literal, Optional, Union

from promptlayer.evaluations.columns import column
from promptlayer.evaluations.scorers._helpers import (
    apply_scorecard_step_options,
    reject_legacy_parameters,
    require_exactly_one_of,
    require_non_empty_source,
    require_non_empty_title,
    require_non_empty_tools,
    require_one_of_modes,
)
from promptlayer.evaluations.trace_output import extract_tool_names
from promptlayer.evaluations.validation import validation_error
from promptlayer.types.table import EvalScorerColumn

TrajectoryMode = Literal["strict", "non_strict"]


def extract_trajectory_tool_names(trace: Any) -> list[str]:
    return extract_tool_names(trace)


def _is_subsequence(required: List[Any], actual: List[Any]) -> bool:
    req_idx = 0
    for name in actual:
        if req_idx < len(required) and name == required[req_idx]:
            req_idx += 1
    return req_idx == len(required)


def _score_tool_sequence(trace: Any, expected_tools: List[str], mode: TrajectoryMode) -> bool:
    actual = extract_trajectory_tool_names(trace)
    if mode == "strict":
        return actual == expected_tools
    return _is_subsequence(expected_tools, actual)


def _parse_json_value(raw: Any) -> Any:
    if isinstance(raw, (dict, list)):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
    return None


def _parse_tool_list(entries: Any) -> Optional[List[str]]:
    if not isinstance(entries, list) or not entries:
        return None
    tools: List[str] = []
    for entry in entries:
        if not isinstance(entry, str):
            return None
        tool_name = entry.strip()
        if not tool_name:
            return None
        tools.append(tool_name)
    return tools


def parse_expected_tool_lists_from_source(raw: Any) -> Optional[List[List[str]]]:
    """Parse expected tool-name sequences from a column cell.

    Accepts only:
    {
      "accepted_scenarios": [
        {"required_tools": ["search", "checkout"]},
        ...
      ]
    }
    """
    parsed = _parse_json_value(raw)
    if not isinstance(parsed, dict):
        return None

    scenarios = parsed.get("accepted_scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        return None

    tool_lists: List[List[str]] = []
    for scenario in scenarios:
        if not isinstance(scenario, dict):
            return None
        tool_list = _parse_tool_list(scenario.get("required_tools"))
        if tool_list is None:
            return None
        tool_lists.append(tool_list)
    return tool_lists


def parse_expected_tools_from_source(raw: Any) -> Optional[List[str]]:
    tool_lists = parse_expected_tool_lists_from_source(raw)
    if not tool_lists or len(tool_lists) != 1:
        return None
    return tool_lists[0]


def score_trajectory(
    trace: Any,
    expected: Union[Dict[str, Any], str],
    mode: TrajectoryMode = "strict",
) -> bool:
    """Score a trace against accepted scenarios from a column cell value."""
    expected_lists = parse_expected_tool_lists_from_source(expected)
    if not expected_lists:
        return False
    return any(_score_tool_sequence(trace, expected_tools, mode) for expected_tools in expected_lists)


def diagnose_trajectory_failure(
    trace: Any,
    expected: Any,
    mode: TrajectoryMode = "strict",
) -> Optional[str]:
    """Return the first trajectory mismatch reason, or ``None`` when it matches."""
    if expected is None:
        return "expected is missing or not a dict"

    expected_lists = parse_expected_tool_lists_from_source(expected)
    if not expected_lists:
        return "expected tools could not be parsed from source"

    if trace is None:
        return "trace is missing or not a dict"

    actual = extract_trajectory_tool_names(trace)
    if any(_score_tool_sequence(trace, expected_tools, mode) for expected_tools in expected_lists):
        return None

    if len(expected_lists) == 1:
        expected_tools = expected_lists[0]
        if mode == "strict":
            return f"expected tools {expected_tools} but observed {actual}"
        return f"required tool order {expected_tools} not satisfied by observed tools {actual}"

    reasons = []
    for index, expected_tools in enumerate(expected_lists):
        if mode == "strict":
            reasons.append(f"scenario {index + 1}: expected tools {expected_tools} but observed {actual}")
        else:
            reasons.append(
                f"scenario {index + 1}: required tool order {expected_tools} not satisfied by observed tools {actual}"
            )
    return "; ".join(reasons)


def trajectory_scorer(
    *,
    expected: Optional[List[List[str]]] = None,
    expected_column: Optional[str] = None,
    source_column: str = "Trace",
    mode: TrajectoryMode = "strict",
    title: str = "Trajectory",
    weight: Optional[float] = None,
    failure_threshold: Optional[float] = None,
    pass_threshold: Optional[float] = None,
    required: Optional[bool] = None,
    thresholds: Optional[Dict[str, float]] = None,
    **settings: Any,
) -> EvalScorerColumn:
    """Build a TRAJECTORY scorer column."""
    reject_legacy_parameters(
        settings,
        names=("accepted_scenarios", "value_source", "source"),
        scorer_name="trajectory_scorer",
    )
    require_non_empty_title(title, "trajectory_scorer")
    require_exactly_one_of(
        expected is not None,
        expected_column is not None,
        names=("expected", "expected_column"),
        scorer_name="trajectory_scorer",
    )
    require_one_of_modes(mode, ("strict", "non_strict"), "trajectory_scorer")
    require_non_empty_source(source_column, "trajectory_scorer", field="source_column")

    config: Dict[str, Any] = {"trace_source": source_column, **settings, "mode": mode}
    if expected is not None:
        if not isinstance(expected, list) or not expected:
            raise validation_error("trajectory_scorer expected must be a non-empty list.")
        normalized: List[List[str]] = []
        for scenario in expected:
            require_non_empty_tools(scenario)
            normalized.append(list(scenario))
        config["accepted_scenarios"] = normalized
    else:
        require_non_empty_source(expected_column, "trajectory_scorer", field="expected_column")
        config["expected_source"] = expected_column

    payload = column(
        title,
        "TRAJECTORY",
        config,
    )
    return apply_scorecard_step_options(
        payload,
        weight=weight,
        failure_threshold=failure_threshold,
        pass_threshold=pass_threshold,
        required=required,
        thresholds=thresholds,
    )
