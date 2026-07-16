"""Trajectory scorer for agent tool-call sequences."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from promptlayer.evaluations.columns import column
from promptlayer.evaluations.scorers._helpers import (
    require_exactly_one_of,
    require_non_empty_source,
    require_non_empty_title,
    require_non_empty_tools,
    require_one_of_modes,
)
from promptlayer.evaluations.trace_output import (
    collect_tool_spans,
    extract_tool_names,
    parse_json_dict,
)
from promptlayer.types.table import EvalScorerColumn

TrajectoryMode = Literal["strict", "non_strict"]


def extract_trajectory_tool_names(trace: Any) -> list[str]:
    return extract_tool_names(trace)


def _score_tool_sequence(trace: Any, expected_tools: List[str], mode: TrajectoryMode) -> bool:
    actual = extract_trajectory_tool_names(trace)
    if mode == "strict":
        return actual == expected_tools
    return _is_subsequence(expected_tools, actual)


def _collect_tools(trace: Any) -> List[Dict[str, Any]]:
    return [
        {"tool": entry.tool, "output": parse_json_dict(entry.output)}
        for entry in collect_tool_spans(trace)
    ]


def _get_path(data: Any, path: Any) -> Any:
    if data is None or not path:
        return None
    current = data
    for part in str(path).split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _is_subsequence(required: List[Any], actual: List[Any]) -> bool:
    req_idx = 0
    for name in actual:
        if req_idx < len(required) and name == required[req_idx]:
            req_idx += 1
    return req_idx == len(required)


def _values_match(actual: Any, expected_value: Any) -> bool:
    if actual == expected_value:
        return True
    if isinstance(expected_value, bool):
        return actual is expected_value
    if isinstance(expected_value, int) and not isinstance(expected_value, bool) and isinstance(actual, str):
        try:
            return int(actual) == expected_value
        except ValueError:
            return False
    return False


def _tool_succeeded(tool_name: Any, tools: List[Dict[str, Any]]) -> bool:
    matching = [entry for entry in tools if entry["tool"] == tool_name]
    if not matching:
        return False
    output = matching[-1]["output"]
    return isinstance(output, dict) and output.get("success") is True


def _check_single(expected_spec: Any, tools: List[Dict[str, Any]], actual_names: List[str]) -> Optional[str]:
    if not isinstance(expected_spec, dict):
        return "expected spec is not a dict"

    required_tools = expected_spec.get("required_tools") or []
    if required_tools and not _is_subsequence(list(required_tools), actual_names):
        return f"required tool order {required_tools} not satisfied by observed tools {actual_names}"

    for check in expected_spec.get("tool_checks") or []:
        if not isinstance(check, dict):
            return "tool_checks entry is not a dict"
        tool_name = check.get("tool")
        if not tool_name:
            return "tool_checks entry is missing tool"

        matching = [entry for entry in tools if entry["tool"] == tool_name]
        if not matching:
            return f"tool {tool_name!r} was not called"

        output = matching[-1]["output"]
        if not isinstance(output, dict):
            return f"tool {tool_name!r} output is not a dict"

        if "success" in check and output.get("success") is not check["success"]:
            return f"tool {tool_name!r} success={output.get('success')!r}, expected {check['success']!r}"

        for path, expected_value in (check.get("output_fields") or {}).items():
            actual_value = _get_path(output, str(path))
            if not _values_match(actual_value, expected_value):
                return f"tool {tool_name!r} field {path!r}={actual_value!r}, expected {expected_value!r}"

        for item in check.get("list_contains") or []:
            if not isinstance(item, dict):
                return "list_contains entry is not a dict"
            list_value = _get_path(output, str(item.get("path") or ""))
            if not isinstance(list_value, list):
                return f"tool {tool_name!r} path {item.get('path')!r} is not a list"
            field = item.get("field")
            value = item.get("value")
            if not any(isinstance(entry, dict) and entry.get(field) == value for entry in list_value):
                return f"tool {tool_name!r} list {item.get('path')!r} missing {field}={value!r}"

    for group in expected_spec.get("any_tool_success") or []:
        if not isinstance(group, list) or not group:
            return "any_tool_success group is empty"
        if not any(_tool_succeeded(tool_name, tools) for tool_name in group):
            return f"none of these tools succeeded: {group}"

    return None


def _score_trajectory_spec(trace: Any, expected: Any) -> bool:
    parsed_trace = parse_json_dict(trace) if not isinstance(trace, dict) else trace
    parsed_expected = parse_json_dict(expected)
    if not isinstance(parsed_expected, dict) or not isinstance(parsed_trace, dict):
        return False

    tools = _collect_tools(parsed_trace)
    actual_names = [entry["tool"] for entry in tools]
    scenarios = parsed_expected.get("scenarios")
    if scenarios:
        return any(_check_single(scenario, tools, actual_names) is None for scenario in scenarios)
    return _check_single(parsed_expected, tools, actual_names) is None


def score_trajectory(
    trace: Any,
    expected: Union[List[str], Dict[str, Any], str],
    mode: TrajectoryMode = "strict",
) -> bool:
    """Score a trace against either a tool sequence or a rich trajectory specification."""
    if isinstance(expected, list):
        return _score_tool_sequence(trace, expected, mode)
    return _score_trajectory_spec(trace, expected)


def diagnose_trajectory_failure(trace: Any, expected: Any) -> Optional[str]:
    """Return the first trajectory mismatch reason, or ``None`` when it matches."""
    if isinstance(expected, list):
        actual = extract_trajectory_tool_names(trace)
        if actual == list(expected):
            return None
        return f"expected tools {list(expected)} but observed {actual}"

    parsed_trace = parse_json_dict(trace) if not isinstance(trace, dict) else trace
    parsed_expected = parse_json_dict(expected)
    if not isinstance(parsed_trace, dict):
        return "trace is missing or not a dict"
    if not isinstance(parsed_expected, dict):
        return "expected is missing or not a dict"

    tools = _collect_tools(parsed_trace)
    actual_names = [entry["tool"] for entry in tools]
    scenarios = parsed_expected.get("scenarios")
    if scenarios:
        reasons = []
        for index, scenario in enumerate(scenarios):
            reason = _check_single(scenario, tools, actual_names)
            if reason is None:
                return None
            reasons.append(f"scenario {index + 1}: {reason}")
        return "; ".join(reasons)
    return _check_single(parsed_expected, tools, actual_names)


def trajectory_scorer(
    expected_tools: Optional[List[str]] = None,
    *,
    expected_source: Optional[str] = None,
    mode: TrajectoryMode = "strict",
    title: str = "Trajectory",
    trace_source: str = "Trace",
    **settings: Any,
) -> EvalScorerColumn:
    """Build a TRAJECTORY scorer column."""
    require_non_empty_title(title, "trajectory_scorer")
    require_exactly_one_of(
        expected_tools is not None,
        expected_source is not None,
        names=("expected_tools", "expected_source"),
        scorer_name="trajectory_scorer",
    )
    require_one_of_modes(mode, ("strict", "non_strict"), "trajectory_scorer")
    require_non_empty_source(trace_source, "trajectory_scorer", field="trace_source")

    config: Dict[str, Any] = {"trace_source": trace_source, **settings}
    if expected_tools is not None:
        require_non_empty_tools(expected_tools)
        config.update({"expected_tools": list(expected_tools), "mode": mode})
    else:
        require_non_empty_source(expected_source, "trajectory_scorer", field="expected_source")
        config["expected_source"] = expected_source

    return column(
        title,
        "TRAJECTORY",
        config,
    )
