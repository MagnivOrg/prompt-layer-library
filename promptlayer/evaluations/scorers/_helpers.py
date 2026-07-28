"""Shared helpers for predefined eval scorers."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from promptlayer.evaluations.validation import validation_error
from promptlayer.types.table import EvalScorerColumn

_SCORECARD_STEP_OPTION_KEYS = frozenset({"weight", "failure_threshold", "pass_threshold", "required", "thresholds"})
_DEFAULT_PASS_THRESHOLD = 0.8
_DEFAULT_WARN_THRESHOLD = 0.6


def require_non_empty_title(title: str, scorer_name: str) -> None:
    if not isinstance(title, str) or not title.strip():
        raise validation_error(f"{scorer_name} title must be a non-empty string.")


def require_non_empty_string(value: Any, *, field: str, scorer_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise validation_error(f"{scorer_name} {field} must be a non-empty string.")
    return value


def require_non_empty_source(source: str, scorer_name: str, *, field: str = "source") -> None:
    if not isinstance(source, str) or not source.strip():
        raise validation_error(f"{scorer_name} {field} must be a non-empty string.")


def require_exactly_one_of(
    *present: bool,
    names: Sequence[str],
    scorer_name: str,
) -> None:
    if len(present) != len(names):
        raise ValueError("require_exactly_one_of present flags must match names.")
    if sum(1 for flag in present if flag) != 1:
        rendered = " or ".join(names)
        raise validation_error(f"{scorer_name} requires exactly one of {rendered}.")


def require_one_of_modes(value: str, allowed: Iterable[str], scorer_name: str, *, field: str = "mode") -> None:
    allowed_list = list(allowed)
    if value not in allowed_list:
        rendered = " or ".join(repr(item) for item in allowed_list)
        raise validation_error(f"{scorer_name} {field} must be {rendered}.")


def require_non_empty_tools(expected_tools: List[str]) -> None:
    if not isinstance(expected_tools, list) or not expected_tools:
        raise validation_error("trajectory_scorer requires a non-empty expected_tools list.")
    for tool in expected_tools:
        if not isinstance(tool, str) or not tool.strip():
            raise validation_error("trajectory_scorer expected_tools must be non-empty strings.")


def require_count_bounds(
    *,
    min_count: Optional[int],
    max_count: Optional[int],
) -> None:
    if min_count is None and max_count is None:
        raise validation_error("count_scorer requires min_count and/or max_count.")
    for name, value in (("min_count", min_count), ("max_count", max_count)):
        if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 0):
            raise validation_error(f"count_scorer {name} must be a non-negative integer.")
    if min_count is not None and max_count is not None and min_count > max_count:
        raise validation_error("count_scorer min_count cannot exceed max_count.")


def _normalize_threshold(value: Any, *, field: str) -> float:
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise validation_error(f"{field} must be a number between 0 and 1.") from exc
    if normalized < 0 or normalized > 1:
        raise validation_error(f"{field} must be a number between 0 and 1.")
    return normalized


def pop_scorecard_step_options(settings: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Split scorecard step options from primitive config settings."""
    step_options = {key: settings[key] for key in list(settings) if key in _SCORECARD_STEP_OPTION_KEYS}
    config_settings = {key: value for key, value in settings.items() if key not in _SCORECARD_STEP_OPTION_KEYS}
    return step_options, config_settings


def apply_scorecard_step_options(
    payload: EvalScorerColumn,
    *,
    weight: Any = None,
    failure_threshold: Any = None,
    pass_threshold: Any = None,
    required: Any = None,
    thresholds: Any = None,
    **_ignored: Any,
) -> EvalScorerColumn:
    """Attach scorecard weight / threshold metadata onto a scorer payload."""
    if weight is not None:
        try:
            normalized_weight = float(weight)
        except (TypeError, ValueError) as exc:
            raise validation_error("weight must be a positive number.") from exc
        if normalized_weight <= 0:
            raise validation_error("weight must be a positive number.")
        payload["weight"] = normalized_weight

    if required is not None:
        if not isinstance(required, bool):
            raise validation_error("required must be a boolean.")
        payload["required"] = required

    resolved_thresholds: Optional[Dict[str, float]] = None
    if thresholds is not None:
        if not isinstance(thresholds, dict):
            raise validation_error("thresholds must be an object with pass/warn values.")
        resolved_thresholds = {
            "pass": _normalize_threshold(
                thresholds.get("pass", _DEFAULT_PASS_THRESHOLD),
                field="thresholds.pass",
            ),
            "warn": _normalize_threshold(
                thresholds.get("warn", _DEFAULT_WARN_THRESHOLD),
                field="thresholds.warn",
            ),
        }
    elif failure_threshold is not None or pass_threshold is not None:
        resolved_thresholds = {
            "pass": _normalize_threshold(
                _DEFAULT_PASS_THRESHOLD if pass_threshold is None else pass_threshold,
                field="pass_threshold",
            ),
            "warn": _normalize_threshold(
                _DEFAULT_WARN_THRESHOLD if failure_threshold is None else failure_threshold,
                field="failure_threshold",
            ),
        }

    if resolved_thresholds is not None:
        if resolved_thresholds["warn"] > resolved_thresholds["pass"]:
            raise validation_error("failure_threshold cannot be higher than pass_threshold.")
        payload["thresholds"] = resolved_thresholds

    return payload
