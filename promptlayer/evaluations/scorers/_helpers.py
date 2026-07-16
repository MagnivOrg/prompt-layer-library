"""Shared helpers for predefined eval scorers."""

from __future__ import annotations

from typing import Any, Iterable, List, Optional, Sequence

from promptlayer.evaluations.validation import validation_error


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
    first_present: bool,
    second_present: bool,
    *,
    names: Sequence[str],
    scorer_name: str,
) -> None:
    if first_present == second_present:
        left, right = names[0], names[1]
        raise validation_error(f"{scorer_name} requires exactly one of {left} or {right}.")


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
