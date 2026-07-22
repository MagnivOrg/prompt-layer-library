"""Shared score value interpretation for evals (parity with JS evaluations/scores.ts)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from promptlayer.types.table import EvalCaseResult

_LLM_ASSERTION_DETAIL_KEYS = frozenset({"value", "reasoning", "explanation", "citation"})


def unwrap_nested_value(value: Any) -> Any:
    current = value
    while isinstance(current, dict) and set(current.keys()) == {"value"}:
        nested = current.get("value")
        if nested is current:
            break
        current = nested
    return current


def _assertion_explanation(detail: Dict[str, Any]) -> Optional[str]:
    for key in ("reasoning", "explanation"):
        text = detail.get(key)
        if isinstance(text, str) and text.strip():
            return text.strip()
    return None


def iter_llm_assertion_details(value: Any) -> List[Tuple[str, Any, Optional[str]]]:
    """Yield ``(label, passed, explanation)`` entries from an LLM Assertion cell value."""
    unwrapped = unwrap_nested_value(value)
    if isinstance(unwrapped, bool):
        return [("Assertion", unwrapped, None)]
    if not isinstance(unwrapped, dict):
        return []

    if "value" in unwrapped and set(unwrapped.keys()) <= _LLM_ASSERTION_DETAIL_KEYS:
        return [("Assertion", unwrapped.get("value"), _assertion_explanation(unwrapped))]

    details: List[Tuple[str, Any, Optional[str]]] = []
    for key, entry in unwrapped.items():
        if isinstance(entry, bool):
            details.append((str(key), entry, None))
        elif isinstance(entry, dict) and "value" in entry:
            details.append((str(key), entry.get("value"), _assertion_explanation(entry)))
    return details


def llm_assertion_verdict(value: Any) -> Optional[bool]:
    """Aggregate nested LLM assertion payload into a single boolean verdict."""
    unwrapped = unwrap_nested_value(value)
    if isinstance(unwrapped, bool):
        return unwrapped
    if not isinstance(unwrapped, dict) or unwrapped.get("status") == "FAILED":
        return None

    if "value" in unwrapped and set(unwrapped.keys()) <= _LLM_ASSERTION_DETAIL_KEYS:
        verdict = unwrapped.get("value")
        return verdict if isinstance(verdict, bool) else None

    verdicts: List[bool] = []
    for detail in unwrapped.values():
        if isinstance(detail, bool):
            verdicts.append(detail)
        elif isinstance(detail, dict) and isinstance(detail.get("value"), bool):
            verdicts.append(detail["value"])
        else:
            return None
    return all(verdicts) if verdicts else None


def scorer_value_failed(value: Any) -> bool:
    if value is False or value == 0 or value == 0.0:
        return True
    if isinstance(value, dict) and value.get("status") == "FAILED":
        return True
    if isinstance(value, dict) and isinstance(value.get("comparison_result"), bool):
        return value["comparison_result"] is False

    assertions = iter_llm_assertion_details(value)
    if assertions:
        return any(passed is False for _label, passed, _explanation in assertions)
    return False


def case_has_failed_scorer(case: EvalCaseResult) -> bool:
    scores = case.get("scores") or {}
    return any(scorer_value_failed(value) for value in scores.values())


def collect_failing_row_indices(case_results: List[EvalCaseResult]) -> List[int]:
    """Return Table ``row_index`` values for rows with any failed scorer."""
    indices: List[int] = []
    for case in case_results:
        if not case_has_failed_scorer(case):
            continue
        row_index = case.get("row_index")
        if row_index is None:
            continue
        indices.append(int(row_index))
    return indices


def collect_failed_cell_row_indices(case_results: List[EvalCaseResult]) -> List[int]:
    """Return row indices containing scorecard evaluators whose execution failed."""
    indices: List[int] = []
    for case in case_results:
        scores = case.get("scores") or {}
        if not any(isinstance(value, dict) and value.get("status") == "FAILED" for value in scores.values()):
            continue
        row_index = case.get("row_index")
        if row_index is not None:
            indices.append(int(row_index))
    return indices


def scorer_pass_rates(case_results: List[EvalCaseResult]) -> List[Dict[str, Any]]:
    """Return per-scorer score cards in first-seen order."""
    totals: Dict[str, int] = {}
    passed_counts: Dict[str, int] = {}
    order: List[str] = []

    for case in case_results:
        scores = case.get("scores") or {}
        for title, value in scores.items():
            if title not in totals:
                order.append(title)
                totals[title] = 0
                passed_counts[title] = 0
            totals[title] += 1
            if not scorer_value_failed(value):
                passed_counts[title] += 1

    return [
        {
            "scorer": title,
            "passed": passed_counts[title],
            "total": totals[title],
            "pass_rate": passed_counts[title] / totals[title] if totals[title] else 0.0,
        }
        for title in order
    ]


def extract_boolean_score_counts(score: Any) -> Optional[Tuple[int, int]]:
    """Return ``(success_count, total_count)`` for boolean aggregates."""
    if not isinstance(score, dict):
        return None
    aggregate = score.get("aggregate")
    if not isinstance(aggregate, dict):
        return None
    success_count = aggregate.get("success_count")
    total_count = aggregate.get("total_count")
    if (
        isinstance(success_count, (int, float))
        and isinstance(total_count, (int, float))
        and not isinstance(success_count, bool)
        and not isinstance(total_count, bool)
        and total_count > 0
    ):
        return int(success_count), int(total_count)
    return None


def extract_overall_score(score: Any) -> Optional[float]:
    """Return the numeric overall sheet score, if present."""
    if score is None:
        return None
    if isinstance(score, (int, float)) and not isinstance(score, bool):
        return float(score)
    if not isinstance(score, dict):
        return None

    for key in ("aggregate_score", "overall_score"):
        if key in score:
            extracted = extract_overall_score(score.get(key))
            if extracted is not None:
                return extracted

    aggregate = score.get("aggregate")
    if isinstance(aggregate, dict):
        if "value" in aggregate:
            extracted = extract_overall_score(aggregate.get("value"))
            if extracted is not None:
                return extracted
        counts = extract_boolean_score_counts(score)
        if counts is not None:
            return float(counts[0]) / float(counts[1])

    columns = score.get("columns")
    if isinstance(columns, list) and columns:
        values = [
            extract_overall_score(column.get("score") if isinstance(column, dict) else None) for column in columns
        ]
        present = [value for value in values if value is not None]
        if present:
            return sum(present) / len(present)

    nested = score.get("score")
    if isinstance(nested, dict):
        if "score" in nested:
            return extract_overall_score(nested.get("score"))
        if "value" in nested:
            return extract_overall_score(nested.get("value"))
        return extract_overall_score(nested)
    if "score" in score:
        return extract_overall_score(score.get("score"))
    return None
