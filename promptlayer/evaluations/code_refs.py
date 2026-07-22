"""Detect Table column title references in CODE_EXECUTION snippets."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Set


def code_references_column_title(code: str, title: str) -> bool:
    """True when JS/Python-ish code references a Table column title."""
    return (
        f'data.get("{title}")' in code
        or f"data.get('{title}')" in code
        or f'data["{title}"]' in code
        or f"data['{title}']" in code
        or f"data.{title}" in code
    )


def infer_referenced_column_titles(code: str, titles: Iterable[str]) -> Set[str]:
    return {title for title in titles if code_references_column_title(code, title)}


def code_references_any_title(code: Any, titles: Iterable[str]) -> bool:
    if not isinstance(code, str):
        return False
    return any(code_references_column_title(code, title) for title in titles)


def config_references_titles(config: Optional[Dict[str, Any]], titles: Iterable[str]) -> bool:
    """True when config named sources or CODE_EXECUTION code reference any title."""
    if not isinstance(config, dict):
        return False
    title_set = set(titles)
    from promptlayer.evaluations.validation import iter_scorer_sources

    if any(source in title_set for source, _key, _meta in iter_scorer_sources(config)):
        return True
    return code_references_any_title(config.get("code"), title_set)
