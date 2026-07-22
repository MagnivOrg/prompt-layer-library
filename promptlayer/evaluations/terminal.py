"""Pytest-style terminal output for eval runs (powered by rich)."""

from __future__ import annotations

import json
import sys
import time
from contextvars import ContextVar
from typing import Any, Dict, List, Optional, Sequence, Tuple

from rich.console import Console
from rich.status import Status
from rich.table import Table
from rich.text import Text

from promptlayer.evaluations.scores import (
    extract_boolean_score_counts,
    extract_overall_score,
    llm_assertion_verdict,
)

_FAILURE_EXAMPLE_LIMIT = 5
_CELL_DISPLAY_LIMIT = 80


class EvalTerminal:
    """Shared status printer for ``evaluate(...)`` and ``promptlayer eval run``."""

    def __init__(self, console: Optional[Console] = None) -> None:
        self.console = console or Console()
        self.err_console = Console(stderr=True)
        self._started_at: Optional[float] = None
        self._status: Optional[Status] = None

    @property
    def _interactive(self) -> bool:
        # Prefer live stdout over the Console created at import time — pytest
        # capture replaces sys.stdout after module import.
        return bool(getattr(sys.stdout, "isatty", lambda: False)())

    def write(self, message: str = "", *, err: bool = False) -> None:
        self._stop_status()
        (self.err_console if err else self.console).print(message)

    def rule(self, title: str = "", *, character: str = "=") -> None:
        self._stop_status()
        self.console.rule(f"[bold cyan]{title}[/]" if title else None, characters=character, style="bold cyan")

    def session_start(self, *, collected: int, label: str = "file") -> None:
        self._started_at = time.monotonic()
        self.rule("eval session starts")
        noun = label if collected == 1 else f"{label}s"
        self.write(f"collected {collected} {noun}")
        self.write()

    def file_start(self, path: str, *, index: int, total: int) -> None:
        label = path if total == 1 else f"{path} ({index}/{total})"
        self.write(Text(label, style="bold"))

    def step(self, message: str) -> None:
        text = message.rstrip(".")
        self.write(Text.assemble(("  • ", "cyan"), text))

    def runners_start(self, total: int) -> None:
        if total <= 0:
            return
        self._stop_status()
        if self._interactive:
            self._status = self.console.status(
                self._runners_text(0, total),
                spinner="dots",
                spinner_style="cyan",
            )
            self._status.start()
            return
        self.console.print(Text.assemble(("    • ", "cyan"), self._runners_text(0, total)))

    def progress(self, completed: int, total: int) -> None:
        done = completed >= total > 0
        if self._status is not None:
            self._status.update(self._runners_text(completed, total))
            if done:
                self._stop_status()
                self.console.print(Text.assemble(("    ✓ ", "green"), self._runners_text(completed, total)))
            return

        icon = "✓" if done else "•"
        style = "green" if done else "cyan"
        self.console.print(Text.assemble((f"    {icon} ", style), self._runners_text(completed, total)))

    def scoring_progress(self, completed: int, total: int, failed: int = 0) -> None:
        self._counted_progress("scorecard rows", completed, total, failed)

    def cell_progress(
        self,
        completed: int,
        total: int,
        failed: int = 0,
        status: Optional[str] = None,
    ) -> None:
        self._counted_progress("cells", completed, total, failed, status=status)

    def _counted_progress(
        self,
        label: str,
        completed: int,
        total: int,
        failed: int = 0,
        *,
        status: Optional[str] = None,
    ) -> None:
        if total <= 0 and not status:
            return
        safe_total = max(int(total), 0)
        clamped = min(max(int(completed), 0), safe_total) if safe_total > 0 else max(int(completed), 0)
        failed = max(int(failed), 0)
        status_value = status.strip().lower() if isinstance(status, str) and status.strip() else None
        text = self._counted_text(label, clamped, safe_total, failed, status_value)
        done = clamped >= safe_total if safe_total > 0 else status_value in {"completed", "failed", "cancelled"}
        if self._status is not None:
            self._status.update(text)
            if done:
                self._stop_status()
                style = "red" if failed or status_value == "failed" else "green"
                icon = "✗" if failed or status_value == "failed" else "✓"
                self.console.print(Text.assemble((f"    {icon} ", style), text))
            return

        if self._interactive and not done:
            self._status = self.console.status(text, spinner="dots", spinner_style="cyan")
            self._status.start()
            return

        if done and (failed or status_value == "failed"):
            icon, style = "✗", "red"
        elif done:
            icon, style = "✓", "green"
        else:
            icon, style = "•", "cyan"
        self.console.print(Text.assemble((f"    {icon} ", style), text))

    def score(self, value: str, *, passed: Optional[bool] = None) -> None:
        icon = "✓" if passed is True else "✗" if passed is False else "•"
        style = "green" if passed is True else "red" if passed is False else "cyan"
        self.write(Text.assemble((f"  {icon} ", style), ("score ", ""), (value, f"bold {style}")))

    def evaluation_results(self, rows: Sequence[Dict[str, Any]]) -> None:
        self.write(Text("Evaluation Results:", style="bold"))
        table = Table(show_header=True, header_style="bold")
        table.add_column("Scorer")
        table.add_column("Result", justify="right")
        for row in rows:
            title = str(row.get("scorer") or "")
            passed = int(row.get("passed") or 0)
            total = int(row.get("total") or 0)
            table.add_row(title, format_pass_rate(passed, total))
        self.write(table)
        self.write()

    def failure_examples(
        self,
        cases: Sequence[Dict[str, Any]],
        *,
        scorer_titles: Sequence[str],
        limit: int = _FAILURE_EXAMPLE_LIMIT,
    ) -> None:
        if not cases:
            return
        self.write(Text("Failure examples:", style="bold"))
        table = Table(show_header=True, header_style="bold", show_lines=True)
        table.add_column("Input")
        table.add_column("Output")
        for title in scorer_titles:
            table.add_column(str(title))
        for case in list(cases)[:limit]:
            scores = case.get("scores") or {}
            cells = [
                format_cell_value(case.get("input")),
                format_cell_value(case.get("output")),
            ]
            for title in scorer_titles:
                cells.append(format_scorer_value(scores.get(title)))
            table.add_row(*cells)
        self.write(table)
        self.write()

    def link(self, url: str) -> None:
        self.write(Text.assemble(("  ↗ ", "cyan"), (url, "cyan")))

    def file_passed(self) -> None:
        self.write(Text("  PASSED", style="bold green"))
        self.write()

    def file_failed(self, path: str, detail: str) -> None:
        self.write(Text.assemble(("  FAILED ", "bold red"), path), err=True)
        for line in detail.rstrip().splitlines() or [detail]:
            self.write(f"    {line}", err=True)
        self.write()

    def session_end(self, *, passed: int, failed: int) -> None:
        elapsed = 0.0 if self._started_at is None else time.monotonic() - self._started_at
        total = passed + failed
        if failed:
            summary = f"{failed} failed"
            if passed:
                summary = f"{failed} failed, {passed} passed"
            style = "bold red"
        elif total == 0:
            summary = "no tests ran"
            style = "bold yellow"
        else:
            summary = f"{passed} passed"
            style = "bold green"
        self._stop_status()
        self.console.rule(
            Text(f"{summary} in {elapsed:.2f}s", style=style),
            characters="=",
            style="bold cyan",
        )

    def _runners_text(self, completed: int, total: int) -> Text:
        return Text.assemble(("runners ", ""), (f"{completed}/{total}", "bold"))

    def _counted_text(
        self,
        label: str,
        completed: int,
        total: int,
        failed: int,
        status: Optional[str],
    ) -> Text:
        count = f"{completed}/{total}" if total > 0 else str(completed)
        parts: List[Tuple[str, str]] = [(f"{label} ", ""), (count, "bold")]
        if failed:
            parts.append((f" ({failed} errors)", "red"))
        if status:
            parts.append((f" · {status}", "dim"))
        return Text.assemble(*parts)

    def _stop_status(self) -> None:
        if self._status is None:
            return
        self._status.stop()
        self._status = None


_DEFAULT_TERMINAL = EvalTerminal()
_TERMINAL: ContextVar[Optional[EvalTerminal]] = ContextVar("promptlayer_eval_terminal", default=None)


def get_terminal() -> EvalTerminal:
    return _TERMINAL.get() or _DEFAULT_TERMINAL


def set_terminal(terminal: Optional[EvalTerminal]) -> None:
    current = get_terminal()
    current._stop_status()
    _TERMINAL.set(terminal or EvalTerminal())


def format_score_value(score: Any) -> str:
    if score is None:
        return "n/a"
    overall = extract_overall_score(score)
    counts = extract_boolean_score_counts(score)
    if overall is not None and counts is not None:
        return f"{overall} ({counts[0]}/{counts[1]})"
    if overall is not None:
        return str(overall)
    if isinstance(score, dict):
        status = score.get("status")
        if status is not None:
            return f"status={status}"
    return str(score)


def format_pass_rate(passed: int, total: int) -> str:
    if total <= 0:
        return "n/a"
    percent = int(round((100.0 * passed) / total))
    return f"{passed}/{total} ({percent}%)"


def format_scorer_value(value: Any) -> str:
    """Format structured LLM assertion results as their boolean verdict."""
    verdict = llm_assertion_verdict(value)
    return format_cell_value(verdict if verdict is not None else value)


def format_cell_value(value: Any, *, limit: int = _CELL_DISPLAY_LIMIT) -> str:
    if value is None:
        text = ""
    elif isinstance(value, bool):
        text = "true" if value else "false"
    elif isinstance(value, (int, float)):
        text = str(value)
    elif isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        except TypeError:
            text = str(value)
    text = " ".join(text.split())
    if len(text) > limit:
        return text[: max(0, limit - 3)] + "..."
    return text
