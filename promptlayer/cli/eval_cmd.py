"""``promptlayer eval run <path>`` — run Python evaluation files."""

from __future__ import annotations

import argparse
import ast
import os
import runpy
import signal
import sys
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence

from promptlayer.exceptions import EvaluationFailedError
from promptlayer.evaluations.terminal import EvalTerminal, set_terminal

_SIGINT_COUNT = 0


def _install_double_sigint_handler() -> None:
    """First Ctrl+C asks for confirmation; second force-kills the process."""

    def _handler(signum, frame):  # noqa: ARG001
        global _SIGINT_COUNT
        _SIGINT_COUNT += 1
        if _SIGINT_COUNT == 1:
            sys.stderr.write("\nAre you sure? Press Ctrl+C again to force quit.\n")
            sys.stderr.flush()
            raise KeyboardInterrupt
        os._exit(130)

    signal.signal(signal.SIGINT, _handler)

_SKIP_DIR_NAMES = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".tox",
        ".venv",
        "venv",
        "node_modules",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        "site-packages",
        "dist-packages",
        "dist",
        "build",
    }
)


def add_eval_parser(subparsers: argparse._SubParsersAction) -> None:
    eval_parser = subparsers.add_parser(
        "eval",
        help="Run PromptLayer evals",
    )
    eval_subparsers = eval_parser.add_subparsers(dest="eval_command", required=True)

    run_parser = eval_subparsers.add_parser(
        "run",
        help="Run Python files containing evaluate, aevaluate, or *_eval calls",
    )
    run_parser.add_argument(
        "paths",
        nargs="+",
        help="Eval file(s) or directories to scan for eval entry points",
    )
    run_parser.set_defaults(handler=run_eval_command)


def run_eval_command(args: argparse.Namespace) -> int:
    terminal = EvalTerminal()
    set_terminal(terminal)
    _install_double_sigint_handler()

    try:
        files = discover_eval_files(args.paths)
    except FileNotFoundError as exc:
        terminal.write(str(exc), err=True)
        return 1

    if not files:
        terminal.write("No files containing evaluate(...), aevaluate(...), or *_eval(...) calls were found.", err=True)
        return 1

    _load_dotenv_files(Path.cwd())
    terminal.session_start(collected=len(files), label="file")

    passed = 0
    failed = 0
    try:
        for index, path in enumerate(files, start=1):
            display = _display_path(path)
            terminal.file_start(display, index=index, total=len(files))
            failure_message: Optional[str] = None
            try:
                _run_eval_file(path)
            except KeyboardInterrupt:
                terminal.write("Eval interrupted.", err=True)
                terminal.session_end(passed=passed, failed=failed + 1)
                return 130
            except SystemExit as exc:
                code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
                if code:
                    failure_message = f"exited with code {code}"
            except EvaluationFailedError as exc:
                failure_message = str(exc)
            except Exception:  # noqa: BLE001 - surface file failures to the CLI
                failure_message = traceback.format_exc()

            if failure_message is None:
                passed += 1
                terminal.file_passed()
            else:
                failed += 1
                terminal.file_failed(display, failure_message)
    except KeyboardInterrupt:
        terminal.write("Eval interrupted.", err=True)
        terminal.session_end(passed=passed, failed=failed + 1)
        return 130

    terminal.session_end(passed=passed, failed=failed)
    return 1 if failed else 0


def discover_eval_files(paths: Sequence[str]) -> List[Path]:
    discovered: List[Path] = []
    seen = set()
    for raw in paths:
        path = Path(raw).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Path not found: {raw}")
        for file_path in _iter_python_files(path):
            key = str(file_path)
            if key in seen or not _contains_eval_call(file_path):
                continue
            seen.add(key)
            discovered.append(file_path)
    return discovered


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def _iter_python_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        if path.suffix == ".py":
            yield path
        return

    for root, dirnames, filenames in os.walk(path):
        dirnames[:] = sorted(name for name in dirnames if name not in _SKIP_DIR_NAMES and not name.startswith("."))
        for filename in sorted(filenames):
            if filename.endswith(".py") and not filename.startswith("."):
                yield Path(root) / filename


def _contains_eval_call(path: Path) -> bool:
    """Return whether a Python file statically calls an eval entry point."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError, UnicodeError):
        return False

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if isinstance(function, ast.Name):
            name = function.id
        elif isinstance(function, ast.Attribute):
            name = function.attr
        else:
            continue
        normalized = name.lower()
        if normalized in {"evaluate", "aevaluate"} or normalized.endswith("_eval"):
            return True
    return False


@contextmanager
def _isolated_eval_runtime(path: Path) -> Iterator[None]:
    path = path.resolve()
    parent = str(path.parent)
    added_to_path = parent not in sys.path
    if added_to_path:
        sys.path.insert(0, parent)

    previous_argv = sys.argv[:]
    cwd = Path.cwd()
    try:
        sys.argv = [str(path)]
        os.chdir(path.parent)
        yield
    finally:
        sys.argv = previous_argv
        os.chdir(cwd)
        if added_to_path:
            try:
                sys.path.remove(parent)
            except ValueError:
                pass


def _run_eval_file(path: Path) -> None:
    with _isolated_eval_runtime(path):
        runpy.run_path(str(path.resolve()), run_name="__main__")


def _load_dotenv_files(start: Path) -> None:
    """Load common .env files without adding a dotenv dependency."""
    candidates = (
        start / ".env.development.local",
        start / ".env.local",
        start / ".env.development",
        start / ".env",
    )
    for env_path in candidates:
        if env_path.is_file():
            _apply_dotenv(env_path)


def _apply_dotenv(path: Path) -> None:
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        os.environ.setdefault(key, value)
