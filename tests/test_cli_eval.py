import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from promptlayer.cli import main
from promptlayer.cli.eval_cmd import (
    _apply_dotenv,
    _load_dotenv_files,
    discover_eval_files,
    run_eval_command,
)


def test_discover_eval_files_single_file(tmp_path: Path):
    eval_file = tmp_path / "my.eval.py"
    eval_file.write_text("from promptlayer import evaluate\nevaluate(...)\n", encoding="utf-8")
    assert discover_eval_files([str(eval_file)]) == [eval_file.resolve()]


def test_discover_eval_files_directory_skips_non_evals_and_venv(tmp_path: Path):
    (tmp_path / "skip.py").write_text("evaluate(...)\n", encoding="utf-8")
    (tmp_path / "legacy_eval.py").write_text("evaluate(...)\n", encoding="utf-8")
    nested = tmp_path / "suite"
    nested.mkdir()
    (nested / "nested.eval.py").write_text("import promptlayer as pl\npl.aevaluate(...)\n", encoding="utf-8")
    venv = tmp_path / ".venv" / "lib"
    venv.mkdir(parents=True)
    (venv / "ignored.eval.py").write_text("evaluate(...)\n", encoding="utf-8")

    found = discover_eval_files([str(tmp_path)])
    names = {path.name for path in found}
    assert names == {"nested.eval.py"}


def test_discover_eval_files_supports_wrapper_calls(tmp_path: Path):
    eval_file = tmp_path / "wrapped.eval.py"
    eval_file.write_text("run_suite_eval(dataset=[])\n", encoding="utf-8")

    assert discover_eval_files([str(eval_file)]) == [eval_file.resolve()]


def test_discover_eval_files_ignores_non_eval_filename(tmp_path: Path):
    eval_file = tmp_path / "my_eval.py"
    eval_file.write_text("evaluate(...)\n", encoding="utf-8")
    assert discover_eval_files([str(eval_file)]) == []


def test_discover_eval_files_missing_path(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        discover_eval_files([str(tmp_path / "missing.eval.py")])


def test_apply_dotenv_does_not_override_existing(tmp_path: Path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        'PROMPTLAYER_API_KEY="from-file"\nexport OTHER=1\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("PROMPTLAYER_API_KEY", "already-set")
    monkeypatch.delenv("OTHER", raising=False)

    _apply_dotenv(env_file)

    assert os.environ["PROMPTLAYER_API_KEY"] == "already-set"
    assert os.environ["OTHER"] == "1"


def test_load_dotenv_files_prefers_local_over_env(tmp_path: Path, monkeypatch):
    (tmp_path / ".env").write_text("FOO=from-env\n", encoding="utf-8")
    (tmp_path / ".env.local").write_text("FOO=from-local\nBAR=2\n", encoding="utf-8")
    monkeypatch.delenv("FOO", raising=False)
    monkeypatch.delenv("BAR", raising=False)

    _load_dotenv_files(tmp_path)

    # .env.local is applied before .env; setdefault means first wins.
    assert os.environ["FOO"] == "from-local"
    assert os.environ["BAR"] == "2"


def test_cli_eval_run_executes_file(tmp_path: Path, capsys):
    marker = tmp_path / "ran.txt"
    eval_file = tmp_path / "sample.eval.py"
    eval_file.write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "evaluate = lambda: None",
                "evaluate()",
                f"Path({str(marker)!r}).write_text('ok')",
                "",
            ]
        ),
        encoding="utf-8",
    )

    code = main(["eval", "run", str(eval_file)])
    assert code == 0
    assert marker.read_text(encoding="utf-8") == "ok"
    out = capsys.readouterr().out
    assert "eval session starts" in out
    assert "sample.eval.py" in out
    assert "PASSED" in out
    assert "1 passed" in out
    assert str(tmp_path) not in sys.path


def test_cli_eval_run_resets_argv_for_script_argparse(tmp_path: Path):
    marker = tmp_path / "argv.txt"
    eval_file = tmp_path / "argparse.eval.py"
    eval_file.write_text(
        "\n".join(
            [
                "import argparse",
                "from pathlib import Path",
                "parser = argparse.ArgumentParser()",
                "parser.add_argument('--ok', action='store_true')",
                "args = parser.parse_args()",
                "aevaluate = lambda: None",
                "aevaluate()",
                f"Path({str(marker)!r}).write_text('ok')",
                "",
            ]
        ),
        encoding="utf-8",
    )

    code = main(["eval", "run", str(eval_file)])
    assert code == 0
    assert marker.read_text(encoding="utf-8") == "ok"


def test_cli_eval_run_reports_failure(tmp_path: Path, capsys):
    eval_file = tmp_path / "bad.eval.py"
    eval_file.write_text("if False:\n    evaluate()\nraise RuntimeError('boom')\n", encoding="utf-8")

    code = main(["eval", "run", str(eval_file)])
    assert code == 1
    captured = capsys.readouterr()
    assert "FAILED" in captured.err or "FAILED" in captured.out
    assert "boom" in captured.err
    assert "Traceback" in captured.err
    assert "1 failed" in captured.out


def test_cli_eval_run_reports_evaluation_failed(tmp_path: Path, capsys):
    eval_file = tmp_path / "failing_score.eval.py"
    eval_file.write_text(
        "\n".join(
            [
                "from promptlayer import EvaluationFailedError",
                "if False:",
                "    evaluate()",
                "raise EvaluationFailedError(",
                "    'Evaluation failed: overall score 0.0 is below passing score 1.0',",
                "    score={'score': {'score': 0.0}},",
                "    passing_score=1.0,",
                ")",
            ]
        ),
        encoding="utf-8",
    )

    code = main(["eval", "run", str(eval_file)])
    assert code == 1
    captured = capsys.readouterr()
    err = " ".join(captured.err.split())
    assert "FAILED" in captured.err or "FAILED" in captured.out
    assert "overall score 0.0 is below passing score 1.0" in err
    assert "Traceback" not in captured.err
    assert "assert_passing_score" not in captured.err
    assert "1 failed" in captured.out


def test_cli_eval_run_no_files(tmp_path: Path):
    empty = tmp_path / "empty"
    empty.mkdir()
    code = main(["eval", "run", str(empty)])
    assert code == 1


def test_cli_eval_run_system_exit_nonzero(tmp_path: Path, capsys):
    eval_file = tmp_path / "exit.eval.py"
    eval_file.write_text(
        "\n".join(
            [
                "import sys",
                "evaluate = lambda: None",
                "evaluate()",
                "sys.exit(1)",
                "",
            ]
        ),
        encoding="utf-8",
    )

    code = main(["eval", "run", str(eval_file)])
    assert code == 1
    captured = capsys.readouterr()
    assert "exited with code 1" in captured.err or "exited with code 1" in captured.out
    assert "1 failed" in captured.out


def test_cli_eval_run_ignores_non_python_file(tmp_path: Path):
    text_file = tmp_path / "notes.txt"
    text_file.write_text("evaluate()\n", encoding="utf-8")
    code = main(["eval", "run", str(text_file)])
    assert code == 1


def test_cli_eval_run_fails_when_no_eval_calls(tmp_path: Path, capsys):
    eval_file = tmp_path / "empty.eval.py"
    marker = tmp_path / "should_not_run.txt"
    eval_file.write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('ran')\n",
        encoding="utf-8",
    )

    code = main(["eval", "run", str(eval_file)])
    assert code == 1
    captured = capsys.readouterr()
    assert "No *.eval.py files containing evaluate(...), aevaluate(...), or *_eval(...)" in captured.err
    assert not marker.exists()


def test_run_eval_command_multiple_files(tmp_path: Path):
    a = tmp_path / "a.eval.py"
    b = tmp_path / "b.eval.py"
    a.write_text("evaluate()\n", encoding="utf-8")
    b.write_text("aevaluate()\n", encoding="utf-8")

    with patch("promptlayer.cli.eval_cmd._run_eval_file") as mock_run:
        args = argparse_namespace(paths=[str(a), str(b)])
        code = run_eval_command(args)

    assert code == 0
    assert mock_run.call_count == 2


def argparse_namespace(**kwargs):
    import argparse

    return argparse.Namespace(**kwargs)
