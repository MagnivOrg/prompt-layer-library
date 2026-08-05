"""``promptlayer setup`` — argparse wiring for coding-agent setup."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from promptlayer.cli.setup.run import run_setup_command


def add_setup_parser(subparsers: argparse._SubParsersAction) -> None:
    setup_parser = subparsers.add_parser(
        "setup",
        help="Install PromptLayer coding-agent skills and the Docs MCP server",
    )
    _add_setup_options(setup_parser)
    setup_parser.set_defaults(handler=run_setup_all, setup_target="all")

    setup_subparsers = setup_parser.add_subparsers(dest="setup_command")

    skills_parser = setup_subparsers.add_parser(
        "skills",
        help="Install PromptLayer skill files for coding agents",
    )
    _add_setup_options(skills_parser)
    skills_parser.set_defaults(handler=run_setup_skills, setup_target="skills")

    mcp_parser = setup_subparsers.add_parser(
        "mcp",
        help="Install the PromptLayer Docs MCP server into agent configs",
    )
    _add_setup_options(mcp_parser)
    mcp_parser.set_defaults(handler=run_setup_mcp, setup_target="mcp")


def _add_setup_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-a",
        "--agent",
        dest="agents",
        action="append",
        help='Agents to configure (cursor, claude, codex, or "*"). Repeatable.',
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Overwrite existing skill or MCP config entries",
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,  # test/cwd override
    )
    parser.add_argument(
        "--evals-url",
        default=None,
        help=argparse.SUPPRESS,  # test override
    )


def run_setup_all(args: argparse.Namespace) -> int:
    return _run(args, target="all")


def run_setup_skills(args: argparse.Namespace) -> int:
    return _run(args, target="skills")


def run_setup_mcp(args: argparse.Namespace) -> int:
    return _run(args, target="mcp")


def _run(args: argparse.Namespace, *, target: str) -> int:
    try:
        kwargs = {
            "cwd": args.path,
            "agents": args.agents,
            "force": args.force,
            "target": target,
        }
        if getattr(args, "evals_url", None):
            kwargs["evals_url"] = args.evals_url
        run_setup_command(**kwargs)
    except (ValueError, RuntimeError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0
