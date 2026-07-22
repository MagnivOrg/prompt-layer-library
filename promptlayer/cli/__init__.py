"""PromptLayer command-line interface."""

from __future__ import annotations

import argparse
from typing import Optional, Sequence

from promptlayer.cli.eval_cmd import add_eval_parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="promptlayer",
        description="PromptLayer CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_eval_parser(subparsers)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
