#!/usr/bin/env python3

import os
import pathlib
import sys


THIS_DIR = pathlib.Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from context import load_context
from handlers import (
    handle_parse_stop_transcript,
    handle_post_tool_use,
    handle_session_end,
    handle_session_start,
    handle_stop_hook,
    handle_user_prompt_submit,
)
from otlp import generate_session_id, generate_span_id, generate_trace_id, probe_endpoint
from settings import write_settings_env


def read_stdin() -> str:
    return sys.stdin.read()


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit("usage: cli.py <command> [args]")

    command = sys.argv[1]
    if command in {"session-start", "user-prompt-submit", "post-tool-use", "session-end", "stop-hook"}:
        ctx = load_context()
        raw_input = read_stdin()
        if command == "session-start":
            output = handle_session_start(ctx, raw_input)
        elif command == "user-prompt-submit":
            output = handle_user_prompt_submit(ctx, raw_input)
        elif command == "post-tool-use":
            output = handle_post_tool_use(ctx, raw_input)
        elif command == "session-end":
            output = handle_session_end(ctx, raw_input)
        else:
            output = handle_stop_hook(ctx, raw_input)
        if output:
            print(output)
        return 0

    if command == "generate-trace-id":
        print(generate_trace_id())
        return 0
    if command == "generate-span-id":
        print(generate_span_id())
        return 0
    if command == "generate-session-id":
        print(generate_session_id())
        return 0
    if command == "write-settings-env":
        if len(sys.argv) != 6:
            raise SystemExit("usage: cli.py write-settings-env <settings_file> <api_key> <endpoint> <debug>")
        print(write_settings_env(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]))
        return 0
    if command == "probe-endpoint":
        if len(sys.argv) != 4:
            raise SystemExit("usage: cli.py probe-endpoint <endpoint> <api_key>")
        print(probe_endpoint(sys.argv[2], sys.argv[3]))
        return 0
    if command == "parse-stop-transcript":
        if len(sys.argv) not in {4, 5}:
            raise SystemExit("usage: cli.py parse-stop-transcript <transcript_path> <turn_start_ns> [session_id]")
        expected_session_id = sys.argv[4] if len(sys.argv) == 5 else None
        print(handle_parse_stop_transcript(sys.argv[2], sys.argv[3], expected_session_id))
        return 0

    raise SystemExit(f"unknown command: {command}")


if __name__ == "__main__":
    raise SystemExit(main())
