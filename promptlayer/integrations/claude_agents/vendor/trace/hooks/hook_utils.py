#!/usr/bin/env python3

import base64
import binascii
import sys
import time


def hex_to_base64(hex_value: str) -> int:
    raw = binascii.unhexlify(hex_value)
    print(base64.b64encode(raw).decode("ascii"))
    return 0


def now_ns() -> int:
    print(time.time_ns())
    return 0


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit("usage: hook_utils.py <command> [args]")

    command = sys.argv[1]
    if command == "hex_to_base64":
        if len(sys.argv) != 3:
            raise SystemExit("usage: hook_utils.py hex_to_base64 <hex>")
        return hex_to_base64(sys.argv[2])
    if command == "now_ns":
        if len(sys.argv) != 2:
            raise SystemExit("usage: hook_utils.py now_ns")
        return now_ns()

    raise SystemExit(f"unknown command: {command}")


if __name__ == "__main__":
    raise SystemExit(main())
