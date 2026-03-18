#!/usr/bin/env python3
"""Parse a Claude transcript and return finalized turn/tool/llm spans as JSON."""

import json
import os
import sys
from datetime import datetime, timezone


def parse_iso_to_ns(raw):
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return int(dt.timestamp() * 1_000_000_000)
    except Exception:
        return None


def safe_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return default


def stringify(value):
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=False)


def content_to_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str) and text:
                    parts.append(text)
                    continue
            serialized = stringify(block)
            if serialized:
                parts.append(serialized)
        return "\n".join(parts).strip()
    if isinstance(content, dict) and content.get("type") == "text":
        text = content.get("text")
        if isinstance(text, str):
            return text
    return stringify(content)


def message_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str) and text:
                    parts.append(text)
        return "\n".join(parts).strip()
    return ""


def flatten_indexed(prefix, items, out):
    for i, item in enumerate(items):
        for key, value in item.items():
            attr_key = f"{prefix}.{i}.{key}"
            if isinstance(value, (dict, list)):
                out[attr_key] = json.dumps(value, ensure_ascii=False)
            else:
                out[attr_key] = value


def append_history_item(history, item):
    if (
        item.get("role") == "user"
        and history
        and history[-1].get("role") == "user"
        and history[-1].get("content") == item.get("content")
    ):
        return
    history.append(item)


def is_tool_result_user(rec):
    if rec.get("type") != "user":
        return False
    content = rec.get("message", {}).get("content")
    return (
        isinstance(content, list)
        and len(content) > 0
        and isinstance(content[0], dict)
        and content[0].get("type") == "tool_result"
    )


def parse_transcript(transcript_path, turn_start_fallback, pending_payloads, expected_session_id=None):
    records = []
    with open(transcript_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if expected_session_id:
                    rec_session_id = rec.get("sessionId") if isinstance(rec, dict) else None
                    if rec_session_id and rec_session_id != expected_session_id:
                        continue
                records.append(rec)
            except Exception:
                continue

    if not records:
        now_ns = int(datetime.now(timezone.utc).timestamp() * 1_000_000_000)
        return {"turn": {"start_ns": now_ns, "end_ns": now_ns}, "tools": [], "llms": []}

    turn_start_idx = 0
    for i in range(len(records) - 1, -1, -1):
        rec = records[i]
        if rec.get("type") != "user":
            continue
        if is_tool_result_user(rec):
            continue
        turn_start_idx = i
        break

    history = []
    tools = []
    llms = []
    pending_tool_uses = []
    pending_payload_idx = 0
    saw_human_input = False

    turn_start_ns = turn_start_fallback
    turn_end_ns = turn_start_fallback
    last_input_ns = turn_start_fallback

    for idx, rec in enumerate(records):
        emit_for_turn = idx >= turn_start_idx
        timestamp_ns = parse_iso_to_ns(rec.get("timestamp"))
        if emit_for_turn and timestamp_ns is not None:
            if turn_start_ns is None or timestamp_ns < turn_start_ns:
                turn_start_ns = timestamp_ns
            if turn_end_ns is None or timestamp_ns > turn_end_ns:
                turn_end_ns = timestamp_ns

        rec_type = rec.get("type")
        if rec_type == "queue-operation":
            operation = stringify(rec.get("operation"))
            if operation == "enqueue":
                content = content_to_text(rec.get("content"))
                if content:
                    append_history_item(history, {"role": "user", "content": content})
                    last_input_ns = timestamp_ns or last_input_ns
                    saw_human_input = True
            continue

        if rec_type == "user":
            content = rec.get("message", {}).get("content")
            if is_tool_result_user(rec):
                block = content[0]
                tool_use_id = stringify(block.get("tool_use_id"))
                tool_result_content = block.get("content")
                is_error = bool(block.get("is_error", False))

                match_idx = None
                for idx, item in enumerate(pending_tool_uses):
                    if tool_use_id and item.get("id") == tool_use_id:
                        match_idx = idx
                        break
                if match_idx is None and pending_tool_uses:
                    match_idx = 0
                tool_use = pending_tool_uses.pop(match_idx) if match_idx is not None else {}

                payload = {}
                if emit_for_turn and pending_payload_idx < len(pending_payloads):
                    maybe_payload = pending_payloads[pending_payload_idx]
                    pending_payload_idx += 1
                    if isinstance(maybe_payload, dict):
                        payload = maybe_payload

                tool_name = stringify(payload.get("tool_name")) or stringify(tool_use.get("name")) or "Tool"
                function_input = payload.get("function_input", tool_use.get("input", {}))
                function_output = payload.get(
                    "function_output",
                    {"content": tool_result_content, "is_error": is_error},
                )

                tool_start_ns = safe_int(tool_use.get("start_ns"), 0) or timestamp_ns or turn_start_ns
                tool_end_ns = timestamp_ns or tool_start_ns
                if tool_start_ns is None:
                    tool_start_ns = tool_end_ns
                if tool_end_ns is None:
                    tool_end_ns = tool_start_ns

                if emit_for_turn:
                    tools.append(
                        {
                            "name": f"Tool: {tool_name}",
                            "start_ns": int(tool_start_ns),
                            "end_ns": int(tool_end_ns),
                            "attributes": {
                                "source": "claude-code",
                                "hook": "PostToolUse",
                                "node_type": "CODE_EXECUTION",
                                "tool_name": tool_name,
                                "function_input": function_input,
                                "function_output": function_output,
                            },
                        }
                    )

                history.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_use_id,
                        "content": stringify(tool_result_content),
                    }
                )
                last_input_ns = timestamp_ns or last_input_ns
                continue

            user_text = content_to_text(content)
            append_history_item(history, {"role": "user", "content": user_text})
            last_input_ns = timestamp_ns or last_input_ns
            saw_human_input = True
            continue

        if rec_type != "assistant":
            continue

        msg = rec.get("message", {})
        model = stringify(msg.get("model")) or "claude"
        msg_id = stringify(msg.get("id"))
        stop_reason = stringify(msg.get("stop_reason"))
        usage = msg.get("usage", {})
        if not isinstance(usage, dict):
            usage = {}

        prompt_tokens = safe_int(usage.get("input_tokens"), 0)
        completion_tokens = safe_int(usage.get("output_tokens"), 0)
        output_text = message_text(msg.get("content"))

        tool_calls = []
        content_blocks = msg.get("content")
        if isinstance(content_blocks, list):
            for block in content_blocks:
                if not isinstance(block, dict) or block.get("type") != "tool_use":
                    continue
                call_id = stringify(block.get("id"))
                call_name = stringify(block.get("name")) or "tool"
                call_input = block.get("input", {})
                tool_calls.append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": call_name,
                            "arguments": json.dumps(call_input, ensure_ascii=False),
                        },
                    }
                )
                pending_tool_uses.append(
                    {
                        "id": call_id,
                        "name": call_name,
                        "input": call_input,
                        "start_ns": timestamp_ns,
                    }
                )

        # Claude can emit intermediate assistant records that contain only
        # empty thinking blocks. Those should not consume the user's prompt.
        if not output_text and not tool_calls:
            continue

        llm_start_ns = last_input_ns or timestamp_ns or turn_start_ns
        llm_end_ns = timestamp_ns or llm_start_ns
        if llm_start_ns is None:
            llm_start_ns = llm_end_ns
        if llm_end_ns is None:
            llm_end_ns = llm_start_ns

        provider = "anthropic" if model.startswith("claude") else "unknown"

        attrs = {
            "source": "claude-code",
            "hook": "Stop",
            "node_type": "PROMPT_TEMPLATE",
            "promptlayer.prompt_history_mode": "full_session",
            "gen_ai.operation.name": "chat",
            "gen_ai.provider.name": provider,
            "gen_ai.request.model": model,
            "gen_ai.response.model": model,
            "gen_ai.usage.input_tokens": prompt_tokens,
            "gen_ai.usage.output_tokens": completion_tokens,
        }
        if msg_id:
            attrs["gen_ai.response.id"] = msg_id
        if stop_reason:
            attrs["gen_ai.completion.0.finish_reason"] = stop_reason

        flatten_indexed("gen_ai.prompt", history, attrs)

        completion_item = {"role": "assistant", "content": output_text}
        if tool_calls:
            completion_item["tool_calls"] = tool_calls
        flatten_indexed("gen_ai.completion", [completion_item], attrs)

        span_name = "LLM Call (User)" if saw_human_input else "LLM call"

        if emit_for_turn:
            llms.append(
                {
                    "name": span_name,
                    "start_ns": int(llm_start_ns),
                    "end_ns": int(llm_end_ns),
                    "attributes": attrs,
                }
            )

        assistant_history = {"role": "assistant", "content": output_text}
        if tool_calls:
            assistant_history["tool_calls"] = tool_calls
        history.append(assistant_history)
        saw_human_input = False

    if turn_start_ns is None:
        turn_start_ns = int(datetime.now(timezone.utc).timestamp() * 1_000_000_000)
    if turn_end_ns is None:
        turn_end_ns = turn_start_ns

    return {
        "turn": {"start_ns": int(turn_start_ns), "end_ns": int(turn_end_ns)},
        "tools": tools,
        "llms": llms,
    }


def main():
    if len(sys.argv) < 3:
        print(
            json.dumps(
                {"error": "Usage: parse_stop_transcript.py <transcript_path> <turn_start_ns> [session_id]"}
            )
        )
        return 1

    transcript_path = sys.argv[1]
    turn_start_fallback = safe_int(sys.argv[2], 0) or None
    expected_session_id = sys.argv[3] if len(sys.argv) > 3 else None

    pending_raw = os.environ.get("PL_PENDING_TOOL_CALLS", "[]")
    try:
        pending_payloads = json.loads(pending_raw)
    except Exception:
        pending_payloads = []
    if not isinstance(pending_payloads, list):
        pending_payloads = []

    parsed = parse_transcript(transcript_path, turn_start_fallback, pending_payloads, expected_session_id)
    print(json.dumps(parsed, ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
