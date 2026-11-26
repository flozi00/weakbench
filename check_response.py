"""Score a saved chat completion locally against a verifiable item in the prepared private release."""
import argparse
import json
from pathlib import Path
import sys

import hidden
import quality


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=hidden.PLAIN_TASKS)
    parser.add_argument("--lang", required=True, choices=hidden.LANGS)
    parser.add_argument("--id", required=True, help="item ID from the run's items-<task>.jsonl")
    parser.add_argument("--response", type=Path, help="chat completion JSON; for adaptive measurements supply "
                        "an object with messages (assistant calls and tool results) and completion (default: stdin)")
    args = parser.parse_args(argv)
    try:
        cells = hidden.release()
    except (OSError, ValueError):
        parser.error("private release unavailable or invalid; run python hidden.py prepare")
    task = hidden.tasks(cells, None)[args.task]
    item = next((item for item in task["load"](args.lang, None, 0) if item["id"] == args.id), None)
    if item is None:
        parser.error("unknown item for this task, language and release")
    try:
        data = args.response.read_bytes() if args.response else sys.stdin.buffer.read()
        completion = hidden.strict_json(data)
        trace = None
        if item.get("capture_tool_transcript") and isinstance(completion, dict) and "completion" in completion:
            if set(completion) != {"messages", "completion"} or not isinstance(completion["messages"], list):
                raise ValueError
            trace, completion = completion["messages"], completion["completion"]
        message = completion["choices"][0]["message"]
        if not isinstance(message, dict) or not any(k in message for k in ("content", "reasoning", "reasoning_content")):
            raise ValueError
        reply = quality.parse_reply(completion)
        if trace is not None:
            reply["observation_trace"] = trace
        result = task["score"](item, reply)
    except (OSError, ValueError, TypeError, KeyError, IndexError, AttributeError, RecursionError):
        parser.error("cannot read or score chat completion JSON")
    print(json.dumps({"correct": result["correct"], "score": result["score"],
                      "checks": hidden.public_checks(args.task, result.get("detail", {}))}, sort_keys=True))
    return 0 if result["correct"] else 1


if __name__ == "__main__":
    sys.exit(main())
