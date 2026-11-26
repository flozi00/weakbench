"""Score a saved chat completion locally against a verifiable item in the prepared private release."""
import argparse
import json
from pathlib import Path
import sys

import hidden
import quality


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=(*hidden.PLAIN_TASKS, *hidden.VISUAL_COUNTS, *hidden.AUDIO_COUNTS))
    parser.add_argument("--lang", required=True, choices=hidden.LANGS)
    parser.add_argument("--id", required=True, help="item ID from the run's items-<task>.jsonl")
    parser.add_argument("--response", type=Path, help="chat completion JSON; for action postconditions, web searches, source inquiries and adaptive measurements supply "
                        "an object with messages (assistant calls and tool results) and completion. For three-turn text dialogues, "
                        "supply an object with the three ordered chat completions under completions (default: stdin)")
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
        if item.get("turns"):
            if (not isinstance(completion, dict) or set(completion) != {"completions"}
                    or not isinstance(completion["completions"], list)
                    or len(completion["completions"]) != len(item["turns"])):
                raise ValueError
            turns = [quality.observe_text_turn(args.task, item, index, entry)
                     for index, entry in enumerate(completion["completions"])]
            reply = turns[-1][0]
            reply["turns"] = [proposal for _, proposal in turns]
        else:
            trace = None
            if item.get("capture_tool_transcript") and isinstance(completion, dict) and "completion" in completion:
                if set(completion) != {"messages", "completion"} or not isinstance(completion["messages"], list):
                    raise ValueError
                trace, completion = completion["messages"], completion["completion"]
            message = completion["choices"][0]["message"]
            if not isinstance(message, dict) or not any(k in message for k in ("content", "reasoning", "reasoning_content")):
                raise ValueError
            reply = (hidden.closed_web_tools.parse_final(completion, quality.parse_reply) if args.task in ("web_currency","action_postconditions")
                     else quality.parse_reply(completion, all_reasoning=args.task == "reader_gamebook"))
            if trace is not None:
                reply["observation_trace"] = trace
        result = (hidden.closed_relay.finish(reply["turns"]) if args.task == "language_relay"
                  else task["score"](item, reply))
    except (OSError, ValueError, TypeError, KeyError, IndexError, AttributeError, RecursionError):
        parser.error("cannot read or score chat completion JSON")
    print(json.dumps({"correct": result["correct"], "score": result["score"],
                      "checks": hidden.public_checks(args.task, result.get("detail", {}))}, sort_keys=True))
    return 0 if result["correct"] else 1


if __name__ == "__main__":
    sys.exit(main())
