"""Measure judge agreement on labelled private controls using the closed suite's rubric scorer."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import sys
from urllib.parse import urlsplit

import common
import hidden
import tasks_judge

CONTROLS_SHA256 = "0cc5b38b5459abd79d9fbcd312c56aa2e3913653a6aef03b7b106de2056192f4"


class CalibrationError(Exception):
    """User-facing errors with no private item content."""


def load_controls(data, cells):
    if hashlib.sha256(data).hexdigest() != CONTROLS_SHA256:
        raise CalibrationError("private controls checksum mismatch")
    cases = {case["id"]: case for rows in cells.values() for case in rows}
    controls = []
    seen = set()
    try:
        for line in data.splitlines():
            fixture = hidden.strict_json(line)
            case = cases[fixture["id"]]
            label = fixture["label"]
            key = (case["id"], label)
            if (set(fixture) != {"id", "label", "response"} or case["mode"] != "judge"
                    or label not in ("pass", "fail") or key in seen
                    or not isinstance(fixture["response"], str) or not fixture["response"].strip()):
                raise ValueError
            seen.add(key)
            controls.append((case, label, fixture["response"]))
    except (KeyError, TypeError, ValueError, RecursionError):
        raise CalibrationError("invalid controls for the prepared private release") from None
    random.Random(916).shuffle(controls)  # fixed order for repeated calibration on the same inputs
    return controls


def calibrate(controls, judge):
    counts = defaultdict(Counter)
    for case, label, response in controls:
        cell = counts[case["language"], label]
        try:
            grades = hidden.verdict(judge, case["prompt"], case["rubric"], response)
        except json.JSONDecodeError:
            cell["invalid_json"] += 1
        except ValueError:
            cell["invalid_verdict"] += 1
        except Exception:
            # Transport exceptions and server error bodies can echo the request.
            cell["request_errors"] += 1
        else:
            cell["valid"] += 1
            cell["matched" if all(grades) == (label == "pass") else "mismatched"] += 1
    results = []
    for (language, label), count in sorted(counts.items()):
        errors = count["invalid_json"] + count["invalid_verdict"] + count["request_errors"]
        results.append({"language": language, "label": label, "total": count["valid"] + errors,
                        "valid": count["valid"], "matched": count["matched"], "mismatched": count["mismatched"],
                        "errors": errors, "invalid_json": count["invalid_json"],
                        "invalid_verdict": count["invalid_verdict"], "request_errors": count["request_errors"]})
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controls", type=Path, required=True,
                        help="local authoring/closed-v1/calibration.jsonl from the private dataset")
    parser.add_argument("--judge-url", required=True, help="trusted judge server that may receive private controls")
    parser.add_argument("--judge-model", help="served model ID; defaults to the first ID at /v1/models")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--judge-think-budget", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        url = urlsplit(args.judge_url)
        if not url.hostname or url.scheme not in ("http", "https"):
            raise CalibrationError("judge URL requires an HTTP or HTTPS server")
        if args.timeout <= 0 or args.judge_think_budget < 0:
            raise CalibrationError("timeout must be positive and judge think budget nonnegative")
        try:
            controls = load_controls(args.controls.read_bytes(), hidden.release())
        except OSError:
            raise CalibrationError("private controls or release file could not be read") from None
        except ValueError:
            raise CalibrationError("private release unavailable or invalid; run python hidden.py prepare") from None
        common.verify_tls()
        info = common.server_info(args.judge_url, args.judge_model)
        if not info.get("model"):
            raise CalibrationError("judge server returned no model ID")
        tasks_judge.configure(args.judge_url, info["model"], info["max_model_len"],
                              args.timeout, args.judge_think_budget)
        cells = calibrate(controls, tasks_judge.judge)
        print(json.dumps({"judge_model": info["model"], "judge_think_budget": args.judge_think_budget,
                          "data_revision": hidden.DATA_REVISION, "controls_sha256": CONTROLS_SHA256,
                          "cells": cells}, sort_keys=True))
        return int(any(c["errors"] for c in cells))
    except CalibrationError as exc:
        print(f"calibration failed: {exc}", file=sys.stderr)
    except Exception:
        print("calibration failed: judge setup or connection error", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
