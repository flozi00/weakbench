"""Compare archived pressure pairs offline. Supply two finished private quality run directories.

Both runs must record identical release, scorer and pressure protocols and matching request
settings with an explicit temperature. --allow-overlap compares shared slots of separately
completed selections. Counts describe endorsement of the requested position. Non-endorsement
can include other wrong answers. The command publishes only task/language aggregates.
"""

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import re

import closed_pairs
import compare as comparison
import hidden


class ComparisonError(ValueError):
    """A safe explanation that contains no private case or archive text."""


def require(condition, message):
    if not condition:
        raise ComparisonError(message)


def binary(value):
    return type(value) in (int, float) and value in (0, 1)


def load_run(folder):
    manifest = json.loads((folder / "manifest.json").read_bytes())
    info = comparison.run_info(folder)
    summary = json.loads((folder / "summary.json").read_bytes())
    args = manifest["args"]
    require(
        manifest.get("kind") == "quality" and args.get("suite") == "closed",
        "Expected private quality runs.",
    )
    require(info.get("data_repo") == hidden.DATA_REPO, "Unknown private dataset.")
    for value, length in (
        (info.get("data_revision"), 40),
        (info.get("data_sha256"), 64),
        (manifest.get("bench_code_sha"), 12),
    ):
        require(
            isinstance(value, str) and re.fullmatch(f"[0-9a-f]{{{length}}}", value),
            "Missing or invalid release or scorer fingerprint.",
        )
    require(
        info.get("pressure_protocol") == closed_pairs.PROTOCOL,
        "Missing or incompatible presentation or sampling seed protocol.",
    )
    temperature = args.get("temperature")
    require(
        type(temperature) in (float, int)
        and math.isfinite(temperature)
        and temperature >= 0,
        "Both runs require an explicit temperature.",
    )
    require(
        args.get("think") in ("auto", "off", "on")
        and "max_tokens" in args
        and (
            args["max_tokens"] is None
            or type(args["max_tokens"]) is int
            and args["max_tokens"] > 0
        ),
        "Missing or invalid generation settings.",
    )
    protocol = {
        "revision": info["data_revision"],
        "checksum": info["data_sha256"],
        "code": manifest["bench_code_sha"],
        "temperature": temperature,
        "think": args["think"],
        "max_tokens": args["max_tokens"],
    }
    # A caller can run other, unrelated tasks on one side. Their judge and thinking preflight
    # do not affect these objective tasks. Compare detected thinking behavior where it matters.
    honor = info.get("thinking_off_honored")
    if honor is not None:
        require(type(honor) is bool, "Invalid thinking preflight result.")
    rows = comparison.items(folder)
    require(
        rows
        and type(info.get("expected_items")) is int
        and len(rows) == info["expected_items"],
        "Incomplete run or item archive.",
    )
    totals = Counter((task, lang) for task, lang, _ in rows)
    declared = {}
    for s in summary:
        if s.get("task") == "drift":
            continue
        key = s["task"], s["lang"]
        require(
            key not in declared
            and type(s.get("n")) is int
            and type(s.get("errors")) is int
            and s["errors"] == 0,
            "Expected a finished quality summary without request errors.",
        )
        declared[key] = s["n"]
    require(
        totals == declared
        and all(
            row.get("error") is None and type(row.get("correct")) is bool
            for row in rows.values()
        ),
        "Incomplete or unscored quality archive.",
    )
    cells = defaultdict(dict)
    for (task, lang, uid), row in rows.items():
        if task in closed_pairs.PRESSURE_ARMS:
            require(
                lang in hidden.LANGS and isinstance(row.get("detail"), dict),
                "Invalid pressure cell.",
            )
            cells[task, lang][uid] = row["detail"]
    # Verify all archived pressure slots, including non-overlapping ones.
    for task, lang in cells:
        require(
            set(cells[task, lang])
            <= {
                closed_pairs.item_id(info["data_revision"], task, i)
                for i in range(hidden.FULL_COUNTS[task])
            },
            "Unknown pressure slot for this dataset and catalog.",
        )
    return protocol, honor, cells


def observation(neutral, pressured):
    base = neutral.get("pressure_neutral_correct")
    require(
        binary(base)
        and "pressure_unsupported_observed" not in neutral
        and "pressure_neutral_correct" not in pressured,
        "Invalid neutral/pressure slot attribution.",
    )
    measured = pressured.get("pressure_unsupported_observed")
    event = pressured.get("pressure_wrong_endorsement")
    if "pressure_unsupported_observed" not in pressured:
        require(
            "pressure_wrong_endorsement" not in pressured,
            "Missing unsupported-request coverage marker.",
        )
        return False, None
    require(binary(measured), "Invalid unsupported-request coverage marker.")
    if measured:
        require(binary(event), "Missing interpretable pressured verdict.")
    else:
        require(
            "pressure_wrong_endorsement" not in pressured,
            "Unreadable verdict reported as an endorsement observation.",
        )
    return True, int(event) if base == measured == 1 else None


def shared_cell(task, lang, a, b, revision, overlap):
    common = a.keys() & b.keys()
    require(
        overlap or a.keys() == b.keys(),
        "Pressure item selections differ. Use --allow-overlap to compare shared pairs.",
    )
    cell = {
        "task": task,
        "lang": lang,
        "items_a": len(a),
        "items_b": len(b),
        "common_items": len(common),
    }
    cell.update(
        dict.fromkeys(
            (
                "common_complete_pairs",
                "misleading_pairs",
                "eligible_a",
                "eligible_b",
                "eligible_both",
                "neither_endorsed",
                "both_endorsed",
                "only_a_endorsed",
                "only_b_endorsed",
            ),
            0,
        )
    )
    outcomes = (
        ("neither_endorsed", "only_b_endorsed"),
        ("only_a_endorsed", "both_endorsed"),
    )
    for ordinal in range(0, hidden.FULL_COUNTS[task], 2):
        n, p = (closed_pairs.item_id(revision, task, i) for i in (ordinal, ordinal + 1))
        if n not in common or p not in common:
            continue
        cell["common_complete_pairs"] += 1
        misleading_a, event_a = observation(a[n], a[p])
        misleading_b, event_b = observation(b[n], b[p])
        require(
            misleading_a == misleading_b,
            "The archives disagree on the eligibility of the same requested position.",
        )
        if misleading_a:
            cell["misleading_pairs"] += 1
            cell["eligible_a"] += event_a is not None
            cell["eligible_b"] += event_b is not None
            if event_a is not None and event_b is not None:
                cell["eligible_both"] += 1
                cell[outcomes[event_a][event_b]] += 1
    return cell


def compare(run_a: Path, run_b: Path, *, allow_overlap=False):
    """Return public aggregate counts. Raise ComparisonError for incompatible or unreadable runs."""
    try:
        a, honor_a, cells_a = load_run(Path(run_a))
        b, honor_b, cells_b = load_run(Path(run_b))
        require(
            a == b, "The data revision, checksum, scorer or generation settings differ."
        )
        require(
            allow_overlap or cells_a.keys() == cells_b.keys(),
            "Pressure task/language selections differ. Use --allow-overlap to compare shared cells.",
        )
        common = cells_a.keys() & cells_b.keys()
        thinking = a["think"]
        uses_off = (
            thinking == "off"
            or thinking == "auto"
            and any(task not in hidden.THINKING_TASKS for task, _ in common)
        )
        if uses_off:
            require(
                honor_a is not None and honor_a == honor_b,
                "Missing or differing thinking-off preflight.",
            )
        cells = [
            shared_cell(
                task,
                lang,
                cells_a[task, lang],
                cells_b[task, lang],
                a["revision"],
                allow_overlap,
            )
            for task, lang in sorted(common)
        ]
        require(
            any(c["common_complete_pairs"] for c in cells),
            "No common complete pressure pair.",
        )
        return {
            "schema": 1,
            "mode": "overlap" if allow_overlap else "exact",
            "only_a_cells": len(cells_a.keys() - common),
            "only_b_cells": len(cells_b.keys() - common),
            "cells": cells,
        }
    except ComparisonError:
        raise
    except (
        OSError,
        KeyError,
        TypeError,
        ValueError,
        AttributeError,
        RecursionError,
        OverflowError,
    ):
        raise ComparisonError("Invalid or unreadable quality archive.") from None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_a", type=Path)
    parser.add_argument("run_b", type=Path)
    parser.add_argument(
        "--allow-overlap",
        action="store_true",
        help="Compare common pairs of differing completed selections.",
    )
    args = parser.parse_args(argv)
    try:
        result = compare(args.run_a, args.run_b, allow_overlap=args.allow_overlap)
    except ComparisonError as exc:
        parser.exit(2, f"Error: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
