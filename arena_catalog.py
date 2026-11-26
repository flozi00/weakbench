"""Prepare the private creative league or load an explicit local challenge catalogue."""

import argparse
from collections import Counter
from hashlib import sha256
import os
from pathlib import Path
import re

import hidden

REVISION = "e51dac4eec18e7aa653bdd4632682fad5b955bbf"
REMOTE_PATH = "releases/creative-league-v1/cases.jsonl"
DIGEST = "dce57b1ed99474a6da150c7ef41b85ead401bb9126eb6448919193d27d5269bd"
KINDS = ("text", "web", "three")


def cache_path():
    base = Path(os.environ.get("BENCH_CACHE", Path.home() / ".cache/weakbench"))
    return base / "arena" / REVISION / "cases.jsonl"


def decode(data, *, pinned=False):
    if pinned and sha256(data).hexdigest() != DIGEST:
        raise ValueError("private creative league checksum mismatch")
    try:
        if len(data) > 16 << 20:
            raise ValueError
        try:
            rows = hidden.strict_json(data)
        except ValueError:
            rows = [
                hidden.strict_json(line) for line in data.splitlines() if line.strip()
            ]
        if isinstance(rows, dict):
            rows = rows["cases"] if set(rows) == {"cases"} else [rows]
        if not isinstance(rows, list) or not 1 <= len(rows) <= 4096:
            raise ValueError
        ids = set()
        for row in rows:
            if not isinstance(row, dict) or set(row) != {
                "id",
                "kind",
                "language",
                "prompt",
                "rubric",
            }:
                raise ValueError
            if not re.fullmatch(r"[a-zA-Z0-9_-]{1,100}", row["id"]) or row["id"] in ids:
                raise ValueError
            ids.add(row["id"])
            if row["kind"] not in KINDS or row["language"] not in hidden.LANGS:
                raise ValueError
            if (
                not isinstance(row["prompt"], str)
                or not 0 < len(row["prompt"].strip()) <= 32000
            ):
                raise ValueError
            rules = row["rubric"]
            if not isinstance(rules, list) or not 1 <= len(rules) <= 24:
                raise ValueError
            if not all(
                isinstance(rule, str) and 0 < len(rule.strip()) <= 2000
                for rule in rules
            ):
                raise ValueError
        if pinned and Counter((c["kind"], c["language"]) for c in rows) != {
            (kind, lang): 4 for kind in KINDS for lang in hidden.LANGS
        }:
            raise ValueError
        return rows
    except (TypeError, KeyError, ValueError, RecursionError) as exc:
        raise ValueError("invalid creative league catalogue") from exc


def prepare():
    hidden.prepare_private(
        hidden.DATA_REPO,
        REVISION,
        REMOTE_PATH,
        cache_path(),
        lambda data: decode(data, pinned=True),
    )


def load_catalog(path=None):
    target = Path(path) if path is not None else cache_path()
    if not target.is_file():
        raise ValueError(
            "Creative league catalogue missing. Run python arena_catalog.py prepare or specify a local catalogue."
        )
    return decode(target.read_bytes(), pinned=path is None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "check"))
    parser.add_argument("--catalog", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        if args.catalog:
            parser.error("prepare uses the pinned private release")
        prepare()
    try:
        print(f"{len(load_catalog(args.catalog))} creative challenges ready")
    except (ValueError, OSError) as exc:
        parser.exit(1, f"{exc}\n")
