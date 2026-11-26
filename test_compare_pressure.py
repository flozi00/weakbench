"""Offline regression controls for paired comparisons across archived quality runs."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

import closed_pairs
import compare_pressure as cp
import hidden


REV = "7" * 40
TASK = "presupposition_repair"


def row(ordinal, detail, task=TASK, lang="en"):
    return {
        "task": task,
        "lang": lang,
        "id": closed_pairs.item_id(REV, task, ordinal),
        "correct": False,
        "error": None,
        "detail": detail,
        "secret": "PRIVATE SENTINEL",
    }


def twins(index, base, event, task=TASK, lang="en"):
    # -1 is an unreadable pressured reply, None a legitimate request.
    d = {}
    if event is not None:
        d["pressure_unsupported_observed"] = int(event != -1)
        if event != -1:
            d["pressure_wrong_endorsement"] = event
    return [
        row(2 * index, {"pressure_neutral_correct": base}, task, lang),
        row(2 * index + 1, d, task, lang),
    ]


def archive(path, rows):
    path.mkdir(exist_ok=True)
    for f in path.glob("items-*.jsonl"):
        f.unlink()
    latest = {(r["task"], r["lang"], r["id"]): r for r in rows}
    manifest = {
        "kind": "quality",
        "bench_code_sha": "1" * 12,
        "server": {"model": "PRIVATE SENTINEL"},
        "args": {
            "suite": "closed",
            "think": "auto",
            "max_tokens": 1024,
            "temperature": 0,
        },
    }
    info = {
        "data_repo": hidden.DATA_REPO,
        "data_sha256": "a" * 64,
        "data_revision": REV,
        "expected_items": len(latest),
        "pressure_protocol": dict(closed_pairs.PROTOCOL),
    }
    cells = {(t, lang) for t, lang, _ in latest}
    summary = [
        {
            "task": t,
            "lang": lang,
            "n": sum(k[:2] == (t, lang) for k in latest),
            "errors": 0,
        }
        for t, lang in cells
    ]
    for name, doc in (("manifest", manifest), ("run_info", info), ("summary", summary)):
        (path / f"{name}.json").write_text(json.dumps(doc))
    for task in {r["task"] for r in rows}:
        (path / f"items-{task}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows if r["task"] == task)
        )


def edit(path, file, mutate):
    target = path / (file + ".json")
    data = json.loads(target.read_bytes())
    mutate(data)
    target.write_text(json.dumps(data))


class CompareTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.a, self.b = (Path(temp.name) / s for s in ("a", "b"))
        self.ra = sum(
            (
                twins(i, n, v)
                for i, (n, v) in enumerate(
                    ((1, 0), (1, 1), (1, 1), (1, 0), (0, 0), (1, -1), (1, None))
                )
            ),
            [],
        )
        self.rb = sum(
            (
                twins(i, n, v)
                for i, (n, v) in enumerate(
                    ((1, 0), (1, 1), (1, 0), (1, 1), (1, 0), (1, -1), (0, None))
                )
            ),
            [],
        )
        archive(self.a, self.ra)
        archive(self.b, self.rb)

    def test_four_matched_outcomes_and_redacted_command(self):
        result = cp.compare(self.a, self.b)
        self.assertEqual(
            result["cells"],
            [
                {
                    "task": TASK,
                    "lang": "en",
                    "items_a": 14,
                    "items_b": 14,
                    "common_items": 14,
                    "common_complete_pairs": 7,
                    "misleading_pairs": 6,
                    "eligible_a": 4,
                    "eligible_b": 5,
                    "eligible_both": 4,
                    "neither_endorsed": 1,
                    "both_endorsed": 1,
                    "only_a_endorsed": 1,
                    "only_b_endorsed": 1,
                }
            ],
        )
        cmd = [
            sys.executable,
            "-BS",
            str(Path(cp.__file__).resolve()),
            str(self.a),
            str(self.b),
        ]
        output = subprocess.run(cmd, capture_output=True, check=True)
        self.assertEqual(json.loads(output.stdout), result)
        self.assertNotIn(b"PRIVATE SENTINEL", output.stdout)
        self.assertNotIn(self.ra[0]["id"].encode(), output.stdout)
        self.assertNotIn(REV.encode(), output.stdout)

    def test_overlap_joins_ids_across_order_language_and_task(self):
        # The first shared pair is in reverse file order. Its neutral half alone is shared for the next slot.
        archive(self.a, self.ra[:4] + twins(0, 1, 0, lang="de"))
        archive(
            self.b,
            [self.rb[2], self.rb[1], self.rb[0]]
            + twins(0, 1, 1, task="evidence_pressure"),
        )
        with self.assertRaises(cp.ComparisonError):
            cp.compare(self.a, self.b)
        result = cp.compare(self.a, self.b, allow_overlap=True)
        self.assertEqual((result["only_a_cells"], result["only_b_cells"]), (1, 1))
        cell = result["cells"][0]
        self.assertEqual(
            [
                cell[k]
                for k in (
                    "items_a",
                    "items_b",
                    "common_items",
                    "common_complete_pairs",
                    "neither_endorsed",
                )
            ],
            [4, 3, 3, 1, 1],
        )
        archive(self.a, self.ra[:4])
        archive(self.b, self.rb[:2])
        with self.assertRaises(cp.ComparisonError):
            cp.compare(self.a, self.b)
        archive(self.b, [self.rb[0]])
        with self.assertRaises(cp.ComparisonError):
            cp.compare(self.a, self.b, allow_overlap=True)

    def test_only_selected_families_matter_and_retries_are_deduplicated(self):
        extra = row(0, {}, task="other")
        first = dict(self.ra[0], error="failed", correct=None)
        archive(self.a, [first, *self.ra, extra])
        self.assertEqual(cp.compare(self.a, self.b)["cells"][0]["common_items"], 14)
        self.assertEqual(cp.compare(self.a, self.b)["only_a_cells"], 0)

    def test_unobserved_is_excluded_from_the_eligible_denominator(self):
        archive(self.a, twins(0, 1, -1))
        archive(self.b, twins(0, 1, 0))
        c = cp.compare(self.a, self.b)["cells"][0]
        self.assertEqual(
            (
                c["misleading_pairs"],
                c["eligible_both"],
                c["eligible_a"],
                c["eligible_b"],
            ),
            (1, 0, 0, 1),
        )
        self.assertEqual(c["neither_endorsed"], 0)

    def test_fingerprints_generation_completion_and_selection(self):
        for file, mutate in (
            ("manifest", lambda d: d.update(kind="speed")),
            ("manifest", lambda d: d.update(bench_code_sha="f" * 12)),
            ("manifest", lambda d: d["args"].update(think="off")),
            ("manifest", lambda d: d["args"].update(max_tokens=64)),
            ("manifest", lambda d: d["args"].update(temperature=None)),
            ("manifest", lambda d: d["args"].update(temperature=0.5)),
            ("run_info", lambda d: d.update(data_sha256="f" * 64)),
            ("run_info", lambda d: d.update(data_revision="a" * 40)),
            ("run_info", lambda d: d.update(data_repo="different")),
            ("run_info", lambda d: d.pop("pressure_protocol")),
            (
                "run_info",
                lambda d: d["pressure_protocol"].update(presentation="serial"),
            ),
            (
                "run_info",
                lambda d: d["pressure_protocol"].update(sampling_seed="per-model"),
            ),
            ("run_info", lambda d: d.update(expected_items=15)),
            ("summary", lambda d: d[0].update(n=11)),
            ("summary", lambda d: d[0].update(errors=1)),
        ):
            archive(self.b, self.rb)
            edit(self.b, file, mutate)
            with (
                self.subTest(file=file, mutation=mutate),
                self.assertRaises(cp.ComparisonError),
            ):
                cp.compare(self.a, self.b, allow_overlap=True)

    def test_rejects_unknown_slots_and_inconsistent_observations(self):
        for index, update in (
            (0, {"id": "f" * 32}),
            (1, {"detail": {"pressure_unsupported_observed": 1}}),
            (
                1,
                {
                    "detail": {
                        "pressure_unsupported_observed": 0,
                        "pressure_wrong_endorsement": 0,
                    }
                },
            ),
            (1, {"detail": {"pressure_wrong_endorsement": 0}}),
            (1, {"detail": {}}),
            (0, {"detail": {"pressure_neutral_correct": True}}),
        ):
            rows = deepcopy(self.rb)
            rows[index].update(update)
            archive(self.b, rows)
            with self.assertRaises(cp.ComparisonError):
                cp.compare(self.a, self.b, allow_overlap=True)


if __name__ == "__main__":
    unittest.main()
