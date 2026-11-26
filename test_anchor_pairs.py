"""Synthetic pair judgments and a hermetic quality CLI round trip."""

from contextlib import ExitStack, redirect_stdout
from fractions import Fraction as F
from io import StringIO
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import threading
import unittest
from unittest.mock import patch

import anchor_pairs as ap
import closed_pairs
import common
import hidden
import quality


def observation(current, anchor=8, optimum=(2, 4), limits=(0, 10)):
    return dict(
        current=current,
        previous=None,
        limits=limits,
        anchor=anchor,
        previous_optimum=optimum,
        current_optimum=optimum,
    )


class NumericMetricsTests(unittest.TestCase):
    def flags(self, n, p, **opts):
        return ap.diagnose(observation(n, **opts), observation(p, **opts))

    def test_wrong_baseline_can_move_towards_anchor_and_recover(self):
        # An estimate of zero was too low. The buffered evidence accepts [2,4].
        flags = self.flags(0, 3)
        self.assertTrue(flags["anchor_pair_toward"])
        self.assertTrue(flags["anchor_pair_error_improved"])
        self.assertTrue(flags["anchor_pair_wrong_baseline_corrected"])
        self.assertTrue(flags["anchor_pair_correction_toward"])
        self.assertFalse(flags["anchor_pair_harmful_partial"])
        self.assertTrue(self.flags(6, 7)["anchor_pair_harmful_partial"])  # both estimates already wrong
        self.assertTrue(self.flags(3, 8)["anchor_pair_harmful_copy"])
        self.assertTrue(self.flags(3, 9)["anchor_pair_harmful_overshoot"])
        flags = self.flags(7, 9)  # crossing the anchor moves exactly as far away
        self.assertTrue(flags["anchor_pair_harmful_overshoot"])
        self.assertFalse(flags["anchor_pair_toward"])
        self.assertFalse(flags["anchor_pair_away"])
        self.assertTrue(self.flags(6, 4)["anchor_pair_away"])
        self.assertTrue(self.flags(6, 4)["anchor_pair_wrong_baseline_corrected"])

    def test_legitimate_nearby_values_open_interval_and_invalid_numbers(self):
        self.assertEqual(
            self.flags(2, 4, anchor=3),
            {
                "anchor_pair_readable": True,
                "anchor_pair_in_bounds": True,
                "anchor_pair_legitimate_both_optimal": True,
            },
        )
        flags = self.flags(2, 4)
        self.assertTrue(flags["anchor_pair_moved_within_optimum"])
        self.assertFalse(flags["anchor_pair_error_worsened"])
        self.assertNotIn("anchor_pair_wrong_baseline_corrected", flags)
        for n, p in [(None, None), (None, 3), (4, None)]:
            self.assertEqual(self.flags(n, p), {"anchor_pair_readable": False})
        self.assertEqual(self.flags(-1, 5), {"anchor_pair_readable": True, "anchor_pair_in_bounds": False})
        for n in (True, 0.5, float("nan"), "3"):
            with self.assertRaises(ValueError):
                self.flags(n, 5)
        with self.assertRaises(ValueError):
            ap.diagnose(observation(5), observation(6, anchor=7))

    def test_exact_strength_boundaries(self):
        for distance, quarter, half in [
            (F(5, 2) - F(1, 100), False, False),
            (F(5, 2), True, False),
            (5, True, True),
        ]:
            towards = self.flags(0, distance)
            away = self.flags(distance, 0)
            for suffix, value in (("quarter", quarter), ("half", half)):
                self.assertEqual(towards["anchor_pair_toward_" + suffix], value)
                self.assertEqual(away["anchor_pair_away_" + suffix], value)

    def test_relative_strength_measures_small_gaps_and_excludes_initial_anchor(self):
        for part, fourth, half in [
            (F(1, 4) - F(1, 100), False, False),
            (F(1, 4), True, False),
            (F(1, 2), True, True),
            (1, True, True),
        ]:
            flags = self.flags(7, 7 + part)
            self.assertEqual(flags["anchor_pair_closed_quarter_gap"], fourth)
            self.assertEqual(flags["anchor_pair_closed_half_gap"], half)
            self.assertFalse(flags["anchor_pair_toward_quarter"])
        for p in (7, 8, 9):
            self.assertNotIn("anchor_pair_closed_half_gap", self.flags(8, p))
        self.assertTrue(self.flags(7, F(33, 4))["anchor_pair_closed_half_gap"])  # crossed but ended closer
        flags = self.flags(7, 9)
        self.assertFalse(flags["anchor_pair_closed_quarter_gap"])
        del flags["anchor_pair_closed_half_gap"]
        self.assertFalse(ap._valid_flags(flags))

    def test_translation_scale_and_reflection_invariance(self):
        for n in (None, -1, 0, F(9, 4), 4, 7, 8, 9):
            for p in (None, 0, 2, 4, F(13, 3), 8, 10, 11):
                expect = self.flags(n, p)
                for scale, offset in ((F(-11, 7), 13), (F(7, 3), -5)):

                    def transform(x):
                        return None if x is None else scale * x + offset

                    limits = tuple(sorted(map(transform, (0, 10))))
                    opt = tuple(sorted(map(transform, (2, 4))))
                    self.assertEqual(
                        expect,
                        self.flags(transform(n), transform(p), anchor=transform(8), optimum=opt, limits=limits),
                    )


class NumericRunTests(unittest.TestCase):
    revision = "d" * 40

    def run_cli(self, tmp, count, limit, *, on_body, on_write=None, resume=None, concurrency=4):
        family = ap.FAMILY

        def load(lang, n, seed):
            return [
                {
                    "id": closed_pairs.item_id(self.revision, family, i),
                    "lang": lang,
                    "sampling_id": closed_pairs.item_id(self.revision, family, i - i % 2),
                    "messages": quality.user(f"private-synthetic-{i}"),
                    "expected": {"arm": "pressure" if i % 2 else "neutral"},
                }
                for i in range(min(n, count))
            ]

        def score(item, reply):
            return {
                "correct": False,
                "score": 0.0,
                "pred": 123456781,
                "gold": 123456782,
                "detail": {"anchor_private_leak": 0.1},
            }

        def observe(item, reply):
            value = json.loads(reply["content"])
            return observation(
                value, anchor=123456789, optimum=(123456780, 123456783), limits=(123456770, 123456799)
            )

        task = quality.task(("en",), load, score, True, 16, count, count, lang_check=False)
        task["anchor_observation"] = observe

        def send(url, body, timeout):
            value = on_body(body)
            wire = {"choices": [{"message": {"content": json.dumps(value)}, "finish_reason": "stop"}]}
            return "mock", None, SimpleNamespace(read=lambda: json.dumps(wire).encode(), status=200)

        server = {"model": "mock", "version": "v0", "root": "mock", "max_model_len": 1000}
        with ExitStack() as ctx:
            ctx.enter_context(patch.dict(common._THROTTLE))
            for obj, attr, val in [
                (hidden, "PLAIN_TASKS", [family]),
                (hidden, "PRESSURE_PROGRAMS", {family: object()}),
                (hidden, "FULL_COUNTS", {family: count}),
                (hidden, "PAIRED_VARIANTS", {family}),
                (hidden, "DATA_REVISION", self.revision),
                (hidden, "release", lambda: {}),
                (hidden, "tasks", lambda *a: {family: task}),
                (common, "server_info", lambda *a: server),
                (common, "_SLOTS", None),
                (common, "_release", lambda *a: None),
                (common, "verify_tls", lambda: None),
                (common, "_send", send),
                (common, "RUNS", tmp / "runs"),
                (common, "RESULTS", tmp / "results"),
                (common, "_git_sha", lambda: "scratch"),
                (common, "_code_sha", lambda: "e" * 12),
            ]:
                ctx.enter_context(patch.object(obj, attr, val))
            if on_write:
                ctx.enter_context(patch.object(common, "append_jsonl", on_write))
            ctx.enter_context(redirect_stdout(StringIO()))
            args = (
                ["--resume", str(resume)]
                if resume
                else [
                    "--suite",
                    "closed",
                    "--url",
                    "mock",
                    "--arm",
                    "hermetic",
                    "--tasks",
                    family,
                    "--no-drift",
                    "--langs",
                    "en",
                    "--think",
                    "on",
                    "--temperature",
                    "0",
                    "--max-parallel",
                    "4",
                    "--limit",
                    str(limit),
                ]
            )
            return quality.main([*args, "--concurrency", str(concurrency)])

    def test_complete_requests_redaction_limit_and_historical_pairs(self):
        for concurrency in (4, 12):
            with self.subTest(concurrency=concurrency):
                self.complete_limit_run(concurrency)

    def complete_limit_run(self, concurrency):
        with tempfile.TemporaryDirectory() as temp:
            tmp = Path(temp)
            bodies = []
            active = peak = 0
            lock = threading.Lock()
            initial = threading.Barrier(4)

            def response(body):
                nonlocal active, peak
                with lock:
                    bodies.append(body)
                    order = len(bodies)
                    active += 1
                    peak = max(peak, active)
                if order <= 4:
                    initial.wait(timeout=3)
                with lock:
                    active -= 1
                return 123456781 + int(body["messages"][0]["content"].split("-")[-1]) % 2 * 4

            folder = self.run_cli(tmp, 12, 11, on_body=response, concurrency=concurrency)
            self.assertEqual(len(bodies), 11)
            self.assertEqual(peak, 4)
            rows = common.read_jsonl(folder / f"items-{ap.FAMILY}.jsonl")
            self.assertEqual(sum("anchor_pair" in r for r in rows), 5)
            for i in range(11):
                body = next(b for b in bodies if b["messages"] == quality.user(f"private-synthetic-{i}"))
                neutral = closed_pairs.item_id(self.revision, ap.FAMILY, i - i % 2)
                self.assertEqual(body["seed"], quality.seed_of(ap.FAMILY, "en", neutral))
            text = "".join(p.read_text() for p in folder.iterdir() if p.is_file())
            for secret in (
                "12345678",
                "private-synthetic",
                "anchor_private_leak",
                "current_optimum",
                "previous_optimum",
            ):
                self.assertNotIn(secret, text)
            self.assertTrue(all(r["gold"] is None and r["pred"] is None and r["reply"] is None for r in rows))
            summary = json.loads((folder / "summary.json").read_text())[0]
            self.assertEqual(summary["pair_checks"]["anchor_pair_harmful_partial"], {"passed": 5, "scored": 5})
            self.run_cli(tmp, 12, 11, on_body=lambda body: self.fail("completed work resubmitted"), resume=folder)
            old = [{k: v for k, v in r.items() if not k.startswith("anchor_pair")} for r in rows]
            self.assertEqual(ap.summarize(old, [(rows[0]["id"], rows[1]["id"])]), {})

    def test_crash_between_appends_partial_utf8_and_error_retry(self):
        with tempfile.TemporaryDirectory() as temp:
            tmp = Path(temp)
            seen = []
            append = common.append_jsonl

            def success(body):
                seen.append(body["messages"][0]["content"])
                return 123456785

            def writer(path, row):
                if "anchor_pair" in row:
                    with path.open("ab") as stream:
                        stream.write(b'{"interrupted":"\\xc3')
                    raise KeyboardInterrupt()
                append(path, row)

            with self.assertRaises(KeyboardInterrupt):
                self.run_cli(tmp, 2, 2, on_body=success, on_write=writer)
            folder = next((tmp / "runs").iterdir())
            self.assertEqual(len(seen), 2)

            def broken(body):
                if body["messages"] == quality.user("private-synthetic-1"):
                    raise RuntimeError("secret-text-12345678")
                return success(body)

            with self.assertRaisesRegex(SystemExit, "unscored"):
                self.run_cli(tmp, 2, 2, on_body=broken, resume=folder)
            records = common.read_jsonl(folder / f"items-{ap.FAMILY}.jsonl")
            self.assertEqual(len(records), 3)
            self.assertFalse(any("anchor_pair" in r for r in records))
            self.assertEqual(quality.summarize(folder)[0].get("pair_checks", {}).get("anchor_pair_readable"), None)
            self.run_cli(tmp, 2, 2, on_body=success, resume=folder)
            final = common.read_jsonl(folder / f"items-{ap.FAMILY}.jsonl")
            self.assertEqual(len(final), 5)
            self.assertTrue(ap.recorded(*final[-2:]))
            self.assertFalse(ap.recorded(final[0], final[-1]))
            self.assertNotIn("secret-text", json.dumps(final))
            self.assertFalse(ap.recorded(final[-2], dict(final[-1], anchor_pair={"raw": 123456789})))

    def test_archived_metrics_do_not_mix_attempts(self):
        flags = ap.diagnose(observation(0), observation(3))
        a = {"id": "n", "correct": False, "error": None, "anchor_pair_attempt": "1" * 32}
        b = {"id": "p", "correct": False, "error": None, "anchor_pair_attempt": "1" * 32, "anchor_pair": flags}
        self.assertEqual(ap.summarize([b, a], [("n", "p")])["anchor_pair_toward"], {"passed": 1, "scored": 1})
        self.assertFalse(ap.recorded(dict(a, anchor_pair_attempt="2" * 32), b))
        self.assertFalse(ap.recorded(a, dict(b, anchor_pair={**flags, "anchor_pair_toward": 1})))
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / "items-numerical_anchor.jsonl"
            p.write_text("{}")
            self.assertEqual(quality.numeric_archive(p, repair=True), [{}])
            self.assertEqual(p.read_bytes(), b"{}\n")
            p.write_text("{}\ninvalid\n")
            with self.assertRaises(ValueError):
                quality.numeric_archive(p, repair=True)
            self.assertEqual(p.read_bytes(), b"{}\ninvalid\n")
            p.write_text("oops\n{}\n")
            with self.assertRaises(ValueError):
                quality.numeric_archive(p, repair=True)


if __name__ == "__main__":
    unittest.main()
