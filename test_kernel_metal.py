"""Hermetic kernel language checks and an opt-in Apple GPU evaluation."""

from copy import deepcopy
import os
import random
import struct
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import kernel_metal as metal


def program(source, threads=64, lanes=1):
    return {"source": source, "threads": threads, "outputs_per_thread": lanes}


def fixture(size, *, timing=False):
    return {
        "shape": [1, size, 1],
        "a": [float(i) for i in range(size)],
        "b": [-0.5] * size,
        "expected": [float(i) - 0.5 for i in range(size)],
        "atol": 1e-5,
        "rtol": 1e-5,
        "timing": timing,
    }


ADD = program("out = A(i) + B(i)")


class CompilerTests(unittest.TestCase):
    def test_checked_reads_and_slot_mapping(self):
        source = metal.lower(program("out = A(i) + B(i)", 128, 4), [(1, 12, 1, 12)])
        self.assertIn("rd(a,s.na,i,bad)", source)
        self.assertIn("gid*4u+lane", source)
        self.assertIn("output[i]=v_out", source)
        self.assertIn("if (i>=L) break", source)
        self.assertIn("constant Shape &s", source)

    def test_bounded_reduction_and_typed_control_flow(self):
        src = """acc = -2.0
for col in range(0, N, 2):
    if col < B(i) and (i % N) != col:
        acc += fma(A(i*K + col), 2.0, -1)
out = acc if acc > 0.0 else 0.0"""
        lowered = metal.lower(program(src), [(2, 4, 4, 2), (8, 100, 100, 8)])
        self.assertIn("v_col+=2u", lowered)
        self.assertIn("fma(", lowered)
        self.assertIn("-1.0f", lowered)
        self.assertIn("float(v_col)", lowered)

    def test_invalid_source_and_uninitialized_registers(self):
        for src in (
            "out = B.__class__",
            "import math\nout = math.sqrt(i)",
            "while i:\n    out=1",
            "for j in range(A(i)):\n    out = 1",
            "for j in range(0,N,0):\n    out = A(j)",
            "if i:\n    tmp = 1.0\nout = tmp",
            "for j in range(K,K):\n    tmp = A(j)\nout = tmp",
            "M = 3\nout = M",
            "for j in range(K):\n    j=0\nout=0.0",
            "a[0] = 1\nout=0.0",
            "out = (lambda: 1)()",
            "out = A(1.5)",
            "if i:\n    out=A(i)",
            "out = 1e999",
            "out = 0\nfor j in range(K):\n    for yy in range(K):\n        out += A(i)",
        ):
            with self.subTest(src=src), self.assertRaises(metal.KernelRejected):
                metal.lower(program(src), [(9, 9, 2500, 81)])

    def test_aliases_nonempty_loops_and_unicode_comments(self):
        two_shapes = [(2, 2, 1, 4), (1, 3, 2, 3)]
        body = "stop = K\nsize = stop\nfor j in range(size):\n    val = A(j)\nout = val"
        plain = metal.lower(program(body), two_shapes)
        self.assertEqual(
            plain,
            metal.lower(
                program("# Größe prüfen\n" + body + " # fin réglée"), two_shapes
            ),
        )
        self.assertIn("v_j<v_size", plain)
        self.assertIn("float(v_val)", plain)
        for source in (
            "z = K\nz += 1\nfor j in range(z):\n    out = 0",
            "if i:\n    z = K\nfor j in range(z):\n    out = 0",
            "for j in range(1,K):\n    out = 2",
            "for j in range(K):\n    if i:\n        value = 2\nout = value",
            "for j in range(K):\n    val = 1\nout = j",
            "größenwert = 1\nout = größenwert",
            "# comment\rｏｕｔ = 1",
        ):
            with self.subTest(source=source), self.assertRaises(metal.KernelRejected):
                metal.lower(program(source), two_shapes)
        with self.assertRaisesRegex(metal.KernelRejected, "static operation bound"):
            metal.lower(
                program(
                    "z = K\nfor x in range(z):\n    for y in range(z):\n        out = A(y)"
                ),
                [(32, 32, 2500, 1024)],
            )

    def test_host_report_requires_complete_consistent_evidence(self):
        shapes = [(1, 3, 1, 3)]
        cases = [fixture(3, timing=True)]
        order = [[0, 1, 0, 1, 0, 1, 0]]
        row = dict(
            shape=[1, 3, 1],
            elements=3,
            dispatches_per_sample=1,
            validation=dict(
                passed=True,
                invalid_operation=False,
                mismatches=0,
                nonfinite=0,
                max_abs_error=0,
            ),
            rounds=[
                dict(baseline_first=first == 0, baseline_us=4, candidate_us=3)
                for first in order[0]
            ],
        )
        result = dict(
            status="correct",
            backend="metal",
            device="Mock",
            simd_width=32,
            registry_id=1,
            cases=[row],
        )
        metal._validate_host_result(result, shapes, cases, order)
        for field, value in (
            (("cases",), []),
            (("cases", 0, "shape"), [1, 2, 1]),
            (("cases", 0, "validation", "passed"), False),
            (("cases", 0, "validation", "mismatches"), 1),
            (("cases", 0, "rounds"), []),
            (("cases", 0, "rounds", 0, "baseline_first"), False),
            (("cases", 0, "rounds", 0, "candidate_us"), 0),
            (("cases", 0, "rounds", 0, "baseline_us"), float("nan")),
            (("status",), "incorrect"),
        ):
            bad = deepcopy(result)
            obj = bad
            for key in field[:-1]:
                obj = obj[key]
            obj[field[-1]] = value
            with self.subTest(field=field), self.assertRaises(metal.BackendUnavailable):
                metal._validate_host_result(bad, shapes, cases, order)

    def test_canonical_profiles_and_case_limits(self):
        for wrong in (
            program("out=1", 12, 1),
            program("out=1", 64, 3),
            program("out=1", True, 1),
            {**ADD, "compiler_flags": "-ffast-math"},
        ):
            with self.assertRaises(metal.KernelRejected):
                metal.lower(wrong, [(1, 4, 1, 4)])
        for change in (
            {"shape": [1, -1, 2]},
            {"expected": []},
            {"rtol": float("nan")},
            {"timing": 3},
        ):
            with self.assertRaises(metal.BackendUnavailable):
                metal._case_shapes([{**fixture(8), **change}])
        with (
            tempfile.TemporaryDirectory() as tmp,
            self.assertRaises(metal.BackendUnavailable),
        ):
            metal._buffers(Path(tmp), [{**fixture(8), "a": [float("inf")]}])

    def test_case_hash_covers_full_precision_cpu_references(self):
        sample = {**fixture(1), "expected": [1.00000004]}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            before = metal._buffers(path, [sample])[0]["case_hash"]
            self.assertEqual(
                (path / "0-expected").read_bytes(), struct.pack("<d", 1.00000004)
            )
            self.assertEqual((path / "0-a").read_bytes(), struct.pack("<f", 0.0))
            self.assertEqual((path / "0-b").read_bytes(), struct.pack("<f", -0.5))
            after = metal._buffers(path, [{**sample, "expected": [1.0]}])[0][
                "case_hash"
            ]
            self.assertNotEqual(before, after)

    def test_unavailable_device_has_no_emulated_rank(self):
        with (
            patch.object(metal.platform, "system", return_value="Linux"),
            self.assertRaises(metal.BackendUnavailable),
        ):
            metal.measure(ADD, ADD, [fixture(3)])

    def test_source_rejection_never_compiles_candidate_host_code(self):
        with (
            patch.object(metal.platform, "system", return_value="Darwin"),
            patch.object(metal.platform, "machine", return_value="arm64"),
            patch.object(metal, "_host", side_effect=AssertionError("host launched")),
        ):
            rejected = metal.measure(program("out=unknown(5)"), ADD, [fixture(5)])
            self.assertEqual(rejected["status"], "invalid_candidate")
            self.assertEqual(rejected["cases"], [])
            with self.assertRaises(metal.BackendUnavailable):
                metal.measure(ADD, program("out=unknown(5)"), [fixture(5)])


@unittest.skipUnless(
    os.getenv("WEAKBENCH_METAL_E2E") == "1",
    "enable the real Apple Metal test explicitly",
)
class HardwareTests(unittest.TestCase):
    def test_comparison_repeats_after_every_shape_passes(self):
        sample = [fixture(11), fixture(8193, timing=True)]
        result = metal.measure(
            program("out=fma(A(i),1.0,B(i))", 32, 2), ADD, sample, seed=19
        )
        self.assertEqual(result["status"], "correct")
        self.assertGreater(result["registry_id"], 0)
        self.assertTrue(all(r["validation"]["passed"] is True for r in result["cases"]))
        small, timed = result["cases"]
        self.assertNotIn("timing", small)
        self.assertEqual(len(timed["rounds"]), 7)
        self.assertGreater(timed["timing"]["ratio_min"], 0)
        self.assertGreaterEqual(
            timed["timing"]["ratio_max"], timed["timing"]["ratio_min"]
        )
        self.assertIn(sum(x["baseline_first"] for x in timed["rounds"]), (3, 4))

    def test_one_bad_shape_and_out_of_bounds_suppress_all_performance(self):
        sample = [fixture(4096, timing=True), fixture(7)]
        for source in (
            "out = A(i) + B(i) if N > 7 else 3.0",
            "out = A(i) + B(i) + B(L)",
            "out=A(i)+B(i)+float(i//(i-i))",
            "out=0.0/0.0",
        ):
            with self.subTest(source=source):
                result = metal.measure(program(source), ADD, sample)
                self.assertEqual(result["status"], "incorrect")
                self.assertTrue(
                    all(
                        "timing" not in r and "rounds" not in r for r in result["cases"]
                    )
                )

    def test_catalog_reference_preserves_both_sides_of_tolerance(self):
        for expected, actual, should_pass in (
            (1.00000004, 1.0000640153884888, True),
            (0.501000007, 0.5009379982948303, False),
        ):
            with self.subTest(expected=expected):
                # The candidate represents the GPU's exact float32 output. Quantizing
                # the CPU reference at either boundary would reverse the verdict.
                self.assertEqual(
                    struct.unpack("<f", struct.pack("<f", actual))[0], actual
                )
                rounded = struct.unpack("<f", struct.pack("<f", expected))[0]
                for witness, verdict in (
                    (expected, should_pass),
                    (rounded, not should_pass),
                ):
                    self.assertEqual(
                        abs(actual - witness) <= 6e-5 + 4e-6 * abs(witness), verdict
                    )
                sample = {
                    **fixture(1),
                    "a": [actual],
                    "b": [expected],
                    "expected": [expected],
                    "atol": 6e-5,
                    "rtol": 4e-6,
                }
                result = metal.measure(
                    program("out=A(i)"), program("out=B(i)"), [sample]
                )
                self.assertEqual(
                    result["status"], "correct" if should_pass else "incorrect"
                )
                check = result["cases"][0]["validation"]
                self.assertEqual(check["passed"], should_pass)
                self.assertAlmostEqual(
                    check["max_abs_error"], abs(actual - expected), places=13
                )

    def test_wrong_baseline_is_infrastructure_error(self):
        with self.assertRaisesRegex(metal.BackendUnavailable, "baseline_disagrees"):
            metal.measure(ADD, program("out=9.0"), [fixture(7)])


FACTORED = """left = 0.0
for j in range(N):
    left += A(j*M+i)
right = 0.0
for k in range(K):
    right += B(k*M+i)
out = left*right"""
EXPANDED = """total = 0.0
for j in range(N):
    left = A(j*M+i)
    for k in range(K):
        total += left*B(k*M+i)
out = total"""


def speed_control_case(m, n, k):
    rng = random.Random(717433 + m)
    a = [rng.randrange(-5, 6) for _ in range(m * n)]
    b = [rng.randrange(-5, 6) for _ in range(m * k)]
    # Every partial sum is an exact binary32 integer for these dimensions.
    assert 25 * n * k < 2**24
    expected = []
    for i in range(m):
        row_a, row_b = a[i::m], b[i::m]
        explicit = sum(x * y for x in row_a for y in row_b)
        assert explicit == sum(row_a) * sum(row_b)
        expected.append(explicit)
    return dict(
        shape=[m, n, k], a=a, b=b, expected=expected, atol=0.0, rtol=0.0, timing=True
    )


@unittest.skipUnless(
    os.environ.get("WEAKBENCH_METAL_SPEED_E2E") == "1", "requires an idle Apple GPU"
)
class SpeedControlTests(unittest.TestCase):
    def test_equivalent_kernels_in_both_directions(self):
        cases = [
            speed_control_case(*shape) for shape in [(2049, 29, 31), (3073, 43, 47)]
        ]
        fast, slow = program(FACTORED), program(EXPANDED)
        for candidate, baseline, expected in (
            (fast, slow, "faster"),
            (slow, fast, "slower"),
        ):
            with self.subTest(expected=expected):
                result = metal.measure(candidate, baseline, cases, seed=717433)
                self.assertEqual(result["status"], "correct")
                for row in result["cases"]:
                    self.assertTrue(row["validation"]["passed"])
                    self.assertEqual(len(row["rounds"]), 7)
                    self.assertEqual(row["timing"]["comparison"], expected)


if __name__ == "__main__":
    unittest.main()
