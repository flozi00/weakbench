"""Hermetic native Metal report and source validation, with opt-in hardware controls."""

from copy import deepcopy
import os
import struct
import tempfile
from pathlib import Path
import unittest

import kernel_msl as msl


HEAD = """#include <metal_stdlib>
using namespace metal;
kernel void evaluate(const device float *a [[buffer(0)]], const device float *b [[buffer(1)]],
                     device float *out [[buffer(2)]], constant uint4 &dims [[buffer(3)]],
                     uint i [[thread_position_in_grid]]) {
"""
SOURCE = HEAD + "if (i < dims.w) out[i] = a[i] + b[i]; }"


def cases():
    return [
        {
            "shape": [1, size, 1],
            "a": list(range(size)),
            "b": [3] * size,
            "expected": [3 + i for i in range(size)],
            "atol": 0,
            "rtol": 0,
            "timing": size > 1,
        }
        for size in (1, 79)
    ]


class NativeSource(unittest.TestCase):
    def test_source_contract(self):
        for source in (
            SOURCE,
            "// Is this all?? Or even ??= ?\n" + SOURCE,
            "#pragma unroll 4\n" + SOURCE,
            "#pragma clang loop unroll(enable)\n" + SOURCE,
            "/* erläutert */ " + SOURCE,
            "#define F(x) ((x)+1)\n" + SOURCE,
            'static_assert(true,"/* example");\n' + SOURCE,
        ):
            self.assertTrue(msl.acceptable_source(source))
        for source in (
            "",
            "\0",
            "ü" * msl.MAX_SOURCE,
            '# include "local.h"',
            '#/**/include "local.h"',
            '#if __has_include("local.h")\n#endif',
            "#import <Foundation/Foundation.h>",
            "??=include <header>",
            "%:include <header>",
            "#pragma once",
        ):
            self.assertFalse(msl.acceptable_source(source))

    def test_witness_precision_and_digest(self):
        c = cases()[0]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            row1 = msl._buffers(path, [{**c, "expected": [1.0]}])[0]
            row2 = msl._buffers(path, [{**c, "expected": [1.00000004]}])[0]
            self.assertNotEqual(row1["case_hash"], row2["case_hash"])
            self.assertEqual(
                (path / "0-expected").read_bytes(), struct.pack("<d", 1.00000004)
            )
            self.assertEqual((path / "0-a").stat().st_size, 4)
        self.assertTrue(msl.acceptable_source("//" + "ü" * 31999))
        self.assertFalse(msl.acceptable_source("//" + "ü" * 32000))

    def test_invalid_native_program(self):
        self.assertFalse(msl.compile_native("")["compiled"])
        with self.assertRaises(msl.BackendUnavailable):
            msl.measure(SOURCE, "", cases())

    def test_native_report_contract(self):
        base = {
            "compiled": True,
            "backend": "metal-native",
            "registry_id": 1,
            "device": "Test GPU",
            "simd_width": 32,
            "status": "compiled",
        }
        msl._check_report(base, False, [], [], [])
        case = cases()[0]
        check = {
            "passed": True,
            "invalid_operation": False,
            "buffers_intact": True,
            "mismatches": 0,
            "nonfinite": 0,
            "max_abs_error": 0,
        }
        row = {"shape": case["shape"], "elements": 1, "validation": check}
        data = {**base, "status": "correct", "cases": [row]}
        msl._check_report(data, True, [(1, 1, 1, 1)], [case], [[]])
        for change in ("count", "status", "guards", "device", "false-success"):
            bad = deepcopy(data)
            if change == "count":
                bad["cases"][0]["elements"] = 2
            elif change == "status":
                bad["status"] = "incorrect"
            elif change == "guards":
                bad["cases"][0]["validation"]["buffers_intact"] = False
            elif change == "device":
                bad["registry_id"] = 0
            else:
                bad["cases"] = []
            with self.assertRaises(msl.BackendUnavailable):
                msl._check_report(bad, True, [(1, 1, 1, 1)], [case], [[]])


@unittest.skipUnless(
    os.environ.get("WEAKBENCH_MSL_E2E") == "1", "requires local Apple GPU"
)
class MetalHardware(unittest.TestCase):
    def test_compiler_and_reflection(self):
        compiled = msl.compile_native(SOURCE)
        self.assertEqual(compiled["status"], "compiled")
        self.assertGreater(compiled["registry_id"], 0)
        with self.subTest("unused input buffer"):
            self.assertTrue(
                msl.compile_native(HEAD + "if(i < dims.w) out[i] = a[i]; }")["compiled"]
            )
        for variant, reason in (
            (SOURCE.replace("evaluate(", "other("), "entry_missing"),
            (SOURCE.replace("a[i] + b[i]", "missing("), "compile_error"),
            (SOURCE.replace("uint4 &dims", "float4 &dims"), "source_contract"),
            (
                SOURCE.replace("const device float *a", "device float *a"),
                "source_contract",
            ),
            (SOURCE.replace("buffer(1)", "buffer(4)"), "source_contract"),
        ):
            with self.subTest(reason=reason):
                result = msl.compile_native(variant)
                self.assertFalse(result["compiled"])
                self.assertEqual(result["reason"], reason)

    def test_correct_device_run(self):
        result = msl.measure(SOURCE, SOURCE, cases(), seed=3)
        self.assertTrue(result["compiled"])
        self.assertEqual(result["status"], "correct")
        self.assertNotIn("timing", result["cases"][0])
        row = result["cases"][1]
        self.assertEqual(len(row["rounds"]), 7)
        self.assertTrue(row["validation"]["buffers_intact"])
        self.assertGreater(row["timing"]["speedup"], 0)

    def test_numerical_guard_and_tail_controls(self):
        # Each wrong shader stays within the allocated memory of the control fixture.
        for body, guard in (
            ("if(i < dims.w) out[i]=a[i]-b[i];", True),
            ("if(i > 0 && i < dims.w) out[i]=a[i]+b[i];", True),
            ("if(i<=dims.w) out[i]=0;", False),
        ):
            with self.subTest(body=body):
                result = msl.measure(HEAD + body + "}", SOURCE, cases())
                self.assertTrue(result["compiled"])
                self.assertEqual(result["status"], "incorrect")
                self.assertTrue(all("timing" not in row for row in result["cases"]))
                self.assertEqual(
                    result["cases"][0]["validation"]["buffers_intact"], guard
                )

    def test_float32_output_against_full_precision_witness(self):
        expected = 1.00000004
        got = 1.0000640153884888  # exactly representable in a float32 device buffer
        case = {
            "shape": [1, 1, 1],
            "a": [got],
            "b": [0],
            "expected": [expected],
            "timing": False,
            "atol": 6e-5,
            "rtol": 4e-6,
        }
        self.assertLess(abs(got - expected), case["atol"] + case["rtol"] * expected)
        self.assertGreater(abs(got - 1), case["atol"] + case["rtol"])
        result = msl.measure(SOURCE, SOURCE, [case])
        self.assertTrue(result["cases"][0]["validation"]["passed"])
        self.assertEqual(result["cases"][0]["elements"], 1)

    def test_bad_reference_is_not_a_model_verdict(self):
        with self.assertRaises(msl.BackendUnavailable):
            msl.measure(SOURCE, HEAD + "if(i < dims.w) out[i]=0; }", cases())


FACTORED = (
    HEAD
    + """
    if (i >= dims.w) return;
    float left = 0.0f;
    for (uint j = 0; j < dims.y; ++j) left += a[j * dims.x + i];
    float right = 0.0f;
    for (uint k = 0; k < dims.z; ++k) right += b[k * dims.x + i];
    out[i] = left * right;
}
"""
)

EXPANDED = (
    HEAD
    + """
    if (i >= dims.w) return;
    float total = 0.0f;
    for (uint j = 0; j < dims.y; ++j) {
        const float left = a[j * dims.x + i];
        for (uint k = 0; k < dims.z; ++k) total += left * b[k * dims.x + i];
    }
    out[i] = total;
}
"""
)


@unittest.skipUnless(
    os.environ.get("WEAKBENCH_MSL_SPEED_E2E") == "1", "requires an idle Apple GPU"
)
class NativeSpeedControl(unittest.TestCase):
    def test_equivalent_kernels_in_both_directions(self):
        from test_kernel_metal import speed_control_case

        cases = [
            speed_control_case(*shape) for shape in ((2049, 29, 31), (3073, 43, 47))
        ]
        for candidate, baseline, expected in (
            (FACTORED, EXPANDED, "faster"),
            (EXPANDED, FACTORED, "slower"),
        ):
            with self.subTest(expected=expected):
                report = msl.measure(candidate, baseline, cases, seed=717433)
                self.assertEqual(report["status"], "correct")
                for row in report["cases"]:
                    self.assertTrue(row["validation"]["passed"])
                    self.assertTrue(row["validation"]["buffers_intact"])
                    self.assertEqual(len(row["rounds"]), 7)
                    self.assertEqual(row["timing"]["comparison"], expected)
                    for pair in row["rounds"]:
                        if expected == "faster":
                            self.assertLess(pair["candidate_us"], pair["baseline_us"])
                        else:
                            self.assertGreater(
                                pair["candidate_us"], pair["baseline_us"]
                            )


if __name__ == "__main__":
    unittest.main()
