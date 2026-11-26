"""Regression controls for the isolated CUDA evaluator; real compilation is opt-in."""

import contextlib
import copy
import ctypes
import importlib.util
import json
import os
from pathlib import Path
import math
import struct
import subprocess
import unittest
from unittest.mock import patch

import kernel_cuda

spec = importlib.util.spec_from_file_location(
    "cuda_worker", Path(__file__).parent / "kernel-cuda" / "worker.py"
)
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)

SOURCE = """extern "C" __global__ void evaluate(const float *a,const float *b,float *out,
unsigned M,unsigned N,unsigned K,unsigned L) {
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<L) out[i]=a[i]+b[i];
}"""
PROGRAM = {"source": SOURCE, "threads": 32, "outputs_per_thread": 1}
CASE = {
    "shape": [1, 2, 1],
    "a": [1.0, 2.0],
    "b": [8.0, 9.0],
    "expected": [9.0, 11.0],
    "atol": 0.0,
    "rtol": 0.0,
    "timing": True,
}


class DeviceMemoryDouble:
    """Exercise byte guards, poisoning and launch ABI using ordinary buffers, without measuring hardware."""

    def __init__(self, mode):
        self.mode = mode
        self.events = (0, 1)
        self.stream = 0
        self.blocks = None

    @contextlib.contextmanager
    def kernel(self, blob):
        yield blob

    @contextlib.contextmanager
    def allocation(self, size):
        buffer = ctypes.create_string_buffer(size)
        yield ctypes.c_uint64(ctypes.addressof(buffer))

    def call(self, name, *args):
        if name == "cuMemcpyHtoD_v2":
            dst, src, n = args
            ctypes.memmove(dst.value, src, n)
        elif name == "cuMemcpyDtoH_v2":
            dst, src, n = args
            ctypes.memmove(dst, src.value, n)
        elif name == "cuLaunchKernel":
            self.blocks = args[1]
            argv = args[-2]
            a, b, out = [ctypes.c_uint64.from_address(argv[i]).value for i in range(3)]
            length = ctypes.c_uint.from_address(argv[6]).value
            if self.mode != "untouched":
                for i in range(length):
                    offset = i * 4
                    ctypes.c_float.from_address(out + offset).value = (
                        ctypes.c_float.from_address(a + offset).value
                        + ctypes.c_float.from_address(b + offset).value
                    )
            if self.mode == "write_input":
                ctypes.c_float.from_address(a).value = 42
            if self.mode == "write_guard":
                ctypes.c_float.from_address(out + 4 * length).value = 42
        elif name == "cuEventElapsedTime":
            ctypes.cast(args[0], ctypes.POINTER(ctypes.c_float))[0] = 1.0


class EvaluatorTest(unittest.TestCase):
    def test_input_metadata_and_opt_in(self):
        self.assertTrue(worker.program_ok(PROGRAM))
        for bad in (
            {"threads": True},
            {"outputs_per_thread": 3},
            {"host_source": ""},
            {"source": "k\0"},
        ):
            self.assertFalse(worker.program_ok({**PROGRAM, **bad}))
        with (
            patch.dict(os.environ, {"WEAKBENCH_CUDA_DEVICE": "all"}),
            patch.object(kernel_cuda, "_docker", side_effect=AssertionError),
        ):
            with self.assertRaisesRegex(kernel_cuda.BackendUnavailable, "dedicated"):
                kernel_cuda.measure(PROGRAM, PROGRAM, [CASE])

    def test_guards_and_unwritten_output(self):
        for mode in ("ok", "untouched", "write_input", "write_guard"):
            with self.subTest(mode=mode):
                dev = DeviceMemoryDouble(mode)
                valid, _ = worker.Trial(dev, CASE).run(0, PROGRAM)
                self.assertEqual(valid["passed"], mode == "ok")
                self.assertEqual(
                    valid["buffers_intact"], mode not in ("write_input", "write_guard")
                )
                self.assertEqual(valid["mismatches"], 2 if mode == "untouched" else 0)
                self.assertEqual(dev.blocks, 1)
        malformed = b"\x7fELF" + bytes(128)
        self.assertFalse(worker.has_entry(malformed))
        self.assertTrue(math.isnan(struct.unpack("<f", worker.POISON)[0]))

    def test_all_shapes_gate_before_reporting_timing(self):
        cases = [{**CASE, "timing": False}, {**CASE, "timing": True}]

        class Device:
            hardware = {"device": "test-only"}

        calls = []
        failure = []

        class MockTrial:
            def __init__(self, driver, case):
                self.case = case

            def run(self, function, program):
                calls.append((function, self.case["timing"]))
                return {
                    "passed": len(calls) not in failure
                }, 2.0 if function == 0 else 1.0

        def run():
            return worker.run_trials(Device(), [PROGRAM, PROGRAM], [0, 1], cases, 42)

        with patch.object(worker, "Trial", MockTrial):
            good = run()
            self.assertEqual(calls[:4], [(0, False), (1, False), (0, True), (1, True)])
            self.assertEqual(len(good["cases"][1]["rounds"]), 7)
            self.assertNotIn("timing", good["cases"][0])
            self.assertEqual(good["cases"][1]["timing"]["speedup"], 2)
            calls.clear()
            failure[:] = [4]
            bad = run()
            self.assertEqual(bad["status"], "incorrect")
            self.assertFalse(any("timing" in case for case in bad["cases"]))
            calls.clear()
            failure[:] = [1]
            with self.assertRaisesRegex(worker.InfraError, "baseline"):
                run()


class HostReportTest(unittest.TestCase):
    device = "GPU-01234567-89ab-cdef-1234-56789abcdef0"
    header = {
        "backend": "cuda",
        "protocol": kernel_cuda.PROTOCOL,
        "target": "sm_120",
        "compiler": "NVRTC 13.0.88",
    }
    inventory = [{**CASE, "shape": [2, 1, 1], "timing": False}, CASE]

    def evaluate(self, report, *, compile_only=False, timeout=False):
        image = "sha256:" + "e" * 64

        def docker(args, **options):
            if "run" in args:
                if timeout:
                    raise subprocess.TimeoutExpired(args, options["timeout"])
                options["stdout"].write(json.dumps(report).encode())
            return subprocess.CompletedProcess(args, 0)

        with (
            patch.object(kernel_cuda, "_docker", return_value=["docker"]),
            patch.object(kernel_cuda, "_image", return_value=image),
            patch.object(
                kernel_cuda, "_device_slot", return_value=contextlib.nullcontext()
            ),
            patch.dict(os.environ, {"WEAKBENCH_CUDA_DEVICE": self.device}),
            patch.object(kernel_cuda.subprocess, "run", side_effect=docker) as cli,
        ):
            try:
                if compile_only:
                    return kernel_cuda.compile_native(PROGRAM)
                return kernel_cuda.measure(PROGRAM, PROGRAM, self.inventory, seed=51)
            finally:
                self.assertEqual(cli.call_args.args[0][1:3], ["rm", "--force"])

    def hardware_result(self):
        valid = {
            "passed": True,
            "mismatches": 0,
            "buffers_intact": True,
            "max_finite_abs_error": 0.0,
        }
        rows = [
            {"shape": case["shape"][:], "elements": 2, "validation": dict(valid)}
            for case in self.inventory
        ]
        rounds = [
            {
                "baseline_first": side == 0,
                "baseline_us": i + 3.0,
                "candidate_us": i + 1.0,
            }
            for i, side in enumerate((1, 0, 0, 1, 1, 0, 0))  # worker order at seed 51
        ]
        rows[1].update(
            rounds=rounds, timing=worker.timing(rounds), dispatches_per_sample=1
        )
        return {
            **self.header,
            "status": "correct",
            "compiled": True,
            "device": "test fixture",
            "device_uuid": self.device[4:].replace("-", ""),
            "compute_capability": "12.0",
            "driver_version": 13000,
            "candidate_hash": "a" * 64,
            "baseline_hash": "b" * 64,
            "timing_source": kernel_cuda.TIMING_SOURCE,
            "seed": 51,
            "cases": rows,
        }

    def test_compile_status_and_native_artifact(self):
        good = {
            **self.header,
            "status": "compiled",
            "compiled": True,
            "cubin_bytes": 5792,
            "cubin_sha256": "a" * 64,
        }
        self.assertEqual(self.evaluate(good, compile_only=True)["status"], "compiled")
        for issue in (
            {"compiled": 1},
            {"cubin_bytes": 1},
            {"cubin_bytes": True},
            {"cubin_sha256": "missing"},
            {"target": "compute_120"},
            {"compiler": "unknown"},
            {"status": "correct"},
            {"cases": []},
        ):
            with (
                self.subTest(issue=issue),
                self.assertRaises(kernel_cuda.BackendUnavailable),
            ):
                self.evaluate({**good, **issue}, compile_only=True)
        rejected = {
            **self.header,
            "status": "compile_error",
            "compiled": False,
            "reason": "entry_missing",
            "cases": [],
        }
        self.assertFalse(self.evaluate(rejected, compile_only=True)["compiled"])
        self.assertFalse(self.evaluate(rejected)["compiled"])
        with self.assertRaises(kernel_cuda.BackendUnavailable):
            self.evaluate({**rejected, "reason": "unknown"}, compile_only=True)
        with self.assertRaises(kernel_cuda.BackendUnavailable):
            self.evaluate(good)

    def test_hardware_requires_complete_correct_results(self):
        good = self.hardware_result()
        self.assertEqual(self.evaluate(good)["status"], "correct")
        # Paths are used to corrupt exactly one field of a coherent worker report.
        changes = [
            (("cases",), []),
            (("cases",), good["cases"][::-1]),
            (("status",), "incorrect"),
            (("device_uuid",), None),
            (("candidate_hash",), "a" * 63),
            (("baseline_hash",), False),
            (("timing_source",), "CPU"),
            (("seed",), 99),
            (("cases", 0, "elements"), 3),
            (("cases", 0, "shape"), [2, True, 1]),
            (("cases", 0, "timing"), {}),
            (("cases", 1, "validation", "passed"), False),
            (("cases", 1, "validation", "mismatches"), 1),
            (("cases", 1, "validation", "buffers_intact"), False),
            (("cases", 1, "validation", "max_finite_abs_error"), float("nan")),
            (("cases", 1, "validation", "max_finite_abs_error"), 1.0),
            (("cases", 1, "dispatches_per_sample"), True),
            (("cases", 1, "rounds"), good["cases"][1]["rounds"][:-1]),
            (("cases", 1, "rounds"), [good["cases"][1]["rounds"][0]] * 7),
            (("cases", 1, "rounds"), good["cases"][1]["rounds"][::-1]),
            (("cases", 1, "rounds", 0, "baseline_us"), 0),
            (("cases", 1, "rounds", 0, "candidate_us"), float("inf")),
            (("cases", 1, "rounds", 0, "candidate_us"), 10**400),
            (("cases", 1, "rounds", 0, "candidate_us"), True),
            (("cases", 1, "timing", "candidate_us"), 100.0),
            (("cases", 1, "timing", "ratio_min"), 1.0),
            (("cases", 1, "timing", "speedup"), 2.0),
            (("cases", 1, "timing", "comparison"), "slower"),
        ]
        for path, value in changes:
            with (
                self.subTest(path=path),
                self.assertRaises(kernel_cuda.BackendUnavailable),
            ):
                report = copy.deepcopy(good)
                obj = report
                for key in path[:-1]:
                    obj = obj[key]
                obj[path[-1]] = value
                self.evaluate(report)
        del good["cases"][1]["timing"]
        with self.assertRaises(kernel_cuda.BackendUnavailable):
            self.evaluate(good)

    def test_failed_verification_suppresses_all_timing(self):
        bad = self.hardware_result()
        bad["status"] = "incorrect"
        bad["cases"][0]["validation"].update(passed=False, buffers_intact=False)
        with self.assertRaises(kernel_cuda.BackendUnavailable):
            self.evaluate(bad)
        for field in ("rounds", "timing", "dispatches_per_sample"):
            del bad["cases"][1][field]
        self.assertEqual(self.evaluate(bad)["status"], "incorrect")
        fault = {
            k: v
            for k, v in bad.items()
            if k not in ("seed", "timing_source", "candidate_hash", "baseline_hash")
        }
        fault.update(cases=[], reason="device_execution_error")
        self.assertEqual(self.evaluate(fault)["status"], "incorrect")
        fault["status"] = "correct"
        with self.assertRaises(kernel_cuda.BackendUnavailable):
            self.evaluate(fault)

    def test_worker_timeout_and_missing_opt_in_remain_distinct(self):
        with self.assertRaises(kernel_cuda.BackendTimeout) as timer:
            self.evaluate(None, timeout=True)
        self.assertEqual(timer.exception.reason_code, "backend_timeout")
        with (
            patch.dict(os.environ, {"WEAKBENCH_CUDA_DEVICE": ""}),
            self.assertRaises(kernel_cuda.BackendUnavailable) as opt_in,
        ):
            kernel_cuda.measure(PROGRAM, PROGRAM, self.inventory)
        self.assertEqual(opt_in.exception.reason_code, "dedicated_device_unselected")


@unittest.skipUnless(
    os.environ.get("WEAKBENCH_CUDA_COMPILE_E2E") == "1",
    "requires local NVRTC OCI image",
)
class CompileControl(unittest.TestCase):
    def test_pinned_native_compiler(self):
        result = kernel_cuda.compile_native(PROGRAM)
        self.assertEqual(result["status"], "compiled")
        self.assertGreater(result["cubin_bytes"], 0)
        for source, reason in [
            (SOURCE.replace("evaluate", "different"), "entry_missing"),
            (SOURCE.replace("unsigned L", "float L"), "compile_error"),
            (SOURCE.replace("a[i]+b[i]", "syntax error"), "compile_error"),
        ]:
            with self.subTest(reason=reason):
                bad = kernel_cuda.compile_native({**PROGRAM, "source": source})
                self.assertEqual(bad["reason"], reason)
                self.assertFalse(bad["compiled"])


@unittest.skipUnless(
    os.environ.get("WEAKBENCH_CUDA_HARDWARE_E2E") == "1",
    "requires dedicated opted-in SM120 GPU",
)
class HardwareControl(unittest.TestCase):
    def test_device_gate(self):
        good = kernel_cuda.measure(PROGRAM, PROGRAM, [CASE], seed=7)
        self.assertEqual(good["status"], "correct")
        self.assertEqual(len(good["cases"][0]["rounds"]), 7)
        bad = kernel_cuda.measure(
            {**PROGRAM, "source": SOURCE.replace("a[i]+b[i]", "0.f")}, PROGRAM, [CASE]
        )
        self.assertEqual(bad["status"], "incorrect")
        self.assertNotIn("timing", bad["cases"][0])


if __name__ == "__main__":
    unittest.main()
