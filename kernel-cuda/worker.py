"""Isolated NVRTC SM120 compiler and dedicated CUDA Driver evaluator."""

import array
import ctypes as C
from contextlib import contextmanager, ExitStack
import hashlib
import json
import math
import random
import statistics
import struct
import sys

PTR = C.c_void_p
INT = C.c_int
UINT = C.c_uint
SIZE = C.c_size_t
DEVICE_PTR = C.c_uint64
TEXT = C.c_char_p

TARGET = "sm_120"
COMPILER = "NVRTC 13.0.88"
PROTOTYPE = b'extern "C" __global__ void evaluate(const float*,const float*,float*,unsigned,unsigned,unsigned,unsigned);\n'
MAX_SOURCE = 96 * 1024
MAX_ITEMS = 1 << 20
MAX_PAYLOAD = 120 << 20
ROUNDS = 7


class InfraError(Exception):
    """Only the fixed code is exported."""


class CandidateError(Exception):
    """Only the fixed code is exported."""


def bind(lib, name, *types):
    try:
        fn = getattr(lib, name)
    except AttributeError:
        raise InfraError("library_entry_unavailable") from None
    fn.argtypes = types
    fn.restype = INT
    return fn


def has_entry(blob):
    """Check the CUDA ELF header and executable, nonempty text section for the exact entry."""
    if (
        len(blob) < 64
        or blob[:6] != b"\x7fELF\x02\x01"
        or struct.unpack_from("<H", blob, 18)[0] != 190
    ):
        return False
    off = struct.unpack_from("<Q", blob, 40)[0]
    stride, count, names_idx = struct.unpack_from("<HHH", blob, 58)
    if (
        stride != 64
        or count < 2
        or names_idx >= count
        or off + stride * count > len(blob)
    ):
        return False
    sections = [
        struct.unpack_from("<IIQQQQIIQQ", blob, off + i * stride) for i in range(count)
    ]
    names = sections[names_idx]
    strings = blob[names[4] : names[4] + names[5]]
    for sec in sections:
        end = strings.find(b"\0", sec[0])
        if (
            strings[sec[0] : end] == b".text.evaluate"
            and sec[1] == 1
            and sec[2] & 4
            and sec[5] > 0
            and sec[4] + sec[5] <= len(blob)
        ):
            return True
    return False


def program_ok(p):
    return (
        isinstance(p, dict)
        and set(p) == {"source", "threads", "outputs_per_thread"}
        and isinstance(p["source"], str)
        and 0 < len(p["source"].encode()) <= MAX_SOURCE
        and "\0" not in p["source"]
        and type(p["threads"]) is int
        and p["threads"] in (32, 64, 128, 256)
        and type(p["outputs_per_thread"]) is int
        and p["outputs_per_thread"] in (1, 2, 4, 8)
    )


def cubin(program):
    if not program_ok(program):
        raise CandidateError("invalid_program")
    try:
        lib = C.CDLL("/opt/nvrtc/libnvrtc.so.13")
    except OSError:
        raise InfraError("nvrtc_unavailable") from None
    prog = PTR()
    create = bind(
        lib,
        "nvrtcCreateProgram",
        C.POINTER(PTR),
        TEXT,
        TEXT,
        INT,
        C.POINTER(TEXT),
        C.POINTER(TEXT),
    )
    source = PROTOTYPE + program["source"].encode()
    if create(C.byref(prog), source, b"submission.cu", 0, None, None):
        raise InfraError("nvrtc_create_failed")
    try:
        options = [b"--gpu-architecture=sm_120", b"--std=c++17"]
        flags = (TEXT * len(options))(*options)
        err = bind(lib, "nvrtcCompileProgram", PTR, INT, C.POINTER(TEXT))(
            prog, len(flags), flags
        )
        if err == 6:  # NVRTC_ERROR_COMPILATION
            raise CandidateError("compile_error")
        if err:
            raise InfraError("nvrtc_compile_failure")
        n = SIZE()
        if (
            bind(lib, "nvrtcGetCUBINSize", PTR, C.POINTER(SIZE))(prog, C.byref(n))
            or not 0 < n.value <= 16 << 20
        ):
            raise InfraError("nvrtc_cubin_size")
        data = C.create_string_buffer(n.value)
        if bind(lib, "nvrtcGetCUBIN", PTR, PTR)(prog, data):
            raise InfraError("nvrtc_cubin_unavailable")
        blob = data.raw
        if not has_entry(blob):
            raise CandidateError("entry_missing")
        return blob
    finally:
        bind(lib, "nvrtcDestroyProgram", C.POINTER(PTR))(C.byref(prog))


def compiled(p):
    blob = cubin(p)
    return {
        "status": "compiled",
        "compiled": True,
        "cubin_sha256": hashlib.sha256(blob).hexdigest(),
        "cubin_bytes": len(blob),
    }


class DriverError(Exception):
    def __init__(self, code):
        self.code = code


class Driver:
    """Only the chosen GPU is visible inside the container; all memory is owned by its context."""

    def __init__(self, expected_uuid):
        try:
            lib = C.CDLL("libcuda.so.1")
        except OSError:
            raise InfraError("nvidia_driver_unavailable") from None
        self.funcs = {}
        signatures = {
            "cuInit": (UINT,),
            "cuDeviceGetCount": (C.POINTER(INT),),
            "cuDeviceGet": (C.POINTER(INT), INT),
            "cuDeviceGetAttribute": (C.POINTER(INT), INT, INT),
            "cuDeviceGetName": (PTR, INT, INT),
            "cuDeviceGetUuid_v2": (PTR, INT),
            "cuDriverGetVersion": (C.POINTER(INT),),
            "cuDevicePrimaryCtxRetain": (C.POINTER(PTR), INT),
            "cuCtxSetCurrent": (PTR,),
            "cuModuleLoadData": (C.POINTER(PTR), PTR),
            "cuModuleGetFunction": (C.POINTER(PTR), PTR, TEXT),
            "cuModuleUnload": (PTR,),
            "cuMemAlloc_v2": (C.POINTER(DEVICE_PTR), SIZE),
            "cuMemFree_v2": (DEVICE_PTR,),
            "cuMemcpyHtoD_v2": (DEVICE_PTR, PTR, SIZE),
            "cuMemcpyDtoH_v2": (PTR, DEVICE_PTR, SIZE),
            "cuLaunchKernel": (
                PTR,
                UINT,
                UINT,
                UINT,
                UINT,
                UINT,
                UINT,
                UINT,
                PTR,
                C.POINTER(PTR),
                C.POINTER(PTR),
            ),
            "cuEventCreate": (C.POINTER(PTR), UINT),
            "cuCtxSynchronize": (),
            "cuStreamCreate": (C.POINTER(PTR), UINT),
            "cuStreamBeginCapture_v2": (PTR, INT),
            "cuStreamEndCapture": (PTR, C.POINTER(PTR)),
            "cuEventRecordWithFlags": (PTR, PTR, UINT),
            "cuGraphInstantiateWithFlags": (C.POINTER(PTR), PTR, C.c_uint64),
            "cuGraphLaunch": (PTR, PTR),
            "cuStreamSynchronize": (PTR,),
            "cuGraphDestroy": (PTR,),
            "cuGraphExecDestroy": (PTR,),
            "cuEventElapsedTime": (C.POINTER(C.c_float), PTR, PTR),
        }
        for name, types in signatures.items():
            self.funcs[name] = bind(lib, name, *types)
        self.call("cuInit", 0)
        count = INT()
        self.call("cuDeviceGetCount", C.byref(count))
        if count.value != 1:
            raise InfraError("one_dedicated_cuda_device_required")
        device = INT()
        self.call("cuDeviceGet", C.byref(device), 0)
        arch = []
        for attr in (75, 76):  # CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_{MAJOR,MINOR}
            v = INT()
            self.call("cuDeviceGetAttribute", C.byref(v), attr, device)
            arch.append(v.value)
        if arch != [12, 0]:
            raise InfraError("hardware_target_mismatch")
        name = C.create_string_buffer(128)
        uuid = (C.c_ubyte * 16)()
        version = INT()
        self.call("cuDeviceGetName", name, 128, device)
        self.call("cuDeviceGetUuid_v2", uuid, device)
        self.call("cuDriverGetVersion", C.byref(version))
        if bytes(uuid).hex() != expected_uuid:
            raise InfraError("device_uuid_mismatch")
        self.context = PTR()
        self.call("cuDevicePrimaryCtxRetain", C.byref(self.context), device)
        self.call("cuCtxSetCurrent", self.context)
        self.stream = PTR()
        self.call("cuStreamCreate", C.byref(self.stream), 1)  # CU_STREAM_NON_BLOCKING
        self.hardware = {
            "device": name.value.decode(errors="replace"),
            "device_uuid": bytes(uuid).hex(),
            "driver_version": version.value,
            "compute_capability": "12.0",
        }
        self.events = (PTR(), PTR())
        for event in self.events:
            self.call("cuEventCreate", C.byref(event), 0)

    def call(self, name, *args):
        result = self.funcs[name](*args)
        if result:
            raise DriverError(result)

    @contextmanager
    def kernel(self, blob):
        module, function = PTR(), PTR()
        self.call("cuModuleLoadData", C.byref(module), C.create_string_buffer(blob))
        try:
            self.call("cuModuleGetFunction", C.byref(function), module, b"evaluate")
            yield function
        finally:
            self.funcs["cuModuleUnload"](module)

    @contextmanager
    def allocation(self, n):
        p = DEVICE_PTR()
        self.call("cuMemAlloc_v2", C.byref(p), n)
        try:
            yield p
        finally:
            # A failed device kernel can poison the whole context; preserve the original error.
            self.funcs["cuMemFree_v2"](p)


GUARD = bytes.fromhex("adf01e83") * 64
POISON = struct.pack(
    "<I", 0x7FC00001
)  # every finite oracle rejects any unwritten element


def fixed_case(case):
    if not isinstance(case, dict):
        raise InfraError("invalid_test_case")
    m = case.get("shape")
    if (
        not isinstance(m, list)
        or len(m) != 3
        or not all(type(x) is int and 0 < x <= MAX_ITEMS for x in m)
    ):
        raise InfraError("invalid_test_shape")
    for key in ("a", "b", "expected"):
        xs = case.get(key)
        if (
            not isinstance(xs, list)
            or not 1 <= len(xs) <= MAX_ITEMS
            or not all(type(x) in (int, float) and math.isfinite(x) for x in xs)
        ):
            raise InfraError("invalid_test_values")
    if len(case["expected"]) > 1 << 19:
        raise InfraError("invalid_test_shape")
    if type(case.get("timing", True)) is not bool or any(
        type(case.get(k)) not in (float, int)
        or not math.isfinite(case[k])
        or case[k] < 0
        for k in ("atol", "rtol")
    ):
        raise InfraError("invalid_test_tolerance")
    return case


def packed(xs):
    try:
        return array.array("f", xs).tobytes()
    except (ValueError, OverflowError):
        raise InfraError("invalid_test_values") from None


class Trial:
    def __init__(self, driver, case):
        self.driver = driver
        self.case = case
        expected = case["expected"]
        blank = POISON * len(expected)
        self.initial = [
            GUARD + raw + GUARD for raw in (packed(case["a"]), packed(case["b"]), blank)
        ]

    def run(self, blob, program):
        # Freshly restored inputs and poisoned non-aliased outputs on every launch.
        d = self.driver
        with ExitStack() as stack:
            # A fresh module restores device globals on every independent call.
            function = stack.enter_context(d.kernel(blob))
            ptrs = [stack.enter_context(d.allocation(len(raw))) for raw in self.initial]
            for ptr, raw in zip(ptrs, self.initial):
                d.call("cuMemcpyHtoD_v2", ptr, raw, len(raw))
            # Pageable host-to-device copies can return after staging. The graph uses
            # a non-blocking stream, so drain all transfers before recording any event.
            d.call("cuCtxSynchronize")
            n = len(self.case["expected"])
            values = [
                *(DEVICE_PTR(p.value + len(GUARD)) for p in ptrs),
                *(UINT(x) for x in (*self.case["shape"], n)),
            ]
            argv = (PTR * len(values))(*(C.cast(C.byref(v), PTR) for v in values))
            t = program["threads"]
            per_block = t * program["outputs_per_thread"]
            blocks = (n + per_block - 1) // per_block
            start, end = d.events
            # Schedule both event records and the kernel in one device graph. Driver/Python
            # submission gaps between separate launches would dominate sub-10us kernels.
            graph, executable = PTR(), PTR()
            d.call("cuStreamBeginCapture_v2", d.stream, 1)  # THREAD_LOCAL
            d.call("cuEventRecordWithFlags", start, d.stream, 1)  # EXTERNAL
            d.call(
                "cuLaunchKernel",
                function,
                blocks,
                1,
                1,
                t,
                1,
                1,
                0,
                d.stream,
                argv,
                None,
            )
            d.call("cuEventRecordWithFlags", end, d.stream, 1)
            d.call("cuStreamEndCapture", d.stream, C.byref(graph))
            stack.callback(d.call, "cuGraphDestroy", graph)
            d.call("cuGraphInstantiateWithFlags", C.byref(executable), graph, 0)
            stack.callback(d.call, "cuGraphExecDestroy", executable)
            d.call("cuGraphLaunch", executable, d.stream)
            d.call("cuStreamSynchronize", d.stream)
            ms = C.c_float()
            d.call("cuEventElapsedTime", C.byref(ms), start, end)
            us = ms.value * 1000.0
            if not math.isfinite(us) or us <= 0:
                raise InfraError("hardware_timestamps_unavailable")
            guarded = True
            out = b""
            for i, (ptr, raw) in enumerate(zip(ptrs, self.initial)):
                host = C.create_string_buffer(len(raw))
                d.call("cuMemcpyDtoH_v2", host, ptr, len(raw))
                value = host.raw
                if i < 2:
                    guarded &= value == raw
                else:
                    guarded &= value.startswith(GUARD) and value.endswith(GUARD)
                    out = value[len(GUARD) : -len(GUARD)]
            f = array.array("f")
            f.frombytes(out)
            max_error = 0.0
            bad = 0
            for got, expected in zip(f, self.case["expected"], strict=True):
                if not math.isfinite(got):
                    bad += 1
                else:
                    delta = abs(got - expected)
                    max_error = max(max_error, delta)
                    bad += int(
                        delta > self.case["atol"] + self.case["rtol"] * abs(expected)
                    )
            return {
                "passed": bool(guarded and not bad),
                "mismatches": bad,
                "buffers_intact": bool(guarded),
                "max_finite_abs_error": max_error,
            }, us


def timing(rounds):
    ratios = [q["baseline_us"] / q["candidate_us"] for q in rounds]
    lo, hi = min(ratios), max(ratios)
    return {
        "baseline_us": statistics.median(q["baseline_us"] for q in rounds),
        "candidate_us": statistics.median(q["candidate_us"] for q in rounds),
        "speedup": statistics.median(ratios),
        "ratio_min": lo,
        "ratio_max": hi,
        "comparison": "faster" if lo > 1 else "slower" if hi < 1 else "overlap",
    }


def run_trials(driver, programs, functions, cases, seed):
    trials = [Trial(driver, case) for case in cases]
    rows = []
    result = {
        "status": "correct",
        "compiled": True,
        "cases": rows,
        **driver.hardware,
        "seed": seed,
        "timing_source": "CUDA graph external device events, one dispatch per trial",
    }

    def verify(i, which):
        try:
            valid, us = trials[i].run(functions[which], programs[which])
        except DriverError as exc:
            if which == 1 and exc.code in (
                700,
                701,
                702,
                710,
                714,
                715,
                716,
                717,
                718,
                719,
                720,
            ):
                raise CandidateError("device_execution_error") from None
            raise InfraError("cuda_trial_unavailable") from None
        if not valid["passed"] and which == 0:
            raise InfraError("baseline_disagrees_with_oracle")
        return valid, us

    # A global correctness gate also covers the larger benchmark shapes.
    for i, case in enumerate(cases):
        verify(i, 0)
        valid, _ = verify(i, 1)
        rows.append(
            {
                "shape": case["shape"],
                "elements": len(case["expected"]),
                "validation": valid,
            }
        )
    if any(not row["validation"]["passed"] for row in rows):
        return {**result, "status": "incorrect"}
    rng = random.Random(seed)
    for i, (case, row) in enumerate(zip(cases, rows)):
        if not case.get("timing", True):
            continue
        order = [0, 1] * (ROUNDS // 2) + [rng.randrange(2)]
        rng.shuffle(order)
        rounds = []
        # Warm both implementations once before timed rounds, reversing first position across rounds.
        for first in [0, *order]:
            elapsed = {}
            for side in (first, 1 - first):
                valid, elapsed[side] = verify(i, side)
                if not valid["passed"]:
                    row["validation"] = valid
                    for prior in rows:
                        prior.pop("timing", None)
                        prior.pop("rounds", None)
                        prior.pop("dispatches_per_sample", None)
                    return {**result, "status": "incorrect"}
            rounds.append(
                {
                    "baseline_first": first == 0,
                    "baseline_us": elapsed[0],
                    "candidate_us": elapsed[1],
                }
            )
        row.update(
            rounds=rounds[1:], timing=timing(rounds[1:]), dispatches_per_sample=1
        )
    return result


def evaluate(req):
    cases = req["cases"]
    if (
        not isinstance(cases, list)
        or not 1 <= len(cases) <= 64
        or type(req.get("seed")) is not int
        or not 0 <= req["seed"] <= 0xFFFFFFFF
    ):
        raise InfraError("invalid_test_configuration")
    cases = [fixed_case(case) for case in cases]
    try:
        reference = cubin(req["baseline"])
    except CandidateError:
        raise InfraError("baseline_compile_failure") from None
    source = cubin(req["candidate"])
    try:
        driver = Driver(req["device_uuid"])
        kernels = [reference, source]
        report = run_trials(
            driver, [req["baseline"], req["candidate"]], kernels, cases, req["seed"]
        )
    except DriverError:
        raise InfraError("cuda_driver_failure") from None
    except CandidateError as exc:
        return {
            "status": "incorrect",
            "compiled": True,
            "cases": [],
            "reason": exc.args[0],
            **driver.hardware,
        }
    return {
        **report,
        "candidate_hash": hashlib.sha256(source).hexdigest(),
        "baseline_hash": hashlib.sha256(reference).hexdigest(),
    }


def main():
    try:
        data = sys.stdin.buffer.read(MAX_PAYLOAD + 1)
        if len(data) > MAX_PAYLOAD:
            raise InfraError("oversized_payload")
        req = json.loads(data)
        if req["op"] == "compile":
            result = compiled(req["program"])
        elif req["op"] == "measure":
            result = evaluate(req)
        else:
            raise InfraError("invalid_operation")
    except CandidateError as exc:
        code = exc.args[0]
        result = {
            "status": "invalid_program"
            if code == "invalid_program"
            else "compile_error",
            "compiled": False,
            "reason": code,
            "cases": [],
        }
    except InfraError as exc:
        result = {"status": "unavailable", "reason": exc.args[0]}
    except Exception:
        result = {"status": "unavailable", "reason": "worker_error"}
    return {
        "backend": "cuda",
        "protocol": "cuda-native-v1",
        "target": TARGET,
        "compiler": COMPILER,
        **result,
    }


if __name__ == "__main__":
    import os

    # Compiler and device printf output never becomes part of the private results channel.
    channel = os.fdopen(os.dup(1), "w")
    with open(os.devnull, "wb") as discard:
        os.dup2(discard.fileno(), 1)
    print(json.dumps(main(), allow_nan=False), file=channel, flush=True)
