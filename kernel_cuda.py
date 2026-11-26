"""Run the isolated native WeakBench CUDA compiler or the opt-in local GPU evaluator."""

import contextlib
import fcntl
import json
import sys
import os
from pathlib import Path
import random
import re
import shutil
import stat
import statistics
import subprocess
import tempfile
import uuid

IMAGE = "weakbench-kernel-cuda:1"
PROTOCOL = "cuda-native-v1"
DEVICE = re.compile(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}")
MAX_INPUT = 120 << 20
MAX_OUTPUT = 256 << 10
ROUNDS = 7
TIMING_SOURCE = "CUDA graph external device events, one dispatch per trial"
HEADER = {"backend", "protocol", "target", "compiler", "status"}
DEVICE_FIELDS = {"device", "device_uuid", "driver_version", "compute_capability"}


class BackendUnavailable(RuntimeError):
    """No verified model result is available from this evaluation."""

    reason_code = "backend_unavailable"


class BackendTimeout(BackendUnavailable):
    """An evaluation exceeded its deadline before a verifiable result was returned."""

    reason_code = "backend_timeout"


class DedicatedDeviceUnselected(BackendUnavailable):
    """The operator must select the dedicated device before timing."""

    reason_code = "dedicated_device_unselected"


def _docker():
    path = os.environ.get("WEAKBENCH_KERNEL_DOCKER_SOCKET") or os.environ.get(
        "WEAKBENCH_ARENA_DOCKER_SOCKET"
    )
    if not path:
        host = os.environ.get("DOCKER_HOST", "unix:///var/run/docker.sock")
        if not host.startswith("unix://"):
            raise BackendUnavailable("a local Unix Docker socket is required")
        path = host[7:]
    if not os.path.isabs(path) or not Path(path).is_socket():
        raise BackendUnavailable("local kernel evaluator Docker socket unavailable")
    cli = shutil.which("docker")
    if not cli:
        raise BackendUnavailable("Docker CLI unavailable")
    return [cli, "--host", "unix://" + path]


def _image(cli):
    try:
        result = subprocess.run(
            [
                *cli,
                "image",
                "inspect",
                "--format",
                '{{.Id}} {{index .Config.Labels "org.weakbench.kernel-cuda"}}',
                IMAGE,
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        raise BackendUnavailable(
            "build the local weakbench-kernel-cuda:1 evaluator image"
        ) from None
    parts = result.stdout.strip().split()
    if (
        len(parts) != 2
        or not re.fullmatch("sha256:[0-9a-f]{64}", parts[0])
        or parts[1] != PROTOCOL
    ):
        raise BackendUnavailable("unexpected CUDA evaluator image")
    return parts[0]


@contextlib.contextmanager
def _device_slot(device):
    folder = Path(tempfile.gettempdir()) / f"weakbench-kernel-cuda-{os.getuid()}"
    folder.mkdir(mode=0o700, exist_ok=True)
    status = folder.lstat()
    if (
        not stat.S_ISDIR(status.st_mode)
        or status.st_uid != os.getuid()
        or status.st_mode & 0o077
    ):
        raise BackendUnavailable("invalid local GPU lock directory")
    fd = os.open(
        folder / (device.lower() + ".lock"),
        os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW,
        0o600,
    )
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise BackendUnavailable("dedicated CUDA evaluator is busy") from None
        yield
    finally:
        os.close(fd)


def _require(condition):
    if not condition:
        raise BackendUnavailable("inconsistent CUDA evaluator report")


def _finite(value):
    # The upper bound also rejects arbitrarily large JSON integers before float conversion.
    return (
        type(value) in (float, int)
        and -sys.float_info.max <= value <= sys.float_info.max
    )


def _sha256(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _check_timing(row, rng):
    rounds = row["rounds"]
    _require(
        type(row["dispatches_per_sample"]) is int and row["dispatches_per_sample"] == 1
    )
    _require(isinstance(rounds, list) and len(rounds) == ROUNDS)
    for pair in rounds:
        _require(
            isinstance(pair, dict)
            and pair.keys() == {"baseline_first", "baseline_us", "candidate_us"}
        )
        _require(
            type(pair["baseline_first"]) is bool
            and all(
                _finite(pair[k]) and pair[k] > 0
                for k in ("baseline_us", "candidate_us")
            )
        )
    order = [0, 1] * (ROUNDS // 2) + [rng.randrange(2)]
    rng.shuffle(order)
    _require(
        [pair["baseline_first"] for pair in rounds] == [side == 0 for side in order]
    )
    ratios = [pair["baseline_us"] / pair["candidate_us"] for pair in rounds]
    _require(all(_finite(ratio) and ratio > 0 for ratio in ratios))
    low, high = min(ratios), max(ratios)
    expected = {
        "baseline_us": statistics.median(pair["baseline_us"] for pair in rounds),
        "candidate_us": statistics.median(pair["candidate_us"] for pair in rounds),
        "speedup": statistics.median(ratios),
        "ratio_min": low,
        "ratio_max": high,
    }
    summary = row["timing"]
    _require(
        isinstance(summary, dict) and summary.keys() == expected.keys() | {"comparison"}
    )
    _require(
        all(
            _finite(summary[k]) and summary[k] == value for k, value in expected.items()
        )
    )
    _require(
        summary["comparison"]
        == ("faster" if low > 1 else "slower" if high < 1 else "overlap")
    )


def _check_cases(rows, requested, correct, seed):
    _require(isinstance(requested, list) and 1 <= len(requested) <= 64)
    _require(isinstance(rows, list) and len(rows) == len(requested))
    rng = random.Random(seed)
    passed = []
    for row, case in zip(rows, requested, strict=True):
        _require(isinstance(case, dict) and isinstance(case.get("expected"), list))
        _require(type(case.get("timing", True)) is bool)
        timed = correct and case.get("timing", True)
        fields = {"shape", "elements", "validation"}
        if timed:
            fields.update(("rounds", "timing", "dispatches_per_sample"))
        _require(isinstance(row, dict) and row.keys() == fields)
        shape = row["shape"]
        _require(
            isinstance(shape, list)
            and len(shape) == 3
            and all(type(n) is int and n > 0 for n in shape)
            and shape == case.get("shape")
            and all(type(n) is int for n in case["shape"])
        )
        elements = row["elements"]
        _require(type(elements) is int and elements == len(case["expected"]) > 0)
        check = row["validation"]
        _require(
            isinstance(check, dict)
            and check.keys()
            == {"passed", "mismatches", "buffers_intact", "max_finite_abs_error"}
        )
        bad = check["mismatches"]
        _require(
            type(bad) is int
            and 0 <= bad <= elements
            and type(check["buffers_intact"]) is bool
        )
        _require(check["passed"] is (bad == 0 and check["buffers_intact"]))
        _require(
            _finite(check["max_finite_abs_error"])
            and check["max_finite_abs_error"] >= 0
        )
        if bad == 0:
            atol, rtol = case.get("atol"), case.get("rtol")
            _require(_finite(atol) and atol >= 0 and _finite(rtol) and rtol >= 0)
            if check["max_finite_abs_error"] > atol:
                _require(rtol > 0 and all(_finite(x) for x in case["expected"]))
                max_allowed = atol + rtol * max(abs(x) for x in case["expected"])
                _require(check["max_finite_abs_error"] <= max_allowed)
        passed.append(check["passed"])
        if timed:
            _check_timing(row, rng)
    _require(all(passed) is correct)


def _validate_report(request, result, device):
    _require(isinstance(result, dict))
    _require(
        result.get("protocol") == PROTOCOL
        and result.get("backend") == "cuda"
        and result.get("target") == "sm_120"
        and result.get("compiler") == "NVRTC 13.0.88"
    )
    status = result.get("status")
    if status == "unavailable":
        reason = result.get("reason")
        _require(
            result.keys() == HEADER | {"reason"}
            and isinstance(reason, str)
            and re.fullmatch(r"[a-z0-9_]{1,96}", reason)
        )
        raise BackendUnavailable(reason)
    if status in ("compile_error", "invalid_program"):
        _require(result.keys() == HEADER | {"reason", "compiled", "cases"})
        _require(result["compiled"] is False and result["cases"] == [])
        reasons = (
            ("compile_error", "entry_missing")
            if status == "compile_error"
            else ("invalid_program",)
        )
        _require(result["reason"] in reasons)
        return
    _require(result.get("compiled") is True)
    if request["op"] == "compile":
        _require(
            status == "compiled"
            and result.keys() == HEADER | {"compiled", "cubin_sha256", "cubin_bytes"}
        )
        _require(
            _sha256(result["cubin_sha256"])
            and type(result["cubin_bytes"]) is int
            and 192 <= result["cubin_bytes"] <= 16 << 20
        )
        return
    _require(
        request["op"] == "measure" and device and status in ("correct", "incorrect")
    )
    _require(
        result.get("device_uuid") == device[4:].replace("-", "").lower()
        and result.get("compute_capability") == "12.0"
    )
    _require(
        isinstance(result.get("device"), str)
        and 0 < len(result["device"].strip()) <= 128
    )
    _require(type(result.get("driver_version")) is int and result["driver_version"] > 0)
    base = HEADER | DEVICE_FIELDS | {"compiled", "cases"}
    # A CUDA execution fault terminates the context before per-shape validation completes.
    if status == "incorrect" and result.get("reason") == "device_execution_error":
        _require(result.keys() == base | {"reason"} and result["cases"] == [])
        return
    _require(
        result.keys()
        == base | {"seed", "timing_source", "candidate_hash", "baseline_hash"}
    )
    _require(_sha256(result["candidate_hash"]) and _sha256(result["baseline_hash"]))
    _require(
        type(result["seed"]) is int
        and type(request["seed"]) is int
        and 0 <= result["seed"] == request["seed"] <= 0xFFFFFFFF
        and result["timing_source"] == TIMING_SOURCE
    )
    _check_cases(result["cases"], request["cases"], status == "correct", result["seed"])


def _run(request, device=None):
    try:
        data = json.dumps(request, allow_nan=False, separators=(",", ":")).encode()
    except (TypeError, ValueError):
        raise BackendUnavailable("invalid evaluator request") from None
    if len(data) > MAX_INPUT:
        raise BackendUnavailable("evaluator request exceeds size limit")
    cli = _docker()
    image = _image(cli)
    name = "weakbench-kernel-cuda-" + uuid.uuid4().hex
    cmd = [
        *cli,
        "run",
        "--rm",
        "--pull=never",
        "--name",
        name,
        "--read-only",
        "--network=none",
        "--cap-drop=ALL",
        "--user=65534:65534",
        "--security-opt=no-new-privileges",
        "--pids-limit=64",
        "--memory=1g",
        "--memory-swap=1g",
        "--cpus=2",
        "--tmpfs=/tmp:rw,nosuid,nodev,noexec,size=128m",
        "--log-driver=none",
        "--ipc=private",
    ]
    if device:
        cmd += ["--gpus", "device=" + device]
    cmd += ["-i", image]
    # Only the trusted worker writes stdout. It discards compiler/device printf output.
    try:
        with tempfile.TemporaryFile() as report:
            completed = subprocess.run(
                cmd,
                input=data,
                stdout=report,
                stderr=subprocess.DEVNULL,
                timeout=180 if device else 35,
            )
            if completed.returncode != 0:
                raise BackendUnavailable("CUDA evaluator container failed")
            if report.tell() > MAX_OUTPUT:
                raise BackendUnavailable("CUDA evaluator report exceeds size limit")
            report.seek(0)
            result = json.load(report)
    except subprocess.TimeoutExpired:
        raise BackendTimeout(
            "CUDA evaluator deadline expired before returning a verified result"
        ) from None
    except (OSError, ValueError, RecursionError):
        raise BackendUnavailable("CUDA evaluator produced no valid result") from None
    finally:
        try:
            subprocess.run(
                [*cli, "rm", "--force", name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            pass
    _validate_report(request, result, device)
    return {**result, "image_id": image}


def compile_native(program):
    """Compile to SM120 SASS and verify the actual evaluate entry. No device is needed."""
    return _run({"op": "compile", "program": program})


def measure(candidate, baseline, cases, seed=0):
    """Validate both kernels against fixed CPU oracles and time pairs on an explicit local SM120 GPU."""
    device = os.environ.get("WEAKBENCH_CUDA_DEVICE", "")
    if not DEVICE.fullmatch(device):
        raise DedicatedDeviceUnselected(
            "set WEAKBENCH_CUDA_DEVICE to a dedicated local SM120 GPU UUID"
        )
    with _device_slot(device):
        return _run(
            {
                "op": "measure",
                "candidate": candidate,
                "baseline": baseline,
                "device_uuid": device[4:].replace("-", "").lower(),
                "cases": cases,
                "seed": seed,
            },
            device,
        )
