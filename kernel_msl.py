"""Compile native Metal compute kernels and validate their output on the local Apple GPU."""

import array
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import statistics
import subprocess
import sys
import tempfile

import kernel_metal as metal

BackendUnavailable = metal.BackendUnavailable
VERSION = "metal-native-v1"
HERE = Path(__file__).resolve().parent
MAX_SOURCE = 64_000
HEADER = re.compile(r"#[ \t]*include[ \t]*<[ \t]*metal_stdlib[ \t]*>[ \t]*")
LOCAL_DIRECTIVE = re.compile(
    r"#[ \t]*(?:define|undef|if|ifdef|ifndef|elif|else|endif)\b"
)
TOKENS = re.compile(
    r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n])*\'', re.S
)
INFRA_ERRORS = {
    "missing_input",
    "report_error",
    "invalid_reference",
    "metal_compiler_unavailable",
    "pipeline_unavailable",
    "reflection_unavailable",
    "metal_allocation",
    "command_encoder_unavailable",
    "command_unavailable",
    "hardware_timestamps_unavailable",
    "baseline_disagrees_with_oracle",
    "apple_gpu_unavailable",
    "queue_unavailable",
    "worker_limits_unavailable",
    "invalid_request",
}


class BackendTimeout(BackendUnavailable):
    reason_code = "backend_timeout"


def acceptable_source(source):
    if not isinstance(source, str) or not source.strip() or len(source) > MAX_SOURCE:
        return False
    try:
        if len(source.encode()) > MAX_SOURCE:
            return False
    except UnicodeEncodeError:
        return False
    if any(ord(c) < 32 and c not in "\t\n\r" for c in source):
        return False
    text = source.removeprefix("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    # Check both translation modes so comments and strings remain harmless even
    # where an older preprocessor recognizes alternative spellings.
    translated = text.replace("??=", "#").replace("??/", "\\").replace("%:", "#")
    return _directives_allowed(text) and (
        text == translated or _directives_allowed(translated)
    )


def _directives_allowed(text):
    normalized = re.sub(r"\\[ \t]*\n", "", text)
    code = TOKENS.sub(
        lambda m: (
            " " + "\n" * m.group().count("\n")
            if m.group().startswith("/")
            else "STRING"
        ),
        normalized,
    )
    if "##" in code or re.search(
        r"\b(?:_Pragma|__pragma|__has_include(?:_next)?|__has_embed)\b", code
    ):
        return False
    pragma = re.compile(
        r"#[ \t]*pragma[ \t]+(?:(?:unroll|nounroll)(?:[ \t]+[A-Za-z0-9_]+)?|clang[ \t]+loop[ \t]+[A-Za-z0-9_() \t]+)[ \t]*"
    )
    for line in code.split("\n"):
        line = line.lstrip()
        if "#" in line and not (
            HEADER.fullmatch(line)
            or LOCAL_DIRECTIVE.match(line)
            or pragma.fullmatch(line)
        ):
            return False
    return True


def _build_host(cache):
    src = HERE / "kernel_msl_host.m"
    command = ["/usr/bin/xcrun", "--sdk", "macosx", "clang"]
    compiler = subprocess.run(
        command + ["--version"], check=True, capture_output=True, timeout=10
    ).stdout
    key = hashlib.sha256(
        src.read_bytes() + compiler + platform.release().encode()
    ).hexdigest()
    binary = cache / ("native-" + key[:24])
    if not binary.is_file():
        with tempfile.TemporaryDirectory(dir=cache) as folder:
            temp = Path(folder) / "host"
            subprocess.run(
                command
                + [
                    "-fobjc-arc",
                    "-O2",
                    "-Wno-deprecated-declarations",
                    "-framework",
                    "Foundation",
                    "-framework",
                    "Metal",
                    src,
                    "-o",
                    temp,
                ],
                check=True,
                capture_output=True,
                timeout=60,
            )
            os.replace(temp, binary)
    return binary, compiler.decode().splitlines()[0]


def _buffers(path, cases):
    """Keep the stored CPU witnesses in float64 and the two shader inputs in float32."""
    metadata = []
    total = 0
    for index, case in enumerate(cases):
        info = {k: case[k] for k in ("shape", "atol", "rtol")} | {
            "timing": case.get("timing", True)
        }
        case_hash = hashlib.sha256(json.dumps(info, sort_keys=True).encode())
        for name, typecode in (("a", "f"), ("b", "f"), ("expected", "d")):
            try:
                values = array.array(typecode, case[name])
            except (OverflowError, TypeError, ValueError) as exc:
                raise BackendUnavailable("invalid native Metal buffer") from exc
            if not all(map(math.isfinite, values)):
                raise BackendUnavailable(
                    "native Metal input and CPU witness values must be finite"
                )
            total += values.itemsize * len(values)
            if total > 256 << 20:
                raise BackendUnavailable("suite exceeds native Metal memory bound")
            if sys.byteorder != "little":
                values.byteswap()
            data = values.tobytes()
            case_hash.update(len(data).to_bytes(8, "little"))
            case_hash.update(data)
            (path / f"{index}-{name}").write_bytes(data)
        metadata.append({**info, "case_hash": case_hash.hexdigest()})
    return metadata


def _fingerprint(source):
    return hashlib.sha256(source.encode()).hexdigest()


def _require(condition):
    if not condition:
        raise BackendUnavailable("inconsistent native Metal host report")


def _check_report(data, measuring, shapes, cases, orders):
    _require(isinstance(data, dict) and type(data.get("compiled")) is bool)
    status = data.get("status")
    if status == "compile_error":
        _require(
            data
            == {
                "status": status,
                "compiled": False,
                "reason": data.get("reason"),
                "cases": [],
            }
            and data["reason"] in ("compile_error", "entry_missing", "source_contract")
        )
        return
    if status == "incorrect" and data.get("reason") == "device_execution_error":
        _require(
            measuring
            and data
            == {
                "status": status,
                "compiled": True,
                "reason": "device_execution_error",
                "cases": [],
            }
        )
        return
    keys = {"status", "compiled", "backend", "device", "registry_id", "simd_width"}
    if measuring:
        keys.add("cases")
    _require(
        data.keys() == keys
        and data["compiled"] is True
        and data["backend"] == "metal-native"
        and isinstance(data["device"], str)
        and data["device"].strip()
        and all(
            type(data[k]) is int and data[k] > 0 for k in ("registry_id", "simd_width")
        )
    )
    if not measuring:
        _require(status == "compiled")
        return
    metal._validate_host_result({**data, "backend": "metal"}, shapes, cases, orders)
    for row in data["cases"]:
        check = row["validation"]
        _require(
            type(check.get("buffers_intact")) is bool
            and check["buffers_intact"] is not check["invalid_operation"]
        )
        if "rounds" in row:
            _require(row["dispatches_per_sample"] == 1)


def _timing(pairs):
    ratios = [v["baseline_us"] / v["candidate_us"] for v in pairs]
    return {
        "baseline_us": statistics.median(v["baseline_us"] for v in pairs),
        "candidate_us": statistics.median(v["candidate_us"] for v in pairs),
        "speedup": statistics.median(ratios),
        "ratio_min": min(ratios),
        "ratio_max": max(ratios),
        "comparison": "faster"
        if min(ratios) > 1
        else "slower"
        if max(ratios) < 1
        else "overlap",
    }


def _run(candidate, baseline, cases, seed):
    measuring = baseline is not None
    if measuring and not acceptable_source(baseline):
        raise BackendUnavailable("reference native Metal source outside contract")
    if not acceptable_source(candidate):
        return {
            "backend": "metal-native",
            "protocol": VERSION,
            "status": "invalid_program",
            "reason": "invalid_source",
            "compiled": False,
            "cases": [],
        }
    if (
        platform.system() != "Darwin"
        or platform.machine() != "arm64"
        or int(platform.mac_ver()[0].split(".")[0]) < 13
    ):
        raise BackendUnavailable(
            "native Metal requires macOS 13 or newer on Apple silicon"
        )
    shapes = metal._case_shapes(cases) if measuring else []
    if type(seed) is not int or not 0 <= seed <= 0xFFFFFFFF:
        raise ValueError("seed must be a uint32")
    orders = []
    rng = random.Random(seed)
    for _ in shapes:
        first = [0, 1] * 3 + [rng.randrange(2)]
        rng.shuffle(first)
        orders.append(first)
    try:
        with metal._device_slot() as cache:
            binary, clang = _build_host(cache)
            with tempfile.TemporaryDirectory(dir=cache) as folder:
                path = Path(folder)
                (path / "candidate.metal").write_text(candidate, encoding="utf8")
                if measuring:
                    (path / "baseline.metal").write_text(baseline, encoding="utf8")
                    metadata = _buffers(path, cases)
                    (path / "run.json").write_text(
                        json.dumps({"cases": metadata, "order": orders})
                    )
                # Source diagnostics and any driver messages remain local to this child.
                with (
                    tempfile.TemporaryFile(dir=path) as channel,
                    tempfile.TemporaryFile(dir=path) as diagnostics,
                ):
                    result = subprocess.run(
                        [binary, "measure" if measuring else "compile"],
                        cwd=path,
                        env={
                            "PATH": "/usr/bin:/bin",
                            "HOME": str(path),
                            "TMPDIR": str(path),
                        },
                        stdin=subprocess.DEVNULL,
                        stdout=channel,
                        stderr=diagnostics,
                        timeout=120 if measuring else 40,
                    )
                    if result.returncode:
                        diagnostics.seek(0)
                        code = diagnostics.read(128).decode(errors="replace").strip()
                        raise BackendUnavailable(
                            code
                            if code in INFRA_ERRORS
                            else "native Metal worker unavailable"
                        )
                    _require(channel.tell() <= 1 << 20)
                    channel.seek(0)
                    data = json.load(channel)
    except subprocess.TimeoutExpired:
        raise BackendTimeout("native Metal evaluator deadline expired") from None
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        raise BackendUnavailable(
            "native Metal toolchain or worker unavailable"
        ) from exc
    _check_report(data, measuring, shapes, cases, orders)
    for row, meta in zip(data.get("cases", []), metadata if measuring else []):
        row["case_hash"] = meta["case_hash"]
        if "rounds" in row:
            row["timing"] = _timing(row["rounds"])
    data.update(
        protocol=VERSION,
        backend="metal-native",
        target="Apple Metal",
        compiler="Apple Metal runtime compiler",
        host_compiler=clang,
        os=platform.mac_ver()[0],
        os_build=platform.version(),
        candidate_hash=_fingerprint(candidate),
    )
    if measuring:
        data.update(
            seed=seed,
            baseline_hash=_fingerprint(baseline),
            timing_source="Metal command-buffer GPU timestamps, one dispatch per trial",
        )
    return data


def compile_native(source):
    """Build the compute pipeline and check reflected bindings on the selected local device."""
    return _run(source, None, (), 17)


def measure(candidate, baseline, cases, *, seed=17):
    """Check all CPU witnesses, then record seven paired trials on every timed shape."""
    if baseline is None:
        raise BackendUnavailable("missing native Metal reference")
    return _run(candidate, baseline, cases, seed)
