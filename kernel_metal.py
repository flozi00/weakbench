"""Compile a bounded Python kernel body into Metal and compare it on the local Apple GPU.

Only AST nodes explicitly lowered below can reach the Metal compiler. The candidate
owns one output element at a time. Reads are bounds-checked against its own input
buffers, and loops have statically bounded trip counts for every supplied shape.
"""

from __future__ import annotations

import array
import ast
from collections import Counter
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
VERSION = "metal-element-v3"
MAX_ELEMENTS = 1 << 20
MAX_OUTPUT = 1 << 19
MAX_WORK = 96_000_000
INTEGERS = {"M", "N", "K", "L", "i"}
THREADS = (32, 64, 128, 256)
LANES = (1, 2, 4, 8)


class KernelRejected(ValueError):
    """Candidate source is outside the bounded kernel contract."""


class BackendUnavailable(RuntimeError):
    """The GPU, host compiler, reference implementation or measurement was unavailable."""


PRELUDE = r"""#include <metal_stdlib>
using namespace metal;

struct Shape { uint M,N,K,L,na,nb; };
inline float rd(device const float *buffer, uint n, uint offset, device atomic_uint *bad) {
    if (offset < n) return buffer[offset];
    atomic_store_explicit(bad, 1u, memory_order_relaxed);
    return 0.0f;
}
inline uint divu(uint x, uint y, device atomic_uint *bad) {
    if (y) return x / y;
    atomic_store_explicit(bad, 1u, memory_order_relaxed);
    return 0u;
}
inline uint modu(uint x, uint y, device atomic_uint *bad) {
    if (y) return x % y;
    atomic_store_explicit(bad, 1u, memory_order_relaxed);
    return 0u;
}

kernel void evaluate(device const float *a [[buffer(0)]], device const float *b [[buffer(1)]],
    device float *output [[buffer(2)]], constant Shape &s [[buffer(3)]],
    device atomic_uint *bad [[buffer(4)]], uint gid [[thread_position_in_grid]]) {
    const uint M=s.M, N=s.N, K=s.K, L=s.L;
"""

OPS = {
    ast.Add: "+",
    ast.Sub: "-",
    ast.Mult: "*",
    ast.BitAnd: "&",
    ast.BitOr: "|",
    ast.BitXor: "^",
    ast.LShift: "<<",
    ast.RShift: ">>",
}
COMPARE = {
    ast.Eq: "==",
    ast.NotEq: "!=",
    ast.Lt: "<",
    ast.LtE: "<=",
    ast.Gt: ">",
    ast.GtE: ">=",
}


def _ensure(test, message):
    if not test:
        raise KernelRejected(message)


def _var(name):
    return name if name in INTEGERS else "v_" + name


class _Compiler:
    def __init__(self, tree, shape, writes):
        self.types = {name: "uint" for name in INTEGERS}
        self.shape = dict(zip(("M", "N", "K", "L"), shape))
        self.bounds = self.shape.copy()
        self.writes = writes
        self.iterators = set()
        self.cost = 0
        self.weight = 1
        self.lines = []
        assigned = self.block(tree.body, set(INTEGERS), 0)
        _ensure("out" in assigned, "out must be assigned on every path")

    def expr(self, node, assigned):
        self.cost += self.weight
        _ensure(
            self.cost * self.shape["L"] <= MAX_WORK,
            "shape exceeds the static operation bound",
        )
        if isinstance(node, ast.Constant):
            v = node.value
            if type(v) is int and 0 <= v <= 0x7FFFFFFF:
                return "uint", str(v) + "u"
            if type(v) is float and math.isfinite(v) and abs(v) <= 3.4e38:
                return "float", repr(v) + "f"
        elif isinstance(node, ast.Name):
            _ensure(node.id in assigned, "uninitialized variable")
            return self.types[node.id], _var(node.id)
        elif isinstance(node, ast.UnaryOp) and type(node.op) in (
            ast.USub,
            ast.UAdd,
            ast.Not,
            ast.Invert,
        ):
            # Negative scalar literals are float32. Index register operations use uint32.
            if (
                isinstance(node.op, ast.USub)
                and isinstance(node.operand, ast.Constant)
                and type(node.operand.value) is int
                and 0 <= node.operand.value <= 0x7FFFFFFF
            ):
                return "float", f"(-{node.operand.value}.0f)"
            t, v = self.expr(node.operand, assigned)
            if isinstance(node.op, ast.Not):
                return "bool", f"(!({v}))"
            if isinstance(node.op, ast.Invert):
                _ensure(t == "uint", "bitwise operand must be an index")
                return "uint", f"(~({v}))"
            _ensure(t != "bool", "numeric operand required")
            return t, f"({'-' if isinstance(node.op, ast.USub) else '+'}({v}))"
        elif isinstance(node, ast.BinOp):
            return self.arithmetic(
                node.op, self.expr(node.left, assigned), self.expr(node.right, assigned)
            )
        elif (
            isinstance(node, ast.Compare)
            and len(node.ops) == 1
            and type(node.ops[0]) in COMPARE
        ):
            left_type, left_src = self.expr(node.left, assigned)
            right_type, right_src = self.expr(node.comparators[0], assigned)
            if "float" in (left_type, right_type):
                left_src, right_src = f"float({left_src})", f"float({right_src})"
            return "bool", f"({left_src} {COMPARE[type(node.ops[0])]} {right_src})"
        elif isinstance(node, ast.BoolOp) and type(node.op) in (ast.And, ast.Or):
            values = [self.expr(v, assigned)[1] for v in node.values]
            return "bool", "(" + (
                " && " if isinstance(node.op, ast.And) else " || "
            ).join(values) + ")"
        elif isinstance(node, ast.IfExp):
            _, c = self.expr(node.test, assigned)
            left_type, left_src = self.expr(node.body, assigned)
            right_type, right_src = self.expr(node.orelse, assigned)
            if "float" in (left_type, right_type):
                left_type, left_src, right_src = (
                    "float",
                    f"float({left_src})",
                    f"float({right_src})",
                )
            else:
                _ensure(left_type == right_type, "incompatible conditional result")
            return left_type, f"({c} ? {left_src} : {right_src})"
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and not node.keywords
        ):
            name = node.func.id
            arguments = [self.expr(arg, assigned) for arg in node.args]
            if name in ("A", "B") and len(arguments) == 1 and arguments[0][0] == "uint":
                base = name.lower()
                return "float", f"rd({base},s.n{base},{arguments[0][1]},bad)"
            count = {
                "min": 2,
                "max": 2,
                "abs": 1,
                "exp": 1,
                "sqrt": 1,
                "rsqrt": 1,
                "fma": 3,
                "float": 1,
            }.get(name)
            if count == len(arguments):
                if name == "float":
                    return "float", f"float({arguments[0][1]})"
                t = (
                    "uint"
                    if name in ("min", "max") and all(t == "uint" for t, _ in arguments)
                    else "float"
                )
                vals = ", ".join(f"{t}({v})" for _, v in arguments)
                return t, f"{name}({vals})"
        raise KernelRejected("unsupported expression or operand type")

    def arithmetic(self, op, left, right):
        left_type, left_src = left
        right_type, right_src = right
        _ensure("bool" not in (left_type, right_type), "numeric operands required")
        optype = type(op)
        if optype in (ast.FloorDiv, ast.Mod):
            _ensure(
                left_type == right_type == "uint",
                "integer quotient and remainder require index operands",
            )
            return (
                "uint",
                f"{'divu' if optype is ast.FloorDiv else 'modu'}({left_src},{right_src},bad)",
            )
        if optype is ast.Div:
            return "float", f"(float({left_src}) / float({right_src}))"
        _ensure(optype in OPS, "unsupported binary operator")
        if optype in (ast.BitAnd, ast.BitOr, ast.BitXor, ast.LShift, ast.RShift):
            _ensure(
                left_type == right_type == "uint", "bitwise operands must be indices"
            )
            if optype in (ast.LShift, ast.RShift):
                right_src = f"({right_src} & 31u)"
        if "float" in (left_type, right_type):
            return "float", f"(float({left_src}) {OPS[optype]} float({right_src}))"
        return "uint", f"({left_src} {OPS[optype]} {right_src})"

    def target(self, node):
        _ensure(
            isinstance(node, ast.Name)
            and re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,39}", node.id)
            and node.id not in INTEGERS | self.iterators,
            "invalid or reserved variable",
        )
        return node.id

    def loop_bound(self, bound, assigned):
        if (
            isinstance(bound, ast.Name)
            and bound.id in assigned
            and bound.id in self.bounds
        ):
            return _var(bound.id), self.bounds[bound.id]
        _ensure(
            isinstance(bound, ast.Constant)
            and type(bound.value) is int
            and 0 <= bound.value <= MAX_ELEMENTS,
            "loop limits must be a literal, a dimension or an immutable alias",
        )
        return str(bound.value) + "u", bound.value

    def block(self, nodes, assigned, depth):
        _ensure(depth <= 8, "too many nested blocks")
        for st in nodes:
            self.cost += self.weight
            if isinstance(st, (ast.Assign, ast.AugAssign)):
                if isinstance(st, ast.Assign):
                    _ensure(len(st.targets) == 1, "one assignment target required")
                    name = self.target(st.targets[0])
                    t, v = self.expr(st.value, assigned)
                else:
                    name = self.target(st.target)
                    _ensure(name in assigned, "uninitialized augmented target")
                    t, v = self.arithmetic(
                        st.op,
                        self.expr(st.target, assigned),
                        self.expr(st.value, assigned),
                    )
                # An integer literal promotes to float in a register first initialized as a float.
                if name == "out":
                    t, v = "float", f"float({v})"
                elif name in self.types and self.types[name] == "float" and t == "uint":
                    t, v = "float", f"float({v})"
                _ensure(
                    self.types.setdefault(name, t) == t, "a register has one fixed type"
                )
                _ensure(len(self.types) <= 96, "too many registers")
                # Only single-write aliases of invariant values can bound another loop.
                if (
                    t == "uint"
                    and self.writes[name] == 1
                    and isinstance(st, ast.Assign)
                ):
                    value = st.value
                    if isinstance(value, ast.Name) and value.id in self.bounds:
                        self.bounds[name] = self.bounds[value.id]
                    elif (
                        isinstance(value, ast.Constant)
                        and type(value.value) is int
                        and 0 <= value.value <= MAX_ELEMENTS
                    ):
                        self.bounds[name] = value.value
                self.lines.append(f"{_var(name)} = {v};")
                assigned.add(name)
            elif isinstance(st, ast.If):
                _, test = self.expr(st.test, assigned)
                self.lines.append(f"if ({test}) {{")
                then = self.block(st.body, assigned.copy(), depth + 1)
                self.lines.append("} else {")
                other = self.block(st.orelse, assigned.copy(), depth + 1)
                self.lines.append("}")
                assigned |= then & other
            elif isinstance(st, ast.For):
                iterator = self.target(st.target)
                _ensure(
                    iterator not in self.types
                    or (iterator not in assigned and self.types[iterator] == "uint"),
                    "loop index must be fresh",
                )
                call = st.iter
                _ensure(
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Name)
                    and call.func.id == "range"
                    and not call.keywords
                    and not st.orelse
                    and 1 <= len(call.args) <= 3,
                    "expected bounded range",
                )
                start = (
                    ("0u", 0)
                    if len(call.args) == 1
                    else self.loop_bound(call.args[0], assigned)
                )
                stop = self.loop_bound(
                    call.args[0] if len(call.args) == 1 else call.args[1], assigned
                )
                step = (
                    self.loop_bound(call.args[2], assigned)
                    if len(call.args) == 3
                    else ("1u", 1)
                )
                _ensure(
                    step[1] > 0
                    and (len(call.args) < 3 or isinstance(call.args[2], ast.Constant)),
                    "positive literal loop stride required",
                )
                self.types[iterator] = "uint"
                self.iterators.add(iterator)
                name = _var(iterator)
                self.lines.append(
                    f"for ({name}={start[0]}; {name}<{stop[0]}; {name}+={step[0]}) {{"
                )
                prev = self.weight
                self.weight *= max(1, len(range(start[1], stop[1], step[1])))
                body_assigned = self.block(st.body, assigned | {iterator}, depth + 1)
                if start[1] < stop[1]:
                    assigned |= body_assigned - {iterator}
                self.weight = prev
                self.lines.append("}")
                self.iterators.remove(iterator)
            else:
                raise KernelRejected("unsupported statement")
        return assigned


def lower(program, shapes):
    _ensure(
        isinstance(program, dict)
        and set(program) == {"source", "threads", "outputs_per_thread"},
        "kernel configuration fields",
    )
    _ensure(
        type(program["threads"]) is int and program["threads"] in THREADS,
        "threadgroup size",
    )
    lanes = program["outputs_per_thread"]
    _ensure(type(lanes) is int and lanes in LANES, "outputs_per_thread")
    text = program["source"]
    _ensure(
        type(text) is str
        and 0 < len(text) <= 8192
        and all(line.partition("#")[0].isascii() for line in text.splitlines()),
        "kernel code must be bounded ASCII; comments may use Unicode",
    )
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, RecursionError) as exc:
        raise KernelRejected("invalid kernel syntax") from exc
    _ensure(len(list(ast.walk(tree))) <= 800, "too many source nodes")
    _ensure(
        isinstance(shapes, (list, tuple)) and bool(shapes),
        "at least one shape is required",
    )
    writes = Counter(
        n.id
        for n in ast.walk(tree)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
    )
    for shape in shapes:
        compiled = _Compiler(tree, shape, writes)
    regs = "\n".join(
        f"    {t} {_var(n)};" for n, t in compiled.types.items() if n not in INTEGERS
    )
    return (
        PRELUDE
        + f"    for (uint lane=0u; lane<{lanes}u; ++lane) {{\n"
        + f"    uint i=gid*{lanes}u+lane; if (i>=L) break;\n"
        + regs
        + "\n"
        + "\n".join("    " + line for line in compiled.lines)
        + "\n    output[i]=v_out;\n    }\n}\n"
    )


def _case_shapes(cases):
    if not isinstance(cases, (tuple, list)) or not 1 <= len(cases) <= 64:
        raise BackendUnavailable("invalid case inventory")
    sizes = []
    total = 0
    for c in cases:
        if (
            not isinstance(c, dict)
            or not {"shape", "a", "b", "expected", "atol", "rtol"} <= c.keys()
        ):
            raise BackendUnavailable("invalid case data")
        dims = c["shape"]
        if (
            not isinstance(dims, (list, tuple))
            or len(dims) != 3
            or not all(type(v) is int and 0 < v <= MAX_ELEMENTS for v in dims)
        ):
            raise BackendUnavailable("invalid shape")
        for name, lim in (
            ("a", MAX_ELEMENTS),
            ("b", MAX_ELEMENTS),
            ("expected", MAX_OUTPUT),
        ):
            minimum = 1 if name == "expected" else 0
            if (
                not isinstance(c[name], (list, tuple, array.array))
                or not minimum <= len(c[name]) <= lim
            ):
                raise BackendUnavailable("invalid buffer length")
            total += len(c[name]) * (8 if name == "expected" else 4)
        if total > 256 * (1 << 20):
            raise BackendUnavailable("suite exceeds memory bound")
        if type(c.get("timing", True)) is not bool or not all(
            type(c[k]) in (float, int) and 0 <= c[k] <= 1 for k in ("atol", "rtol")
        ):
            raise BackendUnavailable("invalid measurement policy")
        sizes.append((*dims, len(c["expected"])))
    return sizes


@contextmanager
def _device_slot():
    path = Path("/tmp") / f"weakbench-metal-{os.geteuid()}"
    path.mkdir(mode=0o700, exist_ok=True)
    info = path.lstat()
    if (
        not path.is_dir()
        or path.is_symlink()
        or info.st_uid != os.geteuid()
        or info.st_mode & 0o077
    ):
        raise BackendUnavailable("private backend cache required")
    fd = os.open(path / "device.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "rb+"):
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield path


def _host(path):
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise BackendUnavailable("this backend requires an Apple silicon GPU and macOS")
    try:
        tool = subprocess.run(
            ["/usr/bin/xcrun", "--find", "clang"],
            capture_output=True,
            timeout=10,
            check=True,
            text=True,
        ).stdout.strip()
        version = subprocess.run(
            [tool, "--version"], capture_output=True, timeout=10, check=True
        ).stdout
        src = HERE / "kernel_metal_host.m"
        key = hashlib.sha256(
            src.read_bytes() + version + platform.release().encode()
        ).hexdigest()[:24]
        dest = path / ("host-" + key)
        if not dest.is_file():
            with tempfile.TemporaryDirectory(dir=path) as tmp:
                host = Path(tmp) / "host"
                result = subprocess.run(
                    [
                        "/usr/bin/xcrun",
                        "--sdk",
                        "macosx",
                        "clang",
                        "-fobjc-arc",
                        "-Wno-deprecated-declarations",
                        "-O2",
                        "-framework",
                        "Foundation",
                        "-framework",
                        "Metal",
                        str(src),
                        "-o",
                        str(host),
                    ],
                    capture_output=True,
                    timeout=60,
                )
                if result.returncode:
                    raise BackendUnavailable("trusted Metal host compiler failed")
                os.replace(host, dest)
        return dest, version.decode().splitlines()[0]
    except (OSError, subprocess.SubprocessError) as e:
        raise BackendUnavailable("trusted Metal toolchain unavailable") from e


def _buffers(path, cases):
    metadata = []
    for i, c in enumerate(cases):
        metadata.append(
            {k: c[k] for k in ("shape", "atol", "rtol")}
            | {"timing": c.get("timing", True)}
        )
        case_hash = hashlib.sha256(json.dumps(metadata[-1], sort_keys=True).encode())
        for name, typecode in (("a", "f"), ("b", "f"), ("expected", "d")):
            try:
                values = array.array(typecode, c[name])
                if not all(map(math.isfinite, values)):
                    raise BackendUnavailable(
                        "GPU inputs and CPU reference values must be finite"
                    )
                if sys.byteorder != "little":
                    values.byteswap()
                data = values.tobytes()
                case_hash.update(len(data).to_bytes(8, "little"))
                case_hash.update(data)
                (path / f"{i}-{name}").write_bytes(data)
            except (OverflowError, TypeError, ValueError) as exc:
                raise BackendUnavailable("invalid case buffer") from exc
        metadata[-1]["case_hash"] = case_hash.hexdigest()
    return metadata


def _validate_host_result(result, shapes, cases, orders):
    def require(condition):
        if not condition:
            raise BackendUnavailable("inconsistent Metal host report")

    def finite(value):
        return type(value) in (int, float) and math.isfinite(value)

    try:
        require(
            isinstance(result, dict)
            and result["backend"] == "metal"
            and result["status"] in ("correct", "incorrect")
        )
        require(isinstance(result["device"], str) and result["device"].strip())
        require(
            all(
                type(result[k]) is int and result[k] > 0
                for k in ("registry_id", "simd_width")
            )
        )
        rows = result["cases"]
        require(isinstance(rows, list) and len(rows) == len(shapes))
        passed = []
        for row, (m, n, k, length), case, order in zip(
            rows, shapes, cases, orders, strict=True
        ):
            timed = result["status"] == "correct" and case.get("timing", True)
            keys = {"shape", "elements", "validation"}
            if timed:
                keys |= {"rounds", "dispatches_per_sample"}
            require(
                isinstance(row, dict)
                and row.keys() == keys
                and row["shape"] == [m, n, k]
                and all(type(x) is int for x in row["shape"])
                and type(row["elements"]) is int
                and row["elements"] == length
            )
            check = row["validation"]
            require(
                type(check["passed"]) is bool
                and type(check["invalid_operation"]) is bool
                and type(check["mismatches"]) is int
                and type(check["nonfinite"]) is int
                and 0 <= check["nonfinite"] <= check["mismatches"] <= length
                and finite(check["max_abs_error"])
                and check["max_abs_error"] >= 0
            )
            require(
                check["passed"]
                == (not check["invalid_operation"] and check["mismatches"] == 0)
            )
            passed.append(check["passed"])
            if timed:
                rounds = row["rounds"]
                require(
                    type(row["dispatches_per_sample"]) is int
                    and 1 <= row["dispatches_per_sample"] <= 64
                    and isinstance(rounds, list)
                    and len(rounds) == len(order) == 7
                )
                for pair, first in zip(rounds, order):
                    require(pair["baseline_first"] is (first == 0))
                    require(
                        all(
                            finite(pair[k]) and pair[k] > 0
                            for k in ("baseline_us", "candidate_us")
                        )
                    )
                    ratio = pair["baseline_us"] / pair["candidate_us"]
                    require(math.isfinite(ratio) and ratio > 0)
        require(all(passed) == (result["status"] == "correct"))
    except (KeyError, TypeError, OverflowError, ValueError) as exc:
        raise BackendUnavailable("inconsistent Metal host report") from exc


def measure(candidate, baseline, cases, *, seed=0):
    """Validate all shapes before benchmarking. Device/host/reference failures raise BackendUnavailable."""
    import random

    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise BackendUnavailable("this backend requires an Apple silicon GPU and macOS")
    if type(seed) is not int or not 0 <= seed <= 0xFFFFFFFF:
        raise ValueError("seed must be a uint32")
    shapes = _case_shapes(cases)
    try:
        ref = lower(baseline, shapes)
    except KernelRejected as e:
        raise BackendUnavailable("reference outside the kernel contract") from e
    try:
        generated = lower(candidate, shapes)
    except KernelRejected as e:
        return {
            "status": "invalid_candidate",
            "backend": "metal",
            "protocol": VERSION,
            "seed": seed,
            "reason": str(e),
            "cases": [],
        }
    rng = random.Random(seed)
    orders = []
    for _ in cases:
        first = [0, 1] * 3 + [rng.randrange(2)]
        rng.shuffle(first)
        orders.append(first)
    try:
        with _device_slot() as cache:
            binary, clang = _host(cache)
            with tempfile.TemporaryDirectory(dir=cache) as folder:
                path = Path(folder)
                (path / "baseline.metal").write_text(ref)
                (path / "candidate.metal").write_text(generated)
                cfg = {
                    "programs": [
                        {k: p[k] for k in ("threads", "outputs_per_thread")}
                        for p in (baseline, candidate)
                    ],
                    "cases": _buffers(path, cases),
                    "order": orders,
                }
                (path / "run.json").write_text(json.dumps(cfg))
                command = subprocess.run(
                    [str(binary)], cwd=path, capture_output=True, timeout=120
                )
                if command.returncode:
                    code = command.stderr.decode(errors="replace").strip()
                    if code not in (
                        "baseline_disagrees_with_oracle",
                        "apple_gpu_unavailable",
                        "hardware_timestamps_unavailable",
                        "threadgroup_unsupported",
                        "metal_allocation",
                        "invalid_buffer_length",
                        "metal_command_failed",
                        "metal_library_compile_failed",
                        "metal_pipeline_compile_failed",
                    ):
                        code = "metal_execution_failed"
                    raise BackendUnavailable(code)
                result = json.loads(command.stdout)
    except (OSError, ValueError, subprocess.SubprocessError) as e:
        raise BackendUnavailable("isolated Metal host execution failed") from e
    _validate_host_result(result, shapes, cases, orders)
    for row, metadata in zip(result["cases"], cfg["cases"], strict=True):
        row["case_hash"] = metadata["case_hash"]
        if "rounds" not in row:
            continue
        rounds = row["rounds"]
        ratios = [v["baseline_us"] / v["candidate_us"] for v in rounds]
        lo, hi = min(ratios), max(ratios)
        row["timing"] = {
            "baseline_us": statistics.median(v["baseline_us"] for v in rounds),
            "candidate_us": statistics.median(v["candidate_us"] for v in rounds),
            "speedup": statistics.median(ratios),
            "ratio_min": lo,
            "ratio_max": hi,
            "comparison": "faster" if lo > 1 else "slower" if hi < 1 else "overlap",
        }
    result.update(
        protocol=VERSION,
        seed=seed,
        compiler=clang,
        os=platform.mac_ver()[0],
        timing_source="Metal command-buffer GPU timestamps, per dispatch",
        os_build=platform.version(),
        baseline_hash=hashlib.sha256(
            f"{baseline['threads']}\n{ref}".encode()
        ).hexdigest(),
        candidate_hash=hashlib.sha256(
            f"{candidate['threads']}\n{generated}".encode()
        ).hexdigest(),
    )
    return result
