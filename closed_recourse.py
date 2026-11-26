"""Verify identified gate mechanisms and minimum-cost interventions across a finite environment set."""

from itertools import product
import re

DETAIL_KEYS = (
    "valid_json",
    "recourse_mechanisms",
    "recourse_feasibility",
    "recourse_legal",
    "recourse_coverage",
    "recourse_cost_report",
    "recourse_optimal",
)


def _label(v):
    return (
        isinstance(v, str) and re.fullmatch(r"[A-Za-z][A-Za-z0-9]{0,11}", v) is not None
    )


def _bits(v, length):
    return isinstance(v, str) and len(v) == length and set(v) <= {"0", "1"}


def _assigned(value, labels):
    return isinstance(value, dict) and all(
        k in labels and type(v) is int and v in (0, 1) for k, v in value.items()
    )


def mechanisms(spec):
    """Recover each module's equation from its unpinned displays."""
    env = spec["environment"]
    names = [entry[0] for entry in spec["modules"]]
    possible = {name: set(choices) for name, _, _, choices in spec["modules"]}
    for setting, pinned, display in spec["calibration"]:
        values = dict(zip([*env, *names], map(int, setting + display)))
        for name, a, b, _ in spec["modules"]:
            if name in pinned:
                if pinned[name] != values[name]:
                    return None
            else:
                index = 2 * values[a] + values[b]
                possible[name] = {
                    op for op in possible[name] if int(op[index]) == values[name]
                }
    if any(len(options) != 1 for options in possible.values()):
        return None
    return {name: next(iter(options)) for name, options in possible.items()}


def validate_expected(spec):
    keys = {"environment", "modules", "calibration", "settings", "controls", "goal"}
    if not isinstance(spec, dict) or set(spec) != keys:
        return False
    env, nodes = spec["environment"], spec["modules"]
    if (
        not isinstance(env, list)
        or not 1 <= len(env) <= 3
        or not all(map(_label, env))
        or len(set(env)) != len(env)
    ):
        return False
    if not isinstance(nodes, list) or not 2 <= len(nodes) <= 7:
        return False
    names = set(env)
    for wire in nodes:
        if not isinstance(wire, list) or len(wire) != 4:
            return False
        name, a, b, opts = wire
        if (
            not all(map(_label, (name, a, b)))
            or name in names
            or a not in names
            or b not in names
            or a == b
        ):
            return False
        if (
            not isinstance(opts, list)
            or not 2 <= len(opts) <= 8
            or not all(_bits(s, 4) for s in opts)
            or len(set(opts)) != len(opts)
        ):
            return False
        names.add(name)
    names -= set(env)
    calibration, settings, controls = (
        spec["calibration"],
        spec["settings"],
        spec["controls"],
    )
    if not isinstance(calibration, list) or not 1 <= len(calibration) <= 24:
        return False
    for line in calibration:
        if not isinstance(line, list) or len(line) != 3:
            return False
        setting, pins, shown = line
        if (
            not _bits(setting, len(env))
            or not _assigned(pins, names)
            or not _bits(shown, len(nodes))
        ):
            return False
    if (
        not isinstance(settings, list)
        or not 1 <= len(settings) <= 2 ** len(env)
        or not all(_bits(s, len(env)) for s in settings)
        or len(set(settings)) != len(settings)
    ):
        return False
    if not isinstance(controls, list) or not 1 <= len(controls) <= 5:
        return False
    actuators = set()
    for ctrl in controls:
        if not isinstance(ctrl, list) or len(ctrl) != 3:
            return False
        name, *prices = ctrl
        if (
            not _label(name)
            or name not in names
            or name in actuators
            or prices == [None, None]
        ):
            return False
        if any(
            c is not None and (type(c) is not int or not 1 <= c <= 100) for c in prices
        ):
            return False
        actuators.add(name)
    return (
        _assigned(spec["goal"], names)
        and bool(spec["goal"])
        and mechanisms(spec) is not None
    )


def coverage(spec, operations, pins):
    """Represent each possible environment by one bit and propagate all displays at once."""
    settings = spec["settings"]
    full = (1 << len(settings)) - 1
    signal = {
        name: sum(1 << j for j, row in enumerate(settings) if row[i] == "1")
        for i, name in enumerate(spec["environment"])
    }
    for name, a, b, _ in spec["modules"]:
        if name in pins:
            signal[name] = full if pins[name] else 0
            continue
        a, b = signal[a], signal[b]
        masks = (~a & ~b, ~a & b, a & ~b, a & b)
        value = 0
        for char, mask in zip(operations[name], masks):
            if char == "1":
                value |= mask
        signal[name] = value & full
    success = full
    for name, bit in spec["goal"].items():
        success &= signal[name] if bit else ~signal[name]
    return success.bit_count()


def candidates(controls):
    options = [
        [
            (None, 0),
            *((value, cost) for value, cost in enumerate(prices) if cost is not None),
        ]
        for _, *prices in controls
    ]
    for assignments in product(*options):
        yield (
            {
                ctrl[0]: value
                for ctrl, (value, _) in zip(controls, assignments)
                if value is not None
            },
            sum(cost for _, cost in assignments),
        )


def optimum(spec, ops):
    best = None
    for action, cost in candidates(spec["controls"]):
        if (best is None or cost < best) and coverage(spec, ops, action) == len(
            spec["settings"]
        ):
            best = cost
    return best


def score(item, reply):
    from hidden import strict_json

    spec = item["expected"]
    ops = mechanisms(spec)
    best = optimum(spec, ops)
    detail = {k: 0.0 for k in DETAIL_KEYS if best is not None or k in DETAIL_KEYS[:3]}
    try:
        text = reply.get("content")
        if not isinstance(text, str) or len(text) > 8192:
            raise ValueError
        ans = strict_json(text)
        detail["valid_json"] = 1.0
    except (ValueError, TypeError, RecursionError):
        return {"correct": False, "score": 0.0, "detail": detail}
    ok = False
    if isinstance(ans, dict) and set(ans) == {"mechanisms", "cost", "pins"}:
        opmap, cost, pins = ans["mechanisms"], ans["cost"], ans["pins"]
        if isinstance(opmap, dict) and set(opmap) == set(ops):
            detail["recourse_mechanisms"] = sum(
                opmap[n] == s for n, s in ops.items()
            ) / len(ops)
        impossible = cost is None and pins is None
        shaped = type(cost) is int and cost >= 0 and isinstance(pins, dict)
        detail["recourse_feasibility"] = float(impossible if best is None else shaped)
        if best is None:
            ok = impossible and detail["recourse_mechanisms"] == 1
        elif shaped:
            tariffs = {name: rates for name, *rates in spec["controls"]}
            legal = _assigned(pins, tariffs) and all(
                tariffs[n][v] is not None for n, v in pins.items()
            )
            detail["recourse_legal"] = float(legal)
            if legal:
                paid = sum(tariffs[n][v] for n, v in pins.items())
                n = coverage(spec, ops, pins)
                detail["recourse_coverage"] = n / len(spec["settings"])
                detail["recourse_cost_report"] = float(cost == paid)
                detail["recourse_optimal"] = float(
                    n == len(spec["settings"]) and paid == best
                )
            ok = all(v == 1 for v in detail.values())
    return {"correct": bool(ok), "score": float(ok), "detail": detail}
