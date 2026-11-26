"""Evaluate scheduled compare-exchange programs on every binary threshold input."""
from functools import lru_cache


DETAIL_KEYS = frozenset((
    "valid_json", "fabric_program", "fabric_wiring", "fabric_budget",
    "fabric_input_coverage", "fabric_all_inputs",
))


def integer(value, low, high):
    return type(value) is int and low <= value <= high


def instruction(value, wires):
    return (isinstance(value, list) and len(value) == 2
            and all(integer(port, 0, wires - 1) for port in value)
            and value[0] != value[1])


def validate_expected(spec):
    if not isinstance(spec, dict) or set(spec) != {"wires", "ticks", "budget", "targets"}:
        return False
    wires, ticks = spec["wires"], spec["ticks"]
    if not integer(wires, 2, 8) or not isinstance(ticks, list) or not 1 <= len(ticks) <= 12:
        return False
    if not integer(spec["budget"], 0, len(ticks) * (wires // 2)):
        return False
    for choices in ticks:
        if not isinstance(choices, list) or len(choices) > wires * (wires-1) // 2:
            return False
        if not all(instruction(edge, wires) and edge[0] < edge[1] for edge in choices):
            return False
        if len(set(map(tuple, choices))) != len(choices):
            return False
    targets = spec["targets"]
    if not isinstance(targets, list) or not 1 <= len(targets) <= wires:
        return False
    if not all(isinstance(t, list) and len(t) == 2 and integer(t[0], 0, wires-1)
               and integer(t[1], 1, wires) for t in targets):
        return False
    return len({port for port, _ in targets}) == len({rank for _, rank in targets}) == len(targets)


def parse_program(program, wires, ticks):
    if not isinstance(program, dict) or set(program) != {"stages"}:
        return False
    stages = program["stages"]
    return (isinstance(stages, list) and len(stages) == ticks
            and all(isinstance(s, list) and len(s) <= wires // 2
                    and all(instruction(op, wires) for op in s) for s in stages))


def schedulable(stages, ticks):
    for ops, edges in zip(stages, ticks):
        used = [port for op in ops for port in op]
        if len(used) != len(set(used)) or any(sorted(op) not in edges for op in ops):
            return False
    return True


@lru_cache(maxsize=7)
def threshold_vectors(wires):
    patterns = 1 << wires
    inputs = tuple(sum(1 << p for p in range(patterns) if (p >> i) & 1)
                   for i in range(wires))
    ranks = tuple(sum(1 << p for p in range(patterns) if p.bit_count() >= wires-r+1)
                  for r in range(1, wires+1))
    return inputs, ranks


def passing_inputs(wires, stages, targets):
    inputs, ranks = threshold_vectors(wires)
    lanes = list(inputs)
    for stage in stages:
        for lo, hi in stage:
            a, b = lanes[lo], lanes[hi]
            lanes[lo], lanes[hi] = a & b, a | b
    errors = 0
    for port, rank in targets:
        errors |= lanes[port] ^ ranks[rank-1]
    return (1 << wires) - errors.bit_count()


def score(item, reply):
    from hidden import strict_json

    spec = item["expected"]
    detail = dict.fromkeys(DETAIL_KEYS, 0.0)
    try:
        content = reply["content"]
        if not isinstance(content, str) or len(content) > 16384:
            raise ValueError("oversize or missing source")
        program = strict_json(content)
    except (ValueError, TypeError, KeyError, RecursionError):
        return {"correct": False, "score": 0.0, "detail": detail}
    detail["valid_json"] = 1.0
    wires, ticks = spec["wires"], spec["ticks"]
    if parse_program(program, wires, len(ticks)):
        detail["fabric_program"] = 1.0
        stages = program["stages"]
        wired = schedulable(stages, ticks)
        funded = sum(map(len, stages)) <= spec["budget"]
        detail.update(fabric_wiring=float(wired), fabric_budget=float(funded))
        if wired and funded:
            count = passing_inputs(wires, stages, spec["targets"])
            detail["fabric_input_coverage"] = count / (1 << wires)
            detail["fabric_all_inputs"] = float(count == 1 << wires)
    correct = bool(detail["fabric_all_inputs"])
    return {"correct": correct, "score": float(correct), "detail": detail}
