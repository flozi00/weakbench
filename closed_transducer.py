"""Grade a total deterministic letter-to-letter transducer by exact finite product search."""

from collections import deque

DETAIL_KEYS = frozenset(
    (
        "valid_json",
        "transducer_schema",
        "transducer_within_budget",
        "transducer_total",
        "transducer_equivalent",
    )
)


def name(x):
    return isinstance(x, str) and 1 <= len(x) <= 40


def alphabet(values):
    return (
        isinstance(values, list)
        and 1 <= len(values) <= 8
        and all(name(v) for v in values)
        and len(set(values)) == len(values)
    )


def structure(machine):
    return (
        isinstance(machine, dict)
        and set(machine) == {"start", "states"}
        and name(machine["start"])
        and isinstance(machine["states"], dict)
    )


def total(machine, inputs, outputs):
    states = machine["states"]
    if machine["start"] not in states or not all(name(q) for q in states):
        return False
    mask = set(inputs)
    for table in states.values():
        if not isinstance(table, dict) or table.keys() != mask:
            return False
        for entry in table.values():
            if (
                not isinstance(entry, list)
                or len(entry) != 2
                or not name(entry[0])
                or entry[0] not in states
                or not isinstance(entry[1], str)
                or entry[1] not in outputs
            ):
                return False
    return True


def validate_expected(expected):
    if not isinstance(expected, dict) or set(expected) != {"inputs", "outputs", "max_states", "reference"}:
        return False
    limit, machine = expected["max_states"], expected["reference"]
    return (
        alphabet(expected["inputs"])
        and alphabet(expected["outputs"])
        and type(limit) is int
        and 1 <= limit <= 16
        and structure(machine)
        and 1 <= len(machine["states"]) <= limit
        and total(machine, expected["inputs"], expected["outputs"])
    )


def equivalent(left, right, inputs):
    """Search only reachable pairs. A disagreement on any edge distinguishes a finite input word."""
    start = left["start"], right["start"]
    queue, visited = deque([start]), {start}
    while queue:
        a, b = queue.popleft()
        for token in inputs:
            na, oa = left["states"][a][token]
            nb, ob = right["states"][b][token]
            if oa != ob:
                return False
            pair = na, nb
            if pair not in visited:
                visited.add(pair)
                queue.append(pair)
    return True


def score(item, reply):
    from hidden import strict_json

    expected = item["expected"]
    try:
        machine = strict_json(reply["content"])
        parsed = True
    except (ValueError, TypeError, KeyError, RecursionError):
        machine, parsed = None, False
    shape = structure(machine)
    bounded = shape and 1 <= len(machine["states"]) <= expected["max_states"]
    complete = bounded and total(machine, expected["inputs"], expected["outputs"])
    correct = complete and equivalent(machine, expected["reference"], expected["inputs"])
    detail = {
        "valid_json": float(parsed),
        "transducer_schema": float(shape),
        "transducer_within_budget": float(bounded),
        "transducer_total": float(complete),
        "transducer_equivalent": float(correct),
    }
    return {"correct": correct, "score": float(correct), "detail": detail}
