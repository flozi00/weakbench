"""A strict, representation-invariant scorer for a three-state fleet estimate."""

from fractions import Fraction
import re

FIELDS = ("risk", "operating", "jam", "signal")
DETAIL_KEYS = (
    "valid_json",
    "incidence_schema",
    "risk_sets_correct",
    "operating_correct",
    "jam_incidence_correct",
    "signal_incidence_correct",
    "incidence_mass_balanced",
)


def validate_expected(expected):
    if not isinstance(expected, dict) or set(expected) != {"fleet", "times"}:
        return False
    fleet, times = expected["fleet"], expected["times"]
    if not isinstance(fleet, list) or not 1 <= len(fleet) <= 32:
        return False
    if (
        not isinstance(times, list)
        or len(times) != 2
        or not all(type(x) is int for x in times)
        or not 0 < times[0] < times[1] <= 1000
    ):
        return False
    seen = set()
    for record in fleet:
        if not isinstance(record, list) or len(record) != 4:
            return False
        unit, start, end, kind = record
        if (
            not isinstance(unit, str)
            or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,23}", unit)
            or unit in seen
        ):
            return False
        if (
            type(start) is not int
            or type(end) is not int
            or not 0 <= start < end <= 1000
        ):
            return False
        if kind not in ("jam", "signal", "censor"):
            return False
        seen.add(unit)
    return min(r[1] for r in fleet) == 0 and times[-1] <= max(r[2] for r in fleet)


def estimate(fleet, times):
    # Left limits determine exposure; the row vector is measured after the step.
    p = (Fraction(1), Fraction(0), Fraction(0))
    result = {k: [] for k in FIELDS}
    failure_times = {end for _, _, end, kind in fleet if kind != "censor"}
    for t in sorted(failure_times | set(times)):
        exposed = sum(start < t <= end for _, start, end, _ in fleet)
        if t in failure_times:
            jam = Fraction(
                sum(end == t and kind == "jam" for _, _, end, kind in fleet), exposed
            )
            signal = Fraction(
                sum(end == t and kind == "signal" for _, _, end, kind in fleet), exposed
            )
            step = ((1 - jam - signal, jam, signal), (0, 1, 0), (0, 0, 1))
            p = tuple(sum(p[i] * step[i][j] for i in range(3)) for j in range(3))
        if t in times:
            result["risk"].append(exposed)
            for k, value in zip(FIELDS[1:], p):
                result[k].append(value)
    return result


def probability(value):
    if type(value) is int:
        return Fraction(value) if value in (0, 1) else None
    if isinstance(value, str) and value in ("0", "1"):
        return Fraction(int(value))
    if not isinstance(value, str):
        return None
    m = re.fullmatch(r"([0-9]+)/([0-9]+)", value)
    if not m:
        return None
    try:
        a, b = map(int, m.groups())
    except ValueError:
        return None
    return Fraction(a, b) if b and 0 <= a <= b else None


def score(item, reply):
    from hidden import strict_json

    try:
        answer = strict_json(reply.get("content"))
        parsed = True
    except (TypeError, ValueError, RecursionError):
        answer, parsed = None, False
    detail = dict.fromkeys(DETAIL_KEYS, 0.0)
    detail["valid_json"] = float(parsed)
    if (
        not isinstance(answer, dict)
        or set(answer) != set(FIELDS)
        or not all(isinstance(row, list) and len(row) == 2 for row in answer.values())
    ):
        return {"correct": False, "score": 0.0, "detail": detail}

    given = {k: [probability(x) for x in answer[k]] for k in FIELDS[1:]}
    given["risk"] = [x if type(x) is int and x >= 0 else None for x in answer["risk"]]
    detail["incidence_schema"] = float(
        all(v is not None for sequence in given.values() for v in sequence)
    )
    correct_values = estimate(item["expected"]["fleet"], item["expected"]["times"])
    for field, metric in zip(FIELDS, DETAIL_KEYS[2:6]):
        detail[metric] = (
            sum(a == b for a, b in zip(given[field], correct_values[field])) / 2
        )
    vectors = zip(*(given[k] for k in FIELDS[1:]))
    detail["incidence_mass_balanced"] = (
        sum(None not in vector and sum(vector) == 1 for vector in vectors) / 2
    )
    correct = all(detail[key] == 1 for key in DETAIL_KEYS)
    return {"correct": correct, "score": float(correct), "detail": detail}
