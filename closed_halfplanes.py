"""Check rational witnesses and linear-combination certificates for planar inequalities."""

from fractions import Fraction
import re

DETAIL_KEYS = (
    "valid_json",
    "certificate_form",
    "point_feasible",
    "bound_violated",
    "nonnegative_weights",
    "combination_proves_claim",
    "combination_proves_conflict",
)


def validate_expected(spec):
    if not isinstance(spec, dict) or set(spec) != {"rows", "claim"}:
        return False
    rows = spec["rows"]
    if not isinstance(rows, list) or not 2 <= len(rows) <= 24:
        return False
    return all(
        isinstance(row, list)
        and len(row) == 3
        and all(type(v) is int and abs(v) <= 10000 for v in row)
        and any(row[:2])
        for row in [*rows, spec["claim"]]
    )


def rational(v):
    if type(v) is int and v.bit_length() <= 1024:
        return Fraction(v)
    if (
        isinstance(v, str)
        and len(v) <= 320
        and re.fullmatch(r"-?[0-9]+(?:/[0-9]+)?", v)
    ):
        try:
            return Fraction(v)
        except (ValueError, ZeroDivisionError):
            pass
    raise ValueError("invalid rational")


def vector(values, size):
    if not isinstance(values, list) or len(values) != size:
        raise ValueError("invalid vector")
    return [rational(v) for v in values]


def score(item, reply):
    from hidden import strict_json

    # Publish branch-specific checks only for the certificate the candidate attempted.
    checks = {"valid_json": 0.0, "certificate_form": 0.0}
    ok = False
    try:
        answer = strict_json(reply.get("content"))
        checks["valid_json"] = 1.0
        rows, claim = item["expected"]["rows"], item["expected"]["claim"]
        if not isinstance(answer, dict):
            raise ValueError
        status = answer.get("status")
        fields = {
            "entailed": {"status", "point", "weights"},
            "refuted": {"status", "point"},
            "inconsistent": {"status", "weights"},
        }
        if (
            not isinstance(status, str)
            or status not in fields
            or set(answer) != fields[status]
        ):
            raise ValueError
        point = vector(answer["point"], 2) if "point" in answer else None
        weights = vector(answer["weights"], len(rows)) if "weights" in answer else None
        checks["certificate_form"] = 1.0
        if point is not None:
            x, y = point
            checks["point_feasible"] = float(
                all(a * x + b * y <= c for a, b, c in rows)
            )
            if status == "refuted":
                a, b, c = claim
                checks["bound_violated"] = float(a * x + b * y > c)
        if weights is not None:
            checks["nonnegative_weights"] = float(all(w >= 0 for w in weights))
            sums = [sum(w * v for w, v in zip(weights, axis)) for axis in zip(*rows)]
            if status == "entailed":
                checks["combination_proves_claim"] = float(
                    sums[:2] == claim[:2] and sums[2] <= claim[2]
                )
            else:
                checks["combination_proves_conflict"] = float(
                    sums[:2] == [0, 0] and sums[2] < 0
                )
        ok = all(checks.values())
    except (KeyError, ValueError, TypeError, RecursionError):
        pass
    return {"correct": ok, "score": float(ok), "detail": checks}
