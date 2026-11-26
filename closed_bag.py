"""Interpret a bounded relational query on a complete finite universe of bag databases."""

import base64
import binascii
import struct
from functools import lru_cache
from itertools import combinations_with_replacement, product

COLUMNS = ("L.k", "L.g", "R.k", "R.v")
FUNCTIONS = ("count", "distinct", "sum", "sum0", "min", "max")
DETAIL_KEYS = (
    "valid_json",
    "bag_program",
    "bag_database_accuracy",
    "bag_empty_accuracy",
    "bag_duplicate_accuracy",
    "bag_null_accuracy",
    "bag_unmatched_accuracy",
    "bag_all_databases",
)


def _key(row):
    return tuple((v is not None, v or 0) for v in row)


def _bags(rows):
    return (
        ((),)
        + tuple((r,) for r in rows)
        + tuple(combinations_with_replacement(rows, 2))
    )


@lru_cache(maxsize=1)
def universe():
    left = _bags(tuple(product((None, 0, 1), (0, 1))))
    right = _bags(tuple(product((None, 0, 1), (None, 0, 1))))
    return tuple(product(left, right))


def validate_expected(spec):
    if not isinstance(spec, dict) or set(spec) != {"outputs", "index"}:
        return False
    table, index = spec["outputs"], spec["index"]
    if not isinstance(table, list) or not 1 <= len(table) <= len(universe()):
        return False
    raw_bytes = len(universe()) * struct.calcsize(">H")
    encoded_chars = 4 * ((raw_bytes + 2) // 3)
    if not isinstance(index, str) or len(index) != encoded_chars:
        return False
    try:
        values = base64.b64decode(index, validate=True)
    except (ValueError, binascii.Error):
        return False
    if len(values) != raw_bytes:
        return False
    ids = [i for (i,) in struct.iter_unpack(">H", values)]
    if set(ids) != set(range(len(table))):
        return False
    width, seen = None, set()
    for bag in table:
        if not isinstance(bag, list) or len(bag) > 4:
            return False
        for row in bag:
            if not isinstance(row, list) or not 1 <= len(row) <= 5:
                return False
            if width is None:
                width = len(row)
            if len(row) != width or any(
                v is not None and (type(v) is not int or not 0 <= v <= 4) for v in row
            ):
                return False
        if bag != sorted(bag, key=_key):
            return False
        packed = tuple(map(tuple, bag))
        if packed in seen:
            return False
        seen.add(packed)
    return True


def _term(v):
    return v in COLUMNS if isinstance(v, str) else type(v) is int and v in (0, 1)


def _predicate(p):
    if not isinstance(p, list) or not p:
        return False
    if p[0] in ("known", "missing"):
        return len(p) == 2 and isinstance(p[1], str) and p[1] in COLUMNS
    return len(p) == 3 and p[0] in ("eq", "ne", "same") and all(_term(t) for t in p[1:])


def _aggregate(a):
    return (
        isinstance(a, list)
        and len(a) == 2
        and isinstance(a[1], str)
        and a[0] in FUNCTIONS
        and (a[1] in COLUMNS or (a[0] == "count" and a[1] == "*"))
    )


def valid_plan(q):
    if not isinstance(q, dict) or set(q) != {
        "join",
        "on",
        "where",
        "group",
        "measures",
        "having",
    }:
        return False
    if q["join"] not in ("inner", "left"):
        return False
    for field in ("on", "where"):
        p = q[field]
        if not isinstance(p, list) or len(p) > 6 or not all(_predicate(x) for x in p):
            return False
    group, measures, having = q["group"], q["measures"], q["having"]
    if (
        not isinstance(group, list)
        or len(group) > 2
        or not all(isinstance(v, str) and v in COLUMNS for v in group)
    ):
        return False
    if (
        len(set(group)) != len(group)
        or not isinstance(measures, list)
        or not 1 <= len(measures) <= 3
    ):
        return False
    if (
        not all(_aggregate(a) for a in measures)
        or not isinstance(having, list)
        or len(having) > 3
    ):
        return False
    return all(
        isinstance(h, list)
        and len(h) == 3
        and h[0] in ("eq", "ge", "gt", "le")
        and type(h[1]) is int
        and 0 <= h[1] < len(measures)
        and type(h[2]) is int
        and 0 <= h[2] <= 4
        for h in having
    )


def _get(term, row):
    return row[COLUMNS.index(term)] if isinstance(term, str) else term


def _accept(predicates, row):
    for op, *args in predicates:
        v = _get(args[0], row)
        if op == "known":
            passed = v is not None
        elif op == "missing":
            passed = v is None
        else:
            w = _get(args[1], row)
            if op == "same":
                passed = v == w
            else:
                passed = (
                    v is not None
                    and w is not None
                    and ((v == w) if op == "eq" else (v != w))
                )
        if not passed:
            return False
    return True


def _number(aggregate, rows):
    kind, field = aggregate
    if field == "*":
        return len(rows)
    values = [
        row[COLUMNS.index(field)]
        for row in rows
        if row[COLUMNS.index(field)] is not None
    ]
    if kind == "count":
        return len(values)
    if kind == "distinct":
        return len(set(values))
    if kind == "sum0":
        return sum(values)
    if not values:
        return None
    return {"sum": sum, "min": min, "max": max}[kind](values)


def evaluate(plan, left, right):
    groups = {} if plan["group"] else {(): []}
    for row in left:
        joined = [row + r for r in right if _accept(plan["on"], row + r)]
        if plan["join"] == "left" and not joined:
            joined.append(row + (None, None))
        for candidate in joined:
            if _accept(plan["where"], candidate):
                key = tuple(_get(v, candidate) for v in plan["group"])
                groups.setdefault(key, []).append(candidate)
    results = []
    for key, rows in groups.items():
        stats = [_number(a, rows) for a in plan["measures"]]

        def qualifies(h):
            op, index, threshold = h
            value = stats[index]
            if value is None:
                return False
            return {
                "eq": value == threshold,
                "ge": value >= threshold,
                "gt": value > threshold,
                "le": value <= threshold,
            }[op]

        if all(map(qualifies, plan["having"])):
            results.append([*key, *stats])
    return sorted(results, key=_key)


@lru_cache(maxsize=1)
def partitions():
    """Overlapping properties of the input bags, independent of the submitted or requested query.

    An unmatched key is a left key without an equal, non-NULL key on the right.
    This partition also applies when the report requests a NULL-safe comparison.
    """
    out = []
    for left, right in universe():
        group = ["bag_database_accuracy"]
        if not left or not right:
            group.append("bag_empty_accuracy")
        if len(set(left)) != len(left) or len(set(right)) != len(right):
            group.append("bag_duplicate_accuracy")
        if any(None in record for record in left + right):
            group.append("bag_null_accuracy")
        if any(key is None or all(key != rk for rk, _ in right) for key, _ in left):
            group.append("bag_unmatched_accuracy")
        out.append(group)
    return tuple(out)


def score(item, reply):
    from hidden import strict_json

    detail = dict.fromkeys(DETAIL_KEYS, 0.0)
    try:
        raw = reply.get("content")
        if not isinstance(raw, str) or len(raw) > 8192:
            raise ValueError
        plan = strict_json(raw)
        detail["valid_json"] = 1.0
    except (ValueError, TypeError, RecursionError):
        return {"correct": False, "score": 0.0, "detail": detail}
    if not valid_plan(plan):
        return {"correct": False, "score": 0.0, "detail": detail}
    detail["bag_program"] = 1.0
    spec = item["expected"]
    refids = (i for (i,) in struct.iter_unpack(">H", base64.b64decode(spec["index"])))
    passed, total = {}, {}
    for (left, right), idx, metrics in zip(universe(), refids, partitions()):
        match = evaluate(plan, left, right) == spec["outputs"][idx]
        for metric in metrics:
            passed[metric] = passed.get(metric, 0) + match
            total[metric] = total.get(metric, 0) + 1
    for metric in total:
        detail[metric] = passed[metric] / total[metric]
    correct = passed["bag_database_accuracy"] == len(universe())
    detail["bag_all_databases"] = float(correct)
    return {"correct": correct, "score": float(correct), "detail": detail}
