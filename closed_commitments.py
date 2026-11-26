"""Evaluate online assignment repairs relative to the model's own preceding booking."""

import re


DETAIL_KEYS = (
    "valid_json",
    "first_feasible",
    "first_optimal",
    "repair1_feasible",
    "repair1_optimal",
    "repair2_feasible",
    "repair2_optimal",
    "full",
)


def minimum_cost(allowed, weights, previous):
    """The lowest (total cancellation weight, number moved) for a complete matching."""
    best = {0: (0, 0)}
    for party, options in enumerate(allowed):
        following = {}
        for used, cost in best.items():
            for slot in options:
                bit = 1 << slot
                if used & bit:
                    continue
                changed = previous is not None and slot != previous[party]
                candidate = (cost[0] + weights[party] * changed, cost[1] + changed)
                mask = used | bit
                if mask not in following or candidate < following[mask]:
                    following[mask] = candidate
        best = following
    return min(best.values(), default=None)


def _ids(values, low, high):
    return (
        isinstance(values, list)
        and low <= len(values) <= high
        and all(
            isinstance(x, str) and re.fullmatch("[a-zA-Z][a-zA-Z0-9_-]{0,23}", x)
            for x in values
        )
        and len(set(values)) == len(values)
    )


def validate_expected(expected):
    if not isinstance(expected, dict) or set(expected) != {
        "parties",
        "slots",
        "weights",
        "stages",
    }:
        return False
    parties, slots = expected["parties"], expected["slots"]
    if not _ids(parties, 5, 6) or not _ids(slots, len(parties), len(parties) + 1):
        return False
    weights = expected["weights"]
    if (
        not isinstance(weights, list)
        or len(weights) != len(parties)
        or not all(type(w) is int and 1 <= w <= 1000 for w in weights)
    ):
        return False
    stages = expected["stages"]
    if not isinstance(stages, list) or len(stages) != 3:
        return False
    preceding = None
    for stage in stages:
        if not isinstance(stage, list) or len(stage) != len(parties):
            return False
        for options in stage:
            if (
                not isinstance(options, list)
                or not 1 <= len(options) <= len(slots)
                or any(type(s) is not int or not 0 <= s < len(slots) for s in options)
                or len(set(options)) != len(options)
            ):
                return False
        if minimum_cost(stage, weights, None) is None:
            return False
        domains = list(map(frozenset, stage))
        if (
            preceding is not None
            and sum(a != b for a, b in zip(preceding, domains)) != 1
        ):
            return False
        preceding = domains
    return True


def _booking(proposal, parties, slots, allowed):
    if not isinstance(proposal, dict) or set(proposal) != set(parties):
        return None
    numbers = []
    for party, options in zip(parties, allowed):
        value = proposal[party]
        if not isinstance(value, str) or value not in slots:
            return None
        position = slots.index(value)
        if position not in options:
            return None
        numbers.append(position)
    return numbers if len(set(numbers)) == len(numbers) else None


def score(item, reply):
    from hidden import strict_json

    detail = dict.fromkeys(DETAIL_KEYS, 0.0)
    turns = reply.get("turns")
    if not isinstance(turns, list) or len(turns) != 3:
        return {"correct": False, "score": 0.0, "detail": detail}
    expected = item["expected"]
    previous = None
    parsed = 0
    for index, (stage, content) in enumerate(zip(expected["stages"], turns)):
        tag = ("first", "repair1", "repair2")[index]
        booking = None
        if isinstance(content, str):
            try:
                proposal = strict_json(content)
            except (ValueError, RecursionError):
                pass
            else:
                parsed += 1
                booking = _booking(
                    proposal, expected["parties"], expected["slots"], stage
                )
        if booking is not None:
            detail[tag + "_feasible"] = 1.0
            if index == 0:
                # Every feasible first commitment is allowed.
                detail["first_optimal"] = 1.0
            elif previous is not None:
                moved = [i for i, slot in enumerate(booking) if slot != previous[i]]
                actual = (sum(expected["weights"][i] for i in moved), len(moved))
                detail[tag + "_optimal"] = float(
                    actual == minimum_cost(stage, expected["weights"], previous)
                )
        previous = booking
    detail["valid_json"] = float(parsed == 3)
    full = all(detail[name + "_optimal"] for name in ("first", "repair1", "repair2"))
    detail["full"] = float(full)
    return {"correct": full, "score": float(full), "detail": detail}
