"""Check a submitted workshop schedule against its resource calendar and optimal finish."""
from graphlib import CycleError, TopologicalSorter
import re


def _identifier(value):
    return isinstance(value, str) and re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,31}", value) is not None


def _integer(value, low, high):
    return type(value) is int and low <= value <= high


def _names(value, allowed):
    return (isinstance(value, list) and all(_identifier(name) and name in allowed for name in value)
            and len(value) == len(set(value)))


def validate_expected(expected):
    """Check the private instance schema before using its calendar or optimal finish."""
    if not isinstance(expected, dict) or set(expected) != {"tasks", "closures", "optimum"}:
        return False
    tasks, closures, optimum = expected["tasks"], expected["closures"], expected["optimum"]
    if not isinstance(tasks, dict) or not 1 <= len(tasks) <= 32 or not all(map(_identifier, tasks)):
        return False
    if not isinstance(closures, dict) or not 1 <= len(closures) <= 16 or not all(map(_identifier, closures)):
        return False
    if not _integer(optimum, 1, 128):
        return False
    for periods in closures.values():
        if not isinstance(periods, list) or len(periods) > 128:
            return False
        for period in periods:
            if (not isinstance(period, list) or len(period) != 2 or not _integer(period[0], 0, 127)
                    or not _integer(period[1], period[0] + 1, 128)):
                return False
    for name, task in tasks.items():
        if not isinstance(task, dict) or set(task) != {"duration", "release", "deadline", "resources", "after"}:
            return False
        if (not _integer(task["duration"], 1, 128) or not _integer(task["release"], 0, 127)
                or not _integer(task["deadline"], task["release"] + task["duration"], 128)
                or task["release"] + task["duration"] > optimum):
            return False
        if (not _names(task["resources"], closures) or not task["resources"]
                or not _names(task["after"], tasks) or name in task["after"]):
            return False
    try:
        TopologicalSorter({name: task["after"] for name, task in tasks.items()}).prepare()
    except CycleError:
        return False
    return optimum <= max(task["deadline"] for task in tasks.values())


def _finish(expected, starts):
    """Return the last finish of a feasible schedule, or None for an invalid certificate."""
    tasks = expected["tasks"]
    if not isinstance(starts, dict) or starts.keys() != tasks.keys():
        return None
    if any(not _integer(starts[name], task["release"], task["deadline"] - task["duration"])
           for name, task in tasks.items()):
        return None
    ends = {name: starts[name] + task["duration"] for name, task in tasks.items()}
    occupied = {resource: list(periods) for resource, periods in expected["closures"].items()}
    for name, task in tasks.items():
        start, end = starts[name], ends[name]
        if any(ends[parent] > start for parent in task["after"]):
            return None
        for resource in task["resources"]:
            if any(start < until and since < end for since, until in occupied[resource]):
                return None
            occupied[resource].append((start, end))
    return max(ends.values())


def score(item, reply):
    from hidden import strict_json

    try:
        starts = strict_json(reply["content"])
        parsed = True
    except (TypeError, ValueError, RecursionError):
        starts, parsed = None, False
    finish = _finish(item["expected"], starts)
    optimal = finish == item["expected"]["optimum"]
    return {"correct": optimal, "score": float(optimal),
            "detail": {"valid_json": float(parsed), "schedule_feasible": float(finish is not None),
                       "schedule_optimal": float(optimal)}}
