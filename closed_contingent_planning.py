"""Evaluate an observation-dependent dispatch policy in every admissible world."""

from functools import lru_cache
import re

FIELDS = {
    "de": ("dauer", "prüfen", "zweige", "abfertigen"),
    "en": ("time", "inspect", "branches", "dispatch"),
}
DETAIL_KEYS = (
    "contingent_decision_correct", "contingent_declared_impossibility",
    "contingent_observation_branches", "contingent_world_coverage", "contingent_all_worlds",
    "contingent_deadline_met", "contingent_cost_report_correct", "contingent_optimal_policy",
)


def label(value):
    return isinstance(value, str) and bool(re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,23}", value))


def duration(value):
    return type(value) is int and 1 <= value <= 200


def validate_expected(spec):
    if not isinstance(spec, dict) or set(spec) != {"worlds", "limit", "checks", "services"}:
        return False
    worlds, checks, services = spec["worlds"], spec["checks"], spec["services"]
    if (not isinstance(worlds, list) or not 2 <= len(worlds) <= 8
            or not all(label(w) for w in worlds) or len(set(worlds)) != len(worlds)
            or not duration(spec["limit"]) or not isinstance(checks, list) or not 1 <= len(checks) <= 5
            or not isinstance(services, list) or not 2 <= len(services) <= 7):
        return False
    ids = set()
    for probe in checks:
        if not isinstance(probe, dict) or set(probe) != {"id", "time", "signals"}:
            return False
        uid, signals = probe["id"], probe["signals"]
        if (not label(uid) or uid in ids or not duration(probe["time"]) or not isinstance(signals, list)
                or len(signals) != len(worlds) or not all(label(s) for s in signals)):
            return False
        ids.add(uid)
    for action in services:
        if not isinstance(action, dict) or set(action) != {"id", "times"}:
            return False
        uid, times = action["id"], action["times"]
        if (not label(uid) or uid in ids or not isinstance(times, list) or len(times) != len(worlds)
                or not all(v is None or duration(v) for v in times)):
            return False
        ids.add(uid)
    return True


def groups(probe, states):
    result = {}
    for i in states:
        result.setdefault(probe["signals"][i], []).append(i)
    return result


def optimum(spec):
    """Uniform information sets admit a dispatch; a useful check strictly reduces the uncertainty."""
    @lru_cache(None)
    def solve(states):
        options = []
        for action in spec["services"]:
            times = [action["times"][i] for i in states]
            if None not in times:
                options.append(max(times))
        for probe in spec["checks"]:
            outcomes = groups(probe, states)
            if len(outcomes) > 1:
                branches = [solve(tuple(part)) for part in outcomes.values()]
                if None not in branches:
                    options.append(probe["time"] + max(branches))
        return min(options, default=None)
    return solve(tuple(range(len(spec["worlds"]))))


def simulate(spec, tree, lang):
    _, inspect, branches, dispatch = FIELDS[lang]
    sensors = {p["id"]: p for p in spec["checks"]}
    services = {a["id"]: a for a in spec["services"]}
    arrivals = {}

    def visit(node, states, used, elapsed):
        if not isinstance(node, dict):
            return False
        if set(node) == {dispatch} and label(node[dispatch]) and node[dispatch] in services:
            times = services[node[dispatch]]["times"]
            arrivals.update((i, elapsed + times[i]) for i in states if times[i] is not None)
            return True
        if set(node) != {inspect, branches} or not isinstance(node[branches], dict):
            return False
        uid = node[inspect]
        if not label(uid) or uid not in sensors or uid in used:
            return False
        sensor = sensors[uid]
        outcomes = groups(sensor, states)
        complete = outcomes.keys() == node[branches].keys()
        for signal, part in outcomes.items():
            if not visit(node[branches].get(signal), part, used | {uid}, elapsed + sensor["time"]):
                complete = False
        return complete

    lawful = visit(tree, range(len(spec["worlds"])), set(), 0)
    return lawful, arrivals


def score(item, reply):
    from hidden import strict_json

    spec, lang = item["expected"], item["lang"]
    best = optimum(spec)
    feasible = best is not None and best <= spec["limit"]
    key = FIELDS[lang][0]
    try:
        text = reply.get("content")
        answer = strict_json(text) if isinstance(text, str) and len(text) <= 64000 else None
    except (ValueError, TypeError, RecursionError):
        answer = None
    detail = {"contingent_decision_correct": 0.0}
    if not feasible:
        detail["contingent_declared_impossibility"] = 0.0
    else:
        detail["contingent_optimal_policy"] = 0.0
    correct = False
    if isinstance(answer, dict) and set(answer) == {key, "plan"}:
        claimed, tree = answer[key], answer["plan"]
        if claimed is None and tree is None:
            correct = not feasible
            detail["contingent_decision_correct"] = float(correct)
            if not feasible:
                detail["contingent_declared_impossibility"] = 1.0
        elif isinstance(tree, dict) and type(claimed) is int and claimed >= 0:
            detail["contingent_decision_correct"] = float(feasible)
            lawful, arrivals = simulate(spec, tree, lang)
            reached = len(arrivals) == len(spec["worlds"]) and lawful
            elapsed = max(arrivals.values()) if reached else None
            on_time = reached and elapsed <= spec["limit"]
            truthful = reached and elapsed == claimed
            detail.update(contingent_observation_branches=float(lawful),
                          contingent_world_coverage=len(arrivals)/len(spec["worlds"]),
                          contingent_all_worlds=float(reached), contingent_deadline_met=float(on_time))
            if reached:
                detail["contingent_cost_report_correct"] = float(truthful)
            if feasible:
                detail["contingent_optimal_policy"] = float(on_time and elapsed == best)
            correct = bool(on_time and truthful and elapsed == best)
    return {"correct": correct, "score": float(correct), "detail": detail}
