"""Run a local binary observation tool and verify an adaptive identification transcript."""

from functools import lru_cache
import json
import re


DETAIL_KEYS = frozenset(
    (
        "valid_json",
        "measurement_transcript_valid",
        "measurement_within_budget",
        "measurement_informative",
        "measurement_strategy_safe",
        "measurement_identified",
        "measurement_answer_ok",
    )
)


def _identifier(value):
    return isinstance(value, str) and re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,31}", value) is not None


def validate_expected(expected):
    if not isinstance(expected, dict) or set(expected) != {"probes", "table", "budget", "target"}:
        return False
    probes, table, budget = expected["probes"], expected["table"], expected["budget"]
    if not isinstance(probes, list) or not 2 <= len(probes) <= 24 or not all(map(_identifier, probes)):
        return False
    if len(probes) != len(set(probes)) or type(budget) is not int or not 1 <= budget <= 5:
        return False
    if not isinstance(table, dict) or not 2 <= len(table) <= 24 or not all(map(_identifier, table)):
        return False
    if not _identifier(expected["target"]) or expected["target"] not in table:
        return False
    if any(
        not isinstance(row, list)
        or len(row) != len(probes)
        or any(type(bit) is not int or bit not in (0, 1) for bit in row)
        for row in table.values()
    ):
        return False
    return len({tuple(row) for row in table.values()}) == len(table)


def valid_pairs(rows):
    """Each pair presents the same evidence and differs only in the unknown specimen."""
    if len(rows) % 2 or len({(row["prompt"], row["expected"]["target"]) for row in rows}) != len(rows):
        return False
    for a, b in zip(rows[::2], rows[1::2]):
        if a["prompt"] != b["prompt"]:
            return False
        if any(a["expected"][key] != b["expected"][key] for key in ("probes", "table", "budget")):
            return False
    return True


def _probe_name(name, arguments, expected):
    from hidden import strict_json

    if name != "observe":
        return None
    if isinstance(arguments, str):
        try:
            arguments = strict_json(arguments)
        except (ValueError, RecursionError):
            return None
    if not isinstance(arguments, dict) or set(arguments) != {"probe"}:
        return None
    probe = arguments["probe"]
    return probe if isinstance(probe, str) and probe in expected["probes"] else None


def _outcome(expected, probe):
    index = expected["probes"].index(probe)
    return {"probe": probe, "value": expected["table"][expected["target"]][index]}


def runtime(expected, lang):
    """Create a fresh measurement allowance for one candidate conversation."""
    used = 0

    def observe(name, arguments):
        nonlocal used
        used += 1
        if used > expected["budget"]:
            return json.dumps({"error": "budget"})
        probe = _probe_name(name, arguments, expected)
        if probe is None:
            return json.dumps({"error": "invalid_call"})
        return json.dumps(_outcome(expected, probe))

    description = (
        "Den binären Messwert der unbekannten Probe abfragen."
        if lang == "de"
        else "Read one binary measurement of the unknown sample."
    )
    function = {
        "name": "observe",
        "description": description,
        "parameters": {
            "type": "object",
            "properties": {"probe": {"type": "string", "enum": expected["probes"]}},
            "required": ["probe"],
            "additionalProperties": False,
        },
    }
    return {
        "tools": [{"type": "function", "function": function}],
        "tool_exec": observe,
        "max_rounds": expected["budget"] + 1,
        "capture_tool_transcript": True,
    }


def _verified_probes(item, trace):
    """Check role order, call/result association, and the local oracle's value on every turn."""
    from hidden import strict_json

    if not isinstance(trace, list):
        return None
    if trace[:1] == item.get("messages", [])[:1] and trace:
        trace = trace[1:]
    budget = item["expected"]["budget"]
    if len(trace) > 2 * (budget + 1) or len(trace) % 2:
        return None
    probes = []
    for ask, result in zip(trace[::2], trace[1::2]):
        if not isinstance(ask, dict) or ask.get("role") != "assistant":
            return None
        if not isinstance(result, dict) or result.get("role") != "tool":
            return None
        calls = ask.get("tool_calls")
        if not isinstance(calls, list) or len(calls) != 1 or not isinstance(calls[0], dict):
            return None
        call = calls[0]
        if (
            call.get("type") != "function"
            or not isinstance(call.get("id"), str)
            or not call["id"]
            or result.get("tool_call_id") != call["id"]
        ):
            return None
        fn = call.get("function")
        if not isinstance(fn, dict):
            return None
        probe = _probe_name(fn.get("name"), fn.get("arguments"), item["expected"])
        if probe is None or not isinstance(result.get("content"), str):
            return None
        try:
            reported = strict_json(result["content"])
        except (ValueError, RecursionError):
            return None
        canonical = _outcome(item["expected"], probe) if len(probes) < budget else {"error": "budget"}
        if (
            not isinstance(reported, dict)
            or reported.keys() != canonical.keys()
            or any(type(reported[k]) is not type(v) or reported[k] != v for k, v in canonical.items())
        ):
            return None
        probes.append(probe)
    return probes


def _evaluate_path(expected, probes):
    """Every chosen probe must preserve a worst-case completion within the remaining allowance."""
    table, budget = expected["table"], expected["budget"]

    def split(states, index):
        ones = frozenset(s for s in states if table[s][index])
        return states - ones, ones

    @lru_cache(maxsize=None)
    def resolvable(states, remaining):
        if len(states) <= 1:
            return True
        if remaining <= 0 or len(states) > (1 << remaining):
            return False
        for index in range(len(expected["probes"])):
            zero, one = split(states, index)
            if zero and one and resolvable(zero, remaining - 1) and resolvable(one, remaining - 1):
                return True
        return False

    current = frozenset(table)
    informative = safe = bool(probes)
    for turn, probe in enumerate(probes, 1):
        index = expected["probes"].index(probe)
        branches = split(current, index)
        informative = informative and all(branches)
        safe = safe and bool(all(branches)) and all(resolvable(group, budget - turn) for group in branches)
        current = branches[table[expected["target"]][index]]
    return informative, safe, len(current) == 1


def score(item, reply):
    from hidden import strict_json

    expected = item["expected"]
    try:
        answer = strict_json(reply.get("content", ""))
        parsed = True
    except (ValueError, TypeError, RecursionError):
        answer, parsed = None, False
    answer_ok = not reply.get("tool_calls") and answer == {"sample": expected["target"]}
    probes = _verified_probes(item, reply.get("observation_trace"))
    authentic = probes is not None
    timely = authentic and len(probes) <= expected["budget"]
    informative, safe, identified = _evaluate_path(expected, probes) if timely else (False, False, False)
    correct = bool(answer_ok and timely and informative and safe and identified)
    return {
        "correct": correct,
        "score": float(correct),
        "detail": {
            "valid_json": float(parsed),
            "measurement_transcript_valid": float(authentic),
            "measurement_within_budget": float(timely),
            "measurement_informative": float(informative),
            "measurement_strategy_safe": float(safe),
            "measurement_identified": float(identified),
            "measurement_answer_ok": float(answer_ok),
        },
    }
