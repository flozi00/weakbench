"""Score a switched-language three-turn chronology using observed reports only."""

import re
import math

import closed_language as language


ANSWERS = {
    "de": {
        "inside": "Der Termin liegt sicher im Prüfzeitraum",
        "outside": "Der Termin liegt sicher außerhalb des Prüfzeitraums",
        "mixed": "Der Termin kann innerhalb oder außerhalb des Prüfzeitraums liegen",
    },
    "en": {
        "inside": "The date is certainly inside the review window",
        "outside": "The date is certainly outside the review window",
        "mixed": "The date can fall inside or outside the review window",
    },
}
OBSERVED = (
    "valid_json",
    "report_fields_present",
    "request_language_ok",
    "justification_language_ok",
    "answer_language_ok",
    "reasoning_present",
    "reasoning_language_decidable",
    "reasoning_language_ok",
    "all_channels_language_ok",
    "all_channels_verified",
    "all_language_ok",
)
TURN_METRICS = (
    *OBSERVED,
    "response_observed",
    "response_format_ok",
    "lower_supported",
    "upper_supported",
    "decision_correct",
    "visible_certificate_complete",
    "turn_correct",
)
DETAIL_KEYS = frozenset(f"turn{i}_{k}" for i in range(1, 4) for k in TURN_METRICS) | {
    "turns_observed",
    "visible_reports_correct",
    "all_requests_language_ok",
    "all_language_ok",
    "all_channels_verified",
    "reasoning_present_fraction",
    "correct_with_all_channels",
    "full",
}


def turn_languages(first):
    return first, ("en" if first == "de" else "de"), first


def stage_data(expected, ordinal):
    state = dict(expected["initial"])
    for update in expected["updates"][:ordinal]:
        state.update(update)
    return state


def limits(records, target):
    # Every note bounds (end - beginning). The distinguished origin has numeric date zero.
    nodes = sorted(
        {target, ""} | {n for a, b, _, _ in records.values() for n in (a, b)}
    )
    distance = {a: {b: 0 if a == b else math.inf for b in nodes} for a in nodes}
    for a, b, lo, hi in records.values():
        distance[a][b] = min(distance[a][b], hi)
        distance[b][a] = min(distance[b][a], -lo)
    for k in nodes:
        for a in nodes:
            for b in nodes:
                distance[a][b] = min(distance[a][b], distance[a][k] + distance[k][b])
    if any(distance[a][a] < 0 for a in nodes):
        return None
    return -distance[target][""], distance[""][target]


def classify(lo, hi, window):
    start, end = window
    if start <= lo and hi <= end:
        return "inside"
    if hi < start or lo > end:
        return "outside"
    return "mixed"


def validate_expected(e):
    if type(e) is not dict or set(e) != {"target", "window", "initial", "updates"}:
        return False

    def named(x):
        return type(x) is str and re.fullmatch("[A-Z][a-z]{1,15}", x) is not None

    if not named(e["target"]) or type(e["window"]) is not list or len(e["window"]) != 2:
        return False
    w = e["window"]
    if not all(type(i) is int for i in w) or not 1 <= w[0] <= w[1] <= 52:
        return False
    if (
        type(e["initial"]) is not dict
        or not 3 <= len(e["initial"]) <= 12
        or type(e["updates"]) is not list
        or len(e["updates"]) != 2
    ):
        return False
    known = {}
    for ordinal, change in enumerate([e["initial"], *e["updates"]]):
        if type(change) is not dict or (
            ordinal and (len(change) != 2 or len(set(change) & set(known)) != 1)
        ):
            return False
        for key, note in change.items():
            if (
                type(key) is not str
                or not re.fullmatch(r"K[0-9]{1,3}", key)
                or type(note) is not list
                or len(note) != 4
            ):
                return False
            a, b, lo, hi = note
            if (
                (a != "" and not named(a))
                or not named(b)
                or a == b
                or type(lo) is not int
                or type(hi) is not int
                or not (1 if a == "" else 0) <= lo <= hi <= 52
            ):
                return False
            if key in known and (note[:2] != known[key][:2] or note == known[key]):
                return False
        known.update(change)
        nodes = {b for _, b, _, _ in known.values()} | {
            a for a, _, _, _ in known.values() if a
        }
        if len(nodes) > 6:
            return False
        bounds = limits(known, e["target"])
        if bounds is None or not (
            math.isfinite(bounds[0]) and math.isfinite(bounds[1])
        ):
            return False
    return True


def support(text, lang, kind, state, target, bounds):
    if lang == "de":
        end = ("frühestmöglicher", "spätestmöglicher")[kind]
        pattern = rf"aus (k\d+(?: und k\d+)*) ergibt sich woche (\d+) als {end} termin"
    else:
        end = ("earliest", "latest")[kind]
        pattern = rf"from (k\d+(?: and k\d+)*) the {end} possible week is (\d+)"
    join = " und " if lang == "de" else " and "
    for sentence in language.clauses(text):
        match = re.fullmatch(pattern, sentence)
        if not match:
            continue
        refs = match[1].upper().split(join.upper())
        if (
            int(match[2]) != bounds[kind]
            or len(set(refs)) != len(refs)
            or not set(refs) <= set(state)
        ):
            continue
        selected = {k: state[k] for k in refs}
        if limits(selected, target)[kind] != bounds[kind]:
            continue
        if all(
            limits({j: v for j, v in selected.items() if j != k}, target)[kind]
            != bounds[kind]
            for k in refs
        ):
            return sentence
    return None


def grade_turn(item, index, reply):
    from hidden import strict_json

    lang = turn_languages(item["lang"])[index]
    prompt = item["turns"][index][0]["content"]
    if reply is None:
        return {
            "response_observed": 0.0,
            "request_language_ok": float(language.request_language_ok(prompt, lang)),
            "turn_correct": 0.0,
            "all_channels_verified": 0.0,
        }
    if (
        not isinstance(reply, dict)
        or not isinstance(reply.get("content"), str)
        or reply.get("reasoning") is not None
        and not isinstance(reply["reasoning"], str)
    ):
        flags = {
            "response_observed": 1.0,
            "response_format_ok": 0.0,
            "request_language_ok": float(language.request_language_ok(prompt, lang)),
            "turn_correct": 0.0,
            "all_channels_verified": 0.0,
        }
        if (
            isinstance(reply, dict)
            and "reasoning" in reply
            and (reply["reasoning"] is None or isinstance(reply["reasoning"], str))
        ):
            native = reply["reasoning"] or ""
            flags["reasoning_present"] = float(bool(native.strip()))
            if native.strip():
                status = language.language_status(native, lang)
                flags["reasoning_language_decidable"] = float(status is not None)
                if status is not None:
                    flags["reasoning_language_ok"] = float(status)
        return flags
    expected = item["expected"]
    state = stage_data(expected, index)
    bounds = limits(state, expected["target"])
    why, _ = language.FIELDS[lang]
    try:
        data = strict_json(reply.get("content", ""))
    except (TypeError, ValueError, RecursionError):
        data = None
    justification = data.get(why, "") if isinstance(data, dict) else ""
    if not isinstance(justification, str):
        justification = ""
    assertions = [
        support(justification, lang, i, state, expected["target"], bounds)
        for i in range(2)
    ]
    conclusion = ANSWERS[lang][classify(*bounds, expected["window"])].casefold()
    # Two citations must stand on their own and together exhaust the stated justification.
    role_item = {"lang": lang, "messages": [{"role": "user", "content": prompt}]}
    result = language.score_report(
        role_item, reply, [{a} if a else set() for a in assertions], [{conclusion}]
    )
    flags = {k: result["detail"][k] for k in OBSERVED if k in result["detail"]}
    flags.update(
        response_observed=1.0,
        response_format_ok=1.0,
        lower_supported=float(assertions[0] is not None),
        upper_supported=float(assertions[1] is not None),
        decision_correct=result["detail"]["answer_facts_ok"],
        visible_certificate_complete=result["detail"]["justification_facts_ok"],
        turn_correct=float(result["correct"]),
    )
    return flags


def finish(turns):
    observed = sum(t.get("response_observed", 0.0) for t in turns)
    detail = {"turns_observed": min(3, observed) / 3}
    if len(turns) != 3:
        detail.update(full=0.0, all_channels_verified=0.0, correct_with_all_channels=0.0)
        return {"correct": False, "score": 0.0, "detail": detail}
    for i, flags in enumerate(turns, 1):
        detail.update(
            {f"turn{i}_{k}": v for k, v in flags.items() if k in TURN_METRICS}
        )
    visible = all(
        t.get("justification_language_ok")
        and t.get("answer_language_ok")
        and t.get("lower_supported")
        and t.get("upper_supported")
        and t.get("decision_correct")
        and t.get("visible_certificate_complete")
        for t in turns
    )
    detail.update(
        visible_reports_correct=float(visible),
        all_requests_language_ok=float(
            all(t.get("request_language_ok") for t in turns)
        ),
        all_language_ok=float(all(t.get("all_language_ok") for t in turns)),
        all_channels_verified=float(all(t.get("all_channels_verified") for t in turns)),
    )
    known = [t for t in turns if "reasoning_present" in t]
    if known:
        detail["reasoning_present_fraction"] = sum(
            t["reasoning_present"] for t in known
        ) / len(known)
    correct = all(t.get("turn_correct") == 1.0 for t in turns)
    detail["full"] = float(correct)
    detail["correct_with_all_channels"] = float(correct and detail["all_channels_verified"] == 1.0)
    return {"correct": correct, "score": float(correct), "detail": detail}


def score(item, reply):
    # Only code parsing actual API envelopes may populate observed_turns. Never accept a submitted score.
    reports = reply.get("observed_turns")
    if not isinstance(reports, list) or len(reports) > 3:
        return finish([])
    return finish(
        [
            grade_turn(item, i, reports[i] if i < len(reports) else None)
            for i in range(3)
        ]
    )
