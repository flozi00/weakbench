"""Verify a bounded, entirely local source search, its quotations and its verdict."""

import json
import re

LABELS = {
    "de": (("dagegen", "offen", "dafür"), ("widerlegt", "offen", "bestätigt", "strittig")),
    "en": (("against", "open", "for"), ("refuted", "unresolved", "supported", "conflicting")),
}
CONDITIONAL_KEYS = frozenset(("inquiry_conflict_recognized", "inquiry_uncertainty_recognized"))
DETAIL_KEYS = CONDITIONAL_KEYS | frozenset(
    (
        "valid_json",
        "inquiry_trace_valid",
        "inquiry_sources_read",
        "inquiry_sources_current",
        "inquiry_quotes_exact",
        "inquiry_ids_authentic",
        "inquiry_quotes_attributed",
        "inquiry_positions_correct",
        "inquiry_verdict_correct",
    )
)


def _string(x):
    return isinstance(x, str) and 0 < len(x) <= 320 and x.strip() == x


def _id(x):
    return isinstance(x, str) and re.fullmatch("[A-Za-z][A-Za-z0-9_-]{0,39}", x) is not None


def _target(spec, doc):
    return next(e for e in doc["entries"] if (e["subject"], e["round"]) == (spec["subject"], spec["round"]))


def _current(shelf):
    leaders = {}
    for doc in shelf["documents"]:
        if not doc["copy"] and (
            doc["origin"] not in leaders or doc["issue"] > leaders[doc["origin"]]["issue"]
        ):
            leaders[doc["origin"]] = doc
    return list(leaders.values())


def validate_expected(s):
    if not isinstance(s, dict) or set(s) != {
        "query",
        "venue",
        "rule",
        "threshold",
        "subject",
        "round",
        "shelves",
        "budget",
    }:
        return False
    if (
        not all(_string(s[k]) for k in ("query", "venue", "subject", "round"))
        or s["rule"] not in ("eq", "ge", "le")
        or type(s["threshold"]) is not int
        or s["threshold"] < 0
        or type(s["budget"]) is not int
        or not 3 <= s["budget"] <= 12
        or not isinstance(s["shelves"], list)
        or not 2 <= len(s["shelves"]) <= 5
    ):
        return False
    ids, keys, venues = set(), set(), set()
    for shelf in s["shelves"]:
        if (
            not isinstance(shelf, dict)
            or set(shelf) != {"key", "venue", "documents"}
            or not _id(shelf["key"])
            or shelf["key"] in keys
            or shelf["key"].casefold() == s["query"].casefold()
            or not _string(shelf["venue"])
            or shelf["venue"] in venues
            or not isinstance(shelf["documents"], list)
            or not 2 <= len(shelf["documents"]) <= 18
        ):
            return False
        keys.add(shelf["key"])
        venues.add(shelf["venue"])
        editions = set()
        for doc in shelf["documents"]:
            if (
                not isinstance(doc, dict)
                or set(doc) != {"id", "origin", "issue", "copy", "entries"}
                or not _id(doc["id"])
                or doc["id"] in ids
                or not _id(doc["origin"])
                or type(doc["issue"]) is not int
                or not 1 <= doc["issue"] <= 25
                or type(doc["copy"]) is not bool
                or not isinstance(doc["entries"], list)
                or not 1 <= len(doc["entries"]) <= 5
            ):
                return False
            ids.add(doc["id"])
            if not doc["copy"]:
                edition = (doc["origin"], doc["issue"])
                if edition in editions:
                    return False
                editions.add(edition)
            seen, texts = set(), set()
            for e in doc["entries"]:
                if (
                    not isinstance(e, dict)
                    or set(e) != {"subject", "round", "lo", "hi", "text"}
                    or not all(_string(e[k]) for k in ("subject", "round", "text"))
                    or any(v is not None and (type(v) is not int or v < 0) for v in (e["lo"], e["hi"]))
                    or (e["lo"] is not None and e["hi"] is not None and e["lo"] > e["hi"])
                ):
                    return False
                key = (e["subject"], e["round"])
                if key in seen or e["text"] in texts:
                    return False
                seen.add(key)
                texts.add(e["text"])
            if (s["subject"], s["round"]) not in seen:
                return False
        if not 1 <= len(_current(shelf)) <= 7:
            return False
    target = next((sh for sh in s["shelves"] if sh["venue"] == s["venue"]), None)
    return target is not None and len(_current(target)) + 2 <= s["budget"]


def _position(rule, value, line):
    lo, hi = line["lo"], line["hi"]
    lo = 0 if lo is None else lo
    if rule == "ge":
        return 1 if lo >= value else -1 if hi is not None and hi < value else 0
    if rule == "le":
        return 1 if hi is not None and hi <= value else -1 if lo > value else 0
    return 1 if lo == value == hi else -1 if lo > value or hi is not None and hi < value else 0


def _result(positive, negative):
    if positive and negative:
        return 2
    if positive >= 2:
        return 1
    return -1 if negative >= 2 else 0


def _arguments(name, raw):
    from hidden import strict_json

    if isinstance(raw, str):
        try:
            raw = strict_json(raw)
        except (ValueError, RecursionError):
            return None
    field = {"find_records": "term", "read_record": "id"}.get(name) if isinstance(name, str) else None
    if field is None or not isinstance(raw, dict) or set(raw) != {field} or not _string(raw[field]):
        return None
    return raw[field]


def _search(spec, name, arg, keys, ids):
    if name == "find_records":
        if arg.casefold() == spec["query"].casefold():
            return {"collections": [{"venue": s["venue"], "key": s["key"]} for s in spec["shelves"]]}
        if arg in keys:
            shelf = next(s for s in spec["shelves"] if s["key"] == arg)
            return {
                "records": [{k: d[k] for k in ("id", "origin", "issue", "copy")} for d in shelf["documents"]]
            }
        return {"matches": []}
    if arg not in ids:
        return {"error": "unlisted_id"}
    doc = next(d for s in spec["shelves"] for d in s["documents"] if d["id"] == arg)
    return {"id": doc["id"], "lines": [e["text"] for e in doc["entries"]]}


def _advance(obj, keys, ids, opened):
    keys.update(rec["key"] for rec in obj.get("collections", []))
    ids.update(rec["id"] for rec in obj.get("records", []))
    if "lines" in obj:
        opened.add(obj["id"])


def runtime(spec, lang):
    descriptions = (
        (
            "Suche nach dem Titel oder einem gefundenen Sammlungsschlüssel.",
            "Lies die Zeilen eines gelisteten Belegs.",
        )
        if lang == "de"
        else ("Look up the title or a discovered collection key.", "Read the lines of a listed record.")
    )
    tools = []
    for (name, field), desc in zip((("find_records", "term"), ("read_record", "id")), descriptions):
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": desc,
                    "parameters": {
                        "type": "object",
                        "properties": {field: {"type": "string"}},
                        "required": [field],
                        "additionalProperties": False,
                    },
                },
            }
        )
    used = 0
    keys, ids, opened = set(), set(), set()
    pending = []

    def finish_batch():
        for obj in pending:
            _advance(obj, keys, ids, opened)
        pending.clear()

    def execute(name, raw):
        nonlocal used
        used += 1
        arg = _arguments(name, raw)
        if used > spec["budget"]:
            obj = {"error": "budget"}
        elif arg is None:
            obj = {"error": "invalid_call"}
        else:
            obj = _search(spec, name, arg, keys, ids)
            pending.append(obj)
        return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))

    nudge = (
        "Schließe die Recherche ab und gib jetzt den Befund aus."
        if lang == "de"
        else "Conclude the search and give your finding now."
    )
    return {
        "tools": tools,
        "tool_exec": execute,
        "max_rounds": spec["budget"] + 1,
        "capture_tool_transcript": True,
        "tool_nudge": nudge,
        "tool_call_budget": spec["budget"],
        "tool_batch_finished": finish_batch,
    }


def _replay(item, trace):
    from hidden import strict_json

    if not isinstance(trace, list):
        return None
    if trace[:1] and trace[:1] == item.get("messages", [])[:1]:
        trace = trace[1:]
    spec = item["expected"]
    if not trace or len(trace) > 2 * spec["budget"]:
        return None
    keys, ids, opened, call_ids = set(), set(), set(), set()
    i = used = 0
    while i < len(trace):
        ask = trace[i]
        i += 1
        if not isinstance(ask, dict) or ask.get("role") != "assistant":
            return None
        calls = ask.get("tool_calls")
        if (
            not isinstance(calls, list)
            or not calls
            or used + len(calls) > spec["budget"]
            or i + len(calls) > len(trace)
        ):
            return None
        pending = []
        for call in calls:
            reply = trace[i]
            i += 1
            if (
                not isinstance(call, dict)
                or call.get("type") != "function"
                or not _string(call.get("id"))
                or call["id"] in call_ids
            ):
                return None
            call_ids.add(call["id"])
            fn = call.get("function")
            if (
                not isinstance(fn, dict)
                or not isinstance(reply, dict)
                or reply.get("role") != "tool"
                or reply.get("tool_call_id") != call["id"]
            ):
                return None
            name = fn.get("name")
            arg = _arguments(name, fn.get("arguments"))
            if arg is None or not isinstance(reply.get("content"), str):
                return None
            actual = _search(spec, name, arg, keys, ids)
            try:
                claimed = strict_json(reply["content"])
                if json.dumps(actual, sort_keys=True, allow_nan=False) != json.dumps(
                    claimed, sort_keys=True, allow_nan=False
                ):
                    return None
            except (ValueError, TypeError, RecursionError):
                return None
            if "error" in actual:
                return None
            pending.append(actual)
        for obj in pending:
            _advance(obj, keys, ids, opened)
        used += len(calls)
    return opened


def score(item, reply):
    from hidden import strict_json

    spec = item["expected"]
    detail = dict.fromkeys(DETAIL_KEYS - CONDITIONAL_KEYS, 0.0)
    opened = _replay(item, reply.get("observation_trace"))
    detail["inquiry_trace_valid"] = float(opened is not None)
    shelf = next(sh for sh in spec["shelves"] if sh["venue"] == spec["venue"])
    current = {d["id"]: _target(spec, d) for d in _current(shelf)}
    detail["inquiry_sources_read"] = float(opened is not None and current.keys() <= opened)
    stances, verdicts = LABELS[item["lang"]]
    pos = {k: _position(spec["rule"], spec["threshold"], e) for k, e in current.items()}
    verdict = verdicts[_result(list(pos.values()).count(1), list(pos.values()).count(-1)) + 1]
    condition = {
        verdicts[1]: "inquiry_uncertainty_recognized",
        verdicts[3]: "inquiry_conflict_recognized",
    }.get(verdict)
    if condition:
        detail[condition] = 0.0
    try:
        ans = strict_json(reply.get("content", ""))
        detail["valid_json"] = 1.0
    except (ValueError, TypeError, RecursionError):
        ans = None
    if isinstance(ans, dict) and set(ans) == {"verdict", "sources"} and isinstance(ans["sources"], list):
        detail["inquiry_verdict_correct"] = float(ans["verdict"] == verdict)
        sources = ans["sources"]
        if sources and all(isinstance(r, dict) and _id(r.get("id")) for r in sources):
            archive = {d["id"]: d for sh in spec["shelves"] for d in sh["documents"]}
            ids = [r["id"] for r in sources]
            detail["inquiry_ids_authentic"] = float(
                len(set(ids)) == len(ids) and all(k in archive for k in ids)
            )
            detail["inquiry_quotes_attributed"] = float(
                all(
                    r["id"] in archive
                    and isinstance(r.get("quote"), str)
                    and any(r["quote"] == line["text"] for line in archive[r["id"]]["entries"])
                    for r in sources
                )
            )
            if condition and all(
                set(r) == {"id", "quote", "position"}
                and isinstance(r["quote"], str)
                and isinstance(r["position"], str)
                for r in sources
            ):
                detail[condition] = detail["inquiry_verdict_correct"]
        listed = {}
        for rec in sources:
            if (
                not isinstance(rec, dict)
                or set(rec) != {"id", "quote", "position"}
                or not _id(rec["id"])
                or rec["id"] in listed
            ):
                break
            listed[rec["id"]] = rec
        else:
            if listed.keys() == current.keys():
                detail["inquiry_sources_current"] = 1.0
                detail["inquiry_quotes_exact"] = float(
                    all(listed[k]["quote"] == e["text"] for k, e in current.items())
                )
                detail["inquiry_positions_correct"] = float(
                    all(listed[k]["position"] == stances[v + 1] for k, v in pos.items())
                )
    correct = all(detail.values()) and not reply.get("tool_calls")
    return {"correct": bool(correct), "score": float(correct), "detail": detail}
