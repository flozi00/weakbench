"""A bounded local document index and replayable native chat-completion search calls."""

from collections import Counter
from datetime import date
import json
import math
import re
import unicodedata


MAX_CALLS = 4
MAX_RESULTS = 3
MAX_QUERY = 512
MAX_RESULT_CHARS = 16000
PUBLIC_FIELDS = (
    "id",
    "title",
    "url",
    "authority",
    "issued",
    "effective_from",
    "effective_until",
    "text",
)
NOISE = set(
    """the this that what which where when why how who its and are from for with was were does
did has had have not under into per most please show find search about give today current latest
der die das dem den des ein eine einem einer einen ist sind war werden wird wann wie wer welche
welcher welches welcher welches was gilt gelten von vom mit für zum zur und oder nach im in am
an um auf aus bei bitte suche finde zeige aktuell aktuelle aktuellen derzeit stand is of on at as
to en de für er sie es gibt eine kann falls""".split()
)
NOISE = {
    "".join(
        c
        for c in unicodedata.normalize("NFKD", w.casefold())
        if not unicodedata.combining(c)
    )
    for w in NOISE
}
EQUIVALENTS = (
    ("fee", "fees", "charge", "charges", "gebuhr", "gebuhren"),
    ("price", "prices", "preis", "preise", "cost", "costs", "kosten"),
    ("limit", "limits", "maximum", "obergrenze", "hochstwert"),
    ("deadline", "deadlines", "frist", "fristen"),
    ("permit", "permits", "zulassung", "genehmigung"),
    ("opening", "offnung", "offnungszeiten"),
    ("hours", "stunden"),
    ("days", "tage", "tagen"),
    ("notice", "bekanntmachung"),
)
LEXICON = {word: group[0] for group in EQUIVALENTS for word in group}


def terms(text):
    plain = "".join(
        c
        for c in unicodedata.normalize("NFKD", text.casefold())
        if not unicodedata.combining(c)
    )
    return {
        LEXICON.get(t, t)
        for t in re.findall(r"[^\W_]+", plain)
        if len(t) > 1 and t not in NOISE
    }


def _query(raw):
    from hidden import strict_json

    if not isinstance(raw, str) or len(raw) > 4096:
        return None
    try:
        obj = strict_json(raw)
    except (ValueError, RecursionError):
        return None
    if (
        not isinstance(obj, dict)
        or set(obj) != {"query"}
        or not isinstance(obj["query"], str)
    ):
        return None
    query = obj["query"].strip()
    return query if 0 < len(query) <= MAX_QUERY else None


def _record(doc):
    return {key: doc[key] for key in PUBLIC_FIELDS}


def search(spec, query):
    """Rank only visible source material. The answer key, question metadata and clock are never consulted."""
    docs = spec["docs"]
    exact = [
        d for d in docs if query.casefold() in (d["id"].casefold(), d["url"].casefold())
    ]
    if exact:
        return [_record(d) for d in sorted(exact, key=lambda d: d["id"])[:MAX_RESULTS]]
    needed = terms(query)
    if not needed:
        return []
    index = []
    frequency = Counter()
    for d in docs:
        zones = [
            terms(" ".join(str(d[k] or "") for k in keys))
            for keys in (
                ("title", "id"),
                ("authority",),
                ("text",),
                ("url", "issued", "effective_from", "effective_until"),
            )
        ]
        vocabulary = set.union(*zones)
        frequency.update(needed & vocabulary)
        index.append((d, zones))
    date_hits = re.findall(r"(?<!\d)\d{4}-\d{2}-\d{2}(?!\d)", query)
    when = None
    if len(date_hits) == 1:
        try:
            when = date.fromisoformat(date_hits[0]).isoformat()
        except ValueError:
            pass
    ranked = []
    for d, zones in index:
        matches = needed & set.union(*zones)
        if not matches:
            continue
        scores = []
        for t in sorted(matches):
            weight = max(w for z, w in zip(zones, (4.0, 3.0, 2.5, 1.0)) if t in z)
            scores.append(weight * math.log1p(len(docs) / frequency[t]))
        applies = bool(
            when
            and d["issued"] <= when
            and (not d["effective_from"] or d["effective_from"] <= when)
            and (not d["effective_until"] or when <= d["effective_until"])
        )
        ranked.append((sum(scores), applies, d["issued"], d["id"], d))
    # Stable ties retain ascending IDs. Issuance and a date explicitly in the query break lexical ties.
    ranked.sort(key=lambda r: r[3])
    ranked.sort(key=lambda r: r[:3], reverse=True)
    return [_record(r[-1]) for r in ranked[:MAX_RESULTS]]


def _result(spec, name, raw):
    if name != "web_search":
        return {"error": "unknown_tool"}
    q = _query(raw)
    if q is None:
        return {"error": "invalid_arguments"}
    hits = search(spec, q)
    out = {"results": hits}
    if len(_wire(out)) > MAX_RESULT_CHARS:
        return {"error": "result_limit"}
    return out


def _wire(obj):
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def runtime(spec, lang):
    setup = {"native_web_search": True, "capture_tool_transcript": True}
    if spec["condition"] == "unavailable":
        return setup
    desc = (
        "Durchsuche das bereitgestellte lokale Webarchiv mit freien Suchbegriffen, Titel, URL oder Belegkennung."
        if lang == "de"
        else "Search the supplied local web archive by keywords, title, URL or source ID."
    )
    setup.update(
        tool_choice="auto",
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": desc,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "maxLength": MAX_QUERY}
                        },
                        "required": ["query"],
                        "additionalProperties": False,
                    },
                },
            }
        ],
    )
    return setup


def _calls(calls, seen, remaining):
    if not isinstance(calls, list) or not 1 <= len(calls) <= remaining:
        return None
    selected = []
    used = set(seen)
    for c in calls:
        if not isinstance(c, dict) or c.get("type") != "function":
            return None
        cid = c.get("id")
        fn = c.get("function")
        if (
            not isinstance(cid, str)
            or not cid.strip()
            or len(cid) > 256
            or cid in used
            or not isinstance(fn, dict)
            or not isinstance(fn.get("name"), str)
            or not fn["name"].strip()
            or len(fn["name"]) > 128
            or not isinstance(fn.get("arguments"), str)
            or len(fn["arguments"]) > 4096
        ):
            return None
        used.add(cid)
        selected.append(
            {
                "id": cid,
                "type": "function",
                "function": {"name": fn["name"], "arguments": fn["arguments"]},
            }
        )
    return selected


def observed(item, trace):
    """Authenticate each reply against the supplied, finite archive; only seen text can ground a quotation."""
    from hidden import strict_json

    spec = item["expected"]
    empty = {"valid": False, "calls": 0, "searches": 0, "errors": 0, "documents": {}}
    if trace is None:
        trace = []
    initial = item.get("messages", [])
    if isinstance(trace, list) and initial and trace[: len(initial)] == initial:
        trace = trace[len(initial) :]
    if (
        not isinstance(trace, list)
        or len(trace) > 2 * MAX_CALLS
        or (trace and spec["condition"] == "unavailable")
    ):
        return empty
    seen = set()
    docs = {}
    cursor = searches = errors = 0
    while cursor < len(trace):
        ask = trace[cursor]
        cursor += 1
        if not isinstance(ask, dict) or ask.get("role") != "assistant":
            return empty
        calls = _calls(ask.get("tool_calls"), seen, MAX_CALLS - len(seen))
        if calls is None or cursor + len(calls) > len(trace):
            return empty
        for call in calls:
            seen.add(call["id"])
            fn = call["function"]
            tool = trace[cursor]
            cursor += 1
            if (
                not isinstance(tool, dict)
                or tool.get("role") != "tool"
                or tool.get("tool_call_id") != call["id"]
                or not isinstance(tool.get("content"), str)
                or len(tool["content"]) > MAX_RESULT_CHARS
            ):
                return empty
            expected = _result(spec, fn["name"], fn["arguments"])
            try:
                returned = strict_json(tool["content"])
                if json.dumps(returned, sort_keys=True, allow_nan=False) != json.dumps(
                    expected, sort_keys=True, allow_nan=False
                ):
                    return empty
            except (ValueError, TypeError, RecursionError):
                return empty
            if "error" in expected:
                errors += 1
            else:
                searches += 1
                docs.update((d["id"], d["text"]) for d in expected["results"])
    return {
        "valid": True,
        "calls": len(seen),
        "searches": searches,
        "errors": errors,
        "documents": docs,
    }


def _choice(completion):
    if (
        not isinstance(completion, dict)
        or not isinstance(completion.get("choices"), list)
        or len(completion["choices"]) != 1
        or not isinstance(completion["choices"][0], dict)
        or not isinstance(completion["choices"][0].get("message"), dict)
    ):
        raise ValueError("expected one web research completion message")
    return completion["choices"][0]


def _malformed_message(msg):
    return (
        msg.get("role", "assistant") != "assistant"
        or msg.get("function_call") is not None
        or (msg.get("refusal") is not None and msg["refusal"] != "")
        or any(
            msg.get(k) is not None and not isinstance(msg[k], str)
            for k in ("content", "reasoning", "reasoning_content")
        )
        or (
            msg.get("tool_calls") is not None
            and not isinstance(msg["tool_calls"], list)
        )
    )


def parse_final(completion, parse_reply=None):
    """Parse a saved or live terminal assistant message and flag actions/refusals separately from its evidence."""
    if parse_reply is None:
        from quality import parse_reply
    choice = _choice(completion)
    msg = choice["message"]
    invalid = (
        _malformed_message(msg)
        or not isinstance(msg.get("content"), str)
        or not msg["content"].strip()
        or bool(msg.get("tool_calls"))
        or choice.get("finish_reason") not in ("stop", "length")
    )
    if invalid:
        # The shared parser expects well-typed native arguments/text. Never use malformed action data as a reply.
        safe = {
            k: v
            for k, v in msg.items()
            if k in ("content", "reasoning", "reasoning_content") and isinstance(v, str)
        }
        completion = {**completion, "choices": [{**choice, "message": safe}]}
    result = parse_reply(completion)
    result["web_invalid_final"] = bool(invalid or not result["content"].strip())
    return result


def run(item, body, request, parse_reply=None):
    """Perform native tool turns using the supplied request(body) callback. Return the normal reply and summed usage."""
    if parse_reply is None:
        from quality import parse_reply

    spec = item["expected"]
    lang = item["lang"]
    setup = runtime(spec, lang)
    base = {
        k: v
        for k, v in body.items()
        if k not in ("messages", "tools", "tool_choice", "parallel_tool_calls")
    }
    first = list(body["messages"])
    trace = []
    seen = set()
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    nudge = (
        "Das Suchbudget ist ausgeschöpft. Gib jetzt deine Antwort anhand der bereits vorliegenden Belege."
        if lang == "de"
        else "The search budget is exhausted. Give your answer now using the evidence already available."
    )
    allowed = "tools" in setup
    while True:
        final_only = allowed and len(seen) == MAX_CALLS
        msgs = [*first, *trace]
        if final_only:
            msgs.append({"role": "user", "content": nudge})
        outgoing = {**base, "messages": msgs}
        if allowed and not final_only:
            outgoing.update(tools=setup["tools"], tool_choice="auto")
        raw = request(outgoing)
        choice = _choice(raw)
        message = choice["message"]
        totals = raw.get("usage")
        if isinstance(totals, dict):
            for k in usage:
                if type(totals.get(k)) is int and totals[k] >= 0:
                    usage[k] += totals[k]
        calls = message.get("tool_calls")
        valid = (
            _calls(calls, seen, MAX_CALLS - len(seen))
            if calls
            and allowed
            and not _malformed_message(message)
            and choice.get("finish_reason") in ("tool_calls", "stop")
            else None
        )
        if valid is None:
            # Calls that could not be executed (bad wire ID, disabled tool or exhausted budget)
            # belong to the final turn. Prior authentic observations retain their ordinary credit.
            reply = parse_final(raw, parse_reply)
            reply["observation_trace"] = trace
            reply["tool_rounds"] = sum(m["role"] == "assistant" for m in trace)
            reply["tool_forced"] = final_only
            return reply, usage
        reply = parse_reply(raw)
        trace.append(
            {
                "role": "assistant",
                "content": reply.get("content") or "",
                "tool_calls": valid,
            }
        )
        for c in valid:
            fn = c["function"]
            seen.add(c["id"])
            trace.append(
                {
                    "role": "tool",
                    "tool_call_id": c["id"],
                    "content": _wire(_result(spec, fn["name"], fn["arguments"])),
                }
            )
