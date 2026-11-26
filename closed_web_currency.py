"""Grade time-sensitive claims using only returned search hits or evidence supplied in the question."""

from datetime import date
import re

import closed_web_tools

CONDITIONS = {"current", "historical_given", "unavailable", "unresolved"}
REQUIRES_SEARCH = {"current", "unresolved"}
KNOWN = {"current", "historical_given"}
DETAIL_KEYS = frozenset(
    {
        "valid_json",
        "web_trace_valid",
        "web_claim_correct",
        "web_search_used",
        "web_required_search_used",
        "web_required_sources_found",
        "web_citations_visible",
        "web_decisive_evidence",
        "web_answer_abstained",
        "web_without_tool_correct",
        "web_given_answered_directly",
    }
)


def _text(value, limit):
    return isinstance(value, str) and bool(value.strip()) and len(value) <= limit


def _day(value):
    try:
        return isinstance(value, str) and date.fromisoformat(value).isoformat() == value
    except ValueError:
        return False


def validate_expected(spec):
    if not isinstance(spec, dict) or set(spec) != {
        "condition",
        "as_of",
        "claim",
        "value",
        "evidence",
        "given_ids",
        "docs",
    }:
        return False
    condition, claim = spec["condition"], spec["claim"]
    if (
        not isinstance(condition, str)
        or condition not in CONDITIONS
        or not _day(spec["as_of"])
    ):
        return False
    if (
        not isinstance(claim, dict)
        or set(claim) != {"entity", "field", "scope"}
        or not all(_text(v, 300) for v in claim.values())
    ):
        return False
    if not (_text(spec["value"], 500) if condition in KNOWN else spec["value"] is None):
        return False
    docs = spec["docs"]
    if not isinstance(docs, list) or len(docs) > 100:
        return False
    by_id = {}
    for doc in docs:
        if not isinstance(doc, dict) or set(doc) != {
            "id",
            "title",
            "url",
            "authority",
            "issued",
            "effective_from",
            "effective_until",
            "entity",
            "scope",
            "field",
            "value",
            "text",
        }:
            return False
        uid = doc["id"]
        if (
            not isinstance(uid, str)
            or not re.fullmatch(r"[A-Za-z0-9_-]{1,96}", uid)
            or uid in by_id
        ):
            return False
        if not all(
            _text(doc[k], 8192 if k == "text" else 2048)
            for k in ("title", "url", "authority", "entity", "scope", "field", "text")
        ):
            return False
        if not _day(doc["issued"]) or any(
            doc[k] is not None and not _day(doc[k])
            for k in ("effective_from", "effective_until")
        ):
            return False
        if (
            doc["effective_from"]
            and doc["effective_until"]
            and doc["effective_from"] > doc["effective_until"]
        ):
            return False
        by_id[uid] = doc
    given = spec["given_ids"]
    if not isinstance(given, list) or not all(
        isinstance(k, str) and k in by_id for k in given
    ):
        return False
    if len(given) != len(set(given)) or bool(given) != (
        condition == "historical_given"
    ):
        return False
    evidence = spec["evidence"]
    if not isinstance(evidence, list) or bool(evidence) != (condition != "unavailable"):
        return False
    for entry in evidence:
        if not isinstance(entry, dict) or set(entry) != {"id", "quote"}:
            return False
        uid, quote = entry["id"], entry["quote"]
        if (
            not isinstance(uid, str)
            or uid not in by_id
            or not _text(quote, 8192)
            or quote not in by_id[uid]["text"]
        ):
            return False
        if condition == "historical_given" and uid not in given:
            return False
    return True


def score(item, reply):
    from hidden import strict_json

    spec = item["expected"]
    condition = spec["condition"]
    transcript = reply.get("observation_trace", [])
    seen = closed_web_tools.observed(item, transcript)
    valid_trace = seen["valid"] and not seen["errors"]
    searched = seen["searches"] > 0
    retrieved = seen["documents"]
    # For supplied historical evidence only the material literally present in the prompt is visible.
    question = item["messages"][0]["content"]
    given = {
        d["id"]: d["text"]
        for d in spec["docs"]
        if d["id"] in spec["given_ids"] and d["id"] in question
    }
    required = spec["evidence"]
    detail = {
        "valid_json": 0.0,
        "web_trace_valid": float(valid_trace),
        "web_search_used": float(searched),
        "web_claim_correct": 0.0,
    }
    if condition in REQUIRES_SEARCH:
        detail["web_required_search_used"] = float(valid_trace and searched)
        detail["web_required_sources_found"] = float(
            valid_trace and all(e["id"] in retrieved for e in required)
        )
    if condition != "unavailable":
        detail.update(web_citations_visible=0.0, web_decisive_evidence=0.0)
    if condition not in KNOWN:
        detail["web_answer_abstained"] = 0.0
    if condition == "unavailable":
        detail["web_without_tool_correct"] = 0.0
    if condition == "historical_given":
        detail["web_given_answered_directly"] = 0.0
    try:
        answer = strict_json(reply.get("content"))
    except (TypeError, ValueError, RecursionError):
        return {"score": 0.0, "correct": False, "detail": detail}
    detail["valid_json"] = 1.0
    if (
        not isinstance(answer, dict)
        or set(answer) != {"status", "value", "citations"}
        or not isinstance(answer["citations"], list)
        or len(answer["citations"]) > 100
    ):
        return {"score": 0.0, "correct": False, "detail": detail}

    unknown = answer["status"] == "unknown" and answer["value"] is None
    claim_ok = (
        (answer["status"] == "known" and answer["value"] == spec["value"])
        if condition in KNOWN
        else unknown
    )
    detail["web_claim_correct"] = float(claim_ok)
    if condition not in KNOWN:
        detail["web_answer_abstained"] = float(unknown)
    # A citation must have been visible in this episode. A plausible ID from the private corpus is insufficient.
    cited = {}
    visible = True
    for citation in answer["citations"]:
        if not isinstance(citation, dict) or set(citation) != {"id", "quote"}:
            visible = False
            continue
        uid, quote = citation["id"], citation["quote"]
        if not _text(uid, 96) or not _text(quote, 8192):
            visible = False
            continue
        if not (
            quote in retrieved.get(uid, "")
            or (uid in given and quote in given[uid] and quote in question)
        ):
            visible = False
        cited.setdefault(uid, []).append(quote)
    complete = set(cited) == {e["id"] for e in required} and all(
        any(e["quote"] in quotation for quotation in cited.get(e["id"], ()))
        for e in required
    )
    if required:
        detail["web_citations_visible"] = float(bool(cited) and visible)
        detail["web_decisive_evidence"] = float(visible and complete)
    policy_ok = (
        searched
        if condition in REQUIRES_SEARCH
        else not seen["calls"]
        if condition == "unavailable"
        else True
    )
    correct = bool(
        claim_ok
        and valid_trace
        and visible
        and complete
        and policy_ok
        and not reply.get("web_invalid_final")
        and not reply.get("tool_calls")
    )
    if condition == "unavailable":
        detail["web_without_tool_correct"] = float(correct)
    if condition == "historical_given":
        detail["web_given_answered_directly"] = float(correct and not seen["calls"])
    return {"score": float(correct), "correct": correct, "detail": detail}
