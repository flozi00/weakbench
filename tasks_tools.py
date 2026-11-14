"""Tool calling: BFCL v3 non-live AST categories in English and German (stdlib).

EN  gorilla-llm/Berkeley-Function-Calling-Leaderboard: simple 400, multiple 200,
    parallel 200, parallel_multiple 200, irrelevance 240 (+ possible_answer/).
DE  hafsteinn/bfcl-v3-multilingual bfcl/de/: only the user turn is translated;
    functions and ground truth are the English ones. Its 4 dropped ids
    (parallel_127, parallel_multiple_103/150/186) leave BOTH languages, so the
    1,236 ids pair item by item.

Prompting is BFCL's OpenAI function-calling handler: the first turn's messages,
no system prompt, the function docs as OpenAI tools (BFCL types mapped to JSON
Schema, "." in names -> "_", BFCL's Python-syntax and float hints). Scoring is
BFCL's AST checker (vendor/bfcl_ast_checker.py) on the calls with the original
names restored; irrelevance is correct iff the model made no tool call.
"""
import json
import random

import common
from vendor.bfcl_ast_checker import ast_checker

EN = ("gorilla-llm/Berkeley-Function-Calling-Leaderboard", "61fc0608cfd831fcfbbaa676ebdfef0ed963eeda")
DE = ("hafsteinn/bfcl-v3-multilingual", "43b71a637dfe264b68fec0839d6b3101a8e59d23")
CATEGORIES = ("simple", "multiple", "parallel", "parallel_multiple", "irrelevance")
QUICK = {"simple": 100, "multiple": 50, "parallel": 50, "irrelevance": 50}
QUICK_N = sum(QUICK.values())
TYPES = {"dict": "object", "float": "number", "tuple": "array", "any": "string"}


def _rows(repo, name):
    # BFCL's .json files are JSON lines, which read_rows would json.load whole.
    return common.read_jsonl(common.hf_file(*repo, name))


def _schema(s, prop=False):
    """BFCL parameter schema -> JSON Schema, recursively (properties, items)."""
    out = dict(s)
    if prop and s.get("type") == "float":  # BFCL's OpenAI handler hints float properties
        out["format"] = "float"
        out["description"] = s.get("description", "") + " This is a float type value."
    if "type" in s:
        out["type"] = TYPES.get(s["type"], s["type"])
    if "properties" in s:
        out["properties"] = {k: _schema(v, True) for k, v in s["properties"].items()}
    if "items" in s:
        out["items"] = _schema(s["items"])
    return out


def to_tool(fn):
    """BFCL function doc -> OpenAI tool, as BFCL's OpenAI FC handler sends it."""
    return {"type": "function", "function": {
        "name": fn["name"].replace(".", "_"),
        "description": fn["description"] + " Note that the provided function is in Python 3 syntax.",
        "parameters": _schema(fn["parameters"])}}


def _items(lang):
    out = []
    for cat in CATEGORIES:
        name = f"BFCL_v3_{cat}.json"
        de = {r["id"]: r for r in _rows(DE, "bfcl/de/" + name)}
        gold = {} if cat == "irrelevance" else {
            r["id"]: r["ground_truth"] for r in _rows(EN, "possible_answer/" + name)}
        for row in _rows(EN, name):
            if row["id"] not in de:
                continue
            fns = row["function"]
            out.append({"id": row["id"], "lang": lang, "category": cat,
                        "messages": (de[row["id"]] if lang == "de" else row)["question"][0],
                        "tools": [to_tool(f) for f in fns], "tool_choice": "auto",
                        "functions": fns, "answer": gold.get(row["id"]),
                        "names": {f["name"].replace(".", "_"): f["name"] for f in fns}})
    return out


def load(lang, n, seed):
    """All items (n None), else n items stratified like the quick mix
    (n=250: simple 100 / multiple 50 / parallel 50 / irrelevance 50); the
    sample depends only on (n, seed), so en and de get the same ids."""
    items = _items(lang)
    if n is None:
        return items
    quota = {c: n * q // QUICK_N for c, q in QUICK.items()}
    for c in list(QUICK)[:n - sum(quota.values())]:  # rounding remainder (< 4)
        quota[c] += 1
    keep = set()
    for cat, k in quota.items():
        ids = sorted(it["id"] for it in items if it["category"] == cat)
        keep.update(random.Random(seed).sample(ids, min(k, len(ids))))
    return [it for it in items if it["id"] in keep]


def _fmt(calls):
    return "; ".join(f"{name}({json.dumps(args, ensure_ascii=False, separators=(',', ':'))})"
                     for name, args in calls)


def _check(item, calls):
    """'' when the calls pass BFCL's AST checker, else the reason."""
    if not calls:
        return "no tool call"
    for c in calls:
        if not isinstance(c.get("arguments"), dict):
            return f"unparseable arguments for {c.get('name')!r}: {str(c.get('arguments'))[:200]}"
        if c.get("name") not in item["names"]:
            return f"undeclared function {c.get('name')!r}"
    decoded = [{item["names"][c["name"]]: c["arguments"]} for c in calls]
    try:
        r = ast_checker(item["functions"], decoded, item["answer"], item["category"])
    except Exception as exc:  # noqa: BLE001 — malformed model output must not kill the run
        return f"checker error: {exc!r}"
    return "" if r["valid"] else f"{r.get('error_type')}: " + "; ".join(map(str, r["error"]))[:500]


def score(item, reply):
    calls = reply.get("tool_calls") or []
    if item["category"] == "irrelevance":
        err = f"{len(calls)} tool call(s) on an irrelevant request" if calls else ""
    else:
        err = _check(item, calls)
    return {"correct": not err, "score": float(not err),
            "pred": _fmt((item["names"].get(c.get("name"), c.get("name")), c.get("arguments")) for c in calls),
            "gold": _fmt(next(iter(g.items())) for g in item["answer"] or []),
            "detail": {"category": item["category"], "error": err}}


TASKS = {"bfcl": {"langs": ("en", "de"), "load": load, "score": score, "think": False, "max_tokens": 1024,
                  "counts": {"quick": {"en": QUICK_N, "de": QUICK_N}, "full": {"en": None, "de": None}},
                  "lang_check": False}}
