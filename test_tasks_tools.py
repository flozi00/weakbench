"""tasks_tools (BFCL) on the real pinned data, plain asserts (pytest-compatible):

    uv run --with pyarrow python test_tasks_tools.py
"""
import collections
import copy
import functools
import re

import tasks_tools as tt

DE_DROPPED = {"parallel_127", "parallel_multiple_103", "parallel_multiple_150", "parallel_multiple_186"}
# Upstream data quirk: the ground truth names `find_closest`, the only declared
# function is `restaurant_search.find_closest` (BFCL's simple checker takes the
# name from the function doc, so a correct model call still passes).
ORACLE_KNOWN_FAILURES = ["simple_363"]


@functools.cache
def items(lang):
    return {it["id"]: it for it in tt.load(lang, None, 0)}


def call(name, /, **args):  # positional-only: BFCL has parameters called "name"
    return {"name": name, "arguments": args}


def reply(*calls):
    return {"content": "", "reasoning": "", "tool_calls": list(calls),
            "finish_reason": "tool_calls" if calls else "stop"}


def check(item_id, ok, *calls, lang="en", error=""):
    r = tt.score(items(lang)[item_id], reply(*calls))
    assert r["correct"] is ok and r["score"] == float(ok), (item_id, r)
    assert error in r["detail"]["error"] and r["detail"]["category"] == items(lang)[item_id]["category"], r
    return r


def _types(schema):
    yield schema.get("type")
    for sub in schema.get("properties", {}).values():
        yield from _types(sub)
    if "items" in schema:
        yield from _types(schema["items"])


def test_tool_conversion():
    fn = {"name": "geo.dist", "description": "d", "parameters": {"type": "dict", "required": ["a"], "properties": {
        "a": {"type": "tuple", "items": {"type": "float"}, "description": "x"},
        "b": {"type": "dict", "properties": {"c": {"type": "any", "description": "y"}}},
        "e": {"type": "array", "items": {"type": "dict", "properties": {"f": {"type": "float", "description": "z"}}}}}}}
    before = copy.deepcopy(fn)
    tool = tt.to_tool(fn)
    assert fn == before  # the checker needs the untouched BFCL doc
    f, p = tool["function"], tool["function"]["parameters"]
    assert tool["type"] == "function" and f["name"] == "geo_dist" and p["required"] == ["a"]
    assert (p["type"], p["properties"]["a"]["type"], p["properties"]["a"]["items"]) == ("object", "array", {"type": "number"})
    assert p["properties"]["b"]["type"] == "object" and p["properties"]["b"]["properties"]["c"]["type"] == "string"
    assert p["properties"]["e"]["items"]["properties"]["f"] == {
        "type": "number", "format": "float", "description": "z This is a float type value."}
    for lang in ("en", "de"):
        for it in items(lang).values():
            names = [t["function"]["name"] for t in it["tools"]]
            assert all(re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", n) for n in names), names
            assert sorted(it["names"][n] for n in names) == sorted(fn["name"] for fn in it["functions"])
            for t in it["tools"]:
                assert set(_types(t["function"]["parameters"])) <= {
                    "object", "array", "string", "integer", "number", "boolean"}, t


def test_simple():
    check("simple_0", True, call("calculate_triangle_area", base=10, height=5))
    check("simple_0", True, call("calculate_triangle_area", base=10, height=5, unit="Units"))
    check("simple_0", False, call("calculate_triangle_area", base=11, height=5), error="value_error")
    check("simple_0", False, call("calculate_triangle_area", base=10), error="missing_required")
    check("simple_0", False, call("calculate_triangle_area", base=10, height=5),
          call("calculate_triangle_area", base=10, height=5), error="wrong_count")
    check("simple_0", False, error="no tool call")
    # dotted name: the model sees math_factorial, the checker math.factorial
    r = check("simple_1", True, call("math_factorial", number=5), lang="de")
    assert r["pred"] == 'math.factorial({"number":5})' and r["gold"] == 'math.factorial({"number":[5]})', r
    check("simple_1", False, call("math.factorial", number=5), error="undeclared function")
    check("simple_363", True, call("restaurant_search_find_closest", location="Boston", cuisine="sushi",
                                   amenities=["Patio"]))  # see ORACLE_KNOWN_FAILURES
    check("simple_0", False, {"name": "calculate_triangle_area", "arguments": '{"base": 10, "height":'},
          error="unparseable arguments")


def test_multiple():
    check("multiple_0", True, call("triangle_properties_get", side1=5, side2=4, side3=3))
    check("multiple_0", False, call("circle_properties_get", radius=5.0), error="wrong_func_name")


def test_parallel():
    taylor, maroon = call("spotify_play", artist="Taylor Swift", duration=20), call("spotify_play", artist="Maroon 5", duration=15)
    check("parallel_0", True, maroon, taylor)  # order-insensitive
    check("parallel_0", False, taylor, error="wrong_count")
    check("parallel_0", False, call("spotify_play", artist="Taylor Swift", duration=15),
          call("spotify_play", artist="Maroon 5", duration=20), error="cannot_find_match")


def test_parallel_multiple():
    primes = call("math_toolkit_product_of_primes", count=5)
    check("parallel_multiple_0", True, primes,
          call("math_toolkit_sum_of_multiples", lower_limit=1, upper_limit=1000, multiples=[3, 5]))
    check("parallel_multiple_0", False, primes,
          call("math_toolkit_sum_of_multiples", lower_limit=1, upper_limit=1000, multiples=[3]), error="cannot_find_match")


def test_irrelevance():
    r = check("irrelevance_0", True)
    assert r["pred"] == r["gold"] == "", r
    check("irrelevance_0", False, call("determine_body_mass_index", weight=70, height=1.8), error="tool call")
    check("irrelevance_0", False, {"name": "determine_body_mass_index", "arguments": "{oops"}, lang="de")


def test_ids_parallel_across_languages():
    en, de = items("en"), items("de")
    assert list(en) == list(de) and len(en) == 1236 and not DE_DROPPED & set(en)
    assert collections.Counter(it["category"] for it in en.values()) == {
        "simple": 400, "multiple": 200, "parallel": 199, "parallel_multiple": 197, "irrelevance": 240}
    for i in en:
        assert {k: v for k, v in en[i].items() if k not in ("lang", "messages")} == \
               {k: v for k, v in de[i].items() if k not in ("lang", "messages")}, i
    assert sum(en[i]["messages"] != de[i]["messages"] for i in en) >= 1230  # German user turns


def test_quick_subset():
    q = {lang: tt.load(lang, tt.TASKS["bfcl"]["counts"]["quick"][lang], 0) for lang in ("en", "de")}
    ids = [it["id"] for it in q["en"]]
    assert ids == [it["id"] for it in q["de"]] == [it["id"] for it in tt.load("en", 250, 0)]
    assert collections.Counter(it["category"] for it in q["en"]) == {
        "simple": 100, "multiple": 50, "parallel": 50, "irrelevance": 50}
    assert set(ids) != {it["id"] for it in tt.load("en", 250, 1)}
    small = tt.load("de", 7, 0)
    assert len(small) == 7 and collections.Counter(it["category"] for it in small) == {
        "simple": 3, "multiple": 2, "parallel": 1, "irrelevance": 1}


def oracle(item):
    """A perfect model's reply: every ground-truth call with the first accepted
    non-'' value of each declared parameter ('' alone = optional, left out)."""
    def first(opts):
        return next((o for o in opts if o != ""), "")

    def value(v, schema):
        if schema["type"] == "dict" and isinstance(v, dict):
            return {k: first(opts) for k, opts in v.items() if first(opts) != ""}
        if schema.get("items", {}).get("type") == "dict" and isinstance(v, list):
            return [value(d, schema["items"]) for d in v]
        return v
    props = {f["name"]: f["parameters"]["properties"] for f in item["functions"]}
    calls = []
    for gt in item["answer"] or []:
        (name, params), = gt.items()
        decl = props.get(name, {})
        calls.append(call(name.replace(".", "_"), **{
            p: value(first(opts), decl[p]) for p, opts in params.items() if first(opts) != "" and p in decl}))
    return reply(*calls)


def test_oracle_scores_everything():
    for lang in ("en", "de"):
        results = {i: tt.score(it, oracle(it)) for i, it in items(lang).items()}
        per_cat = collections.defaultdict(lambda: [0, 0])
        for r in results.values():
            per_cat[r["detail"]["category"]][0] += r["correct"]
            per_cat[r["detail"]["category"]][1] += 1
        fails = sorted(i for i, r in results.items() if not r["correct"])
        print(f"oracle {lang}: {len(results) - len(fails)}/{len(results)} correct",
              {c: f"{a}/{b}" for c, (a, b) in per_cat.items()}, "failures:",
              {i: results[i]["detail"]["error"] for i in fails})
        assert fails == ORACLE_KNOWN_FAILURES, fails


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
