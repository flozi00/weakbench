"""Tests for tasks_if.py on the real pinned data (first run downloads it + nltk punkt_tab
into common.CACHE; no model endpoint is called).

    uv run --with pyyaml --with langdetect --with immutabledict --with "nltk>=3.9.1" \
        --with pyarrow python test_tasks_if.py      (or: pytest test_tasks_if.py)
"""
import functools
import json
import sys
from collections import Counter
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tasks_if import MT_COLLIDES, MT_KINDS, MT_WHAT, TASKS, mt_holds, mt_kind  # noqa: E402

SEED = 1234


@functools.cache
def ifeval_items(lang):
    return {it["id"]: it for it in TASKS["ifeval"]["load"](lang, None, SEED)}


@functools.cache
def ifstruct_items():
    return TASKS["ifstruct"]["load"]("en", None, SEED)


def judge(lang, key, text):
    return TASKS["ifeval"]["score"](ifeval_items(lang)[str(key)], {"content": text, "reasoning": "",
                                                                    "tool_calls": [], "finish_reason": "stop"})


def test_tasks_contract():
    for t in TASKS.values():
        assert set(t) == {"langs", "load", "score", "think", "max_tokens", "counts", "lang_check"}
        assert set(t["counts"]["quick"]) == set(t["counts"]["full"]) == set(t["langs"])
        assert t["think"] is False
    assert TASKS["ifeval"]["max_tokens"] == 4096 and TASKS["ifstruct"]["max_tokens"] == 12288
    assert TASKS["ifeval"]["counts"]["quick"] == {"de": 214, "en": 214} and TASKS["ifeval"]["lang_check"]
    assert TASKS["ifstruct"]["langs"] == ("en",) and not TASKS["ifstruct"]["lang_check"]


def test_ifeval_load_is_parallel():
    de, en = ifeval_items("de"), ifeval_items("en")
    assert len(de) == len(en) == 429 and list(de) == list(en)
    for i in de:
        assert de[i]["lang"] == "de" and en[i]["lang"] == "en"
        assert de[i]["instruction_id_list"] == en[i]["instruction_id_list"]
        assert de[i]["messages"][0]["role"] == "user"
        assert de[i]["messages"][0]["content"] != en[i]["messages"][0]["content"]
    assert de["1000"]["kwargs"] == [{}, {"num_highlights": 3}, {"relation": "at least", "num_words": 300}]
    other = [i for i in de if "language:response_language" in de[i]["instruction_id_list"]]
    assert other and all(de[i]["lang_check"] is False for i in other)  # replying in Kannada is no mismatch
    assert de["1000"]["lang_check"] is True


def test_ifeval_word_count_limits():
    for lang, word in (("en", "word"), ("de", "Grüße")):
        assert judge(lang, 1072, " ".join([word] * 400))["correct"]  # at least 400 words
        assert not judge(lang, 1072, " ".join([word] * 399))["correct"]
        assert judge(lang, 1092, " ".join([word] * 299))["correct"]  # less than 300 words
        assert not judge(lang, 1092, " ".join([word] * 300))["correct"]


def test_ifeval_keywords_come_from_the_item_language():
    assert ifeval_items("de")["2485"]["kwargs"] == [{"keywords": ["Tinte", "Memoiren"]}]
    assert judge("de", 2485, "Die Tinte der Memoiren ist trocken.")["correct"]
    assert not judge("de", 2485, "The ink of the memoirs is dry.")["correct"]
    assert judge("en", 2485, "The ink of the memoirs is dry.")["correct"]


def test_ifeval_constrained_response_uses_item_language():
    assert judge("de", 3753, "Meine Antwort ist ja.")["correct"]
    assert not judge("de", 3753, "My answer is yes.")["correct"]
    assert judge("en", 3753, "My answer is yes.")["correct"]


def test_ifeval_json_format_strict_vs_loose():
    for lang in ("de", "en"):
        assert judge(lang, 1075, '```json\n{"produkt": "Windel", "preis": 9.99}\n```')["correct"]
        r = judge(lang, 1075, 'Hier ist die Werbung:\n```json\n{"produkt": "Windel"}\n```')
        assert not r["correct"] and r["detail"]["prompt_loose"]  # loose drops the first line
        r = judge(lang, 1075, "Kaufen Sie unsere Windel!")
        assert not r["correct"] and not r["detail"]["prompt_loose"] and r["detail"]["inst_loose_frac"] == 0


def test_ifeval_paragraphs_and_bullets():
    assert judge("de", 1082, "Er wachte auf.\n***\nDas Spiel begann.")["correct"]  # exactly 2, *** divider
    assert not judge("de", 1082, "Er wachte auf. Das Spiel begann.")["correct"]
    assert not judge("en", 1082, "One.\n***\nTwo.\n***\nThree.")["correct"]
    assert judge("en", 102, "* one\n* two\n* three")["correct"]  # exactly 3 bullets
    assert not judge("de", 102, "* eins\n* zwei\n* drei\n* vier")["correct"]


def test_ifeval_response_language_german():
    german = ("Google, Facebook und Microsoft geben jedes Jahr viele Millionen Dollar für Lobbyarbeit in "
              "Washington aus. Amazon und Apple folgen dicht dahinter, mit stark steigenden Ausgaben.")
    english = ("Google, Facebook and Microsoft spend many millions of dollars on lobbying in Washington "
               "every year. Amazon and Apple follow closely behind, with sharply rising spending.")
    for lang in ("de", "en"):
        assert judge(lang, 3682, german)["correct"]
        assert not judge(lang, 3682, english)["correct"]


def test_ifeval_all_lowercase():
    # BenchMAX dropped the English-only change_case items; the checker is still registered.
    item = {"id": "0", "lang": "en", "messages": [{"role": "user", "content": "Answer in lowercase."}],
            "instruction_id_list": ["change_case:english_lowercase"], "kwargs": [{}]}
    score = TASKS["ifeval"]["score"]
    assert score(item, {"content": "the weather is sunny and warm today, so we will walk to the park."})["correct"]
    assert not score(item, {"content": "The weather is sunny and warm today, so we will walk to the park."})["correct"]


def test_ifeval_sentence_count_uses_nltk_punkt():
    few = "Heute war ein langer Tag. Ich habe viel gearbeitet. Am Abend bin ich spazieren gegangen."
    many = ("I woke up early. The sun was bright. I made coffee. The dog barked. I read the news. "
            "Then I wrote my journal. Stress fell away. The day felt calm.")
    assert judge("de", 1268, few)["correct"]  # less than 6 sentences
    assert not judge("en", 1268, many)["correct"]


def test_ifeval_partial_instruction_score():
    text = "*Teil eins* und *Teil zwei* und *Teil drei* " + " ".join(["Wort"] * 300)
    r = judge("de", 1000, text)  # no comma, >= 3 highlights, >= 300 words
    assert r["correct"] and r["score"] == 1 and r["detail"]["inst_strict"] == [True, True, True]
    r = judge("de", 1000, text + ", Ende")
    assert not r["correct"] and r["score"] == 2 / 3 and r["detail"]["inst_strict"] == [False, True, True]
    assert r["pred"] is None and r["gold"] is None


def test_ifeval_empty_and_degenerate_replies():
    assert judge("en", 1000, "")["score"] == 0
    r = judge("en", 1075, "[" * 100_000)  # json.loads recursion; upstream would crash
    assert r["correct"] is False and r["detail"]["inst_strict"] == [False]


def test_subsets_are_deterministic_and_parallel():
    load = TASKS["ifeval"]["load"]
    ids = [it["id"] for it in load("de", 214, SEED)]
    assert len(ids) == 214 and ids == [it["id"] for it in load("de", 214, SEED)]
    assert ids == [it["id"] for it in load("en", 214, SEED)]  # same chosen ids in every language
    assert set(ids) != {it["id"] for it in load("de", 214, SEED + 1)}
    quick = TASKS["ifstruct"]["load"]("en", 300, SEED)
    assert Counter(it["output_format"] for it in quick) == {"json": 150, "yaml": 150}
    assert [it["id"] for it in quick] == [it["id"] for it in TASKS["ifstruct"]["load"]("en", 300, SEED)]
    assert len(ifstruct_items()) == 2000 and len({it["id"] for it in ifstruct_items()}) == 2000


def instance(schema):
    """Smallest value a schema accepts: required fields only, first enum value, minItems."""
    t = schema.get("type")
    t = next(x for x in t if x != "null") if isinstance(t, list) else t
    if "enum" in schema:
        return schema["enum"][0]
    if t == "object":
        props = schema.get("properties", {})
        return {k: instance(props.get(k, {})) for k in schema.get("required", props)}
    if t == "array":
        return [instance(schema.get("items", {})) for _ in range(schema.get("minItems", 1))]
    if t in ("integer", "number"):
        return schema.get("minimum", schema.get("maximum", 1))
    return True if t == "boolean" else "x"


def payload(item, extra_key=False, wrap=None):
    count = item["top_level_count"]
    rows = [instance(item["json_schema"]["items"]) for _ in range(count if isinstance(count, int) else count[0])]
    if extra_key:
        rows[0]["not_in_schema"] = "x"
    return {item["top_level_key"]: rows} if (item["require_wrapper_key"] if wrap is None else wrap) else rows


def reply(item, data, prefix=""):
    body = json.dumps(data, indent=2) if item["output_format"] == "json" else yaml.safe_dump(data, sort_keys=False)
    fence = f"```{item['output_format']}\n{body}\n```" if item["require_code_block"] else body
    return {"content": prefix + fence, "reasoning": "", "tool_calls": [], "finish_reason": "stop"}


def first(**want):
    return next(it for it in ifstruct_items() if all(it[k] == v for k, v in want.items()))


def test_ifstruct_json_rows():
    score = TASKS["ifstruct"]["score"]
    js = first(output_format="json", require_wrapper_key=True, require_no_commentary=True, require_code_block=True)
    ok = score(js, reply(js, payload(js)))
    assert ok == {"correct": True, "score": 1.0, "pred": None, "gold": None, "detail": {"errors": []}}
    r = score(js, reply(js, payload(js), prefix="Here is the JSON you asked for:\n"))
    assert not r["correct"] and r["score"] == 0.0 and "text outside JSON" in r["detail"]["errors"][0]
    r = score(js, reply(js, payload(js, extra_key=True)))
    assert not r["correct"] and "extraneous field 'not_in_schema'" in r["detail"]["errors"]
    r = score(js, reply(js, payload(js, wrap=False)))
    assert not r["correct"] and "Expected wrapped object" in r["detail"]["errors"][0]
    chatty = first(output_format="json", require_no_commentary=False, require_code_block=True)
    assert score(chatty, reply(chatty, payload(chatty), prefix="Sure, here you go:\n"))["correct"]
    r = score(js, {"content": "[\n" * 100_000})  # json recursion; upstream would crash
    assert not r["correct"] and r["detail"]["errors"]


def test_ifstruct_yaml_rows():
    score = TASKS["ifstruct"]["score"]
    ym = first(output_format="yaml", require_code_block=True)
    assert score(ym, reply(ym, payload(ym)))["correct"]
    as_json = {"content": "```yaml\n" + json.dumps(payload(ym)) + "\n```"}
    r = score(ym, as_json)
    assert not r["correct"] and "JSON/flow style" in r["detail"]["errors"][0]


def test_every_real_item_scores():
    for lang in ("de", "en"):  # real kwargs (regex keywords, section splitters) never crash a checker
        for it in ifeval_items(lang).values():
            for text in ("", it["messages"][0]["content"]):
                assert isinstance(TASKS["ifeval"]["score"](it, {"content": text})["correct"], bool)
    for it in ifstruct_items():  # every row's flags map right: a schema-minimal reply passes, "" fails
        assert TASKS["ifstruct"]["score"](it, reply(it, payload(it)))["correct"], it["id"]
        assert not TASKS["ifstruct"]["score"](it, {"content": ""})["correct"]


def test_ifstruct_de_is_paired_and_scores():
    """The German cell is the same items behind a German prompt, and the same validator scores it."""
    de = TASKS["ifstruct_de"]["load"]("de", 300, SEED)
    en = TASKS["ifstruct"]["load"]("en", 300, SEED)
    assert [it["id"] for it in de] == [it["id"] for it in en]  # item-paired, same order
    assert {it["lang"] for it in de} == {"de"}
    for d, e in zip(de, en):
        assert d["json_schema"] == e["json_schema"] and d["top_level_count"] == e["top_level_count"]
        assert d["messages"][0]["content"] != e["messages"][0]["content"]
        assert "Erstelle" in d["messages"][0]["content"] and "Create" not in d["messages"][0]["content"]
    score = TASKS["ifstruct_de"]["score"]
    for it in de[:20]:  # a conforming reply passes the same validator
        assert score(it, reply(it, payload(it)))["correct"], it["id"]
    js = next(it for it in de if it["require_wrapper_key"])
    assert not score(js, reply(js, payload(js, wrap=False)))["correct"]  # wrapper still enforced


def test_multiturn_turns_and_scoring():
    """Three turns over the IFEval items: the conversation shape, the pairing, and the two signals."""
    en = TASKS["multiturn"]["load"]("en", None, SEED)
    de = TASKS["multiturn"]["load"]("de", None, SEED)
    assert len(en) == len(de) == 166 and [i["id"] for i in en] == [i["id"] for i in de]  # item-paired
    assert [i["id"] for i in TASKS["multiturn"]["load"]("en", 150, SEED)] == [i["id"] for i in en][:150]
    for i in en:
        assert len(i["turns"]) == 3 and i["turns"][0] == i["messages"]  # turn 1 is the prompt verbatim
        assert len(i["instruction_id_list"]) >= 2
        # the added constraint never collides with one the item already carries, or turn 3 would contradict
        assert MT_COLLIDES[i["mt_kind"]] not in i["instruction_id_list"]
        assert i["mt_kind"] == mt_kind(i["id"], i["instruction_id_list"])
    # the same item gets the same kind in both languages, so the DE/EN cells stay paired
    assert [i["mt_kind"] for i in en] == [i["mt_kind"] for i in de]
    # all four kinds are actually used, and each one's sentence names its own constraint
    kinds = Counter(i["mt_kind"] for i in en)
    assert set(kinds) == set(MT_KINDS), kinds
    for i in en:
        assert MT_WHAT[i["mt_kind"]]["en"] in i["turns"][2][0]["content"]
        assert MT_WHAT[i["mt_kind"]]["de"] in next(
            d["turns"][2][0]["content"] for d in de if d["id"] == i["id"])
    it = en[0]
    # a final answer that keeps nothing but drops the constraint: score = (held + dropped) / (n + 1)
    r = TASKS["multiturn"]["score"](it, {"turns": ["x", "x", "x"]})
    assert r["detail"]["stale"] == 0.0 and not r["correct"]
    assert r["score"] == round((sum(r["detail"]["inst_strict"]) + 1) / (len(it["instruction_id_list"]) + 1), 4)
    # the dropped constraint still present: stale, and it costs one point
    stale_reply = {"done": "x DONE", "upper": "X", "bullets": "- a\n- b\n- c", "nocomma": "abc"}[it["mt_kind"]]
    r2 = TASKS["multiturn"]["score"](it, {"turns": ["x", "x", stale_reply]})
    assert r2["detail"]["stale"] == 1.0 and r2["score"] < r["score"]
    # forgot: a constraint that held at turn 1 and not at turn 3
    words = next(i for i in en if any(x.startswith("length_constraints:number_words") and
                                      kw.get("relation") == "at least"
                                      for x, kw in zip(i["instruction_id_list"], i["kwargs"])))
    long = "word " * 500  # satisfies an "at least N words" constraint
    r3 = TASKS["multiturn"]["score"](words, {"turns": [long, long, "x"]})
    assert r3["detail"]["t1_strict"] > r3["detail"]["t3_strict"] and r3["detail"]["forgot"] > 0


def test_mt_holds_per_kind():
    """Each kind's checker, both directions — `stale` is only as good as these."""
    assert mt_holds("done", "all good\nDONE") and not mt_holds("done", "all good")
    assert mt_holds("upper", "ALL CAPS") and not mt_holds("upper", "Mixed Case")
    assert not mt_holds("upper", "   ")  # empty is not "uppercase"
    assert mt_holds("bullets", "- a\n- b\n- c") and mt_holds("bullets", "1. a\n2. b\n3. c")
    assert not mt_holds("bullets", "- a\n- b")
    assert mt_holds("nocomma", "no punctuation here") and not mt_holds("nocomma", "a, b")
    # the collision rule: whatever kind an item gets, it never carries that kind's instruction
    for ids in ([], ["change_case:all_upper_case"], ["change_case:all_upper_case", "detectable_format:number_bullet_lists"],
                ["punctuation:no_comma"], list(MT_COLLIDES.values())):
        k = mt_kind("k", [i for i in ids if i])
        assert MT_COLLIDES[k] is None or MT_COLLIDES[k] not in ids, (ids, k)
    # and the four kinds are spread, not all `done`
    spread = Counter(mt_kind(str(n), []) for n in range(400))
    assert set(spread) == set(MT_KINDS) and min(spread.values()) > 60, spread


def test_german_json_schema_and_content():
    """German structured output: the schema comes from the prompt, and content is scored, not just shape."""
    full = TASKS["german_json"]["load"]("de", None, SEED)
    quick = TASKS["german_json"]["load"]("de", 150, SEED)
    assert len(full) == 202 and [i["id"] for i in quick] == [i["id"] for i in full][:150]
    assert all(i["lang"] == "de" for i in full)
    assert Counter(i["task_type"] for i in full) == Counter(
        {"ner_extraction": 102, "relation_extraction": 74, "gdpr_anonymization": 25, "schema_following": 1})
    # the system prompt is German and carries the schema block the scorer reads
    for i in full:
        assert i["messages"][0]["role"] == "system" and "JSON-Schema" in i["messages"][0]["content"]
        assert i["required"] == ts_required(i["messages"][0]["content"])
    it = next(i for i in full if i["task_type"] == "ner_extraction")
    assert it["required"] == ["entities"]
    score = TASKS["german_json"]["score"]
    # the gold itself is a perfect reply
    r = score(it, {"content": it["gold"]})
    assert r["correct"] and r["score"] == 1.0 and r["detail"]["content_f1"] == 1.0
    # fenced and prose-embedded JSON parse the same
    assert score(it, {"content": "```json\n" + it["gold"] + "\n```"})["score"] == 1.0
    assert score(it, {"content": "Hier: " + it["gold"] + " fertig."})["score"] == 1.0
    # schema-only is NOT a win: it parses and obeys the schema but recovers no content
    r = score(it, {"content": '{"entities": []}'})
    assert r["detail"]["schema_ok"] == 1.0 and r["detail"]["content_f1"] == 0.0
    assert not r["correct"] and r["score"] == 0.5
    # no JSON at all scores zero
    r = score(it, {"content": "kein json"})
    assert not r["correct"] and r["score"] == 0.0 and r["detail"]["schema_ok"] == 0.0


def ts_required(system):
    """The `required` list of the ```json block, recomputed here so the test does not trust the loader."""
    import re
    m = re.search(r"```json\s*(\{.*?\})\s*```", system, re.S)
    return list(json.loads(m.group(1)).get("required") or []) if m else []


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    print("all tasks_if tests passed")
