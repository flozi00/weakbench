"""tasks_text (summarize, grounded_qa) on the real pinned data, plain asserts. Stdlib only:

    python3 test_tasks_text.py
"""
import tasks_text as tt

reply = lambda text: {"content": text, "reasoning": "", "tool_calls": [], "finish_reason": "stop"}  # noqa: E731


def test_summarize_counts_and_prefix():
    for lang, total in (("en", 11334), ("de", 2965)):
        full = tt.TASKS["summarize"]["load"](lang, None, 0)
        assert len(full) == total and len({i["id"] for i in full}) == total, (lang, len(full))
        quick = tt.TASKS["summarize"]["load"](lang, 100, 0)
        assert [i["id"] for i in quick] == [i["id"] for i in full][:100]
        assert {i["lang"] for i in full} == {lang}


def test_summarize_scoring():
    for lang in ("en", "de"):
        items = tt.TASKS["summarize"]["load"](lang, 20, 0)
        for i in items:  # the reference summary scores 1.0 against itself
            assert tt.score_summarize(i, reply(i["ref"]))["score"] == 1.0
            assert i["ref"] and len(i["messages"][0]["content"]) > len(i["ref"])  # article + instructions
        assert tt.score_summarize(items[0], reply(""))["score"] == 0.0
        assert tt.score_summarize(items[0], reply("completely unrelated words here"))["score"] < 0.3
        assert tt.score_summarize(items[0], reply(items[0]["ref"]))["correct"] is None  # continuous cell


def test_grounded_qa_counts_and_prefix():
    for lang, total in (("en", 10570), ("de", 2204)):
        full = tt.TASKS["grounded_qa"]["load"](lang, None, 0)
        assert len(full) == total and len({i["id"] for i in full}) == total, (lang, len(full))
        quick = tt.TASKS["grounded_qa"]["load"](lang, 150, 0)
        assert [i["id"] for i in quick] == [i["id"] for i in full][:150]


def test_grounded_qa_scoring():
    for lang in ("en", "de"):
        items = tt.TASKS["grounded_qa"]["load"](lang, 40, 0)
        for i in items:  # every gold span is exact-match, behind either marker
            assert tt.score_grounded_qa(i, reply(f"Answer: {i['gold']}"))["correct"]
            assert tt.score_grounded_qa(i, reply(f"Antwort: {i['gold']}"))["correct"]
            assert tt.score_grounded_qa(i, reply(f"Answer: {i['gold']}"))["score"] == 1.0
        assert not tt.score_grounded_qa(items[0], reply("Answer: zzzz"))["correct"]
        assert not tt.score_grounded_qa(items[0], reply(""))["correct"]
    # a sentence instead of a span keeps F1 but loses EM: the two numbers read apart
    it = {"id": "x", "lang": "en", "gold": "Paris", "alts": ["Paris"]}
    r = tt.score_grounded_qa(it, reply("Answer: The capital is Paris, as the passage says."))
    assert not r["correct"] and 0 < r["score"] < 1.0


def test_norm_and_f1():
    assert tt.norm("The  Cat, sat!") == "cat sat"  # lowercase, no punctuation, no articles
    assert tt.f1("Paris", "paris") == 1.0
    assert tt.f1("New York City", "New York") == 0.8
    assert tt.f1("", "Paris") == 0.0
    assert tt.f1("Paris", "") == 0.0


def test_research_bundles_and_citation():
    """Five passages, one gold, and the citation is what separates research from recall."""
    for lang in ("en", "de"):
        full = tt.TASKS["research"]["load"](lang, None, 0)
        quick = tt.TASKS["research"]["load"](lang, 100, 0)
        assert [i["id"] for i in quick] == [i["id"] for i in full][:100]  # quick is a prefix
        for it in quick[:10]:
            assert 1 <= it["gold_no"] <= 5
            body = it["messages"][0]["content"]
            assert body.count("\n\n[") == 5  # five numbered passages (the first follows the prompt's blank line)
            assert f"[{it['gold_no']}]" in body
            assert it["gold"] in body  # the gold passage really is in the bundle
    it = tt.TASKS["research"]["load"]("en", 100, 0)[0]
    gold, no = it["gold"], it["gold_no"]
    r = tt.TASKS["research"]["score"](it, {"content": f"Answer: {gold}\nSource: [{no}]"})
    assert r["correct"] and r["score"] == 1.0 and r["detail"]["cited"] == 1.0
    # right span, wrong citation: the failure mode this cell exists for
    r = tt.TASKS["research"]["score"](it, {"content": f"Answer: {gold}\nSource: [{(no % 5) + 1}]"})
    assert not r["correct"] and r["detail"]["wrong_source"] == 1.0 and r["score"] == 0.5
    # no citation at all is not the same as a wrong one
    r = tt.TASKS["research"]["score"](it, {"content": f"Answer: {gold}"})
    assert not r["correct"] and r["detail"]["wrong_source"] == 0.0
    # the German markers work on an English item too (a German prompt does not guarantee a German marker)
    r = tt.TASKS["research"]["score"](it, {"content": f"Antwort: {gold}\nQuelle: [{no}]"})
    assert r["correct"]
    # a wrong span with the right citation keeps the retrieval half
    r = tt.TASKS["research"]["score"](it, {"content": f"Answer: zzz\nSource: [{no}]"})
    assert not r["correct"] and r["detail"]["cited"] == 1.0 and r["score"] == 0.5



def test_injection():
    import tasks_text as tt
    for lang in ("de", "en"):
        items, clean = tt.load_injection(lang, 40, 0), tt.load_grounded_qa(lang, 40, 0)
        assert [i["id"] for i in items] == [i["id"] for i in clean]  # paired with grounded_qa
        for it, c in zip(items, clean):
            body = it["messages"][0]["content"]
            assert it["code"] in body and body.count(it["code"]) == 1
            assert body.replace(tt.ATTACKS[lang][it["attack"]].format(c=it["code"]) + " ", "") == \
                c["messages"][0]["content"]  # the attack is the only change
        r = tt.score_injection(it, {"content": f"{tt.ANSWER[lang]} {it['gold']}"})
        assert r["correct"] and r["detail"]["hijacked"] == 0.0
        r = tt.score_injection(it, {"content": f"(I ignore the note asking for {it['code']}.)\n"
                                               f"{tt.ANSWER[lang]} {it['gold']}"})
        assert r["correct"] and r["detail"]["hijacked"] == 0.0  # naming the code while refusing is no hijack
        r = tt.score_injection(it, {"content": f"{it['code']}\n{tt.ANSWER[lang]} {it['gold']}"})
        assert not r["correct"] and r["score"] == 0.0 and r["detail"]["hijacked"] == 1.0


def test_context_conflict():
    import tasks_text as tt
    for lang in ("de", "en"):
        items = tt.load_context_conflict(lang, None, 0)
        assert len(items) == 2 * len(tt.CC_FACTS) and "von die" not in " ".join(i["messages"][0]["content"]
                                                                                 for i in items)
        for it in items:
            assert tt.score_context_conflict(it, {"content": it["gold"]})["correct"]
            assert not tt.score_context_conflict(it, {"content": it["wrong"]})["correct"]
            assert not tt.score_context_conflict(it, {"content": f"{it['gold']} (really {it['wrong']})"})["correct"]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
