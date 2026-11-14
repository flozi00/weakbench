"""tasks_reasoning (MuSR) on the real pinned data, plain asserts. Stdlib only, no pyarrow:

    python3 test_tasks_reasoning.py
"""
import tasks_reasoning as tr

reply = lambda text: {"content": text, "reasoning": "", "tool_calls": [], "finish_reason": "stop"}  # noqa: E731


def test_counts_and_prefix():
    full = tr.TASKS["musr"]["load"]("en", None, 0)
    assert len(full) == 756 and len({i["id"] for i in full}) == 756
    quick = tr.TASKS["musr"]["load"]("en", 100, 0)
    assert [i["id"] for i in quick] == [i["id"] for i in full][:100]


def test_items_are_well_formed():
    for i in tr.TASKS["musr"]["load"]("en", None, 0):
        assert i["gold"] in i["choices"] and 2 <= len(i["choices"]) <= 5
        prompt = i["messages"][0]["content"]
        assert i["gold"] in prompt and "Answer: <option>" in prompt


def test_scoring():
    items = tr.TASKS["musr"]["load"]("en", 20, 0)
    for i in items:  # the gold option text always scores, in its own casing and with punctuation
        assert tr.score_musr(i, reply(f"reasoning...\nAnswer: {i['gold']}"))["correct"]
        assert tr.score_musr(i, reply(f"Answer: {i['gold'].upper()}."))["correct"]
        letter = "ABCDE"[i["choices"].index(i["gold"])]
        assert tr.score_musr(i, reply(f"Answer: {letter}"))["correct"]
        wrong = next(c for c in i["choices"] if c != i["gold"])
        assert not tr.score_musr(i, reply(f"Answer: {wrong}"))["correct"]
    assert not tr.score_musr(items[0], reply("I do not know."))["correct"]
    assert not tr.score_musr(items[0], reply(""))["correct"]


def test_math500_counts_and_prefix():
    full = tr.TASKS["math_500"]["load"]("en", None, 0)
    assert len(full) == 500 and len({i["id"] for i in full}) == 500
    quick = tr.TASKS["math_500"]["load"]("en", 150, 0)
    assert [i["id"] for i in quick] == [i["id"] for i in full][:150]
    assert {i["subject"] for i in full} == {"Algebra", "Intermediate Algebra", "Prealgebra", "Number Theory",
                                            "Precalculus", "Geometry", "Counting & Probability"}


def test_math500_scoring():
    items = tr.TASKS["math_500"]["load"]("en", None, 0)
    for i in items:  # every gold scores against itself, and the prompt asks for the '####' convention
        assert tr.score_math500(i, reply(f"work...\n#### {i['gold']}"))["correct"]
        assert "####" in i["messages"][0]["content"]
    it = {"gold": "\\frac{3}{4}", "subject": None, "level": None}
    assert tr.score_math500(it, reply("#### 0.75"))["correct"]  # numeric fallback
    assert tr.score_math500(it, reply("so \\boxed{\\frac{3}{4}}"))["correct"]  # no '####': last \boxed
    assert not tr.score_math500(it, reply("#### \\frac{4}{3}"))["correct"]
    assert not tr.score_math500(it, reply("I am not sure."))["correct"]
    assert tr.tex("\\left( 3, \\frac{\\pi}{2} \\right)") == tr.tex("\\left(3,\\frac{\\pi}{2}\\right)")
    assert tr.value("\\dfrac{1}{2}") == 0.5 and tr.value("3/4") == 0.75 and tr.value("\\pi") is None


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
