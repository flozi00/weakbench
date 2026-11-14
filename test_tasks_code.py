"""tasks_code (CRUXEval-O, LiveCodeBench output) and TruthfulQA on the real pinned data, plain asserts:

    uv run --with pyarrow --with pyyaml --with langdetect --with immutabledict --with 'nltk>=3.9.1' python test_tasks_code.py
"""
import concurrent.futures
import functools
import time

import quality
import tasks_code as tc

reply = lambda text: {"content": text, "reasoning": "", "tool_calls": [], "finish_reason": "stop"}  # noqa: E731


@functools.cache
def items(name):
    return tc.TASKS[name]["load"]("en", None, 0)


def test_counts_and_prefix():
    assert len(items("cruxeval_o")) == 800 and len(items("lcb_output")) == 442
    assert len({i["id"] for i in items("lcb_output")}) == 442
    quick = tc.TASKS["lcb_output"]["load"]("en", 100, 0)
    assert [i["id"] for i in quick] == [i["id"] for i in items("lcb_output")][:100]


def test_oracle_and_negatives():
    for name in ("cruxeval_o", "lcb_output"):
        bad = [i["id"] for i in items(name) if not tc.score(i, reply(f"thinking...\nOutput: {i['gold']}"))["correct"]]
        assert not bad, (name, bad[:5])  # gold in its own format always scores
        i = items(name)[0]
        assert not tc.score(i, reply("no answer line"))["correct"]
        assert not tc.score(i, reply("Output: 'definitely wrong ☃'"))["correct"]
        assert tc.score(i, reply(f"Output: A\nOutput: `{i['gold']}`"))["correct"]  # the last line counts, fences are fine


def test_value_equivalence():
    assert tc.value('"ab"') == tc.value("'ab'") == "ab" and tc.value("[1, 2]") == tc.value("[1,2]")
    assert tc.value("{'a': 1, 'b': 2}") == tc.value('{"b": 2, "a": 1}')
    assert tc.value("(((" * 5000) == "(((" * 5000  # unparseable and deeply nested: text, no crash
    assert tc.value("[1]") != tc.value("[2]") and tc.value("(1, 2)") != tc.value("[1, 2]")


def run_oracle(name):
    """Every canonical solution, as a model would answer it, through the real executor (in parallel)."""
    task = tc.TASKS[name]
    rows = {str(r["task_id"]): r for r in tc.read_rows(tc.hf_file(*(tc.HUMANEVAL if name == "humaneval_plus" else tc.MBPP), (
        "data/test-00000-of-00001-5973903632b82d40.parquet" if name == "humaneval_plus" else
        "data/test-00000-of-00001-d5781c9c51e02795.parquet")))}
    its = task["load"]("en", None, 0)
    assert len(its) == len(rows) - len(tc.UNUSABLE & set(rows))
    ref = {i: (r["prompt"] + r["canonical_solution"] if name == "humaneval_plus" else r["code"]) for i, r in rows.items()}
    with concurrent.futures.ThreadPoolExecutor(16) as ex:
        res = list(ex.map(lambda it: (it["id"], task["score"](it, reply(f"```python\n{ref[it['id']]}\n```"))), its))
    return [(i, r["detail"]["error"]) for i, r in res if not r["correct"]]


def test_exec_oracle_humaneval_plus():
    assert run_oracle("humaneval_plus") == []


def test_exec_oracle_mbpp_plus():
    assert run_oracle("mbpp_plus") == []


def test_exec_failures_and_limits():
    he = {i["id"]: i for i in tc.TASKS["humaneval_plus"]["load"]("en", None, 0)}["HumanEval/0"]
    s = lambda text: tc.TASKS["humaneval_plus"]["score"](he, reply(text))  # noqa: E731
    wrong = "```python\ndef has_close_elements(numbers, threshold):\n    return False\n```"
    assert not s(wrong)["correct"] and "AssertionError" in s(wrong)["detail"]["error"]
    assert not s("no code at all")["correct"]
    body_only = "```python\n    return any(abs(a - b) < threshold for i, a in enumerate(numbers) for b in numbers[i + 1:])\n```"
    assert not s(body_only)["correct"]  # a bare body is no function: fail, no crash
    t0 = time.monotonic()
    tc.WALL_S = 2
    try:
        loop = s("```python\ndef has_close_elements(numbers, threshold):\n    while True:\n        pass\n```")
    finally:
        tc.WALL_S = 30
    assert not loop["correct"] and "timeout" in loop["detail"]["error"] and time.monotonic() - t0 < 10
    # an annotation that needs `from typing import List` still works when the reply drops the import
    typed = ("```python\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n"
             "    return any(abs(a - b) < threshold for i, a in enumerate(numbers) for b in numbers[i + 1:])\n```")
    assert s(typed)["correct"]
    ok, err = tc.run_python("import os\nassert os.getcwd().startswith(('/tmp', '/var', '/private')), os.getcwd()\n"
                            "import numpy as np\nassert np.allclose([1.0, 2.0], [1.0, 2.0 + 1e-9], rtol=1e-7, atol=0)\n")
    assert ok, err


def test_cruxeval_x():
    """Gold recovery from the row itself, balanced sampling, and the two-tier literal comparison."""
    full = tc.TASKS["cruxeval_x"]["load"]("en", None, 0)
    per = {p: sum(1 for i in full if i["id"].startswith(p + "/")) for p in tc.CRUX_X_LANGS}
    assert min(per.values()) > 600, per  # every language is present in the full set
    assert len({i["id"] for i in full}) == len(full)
    quick = tc.TASKS["cruxeval_x"]["load"]("en", 160, 0)
    assert len(quick) == 160 and [i["id"] for i in quick] == [i["id"] for i in full][:160]
    sampled = {p: sum(1 for i in quick if i["id"].startswith(p + "/")) for p in tc.CRUX_X_LANGS}
    assert set(sampled.values()) == {20}, sampled  # the sample is balanced, the pool is not
    for i in full:  # the program block carries exactly one mask, and the instruction names it
        block = i["messages"][0]["content"].split("```")[1]
        assert block.count("????") == 1 and "????" in i["messages"][0]["content"].split("```")[0]
    bad = [i["id"] for i in full if not tc.score_cruxeval_x(i, reply(f"thinking\nOutput: {i['gold']}"))["correct"]]
    assert not bad, bad[:5]  # the gold in its own language always scores
    i = full[0]
    assert not tc.score_cruxeval_x(i, reply("no answer line"))["correct"]
    assert not tc.score_cruxeval_x(i, reply("Output: 'definitely wrong ☃'"))["correct"]


def test_cx_normalisation():
    assert tc.cx_key("vec![(4, 1)]") == tc.cx_key("vec![(4,1)]")  # whitespace
    assert tc.cx_key("'ab'") == tc.cx_key('"ab"')  # quotes
    assert tc.cx_key("None") == tc.cx_key("null") == tc.cx_key("nil")  # null spellings
    assert tc.cx_key("4l") == tc.cx_key("4") and tc.cx_key("1.0f") == tc.cx_key("1.0")  # numeric suffixes
    assert tc.cx_key('"ab"') == tc.cx_key("ab")  # a bare string and a quoted one agree
    assert tc.cx_key("4") != tc.cx_key("5")
    # a Python-style rendering of a Rust literal: the key differs, the atoms agree
    assert tc.cx_key("vec![(4, 1)]") != tc.cx_key("[(4, 1)]")
    assert tc.cx_atoms("vec![(4, 1), (2, 3)]") == tc.cx_atoms("[(4, 1), (2, 3)]")
    assert tc.cx_atoms("[(4, 1)]") != tc.cx_atoms("[(4, 2)]")
    assert tc.cx_atoms('String::from("ab")') == tc.cx_atoms('"ab"')
    assert tc.cx_atoms("") == []


def test_cx_split():
    code = "def f(x):\n    return x\n\nassert f(1) == 1\n"
    masked = "assert f(1) == ????\n"
    assert tc.cx_split(code, masked) == ("def f(x):\n    return x\n\nassert f(1) == ????\n", "1")
    assert tc.cx_split(code, "assert f(2) == ????\n") is None  # not a suffix of the code
    assert tc.cx_split(code, "assert f(1) == 1\n") is None  # no mask


def test_truthfulqa():
    de, en = (quality.TASKS["truthfulqa"]["load"](lang, None, 0) for lang in ("de", "en"))
    assert len(de) == len(en) == 817 and {i["id"] for i in de} == {i["id"] for i in en}
    assert max(i["n_options"] for i in en) == 13 and quality.extract_letter("Answer: M", 13) == "M"
    d, e = ({i["id"]: i for i in x} for x in (de, en))
    for k in ("0", "400", "816"):  # same option count and same shuffle in both languages
        assert d[k]["n_options"] == e[k]["n_options"] and d[k]["gold"] == e[k]["gold"]
    first = e["0"]["messages"][0]["content"]
    assert f"{e['0']['gold']}. Nauru is the smallest country" in first, first


def test_cruxeval_x_de_is_paired():
    """The German cell is the same items behind a German instruction, and both markers are accepted."""
    en = tc.TASKS["cruxeval_x"]["load"]("en", 160, 1234)
    de = tc.TASKS["cruxeval_x_de"]["load"]("de", 160, 1234)
    assert [i["id"] for i in de] == [i["id"] for i in en] and {i["lang"] for i in de} == {"de"}
    for d, e in zip(de, en):
        assert d["gold"] == e["gold"] and d["messages"][0]["content"] != e["messages"][0]["content"]
        assert "Was gibt dieses" in d["messages"][0]["content"]
        assert "Ausgabe: X" in d["messages"][0]["content"]
    score = tc.TASKS["cruxeval_x_de"]["score"]
    for i in de[:20]:  # gold behind either marker, in either language
        assert score(i, {"content": f"Ausgabe: {i['gold']}"})["correct"], i["id"]
        assert score(i, {"content": f"Output: {i['gold']}"})["correct"], i["id"]
    assert not score(de[0], {"content": "Ausgabe: zzz"})["correct"]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
