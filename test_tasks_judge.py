"""tasks_judge (AA-Omniscience, AA-LCR) on the real pinned data with a mock judge server, plain asserts:

    uv run --with pyarrow --with pyyaml --with langdetect --with immutabledict --with 'nltk>=3.9.1' python test_tasks_judge.py
"""
import json
import os
import tempfile
from pathlib import Path

TMP = tempfile.mkdtemp(prefix="judge-test-")
os.environ["BENCH_RUNS"], os.environ["BENCH_RESULTS"] = f"{TMP}/runs", f"{TMP}/results"

import mock_server  # noqa: E402
import quality  # noqa: E402
import tasks_judge as tj  # noqa: E402

reply = lambda text: {"content": text, "reasoning": "", "tool_calls": [], "finish_reason": "stop"}  # noqa: E731
SEEN = []


def serve_judge(answer):
    def responder(path, body):
        SEEN.append(body)
        return {"content": answer if isinstance(answer, str) else answer(body), "finish_reason": "stop"}
    return mock_server.serve(responder)


def test_omniscience_grades():
    for letter, grade, score, correct in (("A", "CORRECT", 1.0, True), ("B", "INCORRECT", -1.0, False),
                                         ("C", "PARTIAL_ANSWER", 0.0, False), ("D", "NOT_ATTEMPTED", 0.0, False)):
        url, stop = serve_judge(f"{letter}")
        try:
            tj.configure(url, "mock-model", 32768)
            it = tj.TASKS["aa_omniscience"]["load"]("en", 5, 0)
            assert len(it) == 5 and "I don't know" in it[0]["messages"][0]["content"]
            r = tj.TASKS["aa_omniscience"]["score"](it[0], reply("Some answer"))
        finally:
            stop()
        assert (r["detail"]["grade"], r["score"], r["correct"]) == (grade, score, correct), r
    assert "Some answer" in SEEN[-1]["messages"][-1]["content"] and it[0]["gold"] in SEEN[-1]["messages"][-1]["content"]
    assert SEEN[-1]["temperature"] == 0 and SEEN[-1]["chat_template_kwargs"]["enable_thinking"] is False
    url, stop = serve_judge("I think maybe")
    try:
        tj.configure(url, "mock-model", 32768)
        try:
            tj.TASKS["aa_omniscience"]["score"](it[0], reply("x"))
            raise AssertionError("an unparseable grade must raise")
        except ValueError as exc:
            assert "unparseable" in str(exc)
    finally:
        stop()
    assert len(tj.TASKS["aa_omniscience"]["load"]("en", None, 0)) == 300  # v1.1 public set: 6 domains x 50


def test_lcr_documents_and_context_filter():
    tj.configure("http://judge", "m", None)
    items = tj.TASKS["aa_lcr"]["load"]("en", None, 0)
    assert len(items) == 100
    for it in items:  # every document of every question resolved (names are mojibake / NFD in the archive)
        text = it["messages"][0]["content"]
        assert text.startswith("BEGIN INPUT DOCUMENTS\nBEGIN DOCUMENT 1:\n") and "END DOCUMENT 1" in text
        assert text.rstrip().endswith("END QUESTION") and it["question"] in text
    quick = tj.TASKS["aa_lcr"]["load"]("en", 20, 0)
    assert [i["id"] for i in quick] == [i["id"] for i in items][:20]  # quick is a prefix of full
    tj.configure("http://judge", "m", 131072)
    fit = tj.TASKS["aa_lcr"]["load"]("en", None, 0)
    assert 0 < len(fit) < 100  # max_model_len filters, conservatively
    tj.configure("http://judge", "m", 8192)
    try:
        tj.TASKS["aa_lcr"]["load"]("en", None, 0)
        raise AssertionError("a context too small for every question must raise")
    except ValueError as exc:
        assert "150k" in str(exc)


def test_lcr_verdict():
    item = {"question": "Which company?", "gold": "Atlassian", "category": "Company_Documents"}
    for verdict, ok in (('{"verdict": "CORRECT"}', True), ('{"verdict": "INCORRECT"}', False), ("incorrect", False),
                        ('Reasoning: the numbers match. {"verdict":"CORRECT"}', True)):
        url, stop = serve_judge(verdict)
        try:
            tj.configure(url, "mock-model", 200000)
            assert tj.TASKS["aa_lcr"]["score"](item, reply("It is Atlassian."))["correct"] is ok, verdict
        finally:
            stop()
    sent = SEEN[-1]["messages"]
    assert sent[0]["role"] == "system" and "OFFICIAL ANSWER" in sent[0]["content"]
    assert "The OFFICIAL ANSWER: Atlassian" in sent[1]["content"] and "It is Atlassian." in sent[1]["content"]


def test_judge_guards():
    tj.SETUP["url"] = None
    for name in tj.TASKS:
        try:
            tj.TASKS[name]["load"]("en", 1, 0)
            raise AssertionError("no judge must refuse")
        except RuntimeError as exc:
            assert "--judge-url" in str(exc)


ARMS = iter(range(100))


def run_quality(url, *extra):  # a run dir is named after the second and the arm: one arm name per call
    return quality.main(["--url", url, "--arm", f"t{next(ARMS)}", "--no-drift", "--limit", "3", *extra])


def canary_aware(body):
    """A judge that grades the preflight canaries right and says A / CORRECT to everything else."""
    last = body["messages"][-1]["content"]
    if "Label (A, B, C or D)" not in last:
        return '{"verdict": "CORRECT"}'
    pred = last.split("PREDICTED ANSWER:")[1]
    return "B" if ("Lyon" in pred or "six legs" in pred) else "D" if "don't know" in pred else "A"


def test_judge_preflight():
    url, stop = serve_judge("A")  # a judge that says CORRECT to everything must not grade a run
    try:
        tj.configure(url, "mock-model", 32768)
        try:
            tj.preflight()
            raise AssertionError("a yes-man judge must fail the canaries")
        except RuntimeError as e:
            assert "canary" in str(e)
    finally:
        stop()
    url, stop = serve_judge(canary_aware)
    try:
        tj.configure(url, "mock-model", 32768)
        assert tj.preflight() == len(tj.CANARIES)
    finally:
        stop()


def test_quality_end_to_end():
    judge_url, stop_judge = serve_judge(canary_aware)
    cand_url, stop_cand = mock_server.serve(lambda path, body: {"content": "42", "finish_reason": "stop"}, max_model_len=200000)
    try:
        run = run_quality(cand_url, "--tasks", "aa_omniscience,aa_lcr,humaneval_plus", "--judge-url", judge_url,
                          "--langs", "en")
        info = json.loads((run / "run_info.json").read_text())
        assert info["judge_model"] == "mock-model"
        rows = {r["task"]: r for r in json.loads((run / "summary.json").read_text())}
        assert rows["aa_omniscience"]["n"] == 3 and rows["aa_omniscience"]["accuracy"] == 1.0, rows
        assert rows["aa_lcr"]["n"] == 3 and rows["humaneval_plus"]["accuracy"] == 0.0, rows
        names = [r["task"] for r in json.loads((run / "summary.json").read_text())]
        run2 = run_quality(cand_url, "--langs", "en", "--tasks", "gsm8k")  # default selection never has opt-in tasks
        assert {r["task"] for r in json.loads((run2 / "summary.json").read_text())} == {"gsm8k"}
        try:
            run_quality(cand_url, "--tasks", "aa_lcr")
            raise AssertionError("aa_lcr without --judge-url must exit")
        except SystemExit as exc:
            assert exc.code == 2
        assert "aa_omniscience" in names
    finally:
        stop_judge()
        stop_cand()
    default = [n for n, t in quality.TASKS.items() if not t.get("opt_in")]
    assert not {"aa_omniscience", "aa_lcr", "humaneval_plus", "mbpp_plus"} & set(default)


def test_eqbench_cw_prompts_and_parser():
    """The pinned prompt file, the item order, and the rubric parser (no judge call)."""
    tj.configure("http://judge.invalid", "judge", 32768)  # check_judge only needs a url
    full = tj.TASKS["eqbench_cw"]["load"]("en", None, 0)
    assert len(full) == 96 and len({i["id"] for i in full}) == 96  # 32 prompts x 3 seed modifiers
    quick = tj.TASKS["eqbench_cw"]["load"]("en", 32, 0)
    assert [i["id"] for i in quick] == [i["id"] for i in full][:32]  # quick = upstream iteration 1
    assert len({i["category"] for i in full}) == 23
    for i in full:
        assert "<SEED>" not in i["messages"][0]["content"]  # every modifier substituted
        assert i["base_prompt"].count("<SEED>") == 1  # the judge sees the base prompt, as upstream does
        assert i["messages"][0]["role"] == "user" and len(i["messages"]) == 1  # no system message
    assert tj.TASKS["eqbench_cw"]["max_tokens"] == 8192 and tj.TASKS["eqbench_cw"]["opt_in"]
    # the parser: 22 names, prose in [Analysis] ignored, name drift and out-of-range values dropped
    reply = ("[Analysis]\nThe voice is strong: 17 is a number I like.\n\n[Scores]\n"
             + "\n".join(f"{c}: [{v}]" for c, v in zip(tj.CRITERIA, [18, 16, 15, 17, 16, 14, 15, 13, 18,
                                                                    4, 6, 5, 7, 3, 4, 5, 2, 3, 14, 16, 15, 16]))
             + "\nTell Don't Show: 5\nElegant Prose: 99\n")
    scores, section = tj.parse_cw_scores(reply)
    assert section == 1 and len(scores) == 22 and "voice" not in scores
    assert scores["Tell-Don't-Show"] == 5 and scores["Elegant Prose"] == 14  # drift mapped, 99 dropped
    pos = [v for c, v in scores.items() if c not in tj.NEGATIVE]
    neg = [20 - v for c, v in scores.items() if c in tj.NEGATIVE]
    assert len(pos) == 13 and len(neg) == 9
    assert round(sum(pos + neg) / 22 / 20, 4) == 0.7818
    assert tj.parse_cw_scores("no scores here")[1] == 0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
