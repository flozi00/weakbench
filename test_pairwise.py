"""pairwise.py with a mock judge and two synthetic runs, plain asserts:

    uv run --with pyarrow --with pyyaml --with langdetect --with immutabledict --with 'nltk>=3.9.1' python test_pairwise.py
"""
import json
import os
import tempfile
from pathlib import Path

TMP = tempfile.mkdtemp(prefix="pairwise-test-")
os.environ["BENCH_RUNS"], os.environ["BENCH_RESULTS"] = f"{TMP}/runs", f"{TMP}/results"

import mock_server  # noqa: E402
import pairwise  # noqa: E402
import tasks_lang  # noqa: E402


def judge(path, body):
    text = body["messages"][-1]["content"]
    if "Label (A, B, C or D)" in text:  # preflight canaries
        pred = text.split("PREDICTED ANSWER:")[1]
        return {"content": "B" if ("Lyon" in pred or "six legs" in pred) else "D" if "don't know" in pred else "A"}
    a, b = text.split("REPLY A:")[1].split("REPLY B:")
    # prefers the reply that says GOOD; with neither, it always says A (pure position bias)
    return {"content": "A" if "GOOD" in a else "B" if "GOOD" in b else "A"}


def test_pairwise():
    items = tasks_lang.load_switch("de", 6, 0)
    runs = {}
    for arm, texts in (("A", ["meh"] * 6), ("B", ["GOOD"] * 3 + ["meh"] * 3)):
        d = Path(TMP) / "runs" / f"run-{arm}"
        d.mkdir(parents=True)
        (d / "items-lang_switch.jsonl").write_text("".join(
            json.dumps({"task": "lang_switch", "lang": "de", "id": it["id"], "error": None, "reply": t}) + "\n"
            for it, t in zip(items, texts)))
        runs[arm] = d.name
    url, stop = mock_server.serve(judge)
    try:
        rows = pairwise.main(["--a", runs["A"], "--b", runs["B"], "--judge-url", url, "--tasks", "lang_switch",
                              "--concurrency", "2"])
    finally:
        stop()
    (r,) = rows
    # 3 B wins; the other 3 are position-bias ties (A first -> A, B first -> A), never A wins
    assert (r["n"], r["b_wins"], r["ties"], r["b_losses"], r["b_win_rate"]) == (6, 3, 3, 0, 0.75), r
    assert r["sign_p"] == 0.25 and r["ci95"][0] < 0.75 < r["ci95"][1]
    assert (Path(TMP) / "results" / "pairwise.jsonl").exists()
    assert pairwise.sign_test(10, 0) < 0.01 and pairwise.wilson(0, 0) == (0.0, 1.0)


def test_cross_lingual():
    items = tasks_lang.load_switch("de", 4, 0)
    d = Path(TMP) / "runs" / "run-xl"
    d.mkdir(parents=True)
    rows = [{"task": "lang_switch", "lang": lang, "id": it["id"], "error": None,
             "reply": f"{'GOOD' if k < 3 else 'BAD'} {lang}"} for k, it in enumerate(items) for lang in ("de", "en")]
    (d / "items-lang_switch.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    def xl_judge(path, body):
        text = body["messages"][-1]["content"]
        if "Label (A, B, C or D)" in text:
            return judge(path, body)
        return {"content": "DIFFERENT" if "BAD" in text.split("GERMAN REPLY:")[1] else "SAME"}
    url, stop = mock_server.serve(xl_judge)
    try:
        (r,) = pairwise.main(["--xl", "run-xl", "--judge-url", url, "--tasks", "lang_switch"])
    finally:
        stop()
    assert (r["n"], r["same_rate"]) == (4, 0.75), r


def test_audit():
    def biased(path, body):  # a judge that loves length and otherwise says A
        text = body["messages"][-1]["content"]
        if "Label (A, B, C or D)" in text:
            return judge(path, body)
        a, b = text.split("REPLY A:")[1].split("REPLY B:")
        return {"content": "B" if len(b) > len(a) + 50 else "A" if len(a) > len(b) + 50 else "TIE"}
    url, stop = mock_server.serve(biased)
    try:
        (r,) = pairwise.main(["--audit", "run-A", "--judge-url", url, "--tasks", "lang_switch"])
    finally:
        stop()
    assert r["identical_tie"] == 1.0 and r["padded_wins"] == 1.0, r


def test_rejudge():
    import tasks_open
    tasks_open.tasks_judge.SETUP["url"] = "x"  # the loaders only check that a judge is configured
    items = tasks_open.load_checklist("de", 2, 0)
    d = Path(TMP) / "runs" / "run-rj"
    d.mkdir(parents=True)
    rows = [{"task": "open_checklist", "lang": "de", "id": it["id"], "error": None, "reply": "x", "score": s}
            for it, s in zip(items, (1.0, 0.5))]
    (d / "items-open_checklist.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    def j(path, body):
        text = body["messages"][-1]["content"]
        if "Label (A, B, C or D)" in text:
            return judge(path, body)
        n = text.count("\n", text.index("CHECKLIST:"))
        return {"content": json.dumps({"verdicts": ["YES"] * n})}
    url, stop = mock_server.serve(j)
    try:
        (r,) = pairwise.main(["--rejudge", "run-rj", "--judge-url", url, "--tasks", "open_checklist"])
    finally:
        stop()
    assert r["n"] == 2 and r["identical"] == 0.5 and r["mad"] == 0.25, r


if __name__ == "__main__":
    test_rejudge()
    test_pairwise()
    test_audit()
    test_cross_lingual()
    print("ok")
