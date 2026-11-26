"""Tests for compare.py (stdlib): python3 test_compare.py, or pytest."""
import atexit
import gzip
import io
import json
import math
import shutil
import subprocess
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import compare  # noqa: E402
from common import append_jsonl  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="compare-test-"))
atexit.register(shutil.rmtree, TMP, True)
ln = math.log
BASE = {"agg_out_tps": [800, 815, 830], "stream_tps_p50": [90, 95, 100], "stream_tps_p10": [60, 62, 64],
        "ttft_p50_ms": [100, 105, 110], "ttft_p99_ms": [300, 320, 340], "itl_p99_ms": [40, 41, 42],
        "e2e_p50_s": [5.0, 5.1, 5.2], "e2e_p90_s": [7.0, 7.2, 7.4], "spec_accept_rate": [0.7, 0.72, 0.71],
        "preemptions": 0, "errors": 0}


def make_run(name, kind, items=(), drift=None, summary=None, info=None, **manifest):
    """A run dir; `info` -> run_info.json, `manifest` overrides manifest keys (args, server, bench_git_sha)."""
    path = TMP / name
    path.mkdir()
    (path / "manifest.json").write_text(json.dumps({
        "run_id": name, "kind": kind, "arm": name.rsplit("-", 1)[-1], "url": "http://mock",
        "server": {"model": "mock-model", "max_model_len": 32768, "version": {"version": "mock-0.0"}}, "args": {},
        **manifest}))
    for it in items:
        append_jsonl(path / f"items-{it['task']}.jsonl", it)
    for lang, docs in (drift or {}).items():
        with gzip.open(path / f"drift-{lang}.jsonl.gz", "wt") as fh:
            fh.writelines(json.dumps(d) + "\n" for d in docs)
    if summary is not None:
        (path / "summary.json").write_text(json.dumps(summary))
    if info is not None:
        (path / "run_info.json").write_text(json.dumps(info))
    return path


def item(task, lang, i, correct=None, score=None, **kw):
    """correct and score both None = request error; score only = continuous task."""
    return {"task": task, "lang": lang, "id": i, "correct": correct,
            "score": float(correct) if correct is not None else score, "pred": "x", "gold": "x",
            "finish_reason": "stop", "prompt_tokens": 50, "completion_tokens": 100, "latency_s": 0.5,
            "junk": False, "lang_ok": True, "error": "HTTP 500" if correct is None and score is None else None,
            "reply": "x", "detail": {}, **kw}


def rounds(scenario, conc, n=3, **series):
    """n speed summary rows; a list gives one value per round, a scalar repeats."""
    return [{"scenario": scenario, "conc": conc, "round": r, "requests": 40,
             **{m: v[r] if isinstance(v, list) else v for m, v in series.items()}} for r in range(n)]


def gate(*argv):
    """compare.main in-process -> (exit code, stdout, --json result)."""
    out, buf = TMP / "result.json", io.StringIO()
    with redirect_stdout(buf):
        code = compare.main([*map(str, argv), "--json", str(out)])
    return code, buf.getvalue(), json.loads(out.read_text())


def speed_gate(name, b_over, n=3, *flags, a1=None, a_runs=2):
    """A1..A<a_runs> (the A/B/A protocol; `a1` overrides A1's series) vs one B run, n rounds per run."""
    a = [make_run(f"{name}-A{k}", "speed", summary=rounds("chat", 8, n, **(BASE | (a1 or {}) if k == 1 else BASE)))
         for k in range(1, a_runs + 1)]
    b = make_run(f"{name}-B", "speed", summary=rounds("chat", 8, n, **{**BASE, **b_over}))
    return gate("--a", *a, "--b", b, *(flags or ("--min-rounds", 3)))  # fixtures: 3 rounds per run


def last(out):
    return out.rstrip().splitlines()[-1]


def test_mcnemar_exact():
    assert compare.mcnemar(0, 10) == compare.mcnemar(10, 0) == 1 / 512  # 0.00195
    assert compare.mcnemar(1, 9) == 11 / 512  # 2 * (C(10,0) + C(10,1)) / 2^10
    assert compare.mcnemar(5, 5) == compare.mcnemar(0, 0) == compare.mcnemar(700, 700) == 1.0


def test_holm():
    adj = compare.holm([0.025, 0.01, 0.02])  # sorted 0.01, 0.02, 0.025 -> x3, x2, x1 under a running max
    assert all(abs(x - y) < 1e-12 for x, y in zip(adj, [0.04, 0.03, 0.04], strict=True))
    assert compare.holm([0.5, 0.6]) == [1.0, 1.0] and compare.holm([]) == []


def test_clear_regression_fails():
    """A 90% vs B 70% on 200 paired items (40 A-only right) next to an identical cell."""
    a = [item("gsm8k", "en", i, i < 180) for i in range(200)] + [item("mmlu", "de", i, i % 2 == 0) for i in range(50)]
    b = [item("gsm8k", "en", i, i < 140) for i in range(200)] + [item("mmlu", "de", i, i % 2 == 0) for i in range(50)]
    code, out, res = gate("--a", make_run("reg-A", "quality", a), "--b", make_run("reg-B", "quality", b))
    g, m = res["cells"]  # sorted: gsm8k/en, mmlu/de
    assert (g["type"], g["n"], g["b"], g["c"], round(g["mean_a"], 9), round(g["mean_b"], 9)) == \
        ("binary", 200, 40, 0, 90.0, 70.0)
    se = math.sqrt(40 - 40 ** 2 / 200) / 200
    assert abs(g["delta_pp"] + 20) < 1e-9 and abs(g["ci_pp"][0] - 100 * (-0.2 - 1.96 * se)) < 1e-9
    assert g["p"] == 2 / 2 ** 40 and g["p_holm"] == 2 * g["p"] and g["status"] == "REGRESSION"  # Holm, 2 cells
    assert m["status"] == "same" and m["p_holm"] == 1.0 and (res["pooled"]["n"], res["pooled"]["b"]) == (250, 40)
    assert code == 1 and last(out).startswith("VERDICT: FAIL - gsm8k/en REGRESSION")


def cell(task, lang, n, a_right, b_right, flips_back=0, prefix=""):
    """n paired items: A right on the first a_right, B right on the first b_right plus `flips_back`
    items A got wrong (so b = A-only right, c = B-only right)."""
    a = [item(task, lang, f"{prefix}{i}", i < a_right) for i in range(n)]
    b = [item(task, lang, f"{prefix}{i}", i < b_right or a_right <= i < a_right + flips_back) for i in range(n)]
    return a, b


def test_relative_tolerance():
    """--max-rel-drop 0.05 (exploratory trials): a real drop inside 5% relative passes, a real drop over it
    fails, and pooled / per-language pools over 5% relative fail even when no single cell is significant."""
    def run(name, cells, *flags):
        a = [x for c in cells for x in c[0]]
        b = [y for c in cells for y in c[1]]
        return gate("--a", make_run(f"{name}-A", "quality", a), "--b", make_run(f"{name}-B", "quality", b), *flags)
    small = [cell("mmlu", "de", 400, 360, 348, flips_back=2)]  # 90.0 -> 87.5: real (b=12, c=2), -2.8% relative
    code, out, res = run("tol-small", small)
    assert code == 1 and res["cells"][0]["status"] == "REGRESSION"  # strict default: any real drop fails
    code, out, res = run("tol-small2", small, "--max-rel-drop", 0.05)
    assert code == 0 and res["cells"][0]["status"] == "worse (tol.)", last(out)
    assert any("inside the 5% tolerance" in w for w in res["warns"]) and res["relative"]["pooled"]["rel_drop_pct"] < 5
    big = [cell("mmlu", "de", 200, 180, 140)]  # 90 -> 70: -22% relative
    code, out, res = run("tol-big", big, "--max-rel-drop", 0.05)
    assert code == 1 and "REGRESSION" in last(out) and "-22.2% relative" in last(out), last(out)
    spread = [cell(t, lang, 30, 27, 25, prefix=lang) for t in ("mmlu", "belebele") for lang in ("de", "en")]
    code, out, res = run("tol-spread", spread, "--max-rel-drop", 0.05)  # 90 -> 83.3 everywhere, no cell significant
    assert all(r["status"] == "worse n.s." for r in res["cells"]) and code == 1
    assert "pooled 90.0 -> 83.3: -7.4% relative, over the 5% tolerance" in last(out), last(out)
    german = [cell("mmlu", "de", 60, 54, 50, prefix="d"), cell("mmlu", "en", 60, 54, 54, prefix="e"),
              cell("bfcl", "de", 60, 54, 50, prefix="d"), cell("bfcl", "en", 60, 54, 54, prefix="e")]
    code, out, res = run("tol-de", german, "--max-rel-drop", 0.05)  # only German loses 7.4%: pooled -3.7%
    assert code == 1 and res["relative"]["pooled"]["rel_drop_pct"] < 5 and "pooled de" in last(out), last(out)
    assert "pooled en" not in last(out)


def test_identical_arms_pass_with_noise_floor():
    base = [*(item("gsm8k", "en", i, i % 3 > 0) for i in range(60)),
            *(item("tools", "de", i, i % 4 > 0) for i in range(40))]
    a2 = [dict(it, correct=not it["correct"]) if it["task"] == "gsm8k" and it["id"] < 3 else it for it in base]
    code, out, res = gate("--a", make_run("same-A1", "quality", base), make_run("same-A2", "quality", a2),
                          "--b", make_run("same-B", "quality", base))
    assert code == 0 and res["verdict"] == "PASS" and {r["status"] for r in res["cells"]} == {"same"}
    noise = {r["task"]: r for r in res["noise"]}
    assert noise["gsm8k"]["A~B"] == [0, 60] and noise["gsm8k"]["A1~A2"] == [3, 60]
    assert noise["pooled"]["A1~A2"] == [3, 100] and "Noise floor" in out and last(out).startswith("VERDICT: PASS")


def test_score_cells_paired_differences():
    """Continuous tasks (correct null, 0..1 score): paired mean difference with a normal-approximation p,
    same Holm family, kept out of the pooled McNemar; n<30 is shown but never judged."""
    shift = [-0.07, -0.06, -0.05, -0.04, -0.03] * 8  # mean -0.05, sample variance 0.0002 * 40/39
    a = [item("chrf", "de", i, score=0.5 + 0.01 * (i % 10)) for i in range(40)]
    b = [dict(it, score=it["score"] + s) for it, s in zip(a, shift)]
    a += [item("chrf", "de", 99, score=0.5), *(item("ifeval", "en", i, score=0.9) for i in range(10)),
          *(item("gsm8k", "en", i, i % 2 == 0) for i in range(20))]
    b += [item("chrf", "de", 99), *(item("ifeval", "en", i, score=0.5) for i in range(10)),  # 99: B request error
          *(item("gsm8k", "en", i, i % 2 == 0) for i in range(20))]
    code, out, res = gate("--a", make_run("score-A", "quality", a), "--b", make_run("score-B", "quality", b))
    cells = {r["task"]: r for r in res["cells"]}
    c, se = cells["chrf"], math.sqrt(0.0002 / 39)  # sd / sqrt(n), n = 40
    assert (c["type"], c["n"], c["errors"]) == ("score", 40, [0, 1]) and abs(c["delta_pp"] + 5) < 1e-9
    assert all(abs(x - 100 * (-0.05 + k * 1.96 * se)) < 1e-9 for x, k in zip(c["ci_pp"], (-1, 1)))
    assert math.isclose(c["p"], math.erfc(0.05 / se / math.sqrt(2)), rel_tol=1e-9) and c["status"] == "REGRESSION"
    assert cells["ifeval"]["status"] == "n<30" and cells["ifeval"]["delta_pp"] < -39  # big drop, too few items
    assert res["pooled"]["n"] == 20 and res["pooled_score"]["n"] == 50  # McNemar pools binary cells only
    assert abs(res["pooled_score"]["delta_pts"] - 100 * (40 * -0.05 + 10 * -0.4) / 50) < 1e-9
    assert code == 1 and "chrf/de REGRESSION" in last(out) and "ifeval" not in last(out)


def tops(*dists):
    return [None, *({t: ln(p) for t, p in d.items()} for d in dists)]


def drift_doc(i, tokens, lp, top):
    return {"task": "drift", "lang": "en", "id": i, "tokens": tokens, "lp": lp, "top": top}


def test_drift_math():
    # pos 1: tails 0.1 vs 0.2, argmax differs | pos 2: A's "6" missing from B -> B's k-th logprob ln 0.1,
    # B's "7" missing from A -> ln 0.2; A's mass > 1, so no tail | pos 3: B ties "4"/"3" -> top-1 agrees
    ta = tops({"1": .6, "2": .3}, {"5": .7, "6": .2}, {"3": .5, "4": .4})
    tb = tops({"2": .5, "1": .3}, {"5": .8, "7": .1}, {"4": .45, "3": .45})
    kls = [.6 * ln(.6 / .3) + .3 * ln(.3 / .5) + .1 * ln(.1 / .2),
           .7 * ln(.7 / .8) + .2 * ln(.2 / .1) + .2 * ln(.2 / .1),
           .5 * ln(.5 / .45) + .4 * ln(.4 / .45)]
    assert all(abs(compare.kl(x, y) - want) < 1e-12 for x, y, want in zip(ta[1:], tb[1:], kls))
    assert compare.kl({"1": ln(.5)}, {"1": 0.0}) == 0.0  # negative estimate clamps at 0
    assert [compare.top1_agree(x, y) for x, y in zip(ta[1:], tb[1:])] == [False, True, True]
    lpa, lpb = [None, ln(.6), ln(.2), ln(.5)], [None, ln(.3), ln(.05), ln(.45)]
    a = [drift_doc("d1", [9, 1, 6, 3], lpa, ta), drift_doc("d2", [1, 2], [None, ln(.5)], tops({"2": .5}))]
    b = [drift_doc("d1", [9, 1, 6, 3], lpb, tb), drift_doc("d2", [1, 3], [None, ln(.5)], tops({"3": .5}))]
    its = [item("gsm8k", "en", i, True) for i in range(10)]
    code, out, res = gate("--a", make_run("drift-A1", "quality", its, {"en": a}),
                          make_run("drift-A2", "quality", its, {"en": a}),
                          "--b", make_run("drift-B", "quality", its, {"en": b}))
    d = res["drift"]["A1~B1"]["en"]
    assert (d["docs"], d["tokenization_differs"], d["positions"]) == (1, ["d2"], 3)
    assert abs(d["top1_pct"] - 200 / 3) < 1e-9 and abs(d["kl_mean"] - sum(kls) / 3) < 1e-12
    assert abs(d["kl_p99"] - max(kls)) < 1e-12
    assert abs(d["ppl_ratio"] - (.6 * .2 * .5 / (.3 * .05 * .45)) ** (1 / 3)) < 1e-9  # exp(mean NLL_B - NLL_A)
    assert d["status"] == "FAIL" and code == 1 and "tokenization differs" in out
    floor = res["drift"]["A1~A2"]["en"]  # replicate row: the noise floor, not gated
    assert (floor["kl_mean"], floor["ppl_ratio"], floor["top1_pct"], "status" in floor) == (0, 1, 100, False)


def test_drift_streams_pairs():
    """Docs pair by id one at a time: ids one side lacks are skipped; ids out of order are a data error."""
    doc = lambda i: drift_doc(i, [9, 1], [None, ln(.5)], tops({"1": .5}))  # noqa: E731
    a, b = make_run("stream-A", "quality", drift={"en": [doc("d1"), doc("d2"), doc("d4")]}), \
        make_run("stream-B", "quality", drift={"en": [doc("d2"), doc("d3"), doc("d4")]})
    assert [(x["id"], y["id"]) for x, y in compare.doc_pairs(a / "drift-en.jsonl.gz", b / "drift-en.jsonl.gz")] \
        == [("d2", "d2"), ("d4", "d4")]
    bad = make_run("stream-bad", "quality", drift={"en": [doc("d2"), doc("d1")]})
    assert cli("--a", a, "--b", bad) == 2


def test_b_request_errors_fail():
    """B errors on 10 items A answered: the pairing drops them (the cell stays "same"), the error McNemar FAILs."""
    a = [item("gsm8k", "en", i, i % 2 == 0) for i in range(60)]
    b = [item("gsm8k", "en", i) if i < 10 else it for i, it in enumerate(a)]
    code, out, res = gate("--a", make_run("err-A", "quality", a), "--b", make_run("err-B", "quality", b))
    assert res["request_errors"] == {"a_only": 0, "b_only": 10, "p": 1 / 512} and res["cells"][0]["status"] == "same"
    assert code == 1 and last(out) == "VERDICT: FAIL - B request errors (10 B-only vs 0 A-only, p=0.002)"


def test_b_side_gaps_inconclusive():
    """A cell A answered but B never paired, or drift documents on A1 only: INCONCLUSIVE, never PASS."""
    en, de = [item("gsm8k", "en", i, i % 2 == 0) for i in range(30)], [item("mmlu", "de", i, True) for i in range(30)]
    code, out, _ = gate("--a", make_run("gapq-A", "quality", en + de), "--b", make_run("gapq-B", "quality", en))
    assert code == 3 and last(out) == "VERDICT: INCONCLUSIVE - mmlu/de: 30 usable A items, no usable B pair"
    docs, check = [drift_doc("d1", [9, 1], [None, ln(.5)], tops({"1": .5}))], {"ok": False, "reason": "no logprobs"}
    code, out, _ = gate("--a", make_run("drift1-A", "quality", en, {"en": docs}),
                        "--b", make_run("drift1-B", "quality", en, info={"drift_self_check": check}))
    assert code == 3 and last(out) == ("VERDICT: INCONCLUSIVE - drift en: documents on A1, none on B1 "
                                       "(B1 drift self-check: {'ok': False, 'reason': 'no logprobs'})")


def test_protocol_mismatch():
    """Different run_info detections: the arms are not comparable (INCONCLUSIVE). Args, served model id and
    suite version (bench_git_sha) only warn; keys outside the compared set (rounds) are ignored."""
    its = [item("gsm8k", "en", i, i % 2 == 0) for i in range(30)]
    code, out, _ = gate("--a", make_run("proto-A", "quality", its, info={"thinking_off_honored": True,
                                                                         "drift_anchor": ""}, args={"tier": "quick"}),
                        "--b", make_run("proto-B", "quality", its, info={"thinking_off_honored": False,
                                                                         "drift_anchor": ""}, args={"tier": "full"}))
    assert code == 3 and last(out) == ("VERDICT: INCONCLUSIVE - A1 and B1 did not run the same protocol: "
                                       "thinking_off_honored True vs False; WARN: A1 and B1 differ in tier: "
                                       "'quick' vs 'full'")
    code, out, _ = gate("--a", make_run("proto-n-A", "quality", its, info={"thinking_off_honored": True}),
                        "--b", make_run("proto-n-B", "quality", its, info={"thinking_off_honored": None}))
    assert code == 0, last(out)  # None = B's preflight failed: sent exactly like True
    # The full framing prefix is part of the sent protocol, even with identical
    # corpus/thresholds and otherwise passing item scores.
    code, out, _ = gate("--a", make_run("proto-g-A", "quality", its, info={"drift_anchor": "<bos>"}),
                       "--b", make_run("proto-g-B", "quality", its, info={"drift_anchor":
                           "<bos><|turn>user\nWrite prose.<turn|>\n<|turn>model\n"}))
    assert code == 3 and "did not run the same protocol: drift_anchor" in last(out)
    # Judge-graded tasks (aa_omniscience / aa_lcr) are only comparable under the same judge model.
    code, out, _ = gate("--a", make_run("proto-j-A", "quality", its, info={"judge_model": "judge-a"}),
                        "--b", make_run("proto-j-B", "quality", its, info={"judge_model": "other-judge"}))
    assert code == 3 and "did not run the same protocol: judge_model 'judge-a' vs 'other-judge'" in last(out)
    code, out, _ = gate("--a", make_run("proto-j2-A", "quality", its, info={"judge_model": "judge-a"}),
                        "--b", make_run("proto-j2-B", "quality", its, info={"judge_model": "judge-a"}))
    assert code == 0, last(out)
    a = [make_run(f"proto-s-A{k}", "speed", summary=rounds("chat", 8, **BASE), args={"seconds": 60, "rounds": 3},
                  bench_git_sha="abc") for k in (1, 2)]
    b = make_run("proto-s-B", "speed", summary=rounds("chat", 8, **BASE), args={"seconds": 30, "rounds": 5},
                 bench_git_sha="def", server={"model": "other-model"})
    code, out, res = gate("--a", *a, "--b", b, "--min-rounds", 3)
    assert res["warns"] == ["A1 and B1 differ in seconds: 60 vs 30",
                            "A1 and B1 differ in served model: 'mock-model' vs 'other-model'",
                            "A1 and B1 differ in bench_git_sha: 'abc' vs 'def'"]
    assert code == 0 and "## Warnings" in out and last(out).startswith(
        "VERDICT: PASS - WIN 0 / TIE 8; WARN: A1 and B1 differ in seconds: 60 vs 30; ")
    a = [make_run(f"proto-c-A{k}", "speed", summary=rounds("chat", 8, **BASE), bench_git_sha="abc",
                  bench_code_sha="111") for k in (1, 2)]
    b = make_run("proto-c-B", "speed", summary=rounds("chat", 8, **BASE), bench_git_sha="abc", bench_code_sha="222")
    _, _, res = gate("--a", *a, "--b", b, "--min-rounds", 3)  # same commit, edited suite: the code hash tells
    assert res["warns"] == ["A1 and B1 differ in bench_code_sha: '111' vs '222'"]


def test_private_release_protocol():
    info = {"data_revision": "a" * 40, "data_sha256": "b" * 64, "expected_items": 2}
    its = [item("hidden", "de", "toy", True), item("hidden", "en", "toy", False)]
    a = make_run("hidden-A", "quality", its, info=info, bench_code_sha="1")
    for name, rows, meta, sha, expected in [
            ("same", its, info, "1", 0),
            ("legacy", its, {}, "1", 3),
            ("rotation", its, dict(info, data_revision="c" * 40), "1", 3),
            ("changed", its, dict(info, data_sha256="d" * 64), "1", 3),
            ("subset", its[:1], info, "1", 3),
            ("scorer", its, info, "2", 3)]:
        b = make_run("hidden-" + name, "quality", rows, info=meta, bench_code_sha=sha)
        rc, output, _ = gate("--a", a, "--b", b)
        assert rc == expected, last(output)


def test_speed_range_verdicts():
    v = compare.range_verdict
    assert v([800, 830], [901, 915], 1) == "WIN" and v([800, 830], [700, 790], 1) == "LOSS"
    assert v([800, 830], [830, 900], 1) == "TIE"  # touching ranges do not separate
    assert v([100, 120], [80, 99], -1) == "WIN" and v([100, 120], [121, 130], -1) == "LOSS"  # lower is better
    code, out, res = speed_gate("sp-win", {"agg_out_tps": [901, 908, 915]})
    m = res["cells"][0]["metrics"]
    assert m["agg_out_tps"]["verdict"] == "WIN" and m["itl_p99_ms"]["verdict"] == "TIE"
    assert code == 0 and "A 800-830 B 901-915 WIN +11.4%" in out and last(out) == "VERDICT: PASS - WIN 1 / TIE 7"
    code, out, _ = speed_gate("sp-loss", {"itl_p99_ms": [50, 51, 52]})
    assert code == 1 and last(out) == "VERDICT: FAIL - 1 untolerated LOSS: chat c=8 itl_p99_ms"


def test_speed_errors_binomial():
    """Error LOSS: one-sided exact binomial, each error lands on B with B's share of all requests."""
    assert compare.binom_sf(0, 5, .3) == 1.0 and math.isclose(compare.binom_sf(1, 2, .3), 1 - .7 ** 2)
    assert math.isclose(compare.binom_sf(10, 10, .5), .5 ** 10)
    assert math.isclose(compare.binom_sf(5000, 10000, .5), .5 + math.comb(10000, 5000) / 2 ** 10001)  # no overflow
    code, out, res = speed_gate("err-1", {"errors": [0, 1, 0]}, a1={"errors": [1, 0, 0]})  # A has 2x the requests
    e = res["cells"][0]["errors"]  # the old per-request-rate rule called this a LOSS
    assert (e["a"], e["b"], e["loss"], code) == ([1, 241], [1, 121], False, 0)
    assert math.isclose(e["p"], 1 - (1 - 121 / 362) ** 2) and last(out) == "VERDICT: PASS - WIN 0 / TIE 8"
    code, out, res = speed_gate("err-10", {"errors": [3, 4, 3]})  # 10 B errors, none on A
    assert code == 1 and last(out) == "VERDICT: FAIL - 1 untolerated LOSS: chat c=8 errors"
    assert math.isclose(res["cells"][0]["errors"]["p"], (130 / 370) ** 10) and "LOSS p=2.9e-05" in out


def test_all_failing_b_cell_fails():
    """Every B request failed (no metric values): judged on its rounds, and the error test runs even when short."""
    dead = {**dict.fromkeys(compare.GATED), "agg_out_tps": 0.0, "requests": 0, "errors": 40}
    code, out, res = speed_gate("dead", dead)
    assert (code, res["cells"][0]["short"], res["cells"][0]["errors"]["b"]) == (1, False, [120, 120])
    assert last(out) == "VERDICT: FAIL - 2 untolerated LOSS: chat c=8 errors, chat c=8 agg_out_tps"
    code, out, _ = speed_gate("dead5", dead, 3, "--min-rounds", 5)
    assert code == 1 and last(out) == ("VERDICT: FAIL - 1 untolerated LOSS: chat c=8 errors; "
                                       "not judged, fewer than 5 rounds: chat c=8 (A 6 / B 3)")


def test_speed_needs_two_a_runs():
    """One A run: no range is judged (identical servers fail the separated-range rule ~40% of the time)."""
    code, out, res = speed_gate("one-a", {"agg_out_tps": [600, 610, 620]}, a_runs=1)  # would be a LOSS
    assert code == 3 and res["losses"] == [] and last(out) == (
        "VERDICT: INCONCLUSIVE - need A1 and A2 runs (A/B/A): with one A run identical servers fail the "
        "separated-range rule ~40% of the time")
    a1 = TMP / "one-a-A1"
    assert gate("--a", a1, a1, "--b", TMP / "one-a-B", "--min-rounds", 3)[0] == 3  # the same run twice is one run


def test_ttft_loss_tolerated_only_by_e2e_win():
    slow_first_token = {"ttft_p50_ms": [120, 125, 130]}
    code, out, res = speed_gate("tt-ok", {**slow_first_token, "e2e_p50_s": [4.0, 4.1, 4.2]})
    t = res["cells"][0]["metrics"]["ttft_p50_ms"]
    assert (t["verdict"], t["tolerated"], code) == ("LOSS", True, 0) and "(tol.)" in out
    assert last(out) == "VERDICT: PASS - WIN 1 / TIE 6 / tolerated TTFT LOSS 1"
    code, out, _ = speed_gate("tt-bad", slow_first_token)  # e2e only TIEs -> the TTFT loss counts
    assert code == 1 and last(out).endswith("chat c=8 ttft_p50_ms")


def test_busy_cells_not_judged():
    code, out, res = speed_gate("busy", {"agg_out_tps": [600, 610, 620], "foreign_load": [0, 2, 0]})
    assert code == 3 and res["busy"] == ["chat c=8 (2 foreign requests)"] and "(busy)" in out  # the LOSS is not counted
    assert last(out) == "VERDICT: INCONCLUSIVE - server busy with other traffic: chat c=8 (2 foreign requests)"
    assert gate("--a", TMP / "busy-A1", TMP / "busy-A2", "--b", TMP / "busy-B")[0] == 3  # default: 5 rounds


def test_too_few_rounds_inconclusive():
    code, out, _ = speed_gate("short", {"agg_out_tps": [700, 710]}, 2)  # a 2-round LOSS proves nothing
    assert code == 3 and last(out) == "VERDICT: INCONCLUSIVE - fewer than 3 rounds: chat c=8 (A 4 / B 2)"
    assert speed_gate("short2", {"agg_out_tps": [700, 710]}, 2, "--min-rounds", 2)[0] == 1
    a = [make_run(f"gap-A{k}", "speed", summary=rounds("chat", 1, **BASE) + rounds("chat", 8, **BASE)) for k in (1, 2)]
    b = make_run("gap-B", "speed", summary=rounds("chat", 8, **BASE))  # B never measured c=1
    code, out, _ = gate("--a", *a, "--b", b, "--min-rounds", 3)
    assert code == 3 and last(out) == "VERDICT: INCONCLUSIVE - fewer than 3 rounds: chat c=1 (A 6 / B 0)"


def cli(*argv):
    return subprocess.run([sys.executable, str(HERE / "compare.py"), *map(str, argv)],
                          capture_output=True, text=True).returncode


def test_exit_codes_subprocess():
    ok = [item("gsm8k", "en", i, i % 2 == 0) for i in range(30)]
    a, same = make_run("cli-A", "quality", ok), make_run("cli-B", "quality", ok)
    worse = make_run("cli-W", "quality", [dict(it, correct=False) for it in ok])  # 15 A-only right
    errors = make_run("cli-E", "quality", [item("gsm8k", "en", i) for i in range(30)])  # all request errors
    two, two2 = (make_run(f"cli-S{k}", "speed", summary=rounds("chat", 1, 2, **BASE)) for k in (1, 2))
    assert cli("--a", a, "--b", same) == 0
    assert cli("--a", a, "--b", worse) == 1
    assert cli("--a", a, "--b", errors) == 1  # B request errors on all 30 items A answered
    assert cli("--a", errors, "--b", errors) == 3  # no paired items
    assert cli("--a", two, two2, "--b", two) == 3  # 2 rounds per run < --min-rounds 5
    assert cli("--a", two, two2, "--b", two, "--min-rounds", 2) == 0
    assert cli("--a", two, "--b", two, "--min-rounds", 2) == 3  # one A run
    assert cli("--a", a, "--b", two) == 2  # kinds differ
    assert cli("--a", a, "--b", TMP / "missing") == 2  # not a run directory
    assert cli("--a", a) == 2  # usage


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
