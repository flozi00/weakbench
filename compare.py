#!/usr/bin/env python3
"""A/B gate: benchmark runs of the reference build (A) vs a candidate (B) of one model.

    python3 compare.py --a RUN [RUN ...] --b RUN [RUN ...] [--alpha 0.05]
                                  [--min-rounds 5] [--json OUT]

RUN = a run directory (or its id under runs/); all runs share one kind.

quality  items-<task>.jsonl paired by (task, lang, id) between the FIRST run of each
         side; later runs are replicates (noise floor). Right/wrong cells: exact
         McNemar; score-only cells (correct null, 0..1 score): paired differences,
         normal-approximation p. Holm over all cells, pooled McNemar, junk McNemar,
         request-error McNemar, logprob drift (KL, top-1, perplexity ratio) from
         drift-<lang>.jsonl.gz. INCONCLUSIVE when a cell with usable A items has no
         usable B pair, a drift language has documents on A1 only, or A1 and B1
         detected different protocols (run_info.json thinking_off_honored, drift_anchor).
speed    summary.json rows of all runs of a side grouped by (scenario, conc). Owner
         gate (port of vllm-suffix-hybrid bench/ovh/gatecmp.py): WIN/LOSS only when
         B's whole range clears A's whole range, judged only with two A runs (A1/B/A2);
         a TTFT LOSS is tolerated when e2e_p50_s WINs in the same cell (gate
         relaxation 2026-10-02). Errors: one-sided exact binomial test, every cell.
both     A1 vs B1 run args, served model id and bench_git_sha: differences warn.

Prints Markdown; the last line is `VERDICT: PASS|FAIL|INCONCLUSIVE - ...`.
Exit 0 PASS, 1 FAIL, 3 INCONCLUSIVE, 2 usage/data error.
"""
import argparse
import gzip
import json
import math
import statistics
import sys
import traceback
from collections import defaultdict
from pathlib import Path

from common import RUNS, read_jsonl

EXIT = {"PASS": 0, "FAIL": 1, "INCONCLUSIVE": 3}
GATED = {"agg_out_tps": 1, "stream_tps_p50": 1, "stream_tps_p10": 1,  # +1 higher is better
         "ttft_p50_ms": -1, "ttft_p99_ms": -1, "itl_p99_ms": -1, "e2e_p50_s": -1, "e2e_p90_s": -1}
INFO = ("spec_accept_rate", "spec_mean_len", "prefix_hit_rate", "preemptions", "tool_call_rate", "mean_out_tokens",
        "foreign_load")


class DataError(Exception):
    pass


def mcnemar(b, c):
    """Exact two-sided McNemar p: Binomial(b+c, 1/2) over the discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2 ** (n - 1))


def binom_sf(k, n, p):
    """One-sided exact binomial tail P(X >= k), X ~ Binomial(n, p), k <= n; log-space terms, so n may be large."""
    if k <= 0 or p >= 1:
        return 1.0
    if p <= 0:
        return 0.0
    lg, lp, lq = math.lgamma(n + 1), math.log(p), math.log1p(-p)
    return min(1.0, math.fsum(math.exp(lg - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * lp + (n - i) * lq)
                              for i in range(k, n + 1)))


def paired_ci(b, c, n):
    """B-A difference of paired proportions + 95% CI (b = A right B wrong, c = A wrong B right)."""
    d = (c - b) / n
    se = math.sqrt((b + c) - (c - b) ** 2 / n) / n
    return d, d - 1.96 * se, d + 1.96 * se


def paired_scores(diffs):
    """Mean paired difference, its 95% CI (± 1.96 sd/sqrt(n)) and the two-sided
    normal-approximation p of the paired t statistic."""
    n, d = len(diffs), statistics.fmean(diffs)
    se = statistics.stdev(diffs) / math.sqrt(n) if n > 1 else 0.0
    p = math.erfc(abs(d) / se / math.sqrt(2)) if se else float(d == 0 or n < 2)
    return d, d - 1.96 * se, d + 1.96 * se, p


def holm(ps):
    """Holm step-down adjusted p values, in input order."""
    adj, run = [1.0] * len(ps), 0.0
    for rank, i in enumerate(sorted(range(len(ps)), key=ps.__getitem__)):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = run
    return adj


def kl(ta, tb):
    """Approximate KL(A||B) at one position from two top-k dicts. Over the union of
    their tokens, a token missing from one side gets that side's k-th logprob; the
    mass left over (1 - sum p) is one tail bucket. Negative estimates clamp at 0."""
    keys, fa, fb = ta.keys() | tb.keys(), min(ta.values()), min(tb.values())
    la, lb = [ta.get(t, fa) for t in keys], [tb.get(t, fb) for t in keys]
    tail_a, tail_b = (max(0.0, 1 - math.fsum(map(math.exp, v))) for v in (la, lb))
    v = math.fsum(math.exp(x) * (x - y) for x, y in zip(la, lb))
    if tail_a > 1e-9 and tail_b > 1e-9:
        v += tail_a * math.log(tail_a / tail_b)
    return max(0.0, v)


def top1_agree(ta, tb):
    """A's argmax is (one of) B's argmax; bf16 logits tie, so 1e-6 counts as equal."""
    return max(tb.values()) - tb.get(max(ta, key=ta.get), -math.inf) <= 1e-6


def range_verdict(a, b, sign):
    if sign < 0:
        a, b = [-v for v in a], [-v for v in b]
    return "WIN" if min(b) > max(a) else "LOSS" if max(b) < min(a) else "TIE"


def p99(xs):
    s = sorted(xs)
    return s[max(0, math.ceil(0.99 * len(s)) - 1)]


def fmt(v):
    return "-" if v is None else f"{v:.0f}" if abs(v) >= 1000 else f"{v:.4g}"


def table(head, rows):
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] +
                     ["| " + " | ".join(map(str, r)) + " |" for r in rows])


def load_runs(names):
    runs = []
    for name in names:
        path = Path(name) if Path(name).is_dir() else RUNS / name
        try:
            runs.append((path, json.loads((path / "manifest.json").read_text())))
        except (OSError, ValueError) as exc:
            raise DataError(f"{name}: not a run directory ({exc})") from None
    return runs


def header(kind, a_runs, b_runs):
    rows = []
    for tag, runs in (("A", a_runs), ("B", b_runs)):
        for i, (path, m) in enumerate(runs, 1):
            srv = m.get("server") or {}
            ver = srv.get("version")
            rows.append([f"{tag}{i}", m.get("run_id") or path.name, m.get("arm"), srv.get("model"),
                         ver.get("version") if isinstance(ver, dict) else ver, srv.get("max_model_len")])
    return f"# A/B {kind} gate\n\n" + table(["side", "run", "arm", "model", "version", "max_model_len"], rows)


# manifest "args" that change what a run measures; compared when both A1 and B1 recorded them
ARGS = {"quality": ("tier", "tasks", "langs", "limit", "think", "temperature", "max_tokens", "image", "no_drift"),
        "speed": ("scenarios", "conc", "seconds", "temperature", "ramp", "vision")}


def protocol(kind, a_runs, b_runs):
    """WARN lines where A1 and B1 differ: run args, served model id, suite code version (bench_git_sha:
    task defaults and extraction rules live in code, not in args)."""
    ma, mb = a_runs[0][1], b_runs[0][1]
    # the code hash pins the suite exactly (uncommitted edits too); older runs only carry the git sha
    ver = "bench_code_sha" if ma.get("bench_code_sha") and mb.get("bench_code_sha") else "bench_git_sha"
    xa, xb = ({**{k: v for k, v in (m.get("args") or {}).items() if k in ARGS[kind]},
               "served model": (m.get("server") or {}).get("model"), ver: m.get(ver)}
              for m in (ma, mb))
    return [f"A1 and B1 differ in {k}: {xa[k]!r} vs {xb[k]!r}" for k in xa if k in xb and xa[k] != xb[k]]


def run_info(path):
    """What a quality run detected (thinking_off_honored, drift_anchor, drift_self_check); {} for older runs."""
    f = path / "run_info.json"
    return json.loads(f.read_text()) if f.exists() else {}


# ---------------------------------------------------------------- quality
def items(path):
    """{(task, lang, id): item} over all items-<task>.jsonl of a run."""
    out = {}
    for f in sorted(path.glob("items-*.jsonl")):
        for it in read_jsonl(f):
            out[(it.get("task") or f.stem[6:], it.get("lang") or "-", str(it.get("id")))] = it
    return out


def drift_docs(path):
    """{lang: its drift-<lang>.jsonl.gz} of a run. quality.py writes a file only
    when it has documents; they are read one at a time (read_docs): one doc's
    top-20 logprobs are ~4 MB of Python objects, and a small server runs
    this gate inside 256 MiB (an 8-doc trial pair peaked at 206 MiB loaded whole)."""
    return {f.name[6:-9]: f for f in sorted(path.glob("drift-*.jsonl.gz"))}


def read_docs(f):
    """The docs of one drift file in file order, ids ascending (quality.py writes 000, 001, ...)."""
    last = None
    with gzip.open(f, "rt") as fh:
        for doc in map(json.loads, filter(str.strip, fh)):
            if last is not None and str(doc["id"]) <= last:
                raise DataError(f"{f}: drift ids not ascending ({last}, {doc['id']})")
            last = str(doc["id"])
            yield doc


def doc_pairs(fx, fy):
    """(x, y) docs with the same id from two drift files, one pair in memory at a time."""
    ys = read_docs(fy)
    y = next(ys, None)
    for x in read_docs(fx):
        while y is not None and str(y["id"]) < str(x["id"]):
            y = next(ys, None)
        if y is not None and str(y["id"]) == str(x["id"]):
            yield x, y


def usable(it):
    """No request error, and graded: right/wrong, or a 0..1 score (continuous tasks: correct null)."""
    return it.get("error") is None and (it.get("correct") is not None or isinstance(it.get("score"), (int, float)))


def value(it):
    return float(bool(it["correct"])) if it.get("correct") is not None else float(it["score"])


def pair(x, y):
    """{(task, lang): [(x_item, y_item)]} over ids usable on both sides."""
    out = defaultdict(list)
    for k, ix in x.items():
        if k in y and usable(ix) and usable(y[k]):
            out[k[:2]].append((ix, y[k]))
    return out


def flips(ps):
    """(b, c) = (x right & y wrong, x wrong & y right)."""
    return (sum(1 for x, y in ps if x["correct"] and not y["correct"]),
            sum(1 for x, y in ps if y["correct"] and not x["correct"]))


def mean100(its):
    v = [value(it) for it in its if usable(it)]
    return 100 * statistics.fmean(v) if v else None


def side(its):
    toks = [it["completion_tokens"] for it in its if it.get("completion_tokens") is not None]
    return {"junk": sum(bool(it.get("junk")) for it in its),
            "truncated": sum(it.get("finish_reason") == "length" for it in its),
            "lang_mismatch": sum(it.get("lang_ok") is False for it in its),
            "out_tokens": statistics.fmean(toks) if toks else None}


def drift_stats(dx, dy):
    """Per lang: drift of run y against reference run x over documents with identical tokens."""
    out = {}
    for lang in sorted(dx.keys() & dy.keys()):
        kls, agree, nll, docs, differs = [], 0, [], 0, []
        for x, y in doc_pairs(dx[lang], dy[lang]):
            if x["tokens"] != y["tokens"]:
                differs.append(str(x["id"]))
                continue
            docs += 1
            nll += [(-a, -b) for a, b in zip(x["lp"], y["lp"]) if a is not None and b is not None]
            for ta, tb in zip(x["top"], y["top"]):
                if ta and tb:
                    kls.append(kl(ta, tb))
                    agree += top1_agree(ta, tb)
        r = {"docs": docs, "tokenization_differs": differs, "positions": len(kls)}
        if kls:
            r |= {"top1_pct": 100 * agree / len(kls), "kl_mean": statistics.fmean(kls), "kl_p99": p99(kls)}
        if nll:
            na, nb = (statistics.fmean(v) for v in zip(*nll))
            r |= {"ppl_a": math.exp(na), "ppl_b": math.exp(nb), "ppl_ratio": math.exp(nb - na)}
        out[lang] = r
    return out


def rel_drop(pairs):
    """(A mean, B mean, relative drop of B vs A) over paired items; drop None when A's mean is 0."""
    ma, mb = (statistics.fmean(value(p[i]) for p in pairs) for i in (0, 1))
    return ma, mb, (ma - mb) / ma if ma else None


def quality(a_runs, b_runs, args):
    A, B = [items(p) for p, _ in a_runs], [items(p) for p, _ in b_runs]
    prs = pair(A[0], B[0])
    rows = []
    for cell in sorted({k[:2] for k in A[0]} | {k[:2] for k in B[0]}):
        ps = prs.get(cell, [])
        r = {"task": cell[0], "lang": cell[1], "type": None, "n": len(ps), "status": "no pairs",
             "errors": [sum(it.get("error") is not None for k, it in X.items() if k[:2] == cell) for X in (A[0], B[0])]}
        if ps:
            score_only = {it.get("correct") is None for p in ps for it in p}
            if len(score_only) > 1:
                raise DataError(f"{cell[0]}/{cell[1]}: mixes right/wrong items with score-only items")
            if True in score_only:
                d, lo, hi, p = paired_scores([value(y) - value(x) for x, y in ps])
                r["type"] = "score"
            else:
                b, c = flips(ps)
                (d, lo, hi), p = paired_ci(b, c, len(ps)), mcnemar(b, c)
                r |= {"type": "binary", "b": b, "c": c}
            sa, sb = side([x for x, _ in ps]), side([y for _, y in ps])
            r |= {"rel_drop": rel_drop(ps)[2],
                  "mean_a": mean100(x for x, _ in ps), "mean_b": mean100(y for _, y in ps), "delta_pp": 100 * d,
                  "ci_pp": [100 * lo, 100 * hi], "p": p, **{k: [sa[k], sb[k]] for k in sa}}
        rows.append(r)
    tested = [r for r in rows if r["n"]]
    tol = args.max_rel_drop
    for r, ph in zip(tested, holm([r["p"] for r in tested])):
        d, sig = r["delta_pp"], ph < args.alpha
        r["p_holm"] = ph
        tolerated = tol is not None and (r["rel_drop"] or 0) <= tol  # a real but small drop inside the tolerance
        r["status"] = ("n<30" if r["type"] == "score" and r["n"] < 30 else "same" if d == 0 else
                       ("worse (tol.)" if sig and tolerated else "REGRESSION" if sig else "worse n.s.") if d < 0
                       else "IMPROVED" if sig else "better n.s.")

    def pairs_of(kind):
        return [p for r in tested if r["type"] == kind for p in prs[(r["task"], r["lang"])]]

    binp, scorep = pairs_of("binary"), pairs_of("score")
    allp = binp + scorep
    b, c = flips(binp)
    pooled = {"n": len(binp), "b": b, "c": c, "p": mcnemar(b, c)}
    if binp:
        d, lo, hi = paired_ci(b, c, len(binp))
        pooled |= {"delta_pp": 100 * d, "ci_pp": [100 * lo, 100 * hi],
                   "macro_delta_pp": statistics.fmean(r["delta_pp"] for r in tested if r["type"] == "binary")}
    pooled_score = {"n": len(scorep)}
    if scorep:
        d, lo, hi, p = paired_scores([value(y) - value(x) for x, y in scorep])
        pooled_score |= {"delta_pts": 100 * d, "ci_pts": [100 * lo, 100 * hi], "p": p}
    ja = sum(bool(x.get("junk")) and not y.get("junk") for x, y in allp)
    jb = sum(bool(y.get("junk")) and not x.get("junk") for x, y in allp)
    junk = {"a_only": ja, "b_only": jb, "p": mcnemar(ja, jb)}
    # request errors on ids both sides ran (the pairing drops them): A-only vs B-only, exact McNemar
    both = [(A[0][k], B[0][k]) for k in A[0].keys() & B[0].keys()]
    ea = sum(x.get("error") is not None and usable(y) for x, y in both)
    eb = sum(usable(x) and y.get("error") is not None for x, y in both)
    errs = {"a_only": ea, "b_only": eb, "p": mcnemar(ea, eb)}

    # noise floor: mean |change| per item between replicates (= flip rate for right/wrong items)
    comps = [("A~B", prs)] + [(f"{t}1~{t}{k}", pair(X[0], Y)) for t, X in (("A", A), ("B", B))
                              for k, Y in enumerate(X[1:], 2)]
    noise = []
    if len(comps) > 1:
        for cell in [*((r["task"], r["lang"]) for r in rows), None]:
            row = {"task": cell[0] if cell else "pooled", "lang": cell[1] if cell else ""}
            for label, pr in comps:
                ps = [p for k, v in pr.items() if cell in (None, k) for p in v]
                row[label] = [math.fsum(abs(value(x) - value(y)) for x, y in ps), len(ps)]
            for key, X in (("mean_a", A), ("mean_b", B)):
                row[key] = [mean100(it for k, it in R.items() if cell in (None, k[:2])) for R in X]
            noise.append(row)

    DA, DB = [drift_docs(p) for p, _ in a_runs], [drift_docs(p) for p, _ in b_runs]
    dcomps = ([("A1~B1", DA[0], DB[0])] if DA[0] and DB[0] else []) + [
        (f"{t}1~{t}{k}", X[0], Y) for t, X in (("A", DA), ("B", DB)) for k, Y in enumerate(X[1:], 2) if X[0] and Y]
    drift = {label: drift_stats(x, y) for label, x, y in dcomps}

    ia, ib = run_info(a_runs[0][0]), run_info(b_runs[0][0])
    fails, warns = [], []
    # compared as sent: quality.py changes thinking-off budgets only on False (None = its preflight failed)
    sent = (("thinking_off_honored", lambda v: v is False), ("drift_anchor", lambda v: v),
            ("judge_model", lambda v: v))
    undecided = [f"A1 and B1 did not run the same protocol: {k} {ia[k]!r} vs {ib[k]!r}"
                 for k, f in sent if k in ia and k in ib and f(ia[k]) != f(ib[k])]
    for r in rows:
        tag = f"{r['task']}/{r['lang']}"
        if r["status"] == "REGRESSION":
            fails.append(f"{tag} REGRESSION {r['delta_pp']:+.1f} points" + (
                f" (-{100 * r['rel_drop']:.1f}% relative)" if tol is not None and r["rel_drop"] else "")
                + f" (Holm p={r['p_holm']:.2g})")
        elif r["status"] == "worse (tol.)":
            warns.append(f"{tag} real drop {r['delta_pp']:+.1f} points, -{100 * r['rel_drop']:.1f}% relative: "
                         f"inside the {100 * tol:g}% tolerance")
        ua = 0 if r["n"] else sum(usable(it) for k, it in A[0].items() if k[:2] == (r["task"], r["lang"]))
        if ua:  # A answered, B gave nothing to pair: B's failure must not pass as "untested"
            undecided.append(f"{tag}: {ua} usable A items, no usable B pair")
        elif not r["n"]:
            warns.append(f"{tag}: no paired items")
        if r["errors"][1] > r["errors"][0]:
            warns.append(f"{tag}: B request errors {r['errors'][1]} > A {r['errors'][0]}")
    relative = {}
    if tol is None:
        if binp and pooled["delta_pp"] < 0 and pooled["p"] < args.alpha:
            fails.append(f"pooled {pooled['delta_pp']:+.2f}pp (McNemar p={pooled['p']:.2g})")
    else:  # pooled pools are large enough for the point estimate itself: any drop over the tolerance fails
        langs = sorted({r["lang"] for r in tested if r["type"] == "binary"})
        pools = [("pooled", binp), ("pooled score", scorep)] + [
            (f"pooled {lang}", [p for r in tested if r["type"] == "binary" and r["lang"] == lang
                                for p in prs[(r["task"], r["lang"])]]) for lang in langs]
        for label, ps in pools:
            if ps:
                ma, mb, rel = rel_drop(ps)
                relative[label] = {"n": len(ps), "mean_a": 100 * ma, "mean_b": 100 * mb,
                                   "rel_drop_pct": None if rel is None else 100 * rel}
                if rel is not None and rel > tol:
                    fails.append(f"{label} {100 * ma:.1f} -> {100 * mb:.1f}: -{100 * rel:.1f}% relative, "
                                 f"over the {100 * tol:g}% tolerance")
    if jb > ja and junk["p"] < args.alpha:
        fails.append(f"more junk on B ({jb} B-only vs {ja} A-only, p={junk['p']:.2g})")
    if eb > ea and errs["p"] < args.alpha:
        fails.append(f"B request errors ({eb} B-only vs {ea} A-only, p={errs['p']:.2g})")
    for lang, d in drift.get("A1~B1", {}).items():
        k, q, t = d.get("kl_mean", 0.0), d.get("ppl_ratio", 1.0), d.get("top1_pct", 100.0)
        d["status"] = ("no data" if not (d["positions"] or "ppl_ratio" in d) else
                       "FAIL" if k > args.kl_fail or q > args.ppl_fail else
                       "WARN" if k > args.kl_warn or q > args.ppl_warn or t < args.top1_warn else "ok")
        what = f"drift {lang} {d['status']} (mean KL {k:.4f}, ppl ratio {q:.4f}, top-1 {t:.1f}%)"
        (fails if d["status"] == "FAIL" else warns if d["status"] != "ok" else []).append(what)
        if d["tokenization_differs"]:
            warns.append(f"drift {lang}: tokenization differs, skipped docs {', '.join(d['tokenization_differs'])}")
    undecided += [f"drift {lang}: documents on A1, none on B1 (B1 drift self-check: "
                  f"{ib.get('drift_self_check') or 'not recorded'})" for lang in sorted(DA[0])
                  if DA[0][lang] and not DB[0].get(lang)]
    warns += [f"drift {lang}: documents on B1 only" for lang in sorted(DB[0]) if DB[0][lang] and not DA[0].get(lang)]
    if not allp:
        undecided.append("no paired items")
    verdict = "FAIL" if fails else "INCONCLUSIVE" if undecided else "PASS"
    line = ("; ".join(fails) + (f"; not judged: {'; '.join(undecided)}" if undecided else "") if fails else
            "; ".join(undecided) if undecided else
            f"{len(allp)} pairs in {len(tested)} cells" +
            (f", pooled {pooled['delta_pp']:+.2f}pp (p={pooled['p']:.2g})" if binp else "") +
            (f", score cells {pooled_score['delta_pts']:+.2f} points" if scorep else ""))

    ab = lambda v, f=str: f"{f(v[0])}/{f(v[1])}"  # noqa: E731
    body = [[r["task"], r["lang"], r["type"] or "-", r["n"], *(["-"] * 6 if not r["n"] else [
        f"{r['mean_a']:.1f}", f"{r['mean_b']:.1f}", f"{r['delta_pp']:+.1f}",
        "[{:+.1f}, {:+.1f}]".format(*r["ci_pp"]), f"{r['p']:.2g}", f"{r['p_holm']:.2g}"]), r["status"],
        ab(r["errors"]), *(["-"] * 4 if not r["n"] else [ab(r["junk"]), ab(r["truncated"]),
                                                          ab(r["lang_mismatch"]), ab(r["out_tokens"], fmt)])]
            for r in rows]
    out = [header("quality", a_runs, b_runs),
           f"\n## Items: A1 vs B1, paired; Holm over {len(tested)} cells, alpha {args.alpha}",
           "A/B = accuracy % (binary) or mean score x100 (score), delta and CI in points B-A. binary: exact "
           "McNemar, paired-proportion CI; score: paired differences, mean ± 1.96 sd/sqrt(n), normal-approximation "
           "p; score cells with n<30 are shown, never judged.\n",
           table(["task", "lang", "type", "pairs", "A", "B", "delta", "95% CI", "p", "p Holm", "status",
                  "errors A/B", "junk A/B", "truncated A/B", "lang mismatch A/B", "out tok A/B"], body)]
    if binp:
        out.append(f"\n**Pooled binary** over {len(binp)} pairs: A-only right {b}, B-only right {c}, delta "
                   f"{pooled['delta_pp']:+.2f}pp [{pooled['ci_pp'][0]:+.2f}, {pooled['ci_pp'][1]:+.2f}], McNemar "
                   f"p={pooled['p']:.2g}; macro-average delta {pooled['macro_delta_pp']:+.2f}pp.")
    if scorep:
        out.append(f"\n**Pooled score** over {len(scorep)} pairs (not gated): mean delta "
                   f"{pooled_score['delta_pts']:+.2f} points [{pooled_score['ci_pts'][0]:+.2f}, "
                   f"{pooled_score['ci_pts'][1]:+.2f}], p={pooled_score['p']:.2g}.")
    out.append(f"\n**Junk** (paired): A-only junk {ja}, B-only junk {jb}, McNemar p={junk['p']:.2g}.")
    out.append(f"\n**Request errors** (ids run on both sides): A-only {ea}, B-only {eb}, McNemar p={errs['p']:.2g}.")
    if noise:
        labels = [label for label, _ in comps]
        out += ["\n## Noise floor: mean |change| per item x100 between two runs (flip rate for right/wrong items)",
                "A~B must clear the replicate change (A1~A2, B1~B2: batch nondeterminism) to mean anything.\n",
                table(["task", "lang", *labels, "A runs", "B runs"],
                      [[r["task"], r["lang"], *(f"{100 * r[x][0] / r[x][1]:.1f}% ({r[x][0]:.4g}/{r[x][1]})"
                                                if r[x][1] else "-" for x in labels),
                        *("/".join(fmt(v) for v in r[key]) for key in ("mean_a", "mean_b"))] for r in noise])]
    if drift:
        cols = (("top1_pct", ".2f"), ("kl_mean", ".5f"), ("kl_p99", ".4f"), ("ppl_a", ".4f"), ("ppl_b", ".4f"),
                ("ppl_ratio", ".4f"))
        out += ["\n## Drift: logprobs over identical token sequences, first run of a pair = reference",
                f"Gate (A1~B1 only): WARN mean KL > {args.kl_warn} or ppl ratio > {args.ppl_warn} or top-1 < "
                f"{args.top1_warn}%; FAIL mean KL > {args.kl_fail} or ppl ratio > {args.ppl_fail}. "
                "Replicate rows are the noise floor.\n",
                table(["lang", "runs", "docs", "tokenization differs", "positions", "top-1 %", "mean KL", "p99 KL",
                       "ppl ref", "ppl cmp", "ppl ratio", "status"],
                      [[lang, label, d["docs"], len(d["tokenization_differs"]), d["positions"],
                        *(f"{d[k]:{f}}" if k in d else "-" for k, f in cols), d.get("status", "noise")]
                       for lang in sorted({lang for v in drift.values() for lang in v})
                       for label, per in drift.items() if (d := per.get(lang))])]
    res = {"kind": "quality", "verdict": verdict, "fails": fails, "inconclusive": undecided, "warns": warns,
           "max_rel_drop": tol, "relative": relative,
           "cells": rows, "pooled": pooled, "pooled_score": pooled_score, "junk": junk, "request_errors": errs,
           "noise": noise, "drift": drift}
    return res, "\n".join(out), line


# ---------------------------------------------------------------- speed
def vals(rows, m):
    return [r[m] for r in rows if r.get(m) is not None]


def span(xs):
    lo, hi = fmt(min(xs)), fmt(max(xs))
    return lo if lo == hi else f"{lo}-{hi}"


def gate_cell(r):
    if r is None:
        return "-"
    pct = r["median_change_pct"]
    return (f"A {span(r['a']) if r['a'] else '-'} B {span(r['b']) if r['b'] else '-'} {r['verdict'] or 'n/a'}"
            + (" (tol.)" if r.get("tolerated") else "") + (f" {pct:+.1f}%" if pct is not None else ""))


def speed(a_runs, b_runs, args):
    groups = {}
    for i, runs in enumerate((a_runs, b_runs)):
        for path, _ in runs:
            try:
                rows = json.loads((path / "summary.json").read_text())
            except (OSError, ValueError) as exc:
                raise DataError(f"{path}: no readable summary.json ({exc})") from None
            for r in rows:
                groups.setdefault((r.get("scenario"), r.get("conc")), ([], []))[i].append(r)
    scenarios = list(dict.fromkeys(s for s, _ in groups))
    cells = []
    for scen, conc in sorted(groups, key=lambda k: (scenarios.index(k[0]), k[1])):
        xa, xb = groups[(scen, conc)]
        metrics = {}
        for m, sign in GATED.items():
            a, b = vals(xa, m), vals(xb, m)
            if a or b:
                ma, mb = (statistics.median(x) if x else None for x in (a, b))
                metrics[m] = {"a": a, "b": b, "verdict": range_verdict(a, b, sign) if a and b else None,
                              "median_change_pct": 100 * (mb / ma - 1) if ma and mb is not None else None}
        if metrics.get("e2e_p50_s", {}).get("verdict") == "WIN":  # better end-to-end beats worse TTFT
            for m in ("ttft_p50_ms", "ttft_p99_ms"):
                if metrics.get(m, {}).get("verdict") == "LOSS":
                    metrics[m]["tolerated"] = True
        ea, eb, qa, qb = (sum(r.get(k) or 0 for r in x) for k in ("errors", "requests") for x in (xa, xb))
        qa, qb = qa + ea, qb + eb  # speed.py's "requests" counts the successful ones
        p = binom_sf(eb, ea + eb, qb / max(qa + qb, 1))
        cells.append({"scenario": scen, "conc": conc, "rounds": [len(xa), len(xb)],
                      "short": min(len(xa), len(xb)) < args.min_rounds, "metrics": metrics,
                      # another client on the server during any round: its numbers say nothing about the build
                      "busy": max((r.get("foreign_load") or 0 for r in xa + xb), default=0),
                      # one-sided exact binomial: under H0 each error lands on B with B's share of all requests
                      "errors": {"a": [ea, qa], "b": [eb, qb], "p": p, "loss": p < args.alpha},
                      "info": {m: [(max if m == "foreign_load" else statistics.median)(v) if v else None
                                   for v in (vals(xa, m), vals(xb, m))]
                               for m in INFO if vals(xa, m) or vals(xb, m)}})

    one_a = len({path.resolve() for path, _ in a_runs}) < 2  # the same run twice is still one A run
    losses, short, busy, count = [], [], [], {"WIN": 0, "TIE": 0, "tolerated": 0}
    for c in cells:
        tag = f"{c['scenario']} c={c['conc']}"
        if c["errors"]["loss"]:  # judged in every cell: a B cell where every request failed must FAIL
            losses.append(f"{tag} errors")
        if c["short"]:  # a range from too few rounds proves nothing either way
            short.append(f"{tag} (A {c['rounds'][0]} / B {c['rounds'][1]})")
        elif c["busy"]:
            busy.append(f"{tag} ({c['busy']} foreign requests)")
        elif not one_a:
            for m, r in c["metrics"].items():
                if r["verdict"] == "LOSS" and r.get("tolerated"):
                    count["tolerated"] += 1
                elif r["verdict"] == "LOSS":
                    losses.append(f"{tag} {m}")
                elif r["verdict"]:
                    count[r["verdict"]] += 1
    verdict = "FAIL" if losses else "INCONCLUSIVE" if one_a or short or busy else "PASS"
    unjudged = "; ".join((["need A1 and A2 runs (A/B/A): with one A run identical servers fail the separated-range "
                           "rule ~40% of the time"] if one_a else [])
                         + ([f"fewer than {args.min_rounds} rounds: {', '.join(short)}"] if short else [])
                         + ([f"server busy with other traffic: {', '.join(busy)}"] if busy else []))
    line = (f"{len(losses)} untolerated LOSS: {', '.join(losses)}" + (f"; not judged, {unjudged}" if unjudged else "")
            if losses else unjudged if unjudged else f"WIN {count['WIN']} / TIE {count['TIE']}"
            + (f" / tolerated TTFT LOSS {count['tolerated']}" if count["tolerated"] else ""))

    out = [header("speed", a_runs, b_runs),
           "\nCells: `A min-max B min-max VERDICT +x%` over all rounds of a side, % = B median vs A median. "
           "WIN/LOSS only when the ranges do not overlap; `(tol.)` = TTFT LOSS tolerated by an e2e_p50_s WIN; "
           f"`(short)` = fewer than {args.min_rounds} rounds on a side, `(busy)` = other clients were "
           "using the server during a round (foreign_load > 0); neither is judged, nor is any range with one A run. "
           "errors/requests (all requests): LOSS when a one-sided exact binomial test (each error lands on B with "
           f"B's share of the requests) gives p < {args.alpha}; judged in every cell."]
    for scen in scenarios:
        cs = [c for c in cells if c["scenario"] == scen]
        gated = [m for m in GATED if any(m in c["metrics"] for c in cs)]
        info = [m for m in INFO if any(m in c["info"] for c in cs)]
        body = [[f"{c['conc']}{' (short)' if c['short'] else ' (busy)' if c['busy'] else ''}",
                 "{}/{}".format(*c["rounds"]),
                 *(gate_cell(c["metrics"].get(m)) for m in gated),
                 "A {}/{} B {}/{}".format(*c["errors"]["a"], *c["errors"]["b"])
                 + (" LOSS p={:.2g}".format(c["errors"]["p"]) if c["errors"]["loss"] else ""),
                 *(f"A {fmt(c['info'][m][0])} B {fmt(c['info'][m][1])}" if m in c["info"] else "-" for m in info)]
                for c in cs]
        out += [f"\n## {scen}", table(["conc", "rounds A/B", *gated, "errors/requests", *info], body)]
    res = {"kind": "speed", "verdict": verdict, "losses": losses, "short": short, "busy": busy, "counts": count,
           "cells": cells, "warns": []}
    return res, "\n".join(out), line


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", nargs="+", required=True, metavar="RUN", help="reference runs (first is paired)")
    ap.add_argument("--b", nargs="+", required=True, metavar="RUN", help="candidate runs (first is paired)")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--max-rel-drop", type=float, metavar="R",
                    help="quality tolerance, e.g. 0.05 (exploratory trials): a cell FAILs only when its drop is "
                         "significant AND over R relative to A; pooled, per-language pooled and pooled-score "
                         "drops over R FAIL outright. Default: strict, any significant regression FAILs")
    # 5 = the A1/B/A2 protocol. With 3 vs 3 rounds two IDENTICAL servers give non-overlapping
    # ranges by chance 10% of the time per metric — an A/A would usually "fail".
    ap.add_argument("--min-rounds", type=int, default=5, help="speed: values per side and cell to judge it")
    ap.add_argument("--kl-warn", type=float, default=0.005, help="drift: mean KL(A||B)")
    ap.add_argument("--kl-fail", type=float, default=0.02)
    ap.add_argument("--ppl-warn", type=float, default=1.005, help="drift: perplexity ratio B/A")
    ap.add_argument("--ppl-fail", type=float, default=1.02)
    ap.add_argument("--top1-warn", type=float, default=98.0, help="drift: top-1 agreement, %%")
    ap.add_argument("--json", metavar="OUT", help="write the machine-readable result here")
    args = ap.parse_args(argv)
    try:
        a, b = load_runs(args.a), load_runs(args.b)
        ka, kb = ({m.get("kind") for _, m in runs} for runs in (a, b))
        if len(ka) != 1 or ka != kb or not ka <= {"quality", "speed"}:
            raise DataError(f"all runs must share one kind (quality|speed): A {sorted(map(str, ka))}, "
                            f"B {sorted(map(str, kb))}")
        kind = ka.pop()
        res, report, line = (quality if kind == "quality" else speed)(a, b, args)
        res["warns"] = warns = protocol(kind, a, b) + res["warns"]
        if warns:
            report += "\n\n## Warnings\n" + "\n".join(f"- {w}" for w in warns)
            if res["verdict"] != "FAIL":
                line += f"; WARN: {'; '.join(warns)}"
        res["verdict_line"] = f"VERDICT: {res['verdict']} - {line}"
        res["runs"] = {"a": [str(p) for p, _ in a], "b": [str(p) for p, _ in b]}
        if args.json:
            Path(args.json).write_text(json.dumps(res, indent=2))
        print(report + "\n\n" + res["verdict_line"])
        return EXIT[res["verdict"]]
    except DataError as exc:
        print(f"compare.py: {exc}", file=sys.stderr)
    except Exception:  # noqa: BLE001 - a crash must never read as FAIL (exit 1)
        traceback.print_exc()
    return 2


if __name__ == "__main__":
    sys.exit(main())
