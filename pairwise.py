"""Pairwise judge A/B: which arm wrote the better free-text reply to the same prompt?

    uv run pairwise.py --a RUN_A --b RUN_B --judge-url URL [--tasks t1,t2] [--limit N]

RUN = a quality run id under runs/ (or a path). For every (task, lang, id) both runs answered without
error, the judge compares the two replies TWICE, once in each order; a side wins only when it wins in both orders,
everything else is a tie. That cancels the judge's position bias instead of measuring it. Per task: B's wins, ties,
losses, the win rate (ties count half) with a 95 % Wilson interval, and a two-sided sign test over the decided
pairs. A rate whose interval holds 0.5 is no evidence either way.

`--xl RUN` instead checks ONE run for cross-lingual consistency: for every item id answered in both German and
English, the judge says whether the two replies agree in substance (same facts, numbers, conclusion, advice).
Reported per task as the share of SAME, with its Wilson interval — a model that knows or decides things
differently per language shows up here even when both languages score alike.

`--audit RUN` checks the JUDGE on one run's own replies: the same reply as A and B must be a TIE (both orders),
and a reply padded with an empty paragraph of filler must not beat the original (length bias). A judge with a
high padded-win rate or a low identical-tie rate is not fit to grade free text.

`--rejudge RUN` re-scores one run's judge-graded items with the judge given here and compares to the scores the
run already holds: per task the mean absolute score difference, the Pearson correlation and the share of
identical scores. Low agreement between two capable judges means the cell measures the judge, not the model.

This is the free-text complement to compare.py: a checklist or chrF can stay flat while one arm writes clearly
better (or worse) text. Replies are the stored first 4,000 characters. The judge must not be either arm's server.
Appends one row per task to results/pairwise.jsonl.
"""
import argparse
import collections
import concurrent.futures
import math
import re
import sys
from pathlib import Path

import common
import quality
import tasks_judge

FREE_TEXT = ("lang_quality", "open_checklist", "summary_faithful", "summarize", "simplify", "lang_switch",
             "eqbench_cw", "multiturn")
PAIR_JUDGE = """You compare two assistant replies to the same user request. Judge helpfulness, correctness, and the
quality of the language (grammar, fluency, the requested language and register). Length is not a merit: a longer
reply wins only if the extra content is needed. Reply with exactly one token: A, B or TIE.

USER REQUEST:
{prompt}

REPLY A:
{a}

REPLY B:
{b}

Better reply (A, B or TIE):"""


XL_JUDGE = """Two replies to the same request, one asked in German and one in English. Ignore the language, style,
length and formatting. Do they agree in substance — the same facts, numbers, conclusions and recommendations?
Minor omissions are fine; a contradiction or a different conclusion is not. Reply with exactly one word: SAME or
DIFFERENT.

REQUEST (English version):
{prompt}

GERMAN REPLY:
{de}

ENGLISH REPLY:
{en}

Verdict:"""
XL_TASKS = ("open_checklist", "lang_quality", "simplify", "summary_faithful", "lang_switch")


def xl_same(prompt, de, en):
    out = tasks_judge.judge("", XL_JUDGE.format(prompt=prompt[:8000], de=de, en=en), max_tokens=8).upper()
    if not re.match(r"\W*(SAME|DIFFERENT)\b", out):
        raise ValueError(f"unparseable consistency verdict {out!r}")
    return out.lstrip(" *`\"").startswith("SAME")


def cross_lingual(run, args, judge_model):
    recs = records(run)
    wanted = set(args.tasks.split(",")) if args.tasks else set(XL_TASKS)
    pairs = sorted({(t, i) for t, lang, i in recs if t in wanted and (t, "de", i) in recs and (t, "en", i) in recs})
    if not pairs:
        raise SystemExit("error: no item answered in both German and English")
    items = {}
    for task in sorted({t for t, _ in pairs}):
        items |= {(task, str(i["id"])): i for i in quality.TASKS[task]["load"]("en", None, quality.SEED)}
    by_task = collections.defaultdict(list)
    for t, i in pairs:
        if (t, i) in items:
            by_task[t].append(i)
    jobs = [(t, i) for t, ids in by_task.items() for i in (ids[:args.limit] if args.limit else ids)]
    same = collections.defaultdict(list)
    with concurrent.futures.ThreadPoolExecutor(args.concurrency) as ex:
        futs = {ex.submit(xl_same, prompt_text(items[k]), recs[(k[0], "de", k[1])]["reply"],
                          recs[(k[0], "en", k[1])]["reply"]): k for k in jobs}
        for f in concurrent.futures.as_completed(futs):
            try:
                same[futs[f][0]].append(f.result())
            except Exception as exc:  # noqa: BLE001
                print(f"warning: {futs[f]}: {exc}", file=sys.stderr)
    print(f"\n## cross-lingual consistency of {run.name}, judge {judge_model}\n")
    print("| task | n | same | 95% CI |\n|---|---|---|---|")
    rows = []
    for t, xs in sorted(same.items()):
        lo, hi = wilson(sum(xs), len(xs))
        rows.append({"kind": "xl", "task": t, "run": run.name, "judge_model": judge_model, "n": len(xs),
                     "same_rate": round(sum(xs) / len(xs), 4), "ci95": [lo, hi]})
        print(f"| {t} | {len(xs)} | {rows[-1]['same_rate']} | {lo}-{hi} |")
    return rows


FILLER = {"en": "\n\nIn summary, it is worth emphasising that the points above are important, and that careful "
                "consideration of all relevant aspects is always advisable. Every situation is different, and it is "
                "generally a good idea to take one's time, to reflect thoroughly, and to consult additional sources "
                "where appropriate, so that a well-rounded understanding can be reached.",
          "de": "\n\nZusammenfassend lässt sich sagen, dass die oben genannten Punkte wichtig sind und dass eine "
                "sorgfältige Abwägung aller relevanten Aspekte stets ratsam ist. Jede Situation ist anders, und es "
                "ist grundsätzlich sinnvoll, sich Zeit zu nehmen, gründlich nachzudenken und gegebenenfalls weitere "
                "Quellen heranzuziehen, um ein umfassendes Verständnis zu gewinnen."}


def audit(run, args, judge_model):
    recs = records(run)
    wanted = set(args.tasks.split(",")) if args.tasks else set(FREE_TEXT)
    keys = sorted(k for k in recs if k[0] in wanted)[:args.limit or 40]
    if not keys:
        raise SystemExit("error: no free-text replies in this run")
    items = {}
    for task, lang in sorted({k[:2] for k in keys}):
        items |= {(task, lang, str(i["id"])): i for i in quality.TASKS[task]["load"](lang, None, quality.SEED)}
    keys = [k for k in keys if k in items]
    out = {"identical_tie": [], "padded_wins": []}
    with concurrent.futures.ThreadPoolExecutor(args.concurrency) as ex:
        same = {ex.submit(compare_pair, prompt_text(items[k]), recs[k]["reply"], recs[k]["reply"]): k for k in keys}
        pad = {ex.submit(compare_pair, prompt_text(items[k]), recs[k]["reply"], recs[k]["reply"] + FILLER.get(k[1], FILLER["en"])): k
               for k in keys}
        for f in concurrent.futures.as_completed(list(same) + list(pad)):
            try:
                v = f.result()
            except Exception as exc:  # noqa: BLE001
                print(f"warning: {exc}", file=sys.stderr)
                continue
            if f in same:
                out["identical_tie"].append(v == 0)
            else:
                out["padded_wins"].append(v == 1)  # B (the padded copy) won in both orders
    row = {"kind": "audit", "run": run.name, "judge_model": judge_model}
    for k, xs in out.items():
        row[k] = round(sum(xs) / len(xs), 4) if xs else None
        row[f"{k}_n"] = len(xs)
    print(f"\n## judge audit, {judge_model} on {run.name}\n\nidentical replies judged TIE: {row['identical_tie']} "
          f"(want 1.0, n={row['identical_tie_n']})\npadded copy beats the original: {row['padded_wins']} "
          f"(want 0.0, n={row['padded_wins_n']})")
    return [row]


JUDGED = ("lang_quality", "open_checklist", "summary_faithful", "summary_xl", "wmt_mqm", "proofs", "roleplay",
          "aa_omniscience", "eqbench_cw")


def rejudge(run, args, judge_model):
    latest = {}
    for f in sorted(run.glob("items-*.jsonl")):
        for r in common.read_jsonl(f):
            latest[(r["task"], r["lang"], str(r["id"]))] = r
    wanted = set(args.tasks.split(",")) if args.tasks else set(JUDGED)
    keys = [k for k, r in sorted(latest.items()) if k[0] in wanted and not r.get("error") and r.get("reply")
            and isinstance(r.get("score"), (int, float))]
    if not keys:
        raise SystemExit("error: no judge-graded item with a score in this run")
    items = {}
    for task, lang in sorted({k[:2] for k in keys}):
        items |= {(task, lang, str(i["id"])): i for i in quality.TASKS[task]["load"](lang, None, quality.SEED)}
    keys = [k for k in keys if k in items][:args.limit or None]

    def one(k):
        rec = latest[k]
        reply = {"content": rec["reply"], "reasoning": "", "tool_calls": [], "finish_reason": "stop",
                 **({"turns": [rec["reply"]]} if items[k].get("turns") else {})}
        return quality.TASKS[k[0]]["score"](items[k], reply)["score"]
    pairs = collections.defaultdict(list)
    with concurrent.futures.ThreadPoolExecutor(args.concurrency) as ex:
        futs = {ex.submit(one, k): k for k in keys}
        for f in concurrent.futures.as_completed(futs):
            k = futs[f]
            try:
                pairs[k[0]].append((float(latest[k]["score"]), float(f.result())))
            except Exception as exc:  # noqa: BLE001
                print(f"warning: {k}: {exc}", file=sys.stderr)
    print(f"\n## judge agreement on {run.name}: stored vs {judge_model}\n")
    print("| task | n | mean abs diff | pearson | identical |\n|---|---|---|---|---|")
    rows = []
    for t, ps in sorted(pairs.items()):
        a, b = [x for x, _ in ps], [y for _, y in ps]
        ma, mb = sum(a) / len(a), sum(b) / len(b)
        va, vb = sum((x - ma) ** 2 for x in a), sum((y - mb) ** 2 for y in b)
        r = sum((x - ma) * (y - mb) for x, y in ps) / math.sqrt(va * vb) if va and vb else None
        row = {"kind": "rejudge", "task": t, "run": run.name, "judge_model": judge_model, "n": len(ps),
               "mad": round(sum(abs(x - y) for x, y in ps) / len(ps), 4), "pearson": None if r is None else round(r, 4),
               "identical": round(sum(x == y for x, y in ps) / len(ps), 4)}
        rows.append(row)
        print(f"| {t} | {row['n']} | {row['mad']} | {row['pearson']} | {row['identical']} |")
    return rows


def records(run):
    latest = {}
    for f in sorted(run.glob("items-*.jsonl")):
        for r in common.read_jsonl(f):
            latest[(r["task"], r["lang"], str(r["id"]))] = r
    return {k: r for k, r in latest.items() if not r.get("error") and r.get("reply")}


def prompt_text(item):
    """The conversation the model saw, as plain text (multi-turn items: every user turn)."""
    msgs = [m for turn in item["turns"] for m in turn] if item.get("turns") else item["messages"]
    return "\n\n".join(f"[{m['role']}] {m['content']}" for m in msgs if isinstance(m.get("content"), str))


def verdict(prompt, a, b):
    out = tasks_judge.judge("", PAIR_JUDGE.format(prompt=prompt[:12000], a=a, b=b), max_tokens=8).upper()
    m = re.match(r"\W*(TIE|A|B)\b", out)
    if not m:
        raise ValueError(f"unparseable pairwise verdict {out!r}")
    return m[1]


def compare_pair(prompt, a, b):
    """+1 B better, -1 A better, 0 tie; B is shown second, then first."""
    first, second = verdict(prompt, a, b), verdict(prompt, b, a)
    b_wins = (first == "B") + (second == "A")
    a_wins = (first == "A") + (second == "B")
    return 1 if b_wins == 2 else -1 if a_wins == 2 else 0


def wilson(k, n, z=1.96):
    if not n:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 4), round(c + h, 4))


def sign_test(wins, losses):
    """Two-sided exact binomial p of wins vs losses (ties dropped)."""
    n = wins + losses
    if not n:
        return 1.0
    k = min(wins, losses)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def summarize(outcomes):
    w, t, lo = (sum(o == v for o in outcomes) for v in (1, 0, -1))
    n = len(outcomes)
    lo_ci, hi_ci = wilson(w + t / 2, n)
    return {"n": n, "b_wins": w, "ties": t, "b_losses": lo, "b_win_rate": round((w + t / 2) / n, 4) if n else None,
            "ci95": [lo_ci, hi_ci], "sign_p": round(sign_test(w, lo), 5)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--a")
    ap.add_argument("--b")
    ap.add_argument("--xl", help="one run: judge DE vs EN replies of the same item for consistency (no --a/--b)")
    ap.add_argument("--audit", help="one run: test the judge for position consistency and length bias")
    ap.add_argument("--rejudge", help="one run: re-score its judge-graded items with this judge, report agreement")
    ap.add_argument("--judge-url", required=True)
    ap.add_argument("--judge-model")
    ap.add_argument("--tasks", help=f"default: the free-text tasks both runs have ({','.join(FREE_TEXT)})")
    ap.add_argument("--limit", type=int, help="pairs per (task, lang)")
    ap.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args(argv)
    if sum(map(bool, (args.xl, args.audit, args.rejudge, args.a and args.b))) != 1:
        ap.error("pass either --a and --b, or --xl, --audit or --rejudge")
    judge_model = common.server_info(args.judge_url, args.judge_model)["model"]
    tasks_judge.configure(args.judge_url, judge_model, None)
    tasks_judge.preflight()
    if args.xl or args.audit or args.rejudge:
        one = args.xl or args.audit or args.rejudge
        run = Path(one) if Path(one).is_dir() else common.RUNS / one
        rows = (cross_lingual if args.xl else audit if args.audit else rejudge)(run, args, judge_model)
        common.RESULTS.mkdir(parents=True, exist_ok=True)
        for row in rows:
            common.append_jsonl(common.RESULTS / "pairwise.jsonl", row)
        return rows
    run_a, run_b = (Path(x) if Path(x).is_dir() else common.RUNS / x for x in (args.a, args.b))  # as compare.py
    ra, rb = records(run_a), records(run_b)
    wanted = set(args.tasks.split(",")) if args.tasks else set(FREE_TEXT)
    keys = sorted(k for k in ra.keys() & rb.keys() if k[0] in wanted)
    if not keys:
        raise SystemExit("error: no free-text item answered by both runs")
    items = {}
    for task, lang in sorted({(k[0], k[1]) for k in keys}):
        items |= {(task, lang, str(i["id"])): i for i in quality.TASKS[task]["load"](lang, None, quality.SEED)}
    by_cell = collections.defaultdict(list)
    for k in keys:
        if k in items:
            by_cell[k[:2]].append(k)
    jobs = [k for ks in by_cell.values() for k in (ks[:args.limit] if args.limit else ks)]
    outcomes = collections.defaultdict(list)
    with concurrent.futures.ThreadPoolExecutor(args.concurrency) as ex:
        futs = {ex.submit(compare_pair, prompt_text(items[k]), ra[k]["reply"], rb[k]["reply"]): k for k in jobs}
        for f in concurrent.futures.as_completed(futs):
            k = futs[f]
            try:
                outcomes[k[:2]].append(f.result())
            except Exception as exc:  # noqa: BLE001 — one bad verdict must not drop the cell silently
                print(f"warning: {k}: {exc}", file=sys.stderr)
    print(f"\n## pairwise {run_b.name} (B) vs {run_a.name} (A), judge {judge_model}\n")
    print("| task | lang | n | B wins | ties | B losses | B win rate | 95% CI | sign p |\n|---|---|---|---|---|---|---|---|---|")
    rows = []
    for (task, lang), os_ in sorted(outcomes.items()):
        s = summarize(os_)
        rows.append({"task": task, "lang": lang, "a": run_a.name, "b": run_b.name, "judge_model": judge_model, **s})
        print(f"| {task} | {lang} | {s['n']} | {s['b_wins']} | {s['ties']} | {s['b_losses']} | {s['b_win_rate']} | "
              f"{s['ci95'][0]}-{s['ci95'][1]} | {s['sign_p']} |")
    common.RESULTS.mkdir(parents=True, exist_ok=True)
    for row in rows:
        common.append_jsonl(common.RESULTS / "pairwise.jsonl", row)
    return rows


if __name__ == "__main__":
    main()
