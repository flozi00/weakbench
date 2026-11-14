#!/usr/bin/env python3
"""Generalization / forgetting report: what a fine-tune did to tasks it never saw.

    python3 generalize.py --base RUN --tuned RUN [--in-dataset bfcl,ifstruct]
                                     [--alpha 0.05] [--json OUT]

RUN = a quality.py run directory (or its id under runs/).

This is NOT the A/B gate (compare.py answers "did the candidate build regress"). The
question here is "did training on task X make the model better or worse at tasks Y it
never saw". Every (task, lang) cell is paired item by item between the two runs:

  binary cells   exact McNemar on the discordant pairs (the test compare.py uses)
  score cells    paired mean difference, normal-approximation p

Holm-corrected across all cells. A cell is IMPROVED or REGRESSED only when the
corrected p < alpha; otherwise UNCHANGED.

Cells are split by whether the task is in the fine-tune's own distribution
(--in-dataset, default bfcl,ifstruct: the SFT corpus is tool-call traces). The
out-of-distribution block is the one that matters: a significant regression there is
the failure mode this report exists to catch, and a significant improvement there is
cross-learning.

The composite index is the unweighted mean of the per-task accuracies (0-100) over the
tasks both runs measured — the same shape as Artificial Analysis's Intelligence Index,
so it moves only when overall competence moves rather than when one task trades
against another.

Refuses to compare runs that did not use the same suite code or protocol: a different
bench_code_sha, thinking_off_honored or drift_anchor makes the pairing meaningless.

Prints Markdown; the last line is
`VERDICT: GENERALIZED|FORGETTING|NEUTRAL|INCONCLUSIVE - ...`.
Exit 0 GENERALIZED/NEUTRAL, 1 FORGETTING, 3 INCONCLUSIVE, 2 usage/data error.
"""
import argparse
import json
import sys
import traceback
from collections import defaultdict

from compare import (DataError, flips, fmt, holm, items, load_runs, mcnemar, mean100,
                     paired_scores, pair, run_info, table, usable, value)

EXIT = {"GENERALIZED": 0, "NEUTRAL": 0, "FORGETTING": 1, "INCONCLUSIVE": 3}
# The SFT corpus is tool-call traces, so these are the only cells the model was trained on.
IN_DATASET = ("bfcl", "ifstruct")


def protocol_diff(base, tuned):
    """Reasons the two runs cannot be paired. Empty list = comparable."""
    mb, mt = base[1], tuned[1]
    out = []
    for key in ("bench_code_sha", "bench_git_sha"):
        vb, vt = mb.get(key), mt.get(key)
        if vb and vt and vb != vt:
            out.append(f"suite code differs ({key} {vb[:12]} vs {vt[:12]})")
            break
    ab, at = mb.get("args") or {}, mt.get("args") or {}
    for key in ("tier", "tasks", "langs", "limit", "think", "temperature", "max_tokens"):
        if key in ab and key in at and ab[key] != at[key]:
            out.append(f"run arg {key} differs ({ab[key]!r} vs {at[key]!r})")
    ib, it = run_info(base[0]), run_info(tuned[0])
    if ib.get("thinking_off_honored") != it.get("thinking_off_honored"):
        out.append(f"thinking_off_honored differs "
                   f"({ib.get('thinking_off_honored')} vs {it.get('thinking_off_honored')})")
    if ib.get("drift_anchor") != it.get("drift_anchor"):
        out.append(f"drift_anchor differs ({ib.get('drift_anchor')} vs {it.get('drift_anchor')})")
    return out


def judge(ps, alpha):
    """(base, tuned, delta, p, kind) for one cell. delta is tuned - base, in points."""
    xs = [x for x, _ in ps]
    ys = [y for _, y in ps]
    base, tuned = mean100(xs), mean100(ys)
    binary = all(x.get("correct") is not None and y.get("correct") is not None for x, y in ps)
    if binary:
        b, c = flips(ps)
        n = len(ps)
        delta = 100.0 * (c - b) / n if n else 0.0
        p = mcnemar(b, c)
        kind = "mcnemar"
    else:
        diffs = [value(y) - value(x) for x, y in ps]
        d, _, _, p = paired_scores(diffs)
        delta = 100.0 * d
        kind = "paired"
    return base, tuned, delta, p, kind


def classify(delta, p, alpha):
    if p >= alpha:
        return "UNCHANGED"
    return "IMPROVED" if delta > 0 else "REGRESSED"


def report(base, tuned, in_dataset, alpha):
    ib, it = items(base[0]), items(tuned[0])
    cells = pair(ib, it)
    if not cells:
        raise DataError("no (task, lang) cell has usable items on both sides")

    names = sorted(cells)
    judged = {k: judge(cells[k], alpha) for k in names}
    adj = dict(zip(names, holm([judged[k][3] for k in names])))

    rows, verdicts, watch = [], {}, []
    for k in names:
        task, lang = k
        b, t, delta, p, kind = judged[k]
        verdict = classify(delta, adj[k], alpha)
        verdicts[k] = verdict
        if verdict == "UNCHANGED" and p < alpha:
            watch.append((k, delta, p))
        rows.append([task, lang, len(cells[k]), fmt(b), fmt(t), f"{delta:+.1f}",
                     f"{p:.4f}", f"{adj[k]:.4f}", verdict, "in" if task in in_dataset else "OOD"])

    def block(pred):
        return [r for r in rows if pred(r[0])]

    ind, ood = block(lambda t: t in in_dataset), block(lambda t: t not in in_dataset)

    # Composite index: mean of per-task accuracy over tasks both runs measured. A task
    # with several languages counts once (its languages are averaged first), so a
    # 5-language task cannot outvote a 1-language one.
    # judge() returns (base, tuned, delta, p, kind), so index 0 is the base arm and
    # index 1 the tuned one. Swapping them flips the sign of the headline delta.
    per_task = defaultdict(list)
    for k in names:
        per_task[k[0]].append(judged[k][0])
    idx_base = sum(sum(v) / len(v) for v in per_task.values()) / len(per_task)
    per_task_b = defaultdict(list)
    for k in names:
        per_task_b[k[0]].append(judged[k][1])
    idx_tuned = sum(sum(v) / len(v) for v in per_task_b.values()) / len(per_task_b)

    ood_reg = [k for k in names if k[0] not in in_dataset and verdicts[k] == "REGRESSED"]
    ood_imp = [k for k in names if k[0] not in in_dataset and verdicts[k] == "IMPROVED"]
    ind_reg = [k for k in names if k[0] in in_dataset and verdicts[k] == "REGRESSED"]
    ind_imp = [k for k in names if k[0] in in_dataset and verdicts[k] == "IMPROVED"]

    if ood_reg:
        verdict = "FORGETTING"
        why = (f"{len(ood_reg)} out-of-distribution cell(s) regressed: "
               + ", ".join(f"{t}/{l}" for t, l in ood_reg[:6]))
    elif ood_imp:
        verdict = "GENERALIZED"
        why = (f"{len(ood_imp)} out-of-distribution cell(s) improved, none regressed: "
               + ", ".join(f"{t}/{l}" for t, l in ood_imp[:6]))
    else:
        verdict = "NEUTRAL"
        why = "no out-of-distribution cell moved beyond the noise floor"

    head = ["task", "lang", "n", "base", "tuned", "delta", "p_raw", "p_holm", "verdict", "set"]
    out = [f"# Generalization report\n",
           table(["", "run", "arm", "model"],
                 [["base", base[1].get("run_id") or base[0].name, base[1].get("arm"),
                   (base[1].get("server") or {}).get("model")],
                  ["tuned", tuned[1].get("run_id") or tuned[0].name, tuned[1].get("arm"),
                   (tuned[1].get("server") or {}).get("model")]]),
           f"\nIn-dataset tasks: {', '.join(in_dataset)} (the SFT corpus is tool-call traces).\n"
           f"Everything else is out-of-distribution. alpha={alpha}, Holm-corrected over {len(names)} cells.\n"]
    if ind:
        out.append("\n## In-dataset (trained on)\n\n" + table(head, ind))
    if ood:
        out.append("\n## Out-of-distribution (never trained on)\n\n" + table(head, ood))
    out.append(f"\n## Composite index\n\n"
               f"| | base | tuned | delta |\n|---|---|---|---|\n"
               f"| mean task accuracy | {idx_base:.2f} | {idx_tuned:.2f} | {idx_tuned - idx_base:+.2f} |\n")
    if watch:
        out.append("\n## Watch list\n\n"
                   f"Cells that moved at raw p < {alpha} but did not survive Holm over "
                   f"{len(names)} cells. Not evidence on their own: re-run with more items "
                   "before acting on one.\n\n"
                   + table(["task", "lang", "delta", "p_raw"],
                           [[t, l, f"{d:+.1f}", f"{p:.4f}"] for (t, l), d, p in watch]))
    out.append(f"\nIn-dataset: {len(ind_imp)} improved, {len(ind_reg)} regressed. "
               f"Out-of-distribution: {len(ood_imp)} improved, {len(ood_reg)} regressed.\n")
    out.append(f"\nVERDICT: {verdict} - {why}")
    return "\n".join(out), verdict, rows


def main(argv=None):
    ap = argparse.ArgumentParser(description="Generalization / forgetting report for a fine-tune")
    ap.add_argument("--base", required=True, help="quality run of the untrained model")
    ap.add_argument("--tuned", required=True, help="quality run of the fine-tuned model")
    ap.add_argument("--in-dataset", default=",".join(IN_DATASET),
                    help="comma-separated tasks the fine-tune trained on")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--json", help="write the per-cell rows here")
    args = ap.parse_args(argv)

    try:
        base = load_runs([args.base])[0]
        tuned = load_runs([args.tuned])[0]
        diffs = protocol_diff(base, tuned)
        if diffs:
            print("# Generalization report\n\nINCONCLUSIVE: the runs are not comparable.\n")
            for d in diffs:
                print(f"- {d}")
            print("\nVERDICT: INCONCLUSIVE - protocol mismatch")
            return EXIT["INCONCLUSIVE"]
        in_dataset = tuple(t.strip() for t in args.in_dataset.split(",") if t.strip())
        text, verdict, rows = report(base, tuned, in_dataset, args.alpha)
        print(text)
        if args.json:
            with open(args.json, "w") as fh:
                json.dump({"verdict": verdict, "rows": rows}, fh, indent=2)
        return EXIT[verdict]
    except DataError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception:
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())
