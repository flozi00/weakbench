"""Where is this model weak? Reads a run's summary.json and ranks the gaps the trap tasks were built to expose.

    uv run weakness_report.py runs/<run>            # one model
    uv run weakness_report.py runs/<A> runs/<B>     # two models side by side

Pairs: control vs trap twins (ok_control/ok_trap, ok_original/ok_mutated, ok_forward/ok_backward, ok_agree/ok_conflict,
ok_base/ok_noop, ok_unique/ok_ambiguous, ...), bait rates (took_*, guessed, chose_longer, naive_sum, ...), and
difficulty cliffs (ok_d2..8, ok_b3..6, ok_ops1..3, ok_attr1..2, ok_neg0..3): the first level that drops below half the
easiest level's accuracy. Consistency cells (group_agree, xl_correct_agree, xt_correct_agree) are listed lowest first.
"""
import json
import pathlib
import re
import sys

PAIRS = [("ok_control", "ok_trap"), ("ok_control", "ok_lookalike"), ("ok_control", "ok_revoked"),
         ("ok_control", "ok_anchored"), ("ok_original", "ok_mutated"), ("ok_forward", "ok_backward"),
         ("ok_agree", "ok_conflict"), ("ok_base", "ok_noop"), ("ok_unique", "ok_ambiguous"), ("ok_sincere", "ok_ironic"),
         ("ok_event", "ok_context"), ("ok_answerable", "ok_missing"), ("ok_fluff_none", "ok_fluff_wrong")]
BAIT = re.compile(r"^(took_|guessed$|chose_longer$|naive_sum$|off_by_one$|leaked_code$|machine_tell$|ai_word$|"
                  r"false_claim$|mixed$|forbidden_used$|words_changed$|illegal$)")
LEVEL = re.compile(r"^ok_((?:[a-z]+_)?(?:d|b|ops|attr|neg|s|n|h|m|e|len|add|mul))(\d+)$|^ok_(\d+)(parts)$")
CONSIST = ("group_agree", "group_all_of_any", "xl_correct_agree", "xl_both_of_any", "xt_correct_agree")


def report(rows):
    gaps, baits, cliffs, cons = [], [], [], []
    for r in rows:
        cell, ex = f"{r['task']}/{r['lang']}", r.get("extra") or {}
        for a, b in PAIRS:
            if a in ex and b in ex and ex[a] is not None and ex[b] is not None:
                gaps.append((ex[a] - ex[b], cell, f"{a} {ex[a]:.2f} -> {b} {ex[b]:.2f}"))
        for k, v in ex.items():
            if BAIT.match(k) and v:
                baits.append((v, cell, k))
        levels = {}
        for k, v in ex.items():
            m = LEVEL.match(k)
            if m and v is not None:
                kind, lvl = (m[1], m[2]) if m[1] else (m[4], m[3])
                levels.setdefault(kind, []).append((int(lvl), v))
        for kind, lv in levels.items():
            lv.sort()
            best = max(range(len(lv)), key=lambda j: lv[j][1])
            top = lv[best][1]
            drop = next((n for n, v in lv[best + 1:] if top and v < top / 2), None)  # only past the best level
            if drop is not None:
                cliffs.append((cell, kind, drop, " ".join(f"{n}:{v:.2f}" for n, v in lv)))
        for k in CONSIST:
            if ex.get(k) is not None:
                cons.append((ex[k], cell, k))
    return sorted(gaps, reverse=True), sorted(baits, reverse=True), cliffs, sorted(cons)


def main():
    runs = [pathlib.Path(a) for a in sys.argv[1:]]
    if not runs:
        sys.exit(__doc__)
    for run in runs:
        rows = json.loads((run / "summary.json").read_text())
        gaps, baits, cliffs, cons = report(rows)
        print(f"== {run.name}")
        print("Trap gaps (control - trap), largest first:")
        for g, cell, txt in gaps[:15]:
            print(f"  {g:+.2f}  {cell:28s} {txt}")
        print("Bait taken (share of items):")
        for v, cell, k in baits[:15]:
            print(f"  {v:.2f}  {cell:28s} {k}")
        print("Difficulty cliffs (first level below half the best):")
        for cell, kind, drop, lv in cliffs:
            print(f"  {cell:28s} {kind} breaks at {drop}   [{lv}]")
        print("Consistency, lowest first:")
        for v, cell, k in cons[:10]:
            print(f"  {v:.2f}  {cell:28s} {k}")


if __name__ == "__main__":
    main()
