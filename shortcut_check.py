"""Shortcut detector: the best accuracy a constant reply ("yes", "A", "die", "cannot be determined", ...) reaches per
generated task, scored offline. >= 0.6 is flagged; 0.5 on a trap/control-twin task is the expected ceiling of always
abstaining (the ok_trap / ok_control split shows it).

    uv run shortcut_check.py [task,task,...]
"""
import importlib
import sys

sys.path.insert(0, ".")
REPLIES = ["Answer: yes", "Answer: no", "Antwort: ja", "Antwort: nein", "Answer: A", "Answer: B", "Answer: C", "Answer: D",
           "Answer: 0", "Answer: 1", "Answer: cannot be determined", "Antwort: nicht bestimmbar", "Answer: ambiguous",
           "Antwort: mehrdeutig", "die", "das", "der", "is", "are", "ist", "sind", "dass", "das", "seit", "seid"]
tasks = sys.argv[1].split(",") if len(sys.argv) > 1 else [n for m in ("tasks_novel", "tasks_lang") for n in importlib.import_module(m).TASKS]
mods = [importlib.import_module(m) for m in ("tasks_novel", "tasks_lang")]
for name in tasks:
    t = next((m.TASKS[name] for m in mods if name in m.TASKS), None)
    if t is None or t.get("needs_judge"):
        print(f"{name:24s} skipped (judge task or not generated)")
        continue
    best = (0, None)
    for lang in t["langs"]:
        items = t["load"](lang, t["counts"]["full"][lang] or None, 0)
        for rep in REPLIES:
            sc = [t["score"](it, {"content": rep, "reasoning": "", "tool_calls": [], "finish_reason": "stop"})["correct"] for it in items]
            acc = sum(bool(x) for x in sc) / len(sc)
            if acc > best[0]:
                best = (acc, f"{lang}:{rep}")
    flag = "  <-- SHORTCUT?" if best[0] >= 0.6 else ""
    print(f"{name:24s} {best[0]:.2f} {best[1]}{flag}")
