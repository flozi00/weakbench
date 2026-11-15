"""Every generated task's full tier holds distinct prompts (a repeated prompt counts twice and skews the score). Plain
asserts:

    python3 test_distinct.py
"""
import json

import tasks_judge
import tasks_lang
import tasks_novel
import tasks_open


# repeats a task may still have (all burnt down); never add one
KNOWN = {}


def test_distinct_prompts():
    tasks_judge.configure("http://127.0.0.1:9", "mock", 32768)  # loaders only check that a judge is set
    bad = []
    try:
        for mod in (tasks_novel, tasks_lang, tasks_open):
            for name, t in mod.TASKS.items():
                for lang in t["langs"]:
                    items = t["load"](lang, t["counts"]["full"][lang], 0)
                    # a multi-turn item's prompt is its turns (messages holds only the first one), as in common.distinct
                    keys = [json.dumps(it.get("turns") or it["messages"], ensure_ascii=False) for it in items]
                    rep = len(keys) - len(set(keys))
                    if rep > KNOWN.get(f"{name}/{lang}", 0):
                        bad.append((name, lang, rep))
    finally:
        tasks_judge.configure(None, None, None)
    assert not bad, bad


if __name__ == "__main__":
    test_distinct_prompts()
    print("ok")
