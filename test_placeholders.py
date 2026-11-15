"""No prompt carries an unfilled template placeholder like "{who}" (a missing f-string prefix). Plain asserts:

    python3 test_placeholders.py
"""
import re

import tasks_judge
import tasks_lang
import tasks_novel
import tasks_open

LEGIT = {"i18n_strings"}  # the task is about keeping {name} placeholders


def test_no_placeholders():
    tasks_judge.configure("http://127.0.0.1:9", "mock", 32768)  # loaders only check that a judge is set
    try:
        for mod in (tasks_novel, tasks_lang, tasks_open):
            for name, t in mod.TASKS.items():
                if name in LEGIT:
                    continue
                for lang in t["langs"]:
                    for it in t["load"](lang, None, 0):
                        for m in it["messages"]:
                            assert not re.search(r"\{[a-z_][a-z_0-9]*\}", m["content"] or ""), (name, lang, it["id"])
    finally:
        tasks_judge.configure(None, None, None)


if __name__ == "__main__":
    test_no_placeholders()
    print("ok")
