"""Every generated task must give byte-identical items in every process: no set iteration order, no unseeded random.
Runs the loaders (8 items per task and language) under two PYTHONHASHSEED values and compares hashes."""
import json
import os
import subprocess
import sys

PROBE = r'''
import hashlib, importlib, json
import tasks_judge
tasks_judge.check_judge = lambda: None
h = {}
for m in ("tasks_open", "tasks_lang", "tasks_novel"):
    mod = importlib.import_module(m)
    for name, t in mod.TASKS.items():
        for lang in t["langs"]:
            try:
                it = t["load"](lang, 8, 0)
            except Exception as e:  # a loader that needs the network is compared by its error
                it = type(e).__name__
            h[f"{name}/{lang}"] = hashlib.sha1(json.dumps(it, sort_keys=True, default=str).encode()).hexdigest()
print(json.dumps(h))
'''


def run(seed):
    env = dict(os.environ, PYTHONHASHSEED=str(seed))
    out = subprocess.run([sys.executable, "-c", PROBE], capture_output=True, text=True, env=env, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_reproducible():
    a, b = run(1), run(2)
    assert a.keys() == b.keys()
    bad = [k for k in a if a[k] != b[k]]
    assert not bad, f"items depend on the process (set order / unseeded random): {bad}"


if __name__ == "__main__":
    test_reproducible()
    print("ok")
