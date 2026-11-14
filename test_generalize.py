"""Tests for generalize.py (stdlib): python3 test_generalize.py, or pytest."""
import atexit
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import generalize  # noqa: E402
from common import append_jsonl  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="generalize-test-"))
atexit.register(shutil.rmtree, TMP, True)


def make_run(name, correct_fn, tasks=("gsm8k", "bfcl"), n=200, **manifest):
    """A quality run dir. correct_fn(task, i) -> bool decides item i's correctness."""
    path = TMP / name
    path.mkdir()
    (path / "manifest.json").write_text(json.dumps({
        "run_id": name, "kind": "quality", "arm": name, "url": "http://mock",
        "server": {"model": "mock-model", "max_model_len": 32768},
        "bench_code_sha": "deadbeef", "args": {"tier": "quick"}, **manifest}))
    for task in tasks:
        for i in range(n):
            append_jsonl(path / f"items-{task}.jsonl",
                         {"task": task, "lang": "en", "id": str(i), "error": None,
                          "correct": correct_fn(task, i)})
    return path


def run(base, tuned, **kw):
    """(verdict, exit code, text) of a report over two run dirs."""
    b, t = generalize.load_runs([str(base)])[0], generalize.load_runs([str(tuned)])[0]
    diffs = generalize.protocol_diff(b, t)
    if diffs:
        return "INCONCLUSIVE", generalize.EXIT["INCONCLUSIVE"], "\n".join(diffs)
    text, verdict, _ = generalize.report(b, t, kw.get("in_dataset", generalize.IN_DATASET),
                                         kw.get("alpha", 0.05))
    return verdict, generalize.EXIT[verdict], text


def test_forgetting():
    """An OOD task collapses while the trained task improves: FORGETTING, exit 1."""
    base = make_run("f-base", lambda t, i: (i % 10) != 0 if t == "gsm8k" else (i % 5) != 0)
    tuned = make_run("f-tuned", lambda t, i: (i % 20) >= 9 if t == "gsm8k" else (i % 20) != 0)
    verdict, code, text = run(base, tuned)
    assert verdict == "FORGETTING", verdict
    assert code == 1, code
    assert "gsm8k/en" in text and "REGRESSED" in text
    assert "bfcl" in text and "IMPROVED" in text


def test_composite_index_can_rise_while_forgetting():
    """The trap this report exists to catch: the mean goes UP and the model still forgot.

    Needs more than one OOD task: with a single one, the forgetting task's loss outweighs the gain and the index
    falls, which is the easy case. Three OOD tasks at 80 -> 95 against gsm8k at 90 -> 55 give 82.5 -> 85.0.
    """
    tasks = ("gsm8k", "bfcl", "ifeval", "musr")
    base = make_run("c-base", lambda t, i: (i % 10) != 0 if t == "gsm8k" else (i % 5) != 0, tasks=tasks)
    tuned = make_run("c-tuned", lambda t, i: (i % 20) >= 9 if t == "gsm8k" else (i % 20) != 0, tasks=tasks)
    verdict, _, text = run(base, tuned)
    assert verdict == "FORGETTING"
    idx = [ln for ln in text.splitlines() if ln.startswith("| mean task accuracy")][0]
    assert idx.split("|")[4].strip().startswith("+"), f"expected a rising index, got {idx}"


def test_generalized():
    """An OOD task improves and nothing regresses: GENERALIZED, exit 0."""
    base = make_run("g-base", lambda t, i: (i % 10) != 0)
    tuned = make_run("g-tuned", lambda t, i: (i % 20) != 0)
    verdict, code, text = run(base, tuned)
    assert verdict == "GENERALIZED", verdict
    assert code == 0, code
    assert "IMPROVED" in text


def test_neutral():
    """Identical runs: nothing moves, NEUTRAL, exit 0."""
    fn = lambda t, i: (i % 10) != 0  # noqa: E731
    verdict, code, _ = run(make_run("n-base", fn), make_run("n-tuned", fn))
    assert verdict == "NEUTRAL", verdict
    assert code == 0, code


def test_in_dataset_improvement_is_not_generalization():
    """Only the trained task improves: that is not cross-learning, so NEUTRAL."""
    base = make_run("i-base", lambda t, i: (i % 5) != 0 if t == "bfcl" else True)
    tuned = make_run("i-tuned", lambda t, i: True)
    verdict, code, _ = run(base, tuned)
    assert verdict == "NEUTRAL", verdict
    assert code == 0, code


def test_protocol_mismatch_is_inconclusive():
    """Different suite code: the pairing is meaningless, so refuse (exit 3)."""
    fn = lambda t, i: (i % 10) != 0  # noqa: E731
    base = make_run("p-base", fn)
    tuned = make_run("p-tuned", fn, bench_code_sha="cafebabe")
    verdict, code, text = run(base, tuned)
    assert verdict == "INCONCLUSIVE", verdict
    assert code == 3, code
    assert "suite code differs" in text


def test_watch_list_holds_near_misses():
    """6 discordant pairs of 200 is raw p=0.031: under alpha, but Holm over 2 cells kills it.
    The cell must stay UNCHANGED and appear on the watch list, not silently vanish."""
    base = make_run("w-base", lambda t, i: True)
    tuned = make_run("w-tuned", lambda t, i: i >= 6)
    verdict, code, text = run(base, tuned)
    assert verdict == "NEUTRAL", verdict
    assert "## Watch list" in text, text
    assert "UNCHANGED" in text


def test_missing_run_is_a_data_error():
    """A run dir that does not exist: exit 2, not a traceback."""
    r = subprocess.run([sys.executable, str(HERE / "generalize.py"),
                        "--base", "no-such-run", "--tuned", "no-such-run-2"],
                       capture_output=True, text=True)
    assert r.returncode == 2, (r.returncode, r.stderr)
    assert "not a run directory" in r.stderr


def test_exit_codes_subprocess():
    """The CLI's exit codes are the contract: 0 neutral, 1 forgetting, 3 inconclusive."""
    fn = lambda t, i: (i % 10) != 0  # noqa: E731
    neutral = subprocess.run([sys.executable, str(HERE / "generalize.py"),
                              "--base", str(make_run("e-base", fn)),
                              "--tuned", str(make_run("e-tuned", fn))],
                             capture_output=True, text=True)
    assert neutral.returncode == 0, neutral.stderr
    assert neutral.stdout.rstrip().splitlines()[-1].startswith("VERDICT: NEUTRAL")

    forget = subprocess.run([sys.executable, str(HERE / "generalize.py"),
                             "--base", str(make_run("e2-base", lambda t, i: True)),
                             "--tuned", str(make_run("e2-tuned", lambda t, i: i >= 60))],
                            capture_output=True, text=True)
    assert forget.returncode == 1, forget.stderr
    assert forget.stdout.rstrip().splitlines()[-1].startswith("VERDICT: FORGETTING")


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        fn()
        print(f"ok {name}")
    print(f"{len(tests)} tests passed ({TMP})")
