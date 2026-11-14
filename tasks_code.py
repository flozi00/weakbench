"""Code tasks of the quality suite (quality.py imports TASKS). English only (code has no language); needs pyarrow.

Output prediction (nothing is run: the answer is compared with the recorded one):
cruxeval_o  CRUXEval-O: 800 short Python functions f and an input; predict f(input).
cruxeval_x  CRUXEval-X: the same task in 18 languages (Python, Rust, Java, C++, TypeScript, Go, Ruby, Shell, ...).
            The gold literal is recovered from the row itself: `output_reasoning` is a suffix of `code` with the
            expected value cut out, so the prompt is `code` with that value replaced by `????`. Scoring is
            language-agnostic (whitespace/quote/numeric-suffix-insensitive, else the ordered atoms must match), so
            the absolute score is approximate while the paired A/B is exact — same trade-off as chrF for wmt24pp.
lcb_output  LiveCodeBench test-output prediction: 442 (problem, test input) pairs from 182 contest problems
            (LeetCode, 2023-05 onward); predict what a correct solution returns.

Code generation (OPT-IN, name them in --tasks): the model writes the code and a local Python subprocess runs it
against the EvalPlus tests.
humaneval_plus  HumanEval+: 163 function-completion problems, ~80x the original tests.
mbpp_plus       MBPP+: 376 short problems with the EvalPlus tests.
The subprocess is no security boundary: a fresh interpreter in an empty temp dir with a minimal environment, a wall
clock timeout, and CPU / memory / file-size limits. It keeps the network, so run these where executing untrusted
code is acceptable (a sandbox container, not a machine with secrets). numpy is replaced by a stdlib stand-in
(the tests only use allclose, ndarray, float32/float64).
"""
import ast
import json
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import tempfile

from common import hf_file, read_rows

CRUX = ("cruxeval-org/cruxeval", "b96af0450242eb4da433032b90998f25588a5d0f")
CRUX_X = ("xhwl/cruxeval-x", "ecebf797cc32aa5d18855d4a75a80eda698486ef")
# The languages a served model is most likely to have seen, one per paradigm. Extending this tuple is the only
# change needed to cover more of CRUXEval-X's 18; the sample is balanced per language, so the count per language
# is quick // len(CRUX_X_LANGS).
CRUX_X_LANGS = ("Python", "Rust", "Java", "Cpp", "TypeScript", "Go", "Ruby", "Shell")
HUMANEVAL = ("evalplus/humanevalplus", "d32357cf319e50e9c8d8dab5ea876c72b0fd321b")
MBPP = ("evalplus/mbppplus", "b2d74c91837c3f2a20c1299ae98133cbe7cfa077")
# Items whose own canonical solution fails here (the oracle test in test_tasks_code.py): HumanEval/32 needs EvalPlus's
# special poly() oracle, MBPP+ 590 compares floats exactly, 596 compares sys.getsizeof (Python-version dependent).
UNUSABLE = {"HumanEval/32", "590", "596"}
LCB = ("livecodebench/test_generation", "6f3ac40bbecf81eba15899139d279b077f2816fd")
END = 'Reason step by step, then finish with a line "Output: X", where X is the exact result written as a literal.'
# The same cell in German: the code is language-agnostic, the instruction is not. A model that reasons
# about code in German must still get the same answer, so the two cells are item-paired (same ids).
CX_ASK = {"en": "What does this {plang} program return? The assertion compares the result with `????`.",
          "de": "Was gibt dieses {plang}-Programm zurück? Die Assertion vergleicht das Ergebnis mit `????`."}
CX_END = {"en": END,
          "de": 'Denke Schritt für Schritt und beende die Antwort mit einer Zeile "Ausgabe: X", wobei X das '
                'exakte Ergebnis als Literal ist.'}
# "**Output**: X" / "Ausgabe: X"; the first colon ends the label
OUTPUT = re.compile(r"(?im)^[^\w\n]*(?:output|ausgabe)[^\w:\n]*:[ \t]*(.+?)\s*$")


def pick(ids, n, seed):
    """quality.pick (not importable: quality imports this module): quick is a prefix of full."""
    perm = random.Random(seed).sample(sorted(ids), len(ids))
    return perm if n is None else perm[:n]


def value(text):
    """A literal as Python or JSON would read it; anything else stays text."""
    text = text.strip().strip("`").strip()
    for parse in (ast.literal_eval, json.loads):
        try:
            return parse(text)
        except Exception:  # noqa: BLE001 — SyntaxError, ValueError, RecursionError, MemoryError: not a literal
            pass
    return text


def score(item, reply):
    hits = OUTPUT.findall(reply["content"])
    ok = bool(hits) and value(hits[-1]) == value(item["gold"])
    return {"correct": ok, "score": float(ok), "pred": hits[-1] if hits else None, "gold": item["gold"], "detail": {}}


def item_of(id_, prompt, gold, lang="en"):
    return {"id": id_, "lang": lang, "messages": [{"role": "user", "content": prompt}], "gold": gold}


def load_cruxeval(lang, n, seed):
    rows = {r["id"]: r for r in read_rows(hf_file(*CRUX, "test.jsonl"))}
    return [item_of(i, f"What does this Python function return for the call f({rows[i]['input']})?\n\n"
                       f"```python\n{rows[i]['code']}\n```\n\n{END}", rows[i]["output"])
            for i in pick(rows, n, seed)]


def load_cruxeval_x(lang, n, seed):
    """CRUXEval-X output prediction, round-robin across CRUX_X_LANGS (id = "<Lang>/<row id>"), so the sample is
    balanced per language and quick stays a prefix of full."""
    per = None if n is None else max(1, n // len(CRUX_X_LANGS))
    pools, picked = [], []
    for plang in CRUX_X_LANGS:
        pool = {}
        for r in read_rows(hf_file(*CRUX_X, f"data/{plang}-00000-of-00001.parquet")):
            split = cx_split(r["code"], r["output_reasoning"])
            # No recoverable gold, or a literal with an embedded newline (unparseable from a free-text reply):
            # leave the row out of both arms alike.
            if split is None or "\n" in split[1] or not split[1].strip():
                continue
            masked, gold = split
            pool[f"{plang}/{r['id']}"] = item_of(
                f"{plang}/{r['id']}",
                f"{CX_ASK[lang].format(plang=plang)}\n\n"
                f"```{plang.lower()}\n{masked}\n```\n\n{CX_END[lang]}", gold, lang)
        pools.append(pool)
        picked.append(pick(pool, per, seed))
    items = []
    for k in range(max(len(p) for p in picked)):  # round-robin: one item per language per round
        items += [pool[p[k]] for pool, p in zip(pools, picked) if k < len(p)]
    return items


def cx_split(code, masked):
    """(code with the expected value replaced by '????', that value). `masked` is a suffix of `code` with the value
    cut out, so the value sits between the last occurrence of the prefix and the shared suffix. The mask is taken
    from the END of `masked`, because a program may itself contain '????' in a string literal."""
    n = 0
    while n < len(masked) and n < len(code) and masked[-1 - n] == code[-1 - n]:
        n += 1
    post = masked[len(masked) - n:]
    head = masked[:len(masked) - n]
    if not head.endswith("????"):
        return None
    pre, end = head[:-4], len(code) - len(post)
    idx = code.rfind(pre, 0, end)
    if idx < 0:
        return None
    return code[:idx + len(pre)] + "????" + code[end:], code[idx + len(pre):end]


# A literal in one of 18 languages: no parser exists for all of them, so compare two normalised forms.
CX_KEYWORD = re.compile(r"(?i)\b(none|nil|null|undefined|true|false)\b")
CX_SUFFIX = re.compile(r"(?<=\d)(?:[lLfFuU]|ll|LL|ul|UL|ull|ULL)\b")
CX_ATOM = re.compile(r'"(?:[^"\\]|\\.)*"|(?<![\w.])-?\d+(?:\.\d+)?')


def cx_key(text):
    """Whitespace-, quote-, numeric-suffix- and null/boolean-spelling-insensitive form of a literal."""
    t = re.sub(r"\s+", "", text).replace("'", '"').replace("`", '"')
    t = CX_KEYWORD.sub(lambda m: "null" if m.group(1).lower() in ("none", "nil", "null", "undefined") else
                       m.group(1).lower(), t)
    t = CX_SUFFIX.sub("", t)
    return t[1:-1] if len(t) > 1 and t[0] == t[-1] == '"' else t  # a bare string and a quoted one agree


def cx_atoms(text):
    """The ordered numbers and string contents of a literal: `vec![(4, 1)]` and `[(4, 1)]` agree."""
    return [m.group(0) if m.group(0).startswith('"') else float(m.group(0))
            for m in CX_ATOM.finditer(text.replace("'", '"'))]


def score_cruxeval_x(item, reply):
    hits = OUTPUT.findall(reply["content"])
    pred = hits[-1] if hits else None
    ok = bool(pred is not None and (cx_key(pred) == cx_key(item["gold"])
                                    or (cx_atoms(pred) and cx_atoms(pred) == cx_atoms(item["gold"]))))
    return {"correct": ok, "score": float(ok), "pred": pred, "gold": item["gold"], "detail": {}}


def load_lcb(lang, n, seed):
    items = {}
    for r in read_rows(hf_file(*LCB, "test.parquet")):
        for t in json.loads(r["test"]):
            items[f"{r['question_id']}#{r['test_id']}"] = item_of(
                f"{r['question_id']}#{r['test_id']}",
                f"{r['question_content']}\n\nA correct solution has this signature:\n```python\n{r['starter_code']}\n```\n\n"
                f"What does a correct solution return for the input below (one argument per line)? Write the "
                f"result as JSON.\n\n{t['input']}\n\n{END}", t["output"])
    return [items[i] for i in pick(items, n, seed)]


# ---------------------------------------------------------------- code generation (runs the model's code)
WALL_S, CPU_S, MEM_BYTES, FILE_BYTES = 30, 30, 4 << 30, 16 << 20
PRELUDE = f"""import resource, sys
for _r, _v in ((resource.RLIMIT_CPU, {CPU_S}), (resource.RLIMIT_AS, {MEM_BYTES}), (resource.RLIMIT_FSIZE, {FILE_BYTES}),
               (resource.RLIMIT_CORE, 0)):
    try:
        resource.setrlimit(_r, (_v, _v))
    except (ValueError, OSError):
        pass  # macOS refuses some limits; the wall clock still holds
sys.setrecursionlimit(10000)
try:
    import numpy
except ImportError:
    import types
    numpy = sys.modules["numpy"] = types.ModuleType("numpy")
    numpy.ndarray, numpy.float64, numpy.float32 = type("ndarray", (), {{}}), "float64", "float32"
    def _close(a, b, rtol, atol):
        if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
            return (isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)) and len(a) == len(b)
                    and all(_close(x, y, rtol, atol) for x, y in zip(a, b)))
        return abs(a - b) <= atol + rtol * abs(b)
    numpy.allclose = lambda a, b, rtol=1e-05, atol=1e-08: _close(a, b, rtol, atol)
"""
FENCE = re.compile(r"```[ \t]*(?:python|py|python3)?[ \t]*\n(.*?)```", re.S | re.I)


def code_of(text):
    """The last fenced code block of the reply, else the reply itself (a bare function)."""
    blocks = FENCE.findall(text)
    return (blocks[-1] if blocks else text).strip()


def run_python(source):
    """(passed, error line): `source` in a fresh interpreter, empty cwd, minimal env, own process group."""
    tmp = tempfile.mkdtemp(prefix="qcode-")
    try:
        with open(os.path.join(tmp, "main.py"), "w", encoding="utf-8") as fh:
            fh.write(PRELUDE + "\n" + source)
        proc = subprocess.Popen([sys.executable, "-I", "main.py"], cwd=tmp, stdin=subprocess.DEVNULL,
                                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, start_new_session=True,
                                env={"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": "0", "HOME": tmp})
        try:
            _, err = proc.communicate(timeout=WALL_S)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.communicate()
            return False, f"timeout after {WALL_S}s"
        lines = err.decode("utf-8", "replace").strip().splitlines()
        return proc.returncode == 0, "" if proc.returncode == 0 else (lines[-1] if lines else f"exit {proc.returncode}")[:300]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def score_exec(item, reply):
    ok, err = run_python(item["build"](code_of(reply["content"])))
    return {"correct": ok, "score": float(ok), "pred": None, "gold": None, "detail": {"error": err}}


def header(prompt):
    """The import lines of a HumanEval prompt: the model's block may lack the names its annotations use."""
    return "\n".join(ln for ln in prompt.splitlines() if re.match(r"(import|from)\s", ln))


def humaneval_source(row, code):
    body = code if re.search(rf"def\s+{re.escape(row['entry_point'])}\b", code) else row["prompt"] + code
    return f"{header(row['prompt'])}\n{body}\n\n{row['test']}\n\ncheck({row['entry_point']})\n"


def load_humaneval(lang, n, seed):
    rows = {r["task_id"]: r for r in read_rows(hf_file(*HUMANEVAL, "data/test-00000-of-00001-5973903632b82d40.parquet"))
            if r["task_id"] not in UNUSABLE}
    return [{"id": i, "lang": "en", "messages": [{"role": "user", "content": (
                 "Complete the following Python function. Reply with the complete function (with any imports it "
                 f"needs) in one ```python code block.\n\n```python\n{rows[i]['prompt']}```")}],
             "build": lambda code, r=rows[i]: humaneval_source(r, code)} for i in pick(rows, n, seed)]


def load_mbpp(lang, n, seed):
    rows = {str(r["task_id"]): r for r in read_rows(hf_file(*MBPP, "data/test-00000-of-00001-d5781c9c51e02795.parquet"))
            if str(r["task_id"]) not in UNUSABLE}
    return [{"id": i, "lang": "en", "messages": [{"role": "user", "content": (
                 f"{rows[i]['prompt']}\nYour code should pass this test:\n{rows[i]['test_list'][0]}\n\n"
                 "Reply with the complete Python code (functions and any imports) in one ```python code block.")}],
             "build": lambda code, r=rows[i]: "\n".join(r["test_imports"]) + f"\n{code}\n\n{r['test']}\n"}
            for i in pick(rows, n, seed)]


def task(load, quick, score=score, opt_in=False, full=None, langs=("en",)):
    return {"langs": langs, "load": load, "score": score, "think": True, "max_tokens": 16384,
            "counts": {"quick": dict.fromkeys(langs, quick), "full": dict.fromkeys(langs, full)},
            "lang_check": False, "opt_in": opt_in}


TASKS = {"cruxeval_o": task(load_cruxeval, 150), "lcb_output": task(load_lcb, 100),
         "cruxeval_x": task(load_cruxeval_x, 160, score_cruxeval_x, full=400),
         # the same items behind a German instruction: item-paired with cruxeval_x, so a DE-vs-EN gap is
         # a language effect and not a sample effect
         "cruxeval_x_de": task(load_cruxeval_x, 160, score_cruxeval_x, full=400, langs=("de",)),
         "humaneval_plus": task(load_humaneval, 163, score_exec, True), "mbpp_plus": task(load_mbpp, 100, score_exec, True)}
