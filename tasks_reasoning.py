"""Hard-reasoning tasks of the quality suite (quality.py imports TASKS). English only; needs pyarrow.

aime25  AIME 2025: 30 competition problems with integer answers 0-999. Artificial Analysis
        includes AIME 2025 in its Intelligence Index, so this is the in-suite analogue of that
        suite's hardest reasoning component. It is deliberately hard: a 27B model scores in the
        single digits to low teens, which makes it a sensitive probe for reasoning damage that
        easier math (gsm8k, mgsm) hides behind a ceiling.
musr    MuSR (Sprague et al., ICLR 2024): 756 long narratives with a multiple-choice question that
        needs multi-step soft reasoning — murder mystery (deduction over motives and alibis),
        object placements (spatial planning) and team allocation (constraint satisfaction). The
        answer is the option's text, not a letter, so no letter extractor is involved; a bare
        letter is accepted too. This is the suite's planning / multi-step cell.
math_500  MATH-500 (Hendrycks MATH test split, OpenAI's 500-problem subset): 7 subjects, 5 levels,
        LaTeX answers. Artificial Analysis reports it as a standalone eval. It fills the band
        between gsm8k (saturated) and aime25 (at ceiling on the 27B class), which is where a
        reasoning regression actually shows up. Scored by normalised LaTeX equality with a
        numeric fallback, so the absolute rate is a lower bound; the paired A/B is exact.

Nothing in this file appears in the SFT corpus, so it is a clean out-of-distribution probe:
a drop here is forgetting, a rise is cross-learning.
"""
import ast
import random
import re

from common import hf_file, read_jsonl, read_rows

AIME25 = ("MathArena/aime_2025", "c94da77eb22bbd6439e62a323bec18493a421302")
AIME_FILE = "data/train-00000-of-00001.parquet"
MATH = "Solve step by step, then give the final answer after '####'."
NUM = re.compile(r"-?\d+(?:[.,  ]\d+)*")


def pick(ids, n, seed):
    """quality.pick (not importable: quality imports this module): quick is a prefix of full."""
    perm = random.Random(seed).sample(sorted(ids), len(ids))
    return perm if n is None else perm[:n]


def user(content):
    return [{"role": "user", "content": content}]


def load_aime25(lang, n, seed):
    """30 problems, ids = problem_idx. `topic` is kept for the report, not for scoring."""
    rows = read_rows(hf_file(*AIME25, AIME_FILE))
    items = [{"id": f"{int(r['problem_idx']):02d}", "lang": lang,
              "messages": user(f"{r['problem']}\n\n{MATH}"),
              "gold": int(r["answer"]),
              "topic": (r.get("problem_type") or [""])[0]}
             for r in rows]
    return [items[i] for i in pick(range(len(items)), n, seed)]


def score_aime25(item, reply):
    """Same '####' convention as score_math: the number right after the last '####' on its
    line, else the last number of the reply. AIME answers are integers 0-999, so a float or
    a fraction is wrong by construction rather than by rounding."""
    c = reply["content"]
    m = "####" in c and re.match(rf"[^\w\n-]*({NUM.pattern})", c.rpartition("####")[2])
    tok = m[1] if m else (NUM.findall(c) or [None])[-1]
    v = None
    if tok:
        try:
            v = int(re.sub(r"[.,  \u00a0\u202f]", "", tok))
        except ValueError:
            v = None
    ok = v is not None and v == item["gold"]
    return {"correct": ok, "score": float(ok), "pred": tok, "gold": str(item["gold"]), "detail": {}}


MUSR = ("TAUR-Lab/MuSR", "7c365b439a222150f317764d4f16ae6c96d7d94a")
MUSR_FILE = "all.csv"
MUSR_PROMPT = ("Read the narrative and answer the question. Reply with the exact text of one of the options.\n\n"
               "Narrative:\n{narrative}\n\nQuestion: {question}\n\nOptions:\n{options}\n\n"
               "Think step by step, then finish with a line \"Answer: <option>\", where <option> is the exact text "
               "of the correct option.")


def load_musr(lang, n, seed):
    """756 narratives; id = row index. `choices` is a Python list literal in the CSV."""
    rows = read_rows(hf_file(*MUSR, MUSR_FILE))
    items = []
    for i in pick(range(len(rows)), n, seed):
        r = rows[i]
        choices = ast.literal_eval(r["choices"])
        items.append({"id": f"{i:03d}", "lang": lang,
                      "messages": user(MUSR_PROMPT.format(
                          narrative=r["narrative"].strip(), question=r["question"].strip(),
                          options="\n".join(f"{chr(65 + k)}. {c}" for k, c in enumerate(choices)))),
                      "gold": choices[int(r["answer_index"])], "choices": choices})
    return items


def norm(text):
    """Lowercase, punctuation to single spaces: 'Mackenzie.' and 'mackenzie' agree."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", text.lower())).strip()


def score_musr(item, reply):
    """The option text after the last 'Answer:' (else the reply's tail), or a bare letter mapped to its option."""
    c = reply["content"]
    tail = norm(c.rpartition("Answer:")[2] if "Answer:" in c else c[-300:])
    gold = norm(item["gold"])
    ok = bool(gold) and gold in tail
    if not ok and re.fullmatch(r"[a-e]", tail):
        ok = item["choices"]["abcde".index(tail)] == item["gold"]
    return {"correct": ok, "score": float(ok), "pred": tail[:120], "gold": item["gold"], "detail": {}}


MATH500 = ("HuggingFaceH4/MATH-500", "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be")
MATH500_FILE = "test.jsonl"
MATH500_PROMPT = ("Solve the problem step by step. Then write the final answer on its own line as "
                  "'#### <answer>', in LaTeX when it is not a plain number (e.g. '#### \\frac{3}{4}').")
# LaTeX noise that never changes the value: spacing macros, sizing, markdown emphasis.
TEX_DROP = ("\\left", "\\right", "\\displaystyle", "\\!", "\\,", "\\;", "\\:", "\\ ", "\\quad", "\\qquad",
            "~", "$", "*", "`")


def tex(s):
    """A LaTeX answer reduced to a comparable form: no spacing macros, no sizing, no whitespace, no trailing dot."""
    s = s.strip()
    for d in TEX_DROP:
        s = s.replace(d, "")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = re.sub(r"\\text\{([^{}]*)\}", r"\1", s)
    return re.sub(r"\s+", "", s).rstrip(".")


def value(s):
    """The number a plain or \\frac answer denotes, else None (so 0.75, 3/4 and \\frac{3}{4} agree)."""
    t = s.replace(",", "").replace("\\%", "").replace("%", "").strip()
    t = t.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    m = re.fullmatch(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", t)
    if m:
        a, b = value(m[1]), value(m[2])
        return a / b if a is not None and b else None
    m = re.fullmatch(r"(-?[\d.]+)/(-?[\d.]+)", t)
    if m:
        return float(m[1]) / float(m[2]) if float(m[2]) else None
    try:
        return float(t)
    except ValueError:
        return None


def load_math500(lang, n, seed):
    """500 competition problems (7 subjects, 5 levels). `answer` is LaTeX; subject/level are kept for the report."""
    rows = read_jsonl(hf_file(*MATH500, MATH500_FILE))
    items = [{"id": f"{i:03d}", "lang": lang, "messages": user(f"{r['problem']}\n\n{MATH500_PROMPT}"),
              "gold": r["answer"], "subject": r.get("subject"), "level": r.get("level")}
             for i, r in enumerate(rows)]
    return [items[i] for i in pick(range(len(items)), n, seed)]


def score_math500(item, reply):
    """The answer after the last '####' (else the last \\boxed{...}, else the reply's tail), compared as LaTeX.

    Two tiers, like cruxeval_x: normalised LaTeX equality, else numeric equality with a 1e-6 relative tolerance.
    A model that writes an equivalent answer in a different notation ('(3, pi/2)' for '\\left(3, \\frac{\\pi}{2}
    \\right)') is scored wrong, so the absolute rate is a lower bound — the paired A/B is exact, because the
    scorer is deterministic and identical on both arms.
    """
    c = reply["content"]
    tail = c.rpartition("####")[2] if "####" in c else ""
    if not tail.strip():
        boxes = re.findall(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}", c)
        tail = boxes[-1] if boxes else c[-200:]
    pred = (tail.strip().splitlines() or [""])[0].strip()
    g, p = tex(item["gold"]), tex(pred)
    ok = bool(p) and p == g
    if not ok and p and g:
        pv, gv = value(p), value(g)
        ok = pv is not None and gv is not None and abs(pv - gv) <= 1e-6 * max(1.0, abs(gv))
    return {"correct": ok, "score": float(ok), "pred": pred[:120], "gold": item["gold"],
            "detail": {"subject": item.get("subject"), "level": item.get("level")}}


def task(load, quick, score=score_aime25, max_tokens=32768):
    return {"langs": ("en",), "load": load, "score": score, "think": True, "max_tokens": max_tokens,
            "counts": {"quick": {"en": quick}, "full": {"en": None}}, "lang_check": False}


# 30 problems is the whole set: quick and full both run all of them, because a 30-item
# cell at ~10% accuracy has too few successes to subsample without losing all signal.
TASKS = {"aime25": task(load_aime25, 30), "musr": task(load_musr, 100, score_musr, max_tokens=8192),
         "math_500": task(load_math500, 150, score_math500, max_tokens=8192)}
