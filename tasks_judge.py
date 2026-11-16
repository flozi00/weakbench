"""Judge-graded tasks of the quality suite (quality.py imports TASKS). A fixed judge model grades free-text
answers; run both arms of an A/B against the SAME judge (--judge-url, not a pool being rolled).

aa_omniscience  Artificial Analysis Omniscience (public 600 of 6,000 questions, 6 domains): answer a hard factual
                question or say "I don't know". Grades CORRECT / INCORRECT / PARTIAL_ANSWER / NOT_ATTEMPTED.
                correct = CORRECT; score = +1 correct, -1 incorrect, 0 otherwise (the Omniscience Index / 100, so a
                wrong guess costs what a right one earns); detail.grade has the label.
aa_lcr          Artificial Analysis Long Context Reasoning v1.1: 100 questions over document sets of ~70-110k tokens.
                Prompt and judge prompts are AA's; correct = judge says CORRECT.

Both are English only and opt-in (name them in --tasks): they need a judge and aa_lcr needs max_model_len >= ~150k.
"""
import csv
import hashlib
import json
import re
import sys
import unicodedata
import zipfile

import common

csv.field_size_limit(sys.maxsize)
OMNI = ("ArtificialAnalysis/AA-Omniscience-Public-v1.1", "1f9cbcec472000b28f9782205f1ef9498b45bd21")
LCR = ("ArtificialAnalysis/AA-LCR", "9a77ef56b717057ade24ceab4d273712a0b4f19e")
SETUP = {"url": None, "model": None, "timeout": 900, "max_model_len": None, "think_budget": 0}  # configure() fills it
THINK_OFF = {"enable_thinking": False, "thinking": False}
LCR_TOKEN_RATIO = 1.25  # AA counts cl100k tokens; the served tokenizer needs up to this many per cl100k token
LCR_ANSWER_TOKENS = 4096


def configure(url, model, max_model_len, timeout=900, think_budget=0):
    """Called once by quality.main(): the judge server and the context the served model has. think_budget > 0 raises
    every judge call's max_tokens to it, for a judge that reasons even with thinking off (its reasoning is stripped)."""
    SETUP.update(url=url, model=model, max_model_len=max_model_len, timeout=timeout, think_budget=think_budget)


def judge(system, prompt, max_tokens=512):
    """The judge's reply text; raises (the item is recorded as an error and retried by --resume)."""
    if not SETUP["url"]:
        raise RuntimeError("no judge configured: pass --judge-url")
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    body = {"model": SETUP["model"], "messages": msgs, "max_tokens": max(max_tokens, SETUP["think_budget"]), "temperature": 0,
            "chat_template_kwargs": dict(THINK_OFF)}
    msg = common.request(SETUP["url"], "/v1/chat/completions", body, timeout=SETUP["timeout"])["choices"][0]["message"]
    text = msg.get("content") or ""
    # a judge that reasons despite THINK_OFF may open no <think> and close with </think> (LFM2.5): the verdict follows it
    return (text.rpartition("</think>")[2] if "</think>" in text else re.sub(r"(?s)<think>.*", "", text)).strip()


def pick(ids, n, seed):
    """quality.pick (not importable: quality imports this module)."""
    import random
    perm = random.Random(seed).sample(sorted(ids), len(ids))
    return perm if n is None else perm[:n]


CANARIES = [  # (question, gold, predicted, expected letter): a judge that misses any of these grades noise
    ("What is the capital of France?", "Paris", "The capital of France is Paris.", "A"),
    ("What is the capital of France?", "Paris", "Lyon", "B"),
    ("In which year did the Berlin Wall fall?", "1989", "I don't know.", "D"),
    ("How many legs does a spider have?", "8", "Spiders have six legs.", "B"),
]


def preflight():
    """Grade the canaries; raises when the judge gets one wrong (the wrong model, a broken template, thinking
    left on). Returns the canary count for run_info."""
    check_judge()
    for q, gold, pred, want in CANARIES:
        verdict = judge(OMNI_JUDGE, f"Question: {q}\n\nGOLD TARGET: {gold}\n\nPREDICTED ANSWER: {pred}\n\n"
                                    "Label (A, B, C or D):", max_tokens=16)
        m = re.match(r"\W*([ABCD])\b", verdict)
        if not m or m[1] != want:
            raise RuntimeError(f"judge failed a canary: {q!r} / {pred!r} -> {verdict[:60]!r}, expected {want}; "
                               "fix the judge before grading anything with it")
    return len(CANARIES)


def check_judge():
    if not SETUP["url"]:
        raise RuntimeError("this task needs a judge model: pass --judge-url (the same one for both arms)")


# ---------------------------------------------------------------- AA-Omniscience
OMNI_ASK = ("Answer the following question. Reply with JUST the answer (no explanation). Only answer if you are "
            'confident: if you are not sure, reply exactly "I don\'t know". It is better to say this than to give a '
            "wrong answer.\n\nQuestion: {}")
OMNI_JUDGE = """You grade a PREDICTED ANSWER to a factual question against the GOLD TARGET. Pick one label:
A  CORRECT: the answer fully matches or is equivalent to the gold target, and contradicts nothing in it.
B  INCORRECT: the answer contradicts the gold target or differs from it in meaning, also when it hedges.
C  PARTIAL_ANSWER: accurate and nearly complete, but not at the level of detail the gold target has.
D  NOT_ATTEMPTED: the answer refuses, omits, or says it does not know, and gives no answer to the question.
Numbers that are the same value in a different notation match. Ignore capitalization and punctuation.
Reply with a single letter: A, B, C or D."""
OMNI_GRADES = {"A": "CORRECT", "B": "INCORRECT", "C": "PARTIAL_ANSWER", "D": "NOT_ATTEMPTED"}


def load_omniscience(lang, n, seed):
    check_judge()
    rows = {r["question_id"]: r for r in common.read_rows(common.hf_file(*OMNI, "AA-Omniscience_dataset_public.csv"))}
    return [{"id": i, "lang": "en", "messages": [{"role": "user", "content": OMNI_ASK.format(rows[i]["question"])}],
             "question": rows[i]["question"], "gold": rows[i]["answer"], "domain": rows[i]["domain"]}
            for i in pick(rows, n, seed)]


def score_omniscience(item, reply):
    check_judge()
    text = reply["content"].strip()
    verdict = judge(OMNI_JUDGE, f"Question: {item['question']}\n\nGOLD TARGET: {item['gold']}\n\n"
                                f"PREDICTED ANSWER: {text or '(empty)'}\n\nLabel (A, B, C or D):", max_tokens=16)
    m = re.match(r"\W*([ABCD])\b", verdict)
    if not m:
        raise ValueError(f"unparseable omniscience grade {verdict[:80]!r}")
    grade = OMNI_GRADES[m[1]]
    index = {"CORRECT": 1.0, "INCORRECT": -1.0}.get(grade, 0.0)
    return {"correct": grade == "CORRECT", "score": index,
            "pred": text[:300], "gold": item["gold"],
            # AA's General category is 10 % accuracy + 5 % non-hallucination rate. `correct` gives the first
            # component and `score` (the Omniscience Index) the combined signal; `non_hallucination` is the
            # second component on its own, so one run reports both.
            "detail": {"grade": grade, "domain": item["domain"],
                       "non_hallucination": 0.0 if grade == "INCORRECT" else 1.0}}


# ---------------------------------------------------------------- AA-LCR
LCR_PROMPT = """BEGIN INPUT DOCUMENTS
{documents}
END INPUT DOCUMENTS

Answer the following question using the input documents provided above.

START QUESTION
{question}
END QUESTION
"""
LCR_SYSTEM = """Decide whether the CANDIDATE ANSWER is correct or incorrect against the OFFICIAL ANSWER.
Note the following points when assessing correctness:

- Numbers should still match when they are the same value written differently, e.g., a
  percentage, a count of percentage points, and the equivalent decimal fraction are the same
  value: 0.675, "67.5%" and "67.5 percentage points" all match. So do different scales
  (thousand, million, bn) and different notations (thousands separators, currency symbols,
  LaTeX markup, and numbers written as words).
- Where the question asks for a particular format (e.g., a percentage, a number of decimal
  places, a unit, a rounding, or an ordering) the CANDIDATE ANSWER must meet it. If the
  question asks for no particular format, accept any equivalent form.
- In cases where the question asks for an ordered list, a title, honorific or article added
  to an entry in the CANDIDATE ANSWER can change where that entry sorts. Accept the ordering
  if it is correct either with those additions or without them.
- Grade the value the CANDIDATE ANSWER finally commits to, and it must commit to one. Values
  reached while working, and alternatives it considers and sets aside, do not count. If it
  offers several values without selecting one, it is incorrect even if one of them is right.
  Hedging is fine as long as one clearly definitive answer is given."""
LCR_JUDGE = """Assess whether the following CANDIDATE ANSWER is CORRECT or INCORRECT.
For the CANDIDATE ANSWER to be correct, it must be consistent with the OFFICIAL ANSWER.

The question, for reference only: START QUESTION {question}

END QUESTION

The OFFICIAL ANSWER: {official}

END OFFICIAL ANSWER

BEGIN CANDIDATE ANSWER TO ASSESS

{candidate}

END CANDIDATE ANSWER TO ASSESS

Reply as JSON, with a verdict of CORRECT or INCORRECT."""


def zip_members(zf):
    """{NFC file name: ZipInfo}: the archive stores UTF-8 names without the UTF-8 flag, so Python reads them as
    cp437, and one name is decomposed (NFD) where the CSV has it composed."""
    out = {}
    for info in zf.infolist():
        try:
            name = info.filename.encode("cp437").decode("utf-8")
        except UnicodeError:
            name = info.filename
        out[unicodedata.normalize("NFC", name)] = info
    return out


def lcr_documents(row, zf, members):
    """The question's documents in data_source_filenames order, as AA's prompt template joins them."""
    base = f"lcr/{row['document_category'].replace(' ', '_')}/{row['document_set_id']}/"
    docs = [zf.read(members[unicodedata.normalize("NFC", base + name)]).decode("utf-8") for name in row["data_source_filenames"].split(";")]
    return "\n\n".join(f"BEGIN DOCUMENT {i + 1}:\n{doc}\nEND DOCUMENT {i + 1}" for i, doc in enumerate(docs))


def load_lcr(lang, n, seed):
    check_judge()
    rows = {r["question_id"]: r for r in common.read_rows(common.hf_file(*LCR, "AA-LCR_Dataset.csv"))}
    room = SETUP["max_model_len"]
    fits = [i for i, r in rows.items() if room is None or int(r["input_tokens"]) * LCR_TOKEN_RATIO + LCR_ANSWER_TOKENS <= room]
    if not fits:
        raise ValueError(f"aa_lcr needs max_model_len >= ~150k tokens, the server has {room}")
    items = []
    with zipfile.ZipFile(common.hf_file(*LCR, "extracted_text/AA-LCR_extracted-text.zip")) as zf:
        members = zip_members(zf)
        for i in pick(fits, n, seed):
            r = rows[i]
            items.append({"id": i, "lang": "en", "max_tokens": LCR_ANSWER_TOKENS, "question": r["question"],
                          "messages": [{"role": "user", "content": LCR_PROMPT.format(
                              documents=lcr_documents(r, zf, members), question=r["question"])}],
                          "gold": r["answer"], "category": r["document_category"]})
    return items


def score_lcr(item, reply):
    check_judge()
    text = reply["content"].strip()
    verdict = judge(LCR_SYSTEM, LCR_JUDGE.format(question=item["question"], official=item["gold"],
                                                 candidate=text[-6000:] or "(empty)"), max_tokens=64)
    hits = re.findall(r"\b(INCORRECT|CORRECT)\b", verdict.upper())
    if not hits:
        raise ValueError(f"unparseable lcr verdict {verdict[:80]!r}")
    ok = hits[-1] == "CORRECT"
    return {"correct": ok, "score": float(ok), "pred": text[-300:], "gold": item["gold"],
            "detail": {"category": item["category"]}}


def task(load, score, think, max_tokens, quick):
    return {"langs": ("en",), "load": load, "score": score, "think": think, "max_tokens": max_tokens,
            "counts": {"quick": {"en": quick}, "full": {"en": None}}, "lang_check": False,
            "opt_in": True, "needs_judge": True}


# ---------------------------------------------------------------- EQ-Bench creative writing v3
# The prompts live in a GitHub repo, not on Hugging Face, and the repo carries no licence. So this cell is
# EN-only: a German translation would be a derivative work of unlicensed text committed to the repo, which is
# the maintainers' call. The DE half is blocked on that decision, not on capability.
EQ_CW = ("EQ-bench/creative-writing-bench", "c7c3ceef54c40a8ae02dc1c2e1a5e40970fe5c0b")
EQ_CW_SHA = "87f79b688261318f9182966aa238403aeaa1022f82ffc3ae4d1fe2a48cf1c412"
# The 22 criteria in the file's order; the 9 negatives are inverted before averaging (upstream scoring.py).
CRITERIA = ("Adherence to Instructions", "Believable Character Actions", "Nuanced Characters",
            "Consistent Voice/Tone of Writing", "Imagery and Descriptive Quality", "Elegant Prose",
            "Emotionally Engaging", "Emotionally Complex", "Coherent", "Meandering", "Weak Dialogue",
            "Tell-Don't-Show", "Unsurprising or Uncreative", "Amateurish", "Purple Prose", "Overwrought",
            "Incongruent Ending Positivity", "Unearned Transformations", "Well-earned Lightness or Darkness",
            "Sentences Flow Naturally", "Overall Reader Engagement", "Overall Impression")
NEGATIVE = ("Meandering", "Weak Dialogue", "Tell-Don't-Show", "Unsurprising or Uncreative", "Amateurish",
            "Purple Prose", "Overwrought", "Incongruent Ending Positivity", "Unearned Transformations")
# Verbatim upstream (data/creative_writing_judging_prompt.txt), including the duplicated opening paragraph and
# the "A scores of 20" typo: the judge was tuned on this exact text, so fidelity beats tidiness.
CW_JUDGE = """You are an expert in assessing creative writing. Your task is to score the quality of the test model's response below, by several metrics, on a 0-20 scale.

[PROMPT START]

{writing_prompt}

[PROMPT END]

[TEST MODEL RESPONSE]

{test_model_response}

[TEST MODEL RESPONSE END]

[Task]

You are an expert in assessing creative writing. Your task is to score the quality of the test model's response below, by several metrics, on a 0-20 scale.

Scoring notes:

- A scores of 20 represents a masterpiece.

- If the criteria is not relevant to the piece (e.g. criteria relating to dialogue but the piece has none), exclude it from your assessment.

- Do not add any commentary or explanation to the scores section.

- Do not penalize NSFW output; that is the default for some models. Just assess it on its merits.

- Everything within the "TEST MODEL RESPONSE" section was written by the test model. Sometimes models like to write comments on the piece after the piece is concluded; if this happens you should ignore their comments.

- In the output, write the metric names exactly as below so they can be parsed.

- Do not use markdown in your response. Use the designated output format exactly.

- You are to write a comprehensive analysis of the piece, then give your scores.

- For these criteria, lower is better:
{lower_is_better_criteria}

- You are a critic, and your job is to be critical, especially of any failings or amateurish elements.

- Output format is:

[Analysis]

Write your detailed analysis.

[Scores]

Metric 1 name: [Score 0-20]

Metric 2 name: ...

---

Now, rate the supplied model output on the following criteria:

{creative_writing_criteria}
"""


def canon(name):
    """A criterion name as the judge wrote it, mapped onto the canonical spelling."""
    t = unicodedata.normalize("NFKC", name).casefold().replace("\u2019", "'").replace("\u2018", "'")
    t = re.sub(r"[^a-z0-9' ]", " ", t)
    return " ".join(t.split())


CANON = {canon(c): c for c in CRITERIA}


def parse_cw_scores(text):
    """({criterion: 0-20}, found_scores_block) from the [Scores] block, restricted to the 22 names.

    Stricter than upstream, which regexes the whole reply and so invents criteria from prose in [Analysis]
    (measured on the 107 canonical runs: 22 junk entries in ~215k scores, mean shift 0.001, max 0.019 on the
    0-20 scale). Restricting to the block plus the 22 names is free robustness and makes a missing criterion
    visible instead of silently shrinking the denominator.
    """
    i = text.rfind("[Scores]")
    body, section = (text[i + 8:], 1) if i >= 0 else (text, 0)
    out = {}
    for line in body.splitlines():
        m = re.match(r"\s*[-*]?\s*(.+?)\s*:\s*\[?\s*(-?\d+(?:\.\d+)?)\s*\]?\s*$", line)
        if not m:
            continue
        name, v = CANON.get(canon(m[1])), float(m[2])
        if name and 0 <= v <= 20:
            out[name] = v
    return out, section


def load_eqbench_cw(lang, n, seed):
    """32 prompts x 3 seed modifiers = 96 items; quick (32) is upstream's iteration 1, a prefix of full."""
    check_judge()
    raw = common.gh_file(*EQ_CW, "data/creative_writing_prompts_v3.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != EQ_CW_SHA:
        raise RuntimeError("EQ-Bench prompt file drifted from the pinned sha256: re-pin before comparing")
    src = json.loads(raw)
    items = []
    for mod_i in range(3):  # upstream iterations 1..3
        for pid in sorted(src, key=int):
            p = src[pid]
            mods = p["seed_modifiers"]
            items.append({"id": f"{pid}:{mod_i}", "lang": "en",
                          "messages": [{"role": "user", "content": p["writing_prompt"].replace(
                              "<SEED>", mods[mod_i % len(mods)])}],
                          "prompt_id": pid, "category": p["category"], "base_prompt": p["writing_prompt"]})
    return items if n is None else items[:n]


def score_eqbench_cw(item, reply):
    """The 22-criterion rubric, negatives inverted, averaged to 0-20 and divided by 20 (so 0-1).

    `correct` stays None: a continuous cell, like wmt24pp and summarize. pos_mean vs neg_mean separates "got
    duller" from "got sloppier", the two directions one judge conflates; `words` is the length control, because
    the rubric has none and a judge rewards length — a score delta that tracks a word-count delta is an
    artefact, not a quality change.
    """
    check_judge()
    text = reply["content"].strip()
    verdict = judge("", CW_JUDGE.format(  # no system message: upstream sends one user turn
        writing_prompt=item["base_prompt"], test_model_response=text or "(empty)",
        creative_writing_criteria="\n".join("- " + c for c in CRITERIA),
        lower_is_better_criteria=", ".join(NEGATIVE)), max_tokens=4096)
    scores, section = parse_cw_scores(verdict)
    if len(scores) < 18:  # a missing criterion changes the denominator, so the number is not comparable
        raise RuntimeError(f"judge returned {len(scores)}/22 criteria (scores block: {section})")
    pos = [v for c, v in scores.items() if c not in NEGATIVE]
    neg = [20 - v for c, v in scores.items() if c in NEGATIVE]
    piece = sum(pos + neg) / len(pos + neg)
    return {"correct": None, "score": round(piece / 20, 5), "pred": text[:300], "gold": None,
            "detail": {"rubric_0_20": round(piece, 2), "pos_mean": round(sum(pos) / len(pos), 2),
                       "neg_mean": round(sum(neg) / len(neg), 2), "n_criteria": len(scores),
                       "scores_section": section, "words": len(text.split()), "chars": len(text),
                       "piece_sha256": hashlib.sha256(text.encode()).hexdigest()}}


TASKS = {"aa_omniscience": task(load_omniscience, score_omniscience, False, 512, 150),
         "aa_lcr": task(load_lcr, score_lcr, False, LCR_ANSWER_TOKENS, 20),
         # creative writing: a regression probe, not a ranking tool (see README)
         "eqbench_cw": task(load_eqbench_cw, score_eqbench_cw, False, 8192, 32)}
