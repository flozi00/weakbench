"""search_rag: retrieval-augmented QA where the model has to *search*, not just read.

MultiHop-RAG (yixuantt/MultiHopRAG, ODC-BY): 2,556 questions over a 609-article news corpus, each
answerable only by combining evidence from 2-4 articles. The point of this cell is the capability the
other research tasks cannot see: **query formulation**. `research` hands the model five passages and
asks it to cite them; `ruler` and `aa_lcr` hand it one long context. None of them make the model decide
what to look for, notice that the first search was not enough, and search again.

So the model gets one tool, `search(query, k)`, backed by a BM25 index over the corpus, and up to four
rounds. It never sees the corpus. Everything is offline and deterministic: BM25 is 40 lines of stdlib,
the corpus is 6.8 MB, and the gold is a string.

A model that is still calling the tool when the budget runs out gets one more turn with a user nudge
("you have used all your searches, answer now") and is scored on that answer, with `tool_forced` set.
Without it the cell measured "did it stop searching" instead of "did it answer": on a 27B model 20 of 24
items hit the cap and returned an empty reply, which scored as wrong and put the cell at 16.7 %. With
the forced answer the same 24 items score 62.5 %, spread over all four types. `tool_choice: "none"` is
not the way to do this — vLLM returns empty content for it once the history carries tool_calls.

Four question types, and the fourth is the reason this task earns its place:
  comparison_query 856  "do A and B both report X" -> Yes/No
  inference_query  816  "who is the person that ..." -> a name
  temporal_query   583  "which happened first" -> an entity
  null_query       301  the evidence does NOT exist in the corpus; the gold answer is literally
                        "Insufficient information." A model that invents an answer here is hallucinating
                        under retrieval pressure, which is the failure mode RAG actually has.

Scoring is deliberately blunt, because a judge would make this opt-in and the gate needs it cheap:
  - null_query: an insufficiency marker in the reply is CORRECT, any confident answer is INCORRECT.
  - Yes/No gold: the polarity of the reply's last sentence must match. Note the always-"Yes" baseline:
    782 of the 1,318 binary items are "Yes" (59%), so read this arm against the other arm, never as an
    absolute score.
  - anything else: the gold answer as a whole word inside the reply.

English only. The corpus is English news; a German arm would need a translated corpus *and* translated
questions, and translating 609 articles changes what BM25 retrieves, so a DE cell here would measure the
translator. German retrieval is covered by `research` (DE/EN paired passages) instead.
"""
import json
import math
import random
import re
from collections import Counter

from common import hf_file

MHR = ("yixuantt/MultiHopRAG", "71ac0d0bd1f951d2d6b70311f7d2ae404e1ffa82")
QUERIES, CORPUS = "MultiHopRAG.json", "corpus.json"
TYPES = ("comparison_query", "inference_query", "temporal_query", "null_query")
INSUFFICIENT = ("insufficient information", "not enough information", "no information", "cannot determine",
                "can't determine", "unable to determine", "not mentioned", "no evidence", "does not contain",
                "doesn't contain", "cannot be determined", "insufficient evidence", "no relevant")
STOP = frozenset("""a an the and or but if of to in on at for with from by as is are was were be been being
this that these those it its their his her they them we you i he she not no do does did have has had will
would can could should may might must than then there here what which who whom whose when where why how""".split())


def _tok(text):
    return [w for w in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(w) > 1 and w not in STOP]


class BM25:
    """Okapi BM25 over a fixed corpus. Small enough that a dependency would be the bigger cost."""

    def __init__(self, docs, k1=1.5, b=0.75):
        self.docs, self.k1, self.b = docs, k1, b
        self.toks = [_tok(f"{d['title']} {d['body']}") for d in docs]
        self.avg = sum(len(t) for t in self.toks) / max(1, len(self.toks))
        df = Counter()
        for t in self.toks:
            df.update(set(t))
        n = len(self.toks)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}

    def search(self, query, k=5):
        qt = _tok(query)
        if not qt:
            return []
        scored = []
        for i, t in enumerate(self.toks):
            tf = Counter(t)
            s = sum(self.idf.get(w, 0.0) * tf[w] * (self.k1 + 1) /
                    (tf[w] + self.k1 * (1 - self.b + self.b * len(t) / self.avg)) for w in qt)
            if s > 0:
                scored.append((s, i))
        scored.sort(key=lambda x: (-x[0], x[1]))  # ties by corpus order: deterministic
        return [self.docs[i] for _, i in scored[:k]]


_INDEX = {}


def _index():
    """The BM25 index, built once per process (609 docs, ~1 s)."""
    if "bm25" not in _INDEX:
        _INDEX["bm25"] = BM25(json.loads(hf_file(*MHR, CORPUS).read_text()))
    return _INDEX["bm25"]


TOOLS = [{"type": "function", "function": {
    "name": "search",
    "description": "Search the news corpus. Returns the best-matching articles with title, source, date "
                   "and an excerpt. Call it as often as you need; one search rarely answers a "
                   "multi-hop question.",
    "parameters": {"type": "object", "properties": {
        "query": {"type": "string", "description": "Keywords or a question. Prefer distinctive terms."},
        "k": {"type": "integer", "description": "How many articles to return (1-10, default 5)."}},
        "required": ["query"]}}}]

ASK = ("Answer the question using the search tool. The corpus is a set of news articles; you cannot see "
       "it directly, so search for the evidence you need. Multi-hop questions usually need more than one "
       "search. If the corpus does not contain the evidence, answer exactly \"Insufficient information.\" "
       "Finish with a single line: \"Answer: <your answer>\".")


def _exec(name, args):
    """The tool implementation. `args` arrives as a JSON string from the model."""
    if name != "search":
        return f"error: unknown tool {name!r}; the only tool is 'search'"
    try:
        a = json.loads(args) if isinstance(args, str) else (args or {})
    except ValueError:
        return "error: arguments were not valid JSON"
    q = a.get("query") or a.get("q") or ""
    k = a.get("k") or 5
    try:
        k = max(1, min(10, int(k)))
    except (TypeError, ValueError):
        k = 5
    hits = _index().search(q, k)
    if not hits:
        return "No articles matched that query."
    return "\n\n".join(f"[{n + 1}] {d['title']} — {d['source']}, {str(d['published_at'])[:10]}\n"
                       f"{d['body'][:600]}" for n, d in enumerate(hits))


def load_search_rag(lang, n, seed):
    """Round-robin over the four question types, so any prefix of the order is balanced."""
    rows = json.loads(hf_file(*MHR, QUERIES).read_text())
    buckets = {t: [r for r in rows if r["question_type"] == t] for t in TYPES}
    rng = random.Random(seed)
    for b in buckets.values():
        rng.shuffle(b)
    order, depth = [], 0
    while any(len(b) > depth for b in buckets.values()):
        order += [b[depth] for b in buckets.values() if len(b) > depth]
        depth += 1
    items = []
    for r in (order if n is None else order[:n]):
        items.append({"id": f"{r['question_type']}:{len(items):04d}", "lang": "en",
                      "messages": [{"role": "system", "content": ASK},
                                   {"role": "user", "content": r["query"]}],
                      "gold": r["answer"], "question_type": r["question_type"],
                      "evidence": [e["title"] for e in r["evidence_list"]],
                      "tools": TOOLS, "tool_exec": _exec, "max_rounds": 4})
    return items


def _polarity(text):
    """'yes' / 'no' / None from the last sentence that carries a polarity."""
    for sent in reversed([s for s in re.split(r"(?<=[.!?])\s+", text or "") if s.strip()]):
        s = sent.lower()
        if re.search(r"\b(yes|true|correct|both do|they do)\b", s):
            return "yes"
        if re.search(r"\b(no|false|incorrect|neither|they do not|do not)\b", s):
            return "no"
    return None


def score_search_rag(item, reply):
    """Insufficiency for null_query, polarity for Yes/No, whole-word containment otherwise."""
    c = reply["content"]
    low = c.lower()
    meta = {"question_type": item["question_type"], "tool_rounds": reply.get("tool_rounds"),
            "tool_forced": reply.get("tool_forced")}
    if item["question_type"] == "null_query":
        ok = any(m in low for m in INSUFFICIENT)
        return {"correct": ok, "score": 1.0 if ok else -1.0, "pred": c.strip()[:200], "gold": item["gold"],
                "detail": {**meta, "verdict": "CORRECT" if ok else "HALLUCINATED"}}
    gold = item["gold"].strip()
    if gold.lower() in ("yes", "no"):
        ok = _polarity(c) == gold.lower()
    else:
        g = re.sub(r"\s+", " ", gold.lower()).strip()
        ok = bool(g) and re.search(rf"(?<![a-z0-9]){re.escape(g)}(?![a-z0-9])", low) is not None
    return {"correct": ok, "score": 1.0 if ok else -1.0, "pred": c.strip()[:200], "gold": gold,
            "detail": {**meta, "verdict": "CORRECT" if ok else "INCORRECT"}}


TASKS = {"search_rag": {"langs": ("en",), "load": load_search_rag, "score": score_search_rag, "think": False,
                        "max_tokens": 4096, "counts": {"quick": {"en": 150}, "full": {"en": None}},
                        "lang_check": False}}
