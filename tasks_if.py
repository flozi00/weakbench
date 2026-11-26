"""Instruction-following tasks of the quality suite (quality.py imports TASKS).

ifeval    BenchMAX_Rule-based: IFEval translated + post-edited, English-specific items
          dropped -> 429 prompts, parallel in de and en (id = IFEval key). Scored by the
          BenchMAX fork of the IFEval checkers (vendor/ifeval/) in the item's language.
          correct = prompt-level strict, score = instruction-level strict fraction.
ifstruct  LiquidAI IFStruct v1.0: 2000 JSON/YAML structured-output prompts (en, id =
          doc_id), scored by Liquid AI's validator (vendor/ifstruct_validator.py).

Needs pyyaml, langdetect, immutabledict, nltk>=3.9.1, pyarrow. load() fetches the
pinned data and nltk's punkt_tab into common.CACHE; score() is offline.
"""
import hashlib
import json
import random
import re

from common import CACHE, hf_file, read_rows

IFEVAL_DATA = ("LLaMAX/BenchMAX_Rule-based", "36a659fc9dddc2b344d98971ea0643527a3658d1")
IFSTRUCT_DATA = ("LiquidAI/ifstruct-v1.0", "2e342ca5f2673fb1287411fdbadcdfde82b35682")

def _ifeval_setup():
    import langdetect
    from vendor.ifeval import instructions_util

    instructions_util.NLTK_DATA_DIR = CACHE / "nltk"
    langdetect.DetectorFactory.seed = 0
    instructions_util.download_nltk_resources()
    langdetect.detector_factory.init_factory()


def _pick(ids, n, seed):
    """Deterministic subset: sorted ids, random.Random(seed).sample (n None = all)."""
    ids = sorted(ids)
    return set(ids if n is None else random.Random(seed).sample(ids, min(n, len(ids))))


def load_ifeval(lang, n, seed):
    rows = read_rows(hf_file(*IFEVAL_DATA, f"ifeval_{lang}.jsonl"))
    chosen = _pick([str(r["key"]) for r in rows], n, seed)
    # Fetch nltk data and langdetect's profiles now rather than in the first,
    # possibly concurrent, score() calls (langdetect's lazy init is not thread-safe).
    _ifeval_setup()
    # Items that ask for a reply in another language opt out of the runner's
    # German/English language check (complying is not a mismatch).
    return [{"id": str(r["key"]), "lang": lang, "messages": [{"role": "user", "content": r["prompt"]}],
             "instruction_id_list": r["instruction_id_list"], "kwargs": r["kwargs"],
             "lang_check": "language:response_language" not in r["instruction_id_list"]}
            for r in rows if str(r["key"]) in chosen]


def score_ifeval(item, reply):
    from vendor.ifeval import utils as ifeval

    inp = ifeval.InputExample(key=int(item["id"]), instruction_id_list=item["instruction_id_list"],
                              prompt=item["messages"][0]["content"], kwargs=item["kwargs"], lang=item["lang"])
    text = reply.get("content") or ""
    try:
        strict = ifeval.test_instruction_following_strict(inp, text).follow_instruction_list
        loose = ifeval.test_instruction_following_loose(inp, text).follow_instruction_list
    except RecursionError:  # json_format check on pathologically nested output ("[[[[..." loop)
        strict = loose = [False] * len(inp.instruction_id_list)
    return {"correct": all(strict), "score": sum(strict) / len(strict), "pred": None, "gold": None,
            "detail": {"inst_strict": strict, "prompt_loose": all(loose), "inst_loose_frac": sum(loose) / len(loose)}}


def load_ifstruct(lang, n, seed):
    rows = read_rows(hf_file(*IFSTRUCT_DATA, "test.parquet"))
    if n is None:
        chosen = {str(r["doc_id"]) for r in rows}
    else:  # stratified by output format: n//2 json + the rest yaml
        ids = lambda fmt: [str(r["doc_id"]) for r in rows if r["output_format"] == fmt]  # noqa: E731
        chosen = _pick(ids("json"), n // 2, seed) | _pick(ids("yaml"), n - n // 2, seed)
    # json_schema and top_level_count are JSON strings in the HF copy (the GitHub
    # data/test.jsonl the upstream eval reads has them parsed).
    return [{"id": str(r["doc_id"]), "lang": lang, "messages": [{"role": "user", "content": r["prompt"]}],
             "entity_type": r["entity_type"], "output_format": r["output_format"],
             "json_schema": json.loads(r["json_schema"]), "top_level_count": json.loads(r["top_level_count"]),
             "top_level_key": r["top_level_key"], "require_wrapper_key": r["require_wrapper_key"],
             "require_code_block": r["require_code_block"], "require_no_commentary": r["require_no_commentary"]}
            for r in rows if str(r["doc_id"]) in chosen]


def score_ifstruct(item, reply):
    from vendor.ifstruct_validator import validate_response

    try:  # argument mapping as in upstream ifstruct/eval.py run_one()
        res = validate_response(
            response=reply.get("content") or "",
            json_schema=item["json_schema"],
            top_level_count=item["top_level_count"],
            require_no_commentary=item["require_no_commentary"],
            output_format=item["output_format"],
            top_level_key=item["top_level_key"],
            require_wrapper_key=item["require_wrapper_key"],
            require_code_block=item["require_code_block"],
        )
        passed, errors = res.passed, res.errors
    except RecursionError:  # pathologically nested JSON ("[[[[..." loop); upstream crashes
        passed, errors = False, ["RecursionError: output nested too deeply to parse"]
    return {"correct": passed, "score": float(passed), "pred": None, "gold": None, "detail": {"errors": errors}}


IFSTRUCT_DE = (
    "Erstelle {n} Beispieldatensätze vom Typ „{entity}“ als {fmt}.\n\n"
    "Halte dich exakt an dieses JSON-Schema (die Feldnamen und Typen sind verbindlich):\n\n"
    "```json\n{schema}\n```\n\n"
    "{wrapper}{count}{code}{commentary}"
    "Antworte nur mit dem {fmt}-Dokument."
)


def _de_prompt(r):
    """A German structured-output prompt rendered from ifstruct's machine-readable fields.

    The upstream prompt is casual English prose that also carries the *content* brief, so it cannot be
    translated by template. What can be rendered from the row itself is every *format* requirement —
    schema, count, wrapper key, code block, commentary — which is exactly what the validator checks.
    So this cell measures "produce schema-conforming JSON/YAML from a German instruction", item-paired
    with `ifstruct` by doc_id, and it does not measure the content brief. Stated here so nobody reads a
    DE-vs-EN gap as a language effect when it is a prompt-richness effect.
    """
    cnt = json.loads(r["top_level_count"]) if isinstance(r["top_level_count"], str) else r["top_level_count"]
    lo, hi = (cnt, cnt) if isinstance(cnt, int) else (cnt[0], cnt[-1])  # int, [n] or [lo, hi]
    fmt = r["output_format"].upper()
    # entity_type is "test__<split>__<name>": drop the split marker, then read the name as words
    entity = " ".join(r["entity_type"].removeprefix("test__").replace("__", " ").replace("_", " ").split())
    schema = json.loads(r["json_schema"]) if isinstance(r["json_schema"], str) else r["json_schema"]
    return IFSTRUCT_DE.format(
        n=f"{lo} bis {hi}" if lo != hi else str(lo), entity=entity, fmt=fmt,
        schema=json.dumps(schema, indent=2, ensure_ascii=False),
        wrapper=(f"Verpacke das Array unter dem Schlüssel `{r['top_level_key']}`.\n\n"
                 if r["require_wrapper_key"] else "Das Wurzelelement ist ein bloßes Array.\n\n"),
        count=f"Erzeuge {lo} bis {hi} Einträge.\n\n" if lo != hi else f"Erzeuge genau {lo} Einträge.\n\n",
        code="Gib das Dokument in einem eingezäunten Codeblock aus.\n\n" if r["require_code_block"] else "",
        commentary="Gib keinen Text außerhalb des Dokuments aus.\n\n" if r["require_no_commentary"] else "")


def load_ifstruct_de(lang, n, seed):
    """The same 2,000 items as `ifstruct`, same ids, German prompt, same validator."""
    rows = read_rows(hf_file(*IFSTRUCT_DATA, "test.parquet"))
    if n is None:
        chosen = {str(r["doc_id"]) for r in rows}
    else:  # stratified by output format, as in load_ifstruct
        ids = lambda fmt: [str(r["doc_id"]) for r in rows if r["output_format"] == fmt]  # noqa: E731
        chosen = _pick(ids("json"), n // 2, seed) | _pick(ids("yaml"), n - n // 2, seed)
    return [{"id": str(r["doc_id"]), "lang": "de", "messages": [{"role": "user", "content": _de_prompt(r)}],
             "entity_type": r["entity_type"], "output_format": r["output_format"],
             "json_schema": json.loads(r["json_schema"]), "top_level_count": json.loads(r["top_level_count"]),
             "top_level_key": r["top_level_key"], "require_wrapper_key": r["require_wrapper_key"],
             "require_code_block": r["require_code_block"], "require_no_commentary": r["require_no_commentary"]}
            for r in rows if str(r["doc_id"]) in chosen]


# --- multi-turn: constraint persistence and override, deterministic (no judge) ------------------------
# Turn 1 is the IFEval prompt verbatim; turn 2 adds one constraint we can check ourselves; turn 3 drops it
# again. The final answer is scored against the *original* instruction set, so `forgot` (a constraint that held
# at turn 1 and not at turn 3) is the multi-turn signal, and `stale` (the dropped constraint still present) is
# the override signal. No instruction text has to be translated: the added constraint is one fixed sentence.
#
# Four kinds, not one, so the cell does not measure a single trick. Each kind names the IFEval instruction it
# would collide with, and an item never gets a kind whose instruction it already carries — otherwise turn 3
# ("drop it") would contradict turn 1 and the item would be unanswerable. `done` collides with nothing, so it
# is always available and is the fallback.
MT_KINDS = ("done", "upper", "bullets", "nocomma")
MT_COLLIDES = {"done": None, "upper": "change_case:all_upper_case",
               "bullets": "detectable_format:number_bullet_lists", "nocomma": "punctuation:no_comma"}
MT_ADD = {
    "done": {"en": "This time, also end your answer with the exact word DONE on its own line.",
             "de": "Beende deine Antwort diesmal zusätzlich mit dem exakten Wort FERTIG in einer eigenen Zeile."},
    "upper": {"en": "This time, also write your entire answer in UPPERCASE letters.",
              "de": "Schreibe deine gesamte Antwort diesmal zusätzlich in GROSSBUCHSTABEN."},
    "bullets": {"en": "This time, also format your answer as at least three bullet points.",
                "de": "Formatiere deine Antwort diesmal zusätzlich als mindestens drei Aufzählungspunkte."},
    "nocomma": {"en": "This time, also avoid using any commas in your answer.",
                "de": "Verwende in deiner Antwort diesmal zusätzlich keine Kommas."},
}
MT_DROP = {
    "en": "Answer the same request one more time. Drop the {what} requirement — do not {how}. Keep every "
          "other constraint from the original request.",
    "de": "Beantworte die Anfrage noch einmal. Lass die Vorgabe „{what}“ weg — {how}. Halte alle anderen "
          "Vorgaben aus der ursprünglichen Anfrage ein.",
}
MT_WHAT = {"done": {"en": "DONE", "de": "FERTIG"}, "upper": {"en": "UPPERCASE", "de": "GROSSBUCHSTABEN"},
           "bullets": {"en": "bullet-point", "de": "Aufzählungspunkt"},
           "nocomma": {"en": "no-comma", "de": "Komma"}}
MT_HOW = {"done": {"en": "write DONE anywhere", "de": "schreibe FERTIG nirgends"},
          "upper": {"en": "use uppercase", "de": "verwende keine Großbuchstaben"},
          "bullets": {"en": "use bullet points", "de": "verwende keine Aufzählungspunkte"},
          "nocomma": {"en": "use commas", "de": "verwende Kommas"}}


def mt_kind(item_id, instruction_ids):
    """A kind this item does not already carry, so turn 3 never contradicts turn 1.

    The starting point comes from the item id, so the four kinds are spread evenly instead of `done`
    taking every item; the walk then skips any kind whose instruction the item already carries.
    """
    start = int(hashlib.sha256(item_id.encode()).hexdigest()[:8], 16) % len(MT_KINDS)
    for off in range(len(MT_KINDS)):
        k = MT_KINDS[(start + off) % len(MT_KINDS)]
        if MT_COLLIDES[k] not in instruction_ids:
            return k
    return "done"


def mt_holds(kind, text):
    """Does the added constraint hold in `text`? Used for `stale`: it must NOT hold at turn 3."""
    if kind == "done":
        return bool(re.search(r"\b(done|fertig)\b", text, re.I))
    if kind == "upper":
        return bool(text.strip()) and text == text.upper()
    if kind == "bullets":
        return sum(1 for ln in text.splitlines() if re.match(r"\s*([-*•]|\d+[.)])\s+", ln)) >= 3
    return "," not in text


def load_multiturn(lang, n, seed):
    """IFEval items with at least two instructions, as a three-turn conversation (see the block comment)."""
    _ifeval_setup()
    rows = read_rows(hf_file(*IFEVAL_DATA, f"ifeval_{lang}.jsonl"))
    rows = [r for r in rows if len(r["instruction_id_list"]) >= 2]  # one to keep, one to lose
    by_key = {str(r["key"]): r for r in rows}
    ids = sorted(by_key)
    order = random.Random(seed).sample(ids, len(ids))  # ordered, so quick is a prefix of full
    items = []
    for key in (order if n is None else order[:n]):
        r = by_key[key]
        kind = mt_kind(key, r["instruction_id_list"])
        first = [{"role": "user", "content": r["prompt"]}]
        add = [{"role": "user", "content": "Answer the same request again. " + MT_ADD[kind][lang]
                if lang == "en" else "Beantworte die Anfrage erneut. " + MT_ADD[kind][lang]}]
        drop = [{"role": "user", "content": MT_DROP[lang].format(what=MT_WHAT[kind][lang], how=MT_HOW[kind][lang])}]
        items.append({"id": key, "lang": lang, "messages": first, "turns": [first, add, drop],
                      "instruction_id_list": r["instruction_id_list"], "kwargs": r["kwargs"], "mt_kind": kind,
                      "lang_check": "language:response_language" not in r["instruction_id_list"]})
    return items


def _strict(item, text):
    """Per-instruction strict flags of `text` against the item's original instruction set."""
    from vendor.ifeval import utils as ifeval

    inp = ifeval.InputExample(key=int(item["id"]), instruction_id_list=item["instruction_id_list"],
                              prompt=item["messages"][0]["content"], kwargs=item["kwargs"], lang=item["lang"])
    try:
        return ifeval.test_instruction_following_strict(inp, text).follow_instruction_list
    except RecursionError:  # json_format on pathologically nested output
        return [False] * len(inp.instruction_id_list)


def score_multiturn(item, reply):
    """Final answer against the original constraints, plus the dropped one must be gone.

    `correct` = every original constraint held on the final answer AND the dropped constraint is absent.
    `score` = (held constraints + dropped-ok) / (n + 1), so the override counts as one more constraint.
    `forgot` = share of constraints that held at turn 1 but not at turn 3 — the multi-turn signal.
    `stale` = 1.0 when the dropped constraint still holds, i.e. the model ignored the override.
    """
    turns = reply.get("turns") or [reply.get("content") or ""]
    final = turns[-1]
    strict = _strict(item, final)
    first = _strict(item, turns[0]) if len(turns) > 1 else strict
    stale = mt_holds(item["mt_kind"], final)
    forgot = [a and not b for a, b in zip(first, strict)]
    return {"correct": all(strict) and not stale,
            "score": round((sum(strict) + (not stale)) / (len(strict) + 1), 4),
            "pred": None, "gold": None,
            "detail": {"t1_strict": round(sum(first) / len(first), 4), "t3_strict": round(sum(strict) / len(strict), 4),
                       "forgot": round(sum(forgot) / len(forgot), 4) if forgot else 0.0,
                       "stale": float(stale), "mt_kind": item["mt_kind"], "inst_strict": strict}}


# --- german_json: German structured output with a content brief ----------------------------------------
# philipp-zettl/german-structured-output: 202 German test items whose system prompt is a German task
# description plus a ```json schema block, and whose user turn is a real German text. The gold is the
# annotation, so this measures *content* extraction under a schema, not just the shape — which is exactly
# what `ifstruct_de` cannot see (it renders a German prompt from machine-readable fields and never checks
# whether the brief was honoured). Task types: ner_extraction 102, relation_extraction 74,
# gdpr_anonymization 25, function_calling_parallel 16, function_calling_single 9, schema_following 1.
GJSON = ("philipp-zettl/german-structured-output", "359b9646e67ffe2da00fc071c22e9a634b06c488")
GJSON_FILE = "data/test-00000-of-00001.parquet"


def _schema_required(system):
    """The `required` list of the ```json schema block in the system prompt, or []."""
    m = re.search(r"```json\s*(\{.*?\})\s*```", system, re.S)
    if not m:
        return []
    try:
        return list(json.loads(m.group(1)).get("required") or [])
    except json.JSONDecodeError:
        return []


def load_german_json(lang, n, seed):
    rows = [r for r in read_rows(hf_file(*GJSON, GJSON_FILE)) if r["language"] == "de"]
    by_id = {}
    for r in rows:
        msgs = [m for m in r["messages"] if isinstance(m, dict)]
        sysm = next(m["content"] for m in msgs if m["role"] == "system")
        user = next(m["content"] for m in msgs if m["role"] == "user")
        gold = next(m["content"] for m in msgs if m["role"] == "assistant")
        by_id[str(r["id"])] = {"id": str(r["id"]), "lang": "de",
                               "messages": [{"role": "system", "content": sysm},
                                            {"role": "user", "content": user}],
                               "gold": gold, "required": _schema_required(sysm),
                               "task_type": r["task_type"]}
    ids = sorted(by_id)
    order = random.Random(seed).sample(ids, len(ids))  # ordered, so quick is a prefix of full
    return [by_id[i] for i in (order if n is None else order[:n])]


def _json_of(text):
    """The JSON object in a reply: bare, fenced, or embedded in prose. None when there is none."""
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", (text or "").strip())
    for cand in (t, (re.search(r"\{.*\}", t, re.S) or [None])[0]):
        if not cand:
            continue
        try:
            v = json.loads(cand)
            return v if isinstance(v, dict) else None
        except json.JSONDecodeError:
            continue
    return None


def _strings(obj, out):
    """Every string leaf, in order. Numbers are skipped: gold offsets are not reproducible."""
    if isinstance(obj, dict):
        for v in obj.values():
            _strings(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _strings(v, out)
    elif isinstance(obj, str):
        out.append(obj)
    return out


def score_german_json(item, reply):
    """Schema compliance plus content F1 against the gold annotation.

    `correct` = the reply parses, carries every `required` key, and recovers at least half the gold
    content — so McNemar has something to bite on instead of a near-constant schema flag.
    `score` = (schema_ok + content_f1) / 2, which keeps a schema-only reply from scoring as a win.
    """
    pred = _json_of(reply["content"])
    req = item["required"]
    schema_ok = isinstance(pred, dict) and all(k in pred for k in req)
    gold = _json_of(item["gold"]) or {}
    g = {s.strip().lower() for s in _strings(gold, []) if s.strip()}
    p = {s.strip().lower() for s in _strings(pred or {}, []) if s.strip()}
    hit = len(g & p)
    f1 = 2 * hit / (len(g) + len(p)) if g and p else 0.0
    return {"correct": bool(schema_ok and f1 >= 0.5), "score": round((float(schema_ok) + f1) / 2, 4),
            "pred": None, "gold": None,
            "detail": {"schema_ok": float(schema_ok), "content_f1": round(f1, 4),
                       "task_type": item["task_type"]}}


TASKS = {
    "ifeval": {"langs": ("de", "en"), "load": load_ifeval, "score": score_ifeval, "think": False,
               "max_tokens": 4096, "counts": {"quick": {"de": 214, "en": 214}, "full": {"de": None, "en": None}},
               "lang_check": True},
    "ifstruct": {"langs": ("en",), "load": load_ifstruct, "score": score_ifstruct, "think": False,
                 "max_tokens": 12288, "counts": {"quick": {"en": 300}, "full": {"en": None}},  # upstream: 16000
                 "lang_check": False},
    # same items, German instruction: the German half of the structured-output cell
    "ifstruct_de": {"langs": ("de",), "load": load_ifstruct_de, "score": score_ifstruct, "think": False,
                    "max_tokens": 12288, "counts": {"quick": {"de": 300}, "full": {"de": None}},
                    "lang_check": False},
    # three turns over the IFEval items: constraint persistence + override, no judge
    "multiturn": {"langs": ("de", "en"), "load": load_multiturn, "score": score_multiturn, "think": False,
                  "max_tokens": 4096, "counts": {"quick": {"de": 150, "en": 150}, "full": {"de": None, "en": None}},
                  "lang_check": True},
    # German structured output with a content brief: schema compliance AND extraction quality
    "german_json": {"langs": ("de",), "load": load_german_json, "score": score_german_json, "think": False,
                    "max_tokens": 4096, "counts": {"quick": {"de": 150}, "full": {"de": None}},
                    "lang_check": False},
}
