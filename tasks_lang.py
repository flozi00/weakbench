"""Language-quality tasks of the quality suite (quality.py imports TASKS): does the model write the RIGHT language,
and does it write it WELL? Both use our own hand-written DE/EN prompt pairs (licence-free, item-paired by id).

lang_switch   Deterministic, no judge. The item's `lang` is the language the reply must be in; the context pulls
              towards the other one. Four scenarios per topic:
                mirror      system prompt in the other language, question in the target language
                switch      two user turns, the first in the other language, the second in the target language
                instructed  question in the other language + "please answer in <target>"
                source      a colleague's note in the other language, the request about it in the target language
              correct = the whole reply is in the target language AND no sentence of >= 8 words is in the other
              one; score = 1 - share of wrong-language sentences (0 when the whole reply is wrong).
              Sub-metrics: ok_<scenario> (accuracy per scenario), wrong_sent_share. Function-word language id
              (common.lang_ok): approximate per sentence, identical on both arms.
lang_quality  Judge-graded. A writing task per genre (formal letter, explanation, complaint, instructions, ...).
              The judge LISTS every error with a verbatim quote (grammar, spelling, punctuation, idiom/translationese,
              language mixing, register) and gives fluency 1-10. A listed error whose quote is not in the reply is
              dropped (judge hallucination guard). Continuous cell: correct = None, score = fluency / 10.
              Sub-metrics: clean (share of error-free texts), errors_per_100w, err_<type>, task_fit, words, umlaut_sub (DE: "fuer"/"koennen" spellings).
              Opt-in, needs --judge-url (the same judge for both arms).
grammar_pairs Deterministic grammaticality judgement on MultiBLiMP (jumelet/multiblimp, CC BY 4.0): real treebank
              sentences next to a copy with one agreement error (subject-verb number or person). Which one is
              correct? Options shuffled per id. Swaps into or out of a
              contraction (is -> 'm, wird -> werd') are left out: there the spelling, not the grammar, gives it away. The DE and EN halves are different sentences (not item-paired).
              Sub-metrics: ok_number / ok_person, ok_attractor (an intervening noun/pronoun with other features).
spell_fix     Deterministic proofreading. Professional WMT24++ news/literary segments (EN source, DE human
              translation, item-paired by segment id) with 2-3 injected errors that are errors in any context:
              a lowercased capitalised word mid-sentence (DE nouns, EN proper nouns), two swapped letters inside a
              long word, a stripped umlaut (ambiguous pairs like schon/schön excluded), a dropped comma before a
              German subordinate clause, "n't" without its apostrophe. correct = every injected error fixed;
              score = share fixed. Sub-metrics: `exact` (the original restored, whitespace and quote style
              normalised), `similarity` (difflib ratio to the original: low with everything fixed means the
              model rewrote what it should have left alone), `fix_<kind>`. The EN sources are web/literary
              originals that carry a few errors of their own, so read `exact` on EN as a lower bound.
simplify      Plain-language rewriting: 12 hand-written DE/EN officialese notices (tax, residence registration,
              rent increase, ...), each with its key facts (numbers, dates, deadlines). The model rewrites it so a
              reader with little reading practice understands it. Deterministic: the reading-ease score (Amstad's
              German Flesch for DE, Flesch for EN, vowel-group syllables) must reach 60 AND rise by >= 15 over the
              source, and every key fact must still be there. correct = both; score = facts kept x readability
              reached (0-1). Sub-metrics `fre`, `fre_gain`, `facts_kept`, `words_per_sentence`.
register_shift German only: rewrite a message from du to Sie (formal) or from Sie to du (informal), 10 hand-written
              messages each way with their facts. Deterministic: no form of the old address may remain (du/dich/
              dir/dein*/euch/euer*, or mid-sentence Sie/Ihnen/Ihr*), the new one must appear, every fact must
              survive. correct = all three; sub-metrics `old_left`, `new_used`, `facts_kept`, `to_<target>`.
de_case       German only, generated: fill a gap with the article and adjective in the right case and gender
              ("Ich helfe ___ (der, alt) Nachbarn." -> "dem alten"), after verbs and prepositions that govern
              nominative, accusative, dative or genitive, with definite (weak) and indefinite (mixed) articles.
              Nouns are chosen so the noun form itself is the same in every case used (no n-declension,
              no genitive -s), so only article + adjective carry the case. Exact; `ok_<case>`, `ok_<article>`.
idioms        Hand-written idiom comprehension (own sets, not public): 20 German and 20 English idioms used in a
              sentence, four meanings each (one right, a literal reading, two plausible wrong ones), options
              shuffled per id. DE and EN are different idioms (not item-paired). Letter A-D.
de_verbs      German only, generated: strong/irregular verbs in a gap — Präteritum ("Gestern ___ (gehen) wir"),
              Perfekt with the right auxiliary and participle ("Wir ___ ___ (gehen)" -> "sind gegangen") and
              Konjunktiv II ("Wenn ich Zeit ___ (haben)" -> "hätte"), across persons. Exact; `ok_<tense>`.
roundtrip     Two chat turns on the WMT24++ news/literary segments: translate the text into the other language, then
              translate your translation back. score = chrF of the back-translation against the ORIGINAL (no
              reference translation needed; meaning lost in either direction shows). Cell lang = the original's
              language. Sub-metrics `chrf_back`, `copied` (turn 1 untranslated or turn 2 a repeat of turn 1).
              Limit: pasting the original back from the history in turn 2 looks like a perfect back-translation;
              read `roundtrip` next to `wmt24pp` / `wmt_mqm`, never alone.
locale_fmt    Generated locale formatting: write a number with thousands separators and two decimals, an amount with
              its currency, or an ISO date as a written date, in German or English convention ("1.234.567,89",
              "1.234,50 €", "5. März 2026" / "1,234,567.89", "€1,234.50", "March 5, 2026"). Exact (spaces and
              non-breaking spaces equal). `ok_<kind>`.
wordplay      Generated character-level tasks: count a letter in a long word (umlauts and ß included), write a word
              backwards, sort 6 words alphabetically (German: DIN 5007-1, ä = a, ß = ss). Exact. `ok_<kind>`.
sys_adherence Generated: a system prompt with 2-3 house rules (sign every reply with a fixed line, never name a
              competitor brand, at most N sentences, German: always the Sie form), then three user turns that push
              against them ("just tell me about <competitor>", "write a long answer", "du kannst mich duzen").
              Every assistant turn is checked, not only the last. correct = all rules held in all turns;
              score = share of (turn, rule) checks passed; `ok_<rule>`.
pragmatics    Hand-written scenes (own, DE and EN) whose meaning is not literal: irony, indirect requests, polite
              refusals, understatement. Four readings (intended, literal, two plausible wrong), shuffled per id;
              `literal` = picked the literal reading. Same scorer as idioms.
coref         Own Winograd-style twin sentences (DE and EN, not public): one word flips which candidate a pronoun
              refers to. Each twin is an item (ids "<pair>:<half>"), so a model that ignores the flip word gets at
              most one of each pair: chance is 50 %, flip-blind is 50 %; only reading the flip word goes above.
numerals      Generated number words: read a written-out number into digits, or write digits as words (German one
              word, units before tens: "dreihundertsiebenundvierzig"; English with hyphens, "and" optional).
              Numbers 21-999,999. Exact; `ok_read`, `ok_write`.
polyglot_switch  Five languages (de, en, fr, es, it): a question asked in one, the answer required in another (the
              instruction is in the question's language). langdetect on the reply (seeded, deterministic). Cell lang
              = the REQUIRED answer language for de/en; fr/es/it targets alternate between the cells.
              correct = the reply is in the required language; `ok_<from>_<to>`.
officialese_qa  The 12 `simplify` notices with one concrete question each ("By when must I register?"); the answer
              must contain the notice's key fact for that question (the fact list's spellings). Reading officialese,
              next to rewriting it. Exact substring on the reply.
de_letter     German business-letter conventions (DIN 5008), checked deterministically on 10 own letter tasks: a
              subject line without the word "Betreff", the salutation "Sehr geehrte(r) ...," with a comma, the first
              body word lowercase (unless a noun/"Sie"), the closing "Mit freundlichen Grüßen" WITHOUT a comma, and
              no du-forms. score = share of the 5 checks; `ok_<check>`.
paraphrase    Generated sentence pairs: a true paraphrase (active/passive, fronting) or a minimal meaning change
              (swapped roles, negation, number, "all"->"some"). Same meaning: YES/NO (JA/NEIN), balanced.
              `ok_<kind>` for each transformation.
de_compounds  German only, own: reversible compound pairs (Schlittenhund / Hundeschlitten, Taschenmesser /
              Messertasche, ...): "Is a <compound> a <A> or a <B>?" — the head is the last part. Both orders of
              every pair are items, so ignoring the order caps at 50 %. Exact head noun.
de_produce    German only, generated: write ONE sentence under grammatical constraints, checked by pattern —
              Perfekt of a given verb (auxiliary form + participle), a given noun after a dative preposition (dem/der
              + noun), a "weil"/"dass" clause with the finite verb at the clause end. correct = every constraint
              met; `ok_<constraint>`.
en_produce    English only, generated: write one sentence in a required construction (present perfect continuous
              with "since", past perfect before a simple past with "when"/"before", third conditional, passive with
              "by") about a given topic word; pattern-checked. correct = construction + topic word + one sentence.
"""
import csv
import difflib
import json
import random
import re

import common
import tasks_judge
import tasks_novel

OTHER = {"de": "en", "en": "de"}
# (de, en) open questions that need a paragraph, not a word: the reply must be long enough to identify
TOPICS = common.OwnSet("tasks_lang.TOPICS")  # private: flozi00/weakbench-data
PARA = {"de": " Antworte in einem Absatz.", "en": " Answer in one paragraph."}
SYSTEM = {"de": "Du bist ein hilfsbereiter Assistent.", "en": "You are a helpful assistant."}
ANSWER_IN = {("de", "en"): " Bitte antworte auf Englisch.", ("en", "de"): " Please answer in German.",
             ("de", "de"): "", ("en", "en"): ""}  # (language of the request, target language)
NOTE = {"de": "Eine Kollegin hat mir diese Notiz geschickt: «{q}» Erkläre mir kurz, worum es ihr geht, und gib "
              "eine Antwort, die ich ihr weiterleiten kann.",
        "en": "A colleague sent me this note: \"{q}\" Briefly tell me what she is asking and give an answer I can "
              "forward to her."}
SCENARIOS = ("mirror", "switch", "instructed", "source")


def q(topic, lang):
    return TOPICS[topic][0 if lang == "de" else 1]


def switch_item(k, scen, lang):
    o = OTHER[lang]
    base = {"id": f"{k}:{scen}", "lang": lang, "scenario": scen}
    user = lambda text: [{"role": "user", "content": text}]  # noqa: E731
    if scen == "mirror":
        return base | {"messages": [{"role": "system", "content": SYSTEM[o]}] + user(q(k, lang) + PARA[lang])}
    if scen == "switch":
        turns = [user(q(k, o) + PARA[o]), user(q((k + 1) % len(TOPICS), lang) + PARA[lang])]
        return base | {"messages": turns[0], "turns": turns}
    if scen == "instructed":
        return base | {"messages": user(q(k, o) + PARA[o] + ANSWER_IN[(o, lang)])}
    return base | {"messages": user(NOTE[lang].format(q=q(k, o)))}


def load_switch(lang, n, seed):
    items = [switch_item(k, s, lang) for k in range(len(TOPICS)) for s in SCENARIOS]  # quick = first topics
    return items if n is None else items[:n]


SENT = re.compile(r"(?<=[.!?])\s+|\n+")


def wrong_sentences(text, lang):
    """(wrong, decidable) sentences of >= 8 words; markdown bullets count as sentences."""
    verdicts = [common.lang_ok(s, lang, min_words=8) for s in SENT.split(text)]
    verdicts = [v for v in verdicts if v is not None]
    return sum(v is False for v in verdicts), len(verdicts)


def score_switch(item, reply):
    text = reply["content"]
    whole = common.lang_ok(text, item["lang"], min_words=15)
    wrong, n = wrong_sentences(text, item["lang"])
    share = wrong / n if n else 0.0
    ok = whole is True and wrong == 0
    detail = {"scenario": item["scenario"], f"ok_{item['scenario']}": float(ok), "wrong_sent_share": round(share, 4)}
    if item.get("turns") and reply.get("turns"):  # the first turn must have stayed in ITS language
        detail["first_turn_ok"] = float(common.lang_ok(reply["turns"][0], OTHER[item["lang"]], min_words=15) is True)
    return {"correct": ok, "score": 0.0 if whole is False else round(1 - share, 4), "pred": text[:300],
            "gold": item["lang"], "lang_ok": whole, "detail": detail}


# ---------------------------------------------------------------- lang_quality (judge)
WRITING = common.OwnSet("tasks_lang.WRITING")  # private: flozi00/weakbench-data
QUALITY_JUDGE = """You are a meticulous native-speaker editor of {language}. Proofread the TEXT below, which was written
for the TASK below. List EVERY language error, each with a verbatim quote (copied exactly, at most 12 words) from the
TEXT and the corrected form. Error types:
  grammar      agreement, case, verb form, word order, wrong article/preposition
  spelling     misspelling, wrong capitalisation, missing umlaut/ß or "ae/oe/ue" substitutes in {language}
  punctuation  missing/wrong commas (e.g. German subordinate clauses), quotes, apostrophes
  idiom        unnatural or translated-sounding phrasing, needless anglicism/calque, wrong collocation
  mixing       words or sentences in another language that a native writer would not use here
  register     inconsistent form of address (Sie/du switching, you/thou) or tone that breaks the task's register
Do not list stylistic preferences that are not errors. Then rate:
  fluency   1-10  how natural and native-like the text reads (10 = flawless native writing)
  task_fit  1-10  how well the text does the TASK (genre, audience, content)
Reply with ONLY a JSON object:
{{"errors": [{{"type": "...", "quote": "...", "fix": "..."}}], "fluency": n, "task_fit": n}}

TASK: {task}

TEXT:
{text}"""
TYPES = ("grammar", "spelling", "punctuation", "idiom", "mixing", "register")
UMLAUT_SUB = re.compile(r"\b(?:fuer|ueber|koennen|moechte|muessen|waere|hoeren|groesse|schoen|natuerlich|"
                        r"gruesse|aendern|frueh|spaeter|maedchen|koennte|wuerde|duerfen|fuehren)\b", re.I)


def load_quality(lang, n, seed):
    tasks_judge.check_judge()
    items = [{"id": g, "lang": lang, "genre": g, "task": de if lang == "de" else en,
              "messages": [{"role": "user", "content": de if lang == "de" else en}]} for g, de, en in WRITING]
    return items if n is None else items[:n]


def norm(s):
    return re.sub(r"\s+", " ", s.replace("’", "'").replace("„", '"').replace("“", '"').replace("”", '"')).strip().lower()


def parse_verdict(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError(f"no JSON in judge reply {text[:120]!r}")
    v = json.loads(m[0])
    if not isinstance(v.get("errors"), list) or not all(isinstance(v.get(k), (int, float)) for k in ("fluency",
                                                                                                     "task_fit")):
        raise ValueError(f"judge JSON lacks errors/fluency/task_fit: {m[0][:200]!r}")
    return v


def score_quality(item, reply):
    tasks_judge.check_judge()
    text = reply["content"].strip()
    v = parse_verdict(tasks_judge.judge("", QUALITY_JUDGE.format(
        language="German" if item["lang"] == "de" else "English", task=item["task"], text=text or "(empty)"),
        max_tokens=4096))
    hay = norm(text)
    errs = [e for e in v["errors"] if isinstance(e, dict) and e.get("type") in TYPES
            and norm(str(e.get("quote", ""))) and norm(str(e["quote"])) in hay]
    words = max(len(text.split()), 1)
    fluency = min(max(float(v["fluency"]), 1), 10)
    detail = {"genre": item["genre"], "clean": float(not errs and bool(text)), "errors_per_100w": round(100 * len(errs) / words, 3),
              "task_fit": min(max(float(v["task_fit"]), 1), 10), "words": words,
              "judge_dropped": len(v["errors"]) - len(errs), "errors": errs[:40],
              **{f"err_{t}": sum(e["type"] == t for e in errs) for t in TYPES}}
    if item["lang"] == "de":
        detail["umlaut_sub"] = len(UMLAUT_SUB.findall(text))
    return {"correct": None, "score": round(fluency / 10, 3), "pred": text[:300], "gold": None,
            "detail": detail}


# ---------------------------------------------------------------- grammar_pairs
BLIMP = ("jumelet/multiblimp", "de923efa8d2483d6b13364ee68e65308e990a991")
GP_TEXT = {"de": "Welcher der beiden Sätze ist grammatisch korrekt?\nA. {a}\nB. {b}\n"
                 "Antworte nur mit einer Zeile \"Antwort: A\" oder \"Antwort: B\".",
           "en": "Which of the two sentences is grammatically correct?\nA. {a}\nB. {b}\n"
                 "Reply with just one line \"Answer: A\" or \"Answer: B\"."}


def load_grammar(lang, n, seed):
    path = common.hf_file(*BLIMP, f"{'deu' if lang == 'de' else 'eng'}/data.tsv")
    with open(path, newline="") as f:
        rows = [r for r in csv.DictReader(f, delimiter="\t") if r["sen"] != r["wrong_sen"]
                and "'" not in r["head"] + r["swap_head"]]  # a contraction swap ("is" -> "'m") is a style cue
    items = []
    for i in tasks_judge.pick(range(len(rows)), n, seed):
        r, flip = rows[i], random.Random(f"grammar_pairs:{lang}:{i}").random() < 0.5
        a, b = (r["wrong_sen"], r["sen"]) if flip else (r["sen"], r["wrong_sen"])
        items.append({"id": str(i), "lang": lang, "gold": "B" if flip else "A",
                      "phen": "number" if r["phenomenon"] == "SV-#" else "person",
                      "attractor": r["num_incongruent_attractors"] not in ("0", ""),
                      "messages": [{"role": "user", "content": GP_TEXT[lang].format(a=a, b=b)}]})
    return items


def score_grammar(item, reply):
    line = tasks_novel.final_answer(reply["content"]) or ""
    m = re.findall(r"\b([AB])\b", line.upper().replace("SATZ", " ").replace("SENTENCE", " "))
    pred = m[-1] if len(set(m)) == 1 else None  # "A or B" is no answer
    ok = pred == item["gold"]
    detail = {f"ok_{item['phen']}": float(ok)}
    if item["attractor"]:
        detail["ok_attractor"] = float(ok)
    return {"correct": ok, "score": float(ok), "pred": pred, "gold": item["gold"], "detail": detail}


# ---------------------------------------------------------------- spell_fix
WMT = ("google/wmt24pp", "fd7405c06494bc66a57b25f55d217a72f96e60dc")
FIX_TEXT = {"de": "Korrigiere alle Rechtschreib-, Grammatik- und Zeichensetzungsfehler im folgenden Text. Ändere "
                  "sonst nichts. Gib nur den korrigierten Text aus.\n\n{t}",
            "en": "Correct all spelling, grammar and punctuation errors in the following text. Change nothing else. "
                  "Output only the corrected text.\n\n{t}"}
UML = {"ä": "a", "ö": "o", "ü": "u"}
UML_AMBIG = {"schon", "zahlen", "zahlt", "drucken", "konnte", "konnten", "wurde", "wurden", "musste", "mussten",
             "hatte", "hatten", "waren", "ware", "fordern", "fur", "uber", "zuruck", "mochte", "mochten", "zahlte",
             "dafur", "naher", "lange", "lasst", "schlagt", "tragt", "fallt", "bar", "mutter", "bruder", "vater",
             "apfel", "garten", "laden", "boden", "ofen", "vogel", "nagel", "kloster", "sattel", "graben", "mangel"}
SUB_CONJ = ("dass", "weil", "obwohl", "damit", "nachdem", "sodass", "ob")


def injections(text, lang):
    """Candidate (kind, start, end, wrong) edits that are errors wherever they appear."""
    out = []
    for m in re.finditer(r"(?<![.!?:]\s)(?<!^)(?<![\"„“'(\-])\b([A-ZÄÖÜ][a-zäöüß]{3,})\b", text):
        if m.start() > 0 and not re.search(r"[.!?:]\s*[\"„“']?\s*$", text[:m.start()]):
            out.append(("case", m.start(1), m.end(1), m[1][0].lower() + m[1][1:]))
    for m in re.finditer(r"\b[a-zäöüß]{7,}\b", text):
        w, k = m[0], 2 + (len(m[0]) - 5) // 2
        if w[k] != w[k + 1]:
            out.append(("swap", m.start(), m.end(), w[:k] + w[k + 1] + w[k] + w[k + 2:]))
    if lang == "de":
        for m in re.finditer(r"\b\w*[äöü]\w*\b", text):
            plain = re.sub("[äöü]", lambda c: UML[c[0]], m[0])
            if plain.lower() not in UML_AMBIG:
                out.append(("umlaut", m.start(), m.end(), plain))
        for m in re.finditer(r",( (?:%s)\b)" % "|".join(SUB_CONJ), text):
            out.append(("comma", m.start(), m.end(), m[1]))
    else:
        for m in re.finditer(r"\b(\w+)n't\b", text):
            out.append(("apostrophe", m.start(), m.end(), m[1] + "nt"))
    return out


def corrupt(text, lang, r):
    cands = injections(text, lang)
    r.shuffle(cands)
    chosen, used = [], []
    for c in cands:  # 2-3 non-overlapping edits, at most one per kind first
        if len(chosen) == 3 or any(c[1] < e and s_ < c[2] for s_, e in used):
            continue
        if sum(x[0] == c[0] for x in chosen) and len({x[0] for x in cands}) > len({x[0] for x in chosen}):
            continue
        chosen.append(c)
        used.append((c[1] - 1, c[2] + 1))
    out = text
    for kind, a, b, wrong in sorted(chosen, key=lambda c: -c[1]):
        out = out[:a] + wrong + out[b:]
    return out, [(kind, text[a:b], wrong) for kind, a, b, wrong in chosen]


def load_spell(lang, n, seed):
    rows = [r for r in common.read_jsonl(common.hf_file(*WMT, "en-de_DE.jsonl"))
            if r["domain"] in ("news", "literary") and r["is_bad_source"] not in (True, "True")]
    items = []
    for row in rows:
        text = row["target" if lang == "de" else "source"]
        if not 60 <= len(text) <= 400 or "\n" in text:
            continue
        r = random.Random(f"spell_fix:{lang}:{row['segment_id']}")
        bad, edits = corrupt(text, lang, r)
        if len(edits) >= 2:
            items.append({"id": str(row["segment_id"]), "lang": lang, "gold": text, "edits": edits,
                          "messages": [{"role": "user", "content": FIX_TEXT[lang].format(t=bad)}]})
    by_id = {i["id"]: i for i in items}
    return [by_id[i] for i in tasks_judge.pick(by_id, n, seed)]


def fix_norm(t):
    t = t.strip().strip("`").strip()
    return re.sub(r"\s+", " ", re.sub(r"[„“”«»]", '"', t.replace("’", "'"))).strip()


def score_spell(item, reply):
    pred, gold = fix_norm(reply["content"]), fix_norm(item["gold"])
    fixed = {}
    for kind, right, wrong in item["edits"]:
        # counts, not membership: ", dass" may occur elsewhere, and it contains " dass"
        ok = pred.count(right) >= gold.count(right) and pred.count(wrong) <= gold.count(wrong)
        fixed.setdefault(kind, []).append(ok)
    share = sum(sum(v) for v in fixed.values()) / len(item["edits"])
    return {"correct": share == 1, "score": round(share, 4), "pred": pred[:300], "gold": gold[:300],
            "detail": {"fixed_share": round(share, 4), "exact": float(pred == gold),
                       "similarity": round(difflib.SequenceMatcher(None, pred, gold).ratio(), 4),
                       **{f"fix_{k}": sum(v) / len(v) for k, v in fixed.items()}}}


# ---------------------------------------------------------------- simplify
NOTICES = common.OwnSet("tasks_lang.NOTICES")  # private: flozi00/weakbench-data
SIMPLIFY = {"de": "Schreibe diesen Text in einfacher Sprache um, sodass ihn auch jemand mit wenig Leseübung "
                  "versteht. Alle wichtigen Angaben müssen erhalten bleiben. Gib nur den neuen Text aus.\n\n{t}",
            "en": "Rewrite this text in plain language so that someone with little reading practice can understand "
                  "it. All important details must be kept. Output only the new text.\n\n{t}"}


def reading_ease(text, lang):
    """Flesch reading ease (EN) / Amstad (DE): higher is easier. Syllables = vowel groups (a heuristic, the same
    on both arms)."""
    words = re.findall(r"[^\W\d_]+", text)
    sents = max(len([x for x in re.split(r"(?<![\d§])(?<!\b[a-z])[.!?:;]+(?:\s|$)|\n+", text) if re.search(r"\w", x)]), 1)
    if not words:
        return 0.0, 0.0
    vow = "aeiouyäöü" if lang == "de" else "aeiouy"
    syl = sum(max(len(re.findall(f"[{vow}]+", w.lower())), 1) for w in words)
    asl, asw = len(words) / sents, syl / len(words)
    fre = 180 - asl - 58.5 * asw if lang == "de" else 206.835 - 1.015 * asl - 84.6 * asw
    return round(fre, 2), round(asl, 2)


OQ = [  # (de question, en question, index into the notice's facts)
    ("Wie viel Zeit habe ich für die Anmeldung?", "How long do I have to register?", 0),
    ("Um wie viel Euro steigt die Miete pro Monat?", "By how many euros per month does the rent go up?", 0),
    ("Bis wann muss die Steuererklärung abgegeben werden?", "By when must the tax return be filed?", 0),
    ("Für welchen Zeitraum muss ich Kontoauszüge vorlegen?", "For what period must I submit bank statements?", 0),
    ("Bis zu welcher Uhrzeit kann man die Briefwahl beantragen?", "Until what time can postal voting be requested?",
     0),
    ("Welchen Betrag muss ich zahlen?", "What amount do I have to pay?", 0),
    ("Um wie viel Prozent steigt die Gebühr bei Verspätung?", "By what percentage does the fee rise when late?", 1),
    ("Für wie viele Monate rückwirkend wird Elterngeld gezahlt?", "For how many months back is the allowance paid?",
     0),
    ("Wie oft wird die Biotonne geleert?", "How often is the organic bin emptied?", 0),
    ("Wie lang ist die Kündigungsfrist?", "How long is the notice period?", 1),
    ("Was kostet der Reisepass?", "How much does the passport cost?", 0),
    ("Wie hoch ist das Verwarnungsgeld ohne Behinderung?", "How high is the fine without obstruction?", 0),
]


def load_officialese(lang, n, seed):
    items = []
    for k, ((de, en, facts), (qde, qen, fi)) in enumerate(zip(NOTICES, OQ)):
        text, q = (de, qde) if lang == "de" else (en, qen)
        items.append({"id": str(k), "lang": lang, "fact": facts[fi],
                      "messages": [{"role": "user", "content": f"{text}\n\n{q} " + (
                          "Antworte kurz." if lang == "de" else "Answer briefly.")}]})
    return items if n is None else items[:n]


def score_officialese(item, reply):
    low = reply["content"].lower()
    ok = any(re.search(rf"(?<![\d\w]){re.escape(alt)}(?![\d])", low) for alt in item["fact"])  # "3" is not "13"
    return {"correct": ok, "score": float(ok), "pred": reply["content"][:120], "gold": item["fact"][0], "detail": {}}


DL_TASKS = common.OwnSet("tasks_lang.DL_TASKS")  # private: flozi00/weakbench-data
DL_ASK = ("Schreibe einen kurzen, formellen Geschäftsbrief an {to} über {topic}. Halte dich an die üblichen "
          "deutschen Konventionen (DIN 5008) für Betreff, Anrede und Grußformel. Lass Briefkopf und Adressfeld weg.")


def load_de_letter(lang, n, seed):
    items = [{"id": str(k), "lang": "de", "to": to,
              "messages": [{"role": "user", "content": DL_ASK.format(to=to, topic=topic)}]}
             for k, (to, topic) in enumerate(DL_TASKS)]
    return items if n is None else items[:n]


def score_de_letter(item, reply):
    lines = [ln.strip().strip("*_ ") for ln in reply["content"].strip().splitlines() if ln.strip()]
    sal = next((k for k, ln in enumerate(lines) if ln.startswith("Sehr geehrte")), None)
    checks = {
        "subject": bool(lines) and sal not in (None, 0) and not re.match(r"(?i)betreff\b", lines[0]),
        "salutation": sal is not None and lines[sal].endswith(","),
        "lowercase": sal is not None and sal + 1 < len(lines) and (
            lines[sal + 1][:1].islower() or re.match(r"(Sie|Ihr|Ihnen|[A-ZÄÖÜ]\w+\b)", lines[sal + 1]) is not None
            and not re.match(r"(Ich|Wir|Hiermit|Bitte|Gerne|Vielen|Leider|Mit|Da|Wie)\b", lines[sal + 1])),
        "closing": any(ln.startswith("Mit freundlichen Grüßen") and not ln.rstrip().endswith(",") for ln in lines),
        "sie": not DU_FORMS.search(reply["content"]),
    }
    return {"correct": all(checks.values()), "score": sum(checks.values()) / len(checks),
            "pred": reply["content"][:200], "gold": None, "detail": {f"ok_{k}": float(v) for k, v in checks.items()}}


DE_TIMES = {2: "zweimal", 3: "dreimal", 4: "viermal", 5: "fünfmal", 6: "sechsmal", 7: "siebenmal", 8: "achtmal"}
PP_PEOPLE = ["Anna", "Ben", "Clara", "David", "Emma", "Felix"]
PP_VERB = {"en": [("gave", "was given", "the book to"), ("sent", "was sent", "the letter to"),
                  ("showed", "was shown", "the photo to")],
           "de": [("gab", "gegeben", "das Buch"), ("schickte", "geschickt", "den Brief"),
                  ("zeigte", "gezeigt", "das Foto")]}


def pp_sentence(lang, kind, a, b, v, k, r):
    """(base, other) sentence pair for one transformation kind."""
    act, part, obj = PP_VERB[lang][v]
    if lang == "en":
        base = f"{a} {act} {obj} {b} {k} times last week."
        other = {"passive": f"Last week, {obj.split(' to')[0]} was {act.replace('gave', 'given').replace('sent', 'sent').replace('showed', 'shown')} to {b} by {a} {k} times.",
                 "front": f"Last week, {a} {act} {obj} {b} {k} times.",
                 "swap": f"{b} {act} {obj} {a} {k} times last week.",
                 "neg": f"{a} did not {act.replace('gave', 'give').replace('sent', 'send').replace('showed', 'show')} "
                        f"{obj} {b} last week.",
                 "number": f"{a} {act} {obj} {b} {k + r.randint(1, 3)} times last week."}[kind]
    else:
        base = f"{a} {act} {b} letzte Woche {DE_TIMES[k]} {obj}."
        nom = {"den Brief": "Der Brief", "das Buch": "Das Buch", "das Foto": "Das Foto"}[obj]  # passive subject
        other = {"passive": f"{nom} wurde {b} letzte Woche {DE_TIMES[k]} von {a} {part}.",
                 "front": f"Letzte Woche {act} {a} {b} {DE_TIMES[k]} {obj}.",
                 "swap": f"{b} {act} {a} letzte Woche {DE_TIMES[k]} {obj}.",
                 "neg": f"{a} {act} {b} letzte Woche {obj} nicht.",
                 "number": f"{a} {act} {b} letzte Woche {DE_TIMES[k + r.randint(1, 3)]} {obj}."}[kind]
    return base, other


PP_SAME = {"passive": True, "front": True, "swap": False, "neg": False, "number": False}
PP_ASK = {"en": "Do these two sentences mean the same thing?\n1. {a}\n2. {b}\nFinish with a line \"Answer: YES\" or "
                "\"Answer: NO\".",
          "de": "Bedeuten diese beiden Sätze dasselbe?\n1. {a}\n2. {b}\nSchließe mit einer Zeile \"Antwort: JA\" oder "
                "\"Antwort: NEIN\" ab."}


def load_paraphrase(lang, n, seed):
    kinds = ["passive", "swap", "front", "neg", "passive", "number", "front", "swap"]  # half same, half changed
    items = []
    for i in range(n or 160):
        r = random.Random(f"paraphrase:{i}")
        kind = kinds[i % len(kinds)]
        a, b = r.sample(PP_PEOPLE, 2)
        s1, s2 = pp_sentence(lang, kind, a, b, r.randrange(3), r.randint(2, 5), r)
        if r.random() < 0.5:
            s1, s2 = s2, s1
        items.append({"id": str(i), "lang": lang, "kind": kind, "gold": PP_SAME[kind],
                      "messages": [{"role": "user", "content": PP_ASK[lang].format(a=s1, b=s2)}]})
    return items


def score_paraphrase(item, reply):
    line = (tasks_novel.final_answer(reply["content"]) or "").upper()
    yes, no = re.search(r"\b(YES|JA)\b", line), re.search(r"\b(NO|NEIN)\b", line)
    pred = True if yes and not no else False if no and not yes else None
    ok = pred == item["gold"]
    return {"correct": ok, "score": float(ok), "pred": pred, "gold": item["gold"], "detail": {f"ok_{item['kind']}": float(ok)}}


DC_PAIRS = common.OwnSet("tasks_lang.DC_PAIRS")  # private: flozi00/weakbench-data
DC_HEADS = {"Hund": "Hund", "Schlitten": "Schlitten", "Messer": "Messer", "Tasche": "Tasche", "Bier": "Bier",
            "Flasche": "Flasche", "Boot": "Boot", "Haus": "Haus", "Baum": "Baum", "Topf": "Topf", "Blume": "Blume",
            "Glas": "Glas", "Wasser": "Wasser", "Zeitung": "Zeitung", "Papier": "Papier", "Lampe": "Lampe",
            "Tisch": "Tisch", "Buch": "Buch", "Kinder": "Kind", "Garten": "Garten"}


def load_de_compounds(lang, n, seed):
    items = []
    for k, (a, b, ab, ba) in enumerate(DC_PAIRS):
        heads = (b, a if a != "Blumen" else "Blume")  # head of ab is b, head of ba is a
        for half, (word, head) in enumerate(((ab, heads[0]), (ba, heads[1]))):
            ca, cb = sorted({DC_HEADS.get(heads[0], heads[0]), DC_HEADS.get(heads[1], heads[1])})
            items.append({"id": f"{k}:{half}", "lang": "de", "gold": DC_HEADS.get(head, head),
                          "options": (ca, cb),
                          "messages": [{"role": "user", "content": f"Was bezeichnet das Wort „{word}“ im Kern: {ca} oder "
                                        f"{cb}? Antworte mit einem Wort in einer Zeile \"Antwort: ...\"."}]})
    return items if n is None else items[:n]


def score_de_compounds(item, reply):
    line = (tasks_novel.final_answer(reply["content"]) or "").strip()
    found = {o for o in item["options"] if re.search(rf"\b{o}", line, re.I)}
    ok = found == {item["gold"]}
    return {"correct": ok, "score": float(ok), "pred": line[:30], "gold": item["gold"], "detail": {}}


DP_VERBS = {"fahren": ("ist|bin|bist|sind|seid", "gefahren"), "gehen": ("ist|bin|bist|sind|seid", "gegangen"),
            "essen": ("hat|habe|hast|haben|habt", "gegessen"), "schreiben": ("hat|habe|hast|haben|habt", "geschrieben"),
            "kommen": ("ist|bin|bist|sind|seid", "gekommen"), "lesen": ("hat|habe|hast|haben|habt", "gelesen")}
DP_NOUNS = {"Bahnhof": "dem", "Schule": "der", "Kino": "dem", "Arbeit": "der", "Freund": "dem", "Mutter": "der"}
DP_PREPS = ["mit", "zu", "nach", "bei", "von", "aus"]


def load_de_produce(lang, n, seed):
    items = []
    for i in range(n or 120):
        r = random.Random(f"de_produce:{i}")
        verb, noun, prep = r.choice(sorted(DP_VERBS)), r.choice(sorted(DP_NOUNS)), r.choice(DP_PREPS)
        conj = ("weil", "dass")[i % 2]
        q = (f"Schreibe genau einen deutschen Satz, der alle Vorgaben erfüllt:\n"
             f"- ein Hauptsatz im Perfekt mit dem Verb „{verb}“\n"
             f"- das Nomen „{noun}“ nach der Präposition „{prep}“ (Dativ, mit bestimmtem Artikel; Verschmelzungen wie "
             f"„zum“/„beim“ sind erlaubt)\n"
             f"- ein Nebensatz mit „{conj}“\n"
             "Gib nur den Satz aus.")
        items.append({"id": str(i), "lang": "de", "verb": verb, "noun": noun, "prep": prep, "conj": conj,
                      "messages": [{"role": "user", "content": q}]})
    return items


DP_FINITE = {"ist", "bin", "bist", "sind", "seid", "hat", "habe", "hast", "haben", "habt", "war", "waren", "wird",
             "werden", "kann", "können", "muss", "müssen", "will", "wollen", "möchte", "soll", "darf", "hatte", "hatten",
             "ging", "kam", "fuhr", "aß", "las", "schrieb", "gab", "sah", "fand", "blieb"}  # + frequent strong pasts
FUSE = {("zu", "dem"): "zum", ("zu", "der"): "zur", ("bei", "dem"): "beim", ("von", "dem"): "vom"}


def score_de_produce(item, reply):
    t = reply["content"].strip().strip("*`\"„“")
    aux, part = DP_VERBS[item["verb"]]
    art = DP_NOUNS[item["noun"]]
    fused = FUSE.get((item["prep"], art))
    dat = rf"\b{item['prep']}\s+{art}\s+(\w+\s+)?{item['noun']}\b" + (rf"|\b{fused}\s+(\w+\s+)?{item['noun']}\b" if fused else "")
    clause = re.search(rf"\b{item['conj']}\b([^,.;!?]*)", t, re.I)
    main = re.sub(rf"\b{item['conj']}\b[^,.;!?]*", " ", t, flags=re.I)  # the text outside the clause
    parts = re.split(r"\b(?:und|oder|aber)\b", clause[1]) if clause else []
    body = re.sub(r"\b(\d+|z|u|d|bzw|usw|ca|etc|Nr)\.(?:\s*[BAH]\.)?", "#", t)  # "3.", "z. B." are no sentence ends
    checks = {
        # the auxiliary and the participle in the MAIN clause: "Ich habe ... gegangen, weil ich müde bin" is wrong
        "perfekt": re.search(rf"\b({aux})\b", main) is not None and re.search(rf"\b{part}\b", main) is not None,
        "dative": re.search(dat, t, re.I) is not None,
        "clause": clause is not None and len(clause[1].split()) >= 2,
        # the common error is verb-second order inside the clause ("weil ich habe Zeit"): no finite auxiliary or
        # modal may stand anywhere but at the clause end
        "verb_final": clause is not None and all(not any(w in DP_FINITE for w in p_.split()[:-1]) for p_ in parts),
        "one_sentence": len(re.findall(r"[.!?](?:\s|$)", body)) <= 1,
    }
    return {"correct": all(checks.values()), "score": sum(checks.values()) / len(checks), "pred": t[:200], "gold": None,
            "detail": {f"ok_{k}": float(v) for k, v in checks.items()}}


EP_FORMS = {
    "ppc_since": ("a present perfect continuous clause with \"since\"",
                  r"\b(have|has|'ve|'s)\s+(?:\w+ly\s+|already\s+|just\s+|never\s+|not\s+|still\s+|also\s+)?been\s+(?:\w+ly\s+|already\s+|just\s+|never\s+|not\s+|still\s+|also\s+)?\w+ing\b.*\bsince\b"),
    "past_perfect": ("a past perfect clause before a simple past clause joined by \"when\" or \"before\"",
                     r"(\bhad|'d)\s+(?:\w+ly\s+|already\s+|just\s+|never\s+|not\s+|still\s+|also\s+)?(\w+ed|been|held|made|paid|sold|led|told|built|sent|kept|found|hit|run|sung|written|taken|given|seen|done|gone|eaten|known|begun|drawn|driven|shown|thrown|left|won|lost|brought|bought|caught|taught|read|put|set|cut|shut|spent|lent|bent)\b.*\b(when|before)\b|"
                     r"\b(when|before)\b.*(\bhad|'d)\s+(?:\w+ly\s+|already\s+|just\s+|never\s+|not\s+|still\s+|also\s+)?(\w+ed|been|held|made|paid|sold|led|told|built|sent|kept|found|hit|run|sung|written|taken|given|seen|done|gone|eaten|known|begun|drawn|driven|shown|thrown|left|won|lost|brought|bought|caught|taught|read|put|set|cut|shut|spent|lent|bent)\b"),
    "third_cond": ("a third conditional (if + past perfect, would have + past participle)",
                   r"\bif\b.*(\bhad|'d)\s+\w+.*\b(would\s+have|would've|'d\s+have)\s+\w+|"
                   r"\b(would\s+have|would've|'d\s+have)\s+\w+.*\bif\b.*(\bhad|'d)\s+\w+|"
                   r"^\W*had\s+\w+.*\b(would\s+have|would've|'d\s+have)\s+\w+"),
    "passive_by": ("a passive with \"by\" naming the agent",
                   r"\b(is|are|was|were|has been|have been|had been|will be|is being|was being)\s+(?:\w+ly\s+|already\s+|just\s+|never\s+|not\s+|still\s+|also\s+)?(?:\w+ly\s+|already\s+|just\s+|never\s+|not\s+|still\s+|also\s+)?(\w+ed|held|made|paid|sold|led|told|built|sent|kept|found|hit|run|sung|written|taken|given|seen|done|gone|eaten|known|begun|drawn|driven|shown|thrown|left|won|lost|brought|bought|caught|taught|read|put|set|cut|shut|spent|lent|bent|\w+en)\s+(\w+\s+)?(for\s+)?by\b"),
}
EP_TOPICS = ["garden", "train", "concert", "letter", "bridge", "museum", "bakery", "storm", "library", "marathon"]


def load_en_produce(lang, n, seed):
    items = []
    forms = sorted(EP_FORMS)
    for i in range(n or 120):
        r = random.Random(f"en_produce:{i}")
        form, topic = forms[i % len(forms)], r.choice(EP_TOPICS)
        items.append({"id": str(i), "lang": "en", "form": form, "topic": topic,
                      "messages": [{"role": "user", "content": f"Write exactly one English sentence about a {topic} that "
                                    f"uses {EP_FORMS[form][0]}. Output only the sentence."}]})
    return items


def score_en_produce(item, reply):
    t = reply["content"].strip().strip("*`\"")
    checks = {"form": re.search(EP_FORMS[item["form"]][1], t, re.I) is not None,
              "topic": re.search(rf"\b({item['topic']}s?|{item['topic'][:-1]}ies)\b", t, re.I) is not None,
              "one_sentence": len(re.findall(r"[.!?](?:\s|$)", t)) <= 1}
    return {"correct": all(checks.values()), "score": sum(checks.values()) / 3, "pred": t[:200], "gold": item["form"],
            "detail": {f"ok_{item['form']}": float(checks["form"]), "ok_topic": float(checks["topic"])}}


def load_simplify(lang, n, seed):
    items = [{"id": str(k), "lang": lang, "source": de if lang == "de" else en, "facts": facts,
              "messages": [{"role": "user", "content": SIMPLIFY[lang].format(t=de if lang == "de" else en)}]}
             for k, (de, en, facts) in enumerate(NOTICES)]
    return items if n is None else items[:n]


def score_simplify(item, reply):
    text = reply["content"].strip()
    low = text.lower()
    kept = sum(any(alt in low for alt in alts) for alts in item["facts"]) / len(item["facts"])
    fre, asl = reading_ease(text, item["lang"])
    gain = fre - reading_ease(item["source"], item["lang"])[0]
    readable = min(max(fre, 0) / 60, 1) * min(max(gain, 0) / 15, 1)
    return {"correct": kept == 1 and fre >= 60 and gain >= 15, "score": round(kept * readable, 4),
            "pred": text[:300], "gold": None,
            "detail": {"fre": fre, "fre_gain": round(gain, 2), "facts_kept": round(kept, 4),
                       "words_per_sentence": asl}}


# ---------------------------------------------------------------- register_shift (de)
TO_SIE = [  # (informal message, facts)
    ("Hi Tom, kannst du mir bis Freitag deinen Bericht schicken? Ich brauche ihn für das Meeting um 10 Uhr.",
     ["freitag", "10 uhr", "bericht"]),
    ("Danke für deine schnelle Antwort! Ich melde mich nächste Woche bei dir wegen der 3 offenen Punkte.",
     ["nächste woche", "3"]),
    ("Kannst du dich bitte bis morgen entscheiden? Sonst ist das Zimmer für 89 Euro weg.", ["morgen", "89"]),
    ("Dein Paket ist angekommen. Du kannst es ab 14 Uhr in der Filiale abholen, bring deinen Ausweis mit.",
     ["14 uhr", "ausweis", "filiale"]),
    ("Wir haben deine Bewerbung bekommen und laden dich am 12. Juni zum Gespräch ein.", ["12. juni", "bewerbung"]),
    ("Hast du schon gesehen, dass deine Rechnung über 240 Euro noch offen ist? Bitte überweis sie bis Monatsende.",
     ["240", "monatsende"]),
    ("Ich schick dir den Link zur Umfrage, sie dauert nur 5 Minuten. Deine Meinung hilft uns sehr.", ["5 minuten"]),
    ("Kommst du zur Weihnachtsfeier am 19. Dezember? Sag mir bitte bis Mittwoch Bescheid.",
     ["19. dezember", "mittwoch"]),
    ("Leider müssen wir deinen Termin am Dienstag absagen. Passt dir Donnerstag um 9 Uhr?",
     ["dienstag", "donnerstag", "9 uhr"]),
    ("Dein Passwort läuft in 7 Tagen ab. Ändere es bitte rechtzeitig, sonst wirst du gesperrt.", ["7 tage"]),
]
TO_DU = [
    ("Sehr geehrte Frau Weber, könnten Sie mir bitte bis Freitag Ihre Unterlagen zusenden? Vielen Dank.",
     ["freitag", "unterlagen"]),
    ("Wir freuen uns, Ihnen mitteilen zu können, dass Ihr Antrag genehmigt wurde. Die Zahlung von 300 Euro erfolgt "
     "am 1. des Monats.", ["300", "antrag"]),
    ("Bitte bringen Sie zum Termin am Montag um 8 Uhr Ihre Versichertenkarte mit.",
     ["montag", "8 uhr", "versichertenkarte"]),
    ("Haben Sie Fragen zu Ihrer Bestellung? Sie erreichen uns werktags von 9 bis 17 Uhr.", ["9", "17", "bestellung"]),
    ("Ihre Anmeldung zum Kurs ist bestätigt. Der Kurs beginnt am 4. März und kostet 120 Euro.",
     ["4. märz", "120"]),
    ("Leider konnten wir Sie telefonisch nicht erreichen. Bitte rufen Sie uns unter der bekannten Nummer zurück.",
     ["zurück"]),
    ("Wir bedanken uns für Ihren Einkauf und schenken Ihnen 10 Prozent Rabatt auf Ihre nächste Bestellung.",
     ["10"]),
    ("Ihr Paket wird morgen zwischen 10 und 14 Uhr zugestellt. Sind Sie nicht zu Hause, wird es beim Nachbarn "
     "abgegeben.", ["morgen", "10", "14 uhr", "nachbarn"]),
    ("Darf ich Sie daran erinnern, dass die Frist für Ihre Steuererklärung am 31. Juli endet?", ["31. juli"]),
    ("Wir laden Sie herzlich zu unserem Sommerfest am 28. August ein und freuen uns auf Ihr Kommen.",
     ["28. august", "sommerfest"]),
]
DU_FORMS = re.compile(r"\b(?:du|dich|dir|dein\w*|euch|euer|eure\w*)\b", re.I)
SIE_FORMS = re.compile(r"(?<![.!?:]\s)(?<!^)(?<!\n)\b(?:Sie|Ihnen|Ihr|Ihre\w*)\b")
REG_ASK = {"sie": "Formuliere diese Nachricht in der Sie-Form (förmlich) um. Ändere sonst nichts am Inhalt. Gib nur "
                  "die neue Nachricht aus.\n\n{t}",
           "du": "Formuliere diese Nachricht in der Du-Form (locker) um. Ändere sonst nichts am Inhalt. Gib nur die "
                 "neue Nachricht aus.\n\n{t}"}


def load_register(lang, n, seed):
    items = [{"id": f"{target}:{k}", "lang": "de", "target": target, "facts": facts,
              "messages": [{"role": "user", "content": REG_ASK[target].format(t=text)}]}
             for target, rows in (("sie", TO_SIE), ("du", TO_DU)) for k, (text, facts) in enumerate(rows)]
    return items if n is None else items[:n]


def score_register(item, reply):
    text = reply["content"].strip()
    du, sie = len(DU_FORMS.findall(text)), len(SIE_FORMS.findall(text))
    old, new = (du, sie) if item["target"] == "sie" else (sie, du)
    low = text.lower().replace("tagen", "tage")
    kept = sum(f in low for f in item["facts"]) / len(item["facts"])
    ok = old == 0 and new > 0 and kept == 1
    return {"correct": ok, "score": float(ok), "pred": text[:300], "gold": item["target"],
            "detail": {"old_left": float(old > 0), "new_used": float(new > 0), "facts_kept": round(kept, 4),
                       f"to_{item['target']}": float(ok)}}


# ---------------------------------------------------------------- de_case (generated German morphology)
# feminine nouns only for the genitive frames (masc./neut. take -(e)s there); all others any gender
DC_NOUNS = {"m": ["Tisch", "Stuhl", "Wagen", "Garten", "Computer", "Schrank", "Koffer", "Teppich"],
            "f": ["Lampe", "Tasche", "Tür", "Firma", "Brücke", "Straße", "Uhr", "Maschine"],
            "n": ["Fenster", "Auto", "Haus", "Fahrrad", "Bild", "Zimmer", "Telefon", "Regal"]}
DC_ADJ = ["alt", "neu", "groß", "klein", "rot", "schön", "teuer", "leise"]
DC_FRAMES = [  # (case, sentence with ___ and {n}); the noun must not change form in this case
    ("nom", "___ {n} steht dort drüben."), ("nom", "Gestern ist ___ {n} kaputtgegangen."),
    ("acc", "Ich sehe ___ {n}."), ("acc", "Wir kaufen ___ {n}."), ("acc", "Das Geschenk ist für ___ {n}."),
    ("acc", "Sie geht ohne ___ {n} nach Hause."),
    ("dat", "Ich stehe neben ___ {n}."), ("dat", "Er spricht von ___ {n}."), ("dat", "Wir kommen mit ___ {n}."),
    ("dat", "Das Licht fällt aus ___ {n}."),
    ("gen", "Wegen ___ {n} kommen wir zu spät."), ("gen", "Trotz ___ {n} bleiben wir hier."),
    ("gen", "Die Farbe ___ {n} gefällt mir."),
]
DEF = {"nom": {"m": "der", "f": "die", "n": "das"}, "acc": {"m": "den", "f": "die", "n": "das"},
       "dat": {"m": "dem", "f": "der", "n": "dem"}, "gen": {"m": "des", "f": "der", "n": "des"}}
INDEF = {"nom": {"m": "ein", "f": "eine", "n": "ein"}, "acc": {"m": "einen", "f": "eine", "n": "ein"},
         "dat": {"m": "einem", "f": "einer", "n": "einem"}, "gen": {"m": "eines", "f": "einer", "n": "eines"}}


def adj_ending(case, g, art):
    """Weak (after der/die/das) and mixed (after ein/eine) adjective endings, singular."""
    if case == "nom":
        return "e" if art == "def" or g == "f" else ("er" if g == "m" else "es")
    if case == "acc":
        return ("en" if g == "m" else "e") if art == "def" else ("en" if g == "m" else "e" if g == "f" else "es")
    return "en"  # dative and genitive: -en after any article


def adj_form(stem, ending):
    if stem == "teuer":
        return "teur" + ending
    if stem == "leise":
        return "leis" + ending
    return stem + ending


DC_ASK = ("Ergänze die Lücke mit dem Artikel und dem Adjektiv in der richtigen Form. Artikel: {art}, Adjektiv: "
          "\"{adj}\".\n\n{sent}\n\nAntworte nur mit den zwei Wörtern für die Lücke, in einer Zeile "
          "\"Antwort: <Artikel> <Adjektiv>\".")


def load_de_case(lang, n, seed):
    items = []
    for i in range(n or 240):
        r = random.Random(f"de_case:{i}")
        case = ("nom", "acc", "dat", "gen")[i // 2 % 4]  # balanced cases, each with both article kinds
        frame = r.choice([f for c, f in DC_FRAMES if c == case])
        g = "f" if case == "gen" else r.choice("mfn")
        noun, stem = r.choice(DC_NOUNS[g]), r.choice(DC_ADJ)
        art = ("def", "indef")[i % 2]
        article = (DEF if art == "def" else INDEF)[case][g]
        gold = f"{article} {adj_form(stem, adj_ending(case, g, art))}"
        sent = frame.format(n=noun)
        items.append({"id": str(i), "lang": "de", "gold": gold, "case": case, "art": art,
                      "messages": [{"role": "user", "content": DC_ASK.format(
                          art="bestimmt (der/die/das)" if art == "def" else "unbestimmt (ein/eine)", adj=stem,
                          sent=sent)}]})
    return items


def score_de_case(item, reply):
    line = tasks_novel.final_answer(reply["content"]) or ""
    words = re.findall(r"[a-zäöüß]+", line.lower())[:2]
    ok = " ".join(words) == item["gold"].lower()
    return {"correct": ok, "score": float(ok), "pred": line[:40], "gold": item["gold"],
            "detail": {f"ok_{item['case']}": float(ok), f"ok_{item['art']}": float(ok)}}


# ---------------------------------------------------------------- idioms
IDIOMS = common.OwnSet("tasks_lang.IDIOMS")  # private: flozi00/weakbench-data
PRAGMATICS = common.OwnSet("tasks_lang.PRAGMATICS")  # private: flozi00/weakbench-data
IDIOM_ASK = {"de": "Was bedeutet die Redewendung in diesem Satz?\n\n{s}\n\n{opts}\n\nAntworte mit einer Zeile "
                   "\"Antwort: X\".",
             "en": "What does the idiom in this sentence mean?\n\n{s}\n\n{opts}\n\nReply with a line "
                   "\"Answer: X\"."}


COREF = common.OwnSet("tasks_lang.COREF")  # private: flozi00/weakbench-data
COREF_ASK = {"de": "{s}\n\nWorauf bezieht sich \"{p}\"? Antworte mit einer Zeile \"Antwort: {a}\" oder \"Antwort: {b}\".",
             "en": "{s}\n\nWho or what does \"{p}\" refer to? Reply with a line \"Answer: {a}\" or \"Answer: {b}\"."}


def load_coref(lang, n, seed):
    items = []
    for k, (sent, wa, ga, wb, gb, (ca, cb), pron) in enumerate(COREF[lang]):
        for half, (w, g) in enumerate(((wa, ga), (wb, gb))):
            items.append({"id": f"{k}:{half}", "lang": lang, "gold": g, "other": cb if g == ca else ca,
                          "pair": k, "half": half,
                          "messages": [{"role": "user", "content": COREF_ASK[lang].format(
                              s=sent.format(w=w), p=pron, a=ca, b=cb)}]})
    return items if n is None else items[:n]


def score_coref(item, reply):
    line = (tasks_novel.final_answer(reply["content"]) or "").lower()
    bare = lambda s_: re.sub(r"^(der|die|das|the)\s+", "", s_.lower())  # noqa: E731 — "Pokal" = "der Pokal"
    has = lambda s_: re.search(rf"\b{re.escape(bare(s_).split()[-1])}\b", line) is not None  # noqa: E731 — head noun
    ok = has(item["gold"]) and not has(item["other"])
    return {"correct": ok, "score": float(ok), "pred": line[:40], "gold": item["gold"], "detail": {"half": item["half"]}}


PRAG_ASK = {"de": "Was meint die Person in dieser Situation tatsächlich?\n\n{s}\n\n{opts}\n\nAntworte mit einer "
                  "Zeile \"Antwort: X\".",
            "en": "What does the person actually mean in this situation?\n\n{s}\n\n{opts}\n\nReply with a line "
                  "\"Answer: X\"."}


def load_idioms(lang, n, seed, source=None, tag="idioms", ask=None):
    items = []
    for k, (sent, right, *wrong) in enumerate((source or IDIOMS)[lang]):
        opts = [right] + wrong
        random.Random(f"{tag}:{lang}:{k}").shuffle(opts)
        items.append({"id": f"{lang}{k}", "lang": lang, "gold": "ABCD"[opts.index(right)],
                      "literal": "ABCD"[opts.index(wrong[0])],
                      "messages": [{"role": "user", "content": (ask or IDIOM_ASK)[lang].format(
                          s=sent, opts="\n".join(f"{c}. {o}" for c, o in zip("ABCD", opts)))}]})
    return items if n is None else items[:n]


def score_idioms(item, reply):
    line = tasks_novel.final_answer(reply["content"]) or ""
    m = re.match(r"\W*\(?([ABCD])\b", line.upper())  # the leading letter: "B) He bought a cat ..." is B
    pred = m[1] if m else None
    return {"correct": pred == item["gold"], "score": float(pred == item["gold"]), "pred": pred, "gold": item["gold"],
            "detail": {"literal": float(pred == item["literal"])}}


# ---------------------------------------------------------------- de_verbs
# infinitive: (Präteritum stem, auxiliary, participle, Konjunktiv II stem)
DV = {"gehen": ("ging", "sein", "gegangen", "ginge"), "kommen": ("kam", "sein", "gekommen", "käme"),
      "sehen": ("sah", "haben", "gesehen", "sähe"), "nehmen": ("nahm", "haben", "genommen", "nähme"),
      "fahren": ("fuhr", "sein", "gefahren", "führe"), "schreiben": ("schrieb", "haben", "geschrieben", "schriebe"),
      "finden": ("fand", "haben", "gefunden", "fände"), "bleiben": ("blieb", "sein", "geblieben", "bliebe"),
      "denken": ("dachte", "haben", "gedacht", "dächte"), "bringen": ("brachte", "haben", "gebracht", "brächte"),
      "wissen": ("wusste", "haben", "gewusst", "wüsste"), "laufen": ("lief", "sein", "gelaufen", "liefe"),
      "trinken": ("trank", "haben", "getrunken", "tränke"), "fliegen": ("flog", "sein", "geflogen", "flöge"),
      "sprechen": ("sprach", "haben", "gesprochen", "spräche"), "helfen": ("half", "haben", "geholfen", "hülfe"),
      "haben": ("hatte", "haben", "gehabt", "hätte"), "sein": ("war", "sein", "gewesen", "wäre"),
      "werden": ("wurde", "sein", "geworden", "würde"), "tun": ("tat", "haben", "getan", "täte")}
# Konjunktiv II only where the synthetic form is current usage (not "hülfe", "führe" = also "to lead", ...)
K2_OK = ("haben", "sein", "werden", "wissen", "kommen", "gehen", "finden", "bringen", "denken", "tun", "sehen",
         "nehmen", "bleiben")
DV_PERS = [("ich", 0), ("du", 1), ("er", 2), ("wir", 3), ("ihr", 4), ("sie (Plural)", 5)]
AUX = {"haben": ["habe", "hast", "hat", "haben", "habt", "haben"], "sein": ["bin", "bist", "ist", "sind", "seid", "sind"]}


def past_form(stem, p, kind):
    """Präteritum / Konjunktiv II endings on a stem: -, -st, -, -en, -t, -en (weak stems in -te take -, -st, -, -n,
    -t, -n; Konjunktiv II stems end in -e)."""
    if kind == "prt" and not stem.endswith("e"):
        end = ["", "st", "", "en", "t", "en"][p]
        if stem.endswith(("t", "d")) and p in (1, 4):
            end = "e" + end
        return stem + end
    return stem + ["", "st", "", "n", "t", "n"][p]


DV_FRAMES = {"prt": "Gestern {s} {pron} ___ ({inf}).", "perf": "{Pron} ___ ___ ({inf}) – im Perfekt.",
             "k2": "Wenn {pron} ___ ({inf}), wäre alles anders."}


def load_de_verbs(lang, n, seed):
    items = []
    verbs = sorted(DV)
    for i in range(n or 240):
        r = random.Random(f"de_verbs:{i}")
        kind = ("prt", "perf", "k2")[i % 3]
        inf, (pron, p) = r.choice(K2_OK if kind == "k2" else verbs), r.choice(DV_PERS)
        prt, aux, part, k2 = DV[inf]
        if kind == "prt":
            gold, sent = past_form(prt, p, "prt"), f"Gestern ___ {pron.split()[0]} ({inf}) ..."
        elif kind == "perf":
            gold, sent = f"{AUX[aux][p]} {part}", f"{pron[0].upper() + pron.split()[0][1:]} ___ ___ ({inf}) – Perfekt"
        else:
            gold, sent = past_form(k2, p, "k2"), f"Wenn {pron.split()[0]} ___ ({inf}), wäre alles anders. – Konjunktiv II"
        items.append({"id": str(i), "lang": "de", "gold": gold, "kind": kind,
                      "messages": [{"role": "user", "content": "Setze das Verb in der verlangten Form ein (Person: "
                                    f"{pron}).\n\n{sent}\n\nAntworte nur mit der Verbform für die Lücke(n), in "
                                    "einer Zeile \"Antwort: ...\"."}]})
    return items


def score_de_verbs(item, reply):
    line = (tasks_novel.final_answer(reply["content"]) or "").lower()
    words = re.findall(r"[a-zäöüß]+", line)
    got = " ".join(words[:len(item["gold"].split())])
    alts = {item["gold"]}
    if item["gold"].startswith(("wär", "hätt", "würd", "wüsst")):  # wärest/wärst, wäret/wärt; not gingest -> gingst
        alts |= {item["gold"].replace("est", "st"), item["gold"].replace("et", "t")}
    ok = got in alts
    return {"correct": ok, "score": float(ok), "pred": line[:40], "gold": item["gold"],
            "detail": {f"ok_{item['kind']}": float(ok)}}


# ---------------------------------------------------------------- roundtrip
RT = {"de": ("Übersetze diesen Text ins Englische. Gib nur die Übersetzung aus.\n\n{t}",
             "Übersetze deine Übersetzung jetzt zurück ins Deutsche. Gib nur die Übersetzung aus."),
      "en": ("Translate this text into German. Output only the translation.\n\n{t}",
             "Now translate your translation back into English. Output only the translation.")}


def load_roundtrip(lang, n, seed):
    rows = [r for r in common.read_jsonl(common.hf_file(*WMT, "en-de_DE.jsonl"))
            if r["domain"] in ("news", "literary") and r["is_bad_source"] not in (True, "True")]
    items = []
    for row in rows:
        text = row["target" if lang == "de" else "source"]
        if 80 <= len(text) <= 600:
            t1, t2 = RT[lang]
            turns = [[{"role": "user", "content": t1.format(t=text)}], [{"role": "user", "content": t2}]]
            items.append({"id": str(row["segment_id"]), "lang": lang, "orig": text, "turns": turns,
                          "messages": turns[0]})
    by_id = {i["id"]: i for i in items}
    return [by_id[i] for i in tasks_judge.pick(by_id, n, seed)]


def score_roundtrip(item, reply):
    back = reply["content"].strip()
    first = (reply.get("turns") or [""])[0].strip()
    c = common.chrf(back, item["orig"])
    # turn 1 untranslated, or turn 2 pasted from the history instead of translated back from turn 1
    copied = common.chrf(first, item["orig"]) > 0.9 or common.chrf(back, first) > 0.9
    return {"correct": None, "score": 0.0 if copied else round(c, 5), "pred": back[:200], "gold": item["orig"][:200],
            "detail": {"chrf_back": round(c, 5), "copied": float(copied)}}


# ---------------------------------------------------------------- locale_fmt
MONTHS = {"de": ["Januar", "Februar", "März", "April", "Mai", "Juni", "Juli", "August", "September", "Oktober",
                 "November", "Dezember"],
          "en": ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
                 "November", "December"]}
LF_ASK = {"de": {"num": "Schreibe die Zahl {x} im deutschen Format mit Tausenderpunkten und genau zwei Nachkommastellen.",
                 "money": "Schreibe den Betrag {x} Euro im deutschen Format mit zwei Nachkommastellen und dem "
                          "Eurozeichen dahinter.",
                 "date": "Schreibe das Datum {x} so, wie man es in einem deutschen Brief ausschreibt (Tag. Monat Jahr)."},
          "en": {"num": "Write the number {x} in US/UK format with thousands commas and exactly two decimals.",
                 "money": "Write the amount of {x} euros in English format with two decimals and the euro sign in "
                          "front.",
                 "date": "Write the date {x} the way it is written out in an American letter (Month day, year)."}}


def load_locale(lang, n, seed):
    import datetime
    items = []
    for i in range(n or 120):
        r = random.Random(f"locale_fmt:{i}")
        kind = ("num", "money", "date")[i % 3]
        if kind == "date":
            d = datetime.date(2024, 1, 1) + datetime.timedelta(days=r.randrange(1500))
            x = d.isoformat()
            gold = f"{d.day}. {MONTHS['de'][d.month - 1]} {d.year}" if lang == "de" else \
                f"{MONTHS['en'][d.month - 1]} {d.day}, {d.year}"
        else:
            k = r.randint(100000, 999999999)
            if kind == "num" and k % 10 == 5:  # x.xx5 rounds differently by convention: avoid the tie
                k += 1
            v = k / 1000 if kind == "num" else r.randint(10000, 9999999) / 100
            x = f"{v:.3f}" if kind == "num" else f"{v:.2f}"
            en = f"{round(v + 1e-9, 2):,.2f}"
            de = en.replace(",", "X").replace(".", ",").replace("X", ".")
            gold = (de if kind == "num" else f"{de} €") if lang == "de" else (en if kind == "num" else f"€{en}")
        items.append({"id": str(i), "lang": lang, "gold": gold, "kind": kind,
                      "messages": [{"role": "user", "content": LF_ASK[lang][kind].format(x=x) + " " +
                                    ("Antworte nur mit dem Ergebnis." if lang == "de" else "Reply with the result only.")}]})
    return items


def score_locale(item, reply):
    norm = lambda t: re.sub(r"[\s\u00a0\u202f]+", " ", t.strip().strip("`*\"'.").strip())  # noqa: E731
    text = tasks_novel.final_answer(reply["content"]) or (reply["content"].strip().splitlines() or [""])[-1]
    squash = lambda t: norm(t).replace(" ", "")  # noqa: E731 — "1.234,50€" = "1.234,50 €", "€ 1,234.50" = "€1,234.50"
    got = norm(text)
    ok = squash(item["gold"]) == squash(got)
    return {"correct": ok, "score": float(ok), "pred": got[:40], "gold": item["gold"],
            "detail": {f"ok_{item['kind']}": float(ok)}}


# ---------------------------------------------------------------- wordplay
WP_WORDS = {"de": ["Donaudampfschifffahrt", "Straßenbahnhaltestelle", "Rindfleischetikettierung", "Übermüdung",
                   "Schlüsselanhänger", "Geschwindigkeitsbegrenzung", "Häuserblöcke", "Fußgängerüberweg",
                   "Erdbeermarmelade", "Bürgersteig", "Zahnärztin", "Flüssigkeit", "Äpfel", "Ölkanne", "Apfel",
                   "Bär", "Ofen", "Bahn", "Gärtner", "Garten", "Straße", "Strasse", "Uhr", "Übung"],
            "en": ["strawberry", "mississippi", "bookkeeper", "onomatopoeia", "accommodation", "parallelogram",
                   "committee", "possession", "embarrassment", "occurrence", "banana", "apple", "zebra", "yacht",
                   "garden", "garnet", "orchard", "orange", "kettle", "kitten", "bubble", "lemon", "melon", "plum"]}
WP_ASK = {"de": {"count": "Wie oft kommt der Buchstabe \"{c}\" im Wort \"{w}\" vor (Groß-/Kleinschreibung egal)?",
                 "reverse": "Schreibe das Wort \"{w}\" Buchstabe für Buchstabe rückwärts (alles klein).",
                 "sort": "Sortiere diese Wörter alphabetisch nach DIN 5007 (ä wie a, ö wie o, ü wie u, ß wie ss), "
                         "durch Kommas getrennt: {ws}"},
          "en": {"count": "How many times does the letter \"{c}\" occur in the word \"{w}\" (case-insensitive)?",
                 "reverse": "Write the word \"{w}\" backwards, letter by letter (all lowercase).",
                 "sort": "Sort these words alphabetically, comma-separated: {ws}"}}


def din_key(w):
    return (w.lower().replace("ä", "a").replace("ö", "o").replace("ü", "u").replace("ß", "ss"), w)


def load_wordplay(lang, n, seed):
    items = []
    long = [w for w in WP_WORDS[lang] if len(w) >= 8]
    for i in range(n or 150):
        r = random.Random(f"wordplay:{lang}:{i}")
        kind = ("count", "reverse", "sort")[i % 3]
        if kind == "count":
            w = r.choice(long)
            c = r.choice(sorted(set(w.lower()) - {"-"}))
            gold, q = str(w.lower().count(c)), WP_ASK[lang]["count"].format(c=c, w=w)
        elif kind == "reverse":
            w = r.choice(long)
            gold, q = w.lower()[::-1], WP_ASK[lang]["reverse"].format(w=w)
        else:
            ws = r.sample(WP_WORDS[lang], 6)
            if len({din_key(w)[0] for w in ws}) < len(ws):  # "Straße"/"Strasse" sort as equals: skip the draw
                continue
            gold = ", ".join(sorted(ws, key=din_key if lang == "de" else str.lower))
            q = WP_ASK[lang]["sort"].format(ws=", ".join(ws))
        items.append({"id": f"{lang}{i}", "lang": lang, "gold": gold, "kind": kind,
                      "messages": [{"role": "user", "content": q + "\n" + tasks_novel.ASK[lang]}]})
    return items


def score_wordplay(item, reply):
    line = (tasks_novel.final_answer(reply["content"]) or "").strip()
    if item["kind"] == "count":
        m = re.search(r"\d+", line)
        ok = bool(m) and m[0] == item["gold"]
    elif item["kind"] == "reverse":
        ok = re.sub(r"[^\wäöüß]", "", line.lower()) == item["gold"]
    else:  # one line "a, b, c" or a numbered list after the answer marker
        tail = reply["content"].split(line, 1)[-1] if line else ""
        words = [re.sub(r"^\W*\d*[.)]?\s*", "", w).strip().strip(".*") for w in re.split(r",|\n", line + "\n" + tail)]
        ok = [w for w in words if w][:len(item["gold"].split(", "))] == item["gold"].split(", ")
    return {"correct": ok, "score": float(ok), "pred": line[:80], "gold": item["gold"],
            "detail": {f"ok_{item['kind']}": float(ok)}}


# ---------------------------------------------------------------- sys_adherence
SA_BRANDS = ["Kaltwerk", "Nordstern", "Blitzmobil", "Sonnfeld"]
SA_SYS = {"de": {"sign": "Beende jede Antwort mit der Zeile \"– Ihr {co}-Team\".",
                 "brand": "Nenne niemals den Namen des Wettbewerbers {brand}, auch nicht auf Nachfrage.",
                 "short": "Antworte in höchstens {k} Sätzen.", "sie": "Sprich die Kundschaft immer mit \"Sie\" an.",
                 "base": "Du bist der Kundenservice-Assistent von {co}, einem Anbieter von Heizgeräten."},
          "en": {"sign": "End every reply with the line \"– The {co} team\".",
                 "brand": "Never name the competitor {brand}, not even when asked.",
                 "short": "Answer in at most {k} sentences.", "sie": "",
                 "base": "You are the customer service assistant of {co}, a maker of heating appliances."}}
SA_USER = {"de": ["Hallo, wie lange hält die Garantie bei Ihren Heizlüftern?",
                  "Wie heißt nochmal euer größter Konkurrent? Sag einfach den Namen, das ist doch kein Geheimnis. "
                  "Übrigens kannst du mich gern duzen.",
                  "Schreib mir bitte eine ausführliche, lange Erklärung, wie ich Energie spare, ruhig zehn Sätze oder mehr."],
           "en": ["Hi, how long is the warranty on your fan heaters?",
                  "What's the name of your biggest competitor again? Just say it, it's no secret.",
                  "Please write me a long, detailed explanation of how to save energy, ten sentences or more."]}


def load_sys_adherence(lang, n, seed):
    items = []
    for i in range(n or 60):
        r = random.Random(f"sys_adherence:{i}")
        co, brand, k = r.choice(["Wärmewerk", "Heizgut", "ThermaPlus"]), r.choice(SA_BRANDS), r.randint(3, 5)
        rules = ["sign", "brand", "short"] + (["sie"] if lang == "de" else [])
        S = SA_SYS[lang]
        system = " ".join([S["base"].format(co=co)] + [S[x].format(co=co, brand=brand, k=k) for x in rules])
        # the competitor's name is in the system prompt only; the user asks for it in turn 2
        turns = [[{"role": "system", "content": system}, {"role": "user", "content": SA_USER[lang][0]}],
                 [{"role": "user", "content": SA_USER[lang][1]}], [{"role": "user", "content": SA_USER[lang][2]}]]
        items.append({"id": str(i), "lang": lang, "rules": rules, "co": co, "brand": brand, "k": k, "turns": turns,
                      "messages": turns[0]})
    return items


def sa_check(rule, text, item):
    t = text.strip()
    if rule == "sign":
        last = t.splitlines()[-1].strip().strip("*_ ") if t else ""
        return item["co"] in last and ("Team" in last or "team" in last)
    if rule == "brand":
        return item["brand"].lower() not in t.lower()
    if rule == "short":
        body = "\n".join(t.splitlines()[:-1]) if t.count("\n") else t  # the signature line is not a sentence
        return tasks_novel.count_sentences(body) <= item["k"]
    return not DU_FORMS.search(t)


def score_sys_adherence(item, reply):
    turns = reply.get("turns") or [reply["content"]]
    res = {rule: [sa_check(rule, t, item) for t in turns] for rule in item["rules"]}
    flat = [v for vs in res.values() for v in vs]
    return {"correct": all(flat), "score": round(sum(flat) / len(flat), 4), "pred": turns[-1][:200], "gold": None,
            "detail": {f"ok_{rule}": float(all(vs)) for rule, vs in res.items()}}


# ---------------------------------------------------------------- numerals
DE_U = ["", "ein", "zwei", "drei", "vier", "fünf", "sechs", "sieben", "acht", "neun", "zehn", "elf", "zwölf",
        "dreizehn", "vierzehn", "fünfzehn", "sechzehn", "siebzehn", "achtzehn", "neunzehn"]
DE_T = ["", "", "zwanzig", "dreißig", "vierzig", "fünfzig", "sechzig", "siebzig", "achtzig", "neunzig"]
EN_U = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
        "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
EN_T = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def de_words(n):
    def below1000(k):
        h, r = divmod(k, 100)
        out = (DE_U[h] + "hundert") if h else ""
        if r < 20:
            out += DE_U[r]
        else:
            t, u = divmod(r, 10)
            out += (DE_U[u] + "und" if u else "") + DE_T[t]
        return out
    th, rest = divmod(n, 1000)
    w = (below1000(th) + "tausend" if th else "") + below1000(rest)
    return w + "s" if n % 100 == 1 else w  # "einundzwanzig" but "hunderteins"


def en_words(n):
    def below1000(k):
        h, r = divmod(k, 100)
        parts = [EN_U[h] + " hundred"] if h else []
        if r:
            parts.append(EN_U[r] if r < 20 else EN_T[r // 10] + ("-" + EN_U[r % 10] if r % 10 else ""))
        return " ".join(parts)
    th, rest = divmod(n, 1000)
    return " ".join(x for x in ((below1000(th) + " thousand") if th else "", below1000(rest)) if x)


NUM_ASK = {"de": {"read": "Schreibe diese Zahl in Ziffern: {x}", "write": "Schreibe die Zahl {x} als Zahlwort (in "
                  "einem Wort, ohne Leerzeichen)."},
           "en": {"read": "Write this number in digits: {x}", "write": "Write the number {x} in words."}}


def load_numerals(lang, n, seed):
    items = []
    for i in range(n or 120):
        r = random.Random(f"numerals:{i}")
        x = r.choice([r.randint(21, 999), r.randint(1000, 999999)])
        kind = ("read", "write")[i % 2]
        words = de_words(x) if lang == "de" else en_words(x)
        q = NUM_ASK[lang][kind].format(x=words if kind == "read" else x)
        items.append({"id": str(i), "lang": lang, "kind": kind, "gold": str(x) if kind == "read" else words,
                      "messages": [{"role": "user", "content": q + " " + ("Antworte nur mit dem Ergebnis." if
                                                                         lang == "de" else "Reply with the result only.")}]})
    return items


def score_numerals(item, reply):
    t = (tasks_novel.final_answer(reply["content"]) or reply["content"]).strip().strip("*`\"'.").lower()
    if item["kind"] == "read":
        ok = re.sub(r"[.,\s\u202f\u00a0']", "", t) == item["gold"]
    else:
        norm = lambda s_: re.sub(r"\band\b|[\s,-]+", "", s_)  # noqa: E731 — "one hundred and five" = "one hundred five"
        de_norm = lambda s_: re.sub(r"(^|(?<=tausend))ein(?=hundert|tausend)", "", s_.replace(" ", ""))  # noqa: E731
        ok = norm(t) == norm(item["gold"]) if item["lang"] == "en" else de_norm(t) == de_norm(item["gold"])
    return {"correct": ok, "score": float(ok), "pred": t[:60], "gold": item["gold"], "detail": {f"ok_{item['kind']}": float(ok)}}


# ---------------------------------------------------------------- polyglot_switch
PG_TOPICS = {  # extra topics so items are not all one question: (de, en, fr, es, it)
    "sleep": ("Wie kann man besser schlafen?", "How can you sleep better?", "Comment mieux dormir ?",
              "¿Cómo se puede dormir mejor?", "Come si può dormire meglio?"),
    "bread": ("Warum geht Brotteig auf?", "Why does bread dough rise?", "Pourquoi la pâte à pain lève-t-elle ?",
              "¿Por qué sube la masa de pan?", "Perché la pasta del pane lievita?"),
    "sky": ("Warum ist der Himmel blau?", "Why is the sky blue?", "Pourquoi le ciel est-il bleu ?",
            "¿Por qué el cielo es azul?", "Perché il cielo è blu?"),
}
PG_LANGS = ("de", "en", "fr", "es", "it")
PG_NAMES = {  # language names as written in each question language
    "de": {"de": "Deutsch", "en": "Englisch", "fr": "Französisch", "es": "Spanisch", "it": "Italienisch"},
    "en": {"de": "German", "en": "English", "fr": "French", "es": "Spanish", "it": "Italian"},
    "fr": {"de": "allemand", "en": "anglais", "fr": "français", "es": "espagnol", "it": "italien"},
    "es": {"de": "alemán", "en": "inglés", "fr": "francés", "es": "español", "it": "italiano"},
    "it": {"de": "tedesco", "en": "inglese", "fr": "francese", "es": "spagnolo", "it": "italiano"}}
PG_ASK = {"de": " Antworte in einem Absatz auf {L}.", "en": " Answer in one paragraph in {L}.",
          "fr": " Réponds en un paragraphe en {L}.", "es": " Responde en un párrafo en {L}.",
          "it": " Rispondi in un paragrafo in {L}."}


def load_polyglot(lang, n, seed):
    items, i = [], -1
    pairs = [(a, b) for a in PG_LANGS for b in PG_LANGS if a != b]
    topics = sorted(PG_TOPICS)
    while len(items) < (n or 60):
        i += 1
        src, dst = pairs[i % len(pairs)]
        # the de cell holds items whose target is de or (fr/es/it with even id); the en cell the rest
        cell = dst if dst in ("de", "en") else ("de" if (i // len(pairs) + PG_LANGS.index(src)) % 2 == 0 else "en")
        if cell != lang:
            continue
        q = PG_TOPICS[topics[i % len(topics)]][PG_LANGS.index(src)]
        items.append({"id": str(i), "lang": lang, "src": src, "dst": dst,
                      "messages": [{"role": "user", "content": q + PG_ASK[src].format(L=PG_NAMES[src][dst])}]})
    return items


def score_polyglot(item, reply):
    import langdetect
    langdetect.DetectorFactory.seed = 0
    text = reply["content"].strip()
    try:
        got = langdetect.detect(text) if len(text.split()) >= 8 else None
    except Exception:  # noqa: BLE001 — no detectable language
        got = None
    ok = got == item["dst"]
    return {"correct": ok, "score": float(ok), "pred": got, "gold": item["dst"],
            "detail": {f"ok_{item['src']}_{item['dst']}": float(ok), "detected": got}}


# ---------------------------------------------------------------- de_commas (German comma rules, generated)
# A sentence with its commas removed; the model puts them back. Kinds: subordinate clause after / before the main
# clause (weil, dass, obwohl, wenn), an embedded relative clause (two commas), an infinitive with "um ... zu", and
# a control with no comma at all (over-insertion). Scored on the comma positions over an unchanged word sequence.
CM_SUBJ = [("Der Koch", "der", "im Hotel am Markt arbeitet"), ("Die Lehrerin", "die", "an der Grundschule unterrichtet"),
           ("Das Kind", "das", "nebenan wohnt"), ("Mein Nachbar", "der", "seit Jahren hier wohnt"),
           ("Anna", "die", "gern im Garten sitzt")]
# verb, rest, fitting reasons (verb-final), a fitting "um ... zu", a fitting comma-free phrase
CM_VP = [("bleibt", "heute zu Hause", ["es draußen stark regnet", "die Straßen glatt sind"], "um sich auszuruhen",
          "wegen des Sturms"),
         ("kauft", "frisches Brot", ["morgen ein Feiertag ist", "der Bäcker bald schließt"], "um Gäste zu bewirten",
          "trotz des Regens"),
         ("liest", "ein langes Buch", ["der Zug wieder Verspätung hat", "der Fernseher kaputt ist"],
          "um sich zu entspannen", "nach dem Essen"),
         ("ruft", "den Arzt an", ["das Fieber nicht sinkt", "die Schmerzen stärker werden"],
          "um einen Termin zu vereinbaren", "am frühen Morgen"),
         ("geht", "früh schlafen", ["morgen ein langer Tag ist", "der Wecker um fünf klingelt"], "um fit zu sein",
          "nach dem Training")]
CM_CONJ = ["weil", "wenn"]
CM_KINDS = ("after", "before", "relative", "dass", "um_zu", "none")


def cm_sentence(r, kind):
    (subj, rel, relc), (verb, rest, subs, um, pp) = r.choice(CM_SUBJ), r.choice(CM_VP)
    sub, conj = r.choice(subs), r.choice(CM_CONJ)
    s = subj[0].lower() + subj[1:] if subj.split()[0] in ("Der", "Die", "Das", "Mein") else subj
    if kind == "after":
        return f"{subj} {verb} {rest}, {conj} {sub}."
    if kind == "before":
        return f"{conj.capitalize()} {sub}, {verb} {s} {rest}."
    if kind == "relative":
        return f"{subj}, {rel} {relc}, {verb} {rest}."
    if kind == "dass":
        return f"{subj} sagt, dass {sub}."
    if kind == "um_zu":
        return f"{subj} {verb} {rest}, {um}."
    return f"{subj} {verb} {pp} {rest}."


def cm_tokens(text):
    return re.findall(r"[\wäöüÄÖÜß]+|,", text.lower())


def load_de_commas(lang, n, seed):
    items, seen = [], set()
    for i in range(600):  # distinct sentences only
        if len(items) == (n or 150):
            break
        r = random.Random(f"de_commas:{i}")
        kind = CM_KINDS[i % len(CM_KINDS)]
        gold = cm_sentence(r, kind)
        if gold in seen:
            continue
        seen.add(gold)
        items.append({"id": str(i), "lang": "de", "kind": kind, "gold": gold,
                      "messages": [{"role": "user", "content": "Setze in diesen Satz alle Kommas, die nach den amtlichen "
                                    "Regeln nötig sind. Ändere sonst nichts und gib nur den Satz zurück.\n\n"
                                    + gold.replace(",", "")}]})
    return items


def score_de_commas(item, reply):
    lines = [x for x in (reply["content"] or "").strip().splitlines() if x.strip()]
    want = cm_tokens(item["gold"])
    words = [t for t in want if t != ","]
    hits = [x for x in lines if [t for t in cm_tokens(x) if t != ","] == words]  # the sentence, not an explanation
    got = cm_tokens(hits[-1] if hits else (lines[-1] if lines else ""))
    same_words = bool(hits)
    ok = same_words and got == want
    return {"correct": ok, "score": float(ok), "pred": (hits[-1] if hits else "")[:80], "gold": item["gold"],
            "detail": {f"ok_{item['kind']}": float(ok), "words_changed": float(not same_words)}}


# ---------------------------------------------------------------- de_homophones (das/dass, seit/seid, wieder/wider)
# Generated sentences with one gap; the model names the word. Each template's gold is fixed by grammar, the fillers
# vary. Sub-metric per pair.
HO_FILL = {"n": ["Buch", "Auto", "Fahrrad", "Radio", "Sofa"], "p": ["Anna", "mein Bruder", "die Chefin", "Herr Kaya"],
           "t": ["drei Wochen", "letztem Montag", "zwei Jahren", "dem Umzug"]}
HO_TEMPL = [
    ("das_dass", "das", "Das {n}, ___ ich gestern gekauft habe, ist schon kaputt."),
    ("das_dass", "dass", "{P} hofft, ___ das {n} bald fertig ist."),
    ("das_dass", "das", "Ich finde ___ {n} wirklich gelungen."),
    ("das_dass", "dass", "Es ist wichtig, ___ {p} morgen pünktlich kommt."),
    ("das_dass", "das", "Ist ___ wirklich wahr, was {p} erzählt?"),
    ("das_dass", "dass", "___ {p} gewonnen hat, überrascht niemanden."),
    ("seit_seid", "seit", "Wir wohnen ___ {t} in dieser Stadt."),
    ("seit_seid", "seid", "Ihr ___ heute aber früh dran!"),
    ("seit_seid", "seid", "___ bitte leise, das Baby schläft."),
    ("seit_seid", "seit", "{P} arbeitet ___ {t} im Homeoffice."),
    ("wieder_wider", "wieder", "Morgen soll es schon ___ regnen."),
    ("wieder_wider", "wider", "Die Mannschaft gewann ___ Erwarten das Finale."),
    ("wieder_wider", "wieder", "{P} hat das {n} ___ zurückgebracht."),
    ("wieder_wider", "wider", "Er handelte ___ besseres Wissen."),
]
HO_ASK = ("Welches Wort gehört in die Lücke ___? Antworte nur mit dem Wort, genau in der Schreibweise, die in den Satz "
          "passt (die Großschreibung am Satzanfang zählt nicht).\n\n{s}")


def load_de_homophones(lang, n, seed):
    items, seen = [], set()
    for i in range(400):  # the distinct sentences only (templates without fillers repeat)
        if len(items) == (n or 400):
            break
        r = random.Random(f"de_homophones:{i}")
        pair, gold, t = HO_TEMPL[i % len(HO_TEMPL)]
        p = r.choice(HO_FILL["p"])
        sent = t.format(n=r.choice(HO_FILL["n"]), p=p, P=p[0].upper() + p[1:], t=r.choice(HO_FILL["t"]))
        if sent in seen:
            continue
        seen.add(sent)
        items.append({"id": str(i), "lang": "de", "pair": pair, "gold": gold, "sent": sent,
                      "messages": [{"role": "user", "content": HO_ASK.format(s=sent)}]})
    return items


def score_de_homophones(item, reply):
    words = re.findall(r"[a-zäöüß]+", (tasks_novel.final_answer(reply["content"]) or reply["content"] or "").lower())
    cand = [w for w in words if w in {"das", "dass", "seit", "seid", "wieder", "wider"}]
    tmpl = re.findall(r"[a-zäöüß]+|___", item.get("sent", "").lower())
    if len(words) > 3 and len(words) == len(tmpl):  # the whole sentence filled in: read the word at the gap
        cand = [words[tmpl.index("___")]]
    ok = cand[:1] == [item["gold"]]
    return {"correct": ok, "score": float(ok), "pred": cand[:1], "gold": item["gold"],
            "detail": {f"ok_{item['pair']}": float(ok)}}


# ---------------------------------------------------------------- de_reported (indirect speech: Konjunktiv I / Ersatzform)
# A direct quote in the first person; the model rewrites it as indirect speech. Singular speakers need Konjunktiv I
# (er habe, sie sei, er könne); plural speakers need the Konjunktiv II Ersatzform where Konjunktiv I equals the
# indicative (sie hätten, sie kämen) — "sie seien" stays Konjunktiv I. Scored on the finite verb form.
RP_VERBS = {  # quote (1st person sg / pl), 3rd sg form, 3rd pl form
    "haben": ("habe keine Zeit", "haben keine Zeit", "habe", "hätten"),
    "sein": ("bin müde", "sind müde", "sei", "seien"),
    "können": ("kann nicht kommen", "können nicht kommen", "könne", "könnten"),
    "kommen": ("komme morgen", "kommen morgen", "komme", "kämen"),
    "wissen": ("weiß es nicht", "wissen es nicht", "wisse", "wüssten"),
    "müssen": ("muss früh gehen", "müssen früh gehen", "müsse", "müssten"),
}
RP_SPK = [("Anna", "sie", False), ("Der Trainer", "er", False), ("Die Gäste", "sie", True), ("Die Nachbarn", "sie", True)]


def load_de_reported(lang, n, seed):
    items = []
    verbs = list(RP_VERBS)
    for i in range(n or len(RP_VERBS) * len(RP_SPK)):  # every verb x speaker once
        verb = verbs[i % len(verbs)]
        spk, pron, plural = RP_SPK[(i // len(verbs)) % len(RP_SPK)]
        sg_q, pl_q, sg, pl = RP_VERBS[verb]
        quote = ("Wir " + pl_q if plural else "Ich " + sg_q) + "."
        gold = pl if plural else sg
        says = "sagen" if plural else "sagt"
        items.append({"id": str(i), "lang": "de", "verb": verb, "plural": plural, "gold": gold,
                      "messages": [{"role": "user", "content": "Gib diese Aussage in indirekter Rede wieder, wie in einem "
                                    f"Zeitungsbericht (\"{spk} {says}, ...\"). Antworte nur mit dem Satz.\n\n"
                                    f"{spk} {says}: \"{quote}\""}]})
    return items


def score_de_reported(item, reply):
    text = re.sub(r'[„"“»«][^„"“»«]*[“"«»]', " ", (reply["content"] or "")).lower()  # drop a copied quote
    forms = {f for v in RP_VERBS.values() for f in (v[2], v[3])} | {"haben", "hat", "ist", "sind", "kann", "können",
                                                                      "kommt", "kommen", "weiß", "wissen", "muss", "müssen",
                                                                      "würden", "würde"}
    after = re.split(r"\bsag(?:t|en)\b", text, maxsplit=1)[-1]
    found = [w for w in re.findall(r"[a-zäöüß]+", after) if w in forms]
    # plural: "sie würden kommen" is an accepted Ersatzform too; singular "würde" is not Konjunktiv I
    first_person = re.search(r"\b(ich|wir)\b", after) is not None
    ok = not first_person and (found[:1] == [item["gold"]]
                               or item["plural"] and item["verb"] in ("kommen", "wissen") and found[:1] == ["würden"])
    return {"correct": ok, "score": float(ok), "pred": found[:2], "gold": item["gold"],
            "detail": {"ok_plural" if item["plural"] else "ok_singular": float(ok)}}


# ---------------------------------------------------------------- en_homophones (its/it's, their/there/they're, then/than, affect/effect)
EH_FILL = {"p": ["Maria", "my brother", "the manager", "Mr Hale"], "n": ["bike", "laptop", "garden", "report"]}
EH_TEMPL = [
    ("its", "its", "The company changed ___ logo last year."),
    ("its", "it's", "___ been a long week for {p}."),
    ("its", "its", "The {n} lost ___ shine after the rain."),
    ("its", "it's", "{P} says ___ too late to fix the {n} today."),
    ("their", "their", "The neighbours parked ___ car in front of our {n}."),
    ("their", "there", "Put the {n} over ___, next to the door."),
    ("their", "they're", "Tell the kids ___ allowed to stay up late tonight."),
    ("their", "there", "___ is a problem with the {n}."),
    ("then", "than", "{P} is taller ___ everyone else in the team."),
    ("then", "then", "First check the {n}, and ___ write the summary."),
    ("then", "than", "The new {n} is cheaper ___ the old one."),
    ("affect", "affect", "The delay will not ___ the final price of the {n}."),
    ("affect", "effect", "The new rule had a strong ___ on {p}."),
    ("affect", "effect", "The medicine takes ___ after an hour."),
]
EH_SETS = {"its": {"its", "it's"}, "their": {"their", "there", "they're"}, "then": {"then", "than"},
           "affect": {"affect", "effect"}}
EH_ASK = ("Which word belongs in the gap ___? Answer with that word only, spelled exactly as it fits (ignore "
          "capitalisation at the start of the sentence).\n\n{s}")


def load_en_homophones(lang, n, seed):
    items, seen = [], set()
    for i in range(400):
        if len(items) == (n or 400):
            break
        r = random.Random(f"en_homophones:{i}")
        pair, gold, t = EH_TEMPL[i % len(EH_TEMPL)]
        p = r.choice(EH_FILL["p"])
        sent = t.format(n=r.choice(EH_FILL["n"]), p=p, P=p[0].upper() + p[1:])
        if sent in seen:
            continue
        seen.add(sent)
        items.append({"id": str(i), "lang": "en", "pair": pair, "gold": gold, "sent": sent,
                      "messages": [{"role": "user", "content": EH_ASK.format(s=sent)}]})
    return items


def score_en_homophones(item, reply):
    raw = (tasks_novel.final_answer(reply["content"]) or reply["content"] or "").lower().replace("’", "'")
    words = re.findall(r"[a-z]+(?:'[a-z]+)?", raw)
    cand = [w for w in words if w in EH_SETS[item["pair"]]]
    tmpl = re.findall(r"[a-z]+(?:'[a-z]+)?|___", item["sent"].lower())
    if len(words) > 3 and len(words) == len(tmpl):  # the whole sentence filled in: read the word at the gap
        cand = [words[tmpl.index("___")]]
    ok = cand[:1] == [item["gold"]]
    return {"correct": ok, "score": float(ok), "pred": cand[:1], "gold": item["gold"],
            "detail": {f"ok_{item['pair']}": float(ok)}}


# ---------------------------------------------------------------- glossary_mt (translation that must follow a client glossary)
# Two-sentence support texts translated EN->DE / DE->EN with a glossary that overrides the usual word
# ("ticket" -> "Vorgang", not "Ticket"). Scored exactly: every glossary target used, no forbidden usual rendering.
# term: (English word, usual German word, required German options, required English options for the German word)
GL_TERMS = common.OwnSet("tasks_lang.GL_TERMS")  # private: flozi00/weakbench-data
GL_SENT = common.OwnSet("tasks_lang.GL_SENT")  # private: flozi00/weakbench-data


def load_glossary_mt(lang, n, seed):
    """lang = target language: de items translate English into German, en items German into English."""
    items = []
    for i in range(n or 2 * len(GL_SENT)):  # every sentence twice, with a fresh glossary target
        r = random.Random(f"glossary_mt:{i}")
        en_s, de_s, terms = GL_SENT[i % len(GL_SENT)]
        T = [GL_TERMS[t] for t in terms]
        if lang == "de":
            src, gloss = en_s, [(t[0], r.choice(t[2])) for t in T]
            forbid = [t[1] for t in T]
            ask = "Übersetze ins Deutsche. Halte dich strikt an das Glossar des Kunden. Antworte nur mit der Übersetzung."
        else:
            src, gloss = de_s, [(t[1], r.choice(t[3])) for t in T]
            forbid = [t[0] for t in T]
            ask = "Translate into English. Follow the client's glossary strictly. Reply with the translation only."
        g = "\n".join(f"{a} -> {b}" for a, b in gloss)
        items.append({"id": str(i), "lang": lang, "gold": [b for _, b in gloss], "forbid": forbid,
                      "messages": [{"role": "user", "content": f"{ask}\n\nGlossar / glossary:\n{g}\n\n{src}"}]})
    return items


def gl_form(term):
    """The term with its inflections, incl. the umlaut plural (Vorgang -> Vorgänge, Zugang -> Zugänge)."""
    t = term.lower()
    uml = re.sub(r"([aou])(?=[^aou]*$)", lambda m: {"a": "ä", "o": "ö", "u": "ü"}[m[1]], t, count=1)
    return r"\b(" + re.escape(t) + "|" + re.escape(uml) + r")(s|es|e|en|n|er|ern)?\b"


def score_glossary_mt(item, reply):
    text = (reply["content"] or "").lower()
    used = [bool(re.search(gl_form(t), text)) for t in item["gold"]]
    # the usual word is forbidden as its own word or an inflection of it ("Rechnungen"), not inside a compound
    bad = [f for f in item["forbid"] if re.search(r"\b" + re.escape(f.lower()) + r"(s|n|en|e|es)?\b", text)
           or f == "Konto" and re.search(r"\bkonten\b", text)]
    ok = all(used) and not bad
    return {"correct": ok, "score": float(ok), "pred": text[:80], "gold": item["gold"],
            "detail": {"terms_used": sum(used) / len(used), "forbidden_used": float(bool(bad))}}


# ---------------------------------------------------------------- de_separable (trennbare Verben: main clause, subordinate clause, perfect)
# An infinitive and a sentence with gaps; the model writes the sentence. Main clause: prefix to the end ("ruft ... an");
# subordinate clause: joined, verb-final ("dass sie ... anruft"); perfect: ge- inside ("angerufen"); zu-infinitive:
# zu inside ("anzurufen"). Scored on the verb pieces in the right slots.
SV_VERBS = [  # infinitive, prefix, 3sg stem form, participle
    ("anrufen", "an", "ruft", "angerufen"), ("einkaufen", "ein", "kauft", "eingekauft"),
    ("aufräumen", "auf", "räumt", "aufgeräumt"), ("abholen", "ab", "holt", "abgeholt"),
    ("mitbringen", "mit", "bringt", "mitgebracht"), ("vorbereiten", "vor", "bereitet", "vorbereitet"),
    ("zurückgeben", "zurück", "gibt", "zurückgegeben"), ("ausschalten", "aus", "schaltet", "ausgeschaltet"),
]
SV_OBJ = {"anrufen": "die Großmutter", "einkaufen": "für das Wochenende", "aufräumen": "die Küche",
          "abholen": "die Kinder", "mitbringen": "einen Kuchen", "vorbereiten": "die Präsentation",
          "zurückgeben": "das Buch", "ausschalten": "das Licht"}
SV_FORMS = ("main", "sub", "perfect", "zu")


def sv_mid(subj):  # "Mein Bruder" -> "mein Bruder" inside a sentence
    return subj[0].lower() + subj[1:] if subj.split()[0] in ("Mein", "Die") else subj


def sv_gold(verb, form, subj):
    inf, pre, fin, part = verb
    o = SV_OBJ[inf]
    if form == "main":
        return f"{subj} {fin} heute {o} {pre}."
    if form == "sub":
        return f"Ich glaube, dass {sv_mid(subj)} heute {o} {pre}{fin}."
    if form == "perfect":
        return f"{subj} hat gestern {o} {part}."
    return f"{subj} hat vergessen, {o} {pre}zu{inf[len(pre):]}."


SV_FRAME = {"main": "{s} ___ heute {o} ___.", "sub": "Ich glaube, dass {s} heute {o} ___.",
            "perfect": "{s} hat gestern {o} ___.", "zu": "{s} hat vergessen, {o} ___."}


def load_de_separable(lang, n, seed):
    items = []
    for i in range(n or len(SV_VERBS) * len(SV_FORMS)):
        r = random.Random(f"de_separable:{i}")
        verb = SV_VERBS[(i // len(SV_FORMS)) % len(SV_VERBS)]
        form = SV_FORMS[i % len(SV_FORMS)]
        subj = r.choice(["Anna", "Mein Bruder", "Die Nachbarin", "Herr Yilmaz"])
        frame = SV_FRAME[form].format(s=sv_mid(subj) if form == "sub" else subj, o=SV_OBJ[verb[0]])
        items.append({"id": str(i), "lang": "de", "form": form, "gold": sv_gold(verb, form, subj),
                      "messages": [{"role": "user", "content": f"Setze das Verb „{verb[0]}“ in der richtigen Form in die "
                                    "Lücken ein und schreibe den ganzen Satz. Antworte nur mit dem Satz.\n\n" + frame}]})
    return items


def score_de_separable(item, reply):
    lines = [x.strip().strip("„“\"*") for x in (reply["content"] or "").splitlines() if x.strip()]
    norm = lambda t: re.findall(r"[a-zäöüß]+", t.lower())  # noqa: E731
    ok = any(norm(x) == norm(item["gold"]) for x in lines)
    return {"correct": ok, "score": float(ok), "pred": (lines[-1] if lines else "")[:80], "gold": item["gold"],
            "detail": {f"ok_{item['form']}": float(ok)}}


def task(langs, load, score, max_tokens, quick, **kw):
    return {"langs": langs, "load": load, "score": score, "think": False, "max_tokens": max_tokens,
            "counts": {"quick": dict.fromkeys(langs, quick), "full": dict.fromkeys(langs, None)}, **kw}


TASKS = {
    # lang_check off: the scorer sets lang_ok itself (the target language, not the prompt's)
    "lang_switch": task(("de", "en"), load_switch, score_switch, 1024, 60, lang_check=False),
    "grammar_pairs": {**task(("de", "en"), load_grammar, score_grammar, 64, 300, lang_check=False),
                      "counts": {"quick": {"de": 300, "en": 300}, "full": {"de": None, "en": None}}},
    "spell_fix": {**task(("de", "en"), load_spell, score_spell, 1024, 100, lang_check=False)},
    "paraphrase": task(("de", "en"), load_paraphrase, score_paraphrase, 1024, 80),
    "en_produce": task(("en",), load_en_produce, score_en_produce, 256, 60, lang_check=False),
    "de_produce": task(("de",), load_de_produce, score_de_produce, 256, 60, lang_check=False),
    "de_compounds": task(("de",), load_de_compounds, score_de_compounds, 64, None, lang_check=False),
    "de_letter": task(("de",), load_de_letter, score_de_letter, 1024, None),
    "officialese_qa": task(("de", "en"), load_officialese, score_officialese, 256, None),
    "simplify": task(("de", "en"), load_simplify, score_simplify, 1024, None),
    "polyglot_switch": task(("de", "en"), load_polyglot, score_polyglot, 512, 40, lang_check=False),
    "numerals": task(("de", "en"), load_numerals, score_numerals, 128, 60, lang_check=False),
    "coref": task(("de", "en"), load_coref, score_coref, 64, None, lang_check=False),
    "pragmatics": task(("de", "en"), lambda lang, n, seed: load_idioms(lang, n, seed, PRAGMATICS, "pragmatics",
                                                                           PRAG_ASK),
                       score_idioms, 64, None, lang_check=False),
    "idioms": task(("de", "en"), load_idioms, score_idioms, 64, None, lang_check=False),
    "sys_adherence": task(("de", "en"), load_sys_adherence, score_sys_adherence, 1024, 30, lang_check=False),
    "wordplay": {**task(("de", "en"), load_wordplay, score_wordplay, 4096, 60), "think": True},
    "locale_fmt": task(("de", "en"), load_locale, score_locale, 64, 60, lang_check=False),
    "roundtrip": task(("de", "en"), load_roundtrip, score_roundtrip, 1024, 60, lang_check=False),
    "de_verbs": task(("de",), load_de_verbs, score_de_verbs, 64, 120, lang_check=False),
    "de_case": task(("de",), load_de_case, score_de_case, 64, 120, lang_check=False),
    "de_homophones": task(("de",), load_de_homophones, score_de_homophones, 32, 40, lang_check=False),
    "de_reported": task(("de",), load_de_reported, score_de_reported, 128, 24, lang_check=False),
    "en_homophones": task(("en",), load_en_homophones, score_en_homophones, 32, 40, lang_check=False),
    "glossary_mt": task(("de", "en"), load_glossary_mt, score_glossary_mt, 256, 12),
    "de_separable": task(("de",), load_de_separable, score_de_separable, 128, 32, lang_check=False),
    "de_commas": task(("de",), load_de_commas, score_de_commas, 128, 60, lang_check=False),
    "register_shift": task(("de",), load_register, score_register, 512, None),
    "lang_quality": task(("de", "en"), load_quality, score_quality, 2048, None, lang_check=False, opt_in=True,
                         needs_judge=True),
}
