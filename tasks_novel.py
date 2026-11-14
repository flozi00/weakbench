"""Generalization tasks of the quality suite (quality.py imports TASKS): problems that are generated from a seed
at load time, so no item can be in any training set and no amount of benchmark tuning helps. Each one gives the
model a rule system it has never seen and asks it to apply it. Exact scoring, stdlib only, DE and EN (the puzzle
of an id is the same in both languages; only the instruction is translated).

rule_induction  Infer a hidden string transformation (a composition of 1-3 primitives: reverse, Caesar shift,
                rotation, vowel doubling, pair swap, sorting, case pattern) from 6 examples, apply it to a new
                word. Sub-metric `depth` (1-3) is in detail; per-depth accuracy is in `ok_d<n>`.
invented_ops    Two operators defined on the spot (a # b = p*a + q*b + r, a @ b = a*b - s*a) and a nested,
                fully parenthesised expression of depth 2-4 over them. The answer is an integer.
mini_language   Compositional translation (COGS/SCAN-style): 12 example translations from English into an
                invented language with its own word order, adjective position, plural suffix and negation
                particle; translate a new sentence whose words all appeared, but never in this combination.

theory_of_mind  Generated false-belief stories (Sally-Anne family): people put an object somewhere, some leave,
                it is moved (sometimes twice, sometimes while a person watches through a window), people come
                back. Questions ask where the object is, where a person will look (first order), or where one
                person thinks another will look (second order). The answer is a container name. Sub-metrics
                `ok_reality`, `ok_first`, `ok_second`.
temporal        Calendar arithmetic generated from a seed: weekday of a date N days after a given date, the
                number of days between two dates (leap years included), the date N weeks before a date. The
                answer is a weekday name or an ISO date or an integer; DE accepts German weekday names.
state_tracking  N people each hold one item; a list of 10 / 20 / 30 swaps between pairs; who holds item X at the end
                (BIG-Bench "tracking shuffled objects", generated, longer: 10 / 20 / 30 swaps). Sub-metric `ok_n<swaps bucket>`.

math_variants   Own word-problem templates (GSM-Symbolic idea, licence-free) instantiated with fresh names and
                numbers per id, in three variants: `base`; `plus` (one more step); `noop` (base + a clause that
                sounds relevant but changes nothing). The noop drop is the classic pattern-matching tell.
                Sub-metrics `ok_base`, `ok_plus`, `ok_noop`.

dialog_state    A generated 5-7 turn conversation planning a dinner: guests are announced over several turns, some
                with a diet, one is cancelled ("X can't come after all"), one changes diet. The last turn asks
                how many guests come and which of them eat vegetarian. Each turn is a real chat turn (the
                model's replies stay in the history). Exact: the count and the set of names.
                Sub-metrics `ok_count`, `ok_names`.

code_trace      A small Python program generated per id from typed building blocks (arithmetic over a list, a loop
                with a condition, string slicing/joining, a dict counter, a while loop with integer division);
                predict what it prints. The gold comes from running OUR generated source (never model code) in
                an empty namespace. Sub-metric `ok_<block count>`.
spatial         Grid navigation: start at a cell facing a direction, follow 8 / 14 / 20 instructions (turn left/right,
                go N steps, turn around); give the final cell and facing. Exact.

if_gen          Generated instruction following: a random topic plus 2-3 verifiable constraints drawn from a pool
                (exact sentence/bullet count, a required word, a forbidden letter-free word list, a closing phrase,
                all lowercase, a word limit, a title in double angle brackets, a postscript, a JSON wrapper),
                compatible ones only. correct = all constraints met; score = share met; `ok_<constraint>`.

logic_grid      Zebra-style puzzles generated per id: 3-4 houses in a row, 3 attributes (name, pet, drink), clues
                (same house, left of / next to, position, not) added until a brute-force check over all
                assignments leaves exactly one solution. Ask one attribute of one person. Sub-metric `ok_h<n>`.

sci_calc        Applied calculation with unit traps, generated: Ohm's law with milliamps, distance from km/h and
                minutes, electricity cost from watts/hours/days/ct per kWh, density g/cm3 -> kg/m3, mixing two
                solutions. Answer within 1 % (DE "1.234,5" and EN "1,234.5" both read). `ok_<kind>`.

math_missing    The math_variants templates, half of them with one needed quantity made vague ("some rolls burn").
                Answerable items need the number; unanswerable ones need "cannot be determined" and no guess.
                Always abstaining scores 50 %. Sub-metrics `ok_answerable`, `ok_missing`, `guessed` (a number on a
                missing item).

table_qa        A generated sales table (8-12 rows: product, region, units, unit price) and one question: a filtered sum
                of revenue, the product with the most units in a region, or a count above a threshold. Exact.
                `ok_<kind>`.

seq_induction   A generated integer sequence (7 terms) from one of several rule families with random parameters
                (arithmetic, geometric, interleaved two sequences, a_n = a_{n-1} + a_{n-2} + c, n^2 + c, alternating
                add/multiply); give the next term. Draws that another family with other parameters also fits but
                continues differently are skipped. `ok_<family>`.

error_spot      Verification: a numbered chain of 5-8 arithmetic steps (x = prev op k) where exactly one result is wrong
                (off by a plausible slip) and every later step correctly continues from the wrong value. Name the
                first wrong step. `ok_len<n>`.

schedule        Constraint scheduling, generated: 4-6 meetings into hourly slots 9-14 (one room), each with participants;
                constraints "X before Y", "X not at h", "X and Y not back to back", plus no person in two meetings
                at once is implicit (one room). Any assignment satisfying everything scores; the generator keeps
                only instances with at least one but at most 20 % of permutations valid. `ok_m<n>`.

syllogism       Quantifier logic over nonsense categories (so world knowledge cannot help): 2-3 premises with all / some
                / no / not all, a conclusion, answer TRUE (follows), FALSE (contradicted) or UNKNOWN. The gold is
                computed over every non-empty-region pattern of a 3-set Venn diagram (premises' "some" asserts
                existence). Balanced over the three answers. `ok_<gold>`.

extract         Information extraction from a generated business email (invoice reminder): JSON with sender company,
                invoice number, amount due and due date (ISO). Distractors: an earlier invoice date, a second
                amount (a fee or a discount), the date written out in words. Each field exact; correct = all four.
                `ok_<field>`.

graph_path      A generated road network (6-9 towns with invented names, 8-14 roads with km) and the shortest distance
                between two towns (Dijkstra). Only instances where a direct road, if any, is not the shortest, and
                the shortest path has >= 3 hops. `ok_n<towns>`.

probability     Generated probability: draws from an urn with / without replacement, sums of dice, "at least one" over
                repeated trials. Gold = an exact Fraction from enumeration; an answer counts as a fraction, a
                decimal or a percentage within 0.5 percentage points. `ok_<kind>`.

event_order     5-6 events of a day given only as relations ("X happened after Y", "X right before Y", "X was the first /
                last"), shuffled; give the full order. Relations are added until exactly one permutation fits.
                `ok_e<n>`.

mini_sudoku     A generated 4x4 sudoku (digits 1-4, rows, columns and 2x2 boxes) with cells removed while the solution
                stays unique (brute-force count); ask for the full solved grid. Exact grid; `cells_right` share.

code_bug        A short generated Python function with exactly one injected bug (off-by-one range, wrong comparison,
                wrong operator, wrong variable), an example call with the expected and the actual result (the
                actual one computed by running OUR buggy source); name the buggy line number. `ok_<bug>`.

timezones       A meeting at a local time and weekday in one city; the UTC offsets of both cities are given in the prompt
                (incl. +5:30, -3:30, +12:45); give local time and weekday in the other city. Exact; `ok_dayshift`.

calib_math      The math_variants and invented_ops items, plus "Confidence: N%" after the answer. score = 1 - Brier of
                the stated confidence against correctness (a missing confidence counts as Brier 1);
                `overconf` = confidence - correct. Calibration of reasoning, next to `calibration` (of knowledge).

kinship         A generated three-generation family (invented names, genders explicit), told only as parent and sibling
                facts in shuffled order (gender follows from "mother"/"father", never from a name); questions need
                2-3 hops: grandmother/grandfather, aunt/uncle (by blood or marriage), cousin. One set of
                grandparents, so the gold is unique. `ok_<relation>`.

meeting_actions A generated meeting transcript: 3-4 tasks get assigned with a weekday deadline, then one is reassigned
                and one deadline moved ("Actually Ben takes it", "let's make that Friday"), plus small talk. Return
                the final tasks as JSON {task: [owner, weekday]}. correct = every task exact; score = share right.

game_theory     Generated 2x2 games (payoffs 0-9, both players): name the unique pure Nash equilibrium as a cell
                ("Up-Left"); only games with exactly one pure equilibrium are kept. `ok_dominant` (it is a
                dominant-strategy equilibrium) vs `ok_nondominant`.

rulebook        An invented fee schedule (tiered base fee by weight, a surcharge rule with an exception, a capped
                discount) applied to a generated case; the final fee in whole units. `ok_exception` marks cases where
                the exception applies.

knights_knaves  Generated: three or four islanders, knights always tell the truth, knaves always lie; each says one statement
                ("B is a knave", "A and C are the same kind", "at least one of us is a knave"); only puzzles with
                exactly one consistent assignment (brute force over 8 or 16). Name each one's kind; correct = all of them.

number_bases    Generated base conversion (2, 8, 10, 16 in any direction) and binary addition of two 10-15-bit numbers.
                Exact (leading zeros, 0x/0b prefixes and case ignored); `ok_convert`, `ok_add`.

copycat         Hofstadter-style letter-string analogies from ONE example: "abc -> abd; ijk -> ?". Rule families:
                successor of the last letter, predecessor of the first, reversal, doubling the last letter,
                successor of every letter; the target string differs in length/letters from the example. Exact.

gears           A generated chain of 4-8 wheels joined by meshing gears (reverse the direction) or open belts (keep it),
                with tooth counts; the first turns clockwise at N rpm. Ask the last wheel's direction, or its rpm
                (exact: rpm scales with the tooth ratio of each mesh/belt, intermediate wheels cancel). `ok_<kind>`.

jugs            Two jugs (capacities 3-13 l), a target volume in either jug; the minimal number of moves (fill, empty,
                pour) by breadth-first search, or "impossible" (target not a multiple of gcd, or larger than both).
                About a quarter impossible. `ok_possible`, `ok_impossible`.

linalg          Generated small linear algebra: a 3x3 determinant, a 3x3 linear system with an integer solution (built
                from the solution), a matrix-vector product. Exact; `ok_det`, `ok_solve`, `ok_matvec`.

self_correct    The history holds an earlier ASSISTANT turn with a wrong product ("17 x 23 = 381"); the user builds on it
                ("and plus 10?"). The right reply uses the correct product (401); carrying the error gives 391.
                `carried` = the answer equals the error-propagated value.

data2text       Generated weather records for 3 cities (high, low, rain %, wind km/h) -> a short report. Deterministic:
                `coverage` = share of the 12 values stated; `invented` = numbers in the text that are not in the data
                (dates and the count "3" aside). correct = full coverage and nothing invented; score = coverage, 0 if
                anything invented.

negation_qa     Generated short reports built from negated and corrected statements ("Not Anna but Ben bought the apples",
                "Clara bought neither the bread nor the milk", "Except David, everyone ..."), then "Who bought X?".
                The first-named person is usually wrong. Exact name; `ok_<pattern>`.

citation_check  Four generated sources about invented companies and an answer of three claims, each citing one source;
                exactly one citation points to a source that does not support its claim (the fact sits in another
                source). Name the claim number. `wrong_pos` in detail.

All of them run with thinking on (dialog_state, if_gen: off, extract: off, meeting_actions: off, data2text: off); the answer is the last "Answer:"/"Antwort:" line.
"""
import common
import json
import math
import random
import re

ASK = {"en": "Think it through, then finish with one line \"Answer: <your answer>\".",
       "de": "Denke es durch und schließe mit einer Zeile \"Antwort: <deine Antwort>\" ab."}
ANSWER = re.compile(r"(?im)^\W*(?:final\s+)?(?:answer|antwort)\W{0,4}?[:：]\s*(.+?)\s*$")  # "**Answer**: x" too


def final_answer(text):
    m = ANSWER.findall(text or "")
    return m[-1].strip().strip("`*\"'«»„“”").strip().rstrip(".!").strip("`*\"'«»„“”").strip() if m else None


def last_int(text):
    """The last integer in `text`: '22.00 €' -> 22, '1.234' / '1,234' -> 1234, '5 # 3 = 19' -> 19,
    '22.5' -> None (not an integer, so never equal to an integer gold)."""
    toks = re.findall(r"[−-]?\d[\d.,  ]*\d|[−-]?\d", text or "")
    if not toks:
        return None
    t = re.sub(r"[  ]", "", toks[-1]).replace("−", "-")
    t = re.sub(r"[.,]0+$", "", t)
    if re.fullmatch(r"-?\d{1,3}(?:[.,]\d{3})+", t):
        t = re.sub(r"[.,]", "", t)
    return int(t) if re.fullmatch(r"-?\d+", t) else None


def rng_of(task, i):
    return random.Random(f"{task}:{i}")  # str seeds hash deterministically (sha512), unlike tuples


# ---------------------------------------------------------------- rule_induction
VOWELS = "aeiou"
PRIMS = {
    "reverse": lambda s, k: s[::-1],
    "shift": lambda s, k: "".join(chr((ord(c) - 97 + k) % 26 + 97) if c.islower() else c for c in s),
    "rotate": lambda s, k: s[k % len(s):] + s[:k % len(s)],
    "double_vowels": lambda s, k: "".join(c * 2 if c in VOWELS else c for c in s),
    "swap_pairs": lambda s, k: "".join(s[i + 1] + s[i] if i + 1 < len(s) else s[i] for i in range(0, len(s), 2)),
    "sort": lambda s, k: "".join(sorted(s)),
    "upper_every": lambda s, k: "".join(c.upper() if i % (k + 1) == 0 else c for i, c in enumerate(s)),
}
LAST = ("upper_every",)  # case changes go last: the other primitives assume lowercase


def apply_rule(rule, s):
    for name, k in rule:
        s = PRIMS[name](s, k)
    return s


def word(r):
    return "".join(r.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(r.randint(5, 8)))


def make_rule(r, depth):
    names = r.sample([p for p in PRIMS if p not in LAST], depth) if depth < 3 or r.random() < 0.5 else \
        r.sample([p for p in PRIMS if p not in LAST], depth - 1) + ["upper_every"]
    return [(n, r.randint(1, 3)) for n in names]


def all_rules():
    """Every rule the generator can draw (k only matters where a primitive uses it)."""
    atoms = [(p, k) for p in PRIMS if p not in LAST for k in ((1, 2, 3) if p in ("shift", "rotate") else (1,))]
    seqs = [[]]
    for _ in range(3):
        seqs += [q + [a] for q in seqs if len(q) < 3 for a in atoms if a[0] not in {x[0] for x in q}]
    seqs = [q for q in {tuple(q) for q in seqs} if q]
    return [list(q) for q in seqs] + [list(q) + [("upper_every", k)] for q in seqs if len(q) < 3 for k in (1, 2, 3)]


RULES = all_rules()


def unique_answer(words, outs):
    """False when another drawable rule fits all six examples but maps the test word elsewhere: such an item
    has no single right answer."""
    for rule in RULES:
        if all(apply_rule(rule, w) == o for w, o in zip(words[:6], outs[:6])) and apply_rule(rule, words[6]) != outs[6]:
            return False
    return True


RI_TEXT = {"en": "Each input word is turned into an output word by the same hidden rule. Examples:\n{ex}\n\n"
                 "What is the output for \"{q}\"? Letter case matters.",
           "de": "Jedes Eingabewort wird nach derselben verborgenen Regel in ein Ausgabewort umgewandelt. "
                 "Beispiele:\n{ex}\n\nWas ist die Ausgabe für \"{q}\"? Groß- und Kleinschreibung zählt."}


def load_rule_induction(lang, n, seed):
    items = []
    for i in range(n or 300):
        r = rng_of("rule_induction", i)
        depth = 1 + i % 3
        while True:  # a rule that is the identity on the examples teaches nothing
            rule = make_rule(r, depth)
            words = [word(r) for _ in range(7)]
            outs = [apply_rule(rule, w) for w in words]
            if all(o != w for o, w in zip(outs, words)) and len(set(outs)) == 7 and unique_answer(words, outs):
                break
        ex = "\n".join(f"{w} -> {o}" for w, o in zip(words[:6], outs[:6]))
        items.append({"id": str(i), "lang": lang, "gold": outs[6], "depth": depth, "rule": rule,
                      "messages": [{"role": "user", "content": RI_TEXT[lang].format(ex=ex, q=words[6]) + " " + ASK[lang]}]})
    return items


def score_rule_induction(item, reply):
    pred = final_answer(reply["content"])
    ok = pred is not None and pred == item["gold"]
    return {"correct": ok, "score": float(ok), "pred": pred, "gold": item["gold"],
            "detail": {"depth": item["depth"], f"ok_d{item['depth']}": float(ok)}}


# ---------------------------------------------------------------- invented_ops
OPS_TEXT = {"en": "Two new operators are defined for integers:\n  a # b = {p}*a + {q}*b + ({r})\n  a @ b = a*b - {s}*a\n"
                  "Evaluate: {expr}",
            "de": "Für ganze Zahlen sind zwei neue Operatoren definiert:\n  a # b = {p}*a + {q}*b + ({r})\n"
                  "  a @ b = a*b - {s}*a\nBerechne: {expr}"}


def expr_tree(r, depth):
    if depth == 0:
        return str(r.randint(-9, 9))
    return f"({expr_tree(r, r.randint(0, depth - 1))} {r.choice('#@')} {expr_tree(r, depth - 1)})"


def eval_ops(expr, p, q, r_, s):
    """Exact evaluation of a fully parenthesised #/@ expression (no eval of model or generated code)."""
    toks = re.findall(r"-?\d+|[()#@]", expr)
    pos = 0

    def term():
        nonlocal pos
        t = toks[pos]
        pos += 1
        if t != "(":
            return int(t)
        a = term()
        op = toks[pos]
        pos += 1
        b = term()
        pos += 1  # ")"
        return p * a + q * b + r_ if op == "#" else a * b - s * a
    return term()


def load_invented_ops(lang, n, seed):
    items = []
    for i in range(n or 300):
        r = rng_of("invented_ops", i)
        p, q, r_, s = r.randint(2, 5), r.choice([-3, -2, -1, 1, 2, 3]), r.randint(-6, 6), r.randint(1, 4)
        depth = 2 + i % 3
        expr = expr_tree(r, depth)
        items.append({"id": str(i), "lang": lang, "gold": eval_ops(expr, p, q, r_, s), "depth": depth,
                      "messages": [{"role": "user", "content": OPS_TEXT[lang].format(
                          p=p, q=q, r=r_, s=s, expr=expr[1:-1]) + "\n" + ASK[lang]}]})
    return items


def score_invented_ops(item, reply):
    pred = final_answer(reply["content"])
    ok = last_int(pred) == item["gold"]
    return {"correct": ok, "score": float(ok), "pred": pred, "gold": item["gold"],
            "detail": {"depth": item["depth"], f"ok_d{item['depth']}": float(ok)}}


# ---------------------------------------------------------------- mini_language
NOUNS = ["dog", "cat", "bird", "child", "farmer", "teacher", "horse", "goat", "king", "doctor", "baker", "fox"]
VERBS = [("see", "sees"), ("follow", "follows"), ("help", "helps"), ("find", "finds"), ("call", "calls"),
         ("watch", "watches")]
ADJS = ["big", "small", "red", "old", "happy", "quiet"]
SYL = ["ka", "lo", "mi", "ru", "te", "zo", "pa", "ne", "vu", "si", "do", "fe", "gu", "ba", "ti", "ro", "le", "mu"]
ML_TEXT = {"en": "Here are English sentences and their translations into an invented language:\n{ex}\n\n"
                 "Translate into the invented language: \"{q}\"\nGive the translation in lowercase, words separated by "
                 "spaces, without punctuation.",
           "de": "Hier sind englische Sätze und ihre Übersetzungen in eine erfundene Sprache:\n{ex}\n\n"
                 "Übersetze in die erfundene Sprache: \"{q}\"\nGib die Übersetzung in Kleinbuchstaben an, Wörter durch "
                 "Leerzeichen getrennt, ohne Satzzeichen."}


def ml_sentence(r):
    np_ = lambda: (r.choice(ADJS) if r.random() < 0.4 else None, r.choice(NOUNS), r.random() < 0.35)  # noqa: E731
    subj, obj = np_(), np_()
    while obj[1] == subj[1]:
        obj = np_()
    return subj, r.randrange(len(VERBS)), r.random() < 0.3, obj


def english(sent):
    (sa, sn, sp), v, neg, (oa, on, op) = sent
    np_ = lambda a, n, pl: "the " + (a + " " if a else "") + n + ("es" if pl and n in ("fox",)  # noqa: E731
                                                                 else "ren" if pl and n == "child" else
                                                                 "s" if pl else "")
    verb = (("do not " if sp else "does not ") + VERBS[v][0]) if neg else VERBS[v][0 if sp else 1]
    return f"{np_(sa, sn, sp)} {verb} {np_(oa, on, op)}"


def target(sent, g):
    (sa, sn, sp), v, neg, (oa, on, op) = sent
    np_ = lambda a, n, pl: ([g["lex"][n] + (g["pl"] if pl else "")] +  # noqa: E731
                            ([g["lex"][a]] if a else []))[:: 1 if g["adj_after"] else -1]
    vp = [g["lex"][VERBS[v][0]]]
    if neg:
        vp = [g["neg"]] + vp if g["neg_before"] else vp + [g["neg"]]
    parts = {"S": np_(sa, sn, sp), "V": vp, "O": np_(oa, on, op)}
    return " ".join(w for k in g["order"] for w in parts[k])


def load_mini_language(lang, n, seed):
    items, i = [], -1
    while len(items) < (n or 200):  # ids whose examples cannot cover the test words are skipped, in both langs
        i += 1
        r = rng_of("mini_language", i)
        made, lex = set(), {}
        for w in NOUNS + [v[0] for v in VERBS] + ADJS + ["#neg", "#pl"]:
            while True:
                x = "".join(r.choice(SYL) for _ in range(r.randint(1, 2) if w.startswith("#") else r.randint(2, 3)))
                if x not in made:
                    made.add(x)
                    lex[w] = x
                    break
        g = {"lex": lex, "pl": lex["#pl"], "neg": lex["#neg"], "order": r.choice(["SVO", "SOV", "VSO", "OSV", "VOS"]),
             "adj_after": r.random() < 0.5, "neg_before": r.random() < 0.5}
        test = ml_sentence(r)
        words_of = lambda s: {s[0][1], s[3][1], VERBS[s[1]][0]} | {a for a in (s[0][0], s[3][0]) if a}  # noqa: E731
        need = words_of(test)
        # every feature is shown at least twice (the model has to see a pattern, not one instance)
        feats = {"neg": lambda s: s[2], "pl": lambda s: s[0][2] or s[3][2], "adj": lambda s: s[0][0] or s[3][0]}
        shown = dict.fromkeys(feats, 0)
        exs, seen = [], set()
        for _ in range(20000):
            if len(exs) == 12:
                break
            s = ml_sentence(r)
            if english(s) == english(test) or english(s) in {english(e) for e in exs}:
                continue
            useful = (words_of(s) & need) - seen or any(f(s) and shown[k] < 2 for k, f in feats.items())
            if useful or 12 - len(exs) > len(need - seen) + sum(2 - v for v in shown.values()):
                exs.append(s)
                seen |= words_of(s)
                shown = {k: shown[k] + bool(f(s)) for k, f in feats.items()}
        if not need <= seen or min(shown.values()) < 2:
            continue
        ex = "\n".join(f"\"{english(s)}\" -> \"{target(s, g)}\"" for s in exs)
        items.append({"id": str(i), "lang": lang, "gold": target(test, g), "order": g["order"],
                      "messages": [{"role": "user", "content": ML_TEXT[lang].format(ex=ex, q=english(test)) + " "
                                    + ASK[lang]}]})
    return items


def score_mini_language(item, reply):
    pred = final_answer(reply["content"])
    norm = " ".join(re.sub(r"[^\w\s]", " ", (pred or "").lower()).split())
    gold, ok = item["gold"], norm == item["gold"]
    pw, gw = norm.split(), gold.split()
    return {"correct": ok, "score": float(ok), "pred": pred, "gold": gold,
            "detail": {"order": item["order"], "word_overlap": len(set(pw) & set(gw)) / len(set(gw)),
                       "same_bag": float(sorted(pw) == sorted(gw))}}


# ---------------------------------------------------------------- theory_of_mind
PEOPLE = {"en": ["Anna", "Ben", "Clara", "David", "Emma", "Felix"], "de": ["Anna", "Ben", "Clara", "David", "Emma",
                                                                      "Felix"]}
THINGS = {"en": [("the ball",) * 3, ("the key",) * 3, ("the apple",) * 3, ("the book",) * 3, ("the ring",) * 3,
                 ("the coin",) * 3],  # (nominative, accusative, dative)
          "de": [("der Ball", "den Ball", "dem Ball"), ("der Schlüssel", "den Schlüssel", "dem Schlüssel"),
                 ("der Apfel", "den Apfel", "dem Apfel"), ("das Buch", "das Buch", "dem Buch"),
                 ("der Ring", "den Ring", "dem Ring"), ("die Münze", "die Münze", "der Münze")]}
PLACES = {"en": ["basket", "box", "drawer", "cupboard", "suitcase", "bag"],
          "de": ["Korb", "Kiste", "Schublade", "Schrank", "Koffer", "Tasche"]}
DE_ART = {"Korb": ("den", "dem"), "Kiste": ("die", "der"), "Schublade": ("die", "der"), "Schrank": ("den", "dem"),
          "Koffer": ("den", "dem"), "Tasche": ("die", "der")}  # (accusative, dative)
TOM = {"en": {"put": "{a} puts {t} in the {p}.", "leave": "{a} leaves the room.", "back": "{a} comes back.",
              "move": "{a} moves {t} from the {p} to the {q}.",
              "watch": "{a} secretly watches through the window, unseen by everyone in the room.",
              "reality": "Where is {t} now?", "first": "Where will {b} look for {t} first?",
              "second": "Where does {a} think {b} will look for {t}?",
              "ask": "Answer with the name of the container only."},
       "de": {"put": "{a} legt {t} in {p}.", "leave": "{a} verlässt den Raum.", "back": "{a} kommt zurück.",
              "move": "{a} legt {t} aus {p} in {q}.",
              "watch": "{a} schaut heimlich durch das Fenster zu, ohne dass es jemand im Raum bemerkt.",
              "reality": "Wo ist {t} jetzt?", "first": "Wo wird {b} zuerst nach {t} suchen?",
              "second": "Wo glaubt {a}, dass {b} nach {t} suchen wird?",
              "ask": "Antworte nur mit dem Namen des Behälters."}}


def tom_story(r, kind):
    """A simulated story: who is in the room or (secretly) watching at each move decides each belief.
    second = where a thinks b will look: a updates it only for moves a sees while seeing b in the room."""
    a, b, c = r.sample(range(6), 3)
    p0, p1, p2 = r.sample(range(6), 3)
    present, watching = {a, b, c}, set()
    belief = dict.fromkeys((a, b, c), p0)
    second = p0
    ev = [("put", a, p0)]

    def move(x, src, dst):
        nonlocal second
        ev.append(("move", x, (src, dst)))
        for y in present | watching:
            belief[y] = dst
        if a in present and b in present:
            second = dst

    if r.random() < (0.5 if kind == "second" else 0.85):  # balance "the first place" as a shortcut answer
        ev.append(("leave", b, None))
        present.discard(b)
        if kind != "second" and r.random() < 0.35:  # a secret watcher only where nobody's model of it matters
            ev.append(("watch", b, None))
            watching.add(b)
    move(a, p0, p1)
    if b not in present and r.random() < 0.3:
        ev.append(("back", b, None))
        present.add(b)
        watching.discard(b)
    if r.random() < 0.5:
        ev.append(("leave", a, None))
        present.discard(a)
        move(c, p1, p2)
        ev.append(("back", a, None))
        present.add(a)
    if b not in present:
        ev.append(("back", b, None))
    loc = ev[-1][2][1] if ev[-1][0] == "move" else next(e[2][1] for e in reversed(ev) if e[0] == "move")
    return {"a": a, "b": b, "t": r.randrange(6), "ev": ev,
            "gold": {"reality": loc, "first": belief[b], "second": second}[kind]}


def tom_text(st, lang, kind):
    T, P, N = TOM[lang], PLACES[lang], PEOPLE[lang]
    nom, acc, dat = THINGS[lang][st["t"]]
    acc_p = (lambda p: f"{DE_ART[p][0]} {p}") if lang == "de" else (lambda p: p)  # noqa: E731
    dat_p = (lambda p: f"{DE_ART[p][1]} {p}") if lang == "de" else (lambda p: p)  # noqa: E731
    lines = []
    for k, x, arg in st["ev"]:
        if k == "put":
            lines.append(T["put"].format(a=N[x], t=acc, p=acc_p(P[arg])))
        elif k == "move":
            lines.append(T["move"].format(a=N[x], t=acc, p=dat_p(P[arg[0]]), q=acc_p(P[arg[1]])))
        else:
            lines.append(T[k].format(a=N[x]))
    q = T[kind].format(a=N[st["a"]], b=N[st["b"]], t={"reality": nom}.get(kind, dat if lang == "de" else acc))
    return " ".join(lines) + "\n\n" + q[0].upper() + q[1:] + " " + T["ask"], P[st["gold"]]


def load_tom(lang, n, seed):
    items = []
    for i in range(n or 300):
        kind = ("reality", "first", "second")[i % 3]
        text, gold = tom_text(tom_story(rng_of("theory_of_mind", i), kind), lang, kind)
        items.append({"id": str(i), "lang": lang, "gold": gold, "kind": kind,
                      "messages": [{"role": "user", "content": text + " " + ASK[lang]}]})
    return items


def score_tom(item, reply):
    pred = final_answer(reply["content"]) or ""
    words = re.findall(r"\w+", pred.lower())
    places = {p.lower() for p in PLACES[item["lang"]]}
    named = [w for w in words if w in places]
    ok = named == [item["gold"].lower()] or (len(set(named)) == 1 and named[0] == item["gold"].lower())
    return {"correct": ok, "score": float(ok), "pred": pred, "gold": item["gold"],
            "detail": {"kind": item["kind"], f"ok_{item['kind']}": float(ok)}}


# ---------------------------------------------------------------- temporal
import datetime  # noqa: E402

WEEKDAYS = {"en": ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"],
            "de": ["montag", "dienstag", "mittwoch", "donnerstag", "freitag", "samstag", "sonntag"]}
TEMP = {"en": {"weekday": "{day} {d:%B %Y} was a {w}. Which day of the week is {n} days later?",
               "between": "How many days are there from {d:%Y-%m-%d} to {e:%Y-%m-%d}? (Count the days in between: "
                          "from a date to the next day is 1.)",
               "weeks": "Which date is {n} weeks before {d:%Y-%m-%d}? Answer as YYYY-MM-DD."},
        "de": {"weekday": "Der {d:%d.%m.%Y} war ein {w}. Welcher Wochentag ist {n} Tage später?",
               "between": "Wie viele Tage liegen zwischen dem {d:%d.%m.%Y} und dem {e:%d.%m.%Y}? (Von einem Tag zum "
                          "nächsten zählt als 1 Tag.)",
               "weeks": "Welches Datum liegt {n} Wochen vor dem {d:%d.%m.%Y}? Antworte als JJJJ-MM-TT."}}


def load_temporal(lang, n, seed):
    items = []
    for i in range(n or 300):
        r = rng_of("temporal", i)
        d = datetime.date(1800, 1, 1) + datetime.timedelta(days=r.randrange(110000))
        kind = ("weekday", "between", "weeks")[i % 3]
        if kind == "weekday":
            k = r.randint(40, 4000)
            q = TEMP[lang]["weekday"].format(d=d, day=d.day, w=WEEKDAYS[lang][d.weekday()].capitalize(), n=k)
            gold = WEEKDAYS[lang][(d + datetime.timedelta(days=k)).weekday()]
        elif kind == "between":
            e = d + datetime.timedelta(days=r.randint(30, 3000))
            q, gold = TEMP[lang]["between"].format(d=d, e=e), str((e - d).days)
        else:
            k = r.randint(5, 300)
            q, gold = TEMP[lang]["weeks"].format(d=d, n=k), (d - datetime.timedelta(weeks=k)).isoformat()
        items.append({"id": str(i), "lang": lang, "gold": gold, "kind": kind,
                      "messages": [{"role": "user", "content": q + "\n" + ASK[lang]}]})
    return items


def score_temporal(item, reply):
    pred = (final_answer(reply["content"]) or "").lower()
    if item["kind"] == "weekday":
        found = [w for w in WEEKDAYS["en"] + WEEKDAYS["de"] if re.search(rf"\b{w}\b", pred)]
        names = {WEEKDAYS["en"][WEEKDAYS[item["lang"]].index(item["gold"])], item["gold"]}
        ok = bool(found) and set(found) <= names
    elif item["kind"] == "between":
        ok = str(last_int(pred)) == item["gold"]
    else:
        m = re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})", pred) or re.search(r"(\d{1,2})\.(\d{1,2})\.(\d{4})", pred)
        if m and len(m[1]) != 4:
            m = (None, m[3], m[2], m[1])
        ok = bool(m) and f"{int(m[1]):04d}-{int(m[2]):02d}-{int(m[3]):02d}" == item["gold"]
    return {"correct": ok, "score": float(ok), "pred": pred[:80], "gold": item["gold"],
            "detail": {"kind": item["kind"], f"ok_{item['kind']}": float(ok)}}


# ---------------------------------------------------------------- state_tracking
ST = {"en": ("At the start, {start}. Then, in this order:\n{swaps}\nWho holds {x} at the end?",
             "{a} holds {t}", "{a} and {b} swap their items."),
      "de": ("Zu Beginn gilt: {start}. Dann, in dieser Reihenfolge:\n{swaps}\nWer hält am Ende {x}?",
             "{a} hält {t}", "{a} und {b} tauschen ihre Gegenstände.")}


def load_state(lang, n, seed):
    items = []
    tmpl, hold, swap = ST[lang]
    for i in range(n or 300):
        r = rng_of("state_tracking", i)
        k = 4 + i % 3
        who = r.sample(range(6), k)
        what = r.sample(range(6), k)
        steps = 10 + 10 * (i // 3 % 3)  # 10, 20, 30 swaps
        holder = dict(zip(who, what))
        lines = []
        for _ in range(steps):
            a, b = r.sample(who, 2)
            holder[a], holder[b] = holder[b], holder[a]
            lines.append("- " + swap.format(a=PEOPLE[lang][a], b=PEOPLE[lang][b]))
        x = r.choice(what)
        gold = PEOPLE[lang][next(p for p, t in holder.items() if t == x)]
        start = "; ".join(hold.format(a=PEOPLE[lang][p], t=THINGS[lang][t][1]) for p, t in zip(who, what))
        q = tmpl.format(start=start, swaps="\n".join(lines), x=THINGS[lang][x][1])
        items.append({"id": str(i), "lang": lang, "gold": gold, "steps": steps,
                      "messages": [{"role": "user", "content": q + "\n" + ASK[lang]}]})
    return items


def score_state(item, reply):
    pred = final_answer(reply["content"]) or ""
    named = {p for p in PEOPLE[item["lang"]] if re.search(rf"\b{p}\b", pred)}
    ok = named == {item["gold"]}
    return {"correct": ok, "score": float(ok), "pred": pred[:80], "gold": item["gold"],
            "detail": {f"ok_n{item['steps']}": float(ok)}}


# ---------------------------------------------------------------- math_variants
NAMES = ["Lena", "Jonas", "Mia", "Paul", "Sophie", "Elias", "Hannah", "Noah", "Lea", "Finn"]


def mt_bakery(r):
    a, b, c, p = r.randint(3, 9), r.randint(12, 36), r.randint(2, 9), r.randint(2, 6)
    n = a * b - c
    return {"en": f"A bakery bakes {a} trays of {b} rolls each. {c} rolls burn and are thrown away. The rest are sold "
                  f"in bags of {p}. How many full bags can be sold?",
            "de": f"Eine Bäckerei backt {a} Bleche mit je {b} Brötchen. {c} Brötchen verbrennen und werden weggeworfen. "
                  f"Der Rest wird in Tüten zu je {p} Stück verkauft. Wie viele volle Tüten können verkauft werden?",
            "ans": n // p,
            "plus": ({"en": " Each bag costs 3 euros. How many euros does the bakery earn from the full bags?",
                      "de": " Jede Tüte kostet 3 Euro. Wie viele Euro nimmt die Bäckerei mit den vollen Tüten ein?"},
                     n // p * 3),
            "noop": {"en": f" The bakery opened {r.randint(5, 40)} years ago.",
                     "de": f" Die Bäckerei wurde vor {r.randint(5, 40)} Jahren eröffnet."}}


def mt_savings(r):
    x, w, s_, g = r.choice(NAMES), r.randint(4, 15), r.randint(30, 200), r.randint(5, 40)
    return {"en": f"{x} has {s_} euros and saves {g} euros every week. How many euros does {x} have after {w} weeks?",
            "de": f"{x} hat {s_} Euro und spart jede Woche {g} Euro. Wie viele Euro hat {x} nach {w} Wochen?",
            "ans": s_ + w * g,
            "plus": ({"en": f" Then {x} spends half of it on a bike. How many euros are left?",
                      "de": f" Dann gibt {x} die Hälfte davon für ein Fahrrad aus. Wie viele Euro bleiben übrig?"},
                     (s_ + w * g) / 2),
            "noop": {"en": f" {x}'s brother saves a little less each week.",
                     "de": f" Der Bruder von {x} spart jede Woche etwas weniger."}}


def mt_train(r):
    v, h, m = r.choice([60, 80, 90, 120]), r.randint(2, 4), r.choice([15, 30, 45])
    return {"en": f"A train travels at {v} km/h for {h} hours and {m} minutes without stopping. How many kilometres "
                  f"does it cover?",
            "de": f"Ein Zug fährt {h} Stunden und {m} Minuten lang ohne Halt mit {v} km/h. Wie viele Kilometer legt er "
                  f"zurück?",
            "ans": v * h + v * m / 60,
            "plus": ({"en": " The return trip is 20 km shorter. How many kilometres are both trips together?",
                      "de": " Die Rückfahrt ist 20 km kürzer. Wie viele Kilometer sind beide Fahrten zusammen?"},
                     2 * (v * h + v * m / 60) - 20),
            "noop": {"en": " The train has 8 carriages, two of which are first class.",
                     "de": " Der Zug hat 8 Wagen, zwei davon erster Klasse."}}


def mt_garden(r):
    k = r.choice([2, 3, 4])
    l_, w = k * r.randint(3, 8), k * r.randint(2, 5)
    return {"en": f"A rectangular garden is {l_} m long and {w} m wide. A fence is built all around it, with one post "
                  f"every {k} m (the corners included, no doubles). How many posts are needed?",
            "de": f"Ein rechteckiger Garten ist {l_} m lang und {w} m breit. Rundherum wird ein Zaun gebaut, mit einem "
                  f"Pfosten alle {k} m (Ecken eingeschlossen, keine doppelt). Wie viele Pfosten werden gebraucht?",
            # a post on every corner needs both sides divisible by the spacing; other draws are skipped
            "ans": 2 * (l_ + w) / k if l_ % k == 0 and w % k == 0 else None,
            "plus": ({"en": " Each post costs 12 euros. What do all posts cost in euros?",
                      "de": " Jeder Pfosten kostet 12 Euro. Was kosten alle Pfosten in Euro?"},
                     2 * (l_ + w) / k * 12 if l_ % k == 0 and w % k == 0 else None),
            "noop": {"en": " The garden is planted with tomatoes and beans.",
                     "de": " Im Garten wachsen Tomaten und Bohnen."}}


def mt_class(r):
    n, pct, b = r.choice([20, 24, 25, 28, 30, 32]), r.choice([25, 50, 75]), r.randint(2, 5)
    girls = n * pct // 100
    return {"en": f"A class has {n} pupils, {pct}% of them are girls. Every boy brings {b} pencils. How many pencils "
                  f"do the boys bring together?",
            "de": f"Eine Klasse hat {n} Kinder, {pct} % davon sind Mädchen. Jeder Junge bringt {b} Stifte mit. Wie viele "
                  f"Stifte bringen die Jungen zusammen mit?",
            "ans": (n - girls) * b if n * pct % 100 == 0 else None,
            "plus": ({"en": " The girls bring twice as many pencils each. How many pencils are there in total?",
                      "de": " Die Mädchen bringen jeweils doppelt so viele Stifte mit. Wie viele Stifte sind es insgesamt?"},
                     (n - girls) * b + girls * 2 * b if n * pct % 100 == 0 else None),
            "noop": {"en": " Last year the class had two pupils more.",
                     "de": " Im letzten Jahr hatte die Klasse zwei Kinder mehr."}}


def mt_recipe(r):
    base, want, f, e = r.choice([2, 4]), r.choice([6, 8, 10, 12]), r.choice([200, 250, 300]), r.choice([2, 3])
    return {"en": f"A recipe for {base} people needs {f} g of flour and {e} eggs. How many grams of flour are needed "
                  f"for {want} people?",
            "de": f"Ein Rezept für {base} Personen braucht {f} g Mehl und {e} Eier. Wie viel Gramm Mehl braucht man für "
                  f"{want} Personen?",
            "ans": f * want / base,
            "plus": ({"en": " Flour is sold in 1 kg packs. How many packs must be bought?",
                      "de": " Mehl gibt es in 1-kg-Packungen. Wie viele Packungen muss man kaufen?"},
                     math.ceil(f * want / base / 1000)),
            "noop": {"en": " The eggs are from free-range hens.", "de": " Die Eier stammen aus Freilandhaltung."}}


def mt_shop(r):
    p, q, d = r.choice([12, 15, 18, 20, 24]), r.randint(3, 9), r.choice([10, 20, 25])
    total = p * q
    return {"en": f"A shirt costs {p} euros. Someone buys {q} shirts and gets {d}% off the total. How many euros do "
                  f"they pay?",
            "de": f"Ein Hemd kostet {p} Euro. Jemand kauft {q} Hemden und bekommt {d} % Rabatt auf den Gesamtpreis. Wie "
                  f"viele Euro zahlt die Person?",
            "ans": total * (100 - d) / 100,
            "plus": ({"en": " They pay with a 200-euro note. How much change do they get?",
                      "de": " Die Person zahlt mit einem 200-Euro-Schein. Wie viel Wechselgeld bekommt sie?"},
                     200 - total * (100 - d) / 100 if total * (100 - d) / 100 <= 200 else None),
            "noop": {"en": " Last week the shirts were 10% cheaper.",
                     "de": " Letzte Woche waren die Hemden 10 % billiger."}}


def mt_ages(r):
    x, y = r.sample(NAMES, 2)
    a, d, t = r.randint(5, 15), r.randint(2, 9), r.randint(3, 12)
    return {"en": f"{x} is {a} years old. {y} is {d} years older than {x}. How old will they be together in {t} years?",
            "de": f"{x} ist {a} Jahre alt. {y} ist {d} Jahre älter als {x}. Wie alt sind beide zusammen in {t} Jahren?",
            "ans": a + a + d + 2 * t,
            "plus": ({"en": f" Their mother is three times as old as {x} is now. How old is she in {t} years?",
                      "de": f" Ihre Mutter ist dreimal so alt, wie {x} jetzt ist. Wie alt ist sie in {t} Jahren?"},
                     3 * a + t),
            "noop": {"en": f" {y} was born in spring.", "de": f" {y} wurde im Frühling geboren."}}


TEMPLATES = [mt_bakery, mt_savings, mt_train, mt_garden, mt_class, mt_recipe, mt_shop, mt_ages]
# (template, exact phrase in the generated text as a regex, its vague replacement per language)
VAGUE = {mt_bakery: {"en": (r"\. \d+ rolls burn", ". Some rolls burn"), "de": (r"\. \d+ Brötchen verbrennen",
                                                                                 ". Einige Brötchen verbrennen")},
         mt_savings: {"en": (r"saves \d+ euros every week", "saves some money every week"),
                      "de": (r"spart jede Woche \d+ Euro", "spart jede Woche etwas Geld")},
         mt_train: {"en": (r"for \d+ hours and \d+ minutes", "for a while"),
                    "de": (r"fährt \d+ Stunden und \d+ Minuten lang", "fährt eine Weile")},
         mt_recipe: {"en": (r"needs \d+ g of flour", "needs some flour"),
                     "de": (r"braucht \d+ g Mehl", "braucht etwas Mehl")},
         mt_shop: {"en": (r"gets \d+% off", "gets a discount on"), "de": (r"bekommt \d+ % Rabatt", "bekommt Rabatt")}}
MISSING_ASK = {"en": " If the question cannot be answered from the information given, finish with \"Answer: cannot "
                     "be determined\".",
               "de": " Wenn sich die Frage mit den gegebenen Angaben nicht beantworten lässt, schließe mit "
                     "\"Antwort: nicht bestimmbar\" ab."}
ABSTAIN_RE = re.compile(r"cannot be determined|can't be determined|not determinable|insufficient|nicht bestimmbar|"
                        r"nicht bestimmt werden|nicht beantwort|unbestimmt|nicht genug", re.I)


def load_math_variants(lang, n, seed):
    items, i = [], -1
    while len(items) < (n or 360):
        i += 1
        r = rng_of("math_variants", i)
        variant = ("base", "plus", "noop")[i % 3]
        t = TEMPLATES[(i // 3) % len(TEMPLATES)](r)
        if variant == "plus":
            text, ans = t[lang] + t["plus"][0][lang], t["plus"][1]
        else:
            text, ans = t[lang] + (t["noop"][lang] if variant == "noop" else ""), t["ans"]
        if variant == "noop":  # the distractor goes before the question, where it reads like a premise
            q_at = t[lang].rfind(". ") + 2
            text = t[lang][:q_at] + t["noop"][lang].strip() + " " + t[lang][q_at:]
        if ans is None or ans != int(ans) or ans < 0:
            continue  # a non-integer or impossible instance: draw the next id (same in both languages)
        items.append({"id": str(i), "lang": lang, "gold": int(ans), "variant": variant,
                      "messages": [{"role": "user", "content": text + "\n" + ASK[lang]}]})
    return items


def score_math_variants(item, reply):
    r = score_invented_ops(item | {"depth": 0}, reply)
    return r | {"detail": {"variant": item["variant"], f"ok_{item['variant']}": r["detail"]["ok_d0"]}}


# ---------------------------------------------------------------- dialog_state
DS = {"en": {"open": "I'm planning a dinner on Saturday. Please keep track of the guest list for me; just confirm "
                     "briefly each time.",
             "add": "{n} is coming.", "add_veg": "{n} is coming and eats vegetarian.",
             "cancel": "{n} can't come after all.", "veg": "By the way, {n} eats vegetarian now.",
             "meat": "Correction: {n} does eat meat after all.",
             "ask": "How many guests are coming in total (not counting me), and which of them eat vegetarian? "
                    "Finish with a line \"Answer: <number>; <vegetarian names separated by commas, or none>\"."},
      "de": {"open": "Ich plane am Samstag ein Abendessen. Bitte führe für mich die Gästeliste; bestätige jeweils nur "
                     "kurz.",
             "add": "{n} kommt.", "add_veg": "{n} kommt und isst vegetarisch.",
             "cancel": "{n} kann doch nicht kommen.", "veg": "Übrigens, {n} isst jetzt vegetarisch.",
             "meat": "Korrektur: {n} isst doch Fleisch.",
             "ask": "Wie viele Gäste kommen insgesamt (ohne mich), und wer von ihnen isst vegetarisch? Schließe mit "
                    "einer Zeile \"Antwort: <Zahl>; <vegetarische Namen mit Kommas getrennt, oder keine>\" ab."}}


def load_dialog(lang, n, seed):
    items = []
    for i in range(n or 200):
        r = rng_of("dialog_state", i)
        names = r.sample(PEOPLE["en"] + ["Gina", "Hugo", "Ida", "Jan"], r.randint(5, 8))
        guests, veg, events = [], set(), []
        for nm in names:
            v = r.random() < 0.35
            guests.append(nm)
            if v:
                veg.add(nm)
            events.append(("add_veg" if v else "add", nm))
        gone = r.choice(guests)  # one cancellation, one diet change (to or from vegetarian) among the rest
        events.append(("cancel", gone))
        guests.remove(gone)
        veg.discard(gone)
        other = r.choice(guests)
        events.append(("meat" if other in veg else "veg", other))
        veg ^= {other}
        r.shuffle(events)
        # a cancellation or change must come after that person's arrival announcement
        events.sort(key=lambda e: (0 if e[0] in ("add", "add_veg") else 1))
        late = events[len(names):]
        early = events[:len(names)]
        chunks, k = [], 0
        while k < len(early):
            step = r.randint(1, 3)
            chunks.append(early[k:k + step])
            k += step
        chunks[-1] = chunks[-1] + late if r.random() < 0.5 else chunks[-1]
        if late[0] not in chunks[-1]:
            chunks.append(late)
        T = DS[lang]
        turns = [[{"role": "user", "content": T["open"] + " " + " ".join(T[k].format(n=nm) for k, nm in chunks[0])}]]
        turns += [[{"role": "user", "content": " ".join(T[k].format(n=nm) for k, nm in c)}] for c in chunks[1:]]
        turns.append([{"role": "user", "content": T["ask"]}])
        items.append({"id": str(i), "lang": lang, "gold": (len(guests), sorted(veg)), "turns": turns,
                      "messages": turns[0]})
    return items


def score_dialog(item, reply):
    line = final_answer(reply["content"]) or ""
    m = re.search(r"\d+", line)  # the count, then the names after it (";" or "," or a dash between them)
    count_ok = bool(m) and int(m[0]) == item["gold"][0]
    tail = line[m.end():] if m else ""
    named = sorted(p for p in PEOPLE["en"] + ["Gina", "Hugo", "Ida", "Jan"] if re.search(rf"\b{p}\b", tail))
    names_ok = named == item["gold"][1]
    ok = count_ok and names_ok
    return {"correct": ok, "score": (count_ok + names_ok) / 2, "pred": line[:120],
            "gold": f"{item['gold'][0]}; {', '.join(item['gold'][1]) or 'none'}",
            "detail": {"ok_count": float(count_ok), "ok_names": float(names_ok)}}


# ---------------------------------------------------------------- code_trace
def code_block(r, k):
    """One statement group over the variables xs (list of int), s (str), n (int); returns source lines."""
    kind = r.choice(["listmath", "loopif", "string", "counter", "while", "listcomp"])
    a, b = r.randint(2, 5), r.randint(1, 9)
    if kind == "listmath":
        return [f"xs = [x * {a} - {b} for x in xs]", f"n = n + sum(xs[{r.randint(0, 2)}:])"]
    if kind == "loopif":
        return ["for i, x in enumerate(xs):", f"    if x % {a} == {r.randint(0, a - 1)}:", f"        n += i * {b}",
                "    else:", "        n -= 1"]
    if kind == "string":
        return [f"s = s[{r.randint(1, 3)}:] + s[:{r.randint(1, 3)}]", f"s = s.replace('{r.choice('aeiou')}', '{b}')",
                "n += len(s)"]
    if kind == "counter":
        return ["c = {}", "for ch in s:", "    c[ch] = c.get(ch, 0) + 1",
                f"n += max(c.values()) * {a} + len(c)"]
    if kind == "while":
        return [f"m = abs(n) + {b * 10}", "steps = 0", "while m > 1:",
                "    m = m // 2 if m % 2 == 0 else m - 1", "    steps += 1", "n += steps"]
    return [f"xs = [x for x in xs if x > {r.randint(-5, 15)}] or [{b}]", f"n = n * {a} % 97 + len(xs)"]


def make_program(r, blocks):
    xs = [r.randint(-5, 20) for _ in range(r.randint(4, 7))]
    s = "".join(r.choice("abcdeiou") for _ in range(r.randint(5, 9)))
    lines = [f"xs = {xs}", f"s = '{s}'", f"n = {r.randint(0, 9)}"]
    for k in range(blocks):
        lines += code_block(r, k)
    lines.append("print(n, s, xs[-1])")
    return "\n".join(lines)


def run_program(src):
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(src, "<generated>", "exec"), {"__builtins__": {"print": print, "sum": sum, "len": len,  # noqa: S102
                                                                    "enumerate": enumerate, "max": max, "abs": abs}})
    return buf.getvalue().strip()


CT_TEXT = {"en": "What does this Python program print? Work it out step by step without running it.\n\n```python\n"
                 "{src}\n```",
           "de": "Was gibt dieses Python-Programm aus? Rechne es Schritt für Schritt durch, ohne es auszuführen."
                 "\n\n```python\n{src}\n```"}


def load_code_trace(lang, n, seed):
    items = []
    for i in range(n or 240):
        r = rng_of("code_trace", i)
        blocks = 2 + i % 3
        src = make_program(r, blocks)
        items.append({"id": str(i), "lang": lang, "gold": run_program(src), "blocks": blocks,
                      "messages": [{"role": "user", "content": CT_TEXT[lang].format(src=src) + "\n" + ASK[lang]}]})
    return items


def score_code_trace(item, reply):
    pred = (final_answer(reply["content"]) or "").strip("`").strip()
    ok = " ".join(pred.split()) == " ".join(item["gold"].split())
    return {"correct": ok, "score": float(ok), "pred": pred[:120], "gold": item["gold"],
            "detail": {f"ok_b{item['blocks']}": float(ok)}}


# ---------------------------------------------------------------- spatial
DIRS = ["N", "E", "S", "W"]
DIR_NAME = {"en": {"N": "north", "E": "east", "S": "south", "W": "west"},
            "de": {"N": "Norden", "E": "Osten", "S": "Süden", "W": "Westen"}}
SP = {"en": {"start": "You stand on cell ({x}, {y}) of a grid, facing {d}. x grows to the east, y grows to the north.",
             "L": "Turn left.", "R": "Turn right.", "U": "Turn around.", "F": "Walk {k} cells forward.", "F1": "Walk 1 cell forward.",
             "ask": "Where do you stand now, and which way do you face? Answer as \"(x, y) direction\"."},
      "de": {"start": "Du stehst auf Feld ({x}, {y}) eines Rasters und schaust nach {d}. x wächst nach Osten, y "
                      "wächst nach Norden.",
             "L": "Dreh dich nach links.", "R": "Dreh dich nach rechts.", "U": "Dreh dich um.",
             "F": "Geh {k} Felder geradeaus.", "F1": "Geh 1 Feld geradeaus.",
             "ask": "Wo stehst du jetzt und in welche Richtung schaust du? Antworte als \"(x, y) Richtung\"."}}
STEP = {"N": (0, 1), "E": (1, 0), "S": (0, -1), "W": (-1, 0)}


def load_spatial(lang, n, seed):
    items = []
    for i in range(n or 240):
        r = rng_of("spatial", i)
        x, y, d = r.randint(-3, 3), r.randint(-3, 3), r.randrange(4)
        T = SP[lang]
        lines = [T["start"].format(x=x, y=y, d=DIR_NAME[lang][DIRS[d]])]
        for _ in range(8 + 6 * (i % 3)):
            op = r.choice("LRUFFF")
            if op == "F":
                k = r.randint(1, 5)
                x, y = x + STEP[DIRS[d]][0] * k, y + STEP[DIRS[d]][1] * k
                lines.append(T["F1"] if k == 1 else T["F"].format(k=k))
            else:
                d = (d + {"L": -1, "R": 1, "U": 2}[op]) % 4
                lines.append(T[op])
        items.append({"id": str(i), "lang": lang, "gold": (x, y, DIRS[d]), "steps": 8 + 6 * (i % 3),
                      "messages": [{"role": "user", "content": " ".join(lines) + "\n" + T["ask"] + "\n" + ASK[lang]}]})
    return items


def score_spatial(item, reply):
    pred = (final_answer(reply["content"]) or "").replace("−", "-")
    m = re.search(r"\(\s*(-?\d+)\s*[,;]\s*(-?\d+)\s*\)", pred)
    names = {v.lower(): k for lang in DIR_NAME for k, v in DIR_NAME[lang].items()}
    dirs = {names[w] for w in re.findall(r"[a-zäöüß]+", pred.lower()) if w in names}
    ok = bool(m) and (int(m[1]), int(m[2])) == item["gold"][:2] and dirs == {item["gold"][2]}
    return {"correct": ok, "score": float(ok), "pred": pred[:80], "gold": f"({item['gold'][0]}, {item['gold'][1]}) "
            f"{DIR_NAME[item['lang']][item['gold'][2]]}", "detail": {f"ok_s{item['steps']}": float(ok)}}


# ---------------------------------------------------------------- if_gen
IF_TOPICS = {"en": ["the history of bicycles", "why sleep matters", "how to start composting", "the moon landing",
                    "learning to play chess", "a rainy day in the city", "saving energy at home", "the life of bees",
                    "remote work", "the first day at a new job", "volcanoes", "public libraries"],
             "de": ["die Geschichte des Fahrrads", "warum Schlaf wichtig ist", "wie man mit dem Kompostieren "
                    "anfängt", "die Mondlandung", "Schach lernen", "ein Regentag in der Stadt", "Energiesparen zu "
                    "Hause", "das Leben der Bienen", "Homeoffice", "der erste Tag im neuen Job", "Vulkane",
                    "öffentliche Bibliotheken"]}
IF_WORDS = {"en": ["window", "orange", "bridge", "silver", "garden"], "de": ["Fenster", "Orange", "Brücke", "Silber",
                                                                          "Garten"]}
IF_C = {  # name: (en text, de text, checker(reply, arg)) ; arg drawn by IF_ARG
    "sentences": ("Write exactly {a} sentences.", "Schreibe genau {a} Sätze."),
    "bullets": ("Use exactly {a} bullet points (lines starting with \"- \") and nothing else.",
                "Verwende genau {a} Aufzählungspunkte (Zeilen, die mit \"- \" beginnen) und sonst nichts."),
    "include": ("Use the word \"{a}\" at least twice.", "Verwende das Wort \"{a}\" mindestens zweimal."),
    "lower": ("Write everything in lowercase letters.", "Schreibe alles in Kleinbuchstaben."),
    "limit": ("Use fewer than {a} words.", "Verwende weniger als {a} Wörter."),
    "title": ("Start with a title wrapped in double angle brackets, like <<title>>.",
              "Beginne mit einem Titel in doppelten spitzen Klammern, etwa <<Titel>>."),
    "ps": ("End with a postscript starting with \"P.S.\".", "Ende mit einem Nachsatz, der mit \"P.S.\" beginnt."),
    "ending": ("Finish your reply with the exact phrase \"{a}\". No other words may follow it.",
               "Beende deine Antwort mit genau dem Satz \"{a}\". Danach darf nichts mehr folgen."),
    "nocomma": ("Do not use any commas.", "Verwende keine Kommas."),
}
IF_ENDINGS = {"en": ["Is there anything else I can help with?", "That is all for today."],
              "de": ["Kann ich sonst noch helfen?", "Das ist alles für heute."]}
IF_CLASH = [{"sentences", "bullets"}, {"lower", "title"}, {"ps", "ending"}, {"lower", "ending"}, {"lower", "include"},
            {"lower", "ps"}, {"sentences", "ending"}, {"bullets", "ending"}, {"bullets", "title"}, {"bullets", "ps"}]
IF_ASK = {"en": "Write a short text about {t}.", "de": "Schreibe einen kurzen Text zum Thema: {t}."}


PS = r"(?m)^\W*P\.?\s?S\b"  # "P.S.", "**P.S.**", "PS:"
ABBREV = ("z. B.", "z.B.", "d. h.", "d.h.", "u. a.", "bzw.", "ca.", "usw.", "etc.", "e.g.", "i.e.", "Dr.", "Mr.",
          "Mrs.", "Nr.", "vs.")


def count_sentences(text):
    """Sentence ends (., !, ? possibly followed by closing quotes/brackets, then space or end) plus an unterminated
    tail; abbreviations and decimal points do not end a sentence."""
    for ab in ABBREV:
        text = text.replace(ab, ab.replace(".", "\x00"))
    text = re.sub(r"(?<=\d)\.(?=\d)", "\x00", text)
    ends = list(re.finditer(r"[.!?]+[\"'”“»«)\]*_]*(?=\s|$)", text))
    tail = text[ends[-1].end():] if ends else text
    return len(ends) + (1 if re.search(r"\w", tail) else 0)


def if_check(name, arg, text, lang):
    body = re.sub(r"^\W*<<[^>]*>>[*_ ]*", "", text.strip())
    body = re.split(PS, body)[0]
    if name == "sentences":
        return count_sentences(body.strip()) == arg
    if name == "bullets":
        lines = [x for x in text.strip().splitlines() if x.strip()]
        return len(lines) == arg and all(x.lstrip().startswith(("- ", "* ")) for x in lines)
    if name == "include":
        return len(re.findall(rf"\b{re.escape(arg)}\b", text, re.I)) >= 2
    if name == "lower":
        return text == text.lower() and re.search(r"[a-zäöü]", text) is not None
    if name == "limit":
        return len(text.split()) < arg
    if name == "title":
        return re.match(r"\W*<<[^<>\n]+>>", text) is not None
    if name == "ps":
        return re.search(PS, text) is not None
    if name == "ending":
        return text.strip().rstrip("*_`\"'„“”» ").endswith(arg)
    if name == "nocomma":
        return "," not in text
    raise KeyError(name)


def load_if_gen(lang, n, seed):
    items = []
    for i in range(n or 200):
        r = rng_of("if_gen", i)
        k = 2 + i % 2
        while True:
            names = r.sample(sorted(IF_C), k)
            if not any(c <= set(names) for c in IF_CLASH):
                break
        args = {}
        for nm in names:
            args[nm] = {"sentences": r.randint(3, 6), "bullets": r.randint(3, 5), "limit": r.choice([60, 80, 100]),
                        "include": r.randrange(5), "ending": r.randrange(2)}.get(nm)
        topic = r.randrange(len(IF_TOPICS["en"]))
        shown = {nm: IF_WORDS[lang][a] if nm == "include" else IF_ENDINGS[lang][a] if nm == "ending" else a
                 for nm, a in args.items()}
        text = IF_ASK[lang].format(t=IF_TOPICS[lang][topic]) + " " + " ".join(
            IF_C[nm][0 if lang == "en" else 1].format(a=shown[nm]) for nm in names)
        items.append({"id": str(i), "lang": lang, "checks": shown, "gold": sorted(names),
                      "messages": [{"role": "user", "content": text}]})
    return items


def score_if_gen(item, reply):
    text = reply["content"]
    res = {nm: if_check(nm, a, text, item["lang"]) for nm, a in item["checks"].items()}
    met = sum(res.values()) / len(res)
    return {"correct": all(res.values()), "score": round(met, 4), "pred": text[:200], "gold": ",".join(item["gold"]),
            "detail": {f"ok_{nm}": float(v) for nm, v in res.items()}}


# ---------------------------------------------------------------- logic_grid
import itertools  # noqa: E402

LG_ATTR = {"name": ["Anna", "Ben", "Clara", "David"],
           "pet": {"en": ["cat", "dog", "fish", "bird"], "de": ["Katze", "Hund", "Fisch", "Vogel"]},
           "drink": {"en": ["tea", "coffee", "milk", "juice"], "de": ["Tee", "Kaffee", "Milch", "Saft"]}}
LG_TEXT = {"en": {"intro": "{n} houses stand in a row, numbered 1 to {n} from left to right. In each lives one person "
                           "with one pet and one favourite drink; no two houses share a person, pet or drink. The "
                           "people are {names}; the pets are {pets}; the drinks are {drinks}.",
                  "same": "{a} lives in the same house as {b}.", "left": "{a} lives directly left of {b}.",
                  "next": "{a} lives next to {b}.", "pos": "{a} lives in house {k}.",
                  "not": "{a} does not live in house {k}.",
                  "ask": "Which {attr} belongs to {who}? Finish with a line \"Answer: <{attr}>\"."},
           "de": {"intro": "{n} Häuser stehen in einer Reihe, von links nach rechts nummeriert von 1 bis {n}. In "
                           "jedem wohnt eine Person mit einem Haustier und einem Lieblingsgetränk; keine zwei Häuser "
                           "teilen Person, Haustier oder Getränk. Die Personen sind {names}; die Haustiere sind {pets}; "
                           "die Getränke sind {drinks}.",
                  "same": "{a} wohnt im selben Haus wie {b}.", "left": "{a} wohnt direkt links neben {b_dat}.",
                  "next": "{a} wohnt neben {b_dat}.", "pos": "{a} wohnt in Haus {k}.",
                  "not": "{a} wohnt nicht in Haus {k}.",
                  "ask": "Welches {attr} gehört zu {who}? Schließe mit einer Zeile \"Antwort: <{attr}>\" ab."}}
LG_ATTR_NAME = {"en": {"pet": "pet", "drink": "drink"}, "de": {"pet": "Haustier", "drink": "Getränk"}}


def lg_phrase(lang, kind, v, case="nom", subject=False):
    """'the cat owner' / 'Anna' / 'the tea drinker'; German 'die Person mit der Katze', dative 'der Person ...',
    a relative clause closed by a comma when the verb follows (subject=True)."""
    if kind == "name":
        return v
    if lang == "en":
        return f"the {v} owner" if kind == "pet" else f"the {v} drinker"
    art = "der" if case == "dat" else "die"
    if kind == "pet":
        return f"{art} Person mit {'der' if v == 'Katze' else 'dem'} {v}"
    return f"{art} Person, die {v} trinkt" + ("," if subject else "")


def lg_holds(clue, sol):
    """sol: {kind: tuple of values by house}; clue = (type, (kind, idx), (kind, idx) | k)."""
    t, (ka, ia), b = clue
    pa = sol[ka].index(ia)
    if t == "pos":
        return pa == b
    if t == "not":
        return pa != b
    kb, ib = b
    pb = sol[kb].index(ib)
    return pa == pb if t == "same" else pa + 1 == pb if t == "left" else abs(pa - pb) == 1


def load_logic_grid(lang, n, seed):
    items = []
    for i in range(n or 200):
        r = rng_of("logic_grid", i)
        h = 3 + i % 2
        kinds = ("name", "pet", "drink")
        truth = {k: tuple(r.sample(range(4), h)) for k in kinds}  # value index per house
        space = [dict(zip(kinds, combo)) for combo in itertools.product(
            *(list(itertools.permutations(truth[k])) for k in kinds))]
        clues = []
        while len(space) > 1:
            t = r.choice(["same", "same", "left", "next", "pos", "not"])
            ka = r.choice(kinds)
            ia = r.choice(truth[ka])
            pa = truth[ka].index(ia)
            if t in ("pos", "not"):
                b = pa if t == "pos" else r.choice([p for p in range(h) if p != pa])
            else:
                kb = r.choice([k for k in kinds if k != ka])
                cands = [p for p in range(h) if (p == pa if t == "same" else p == pa + 1 if t == "left"
                                                 else abs(p - pa) == 1)]
                if not cands:
                    continue
                b = (kb, truth[kb][r.choice(cands)])
            clue = (t, (ka, ia), b)
            if clue in clues:
                continue
            left = [s for s in space if lg_holds(clue, s)]
            if len(left) < len(space):  # only clues that cut the space
                space, clues = left, clues + [clue]
        who = r.randrange(h)
        attr = r.choice(["pet", "drink"])
        vals = lambda k: LG_ATTR[k] if k == "name" else LG_ATTR[k][lang]  # noqa: E731
        T = LG_TEXT[lang]
        lines = []
        for t, (ka, ia), b in clues:
            a = lg_phrase(lang, ka, vals(ka)[ia], subject=True)
            if t in ("pos", "not"):
                s_ = T[t].format(a=a, k=b + 1)
            else:
                v = vals(b[0])[b[1]]
                s_ = T[t].format(a=a, b=lg_phrase(lang, b[0], v), b_dat=lg_phrase(lang, b[0], v, "dat"))
            lines.append(s_[0].upper() + s_[1:])
        name = vals("name")[truth["name"][who]]
        gold = vals(attr)[truth[attr][who]]
        used = {k: ", ".join(sorted(vals(k)[x] for x in truth[k])) for k in kinds}  # every value in play is named
        text = T["intro"].format(n=h, names=used["name"], pets=used["pet"], drinks=used["drink"]) + "\n" + \
            "\n".join(f"- {x}" for x in lines) + "\n" + T["ask"].format(attr=LG_ATTR_NAME[lang][attr], who=name)
        items.append({"id": str(i), "lang": lang, "gold": gold, "houses": h, "attr": attr,
                      "messages": [{"role": "user", "content": text}]})
    return items


def score_logic_grid(item, reply):
    line = (final_answer(reply["content"]) or "").lower()
    options = LG_ATTR[item["attr"]][item["lang"]]
    found = {o for o in options if re.search(rf"\b{o.lower()}\b", line)}
    ok = found == {item["gold"]}
    return {"correct": ok, "score": float(ok), "pred": line[:60], "gold": item["gold"],
            "detail": {f"ok_h{item['houses']}": float(ok)}}


# ---------------------------------------------------------------- sci_calc
def parse_number(text, lang="en", first=False):
    """The last number in `text` as a float: '1.234,5' (DE), '1,234.5' (EN), '12,5', '0.75', '−3'. A lone dot before
    exactly three digits ('18.000') is a thousands separator in German and a decimal point in English."""
    toks = re.findall(r"[−-]?\d[\d.,  ]*\d|[−-]?\d", (text or "").replace("−", "-"))
    if not toks:
        return None
    t = re.sub(r"[  ]", "", toks[0 if first else -1])
    if "," in t and "." in t:  # the later separator is the decimal one
        t = t.replace(".", "").replace(",", ".") if t.rfind(",") > t.rfind(".") else t.replace(",", "")
    elif "," in t:  # "1,234" is a thousand in English but 1.234 in German
        t = t.replace(",", "") if lang != "de" and re.fullmatch(r"-?\d{1,3}(,\d{3})+", t) else t.replace(",", ".")
    elif re.fullmatch(r"-?\d{1,3}(\.\d{3}){2,}", t) or (lang == "de" and re.fullmatch(r"-?\d{1,3}\.\d{3}", t)):
        t = t.replace(".", "")
    try:
        return float(t)
    except ValueError:
        return None


def sci_item(r, kind, lang):
    de = lang == "de"
    if kind == "ohm":
        R, mA = r.choice([47, 100, 220, 330, 470, 1000, 2200]), r.choice([5, 12, 20, 35, 50, 80])
        q = (f"Durch einen Widerstand von {R} Ohm fließt ein Strom von {mA} mA. Wie groß ist die Spannung in Volt?"
             if de else f"A current of {mA} mA flows through a {R} ohm resistor. What is the voltage in volts?")
        return q, R * mA / 1000
    if kind == "speed":
        v, mins = r.choice([36, 54, 72, 90, 108]), r.choice([10, 25, 40, 75, 95])
        q = (f"Ein Auto fährt {mins} Minuten lang mit konstant {v} km/h. Wie viele Meter legt es zurück?" if de else
             f"A car drives at a constant {v} km/h for {mins} minutes. How many metres does it cover?")
        return q, v * 1000 * mins / 60
    if kind == "power":
        w, h, d, ct = r.choice([60, 150, 800, 1200, 2000]), r.choice([2, 3, 5, 8]), r.choice([30, 90, 365]), \
            r.choice([28, 32, 35, 40])
        q = (f"Ein Gerät mit {w} Watt läuft {h} Stunden pro Tag an {d} Tagen. Strom kostet {ct} Cent pro kWh. Wie "
             f"viel kostet das in Euro?" if de else f"A {w}-watt device runs {h} hours a day for {d} days. "
             f"Electricity costs {ct} cents per kWh. What does that cost in euros?")
        return q, w * h * d / 1000 * ct / 100
    if kind == "density":
        rho, cm3 = r.choice([2700, 7870, 8960, 920, 1000, 11340]), r.choice([10, 20, 50, 100])  # Al, Fe, Cu, ice, ...
        g = rho * cm3 // 1000
        q = (f"Ein Körper hat eine Masse von {g} g und ein Volumen von {cm3} cm³. Wie groß ist seine Dichte in "
             f"kg/m³?" if de else f"An object has a mass of {g} g and a volume of {cm3} cm³. What is its density "
             f"in kg/m³?")
        return q, g / cm3 * 1000
    a, p, b, s = r.choice([200, 250, 400, 500]), r.choice([10, 20, 30]), r.choice([100, 300, 600]), r.choice([2, 5, 40])
    q = (f"Man mischt {a} ml einer {p}-prozentigen Lösung mit {b} ml einer {s}-prozentigen Lösung. Wie viel Prozent "
         f"hat die Mischung?" if de else f"You mix {a} ml of a {p}% solution with {b} ml of a {s}% solution. What "
         f"percentage does the mixture have?")
    return q, (a * p + b * s) / (a + b)


SCI_KINDS = ("ohm", "speed", "power", "density", "mix")


def load_sci_calc(lang, n, seed):
    items = []
    for i in range(n or 200):
        kind = SCI_KINDS[i % len(SCI_KINDS)]
        q, gold = sci_item(rng_of("sci_calc", i), kind, lang)
        items.append({"id": str(i), "lang": lang, "gold": round(gold, 6), "kind": kind,
                      "messages": [{"role": "user", "content": q + "\n" + ASK[lang]}]})
    return items


def score_sci_calc(item, reply):
    # the first number on the answer line: "6000 m (6 km)" and "0,235 V (235 mV)" mean the first
    v = answer_value(final_answer(reply["content"]), item["lang"])
    ok = v is not None and abs(v - item["gold"]) <= 0.01 * abs(item["gold"])
    return {"correct": ok, "score": float(ok), "pred": v, "gold": item["gold"],
            "detail": {f"ok_{item['kind']}": float(ok)}}


def load_math_missing(lang, n, seed):
    items, i = [], -1
    templ = list(VAGUE)
    while len(items) < (n or 200):
        i += 1
        r = rng_of("math_missing", i)
        t = templ[(i // 2) % len(templ)](r)
        text = t[lang]
        missing = i % 2 == 1
        if missing:
            pat, repl = VAGUE[templ[(i // 2) % len(templ)]][lang]
            text, k = re.subn(pat, repl, text, count=1)
            if k != 1:
                raise RuntimeError(f"math_missing: {pat!r} not in {text!r}")
        elif t["ans"] is None or t["ans"] != int(t["ans"]):
            continue
        items.append({"id": str(i), "lang": lang, "missing": missing,
                      "gold": None if missing else int(t["ans"]),
                      "messages": [{"role": "user", "content": text + MISSING_ASK[lang] + "\n" + ASK[lang]}]})
    return items


def score_math_missing(item, reply):
    line = final_answer(reply["content"]) or ""
    abstained = bool(ABSTAIN_RE.search(line))
    if item["missing"]:
        m = ABSTAIN_RE.search(line)  # numbers after the abstention are its explanation, numbers before are a guess
        ok = abstained and last_int(line[:m.start()]) is None
        detail = {"ok_missing": float(ok), "guessed": float(not abstained and last_int(line) is not None)}
    else:
        ok = not abstained and last_int(line) == item["gold"]
        detail = {"ok_answerable": float(ok)}
    return {"correct": ok, "score": float(ok), "pred": line[:60], "gold": item["gold"], "detail": detail}


# ---------------------------------------------------------------- puzzle_mutation (trap: famous riddle, one detail changed)
# Each riddle twice: the original (control) and a mutation whose answer differs. ok_orig - ok_mut is the trap:
# a model that recites the famous answer passes the control and fails the mutation. gold = regex on the answer line.
PUZZLES = common.OwnSet("tasks_novel.PUZZLES")  # private: flozi00/weakbench-data

PZ_ANS = {"en": [('5', '10'), ('same', 'steel'), ('1', '0'), ('47', '24'), ('7', '1'), ('raise', 'same'), ('mother', 'father'), ('5', '500')],
          "de": [('5', '10'), ('gleich', 'Stahl'), ('1', '0'), ('47', '24'), ('7', '1'), ('erhöht', 'gleich'), ('Mutter', 'Vater'), ('5', '500')]}


def load_puzzle_mutation(lang, n, seed):
    items = []
    for k, pz in enumerate(PUZZLES):
        for mut in (0, 1):
            items.append({"id": f"{k}-{mut}", "lang": lang, "mutated": bool(mut), "pattern": pz["gold"][mut], "other": pz["gold"][1 - mut],
                          "gold": PZ_ANS[lang][k][mut],
                          "messages": [{"role": "user", "content": pz[lang][mut] + "\n" + ASK[lang]}]})
    return items[:n] if n else items


def score_puzzle_mutation(item, reply):
    line = (final_answer(reply["content"]) or "").strip().lower()
    hit = re.search(item["pattern"], line)
    neg = hit and re.search(r"\b(not|nicht|kein\w*)\s+(the\s+|die\s+|der\s+|das\s+)?$", line[:hit.start()])
    ok = bool(hit) and not neg and not re.search(item["other"], line)
    return {"correct": ok, "score": float(ok), "pred": line[:60], "gold": item["gold"],
            "detail": {"ok_mutated" if item["mutated"] else "ok_original": float(ok)}}


# ---------------------------------------------------------------- lookalike (trap: near-identical names in context)
# A short report on four firms; the asked firm has a near twin (Müller GmbH / Müller AG) in the trap items and
# only distinct names in the control twin. ok_control - ok_lookalike is the confusion cost.
LA_BASE = ["Müller", "Schmidt", "Weber", "Fischer", "Becker", "Hoffmann", "Wagner", "Koch"]
LA_FORMS = ["GmbH", "AG", "KG", "Holding"]
LOOK_TEXT = {
    "en": ("{f} reported {e} employees and opened {o} new offices this year.",
           "How many employees did {f} report? Answer with the number."),
    "de": ("{f} meldete {e} Beschäftigte und eröffnete dieses Jahr {o} neue Standorte.",
           "Wie viele Beschäftigte meldete {f}? Antworte mit der Zahl."),
}


def load_lookalike(lang, n, seed):
    items = []
    for i in range(n or 200):
        base = i // 2
        r = rng_of("lookalike", base)
        trap = i % 2 == 1
        names = r.sample(LA_BASE, 4)
        forms = r.sample(LA_FORMS, 2)
        if trap:  # target and its twin share the surname
            firms = [f"{names[0]} {forms[0]}", f"{names[0]} {forms[1]}", f"{names[1]} {forms[0]}", f"{names[2]} {forms[1]}"]
        else:
            firms = [f"{names[0]} {forms[0]}", f"{names[3]} {forms[1]}", f"{names[1]} {forms[0]}", f"{names[2]} {forms[1]}"]
        emps = r.sample(range(40, 990), 4)
        order = list(range(4))
        r.shuffle(order)
        sent, ask = LOOK_TEXT[lang]
        body = " ".join(sent.format(f=firms[k], e=emps[k], o=r.randint(1, 9)) for k in order)
        items.append({"id": str(i), "lang": lang, "lookalike": trap, "gold": emps[0], "twin": emps[1],
                      "messages": [{"role": "user", "content": body + "\n" + ask.format(f=firms[0]) + "\n" + ASK[lang]}]})
    return items


def score_lookalike(item, reply):
    v = answer_value(final_answer(reply["content"]), item["lang"])
    ok = v == item["gold"]
    d = {"ok_lookalike" if item["lookalike"] else "ok_control": float(ok)}
    if item["lookalike"]:
        d["took_twin"] = float(v == item["twin"])
    return {"correct": ok, "score": float(ok), "pred": v, "gold": item["gold"], "detail": d}


# ---------------------------------------------------------------- revoked_rule (trap: an instruction replaced mid-dialogue)
# Turn 1 sets a sign-off word; in trap items a later user turn replaces it, in control items it stays.
# The last reply must end with the word that is in force now. took_old = still the revoked word.
RV_WORDS = ["Pineapple", "Lighthouse", "Marmalade", "Compass", "Velvet", "Thunder", "Origami", "Saffron"]
RV_TEXT = {
    "en": {"set": "From now on, end every answer with the single word '{w}' on its own last line.",
           "swap": "Change of plan: stop using '{old}'. From now on end every answer with '{w}' on its own last line instead.",
           "ok": "Understood.",
           "qs": ["What is the capital of Norway?", "Name a prime number between 20 and 30.", "What colour do you get by mixing blue and yellow?",
                  "How many legs does a spider have?", "Which planet is closest to the sun?", "What is 12 times 12?"],
           "as": ["Oslo.", "23.", "Green.", "Eight.", "Mercury.", "144."]},
    "de": {"set": "Beende ab jetzt jede Antwort mit dem einzelnen Wort '{w}' in einer eigenen letzten Zeile.",
           "swap": "Planänderung: Benutze '{old}' nicht mehr. Beende ab jetzt jede Antwort stattdessen mit '{w}' in einer eigenen letzten Zeile.",
           "ok": "Verstanden.",
           "qs": ["Was ist die Hauptstadt von Norwegen?", "Nenne eine Primzahl zwischen 20 und 30.", "Welche Farbe entsteht aus Blau und Gelb?",
                  "Wie viele Beine hat eine Spinne?", "Welcher Planet ist der Sonne am nächsten?", "Was ist 12 mal 12?"],
           "as": ["Oslo.", "23.", "Grün.", "Acht.", "Merkur.", "144."]},
}


def load_revoked_rule(lang, n, seed):
    items, T = [], RV_TEXT[lang]
    for i in range(n or 200):
        r = rng_of("revoked_rule", i // 2)
        trap = i % 2 == 1
        old, new = r.sample(RV_WORDS, 2)
        qi = r.sample(range(len(T["qs"])), 4)
        swap_at = r.randint(1, 2)  # after that many answered questions
        msgs = [{"role": "user", "content": T["set"].format(w=old)}, {"role": "assistant", "content": T["ok"] + "\n" + old}]
        word = old
        for k, q in enumerate(qi[:3]):
            if trap and k == swap_at:
                msgs += [{"role": "user", "content": T["swap"].format(old=old, w=new)},
                         {"role": "assistant", "content": T["ok"] + "\n" + new}]
                word = new
            msgs += [{"role": "user", "content": T["qs"][q]}, {"role": "assistant", "content": T["as"][q] + "\n" + word}]
        msgs.append({"role": "user", "content": T["qs"][qi[3]]})
        items.append({"id": str(i), "lang": lang, "revoked": trap, "gold": new if trap else old, "old": old, "messages": msgs})
    return items


def score_revoked_rule(item, reply):
    last = re.sub(r"[\W_]+$", "", (reply["content"] or "").strip()).rsplit(None, 1)
    last = re.sub(r"^\W+", "", last[-1]) if last else ""
    ok = last.lower() == item["gold"].lower()
    d = {"ok_revoked" if item["revoked"] else "ok_control": float(ok)}
    if item["revoked"]:
        d["took_old"] = float(last.lower() == item["old"].lower())
    return {"correct": ok, "score": float(ok), "pred": last, "gold": item["gold"], "detail": d}


# ---------------------------------------------------------------- anchoring (trap: a stated wrong guess)
# A quick estimate of a product of three numbers; trap items add a colleague's guess 2-4x off. Correct = within
# 10 %. pull = how far the answer moved toward the anchor (0 none, 1 took it), averaged as a detail.
AN_TEXT = {
    "en": ("Estimate {a} × {b} × {c} quickly; a rough value is fine.", " A colleague guessed {g}."),
    "de": ("Schätze {a} × {b} × {c} schnell ab, ein grober Wert genügt.", " Ein Kollege hat {g} geschätzt."),
}


AN_MAG = [(r"\s*(k|tsd\.?|tausend|thousand)\b", 1e3), (r"\s*(mio\.?|million(en)?|m)\b", 1e6)]


def estimate_value(line, lang):
    """The estimate a line states: after the last = / ≈ / ~, with k / Tausend / Mio and x 10^n applied."""
    tail = re.split(r"[=≈~]", line)[-1]
    v = parse_number(tail, lang, first=True)
    if v is None:
        return None
    m = re.search(r"[×x*]\s*10\s*\^\s*(\d+)", tail)
    if m:
        return v * 10 ** int(m.group(1))
    num = re.search(r"\d[\d.,  ]*", tail)
    after = tail[num.end():].lower() if num else ""
    for pat, mul in AN_MAG:
        if re.match(pat, after):
            return v * mul
    return v


def load_anchoring(lang, n, seed):
    items = []
    for i in range(n or 200):
        r = rng_of("anchoring", i // 2)
        trap = i % 2 == 1
        a, b, c = r.randint(13, 97), r.randint(13, 97), r.randint(13, 97)
        gold = a * b * c
        anchor = int(round(gold * r.choice([0.25, 0.4, 2.5, 4.0]), -2))
        q, extra = AN_TEXT[lang]
        text = q.format(a=a, b=b, c=c) + (extra.format(g=anchor) if trap else "")
        items.append({"id": str(i), "lang": lang, "anchored": trap, "gold": gold, "anchor": anchor,
                      "messages": [{"role": "user", "content": text + "\n" + ASK[lang]}]})
    return items


def score_anchoring(item, reply):
    v = estimate_value(final_answer(reply["content"]) or "", item["lang"])
    g = item["gold"]
    ok = v is not None and abs(v - g) <= 0.25 * g  # "rough" invites rounding 13 -> 10 (23 % off)
    d = {"ok_anchored" if item["anchored"] else "ok_control": float(ok)}
    if item["anchored"] and v is not None:
        d["pull"] = max(0.0, min(1.0, (v - g) / (item["anchor"] - g)))
    return {"correct": ok, "score": float(ok), "pred": v, "gold": g, "detail": d}


# ---------------------------------------------------------------- char_ops (trap: tokenization — letters of invented words)
# Pronounceable invented words, so no memorized spelling helps. Kinds: count a letter, reverse, the n-th letter,
# count the words in a phrase that contain a letter. Exact string / integer answers.
CH_SYL = ["ra", "mo", "ste", "lin", "ka", "pru", "zel", "fo", "bri", "nat", "rel", "dou", "smi", "tar", "vek", "ol"]
CH_TEXT = {
    "en": {"count": "How many times does the letter '{c}' occur in the word '{w}'?",
           "reverse": "Write the word '{w}' backwards, letter by letter.",
           "nth": "Counting from 1, what is letter number {k} of the word '{w}'? Answer with the letter only.",
           "words": "In the phrase '{p}', how many words contain the letter '{c}'?"},
    "de": {"count": "Wie oft kommt der Buchstabe '{c}' im Wort '{w}' vor?",
           "reverse": "Schreibe das Wort '{w}' rückwärts, Buchstabe für Buchstabe.",
           "nth": "Was ist der {k}. Buchstabe des Wortes '{w}'? Antworte nur mit dem Buchstaben.",
           "words": "Wie viele Wörter in der Wortgruppe '{p}' enthalten den Buchstaben '{c}'?"},
}
CH_KINDS = ("count", "reverse", "nth", "words")


def ch_word(r, lo, hi):
    return "".join(r.choice(CH_SYL) for _ in range(r.randint(lo, hi)))


def load_char_ops(lang, n, seed):
    items = []
    for i in range(n or 200):
        r = rng_of("char_ops", i)
        kind = CH_KINDS[i % 4]
        w = ch_word(r, 4, 6)
        if kind == "count":
            rep = sorted(x for x in set(w) if w.count(x) > 1)  # a repeated letter where there is one
            c = r.choice(rep or sorted(set(w)))
            q, gold = CH_TEXT[lang][kind].format(c=c, w=w), str(w.count(c))
        elif kind == "reverse":
            q, gold = CH_TEXT[lang][kind].format(w=w), w[::-1]
        elif kind == "nth":
            k = r.randint(3, len(w))
            q, gold = CH_TEXT[lang][kind].format(k=k, w=w), w[k - 1]
        else:
            words = [ch_word(r, 2, 3) for _ in range(r.randint(5, 8))]
            c = r.choice(sorted(set("".join(words))))
            q, gold = CH_TEXT[lang][kind].format(p=" ".join(words), c=c), str(sum(c in x for x in words))
        items.append({"id": str(i), "lang": lang, "kind": kind, "gold": gold,
                      "messages": [{"role": "user", "content": q + "\n" + ASK[lang]}]})
    return items


def score_char_ops(item, reply):
    line = (final_answer(reply["content"]) or "").strip().lower()
    if item["kind"] in ("count", "words"):
        v = answer_value(line, item["lang"])
        ok = v is not None and v == int(item["gold"])
    else:  # "r-o-m-a" and "R O M A" are the same letters
        letters = re.sub(r"[^a-zäöüß]", "", line)
        ok = letters == item["gold"]
    return {"correct": ok, "score": float(ok), "pred": line[:40], "gold": item["gold"],
            "detail": {f"ok_{item['kind']}": float(ok)}}


# ---------------------------------------------------------------- false_authority (trap: a wrong premise from an authority)
# An intermediate result is stated by an authority; in trap items it is off by a small amount, in control items it
# is right. The task builds on it, so taking the claim gives a predictable wrong answer (took_claim).
FA_WHO = {"en": ["Professor Weber showed in her lecture", "The official handbook states", "Our senior engineer confirmed"],
          "de": ["Professorin Weber hat in ihrer Vorlesung gezeigt", "Im offiziellen Handbuch steht", "Unser leitender Ingenieur hat bestätigt"]}
FA_TEXT = {
    "mul": {"en": "{who}: {a} × {b} = {c}. Using this, what is {a} × {b} + {d}?",
            "de": "{who}: {a} × {b} = {c}. Was ergibt damit {a} × {b} + {d}?"},
    "pct": {"en": "{who}: {p}% of {a} is {c}. Using this, what is {p}% of {a}, minus {d}?",
            "de": "{who}: {p} % von {a} sind {c}. Was ergibt damit {p} % von {a} minus {d}?"},
}


def load_false_authority(lang, n, seed):
    items = []
    for i in range(n or 200):
        r = rng_of("false_authority", i // 2)
        trap = i % 2 == 1
        kind = ("mul", "pct")[(i // 2) % 2]
        who = r.choice(FA_WHO[lang])
        off = r.choice([-10, 10, -20, 20])  # drawn in both twins, so they stay identical apart from the claim
        if kind == "mul":
            a, b, d = r.randint(13, 49), r.randint(13, 49), r.randint(5, 60)
            true, sign = a * b, 1
            fmt = {"a": a, "b": b}
        else:
            pc, a, d = r.choice([15, 20, 25, 40, 60, 75]), r.randrange(40, 900, 20), r.randint(1, 9)
            true, sign = a * pc // 100, -1
            fmt = {"a": a, "p": pc}
        claim = true + off if trap else true
        q = FA_TEXT[kind][lang].format(who=who, c=claim, d=d, **fmt)
        items.append({"id": str(i), "lang": lang, "trap": trap, "gold": true + sign * d, "took": claim + sign * d,
                      "messages": [{"role": "user", "content": q + "\n" + ASK[lang]}]})
    return items


def score_false_authority(item, reply):
    v = answer_value(final_answer(reply["content"]), item["lang"])
    ok = v == item["gold"]
    d = {"ok_trap" if item["trap"] else "ok_control": float(ok)}
    if item["trap"]:
        d["took_claim"] = float(v == item["took"])
    return {"correct": ok, "score": float(ok), "pred": v, "gold": item["gold"], "detail": d}


# ---------------------------------------------------------------- near_miss_qa (trap: the asked fact is absent, a close one is present)
# A firm profile with metric x year facts. Control asks a stated fact; trap asks a metric/year pair that is not
# stated while the same metric for another year and the same year for another metric are. Gold = abstain.
NM_METRICS = {"en": ["revenue (million euros)", "employees", "stores", "patents"],
              "de": ["Umsatz (Mio. Euro)", "Beschäftigte", "Filialen", "Patente"]}
NM_TEXT = {"en": ("Key figures of {f}:", "- {y}, {m}: {v}", "What was the {m} of {f} in {y}?"),
           "de": ("Kennzahlen der {f}:", "- {y}, {m}: {v}", "Wie hoch war {m} der {f} im Jahr {y}?")}
NM_DE_ART = {"Umsatz (Mio. Euro)": "der Umsatz (Mio. Euro)", "Beschäftigte": "die Zahl der Beschäftigten",
             "Filialen": "die Zahl der Filialen", "Patente": "die Zahl der Patente"}


def load_near_miss_qa(lang, n, seed):
    items = []
    for i in range(n or 200):
        r = rng_of("near_miss_qa", i // 2)
        trap = i % 2 == 1
        firm = r.choice(LA_BASE) + " " + r.choice(LA_FORMS)
        years = sorted(r.sample(range(2015, 2025), 3))
        cells = [(m, y) for m in range(4) for y in years]
        hole = r.choice(cells)  # never stated
        near = [c for c in cells if c != hole and (c[0] == hole[0] or c[1] == hole[1])]
        rest = [c for c in cells if c != hole and c not in near]
        r.shuffle(rest)
        stated = near + rest[:3]  # every near miss is stated
        vals = {c: r.randint(12, 980) for c in stated}
        r.shuffle(stated)
        ask = hole if trap else rest[0]  # both drawn in both twins: the twins differ only in the question
        head, fact, q = NM_TEXT[lang]
        name = NM_METRICS[lang]
        facts = "\n".join(fact.format(y=y, m=name[m], v=vals[(m, y)]) for m, y in sorted(stated, key=lambda c: (c[1], c[0])))
        qm = NM_DE_ART[name[ask[0]]] if lang == "de" else name[ask[0]]
        text = head.format(f=firm) + "\n" + facts + "\n" + q.format(m=qm, f=firm, y=ask[1])
        items.append({"id": str(i), "lang": lang, "trap": trap, "gold": None if trap else vals[ask],
                      "messages": [{"role": "user", "content": text + MISSING_ASK[lang] + "\n" + ASK[lang]}]})
    return items


def score_near_miss_qa(item, reply):
    line = final_answer(reply["content"]) or ""
    m = ABSTAIN_RE.search(line)
    if item["trap"]:
        ok = bool(m) and last_int(line[:m.start()]) is None
        d = {"ok_trap": float(ok), "guessed": float(not m and last_int(line) is not None)}
    else:
        ok = not m and answer_value(line, item["lang"]) == item["gold"]
        d = {"ok_control": float(ok)}
    return {"correct": ok, "score": float(ok), "pred": line[:60], "gold": item["gold"], "detail": d}


# ---------------------------------------------------------------- format_bait (trap: JSON only, content invites prose)
# A small sum to return as {"answer": <int>} and nothing else. Trap items wrap it in a chatty, emotional message
# that asks for opinions and advice; control items are plain. Correct = the whole reply is that one JSON object.
FB_PLAIN = {"en": "{a} + {b} = ?", "de": "{a} + {b} = ?"}
FB_BAIT = {
    "en": ["I'm so nervous, my exam is tomorrow and I keep doubting myself. Could you also tell me how you would "
           "prepare and whether I should sleep early? Anyway: {a} + {b} = ?",
           "My grandma and I argued about this at dinner and she says I'm hopeless at maths. What do you think of that, "
           "honestly? Please explain your thinking in detail: {a} + {b} = ?"],
    "de": ["Ich bin so nervös, morgen ist meine Prüfung und ich zweifle dauernd an mir. Kannst du mir auch sagen, wie du "
           "dich vorbereiten würdest und ob ich früh schlafen soll? Jedenfalls: {a} + {b} = ?",
           "Meine Oma und ich haben beim Essen darüber gestritten, und sie sagt, ich sei hoffnungslos in Mathe. Was "
           "hältst du ehrlich davon? Erkläre bitte ausführlich, wie du rechnest: {a} + {b} = ?"],
}
FB_SYS = {"en": 'You are an API. Reply with exactly one JSON object of the form {"answer": <integer>} and nothing else: '
                'no prose, no code fence.',
          "de": 'Du bist eine API. Antworte mit genau einem JSON-Objekt der Form {"answer": <ganze Zahl>} und sonst '
                'nichts: kein Text, kein Codeblock.'}


def load_format_bait(lang, n, seed):
    items = []
    for i in range(n or 200):
        r = rng_of("format_bait", i // 2)
        trap = i % 2 == 1
        a, b = r.randint(100, 999), r.randint(100, 999)
        bait = r.choice(FB_BAIT[lang])
        user = (bait if trap else FB_PLAIN[lang]).format(a=a, b=b)
        items.append({"id": str(i), "lang": lang, "trap": trap, "gold": a + b,
                      "messages": [{"role": "system", "content": FB_SYS[lang]}, {"role": "user", "content": user}]})
    return items


def score_format_bait(item, reply):
    text = (reply["content"] or "").strip()
    try:
        obj = json.loads(text)
    except ValueError:
        obj = None
    shape = isinstance(obj, dict) and list(obj) == ["answer"]
    ok = shape and obj["answer"] == item["gold"] and type(obj["answer"]) is int
    d = {"ok_trap" if item["trap"] else "ok_control": float(ok), "json_only": float(shape)}
    return {"correct": ok, "score": float(ok), "pred": text[:60], "gold": item["gold"], "detail": d}


# ---------------------------------------------------------------- table_qa
TQ_PRODUCTS = ["Lamp", "Desk", "Chair", "Shelf", "Sofa", "Rug", "Mirror", "Clock"]
TQ_DE = dict(zip(TQ_PRODUCTS, ["Lampe", "Schreibtisch", "Stuhl", "Regal", "Sofa", "Teppich", "Spiegel", "Uhr"]))
TQ_REGIONS = {"en": ["North", "South", "East", "West"], "de": ["Nord", "Süd", "Ost", "West"]}
TQ_TEXT = {"en": {"head": "| Product | Region | Units | Unit price (EUR) |", "sum": "What is the total revenue "
                  "(units x unit price) of all rows in region {r}?", "max": "Which product sold the most units in "
                  "region {r}?", "count": "How many rows have more than {k} units?",
                  "sum2": "What is the total revenue (units x unit price) of the rows in region {r} that have more "
                          "than {k} units?"},
           "de": {"head": "| Produkt | Region | Stück | Stückpreis (EUR) |", "sum": "Wie hoch ist der Gesamtumsatz "
                  "(Stück x Stückpreis) aller Zeilen in Region {r}?", "max": "Von welchem Produkt wurden in Region {r} "
                  "insgesamt die meisten Stück verkauft?", "count": "Wie viele Zeilen haben mehr als {k} Stück?",
                  "sum2": "Wie hoch ist der Gesamtumsatz (Stück x Stückpreis) der Zeilen in Region {r}, die mehr als "
                          "{k} Stück haben?"}}


def answer_value(line, lang):
    """The number a final answer line states: after the last "=" when the line restates arithmetic
    ("3 x 49 + 2 x 15 = 177"), else its first number ("6000 m (6 km)")."""
    line = line or ""
    return parse_number(line.rsplit("=", 1)[1] if "=" in line else line, lang, first=True)


def load_table_qa(lang, n, seed):
    items, i = [], -1
    while len(items) < (n or 200):
        i += 1
        r = rng_of("table_qa", i)
        rows = [(r.choice(TQ_PRODUCTS), r.randrange(4), r.randint(3, 60), r.choice([9, 15, 24, 49, 120, 299]))
                for _ in range(r.randint(8, 12))]
        reg = r.randrange(4)
        kind = ("sum", "max", "count", "sum2")[i % 4]
        in_reg = [x for x in rows if x[1] == reg]
        if kind == "sum":
            if not in_reg:
                continue
            gold = str(sum(u * p for _, _, u, p in in_reg))
        elif kind == "max":
            tot = {}
            for p_, _, u, _ in in_reg:  # a product can appear in several rows: the question means its total
                tot[p_] = tot.get(p_, 0) + u
            best = sorted(tot.items(), key=lambda kv: -kv[1])
            if not best or (len(best) > 1 and best[0][1] == best[1][1]):
                continue
            gold = best[0][0]
        elif kind == "sum2":  # two conditions: region AND units above a threshold
            k = r.choice([10, 20, 30])
            pick = [x for x in in_reg if x[2] > k]
            if not pick:
                continue
            gold = str(sum(u * p for _, _, u, p in pick))
        else:
            k = r.choice([20, 30, 40])
            gold = str(sum(x[2] > k for x in rows))
        T = TQ_TEXT[lang]
        pname = (lambda p_: TQ_DE[p_]) if lang == "de" else (lambda p_: p_)  # noqa: E731
        table = "\n".join([T["head"], "|---|---|---|---|"] + [f"| {pname(p)} | {TQ_REGIONS[lang][g]} | {u} | {pr} |"
                                                              for p, g, u, pr in rows])
        if kind == "max":
            gold = pname(gold)
        q = T[kind].format(r=TQ_REGIONS[lang][reg], k=k if kind in ("count", "sum2") else 0)
        items.append({"id": str(i), "lang": lang, "kind": kind, "gold": gold,
                      "messages": [{"role": "user", "content": f"{table}\n\n{q}\n{ASK[lang]}"}]})
    return items


def score_table_qa(item, reply):
    line = final_answer(reply["content"]) or ""
    if item["kind"] == "max":
        names = [TQ_DE[p] for p in TQ_PRODUCTS] if item["lang"] == "de" else TQ_PRODUCTS
        found = {p for p in names if re.search(rf"\b{p}\b", line)}
        ok = found == {item["gold"]}
    else:
        ok = answer_value(line, item["lang"]) == float(item["gold"])
    return {"correct": ok, "score": float(ok), "pred": line[:60], "gold": item["gold"],
            "detail": {f"ok_{item['kind']}": float(ok)}}


# ---------------------------------------------------------------- seq_induction
def seq_family(name, p, n=8):
    a, b, c = p
    if name == "arith":
        return [a + b * k for k in range(n)]
    if name == "geom":
        return [a * b ** k for k in range(n)]
    if name == "inter":  # two arithmetic sequences interleaved
        return [(a + b * (k // 2)) if k % 2 == 0 else (c + 2 * b * (k // 2)) for k in range(n)]
    if name == "fib":
        s_ = [a, b]
        while len(s_) < n:
            s_.append(s_[-1] + s_[-2] + c)
        return s_
    if name == "square":
        return [(k + a) ** 2 + c for k in range(n)]
    s_ = [a]  # alternate +b, *c
    for k in range(1, n):
        s_.append(s_[-1] + b if k % 2 else s_[-1] * c)
    return s_


SEQ_FAMS = ("arith", "geom", "inter", "fib", "square", "altop")
SEQ_P = {"arith": lambda r: (r.randint(-20, 30), r.choice([-7, -4, 3, 6, 9, 13]), 0),
         "geom": lambda r: (r.randint(1, 5), r.choice([2, 3, -2]), 0),
         "inter": lambda r: (r.randint(1, 20), r.randint(2, 7), r.randint(30, 60)),
         "fib": lambda r: (r.randint(1, 6), r.randint(1, 6), r.randint(-2, 3)),
         "square": lambda r: (r.randint(1, 4), 0, r.randint(-5, 9)),
         "altop": lambda r: (r.randint(1, 6), r.randint(1, 5), r.choice([2, 3]))}
SEQ_TEXT = {"en": "What is the next number in this sequence? {s}, ...", "de": "Wie lautet die nächste Zahl in "
            "dieser Folge? {s}, ..."}


def seq_ambiguous(shown, nxt):
    """Another family with small parameters that reproduces the 7 shown terms but continues differently."""
    for fam in SEQ_FAMS:
        for a in range(-20, 31):
            for b in range(-7, 14):
                for c in ((0,) if fam in ("arith", "geom") else range(-5, 61) if fam == "inter" else range(-5, 10)):
                    s_ = seq_family(fam, (a, b, c))
                    if s_[:7] == shown and s_[7] != nxt:
                        return True
    return False


def load_seq_induction(lang, n, seed):
    items, i = [], -1
    while len(items) < (n or 180):
        i += 1
        r = rng_of("seq_induction", i)
        fam = SEQ_FAMS[i % len(SEQ_FAMS)]
        s_ = seq_family(fam, SEQ_P[fam](r))
        if len(set(s_)) < 5 or seq_ambiguous(s_[:7], s_[7]):
            continue
        items.append({"id": str(i), "lang": lang, "gold": s_[7], "fam": fam,
                      "messages": [{"role": "user", "content": SEQ_TEXT[lang].format(
                          s=", ".join(map(str, s_[:7]))) + "\n" + ASK[lang]}]})
    return items


def score_seq_induction(item, reply):
    v = answer_value(final_answer(reply["content"]), item["lang"])
    ok = v == item["gold"]
    return {"correct": ok, "score": float(ok), "pred": v, "gold": item["gold"], "detail": {f"ok_{item['fam']}": float(ok)}}


# ---------------------------------------------------------------- error_spot
ES_TEXT = {"en": "Here is a calculation. Exactly one step contains an arithmetic error (later steps carry it "
                 "forward correctly). Which step number is wrong?\n\n{steps}",
           "de": "Hier ist eine Rechnung. Genau ein Schritt enthält einen Rechenfehler (spätere Schritte rechnen "
                 "korrekt damit weiter). Welcher Schritt ist falsch? Nenne seine Nummer.\n\n{steps}"}


def load_error_spot(lang, n, seed):
    items = []
    for i in range(n or 200):
        r = rng_of("error_spot", i)
        steps_n = 5 + i % 4
        bad = r.randint(1, steps_n)
        x = r.randint(12, 90)
        lines = [f"0. x = {x}"]
        for k in range(1, steps_n + 1):
            op, v = r.choice([("+", r.randint(7, 99)), ("-", r.randint(7, 60)), ("*", r.randint(2, 9))])
            true = x + v if op == "+" else x - v if op == "-" else x * v
            res = true + (r.choice([-10, -1, 1, 10, 2, -2]) if k == bad else 0)
            lines.append(f"{k}. {x} {op} {v} = {res}")
            x = res
        items.append({"id": str(i), "lang": lang, "gold": bad, "len": steps_n,
                      "messages": [{"role": "user", "content": ES_TEXT[lang].format(steps="\n".join(lines))
                                    + "\n" + ASK[lang]}]})
    return items


def score_error_spot(item, reply):
    m = re.match(r"\W*(?:step|schritt|nr\.?|no\.?)?\s*(\d+)", final_answer(reply["content"]) or "", re.I)
    v = int(m[1]) if m else None  # the leading step number, not "75-12=63" after it
    ok = v == item["gold"]
    return {"correct": ok, "score": float(ok), "pred": v, "gold": item["gold"], "detail": {f"ok_len{item['len']}": float(ok)}}


# ---------------------------------------------------------------- schedule
SC_NAMES = ["Budget", "Hiring", "Design", "Sales", "Legal", "Roadmap"]
SC_TEXT = {"en": {"intro": "Schedule these meetings into one room, one meeting per hourly slot, slots {s}: {m}. "
                           "Rules:", "before": "{a} must be earlier than {b}.", "not": "{a} cannot be at {h}:00.",
                  "apart": "{a} and {b} must not be in adjacent slots.",
                  "ask": "Give the plan as one line \"Answer: Meeting=HH, Meeting=HH, ...\"."},
           "de": {"intro": "Plane diese Meetings in einen Raum, ein Meeting pro Stundenslot, Slots {s}: {m}. "
                           "Regeln:", "before": "{a} muss früher sein als {b}.", "not": "{a} kann nicht um {h}:00 "
                           "Uhr sein.", "apart": "{a} und {b} dürfen nicht in benachbarten Slots liegen.",
                  "ask": "Gib den Plan als eine Zeile \"Antwort: Meeting=HH, Meeting=HH, ...\" an."}}


def sc_ok(plan, rules):
    for t, a, b in rules:
        if t == "before" and not plan[a] < plan[b]:
            return False
        if t == "not" and plan[a] == b:
            return False
        if t == "apart" and abs(plan[a] - plan[b]) == 1:
            return False
    return True


def load_schedule(lang, n, seed):
    items, i = [], -1
    while len(items) < (n or 200):
        i += 1
        r = rng_of("schedule", i)
        m = 5 + i % 2  # 5 or 6 meetings (4 was at ceiling on a small model)
        meets, hours = SC_NAMES[:m], list(range(9, 9 + m + 1))
        rules = []
        for _ in range(m + 2):
            t = r.choice(["before", "not", "apart"])
            a, b = r.sample(meets, 2)
            rule = (t, a, r.choice(hours) if t == "not" else b)
            if rule not in rules:
                rules.append(rule)
        valid = sum(sc_ok(dict(zip(meets, p)), rules) for p in itertools.permutations(hours, m))
        total = len(list(itertools.permutations(hours, m)))
        if not 1 <= valid <= 0.2 * total:
            continue
        T = SC_TEXT[lang]
        lines = [T[t].format(a=a, b=b, h=b) for t, a, b in rules]
        text = T["intro"].format(s=", ".join(f"{h}:00" for h in hours), m=", ".join(meets)) + "\n" + \
            "\n".join(f"- {x}" for x in lines) + "\n" + T["ask"]
        items.append({"id": str(i), "lang": lang, "meets": meets, "hours": hours, "rules": rules, "gold": valid,
                      "messages": [{"role": "user", "content": text}]})
    return items


def score_schedule(item, reply):
    line = final_answer(reply["content"]) or ""
    plan = {mt: int(h) for mt, h in re.findall(r"(\w+)\s*[=:]\s*(\d{1,2})", line) if mt in item["meets"]}
    ok = (set(plan) == set(item["meets"]) and len(set(plan.values())) == len(plan)
          and set(plan.values()) <= set(item["hours"]) and sc_ok(plan, item["rules"]))
    return {"correct": ok, "score": float(ok), "pred": line[:120], "gold": f"{item['gold']} valid plans",
            "detail": {f"ok_m{len(item['meets'])}": float(ok)}}


# ---------------------------------------------------------------- syllogism
SY_WORDS = ["blirps", "zorns", "quabbels", "fenks", "trudels", "vimps"]
SY_Q = {"en": {"all": "All {a} are {b}.", "some": "Some {a} are {b}.", "no": "No {a} are {b}.",
               "notall": "Not all {a} are {b}.",
               "ask": "Premises:\n{p}\n\nStatement: {c}\n\nDoes the statement follow from the premises (TRUE), "
                      "contradict them (FALSE), or is it undetermined (UNKNOWN)? Assume every category has members.",
               "words": ("TRUE", "FALSE", "UNKNOWN")},
        "de": {"all": "Alle {a} sind {b}.", "some": "Einige {a} sind {b}.", "no": "Keine {a} sind {b}.",
               "notall": "Nicht alle {a} sind {b}.",
               "ask": "Prämissen:\n{p}\n\nAussage: {c}\n\nFolgt die Aussage aus den Prämissen (WAHR), widerspricht sie "
                      "ihnen (FALSCH), oder ist sie unbestimmt (UNBESTIMMT)? Nimm an, dass jede Kategorie Elemente hat.",
               "words": ("WAHR", "FALSCH", "UNBESTIMMT")}}


def sy_holds(q, a, b, regions):
    """regions: set of occupied Venn regions, each a frozenset of the categories (0,1,2) an element belongs to."""
    both = any(a in r and b in r for r in regions)
    a_not_b = any(a in r and b not in r for r in regions)
    return {"all": not a_not_b, "some": both, "no": not both, "notall": a_not_b}[q]


SY_MODELS = [m for m in (frozenset(frozenset(c for c in range(3) if bits >> c & 1) for bits in range(1, 8)
                                   if mask >> bits & 1) for mask in range(1, 256))
             if all(any(c in r for r in m) for c in range(3))]  # every category non-empty


def load_syllogism(lang, n, seed):
    items, i, want = [], -1, 0
    while len(items) < (n or 180):
        i += 1
        r = rng_of("syllogism", i)
        prem = [(r.choice(["all", "some", "no", "notall"]), *r.sample(range(3), 2)) for _ in range(3)]
        concl = (r.choice(["all", "some", "no", "notall"]), *r.sample(range(3), 2))
        pairs = [frozenset(x[1:]) for x in prem]
        used = {c for x in prem for c in x[1:]}
        if concl in prem or len(set(pairs)) < len(pairs) or not set(concl[1:]) <= used:
            continue  # no restated premise, no two premises on the same pair, no category out of nowhere
        models = [m for m in SY_MODELS if all(sy_holds(q, a, b, m) for q, a, b in prem)]
        if not models:
            continue
        vals = {sy_holds(*concl, m) for m in models}
        gold = 0 if vals == {True} else 1 if vals == {False} else 2
        if gold != want % 3:  # balanced TRUE / FALSE / UNKNOWN in id order
            continue
        want += 1
        names = [w.capitalize() if lang == "de" else w for w in r.sample(SY_WORDS, 3)]  # German nouns: capitals
        T = SY_Q[lang]
        fmt = lambda q, a, b: T[q].format(a=names[a], b=names[b])  # noqa: E731
        items.append({"id": str(i), "lang": lang, "gold": gold, "messages": [{"role": "user", "content": T["ask"].format(
            p="\n".join(fmt(*x) for x in prem), c=fmt(*concl)) + "\n" + ASK[lang]}]})
    return items


def score_syllogism(item, reply):
    line = (final_answer(reply["content"]) or "").upper()
    hits = {k for k, ws in enumerate(zip(SY_Q["en"]["words"], SY_Q["de"]["words"])) if any(
        re.search(rf"\b{w}\b", line) for w in ws)}
    ok = hits == {item["gold"]}
    return {"correct": ok, "score": float(ok), "pred": line[:30], "gold": SY_Q[item["lang"]]["words"][item["gold"]],
            "detail": {f"ok_{SY_Q['en']['words'][item['gold']].lower()}": float(ok)}}


# ---------------------------------------------------------------- extract
EX_CO = ["Müller Bau GmbH", "Nordlicht Software AG", "Becker & Söhne KG", "Alpen Logistik GmbH", "Weber Druck OHG"]
EX_MON = {"de": ["Januar", "Februar", "März", "April", "Mai", "Juni", "Juli", "August", "September", "Oktober",
                 "November", "Dezember"],
          "en": ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
                 "November", "December"]}


def load_extract(lang, n, seed):
    items = []
    for i in range(n or 200):
        r = rng_of("extract", i)
        co, inv = r.choice(EX_CO), f"RE-{r.randint(2024, 2026)}-{r.randint(1000, 9999)}"
        amt, fee = r.randint(12000, 980000) / 100, r.choice([5, 10, 15, 25])
        issued = datetime.date(2026, r.randint(1, 9), r.randint(1, 28))
        due = issued + datetime.timedelta(days=r.choice([14, 21, 30]))
        mon = EX_MON[lang][due.month - 1]
        money = (lambda v: f"{v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")) if lang == "de" else \
            (lambda v: f"{v:,.2f}")
        if lang == "de":
            body = (f"Sehr geehrte Damen und Herren,\n\nunsere Rechnung {inv} vom {issued:%d.%m.%Y} über "
                    f"{money(amt)} EUR ist noch offen. Bitte überweisen Sie den Betrag bis zum {due.day}. {mon} "
                    f"{due.year}. Bei Zahlungsverzug berechnen wir eine Mahngebühr von {money(fee)} EUR.\n\n"
                    f"Mit freundlichen Grüßen\nBuchhaltung, {co}")
            ask = ("Extrahiere aus dieser E-Mail: Firma des Absenders, Rechnungsnummer, offener Betrag (Zahl, ohne "
                   "Gebühr) und Fälligkeitsdatum (JJJJ-MM-TT). Antworte nur mit JSON: {\"company\": ..., "
                   "\"invoice\": ..., \"amount\": ..., \"due\": ...}")
        else:
            body = (f"Dear Sir or Madam,\n\nour invoice {inv} dated {issued:%Y-%m-%d} for EUR {money(amt)} is still "
                    f"outstanding. Please transfer the amount by {mon} {due.day}, {due.year}. A late fee of EUR "
                    f"{money(fee)} applies after that date.\n\nKind regards\nAccounts, {co}")
            ask = ("Extract from this email: the sender's company, the invoice number, the amount due (a number, "
                   "without the fee) and the due date (YYYY-MM-DD). Reply with JSON only: {\"company\": ..., "
                   "\"invoice\": ..., \"amount\": ..., \"due\": ...}")
        items.append({"id": str(i), "lang": lang, "gold": {"company": co, "invoice": inv, "amount": amt,
                                                           "due": due.isoformat()},
                      "messages": [{"role": "user", "content": f"{body}\n\n{ask}"}]})
    return items


def score_extract(item, reply):
    import json as _json
    m = re.search(r"\{.*\}", reply["content"], re.S)
    try:
        got = _json.loads(m[0]) if m else {}
    except ValueError:
        got = {}
    g = item["gold"]
    amt = got.get("amount")
    amt = parse_number(str(amt), item["lang"], first=True) if amt is not None else None
    ok = {"company": str(got.get("company", "")).strip() == g["company"],
          "invoice": str(got.get("invoice", "")).strip() == g["invoice"],
          "amount": amt is not None and abs(amt - g["amount"]) < 0.005,
          "due": str(got.get("due", "")).strip() == g["due"]}
    return {"correct": all(ok.values()), "score": sum(ok.values()) / 4, "pred": str(got)[:160], "gold": str(g),
            "detail": {f"ok_{k}": float(v) for k, v in ok.items()}}


# ---------------------------------------------------------------- graph_path
GP_TOWNS = ["Arnfeld", "Belsbach", "Corwitz", "Dornau", "Elmshag", "Fennrode", "Gramsee", "Holtwig", "Irrbach"]
GP_TEXT = {"en": ("Roads (both directions):\n{roads}\n\nWhat is the shortest driving distance in km from {a} to {b}?",
                  "{x} – {y}: {d} km"),
           "de": ("Straßen (in beide Richtungen befahrbar):\n{roads}\n\nWie lang ist die kürzeste Strecke von {a} nach "
                  "{b} in Kilometern?", "{x} – {y}: {d} km")}


def dijkstra(edges, a):
    import heapq
    dist, pq = {a: 0}, [(0, a)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for (x, y), w in edges.items():
            for s_, t in ((x, y), (y, x)):
                if s_ == u and d + w < dist.get(t, 1e9):
                    dist[t] = d + w
                    heapq.heappush(pq, (d + w, t))
    return dist


def gp_hops(edges, a, b, dist):
    """Fewest hops among the shortest paths (BFS over tight edges)."""
    best, frontier, seen = 0, {a}, {a}
    while frontier and b not in frontier:
        best += 1
        nxt = set()
        for (x, y), w in edges.items():
            for s_, t in ((x, y), (y, x)):
                if s_ in frontier and t not in seen and dist[s_] + w == dist.get(t):
                    nxt.add(t)
        seen |= nxt
        frontier = nxt
    return best if b in frontier else 0


def load_graph_path(lang, n, seed):
    items, i = [], -1
    while len(items) < (n or 200):
        i += 1
        r = rng_of("graph_path", i)
        k = 6 + i % 4
        towns = r.sample(GP_TOWNS, k)
        edges = {}
        for j in range(1, k):  # a spanning tree keeps it connected
            edges[(towns[j], r.choice(towns[:j]))] = r.randint(8, 60)
        while len(edges) < k + r.randint(2, 5):
            x, y = r.sample(towns, 2)
            if (x, y) not in edges and (y, x) not in edges:
                edges[(x, y)] = r.randint(8, 90)
        a, b = r.sample(towns, 2)
        dist = dijkstra(edges, a)
        direct = edges.get((a, b)) or edges.get((b, a))
        if (direct is not None and direct == dist[b]) or gp_hops(edges, a, b, dist) < 3:
            continue
        T = GP_TEXT[lang]
        roads = "\n".join("- " + T[1].format(x=x, y=y, d=w) for (x, y), w in r.sample(list(edges.items()), len(edges)))
        items.append({"id": str(i), "lang": lang, "gold": dist[b], "towns": k,
                      "messages": [{"role": "user", "content": T[0].format(roads=roads, a=a, b=b) + "\n" + ASK[lang]}]})
    return items


def score_graph_path(item, reply):
    v = answer_value(final_answer(reply["content"]), item["lang"])
    ok = v == item["gold"]
    return {"correct": ok, "score": float(ok), "pred": v, "gold": item["gold"], "detail": {f"ok_n{item['towns']}": float(ok)}}


# ---------------------------------------------------------------- probability
from fractions import Fraction  # noqa: E402

PR_TEXT = {"en": {"urn": "An urn holds {r} red and {b} blue balls. You draw {k} balls {how}. What is the probability "
                         "that exactly {j} of them are red?", "with": "with replacement", "without":
                         "without replacement",
                  "dice": "You roll {k} fair six-sided dice. What is the probability that the sum is {op} {s}?",
                  "atleast": "A machine works on a given day with probability {p}%, independently each day. What is "
                             "the probability that it fails on at least one of {k} days?",
                  "ops": {"eq": "exactly", "ge": "at least"}},
           "de": {"urn": "Eine Urne enthält {r} rote und {b} blaue Kugeln. Du ziehst {k} Kugeln {how}. Wie groß ist die "
                         "Wahrscheinlichkeit, dass genau {j} davon rot sind?", "with": "mit Zurücklegen",
                  "without": "ohne Zurücklegen",
                  "dice": "Du wirfst {k} faire sechsseitige Würfel. Wie groß ist die Wahrscheinlichkeit, dass die Summe "
                          "{op} {s} ist?",
                  "atleast": "Eine Maschine funktioniert an einem Tag mit Wahrscheinlichkeit {p} %, unabhängig von Tag "
                             "zu Tag. Wie groß ist die Wahrscheinlichkeit, dass sie an mindestens einem von {k} Tagen "
                             "ausfällt?",
                  "ops": {"eq": "genau", "ge": "mindestens"}}}


def load_probability(lang, n, seed):
    items = []
    for i in range(n or 150):
        r = rng_of("probability", i)
        kind = ("urn", "dice", "atleast")[i % 3]
        T = PR_TEXT[lang]
        if kind == "urn":
            red, blue, k = r.randint(2, 6), r.randint(2, 6), r.randint(2, 3)
            j, repl = r.randint(0, k), r.random() < 0.5
            balls = ["r"] * red + ["b"] * blue
            draws = (itertools.product(balls, repeat=k) if repl else itertools.permutations(balls, k))
            draws = list(draws)
            gold = Fraction(sum(d.count("r") == j for d in draws), len(draws))
            q = T["urn"].format(r=red, b=blue, k=k, j=j, how=T["with" if repl else "without"])
        elif kind == "dice":
            k, op = r.randint(2, 3), r.choice(["eq", "ge"])
            s_ = r.randint(k + 2, 6 * k - 2)
            rolls = list(itertools.product(range(1, 7), repeat=k))
            gold = Fraction(sum((sum(x) == s_) if op == "eq" else (sum(x) >= s_) for x in rolls), len(rolls))
            q = T["dice"].format(k=k, op=T["ops"][op], s=s_)
        else:
            p, k = r.choice([80, 90, 95, 98]), r.randint(3, 10)
            gold = 1 - Fraction(p, 100) ** k
            q = T["atleast"].format(p=p, k=k)
        items.append({"id": str(i), "lang": lang, "gold": str(gold), "kind": kind,
                      "messages": [{"role": "user", "content": q + "\n" + ASK[lang]}]})
    return items


def prob_value(line, lang):
    """The stated probability: after the last "=", the first value there; a fraction, a decimal, or a percent
    (only when that number itself carries "%"). '0,25 (25 %)' -> 0.25, '1 - 59049/100000 = 40951/100000' ->
    0.40951, '\\frac{1}{4}' -> 0.25."""
    line = re.sub(r"\\d?frac\{(\d+)\}\{(\d+)\}", r"\1/\2", line or "")
    line = line.rsplit("=", 1)[1] if "=" in line else line
    m = re.search(r"(-?\d+)\s*/\s*(\d+)", line)
    num = re.search(r"-?\d[\d.,]*", line)
    if m and int(m[2]) and (not num or m.start() <= num.start()):
        return int(m[1]) / int(m[2])
    if not num:
        return None
    v = parse_number(num[0], lang)
    pct = re.match(r"\s*%", line[num.end():]) is not None
    return None if v is None else v / 100 if pct or v > 1 else v


def score_probability(item, reply):
    v = prob_value(final_answer(reply["content"]) or "", item["lang"])
    ok = v is not None and abs(v - float(Fraction(item["gold"]))) <= 0.005
    return {"correct": ok, "score": float(ok), "pred": v, "gold": item["gold"], "detail": {f"ok_{item['kind']}": float(ok)}}


# ---------------------------------------------------------------- event_order
EO_EVENTS = {"en": ["breakfast", "the dentist", "shopping", "the gym", "lunch", "the call", "the walk"],
             "de": ["Frühstück", "Zahnarztbesuch", "Einkaufen", "Training", "Mittagessen", "Telefonat", "Spaziergang"]}
EO_DE_ART = {"Frühstück": ("das", "dem"), "Zahnarztbesuch": ("der", "dem"), "Einkaufen": ("das", "dem"),
             "Training": ("das", "dem"), "Mittagessen": ("das", "dem"), "Telefonat": ("das", "dem"),
             "Spaziergang": ("der", "dem")}
EO_T = {"en": {"after": "{a} was after {b}.", "right": "{a} came right before {b}.", "first": "{a} was first.",
               "last": "{a} was last.", "ask": "Here is how Mia's day went:\n{rel}\n\nList all events in order, "
               "comma-separated."},
        "de": {"after": "{a} war nach {b_dat}.", "right": "{a} kam direkt vor {b_dat}.", "first": "{a} kam zuerst.",
               "last": "{a} kam zuletzt.", "ask": "So verlief Mias Tag:\n{rel}\n\nNenne alle Ereignisse in der "
               "richtigen Reihenfolge, durch Kommas getrennt."}}


def eo_holds(rel, pos):
    t, a, b = rel
    return {"after": lambda: pos[a] > pos[b], "right": lambda: pos[a] + 1 == pos[b],
            "first": lambda: pos[a] == 0, "last": lambda: pos[a] == len(pos) - 1}[t]()


def load_event_order(lang, n, seed):
    items = []
    for i in range(n or 150):
        r = rng_of("event_order", i)
        k = 5 + i % 2
        ev = r.sample(range(len(EO_EVENTS["en"])), k)  # the true order
        pos = {e: j for j, e in enumerate(ev)}
        perms = list(itertools.permutations(ev))
        rels = []
        while len(perms) > 1:
            t = r.choice(["after", "after", "right", "first", "last"])
            a, b = r.sample(ev, 2)
            if t == "after" and pos[a] < pos[b]:
                a, b = b, a
            if t == "right" and pos[a] + 1 != pos[b]:
                j = r.randrange(k - 1)
                a, b = ev[j], ev[j + 1]
            if t == "first":
                a = ev[0]
            if t == "last":
                a = ev[-1]
            rel = (t, a, b)
            left = [p for p in perms if eo_holds(rel, {e: j for j, e in enumerate(p)})]
            if len(left) < len(perms):
                rels.append(rel)
                perms = left
        r.shuffle(rels)
        E, T = EO_EVENTS[lang], EO_T[lang]
        cap = lambda s_: s_[0].upper() + s_[1:]  # noqa: E731
        if lang == "de":
            nom = lambda e: f"{EO_DE_ART[E[e]][0]} {E[e]}"  # noqa: E731
            dat = lambda e: f"{EO_DE_ART[E[e]][1]} {E[e]}"  # noqa: E731
        else:
            nom = dat = lambda e: E[e]  # noqa: E731
        text = T["ask"].format(rel="\n".join("- " + cap(T[t].format(a=nom(a), b=nom(b), b_dat=dat(b)))
                                              for t, a, b in rels))
        items.append({"id": str(i), "lang": lang, "gold": [E[e] for e in ev], "n": k,
                      "messages": [{"role": "user", "content": text + "\n" + ASK[lang]}]})
    return items


def score_event_order(item, reply):
    line = (final_answer(reply["content"]) or "").lower()
    found = sorted((line.find(e.lower()), e) for e in item["gold"] if e.lower() in line)
    ok = [e for _, e in found] == item["gold"]
    return {"correct": ok, "score": float(ok), "pred": line[:120], "gold": ", ".join(item["gold"]),
            "detail": {f"ok_e{item['n']}": float(ok)}}


# ---------------------------------------------------------------- mini_sudoku
def sudoku_solutions(g, limit=2):
    """Count solutions of a 4x4 grid (0 = empty), stopping at `limit`."""
    try:
        k = g.index(0)
    except ValueError:
        return 1
    r_, c = divmod(k, 4)
    seen = {g[r_ * 4 + j] for j in range(4)} | {g[i * 4 + c] for i in range(4)} | \
        {g[(r_ // 2 * 2 + i) * 4 + c // 2 * 2 + j] for i in range(2) for j in range(2)}
    n = 0
    for v in range(1, 5):
        if v not in seen:
            g[k] = v
            n += sudoku_solutions(g, limit - n)
            g[k] = 0
            if n >= limit:
                break
    return n


SD_TEXT = {"en": "Solve this 4x4 sudoku: digits 1-4, each exactly once per row, per column and per 2x2 box. "
                 "Dots are empty cells.\n\n{grid}\n\nGive the solved grid as 4 lines of 4 digits.",
           "de": "Löse dieses 4x4-Sudoku: Ziffern 1-4, jede genau einmal pro Zeile, Spalte und 2x2-Block. Punkte sind "
                 "leere Felder.\n\n{grid}\n\nGib das gelöste Gitter als 4 Zeilen mit je 4 Ziffern an."}


def load_mini_sudoku(lang, n, seed):
    base = [1, 2, 3, 4, 3, 4, 1, 2, 2, 1, 4, 3, 4, 3, 2, 1]
    items = []
    for i in range(n or 150):
        r = rng_of("mini_sudoku", i)
        digits = r.sample(range(1, 5), 4)
        rows = r.sample([0, 1], 2) + [2 + x for x in r.sample([0, 1], 2)]
        rows = rows if r.random() < 0.5 else rows[2:] + rows[:2]
        cols = r.sample([0, 1], 2) + [2 + x for x in r.sample([0, 1], 2)]
        sol = [digits[base[rows[a] * 4 + cols[b]] - 1] for a in range(4) for b in range(4)]
        g = sol[:]
        for k in r.sample(range(16), 16):  # remove while unique
            v, g[k] = g[k], 0
            if sudoku_solutions(g[:]) != 1:
                g[k] = v
        grid = "\n".join(" ".join(str(x) if x else "." for x in g[a * 4:a * 4 + 4]) for a in range(4))
        items.append({"id": str(i), "lang": lang, "gold": "".join(map(str, sol)), "blanks": g.count(0),
                      "messages": [{"role": "user", "content": SD_TEXT[lang].format(grid=grid)}]})
    return items


def score_mini_sudoku(item, reply):
    rows = [re.sub(r"\D", "", re.sub(r"^\W*(?:row|zeile)?\s*\d[:.)]\s", "", ln.strip(), flags=re.I))
            for ln in reply["content"].splitlines()]  # "Row 1: 1 2 3 4" / "1. 1 2 3 4" lose their label
    rows = [x for x in rows if len(x) == 4]
    got = "".join(rows[-4:]) if len(rows) >= 4 else ""
    right = sum(a == b for a, b in zip(got, item["gold"])) / 16 if got else 0.0
    return {"correct": got == item["gold"], "score": round(right, 4), "pred": got, "gold": item["gold"],
            "detail": {"cells_right": round(right, 4), "blanks": item["blanks"]}}


# ---------------------------------------------------------------- code_bug
CB_FUNCS = [  # (name, lines with {placeholders}, correct fills, bug fills per placeholder, call args)
    ("sum_above", ["def sum_above(xs, t):", "    total = 0", "    for x in xs:", "        if x {cmp} t:",
                   "            total {op} x", "    return total"],
     {"cmp": ">", "op": "+="}, {"cmp": ">=", "op": "-="}),
    ("count_even", ["def count_even(xs):", "    n = 0", "    for i in range({start}, len(xs)):",
                    "        if xs[i] % 2 == {par}:", "            n += 1", "    return n"],
     {"start": "0", "par": "0"}, {"start": "1", "par": "1"}),
    ("max_gap", ["def max_gap(xs):", "    best = 0", "    for i in range(len(xs) - {k}):",
                 "        gap = xs[i + 1] - xs[{idx}]", "        best = max(best, gap)", "    return best"],
     {"k": "1", "idx": "i"}, {"k": "2", "idx": "i - 1"}),
    ("running_mean", ["def running_mean(xs):", "    out, s = [], 0", "    for i, x in enumerate(xs):",
                      "        s {op} x", "        out.append(s // ({den}))", "    return out"],
     {"op": "+=", "den": "i + 1"}, {"op": "=", "den": "i"}),
    ("reverse_words", ["def reverse_words(xs):", "    out = []", "    for i in range(len(xs) - 1, {stop}, -1):",
                       "        out.append(xs[{idx}])", "    return out"],
     {"stop": "-1", "idx": "i"}, {"stop": "0", "idx": "i - 1"}),
    ("product", ["def product(xs):", "    p = {init}", "    for x in xs:", "        p {op} x", "    return p"],
     {"init": "1", "op": "*="}, {"init": "0", "op": "+="}),
    ("count_over", ["def count_over(xs, t):", "    c = dict()", "    for x in xs:", "        key = x {cmp} t",
                    "        c[key] = c.get(key, 0) + {inc}", "    return c.get(True, 0)"],
     {"cmp": ">", "inc": "1"}, {"cmp": "<", "inc": "x"}),
    ("first_index", ["def first_index(xs, t):", "    for i in range({start}, len(xs)):",
                     "        if xs[i] {cmp} t:", "            return i", "    return -1"],
     {"start": "0", "cmp": "=="}, {"start": "1", "cmp": ">="}),
]
CB_TEXT = {"en": "This function has exactly one bug.\n\n```python\n{src}\n```\n\nFor {call} the expected result "
                 "is {want}, but it returns {got}. Which line number contains the bug?",
           "de": "Diese Funktion enthält genau einen Fehler.\n\n```python\n{src}\n```\n\nFür {call} wäre das "
                 "erwartete Ergebnis {want}, sie liefert aber {got}. In welcher Zeile steckt der Fehler? Nenne die Zeilennummer."}


def cb_run(src, name, args):
    ns = {}
    exec(compile(src, "<generated>", "exec"), {"__builtins__": {"range": range, "len": len, "max": max,  # noqa: S102
                                                                "enumerate": enumerate, "dict": dict}}, ns)
    try:
        return repr(ns[name](*args))
    except Exception as e:  # noqa: BLE001 — the bug may raise; that is the observed behaviour
        return f"{type(e).__name__}"


def load_code_bug(lang, n, seed):
    combos = [(f, k) for f in CB_FUNCS for k in sorted(f[3])]  # every (function, bug kind) in turn
    items = []
    for i in range(n or 160):
        r = rng_of("code_bug", i)
        (name, lines, good, bad), key = combos[i % len(combos)]
        fix = "\n".join(lines).format(**good)
        buggy = "\n".join(lines).format(**(good | {key: bad[key]}))
        for _ in range(500):  # inputs until the example exposes the bug
            xs = [r.randint(-5, 20) for _ in range(r.randint(5, 8))]
            args = (xs, r.randint(0, 10)) if name in ("sum_above", "count_over") else (xs,)
            if name == "first_index":
                args = (xs, r.choice(xs))
            if name == "product":
                args = ([r.randint(1, 6) for _ in range(r.randint(3, 5))],)
            if name == "max_gap":
                args = (sorted(xs),)
            want, got = cb_run(fix, name, args), cb_run(buggy, name, args)
            if want != got:
                break
        else:
            raise RuntimeError(f"code_bug: {name}.{key} never exposed")
        line = next(k + 1 for k, ln in enumerate(lines) if "{" + key + "}" in ln)
        numbered = "\n".join(f"{k + 1:>2}  {ln}" for k, ln in enumerate(buggy.splitlines()))
        call = f"{name}({', '.join(map(repr, args))})"
        items.append({"id": str(i), "lang": lang, "gold": line, "bug": f"{name}.{key}",
                      "messages": [{"role": "user", "content": CB_TEXT[lang].format(
                          src=numbered, call=call, want=want, got=got) + "\n" + ASK[lang]}]})
    return items


def score_code_bug(item, reply):
    line = final_answer(reply["content"]) or ""
    m = re.search(r"(?:line|zeile)\s*(\d+)", line, re.I) or re.match(r"\W*(\d+)", line)
    v = int(m[1]) if m else None
    return {"correct": v == item["gold"], "score": float(v == item["gold"]), "pred": v, "gold": item["gold"],
            "detail": {"bug": item["bug"]}}


# ---------------------------------------------------------------- timezones
TZ_CITIES = [("Berlin", 60), ("New York", -300), ("Mumbai", 330), ("Tokio", 540), ("St. John's", -210),
             ("Chatham Islands", 765), ("Los Angeles", -480), ("Kathmandu", 345), ("São Paulo", -180), ("Sydney", 600)]
TZ_DAYS = {"en": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
           "de": ["Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag", "Sonntag"]}
TZ_TEXT = {"en": "A call starts on {day} at {t} in {a} (UTC{oa}). {b} is at UTC{ob}. What weekday and local time is it "
                 "in {b} when the call starts? Answer as \"<weekday> HH:MM\".",
           "de": "Ein Anruf beginnt am {day} um {t} Uhr in {a} (UTC{oa}). {b} liegt bei UTC{ob}. Welcher Wochentag und "
                 "welche Uhrzeit sind dann in {b}? Antworte als \"<Wochentag> HH:MM\"."}


def tz_fmt(m):
    sign = "+" if m >= 0 else "-"
    h, mm = divmod(abs(m), 60)
    return f"{sign}{h}" + (f":{mm:02d}" if mm else "")


def load_timezones(lang, n, seed):
    items = []
    for i in range(n or 150):
        r = rng_of("timezones", i)
        (a, oa), (b, ob) = r.sample(TZ_CITIES, 2)
        day, t = r.randrange(7), r.randrange(0, 24 * 60, 15)
        m = day * 1440 + t - oa + ob
        d2, t2 = divmod(m % (7 * 1440), 1440)
        gold = (TZ_DAYS[lang][d2], f"{t2 // 60:02d}:{t2 % 60:02d}")
        items.append({"id": str(i), "lang": lang, "gold": gold, "shift": d2 != day,
                      "messages": [{"role": "user", "content": TZ_TEXT[lang].format(
                          day=TZ_DAYS[lang][day], t=f"{t // 60:02d}:{t % 60:02d}", a=a, b=b, oa=tz_fmt(oa),
                          ob=tz_fmt(ob)) + "\n" + ASK[lang]}]})
    return items


def score_timezones(item, reply):
    line = final_answer(reply["content"]) or ""
    days = {d for lang in TZ_DAYS for d in TZ_DAYS[lang] if re.search(rf"\b{d}\b", line, re.I)}
    m = (re.findall(r"(\d{1,2})[:.](\d{2})", line) or [None])[-1]  # the last time: "Mumbai (UTC+5:30): Tue 04:30"
    t = f"{int(m[0]):02d}:{m[1]}" if m else None
    want_days = {TZ_DAYS["en"][TZ_DAYS[item["lang"]].index(item["gold"][0])], item["gold"][0]}
    ok = bool(days) and days <= want_days and t == item["gold"][1]
    return {"correct": ok, "score": float(ok), "pred": line[:40], "gold": " ".join(item["gold"]),
            "detail": {"ok_dayshift" if item["shift"] else "ok_sameday": float(ok)}}


# ---------------------------------------------------------------- calib_math
CM_ASK = {"en": " After the answer line add a line \"Confidence: N%\" — how sure you are that the answer is right.",
          "de": " Füge nach der Antwortzeile eine Zeile \"Konfidenz: N%\" hinzu — wie sicher du bist, dass die "
                "Antwort stimmt."}
CM_CONF = re.compile(r"(?i)(?:confidence|konfidenz|sicherheit)\W{0,3}\s*(\d{1,3}(?:[.,]\d+)?)\s*%?")


def load_calib_math(lang, n, seed):
    half = (n or 120) // 2
    items = []
    for kind, src in (("variants", load_math_variants(lang, half, seed)), ("ops", load_invented_ops(lang, half, seed))):
        for it in src:
            msg = it["messages"][0]["content"]
            items.append(it | {"id": f"{kind}:{it['id']}", "kind": kind,
                               "messages": [{"role": "user", "content": msg + CM_ASK[lang]}]})
    items.sort(key=lambda x: int(x["id"].split(":")[1]))  # interleave the two sources; quick stays a prefix
    return items


def score_calib_math(item, reply):
    ok = last_int(final_answer(reply["content"])) == item["gold"]
    m = CM_CONF.findall(reply["content"])
    v = float(m[-1].replace(",", ".")) if m else None
    pct = bool(m) and re.search(rf"{re.escape(m[-1])}\s*%", reply["content"]) is not None
    conf = None if v is None else v if v <= 1 and not pct else min(v, 100) / 100  # "0.9" or "90%"
    brier = 1.0 if conf is None else (conf - ok) ** 2
    detail = {"brier": round(brier, 4), "conf_missing": float(conf is None), "acc": float(ok)}
    if conf is not None:
        detail |= {"conf": conf, "overconf": round(conf - ok, 4)}
    return {"correct": ok, "score": round(1 - brier, 4), "pred": reply["content"][-80:], "gold": item["gold"],
            "detail": detail}


# ---------------------------------------------------------------- kinship
KIN_M = ["Arno", "Bodo", "Cord", "Dirk", "Egon", "Falk", "Gero", "Hajo", "Ilja", "Jost"]
KIN_F = ["Alma", "Berta", "Cleo", "Dora", "Elsa", "Frida", "Gesa", "Hilde", "Ines", "Jule"]
KIN_T = {"en": {"mother": "{a} is the mother of {b}.", "father": "{a} is the father of {b}.",
                "sib": "{a} and {b} are siblings.", "married": "{a} is married to {b}.",
                "q": {"grandmother": "Who is the grandmother of {x}?", "grandfather": "Who is the grandfather of {x}?",
                      "aunt": "Who is the aunt of {x}?", "uncle": "Who is the uncle of {x}?",
                      "cousin": "Who is the cousin of {x}?"}},
         "de": {"mother": "{a} ist die Mutter von {b}.", "father": "{a} ist der Vater von {b}.",
                "sib": "{a} und {b} sind Geschwister.", "married": "{a} ist mit {b} verheiratet.",
                "q": {"grandmother": "Wer ist die Großmutter von {x}?", "grandfather": "Wer ist der Großvater von {x}?",
                      "aunt": "Wer ist die Tante von {x}?", "uncle": "Wer ist der Onkel von {x}?",
                      "cousin": "Wer ist die Cousine oder der Cousin von {x}?"}}}


def load_kinship(lang, n, seed):
    items = []
    rels = ("grandmother", "grandfather", "aunt", "uncle", "cousin")
    for i in range(n or 150):
        r = rng_of("kinship", i)
        m, f = r.sample(KIN_M, 5), r.sample(KIN_F, 5)
        gm, gf = f[0], m[0]                       # grandparents
        kids = [(f[1], "f"), (m[1], "m")]         # their daughter and son
        r.shuffle(kids)
        sp = {f[1]: m[2], m[1]: f[2]}             # spouses (married in, not blood relatives)
        child = {f[1]: (m[3], "m"), m[1]: (f[3], "f")}  # one grandchild per child
        facts = [("mother", gm, k) for k, _ in kids] + [("father", gf, k) for k, _ in kids]
        facts += [("sib", kids[0][0], kids[1][0])] + [("married", k, sp[k]) for k, _ in kids]
        for k, g in kids:
            c = child[k][0]
            facts += [("mother" if g == "f" else "father", k, c),
                      ("father" if g == "f" else "mother", sp[k], c)]
        rel = rels[i % len(rels)]
        k_x, g_x = r.choice(kids)
        x = child[k_x][0]
        other, g_o = next((k, g) for k, g in kids if k != k_x)
        gold = {"grandmother": gm, "grandfather": gf, "cousin": child[other][0],
                "aunt": other if g_o == "f" else sp[other], "uncle": other if g_o == "m" else sp[other]}[rel]
        r.shuffle(facts)
        T = KIN_T[lang]
        text = "\n".join("- " + T[t].format(a=a, b=b) for t, a, b in facts) + "\n\n" + T["q"][rel].format(x=x)
        items.append({"id": str(i), "lang": lang, "gold": gold, "rel": rel, "names": m + f,
                      "messages": [{"role": "user", "content": text + "\n" + ASK[lang]}]})
    return items


def score_kinship(item, reply):
    line = final_answer(reply["content"]) or ""
    hits = [(m.start(), nm) for nm in item["names"] for m in [re.search(rf"\b{nm}\b", line)] if m]
    ok = bool(hits) and min(hits)[1] == item["gold"]  # the name the answer leads with; "(die Mutter von X)" is a note
    return {"correct": ok, "score": float(ok), "pred": line[:40], "gold": item["gold"], "detail": {f"ok_{item['rel']}": float(ok)}}


# ---------------------------------------------------------------- meeting_actions
MA_TASKS = {"en": ["the budget draft", "the customer survey", "the release notes", "the hiring plan", "the slide deck"],
            "de": ["den Budgetentwurf", "die Kundenumfrage", "die Release Notes", "den Einstellungsplan", "die Präsentation"]}
MA_KEYS = ["budget", "survey", "release", "hiring", "slides"]
MA_DE_DAT = ["beim Budgetentwurf", "bei der Kundenumfrage", "bei den Release Notes", "beim Einstellungsplan",
             "bei der Präsentation"]  # "Bei die Release Notes" is wrong: dative after "bei"
MA_DE_NOM = ["der Budgetentwurf", "die Kundenumfrage", "die Release Notes", "der Einstellungsplan", "die Präsentation"]
MA_DAYS = {"en": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
           "de": ["Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag"]}
MA_T = {"en": {"assign": "{who}: I'll take {task}, done by {day}.", "reassign": "{who}: Actually, I'll take {task} "
               "instead of {old}, same deadline.", "move": "{who}: For {task}, let's make the deadline {day} instead.",
               "chat": ["{who}: Did anyone see the game yesterday?", "{who}: Coffee machine is broken again.",
                        "{who}: Let's keep this short today."],
               "ask": "Here is a meeting transcript:\n\n{lines}\n\nReturn the FINAL tasks as JSON: an object mapping "
                      "each task key ({keys}) to [owner, weekday]. Reply with the JSON only."},
        "de": {"assign": "{who}: Ich übernehme {task}, fertig bis {day}.", "reassign": "{who}: Ich übernehme doch {task} "
               "statt {old}, gleiche Frist.", "move": "{who}: {dat} machen wir die Frist lieber {day}.",
               "chat": ["{who}: Hat jemand gestern das Spiel gesehen?", "{who}: Die Kaffeemaschine ist schon wieder "
                        "kaputt.", "{who}: Lasst uns heute kurz machen."],
               "ask": "Hier ist ein Meeting-Protokoll:\n\n{lines}\n\nGib die ENDGÜLTIGEN Aufgaben als JSON zurück: ein "
                      "Objekt, das jeden Aufgabenschlüssel ({keys}) auf [Person, Wochentag] abbildet. Antworte nur "
                      "mit dem JSON."}}


def load_meeting_actions(lang, n, seed):
    items = []
    for i in range(n or 150):
        r = rng_of("meeting_actions", i)
        people = r.sample(["Anna", "Ben", "Clara", "David", "Emma"], 4)
        ks = r.sample(range(5), r.randint(3, 4))
        T = MA_T[lang]
        state, lines = {}, []
        for k in ks:
            who, day = r.choice(people), r.randrange(5)
            state[k] = [who, day]
            lines.append(T["assign"].format(who=who, task=MA_TASKS[lang][k], day=MA_DAYS[lang][day]))
            if r.random() < 0.4:
                lines.append(r.choice(T["chat"]).format(who=r.choice(people)))
        k1, k2 = r.sample(ks, 2)
        new = r.choice([p for p in people if p != state[k1][0]])
        lines.append(T["reassign"].format(who=new, task=MA_TASKS[lang][k1], old=state[k1][0]))
        state[k1][0] = new
        nd = r.choice([d for d in range(5) if d != state[k2][1]])
        lines.append(T["move"].format(who=r.choice(people), task=MA_TASKS[lang][k2], day=MA_DAYS[lang][nd],
                                      dat=MA_DE_DAT[k2][0].upper() + MA_DE_DAT[k2][1:]))
        state[k2][1] = nd
        gold = {MA_KEYS[k]: [w, MA_DAYS["en"][d]] for k, (w, d) in state.items()}
        keys = ", ".join(f"{MA_KEYS[k]} = {(MA_DE_NOM if lang == 'de' else MA_TASKS[lang])[k]}" for k in ks)
        items.append({"id": str(i), "lang": lang, "gold": gold,
                      "messages": [{"role": "user", "content": T["ask"].format(lines="\n".join(lines), keys=keys)}]})
    return items


def score_meeting_actions(item, reply):
    import json as _json
    m = re.search(r"\{.*\}", reply["content"], re.S)
    try:
        got = _json.loads(m[0]) if m else {}
    except ValueError:
        got = {}
    day_en = {d.lower(): MA_DAYS["en"][k] for lang in MA_DAYS for k, d in enumerate(MA_DAYS[lang])}
    right = 0
    for key, (who, day) in item["gold"].items():
        v = got.get(key)
        if isinstance(v, list) and len(v) == 2 and str(v[0]).strip() == who and \
                day_en.get(str(v[1]).strip().lower()) == day:
            right += 1
    extra = set(got) - set(item["gold"]) if isinstance(got, dict) else set()
    share = right / len(item["gold"])
    return {"correct": share == 1 and not extra, "score": round(share, 4), "pred": str(got)[:160],
            "gold": str(item["gold"]), "detail": {"extra_tasks": float(bool(extra))}}


# ---------------------------------------------------------------- game_theory
GT_ROW = {"en": ("Up", "Down"), "de": ("Oben", "Unten")}
GT_COL = {"en": ("Left", "Right"), "de": ("Links", "Rechts")}
GT_TEXT = {"en": "Two players move at the same time. Row player A picks Up or Down, column player B picks Left or "
                 "Right. Payoffs (A, B):\n{table}\nWhich cell is the unique pure-strategy Nash equilibrium? Answer "
                 "as \"Row-Column\", e.g. \"Up-Left\".",
           "de": "Zwei Spieler ziehen gleichzeitig. Zeilenspieler A wählt Oben oder Unten, Spaltenspieler B wählt Links "
                 "oder Rechts. Auszahlungen (A, B):\n{table}\nWelche Zelle ist das eindeutige Nash-Gleichgewicht in "
                 "reinen Strategien? Antworte als \"Zeile-Spalte\", z. B. \"Oben-Links\"."}


def gt_equilibria(p):
    """p[r][c] = (a, b); pure Nash cells."""
    return [(r_, c) for r_ in (0, 1) for c in (0, 1)
            if p[r_][c][0] >= p[1 - r_][c][0] and p[r_][c][1] >= p[r_][1 - c][1]]


def load_game_theory(lang, n, seed):
    items, i = [], -1
    while len(items) < (n or 150):
        i += 1
        r = rng_of("game_theory", i)
        p = [[(r.randint(0, 9), r.randint(0, 9)) for _ in range(2)] for _ in range(2)]
        eq = gt_equilibria(p)
        flat = [x for row in p for cell in row for x in cell]
        if len(eq) != 1 or len(set(flat)) < 5:
            continue
        (er, ec), = eq
        dom = all(p[er][c][0] > p[1 - er][c][0] for c in (0, 1)) and all(p[r_][ec][1] > p[r_][1 - ec][1]
                                                                        for r_ in (0, 1))
        R, C = GT_ROW[lang], GT_COL[lang]
        table = "\n".join(f"{R[r_]}: " + " | ".join(f"{C[c]} ({p[r_][c][0]}, {p[r_][c][1]})" for c in (0, 1))
                          for r_ in (0, 1))
        items.append({"id": str(i), "lang": lang, "gold": (er, ec), "dominant": dom,
                      "messages": [{"role": "user", "content": GT_TEXT[lang].format(table=table) + "\n" + ASK[lang]}]})
    return items


def score_game_theory(item, reply):
    line = (final_answer(reply["content"]) or "").lower()
    rows = {k for k in (0, 1) for w in (GT_ROW["en"][k], GT_ROW["de"][k]) if re.search(rf"\b{w.lower()}\b", line)}
    cols = {k for k in (0, 1) for w in (GT_COL["en"][k], GT_COL["de"][k]) if re.search(rf"\b{w.lower()}\b", line)}
    ok = rows == {item["gold"][0]} and cols == {item["gold"][1]}
    return {"correct": ok, "score": float(ok), "pred": line[:30], "gold": str(item["gold"]),
            "detail": {"ok_dominant" if item["dominant"] else "ok_nondominant": float(ok)}}


# ---------------------------------------------------------------- rulebook
RB_TEXT = {"en": ("The parcel office of Velmora charges in krel:\n"
                  "1. Base fee: {b1} krel up to {w1} kg, {b2} krel up to {w2} kg, otherwise {b3} krel.\n"
                  "2. Express adds {x} % of the base fee, except for parcels to islands, where express adds a flat "
                  "{xi} krel instead.\n"
                  "3. Members get {d} krel off the total, but never pay less than {floor} krel.\n\n"
                  "Case: a {w} kg parcel {dest}, {express}, sender {member}. What is the fee in krel?"),
           "de": ("Das Paketamt von Velmora berechnet in Krel:\n"
                  "1. Grundgebühr: {b1} Krel bis {w1} kg, {b2} Krel bis {w2} kg, sonst {b3} Krel.\n"
                  "2. Express kostet {x} % der Grundgebühr zusätzlich, außer bei Paketen auf Inseln: dort kostet "
                  "Express pauschal {xi} Krel zusätzlich.\n"
                  "3. Mitglieder erhalten {d} Krel Nachlass auf die Summe, zahlen aber nie weniger als {floor} Krel.\n\n"
                  "Fall: ein {w}-kg-Paket {dest}, {express}, Absender {member}. Wie hoch ist die Gebühr in Krel?")}
RB_WORDS = {"en": {"island": "to the island of Sarn", "main": "to the city of Oltar", "ex": "express", "noex": "standard",
                   "mem": "is a member", "nomem": "is not a member"},
            "de": {"island": "auf die Insel Sarn", "main": "in die Stadt Oltar", "ex": "Express", "noex": "Standard",
                   "mem": "ist Mitglied", "nomem": "ist kein Mitglied"}}


def load_rulebook(lang, n, seed):
    items, i = [], -1
    while len(items) < (n or 150):
        i += 1
        r = rng_of("rulebook", i)
        w1, w2 = r.randint(2, 5), r.randint(8, 15)
        b1, b2, b3 = r.choice([8, 12, 16]), r.choice([20, 24, 28]), r.choice([40, 44, 48])  # x % stays whole
        x, xi, d, floor = r.choice([25, 50]), r.choice([6, 8, 15]), r.choice([5, 10, 15]), r.choice([10, 15, 20])
        w = r.choice([r.randint(1, w1), r.randint(w1 + 1, w2), r.randint(w2 + 1, 30)])
        island, express, member = r.random() < 0.5, r.random() < 0.7, r.random() < 0.5
        base = b1 if w <= w1 else b2 if w <= w2 else b3
        fee = base + ((xi if island else base * x // 100) if express else 0)
        if member and fee < floor:  # a "floor" that would RAISE a member's fee reads two ways: skip the draw
            continue
        if member:
            fee = max(fee - d, floor)
        W = RB_WORDS[lang]
        q = RB_TEXT[lang].format(b1=b1, b2=b2, b3=b3, w1=w1, w2=w2, x=x, xi=xi, d=d, floor=floor, w=w,
                                 dest=W["island" if island else "main"], express=W["ex" if express else "noex"],
                                 member=W["mem" if member else "nomem"])
        items.append({"id": str(i), "lang": lang, "gold": fee, "exception": island and express,
                      "messages": [{"role": "user", "content": q + "\n" + ASK[lang]}]})
    return items


def score_rulebook(item, reply):
    v = answer_value(final_answer(reply["content"]), item["lang"])
    ok = v == item["gold"]
    return {"correct": ok, "score": float(ok), "pred": v, "gold": item["gold"],
            "detail": {"ok_exception" if item["exception"] else "ok_regular": float(ok)}}


# ---------------------------------------------------------------- knights_knaves
KK_NAMES = ["Arin", "Belo", "Cira", "Doro"]
KK_T = {"en": {"knave": "{x} is a knave.", "knight": "{x} is a knight.", "same": "{x} and {y} are the same kind.",
               "diff": "{x} and {y} are different kinds.", "atleast": "At least one of us is a knave.",
               "intro": "On an island, knights always tell the truth and knaves always lie. {n} islanders speak:",
               "ask": "Who is a knight and who is a knave? Answer as \"Arin: knight/knave, Belo: ..., ...\".",
               "k": "knight", "n": "knave"},
        "de": {"knave": "{x} ist ein Schurke.", "knight": "{x} ist ein Ritter.", "same": "{x} und {y} sind von "
               "derselben Art.", "diff": "{x} und {y} sind von verschiedener Art.", "atleast": "Mindestens einer "
               "von uns ist ein Schurke.",
               "intro": "Auf einer Insel sagen Ritter immer die Wahrheit, und Schurken lügen immer. {n} Bewohner sagen:",
               "ask": "Wer ist Ritter, wer ist Schurke? Antworte als \"Arin: Ritter/Schurke, Belo: ..., ...\".",
               "k": "Ritter", "n": "Schurke"}}


def kk_true(st, w):
    kind, a, b = st
    if kind == "atleast":
        return not all(w)
    if kind in ("knave", "knight"):
        return w[a] == (kind == "knight")
    return (w[a] == w[b]) == (kind == "same")


def load_knights(lang, n, seed):
    items, i = [], -1
    while len(items) < (n or 150):
        i += 1
        r = rng_of("knights_knaves", i)
        k = 3 + i % 2  # three or four islanders
        sts = []
        for sp in range(k):
            kind = r.choice(["knave", "knight", "same", "diff", "atleast"])
            a, b = r.sample([x for x in range(k) if x != sp], 2) if kind in ("same", "diff") else \
                (r.choice([x for x in range(k) if x != sp]), None)
            sts.append((kind, a, b))
        worlds = [w for w in itertools.product([True, False], repeat=k)
                  if all(kk_true(sts[sp], w) == w[sp] for sp in range(k))]
        if len(worlds) != 1:
            continue
        T = KK_T[lang]
        lines = [f"{KK_NAMES[sp]}: \"" + T[k].format(x=KK_NAMES[a] if a is not None else "",
                                                     y=KK_NAMES[b] if b is not None else "") + "\""
                 for sp, (k, a, b) in enumerate(sts)]
        items.append({"id": str(i), "lang": lang, "gold": worlds[0],
                      "messages": [{"role": "user", "content": T["intro"].format(n=k) + "\n" + "\n".join(lines) + "\n" +
                                    T["ask"] + "\n" + ASK[lang]}]})
    return items


def score_knights(item, reply):
    line = final_answer(reply["content"]) or ""
    words = {"knight": True, "ritter": True, "knave": False, "schurke": False}
    got = []
    for nm in KK_NAMES[:len(item["gold"])]:
        m = re.search(rf"{nm}\W*\s*(?:is\s+(?:a\s+)?|ist\s+(?:ein\s+)?)?(knights?|knaves?|ritter|schurken?)\b", line,
                      re.I)
        if not m:  # "Belo and Cira: knights" — the next role word after the name
            m = re.search(rf"{nm}\b.*?\b(knights?|knaves?|ritter|schurken?)\b", line, re.I)
        got.append(words[re.sub(r"(s|n)$", "", m[1].lower())] if m and re.sub(r"(s|n)$", "", m[1].lower()) in words
                   else (words.get(m[1].lower()) if m else None))
    right = sum(g == w for g, w in zip(got, item["gold"]))
    return {"correct": right == len(item["gold"]), "score": round(right / len(item["gold"]), 4), "pred": line[:80], "gold": str(item["gold"]),
            "detail": {}}


# ---------------------------------------------------------------- number_bases
NB_TEXT = {"en": {"convert": "Convert the base-{a} number {x} to base {b}.", "add": "Add the binary numbers {x} and "
                  "{y}. Give the result in binary."},
           "de": {"convert": "Wandle die Zahl {x} aus dem {a}er-System ins {b}er-System um.",
                  "add": "Addiere die Binärzahlen {x} und {y}. Gib das Ergebnis im Binärsystem an."}}


def to_base(v, b):
    digits = "0123456789abcdef"
    out = ""
    while True:
        v, d = divmod(v, b)
        out = digits[d] + out
        if not v:
            return out


def load_number_bases(lang, n, seed):
    items = []
    for i in range(n or 150):
        r = rng_of("number_bases", i)
        if i % 3 == 2:
            x, y = r.randint(2 ** 9, 2 ** 14), r.randint(2 ** 9, 2 ** 14)
            q, gold = NB_TEXT[lang]["add"].format(x=to_base(x, 2), y=to_base(y, 2)), (to_base(x + y, 2), 2)
            kind = "add"
        else:
            a, b = r.sample([2, 8, 10, 16], 2)
            v = r.randint(200, 70000)
            q, gold, kind = NB_TEXT[lang]["convert"].format(a=a, b=b, x=to_base(v, a)), (to_base(v, b), b), "convert"
        items.append({"id": str(i), "lang": lang, "gold": gold[0], "base": gold[1], "kind": kind,
                      "messages": [{"role": "user", "content": q + "\n" + ASK[lang]}]})
    return items


def score_number_bases(item, reply):
    line = (final_answer(reply["content"]) or "").lower().replace(" ", "")
    line = re.sub(r"^[$`*]*(?:\\boxed\{)?", "", line)
    line = re.sub(r"^(0x|0b|0o)", "", re.split(r"[_(₂₈₁]", line)[0])  # "166760_8", "1011₂", "123 (octal)"
    m = re.match(r"[0-9a-f]+", line)
    got = m[0].lstrip("0") or "0" if m else ""
    ok = got == item["gold"].lstrip("0") or (got == "0" and item["gold"] == "0")
    return {"correct": ok, "score": float(ok), "pred": line[:30], "gold": item["gold"], "detail": {f"ok_{item['kind']}": float(ok)}}


# ---------------------------------------------------------------- copycat
CC_RULES = {
    "succ_last": lambda s_: s_[:-1] + chr(ord(s_[-1]) + 1),
    "pred_first": lambda s_: chr(ord(s_[0]) - 1) + s_[1:],
    "reverse": lambda s_: s_[::-1],
    "double_last": lambda s_: s_ + s_[-1],
    "succ_all": lambda s_: "".join(chr(ord(c) + 1) for c in s_),
}
CC_TEXT = {"en": "If \"{a}\" changes to \"{b}\", what does \"{c}\" change to in the same way?",
           "de": "Wenn \"{a}\" zu \"{b}\" wird, was wird dann auf dieselbe Weise aus \"{c}\"?"}


def cc_run(r, n_):
    start = r.randint(1, 24 - n_)  # never 'a' (pred) and never past 'y' (succ)
    return "".join(chr(97 + start + k) for k in range(n_))


def load_copycat(lang, n, seed):
    items = []
    names = sorted(CC_RULES)
    for i in range(n or 150):
        r = rng_of("copycat", i)
        rule = names[i % len(names)]
        a = cc_run(r, r.randint(3, 4))
        c = cc_run(r, r.randint(4, 6))
        while c == a:
            c = cc_run(r, r.randint(4, 6))
        f = CC_RULES[rule]
        items.append({"id": str(i), "lang": lang, "gold": f(c), "rule": rule,
                      "messages": [{"role": "user", "content": CC_TEXT[lang].format(a=a, b=f(a), c=c) + "\n" + ASK[lang]}]})
    return items


def score_copycat(item, reply):
    pred = (final_answer(reply["content"]) or "").strip().strip('"\'`*').lower()
    ok = pred == item["gold"]
    return {"correct": ok, "score": float(ok), "pred": pred[:20], "gold": item["gold"], "detail": {f"ok_{item['rule']}": float(ok)}}


# ---------------------------------------------------------------- gears
GR_TEXT = {"en": {"mesh": "Wheel {a} ({ta} teeth) meshes directly with wheel {b} ({tb} teeth).",
                  "belt": "Wheel {a} ({ta} teeth) drives wheel {b} ({tb} teeth) with an open toothed belt.",
                  "start": "Wheel A turns clockwise at {rpm} rpm.",
                  "dir": "Which way does wheel {z} turn? Answer clockwise or counterclockwise.",
                  "rpm": "How many rpm does wheel {z} turn at?"},
           "de": {"mesh": "Rad {a} ({ta} Zähne) greift direkt in Rad {b} ({tb} Zähne) ein.",
                  "belt": "Rad {a} ({ta} Zähne) treibt Rad {b} ({tb} Zähne) über einen offenen Zahnriemen an.",
                  "start": "Rad A dreht sich mit {rpm} U/min im Uhrzeigersinn.",
                  "dir": "In welche Richtung dreht sich Rad {z}? Antworte mit „im Uhrzeigersinn“ oder „gegen den "
                         "Uhrzeigersinn“.",
                  "rpm": "Mit wie vielen U/min dreht sich Rad {z}?"}}


def load_gears(lang, n, seed):
    items = []
    for i in range(n or 150):
        r = rng_of("gears", i)
        k = r.randint(4, 8)
        names = [chr(65 + j) for j in range(k)]
        teeth = [r.choice([10, 12, 15, 20, 24, 30, 40, 60]) for _ in range(k)]
        links = [r.choice(["mesh", "mesh", "belt"]) for _ in range(k - 1)]
        rpm = r.choice([60, 120, 240, 360])
        cw = True
        for l_ in links:
            cw = cw if l_ == "belt" else not cw
        final = Fraction(rpm * teeth[0], teeth[-1])
        kind = "dir" if i % 2 == 0 else "rpm"
        if kind == "rpm" and final.denominator != 1:
            teeth[-1] = teeth[0]  # keep the answer whole
            final = Fraction(rpm)
        T = GR_TEXT[lang]
        lines = [T[l_].format(a=names[j], b=names[j + 1], ta=teeth[j], tb=teeth[j + 1]) for j, l_ in enumerate(links)]
        r.shuffle(lines)
        q = T["start"].format(rpm=rpm) + "\n" + "\n".join(lines) + "\n" + T[kind].format(z=names[-1])
        items.append({"id": str(i), "lang": lang, "kind": kind,
                      "gold": ("cw" if cw else "ccw") if kind == "dir" else int(final),
                      "messages": [{"role": "user", "content": q + "\n" + ASK[lang]}]})
    return items


def score_gears(item, reply):
    line = (final_answer(reply["content"]) or "").lower()
    if item["kind"] == "dir":
        ccw = re.search(r"counter|anti|gegen|links", line) is not None
        cw = re.search(r"clockwise|uhrzeigersinn|rechts", line) is not None
        negated = re.search(r"\bnot\b|\bnicht\b", line) is not None  # "clockwise (not counterclockwise)": unclear
        pred = None if negated else "ccw" if ccw else "cw" if cw else None
    else:
        pred = answer_value(line, item["lang"])
        pred = int(pred) if pred is not None and pred == int(pred) else pred
    ok = pred == item["gold"]
    return {"correct": ok, "score": float(ok), "pred": pred, "gold": item["gold"], "detail": {f"ok_{item['kind']}": float(ok)}}


# ---------------------------------------------------------------- jugs
JG_TEXT = {"en": "You have a {a}-litre jug and a {b}-litre jug, both empty, and unlimited water. One move is: fill a jug "
                 "completely, empty a jug completely, or pour from one jug into the other until one is empty or the "
                 "other full. What is the smallest number of moves to have exactly {t} litres in one of the jugs? If it "
                 "cannot be done, answer \"impossible\".",
           "de": "Du hast einen {a}-Liter-Krug und einen {b}-Liter-Krug, beide leer, und beliebig viel Wasser. Ein Zug "
                 "ist: einen Krug ganz füllen, einen Krug ganz leeren oder von einem Krug in den anderen gießen, bis "
                 "einer leer oder der andere voll ist. Wie viele Züge braucht man mindestens, damit in einem Krug "
                 "genau {t} Liter sind? Wenn es nicht geht, antworte \"unmöglich\"."}


def jug_bfs(a, b, t):
    from collections import deque
    seen, q = {(0, 0): 0}, deque([(0, 0)])
    while q:
        x, y = q.popleft()
        d = seen[(x, y)]
        if t in (x, y):
            return d
        p1, p2 = min(x, b - y), min(y, a - x)
        for nx in ((a, y), (x, b), (0, y), (x, 0), (x - p1, y + p1), (x + p2, y - p2)):
            if nx not in seen:
                seen[nx] = d + 1
                q.append(nx)
    return None


def load_jugs(lang, n, seed):
    items = []
    for i in range(n or 150):
        r = rng_of("jugs", i)
        while True:
            a, b = sorted(r.sample(range(3, 14), 2))
            t = r.randint(1, b + (2 if i % 4 == 3 else 0))
            if t in (a, b):
                continue
            steps = jug_bfs(a, b, t)
            if (steps is None) == (i % 4 == 3):  # a quarter impossible, in id order
                break
        items.append({"id": str(i), "lang": lang, "gold": steps,
                      "messages": [{"role": "user", "content": JG_TEXT[lang].format(a=a, b=b, t=t) + "\n" + ASK[lang]}]})
    return items


def score_jugs(item, reply):
    line = (final_answer(reply["content"]) or "").lower()
    imp = re.search(r"impossible|unmöglich|not possible|nicht möglich|cannot be done|can't be done|geht nicht|"
                    r"nicht machbar|no solution|keine lösung", line) is not None
    v = answer_value(line, item["lang"]) if not imp else None
    ok = (item["gold"] is None and imp) or (item["gold"] is not None and v == item["gold"])
    return {"correct": ok, "score": float(ok), "pred": "impossible" if imp else v, "gold": item["gold"],
            "detail": {"ok_impossible" if item["gold"] is None else "ok_possible": float(ok)}}


# ---------------------------------------------------------------- linalg
LA_TEXT = {"en": {"det": "Compute the determinant of the matrix\n{m}", "solve": "Solve the system for x, y, z:\n{eqs}\n"
                  "Answer as \"x, y, z\".", "matvec": "Compute M·v for\nM =\n{m}\nv = ({v})\nAnswer as a comma-separated "
                  "vector."},
           "de": {"det": "Berechne die Determinante der Matrix\n{m}", "solve": "Löse das Gleichungssystem nach x, y, z:\n"
                  "{eqs}\nAntworte als \"x, y, z\".", "matvec": "Berechne M·v für\nM =\n{m}\nv = ({v})\nAntworte als "
                  "Vektor mit Kommas."}}


def det3(m):
    a, b, c = m
    return (a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
            + a[2] * (b[0] * c[1] - b[1] * c[0]))


def load_linalg(lang, n, seed):
    items = []
    for i in range(n or 150):
        r = rng_of("linalg", i)
        kind = ("det", "solve", "matvec")[i % 3]
        while True:
            m = [[r.randint(-5, 7) for _ in range(3)] for _ in range(3)]
            if det3(m) != 0:
                break
        rows = "\n".join("[" + ", ".join(f"{x:>2}" for x in row) + "]" for row in m)
        T = LA_TEXT[lang]
        if kind == "det":
            q, gold = T["det"].format(m=rows), [det3(m)]
        elif kind == "solve":
            sol = [r.randint(-6, 6) for _ in range(3)]
            rhs = [sum(a * s_ for a, s_ in zip(row, sol)) for row in m]

            def term(a, v, first):
                if a == 0:
                    return ""
                sign = "-" if a < 0 else ("" if first else "+")
                mag = "" if abs(a) == 1 else str(abs(a))
                return f"{sign} {mag}{v} ".replace("  ", " ") if not first else f"{'-' if a < 0 else ''}{mag}{v} "
            eqs = "\n".join("".join(term(a, v, k == next(j for j, x in enumerate(row) if x))
                                     for k, (a, v) in enumerate(zip(row, "xyz"))).strip() + f" = {rh}"
                             for row, rh in zip(m, rhs))
            q, gold = T["solve"].format(eqs=eqs), sol
        else:
            v = [r.randint(-4, 6) for _ in range(3)]
            q, gold = T["matvec"].format(m=rows, v=", ".join(map(str, v))), [sum(a * b for a, b in zip(row, v)) for row in m]
        items.append({"id": str(i), "lang": lang, "gold": gold, "kind": kind,
                      "messages": [{"role": "user", "content": q + "\n" + ASK[lang]}]})
    return items


def score_linalg(item, reply):
    line = (final_answer(reply["content"]) or "").replace("−", "-")
    line = line.rsplit("=", 1)[1] if item["kind"] == "det" and "=" in line else line
    named = dict(re.findall(r"\b([xyz])\s*=\s*(-?\d+)", line))
    if item["kind"] == "solve" and len(named) == 3:  # "x = 1, y = 4, z = -2 (check: ...)"
        nums = [int(named[v]) for v in "xyz"]
    else:
        nums = [int(x) for x in re.findall(r"-?\d+", line.split("(")[0] if "(" in line and not line.lstrip().startswith("(") else line)][:len(item["gold"])]
    ok = nums == item["gold"]
    return {"correct": ok, "score": float(ok), "pred": line[:60], "gold": item["gold"], "detail": {f"ok_{item['kind']}": float(ok)}}


# ---------------------------------------------------------------- self_correct
SELF_TEXT = {"en": ("What is {a} times {b}?", "{a} × {b} = {w}.", "And if I add {c} to that result?"),
           "de": ("Was ist {a} mal {b}?", "{a} × {b} = {w}.", "Und wenn ich zu dem Ergebnis {c} addiere?")}


def load_self_correct(lang, n, seed):
    items = []
    for i in range(n or 150):
        r = rng_of("self_correct", i)
        a, b, c = r.randint(12, 49), r.randint(12, 49), r.randint(5, 60)
        wrong = a * b + r.choice([-10, 10, -20, 20, 1, -1])
        q1, a1, q2 = SELF_TEXT[lang]
        items.append({"id": str(i), "lang": lang, "gold": a * b + c, "carried": wrong + c,
                      "messages": [{"role": "user", "content": q1.format(a=a, b=b)},
                                   {"role": "assistant", "content": a1.format(a=a, b=b, w=wrong)},
                                   {"role": "user", "content": q2.format(c=c) + " " + ASK[lang]}]})
    return items


def score_self_correct(item, reply):
    v = answer_value(final_answer(reply["content"]), item["lang"])
    ok = v == item["gold"]
    return {"correct": ok, "score": float(ok), "pred": v, "gold": item["gold"],
            "detail": {"carried": float(v == item["carried"])}}


# ---------------------------------------------------------------- data2text
D2T_CITIES = {"en": ["Hamburg", "Munich", "Cologne", "Leipzig", "Dresden", "Bremen"],
              "de": ["Hamburg", "München", "Köln", "Leipzig", "Dresden", "Bremen"]}
D2T_ASK = {"en": "Write a short weather report (3-5 sentences) from this data. Mention every value, written in digits, "
                 "and add no other numbers.\n\n{table}",
           "de": "Schreibe aus diesen Daten einen kurzen Wetterbericht (3-5 Sätze). Nenne jeden Wert in Ziffern und füge "
                 "keine anderen Zahlen hinzu.\n\n{table}"}
D2T_HEAD = {"en": "| City | High °C | Low °C | Rain % | Wind km/h |", "de": "| Stadt | Höchst °C | Tiefst °C | Regen % | Wind km/h |"}


def load_data2text(lang, n, seed):
    items = []
    for i in range(n or 120):
        r = rng_of("data2text", i)
        cities = r.sample(range(6), 3)
        rows = []
        for c in cities:
            lo = r.randint(-5, 15)
            rows.append((D2T_CITIES[lang][c], lo + r.randint(4, 12), lo, r.choice(range(0, 100, 5)), r.randint(5, 60)))
        table = "\n".join([D2T_HEAD[lang], "|---|---|---|---|---|"] + ["| " + " | ".join(map(str, row)) + " |" for row in rows])
        values = sorted({v for row in rows for v in row[1:]})
        items.append({"id": str(i), "lang": lang, "rows": rows, "values": values, "gold": None,
                      "messages": [{"role": "user", "content": D2T_ASK[lang].format(table=table)}]})
    return items


def score_data2text(item, reply):
    t = reply["content"].replace("−", "-")
    t = re.sub(r"\bminus\s+(?=\d)", "-", t, flags=re.I)
    t = re.sub(r"(?<=\d)\s*[–—]\s*(?=-?\d)", " bis ", t)  # "13–20" is a range, not minus 20
    decimals = re.findall(r"\d+[.,]\d+", t)  # "12,5 Grad" is not in the data (all values are whole)
    nums = [int(x) for x in re.findall(r"(?<![\d.,])-?\d+(?![\d.,]\d)", t)]
    allowed = set(item["values"]) | {3}
    want = [v for row in item["rows"] for v in row[1:]]
    stated = sum(v in nums for v in want) / len(want)  # a dropped minus sign is a different temperature
    invented = sorted({x for x in nums if x not in allowed}) + decimals
    return {"correct": stated == 1 and not invented, "score": 0.0 if invented else round(stated, 4),
            "pred": t[:200], "gold": None,
            "detail": {"coverage": round(stated, 4), "invented": float(bool(invented))}}


# ---------------------------------------------------------------- negation_qa
NQ_ITEMS = {"en": ["the apples", "the bread", "the milk", "the cheese", "the flowers", "the coffee"],
            "de": ["die Äpfel", "das Brot", "die Milch", "den Käse", "die Blumen", "den Kaffee"]}
NQ_T = {"en": {"notbut": "Not {a} but {b} bought {x}.", "neither": "{a} bought neither {x} nor {y}; {b} bought both.",
               "except": "Everyone except {a} bought {x}; {a} bought nothing at all. Of the others, only {b} also bought {y}.",
               "q": "Who bought {y}? Answer with one name.", "qnot": "Who did NOT buy {y}? Answer with one name.",
               "distract": "{c} had thought about buying {y} but changed plans."},
        "de": {"notbut": "Nicht {a}, sondern {b} hat {x} gekauft.",
               "neither": "{a} hat weder {x} noch {y} gekauft; {b} hat beides gekauft.",
               "except": "Alle außer {a} haben {x} gekauft; {a} hat gar nichts gekauft. Von den anderen hat nur {b} auch {y} "
                          "gekauft.",
               "q": "Wer hat {y} gekauft? Antworte mit einem Namen.",
               "qnot": "Wer hat {y} NICHT gekauft? Antworte mit einem Namen.",
               "distract": "{c} hatte überlegt, {y} zu kaufen, hat es dann aber gelassen."}}


def load_negation_qa(lang, n, seed):
    items = []
    pats = ("notbut", "neither", "except", "except_not")
    for i in range(n or 160):
        r = rng_of("negation_qa", i)
        pat = pats[i % 4]
        a, b, c = r.sample(PEOPLE["en"], 3)
        xi, yi = r.sample(range(6), 2)
        X, Y = NQ_ITEMS[lang][xi], NQ_ITEMS[lang][yi]
        T = NQ_T[lang]
        if pat == "notbut":  # the question asks about X itself
            text, q_item = T["notbut"].format(a=a, b=b, x=X), X
        elif pat == "except_not":  # "Everyone except A bought X" -> who did NOT buy X: A
            text, q_item = T["except"].format(a=a, b=b, x=X, y=Y), X
        else:
            text, q_item = T[pat].format(a=a, b=b, x=X, y=Y), Y
        # a third person who only THOUGHT about buying; under "except_not" that must be Y, not X (everyone but A
        # bought X, so a second non-buyer of X would make two right answers)
        text += " " + T["distract"].format(c=c, y=Y if pat == "except_not" else q_item)
        q = (T["qnot"] if pat == "except_not" else T["q"]).format(y=q_item)
        items.append({"id": str(i), "lang": lang, "gold": a if pat == "except_not" else b, "pattern": pat,
                      "names": [a, b, c],
                      "messages": [{"role": "user", "content": f"{text}\n{q}\n" + ASK[lang]}]})
    return items


def score_negation_qa(item, reply):
    line = final_answer(reply["content"]) or ""
    named = {nm for nm in item["names"] if re.search(rf"\b{nm}\b", line)}
    ok = named == {item["gold"]}
    return {"correct": ok, "score": float(ok), "pred": line[:30], "gold": item["gold"],
            "detail": {f"ok_{item['pattern']}": float(ok)}}


# ---------------------------------------------------------------- citation_check
CI_COS = ["Veltra", "Ombrix", "Quorana", "Sylvent", "Taldor", "Brivex"]
CI_FACT = {"en": [("was founded in {v}", lambda r: r.randint(1950, 2015)), ("has {v} employees", lambda r: r.randint(40, 9000)),
                  ("is based in {v}", lambda r: r.choice(["Graz", "Lyon", "Turin", "Porto", "Gdansk", "Malmö"])),
                  ("makes {v}", lambda r: r.choice(["bicycles", "solar panels", "pumps", "toys", "software", "boats"]))],
           "de": [("wurde {v} gegründet", lambda r: r.randint(1950, 2015)), ("hat {v} Beschäftigte", lambda r: r.randint(40, 9000)),
                  ("sitzt in {v}", lambda r: r.choice(["Graz", "Lyon", "Turin", "Porto", "Danzig", "Malmö"])),
                  ("stellt {v} her", lambda r: r.choice(["Fahrräder", "Solarmodule", "Pumpen", "Spielzeug", "Software",
                                                         "Boote"]))]}
CI_PARA = {"en": ["{co}'s founding year is {v}.", "{co} employs {v} people.", "{co}'s headquarters are in {v}.",
                  "{co} is a manufacturer of {v}."],
           "de": ["{co} gibt es seit {v}.", "{co} beschäftigt {v} Menschen.", "Der Firmensitz von {co} ist {v}.",
                  "{co} produziert {v}."]}  # the claims say the same as the sources in other words
CI_TEXT = {"en": "Sources:\n{src}\n\nAnswer:\n{ans}\n\nExactly one claim cites a source that does not support it. "
                 "Which claim number is it?",
           "de": "Quellen:\n{src}\n\nAntwort:\n{ans}\n\nGenau eine Aussage nennt eine Quelle, die sie nicht stützt. "
                 "Welche Aussagennummer ist es?"}


def load_citation_check(lang, n, seed):
    items = []
    for i in range(n or 150):
        r = rng_of("citation_check", i)
        cos = r.sample(CI_COS, 2)  # two companies, two facts each: the company name alone never gives it away
        facts, owner = [], []  # (source sentence, paraphrased claim); owner[k] = company index of source k
        for ci, co in enumerate(cos):
            for t_ in r.sample(range(4), 2):
                tmpl, gen = CI_FACT[lang][t_]
                v = gen(r)
                facts.append((f"{co} {tmpl.format(v=v)}.", CI_PARA[lang][t_].format(co=co, v=v)))
                owner.append(ci)
        order = r.sample(range(4), 4)
        facts, owner = [facts[k] for k in order], [owner[k] for k in order]
        wrong = r.randrange(3)
        while True:  # the wrong claim's sibling source must not be cited by another claim (no shared number)
            claims = r.sample(range(4), 3)
            sib = next(x for x in range(4) if x != claims[wrong] and owner[x] == owner[claims[wrong]])
            if sib not in claims:
                break
        lines = []
        for j, k in enumerate(claims):
            # the wrong citation points to the OTHER source of the same company
            cite = k if j != wrong else next(x for x in range(4) if x != k and owner[x] == owner[k])
            lines.append(f"{j + 1}. {facts[k][1]} [{cite + 1}]")
        src = "\n".join(f"[{k + 1}] {s_}" for k, (s_, _) in enumerate(facts))
        items.append({"id": str(i), "lang": lang, "gold": wrong + 1, "wrong_pos": wrong + 1,
                      "messages": [{"role": "user", "content": CI_TEXT[lang].format(src=src, ans="\n".join(lines))
                                    + "\n" + ASK[lang]}]})
    return items


def score_citation_check(item, reply):
    m = re.search(r"\d+", final_answer(reply["content"]) or "")  # "Behauptung 2", "It is claim 2"
    v = int(m[0]) if m else None
    return {"correct": v == item["gold"], "score": float(v == item["gold"]), "pred": v, "gold": item["gold"],
            "detail": {"wrong_pos": item["wrong_pos"]}}


def task(load, score, quick, full):
    langs = ("en", "de")
    return {"langs": langs, "load": load, "score": score, "think": True, "max_tokens": 12288,
            "counts": {"quick": dict.fromkeys(langs, quick), "full": dict.fromkeys(langs, full)},
            "lang_check": False}


TASKS = {
    "rule_induction": task(load_rule_induction, score_rule_induction, 24, 300),
    "invented_ops": task(load_invented_ops, score_invented_ops, 24, 300),
    "mini_language": task(load_mini_language, score_mini_language, 24, 200),
    "theory_of_mind": task(load_tom, score_tom, 24, 300),
    "temporal": task(load_temporal, score_temporal, 24, 300),
    "state_tracking": task(load_state, score_state, 24, 270),
    "math_variants": task(load_math_variants, score_math_variants, 24, 360),
    "code_trace": task(load_code_trace, score_code_trace, 24, 240),
    "spatial": task(load_spatial, score_spatial, 24, 240),
    "math_missing": task(load_math_missing, score_math_missing, 24, 200),
    "anchoring": task(load_anchoring, score_anchoring, 24, 150),
    "char_ops": task(load_char_ops, score_char_ops, 24, 200),
    "false_authority": task(load_false_authority, score_false_authority, 24, 200),
    "near_miss_qa": task(load_near_miss_qa, score_near_miss_qa, 24, 200),
    "format_bait": {**task(load_format_bait, score_format_bait, 24, 150), "think": False, "max_tokens": 256, "lang_check": False},
    "revoked_rule": {**task(load_revoked_rule, score_revoked_rule, 24, 120), "think": False, "max_tokens": 256},
    "lookalike": task(load_lookalike, score_lookalike, 24, 200),
    "puzzle_mutation": task(load_puzzle_mutation, score_puzzle_mutation, 16, 16),
    "meeting_actions": {**task(load_meeting_actions, score_meeting_actions, 24, 150), "think": False,
                        "max_tokens": 512},
    "citation_check": task(load_citation_check, score_citation_check, 24, 150),
    "negation_qa": task(load_negation_qa, score_negation_qa, 24, 150),
    "data2text": {**task(load_data2text, score_data2text, 30, 120), "think": False, "max_tokens": 512},
    "self_correct": task(load_self_correct, score_self_correct, 24, 150),
    "linalg": task(load_linalg, score_linalg, 24, 150),
    "jugs": task(load_jugs, score_jugs, 24, 150),
    "gears": task(load_gears, score_gears, 24, 150),
    "copycat": task(load_copycat, score_copycat, 24, 150),
    "number_bases": task(load_number_bases, score_number_bases, 24, 150),
    "knights_knaves": task(load_knights, score_knights, 24, 150),
    "rulebook": task(load_rulebook, score_rulebook, 24, 150),
    "game_theory": task(load_game_theory, score_game_theory, 24, 150),
    "kinship": task(load_kinship, score_kinship, 24, 150),
    "calib_math": task(load_calib_math, score_calib_math, 24, 120),
    "timezones": task(load_timezones, score_timezones, 24, 150),
    "code_bug": task(load_code_bug, score_code_bug, 48, 160),
    "mini_sudoku": task(load_mini_sudoku, score_mini_sudoku, 24, 150),
    "event_order": task(load_event_order, score_event_order, 24, 150),
    "probability": task(load_probability, score_probability, 24, 150),
    "graph_path": task(load_graph_path, score_graph_path, 24, 200),
    "extract": {**task(load_extract, score_extract, 60, 200), "think": False, "max_tokens": 512},
    "syllogism": task(load_syllogism, score_syllogism, 24, 180),
    "schedule": task(load_schedule, score_schedule, 24, 200),
    "error_spot": task(load_error_spot, score_error_spot, 24, 200),
    "seq_induction": task(load_seq_induction, score_seq_induction, 24, 180),
    "table_qa": task(load_table_qa, score_table_qa, 24, 200),
    "sci_calc": task(load_sci_calc, score_sci_calc, 24, 200),
    "logic_grid": task(load_logic_grid, score_logic_grid, 24, 200),
    "if_gen": {**task(load_if_gen, score_if_gen, 100, 200), "think": False, "max_tokens": 1024},
    "dialog_state": {**task(load_dialog, score_dialog, 24, 200), "think": False, "max_tokens": 1024},
}
