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
    # "not 5, but 10" / "father, not mother": drop the negated alternative, then require gold and not the other
    line = re.sub(r"\b(not|nicht|kein\w*)\s+(?!change|affect|matter)((the|die|der|das|his|seine?)\s+)?[\w$€.,/%]+", " ",
                  line).strip(" ,;:-—")
    ok = bool(re.search(item["pattern"], line)) and not re.search(item["other"], line)
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
    line = re.split(r"\(|\bvs\b|\bstatt\b|\binstead\b|\bnicht\b|\bnot\b", line)[0]
    line = line.translate(str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789"))
    tail = re.split(r"[=≈~]", line)[-1]
    e = re.search(r"(\d+(?:[.,]\d+)?)\s*e\s*\+?(\d+)", tail, re.I)
    if e:
        return float(e.group(1).replace(",", ".")) * 10 ** int(e.group(2))
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
        if item["kind"] == "nth":  # "Der Buchstabe ist L": the last standalone letter
            single = re.findall(r"(?<![a-zäöüß])[a-zäöüß](?![a-zäöüß])", line)
            ok = letters == item["gold"] or bool(single) and single[-1] == item["gold"]
        else:  # "The reversed word is x-y-z": the letters end with it
            ok = letters.endswith(item["gold"])
    return {"correct": ok, "score": float(ok), "pred": line[:40], "gold": item["gold"],
            "detail": {f"ok_{item['kind']}": float(ok)}}


# ---------------------------------------------------------------- false_authority (trap: a wrong premise from an authority)
# An intermediate result is stated by an authority; in trap items it is off by a small amount, in control items it
# is right. The task builds on it, so taking the claim gives a predictable wrong answer (took_claim).
FA_WHO = {"en": ["Professor Weber showed in her lecture", "The official handbook states", "Our senior engineer confirmed"],
          "de": ["Professorin Weber hat in ihrer Vorlesung gezeigt", "Im offiziellen Handbuch steht", "Unser leitender Ingenieur hat bestätigt"]}
FA_TEXT = {
    "mul": {"en": "{who}: {a} × {b} = {c}. What is {a} × {b} + {d}?",
            "de": "{who}: {a} × {b} = {c}. Was ergibt {a} × {b} + {d}?"},
    "pct": {"en": "{who}: {p}% of {a} is {c}. What is {p}% of {a}, minus {d}?",
            "de": "{who}: {p} % von {a} sind {c}. Was ergibt {p} % von {a} minus {d}?"},
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
    v = answer_value(re.sub(r"\([^)]*\)", "", final_answer(reply["content"]) or ""), item["lang"])
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


NM_ABSTAIN = re.compile(ABSTAIN_RE.pattern + r"|not (stated|given|provided|mentioned|listed|available)|unknown|"
                        r"cannot be answered|nicht (angegeben|genannt|enthalten|aufgeführt|verfügbar)|keine angabe|"
                        r"unbekannt", re.I)


def score_near_miss_qa(item, reply):
    line = re.sub(r"\b20[12]\d\b", "", final_answer(reply["content"]) or "")  # naming the year is not a guess
    m = NM_ABSTAIN.search(line)
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


# ---------------------------------------------------------------- rephrase_consistency (trap: same fact, three phrasings)
# A chain of "older than" facts over five people (shuffled); one pair is asked three ways: "Is X older than Y?",
# "Is Y younger than X?", "Who is older, X or Y?". Ids "<group>-<form>"; quality.summarize reports group_agree
# (all three verdicts equal) and group_all (all three right) for tasks with "grouped".
RC_TEXT = {
    "en": {"fact": "{a} is older than {b}.", "older": "Is {x} older than {y}? Answer yes or no.",
           "younger": "Is {y} younger than {x}? Answer yes or no.", "who": "Who is older, {x} or {y}? Answer with the name."},
    "de": {"fact": "{a} ist älter als {b}.", "older": "Ist {x} älter als {y}? Antworte mit ja oder nein.",
           "younger": "Ist {y} jünger als {x}? Antworte mit ja oder nein.", "who": "Wer ist älter, {x} oder {y}? Antworte mit dem Namen."},
}
RC_FORMS = ("older", "younger", "who")


def load_rephrase_consistency(lang, n, seed):
    items = []
    for i in range(n or 150):
        g, form = divmod(i, 3)
        r = rng_of("rephrase_consistency", g)
        people = r.sample(NAMES, 5)  # oldest first
        facts = [RC_TEXT[lang]["fact"].format(a=people[k], b=people[k + 1]) for k in range(4)]
        r.shuffle(facts)
        i1, i2 = sorted(r.sample(range(5), 2))
        while i2 - i1 < 2:  # at least one inference step
            i1, i2 = sorted(r.sample(range(5), 2))
        x, y = (people[i1], people[i2]) if r.random() < 0.5 else (people[i2], people[i1])
        x_older = people.index(x) < people.index(y)
        f = RC_FORMS[form]
        gold = (("yes" if lang == "en" else "ja") if x_older else ("no" if lang == "en" else "nein")) if f != "who" \
            else (x if x_older else y)
        q = " ".join(facts) + "\n" + RC_TEXT[lang][f].format(x=x, y=y)
        items.append({"id": f"{g}-{form}", "lang": lang, "form": f, "gold": gold, "x": x, "y": y,
                      "messages": [{"role": "user", "content": q + "\n" + ASK[lang]}]})
    return items


def score_rephrase_consistency(item, reply):
    line = (final_answer(reply["content"]) or "").strip().lower()
    words = re.findall(r"[a-zäöüß]+", line)
    gold = item["gold"].lower()
    if item["form"] == "who":
        pair = {item["x"].lower(), item["y"].lower()}
        names = [w for w in words if w in pair]
        older = re.search(r"([a-zäöüß]+)\s+(is|ist)\s+(the\s+)?(older|älter)", line)
        pick = older.group(1) if older and older.group(1) in pair else (names[-1] if names else None)
        ok = pick == gold
    else:
        yn = [w for w in words if w in ("yes", "no", "ja", "nein")]
        ok = yn[:1] == [gold]
    return {"correct": ok, "score": float(ok), "pred": line[:30], "gold": item["gold"],
            "detail": {f"ok_{item['form']}": float(ok)}}


# ---------------------------------------------------------------- reversal (trap: a learned relation asked backwards)
# Six invented "A's mentor is B" facts in context; control asks forward (who is A's mentor), trap asks backward
# (whose mentor is B). Same fact, same context; ok_forward - ok_backward is the reversal gap.
RV2_TEXT = {"en": ("The mentor of {a} is {b}.", "Who is the mentor of {a}?", "Whose mentor is {b}?"),
            "de": ("Die Mentorin von {a} ist {b}.", "Wer ist die Mentorin von {a}?", "Wessen Mentorin ist {b}?")}


def load_reversal(lang, n, seed):
    items = []
    for i in range(n or 200):
        r = rng_of("reversal", i // 2)
        back = i % 2 == 1
        names = []
        while len(names) < 12:  # 3 syllables: fewer real words; no name a prefix of another
            w = ch_word(r, 3, 3).capitalize()
            if not any(w.startswith(o) or o.startswith(w) for o in names):
                names.append(w)
        pairs = list(zip(names[:6], names[6:]))  # every person appears once: no chains
        k = r.randrange(6)
        fact, fwd, bwd = RV2_TEXT[lang]
        a, b = pairs[k]
        q = "\n".join(fact.format(a=x, b=y) for x, y in pairs) + "\n" + (bwd.format(b=b) if back else fwd.format(a=a))
        items.append({"id": str(i), "lang": lang, "backward": back, "gold": a if back else b, "names": names,
                      "asked": b if back else a,
                      "messages": [{"role": "user", "content": q + "\n" + ASK[lang]}]})
    return items


def score_reversal(item, reply):
    line = final_answer(reply["content"]) or ""
    said = {w for w in re.findall(r"[A-Za-zäöüß]+", line) if w in item["names"]} - {item["asked"]}
    ok = said == {item["gold"]}
    return {"correct": ok, "score": float(ok), "pred": line[:30], "gold": item["gold"],
            "detail": {"ok_backward" if item["backward"] else "ok_forward": float(ok)}}


# ---------------------------------------------------------------- off_by_one (trap: fence posts and inclusive ranges)
OB_KINDS = ("pages", "between", "fence", "nights", "loop")
OB_TEXT = {
    "en": {"pages": "Read pages {a} to {b} of the book, both included. How many pages is that?",
           "between": "How many whole numbers lie strictly between {a} and {b}?",
           "fence": "A straight fence is {L} m long with a post every {d} m, including one at each end. How many posts?",
           "nights": "A hotel guest checks in on May {a} and checks out on May {b}. How many nights?",
           "loop": "In Python, how many times does the body of `for i in range({a}, {b}, {d}):` run?"},
    "de": {"pages": "Lies die Seiten {a} bis {b} des Buches, beide eingeschlossen. Wie viele Seiten sind das?",
           "between": "Wie viele ganze Zahlen liegen echt zwischen {a} und {b}?",
           "fence": "Ein gerader Zaun ist {L} m lang, alle {d} m steht ein Pfosten, auch an beiden Enden. Wie viele Pfosten?",
           "nights": "Ein Hotelgast checkt am {a}. Mai ein und am {b}. Mai aus. Wie viele Nächte sind das?",
           "loop": "Wie oft läuft in Python der Rumpf von `for i in range({a}, {b}, {d}):`?"},
}


def load_off_by_one(lang, n, seed):
    items = []
    for i in range(n or 200):
        r = rng_of("off_by_one", i)
        kind = OB_KINDS[i % len(OB_KINDS)]
        a = r.randint(4, 120)
        if kind == "pages":
            b = a + r.randint(5, 60)
            gold, f = b - a + 1, {"a": a, "b": b}
        elif kind == "between":
            b = a + r.randint(5, 60)
            gold, f = b - a - 1, {"a": a, "b": b}
        elif kind == "fence":
            d = r.choice([2, 3, 4, 5])
            L = d * r.randint(6, 30)
            gold, f = L // d + 1, {"L": L, "d": d}
        elif kind == "nights":
            a = r.randint(4, 15)
            b = a + r.randint(3, 12)
            gold, f = b - a, {"a": a, "b": b}
        else:
            d = r.choice([2, 3, 5])
            b = a + r.randint(10, 60)
            gold, f = len(range(a, b, d)), {"a": a, "b": b, "d": d}
        items.append({"id": str(i), "lang": lang, "kind": kind, "gold": gold,
                      "messages": [{"role": "user", "content": OB_TEXT[lang][kind].format(**f) + "\n" + ASK[lang]}]})
    return items


def score_off_by_one(item, reply):
    v = answer_value(final_answer(reply["content"]), item["lang"])
    ok = v == item["gold"]
    d = {f"ok_{item['kind']}": float(ok), "off_by_one": float(v is not None and abs(v - item["gold"]) == 1)}
    return {"correct": ok, "score": float(ok), "pred": v, "gold": item["gold"], "detail": d}


# ---------------------------------------------------------------- mixed_units (trap: one value in another unit)
# Three lengths / durations / masses to add, asked in the main unit; trap items state one of them in the smaller
# unit, control items state all in the main unit (same quantities). naive = the plain sum of the numbers.
MU_KINDS = {
    "len": {"en": ("board", "m", "cm", 100, "How long are the three boards together, in metres?"),
            "de": ("Brett", "m", "cm", 100, "Wie lang sind die drei Bretter zusammen, in Metern?")},
    "time": {"en": ("task", "h", "min", 60, "How long do the three tasks take together, in hours?"),
             "de": ("Aufgabe", "h", "min", 60, "Wie lange dauern die drei Aufgaben zusammen, in Stunden?")},
    "mass": {"en": ("parcel", "kg", "g", 1000, "How heavy are the three parcels together, in kilograms?"),
             "de": ("Paket", "kg", "g", 1000, "Wie schwer sind die drei Pakete zusammen, in Kilogramm?")},
}
MU_ITEM = "{noun} {k}: {v} {u}"


def load_mixed_units(lang, n, seed):
    items = []
    kinds = list(MU_KINDS)
    for i in range(n or 200):
        r = rng_of("mixed_units", i // 2)
        trap = i % 2 == 1
        kind = kinds[(i // 2) % 3]
        noun, big, small, f, q = MU_KINDS[kind][lang]
        vals = [r.randint(2, 9) * f // 4 for _ in range(3)]  # quarters of the big unit
        k = r.randrange(3)
        lines = []
        for j, v in enumerate(vals):
            if trap and j == k:
                lines.append(MU_ITEM.format(noun=noun.capitalize(), k=j + 1, v=v, u=small))
            else:
                lines.append(MU_ITEM.format(noun=noun.capitalize(), k=j + 1, v=fmt_num(v / f, lang), u=big))
        gold = sum(vals) / f
        naive = sum(vals[j] / f for j in range(3) if j != k) + vals[k]
        items.append({"id": str(i), "lang": lang, "trap": trap, "gold": round(gold, 4), "naive": round(naive, 4),
                      "messages": [{"role": "user", "content": "\n".join(lines) + "\n" + q + "\n" + ASK[lang]}]})
    return items


def fmt_num(x, lang):  # 1.5 -> '1,5' in German
    t = f"{x:g}"
    return t.replace(".", ",") if lang == "de" else t


def score_mixed_units(item, reply):
    line = final_answer(reply["content"]) or ""
    hm = re.search(r"(\d+)\s*(h|std\.?|stunden?|hours?)\s*(und|and)?\s*(\d+)\s*min", line, re.I)
    v = int(hm.group(1)) + int(hm.group(4)) / 60 if hm else answer_value(line, item["lang"])
    ok = v is not None and abs(v - item["gold"]) < 1e-6
    d = {"ok_trap" if item["trap"] else "ok_control": float(ok)}
    if item["trap"]:
        d["naive_sum"] = float(v is not None and abs(v - item["naive"]) < 1e-6)
    return {"correct": ok, "score": float(ok), "pred": v, "gold": item["gold"], "detail": d}


# ---------------------------------------------------------------- double_negation (trap: stacked negations)
# Four people, each with a statement about team membership wrapped in 0-3 negations; one person is asked about.
# Sub-metric ok_neg<k> by the number of negations on the asked statement.
# (main clause, subordinate clause) forms, so German "dass" clauses put the verb last
DN_TEXT = {
    "en": {"base": [("{x} is on the team", "{x} is on the team"), ("{x} is not on the team", "{x} is not on the team")],
           "wrap": [("It is not true that {c}", "it is not true that {c}"), ("It is false that {c}", "it is false that {c}")],
           "q": ["Is {x} on the team? Answer yes or no.", "Is it true that {x} is not on the team? Answer yes or no."]},
    "de": {"base": [("{x} ist im Team", "{x} im Team ist"), ("{x} ist nicht im Team", "{x} nicht im Team ist")],
           "wrap": [("Es stimmt nicht, dass {c}", "es nicht stimmt, dass {c}"), ("Es ist falsch, dass {c}", "es falsch ist, dass {c}")],
           "q": ["Ist {x} im Team? Antworte mit ja oder nein.", "Stimmt es, dass {x} nicht im Team ist? Antworte mit ja oder nein."]},
}


def dn_statement(r, x, negs, lang):
    """A statement about x with `negs` negations (0-3: an inner "not" plus up to two wrappers). Its truth for
    "x is on the team" is negs even; the question form decides the yes/no, so parity alone is no shortcut."""
    T = DN_TEXT[lang]
    inner = 1 if negs == 3 else 0 if negs == 0 else r.randint(0, 1)
    main, sub = (f.format(x=x) for f in T["base"][inner])
    for _ in range(negs - inner):
        w = r.choice(T["wrap"])
        main, sub = w[0].format(c=sub), w[1].format(c=sub)
    return main + ".", negs % 2 == 0


def load_double_negation(lang, n, seed):
    items = []
    for i in range(n or 200):
        r = rng_of("double_negation", i)
        people = r.sample(NAMES, 4)
        target_negs = i % 4
        stmts = [dn_statement(r, x, target_negs if j == 0 else r.randrange(4), lang)[0] for j, x in enumerate(people)]
        on = target_negs % 2 == 0
        r.shuffle(stmts)
        q_neg = r.random() < 0.5
        yes, no = ("yes", "no") if lang == "en" else ("ja", "nein")
        x = people[0]
        items.append({"id": str(i), "lang": lang, "negs": target_negs, "gold": yes if on != q_neg else no,
                      "messages": [{"role": "user", "content": " ".join(stmts) + "\n" + DN_TEXT[lang]["q"][q_neg].format(x=x)
                                    + "\n" + ASK[lang]}]})
    return items


def score_double_negation(item, reply):
    yn = [w for w in re.findall(r"[a-z]+", (final_answer(reply["content"]) or "").lower()) if w in ("yes", "no", "ja", "nein")]
    ok = yn[:1] == [item["gold"]]
    return {"correct": ok, "score": float(ok), "pred": yn[:1], "gold": item["gold"],
            "detail": {f"ok_neg{item['negs']}": float(ok)}}


# ---------------------------------------------------------------- false_premise (trap: the question presupposes an event)
# Three short careers; the question asks in which year the target resigned. Control: the target did resign (gold =
# the year). Trap: the target is stated to still be there, another person resigned (gold = reject the premise).
FP_TEXT = {
    "en": {"join": "{x} joined the company in {a}.", "promo": "{x} became team lead in {b}.",
           "quit": "{x} resigned in {c}.", "stay": "{x} still works there today.",
           "q": "In which year did {x} resign?"},
    "de": {"join": "{x} kam {a} in die Firma.", "promo": "{x} übernahm {b} die Teamleitung.",
           "quit": "{x} kündigte {c}.", "stay": "{x} arbeitet bis heute dort.",
           "q": "In welchem Jahr kündigte {x}?"},
}
FP_REJECT = re.compile(r"\b(never|not|did ?n.t|has ?n.t|has not|no such|no year|false premise|still|to this day|nowhere|"
                       r"nie|niemals|nicht|gar nicht|kein\w*|noch|bis heute|weiterhin)\b|^\s*(-|n/?a)\s*$", re.I)
FP_YEAR = re.compile(r"\b(19|20)\d\d\b")


def load_false_premise(lang, n, seed):
    items = []
    for i in range(n or 200):
        r = rng_of("false_premise", i // 2)
        trap = i % 2 == 1
        people = r.sample(NAMES, 3)
        T = FP_TEXT[lang]
        blocks, quit_year = [], {}
        for j, x in enumerate(people):
            a = r.randint(2003, 2012)
            b = a + r.randint(2, 5)
            c = b + r.randint(1, 6)
            quits = (not trap, True, False)[j]  # the target quits only in control; person 1 always quits
            blocks.append(" ".join([T["join"].format(x=x, a=a), T["promo"].format(x=x, b=b),
                                    T["quit"].format(x=x, c=c) if quits else T["stay"].format(x=x)]))
            quit_year[x] = c if quits else None
        x = people[0]
        r.shuffle(blocks)  # drawn in both twins: same order
        items.append({"id": str(i), "lang": lang, "trap": trap, "gold": "-" if trap else quit_year[x],
                      "messages": [{"role": "user", "content": " ".join(blocks) + "\n" + T["q"].format(x=x)
                                    + "\n" + ASK[lang]}]})
    return items


def score_false_premise(item, reply):
    line = final_answer(reply["content"]) or ""
    year = FP_YEAR.search(line)
    if item["trap"]:  # right: a rejection, before the first year if one is named ("never; still there since 2010")
        rej = FP_REJECT.search(line)
        ok = bool(rej) and (not year or rej.start() < year.start())
        d = {"ok_trap": float(ok), "took_premise": float(bool(year) and not ok)}
    else:  # the first year stated is the answer ("2013, not 2011")
        ok = bool(year) and int(year.group()) == item["gold"]
        d = {"ok_control": float(ok)}
    return {"correct": ok, "score": float(ok), "pred": line[:60], "gold": item["gold"], "detail": d}


# ---------------------------------------------------------------- option_order (trap: position bias in multiple choice)
# One question, four options, asked four times with the right option at A, B, C and D (ids "<group>-<pos>",
# "grouped": group_agree = the verdict survives reordering). Sub-metric ok_pos<A-D> exposes a favourite letter.
OO_TEXT = {"en": {"div": "Which of these numbers is divisible by {k}?", "ans": "The answer is a single letter."},
           "de": {"div": "Welche dieser Zahlen ist durch {k} teilbar?", "ans": "Die Antwort ist ein einzelner Buchstabe."}}


def load_option_order(lang, n, seed):
    items = []
    for i in range(n or 200):
        g, pos = divmod(i, 4)
        r = rng_of("option_order", g)
        k = r.choice([7, 11, 13, 17])
        right = k * r.randint(8, 70)
        wrong = []
        while len(wrong) < 3:  # near misses: right +- a little, never divisible
            w = right + r.choice([-1, 1]) * r.randint(1, k - 1) + k * r.randint(-3, 3)
            if w % k and w > 0 and w not in wrong:
                wrong.append(w)
        opts = wrong[:pos] + [right] + wrong[pos:]
        body = "\n".join(f"{'ABCD'[j]}) {v}" for j, v in enumerate(opts))
        q = OO_TEXT[lang]["div"].format(k=k) + "\n" + body + "\n" + OO_TEXT[lang]["ans"]
        items.append({"id": f"{g}-{pos}", "lang": lang, "gold": "ABCD"[pos],
                      "messages": [{"role": "user", "content": q + "\n" + ASK[lang]}]})
    return items


def score_option_order(item, reply):
    line = (final_answer(reply["content"]) or "").strip()
    m = (re.fullmatch(r"\(?([abcdABCD])\)?\.?", line) or re.match(r"\(?([ABCD])\)", line)
         or re.search(r"(?:option|antwort|answer)\s+\(?([ABCD])\b", line, re.I) or re.search(r"\b([ABCD])\)", line))
    if not m:
        caps = re.findall(r"\b([ABCD])\b", line)
        m = re.match(r"(.*)", caps[-1]) if caps else None
    pred = m.group(1).upper() if m else None
    ok = pred == item["gold"]
    return {"correct": ok, "score": float(ok), "pred": pred, "gold": item["gold"],
            "detail": {f"ok_pos{item['gold']}": float(ok), **({f"chose_{pred}": 1.0} if pred else {})}}


# ---------------------------------------------------------------- source_trust (trap: a wrong majority vs the trusted source)
# A rule says which source counts (the calibrated lab measurement). Four sources report a device's weight; in trap
# items three blogs agree on a wrong value, in control items they agree with the lab. took_majority = the blogs' value.
ST_TEXT = {
    "en": {"rule": "Rule: if sources disagree, only the calibrated lab measurement counts.",
           "lab": "Calibrated lab measurement: {v} g.", "blog": "{who}: {v} g.",
           "who": ["Tech blog", "Online shop listing", "Forum post", "Review video"],
           "q": "How much does the device weigh, in grams?"},
    "de": {"rule": "Regel: Wenn Quellen sich widersprechen, zählt nur die kalibrierte Labormessung.",
           "lab": "Kalibrierte Labormessung: {v} g.", "blog": "{who}: {v} g.",
           "who": ["Technikblog", "Angebot im Onlineshop", "Forumsbeitrag", "Testvideo"],
           "q": "Wie viel wiegt das Gerät in Gramm?"},
}


def load_source_trust(lang, n, seed):
    items = []
    for i in range(n or 200):
        r = rng_of("source_trust", i // 2)
        trap = i % 2 == 1
        T = ST_TEXT[lang]
        true = r.randint(120, 950)
        wrong = true + r.choice([-1, 1]) * r.randint(15, 80)
        who = r.sample(T["who"], 3)
        # trap: the three others agree on a wrong value; control: two agree with the lab, one is off
        srcs = [T["blog"].format(who=w, v=wrong if trap or k == 2 else true) for k, w in enumerate(who)]
        srcs.insert(r.randrange(4), T["lab"].format(v=true))
        q = T["rule"] + "\n" + "\n".join(srcs) + "\n" + T["q"]
        items.append({"id": str(i), "lang": lang, "trap": trap, "gold": true, "majority": wrong,
                      "messages": [{"role": "user", "content": q + "\n" + ASK[lang]}]})
    return items


def score_source_trust(item, reply):
    line = final_answer(reply["content"]) or ""
    g = re.findall(r"(\d[\d.,]*)\s*(?:g\b|gramm|grams?)", line, re.I)
    v = parse_number(g[-1], item["lang"]) if g else answer_value(line, item["lang"])
    ok = v == item["gold"]
    d = {"ok_trap" if item["trap"] else "ok_control": float(ok)}
    if item["trap"]:
        d["took_majority"] = float(v == item["majority"])
    return {"correct": ok, "score": float(ok), "pred": v, "gold": item["gold"], "detail": d}


# ---------------------------------------------------------------- switched_question (trap: the preamble sets up another question)
# A shopping list with prices. Trap: a long preamble builds up to the total ("we need the total for the budget,
# add everything up carefully ...") and the last sentence asks something else (how many items cost more than X).
# Control: only the last question. took_total = answered the preamble's question.
SQ_ITEMS = {"en": ["paint", "brushes", "tape", "ladder", "gloves", "sponges", "rollers", "buckets"],
            "de": ["Farbe", "Pinsel", "Klebeband", "Leiter", "Handschuhe", "Schwämme", "Rollen", "Eimer"]}
SQ_TEXT = {
    "en": {"pre": "We are planning the renovation budget and the treasurer needs the total cost of this list, so "
                  "please be careful with the sum, last time an item was forgotten and the total was wrong. The total "
                  "goes straight into the budget sheet.",
           "q": "But before that, just tell me: how many items cost more than {x} euros?"},
    "de": {"pre": "Wir planen das Renovierungsbudget, und der Kassenwart braucht die Gesamtkosten dieser Liste. Bitte "
                  "rechne die Summe sorgfältig, letztes Mal wurde ein Posten vergessen und die Summe stimmte nicht. Die "
                  "Summe kommt direkt in die Budgettabelle.",
           "q": "Aber vorher sag mir nur: Wie viele Posten kosten mehr als {x} Euro?"},
}


def load_switched_question(lang, n, seed):
    items = []
    for i in range(n or 200):
        r = rng_of("switched_question", i // 2)
        trap = i % 2 == 1
        names = r.sample(SQ_ITEMS[lang], 6)
        prices = [r.randint(3, 90) for _ in names]
        x = r.choice([20, 30, 40, 50])
        prices = [p + 1 if p == x else p for p in prices]  # no price exactly at the threshold
        lst = "\n".join(f"- {a}: {p} €" for a, p in zip(names, prices))
        q = (SQ_TEXT[lang]["pre"] + "\n" if trap else "") + lst + "\n" + SQ_TEXT[lang]["q"].format(x=x)
        if not trap:
            q = q.replace("But before that, just tell me: h", "H").replace("Aber vorher sag mir nur: W", "W")
        items.append({"id": str(i), "lang": lang, "trap": trap, "gold": sum(p > x for p in prices), "total": sum(prices),
                      "messages": [{"role": "user", "content": q + "\n" + ASK[lang]}]})
    return items


SQ_WORDS = {w: k for k, ws in enumerate([("zero", "null", "keine", "keiner", "none"), ("one", "eins", "einer", "ein"),
                                         ("two", "zwei"), ("three", "drei"), ("four", "vier"), ("five", "fünf"),
                                         ("six", "sechs")]) for w in ws}


def score_switched_question(item, reply):
    tail = (final_answer(reply["content"]) or "").rsplit(":", 1)[-1]
    word = re.match(r"\s*\**([a-zäöü]+)", tail.lower())
    v = SQ_WORDS[word.group(1)] if word and word.group(1) in SQ_WORDS else answer_value(tail, item["lang"])
    ok = v == item["gold"]
    d = {"ok_trap" if item["trap"] else "ok_control": float(ok)}
    if item["trap"]:
        d["took_total"] = float(v == item["total"])
    return {"correct": ok, "score": float(ok), "pred": v, "gold": item["gold"], "detail": d}


# ---------------------------------------------------------------- causal_graph (interventions on an invented causal net)
# Six invented quantities, 6-8 direct effects (a random DAG). Questions: does X influence Y at all (a directed path),
# and does X still influence Y when Z is held fixed (a path avoiding Z)? Yes/no, balanced; correlation-only links
# ("X and Y rise together because both depend on W") are listed too and never count as influence.
CG_TEXT = {
    "en": {"edge": "The {a} directly affects the {b}.",
           "conf": "The {a} and the {b} tend to rise together, only because both depend on the {w}.",
           "any": "Can changing the {x} change the {y}? Answer yes or no.",
           "fix": "Suppose the {z} is held fixed by an experimenter. Can changing the {x} still change the {y}? Answer yes or no."},
    "de": {"edge": "Der {a} wirkt direkt auf den {b}.",
           "conf": "Der {a} und der {b} steigen oft gemeinsam, nur weil beide vom {w} abhängen.",
           "any": "Kann eine Änderung des {x} den {y} verändern? Antworte mit ja oder nein.",
           "fix": "Angenommen, der {z} wird von einer Experimentatorin festgehalten. Kann eine Änderung des {x} dann noch "
                  "den {y} verändern? Antworte mit ja oder nein."},
}


def cg_reach(edges, x, y, avoid=None):
    seen, todo = {x}, [x]
    while todo:
        u = todo.pop()
        for a, b in edges:
            if a == u and b != avoid and b not in seen:
                seen.add(b)
                todo.append(b)
    return y in seen


def load_causal_graph(lang, n, seed):
    items, i = [], -1
    while len(items) < (n or 200):
        i += 1
        r = rng_of("causal_graph", i)
        names = []
        while len(names) < 6:
            w = ch_word(r, 2, 3).capitalize() + ("-Index" if lang == "de" else " index")  # "der X-Index" / "the X index"
            if w not in names:
                names.append(w)
        edges = set()
        while len(edges) < r.randint(6, 8):
            a, b = sorted(r.sample(range(6), 2))  # a < b: acyclic
            edges.add((a, b))
        edges = sorted(edges)
        k = len(items)  # kind and answer follow the items made, so skipped draws keep the balance
        kind = ("any", "fix")[k % 2]
        want = (k // 2) % 2 == 0  # balanced yes / no
        used = {v for e in edges for v in e}
        # never a direct edge X -> Y: "yes if there is a direct link" must not be a shortcut
        pairs = [(x, y) for x in used for y in used if x != y and (x, y) not in edges]
        r.shuffle(pairs)
        pick = None
        for x, y in pairs:
            if kind == "any" and cg_reach(edges, x, y) == want:
                pick = (x, y, None)
                break
            if kind == "fix":
                zs = [z for z in range(6) if z not in (x, y) and cg_reach(edges, x, z) and cg_reach(edges, z, y)]
                if zs and cg_reach(edges, x, y):
                    z = r.choice(zs)
                    if cg_reach(edges, x, y, avoid=z) == want:
                        pick = (x, y, z)
                        break
        if pick is None:
            continue
        x, y, z = pick
        assert (x, y) not in edges
        T = CG_TEXT[lang]
        lines = [T["edge"].format(a=names[a], b=names[b]) for a, b in edges]
        # a confounded pair without a causal path between them
        conf = [(a, b, w) for w in range(6) for a in range(6) for b in range(6) if a < b and (w, a) in edges
                and (w, b) in edges and not cg_reach(edges, a, b) and not cg_reach(edges, b, a)]
        if conf:
            a, b, w = r.choice(conf)
            lines.append(T["conf"].format(a=names[a], b=names[b], w=names[w]))
        r.shuffle(lines)
        q = T[kind].format(x=names[x], y=names[y], z=names[z] if z is not None else "")
        yes, no = ("yes", "no") if lang == "en" else ("ja", "nein")
        items.append({"id": str(i), "lang": lang, "kind": kind, "gold": yes if want else no,
                      "messages": [{"role": "user", "content": "\n".join(lines) + "\n" + q + "\n" + ASK[lang]}]})
    return items


def score_causal_graph(item, reply):
    yn = [w for w in re.findall(r"[a-z]+", (final_answer(reply["content"]) or "").lower()) if w in ("yes", "no", "ja", "nein")]
    ok = yn[:1] == [item["gold"]]
    return {"correct": ok, "score": float(ok), "pred": yn[:1], "gold": item["gold"], "detail": {f"ok_{item['kind']}": float(ok)}}


# ---------------------------------------------------------------- ambiguous_ref (notice when a reference has two matches)
# A room with 6 objects (colour, kind, place). The question names an object; in half of the items the description
# fits two objects (gold: "ambiguous"), otherwise exactly one (gold: its place). The near-twin makes guessing tempting.
AR_KIND = {"en": ["cup", "book", "lamp", "bag", "plant"], "de": ["Tasse", "Buch", "Lampe", "Tasche", "Pflanze"]}
AR_COL = {"en": ["red", "blue", "green", "black"], "de": ["rot", "blau", "grün", "schwarz"]}
AR_PLACE = {"en": ["on the shelf", "on the desk", "by the window", "under the bed", "in the hallway", "on the sofa", "in the kitchen"],
            "de": ["im Regal", "auf dem Schreibtisch", "am Fenster", "unter dem Bett", "im Flur", "auf dem Sofa", "in der Küche"]}
AR_TEXT = {"en": ("There is a {c} {k} {p}.", "Where is the {d}? If the description fits more than one object, answer "
                  "\"ambiguous\"."),
           "de": ("{p_cap} befindet sich ein{e} {c}{end} {k}.", "Wo ist {art} {d}? Wenn die Beschreibung auf mehr als "
                  "einen Gegenstand passt, antworte \"mehrdeutig\".")}
AR_GENDER = {"Tasse": "f", "Buch": "n", "Lampe": "f", "Tasche": "f", "Pflanze": "f"}
AR_ABSTAIN = re.compile(r"ambiguous|unclear|more than one|\btwo\b|\bboth\b|which one|mehrdeutig|unklar|mehr als ein|"
                        r"nicht eindeutig|\bzwei\b|\bbeide|welche[nrs]?\b", re.I)
AR_NOT_ABSTAIN = re.compile(r"not ambiguous|nicht mehrdeutig|eindeutig", re.I)
# variants: 0 bare kind, two of it (ambiguous) / 1 colour + kind, two of it in other colours (unique) /
#           2 bare kind, only one of it (unique) / 3 colour + kind, two identical (ambiguous)
AR_VARIANTS = ((True, False, 2), (False, True, 2), (False, False, 1), (True, True, 2))


def ar_obj(lang, c, k, p):
    if lang == "en":
        return AR_TEXT["en"][0].format(c=AR_COL["en"][c], k=AR_KIND["en"][k], p=AR_PLACE["en"][p])
    kind = AR_KIND["de"][k]
    g = AR_GENDER[kind]
    place = AR_PLACE["de"][p]
    return AR_TEXT["de"][0].format(p_cap=place[0].upper() + place[1:], e="e" if g == "f" else "", c=AR_COL["de"][c],
                                   end="e" if g == "f" else "es", k=kind)


def load_ambiguous_ref(lang, n, seed):
    items = []
    for i in range(n or 200):
        r = rng_of("ambiguous_ref", i // 4)
        amb, coloured, count = AR_VARIANTS[i % 4]
        places = r.sample(range(7), 6)
        k, c, c2 = r.randrange(5), *r.sample(range(4), 2)
        objs = [(c, k), (c if amb and coloured else c2, k)][:count]
        while len(objs) < 6:
            o = (r.randrange(4), r.randrange(5))
            if o[1] != k and o not in objs:
                objs.append(o)
        lines = [ar_obj(lang, oc, ok, pl) for (oc, ok), pl in zip(objs, places)]
        r.shuffle(lines)
        kind = AR_KIND[lang][k]
        if lang == "en":
            d = f"{AR_COL['en'][c]} {kind}" if coloured else kind
            q = AR_TEXT["en"][1].format(d=d)
        else:
            d = f"{AR_COL['de'][c]}e {kind}" if coloured else kind  # "die rote Tasse", "das rote Buch"
            q = AR_TEXT["de"][1].format(art="die" if AR_GENDER[kind] == "f" else "das", d=d)
        gold = None if amb else AR_PLACE[lang][places[0]]
        items.append({"id": str(i), "lang": lang, "ambiguous": amb, "variant": i % 4, "gold": gold,
                      "messages": [{"role": "user", "content": " ".join(lines) + "\n" + q + "\n" + ASK[lang]}]})
    return items


def score_ambiguous_ref(item, reply):
    line = (final_answer(reply["content"]) or "").lower()
    abst = bool(AR_ABSTAIN.search(AR_NOT_ABSTAIN.sub(" ", line)))
    if item["ambiguous"]:
        ok = abst
        d = {"ok_ambiguous": float(ok), f"ok_v{item['variant']}": float(ok)}
    else:
        noun = item["gold"].lower().split()[-1]  # "shelf", "regal", "schreibtisch": the place noun is enough
        ok = not abst and re.search(r"\b" + re.escape(noun), line) is not None
        d = {"ok_unique": float(ok), f"ok_v{item['variant']}": float(ok)}
    return {"correct": ok, "score": float(ok), "pred": line[:50], "gold": item["gold"], "detail": d}


# ---------------------------------------------------------------- counterfactual_world (reason inside stated rules that contradict the real world)
# A fictional world states one rule that differs from reality (water freezes at 40 °C there, adults are 25 ...). The
# answer follows the stated rule; ok_conflict (rule and reality disagree: the trap) vs ok_agree (they agree: the
# control, so 'always answer against reality' is no shortcut).
CW_RULES = [  # (en rule, de rule, en q, de q, in-world truth fn(v), real truth fn(v), value range)
    ("In Veloria, water freezes at 40 °C and boils at 180 °C.", "In Veloria gefriert Wasser bei 40 °C und kocht bei 180 °C.",
     "A lake in Veloria is at {v} °C. Is its water frozen?", "Ein See in Veloria hat {v} °C. Ist sein Wasser gefroren?",
     lambda v: v <= 40, lambda v: v <= 0, (-10, 60)),
    ("In Veloria, people come of age at 25.", "In Veloria wird man mit 25 volljährig.",
     "Mira lives in Veloria and is {v}. Is she of age?", "Mira lebt in Veloria und ist {v}. Ist sie volljährig?",
     lambda v: v >= 25, lambda v: v >= 18, (14, 32)),
    ("In Veloria, a week has 9 days.", "In Veloria hat eine Woche 9 Tage.",
     "In Veloria, is a stay of {v} days longer than two weeks?", "Ist in Veloria ein Aufenthalt von {v} Tagen länger als zwei Wochen?",
     lambda v: v > 18, lambda v: v > 14, (10, 24)),
    ("In Veloria, heavier objects fall more slowly than light ones.", "In Veloria fallen schwere Gegenstände langsamer als leichte.",
     "In Veloria, a {v} kg stone and a 1 kg ball are dropped together from a tower. {fq}",
     "In Veloria lässt man einen {v}-kg-Stein und einen 1-kg-Ball gleichzeitig von einem Turm fallen. {fq}",
     None, None, (5, 50)),
    ("In Veloria, the sun rises in the west.", "In Veloria geht die Sonne im Westen auf.",
     "In Veloria, a house faces {v}. Does it get the morning sun?",
     "In Veloria ist ein Haus nach {v} ausgerichtet. Bekommt es die Morgensonne?",
     None, None, None),
]


CW_FALL = [("Does the ball land first?", "Landet der Ball zuerst?", True, False),
           ("Does the stone land first?", "Landet der Stein zuerst?", False, False),
           ("Do both land at the same moment?", "Landen beide im selben Moment?", False, True),
           ("Does the stone land last?", "Landet der Stein zuletzt?", True, False)]
CW_COMBOS = [(True, True), (True, False), (False, True), (False, False)]  # (conflict, in-world answer)


def load_counterfactual_world(lang, n, seed):
    items = []
    for i in range(n or 200):
        r = rng_of("counterfactual_world", i)
        k = i % len(CW_RULES)
        en_r, de_r, en_q, de_q, world, real, rng = CW_RULES[k]
        want = CW_COMBOS[(i // len(CW_RULES)) % 4]  # balanced conflict / agree and yes / no where the rule allows it
        if k == 3:  # falling: the question form carries the answer
            fe, fd, ans, real_ans = r.choice([f for f in CW_FALL if (f[2] != f[3], f[2]) == want] or CW_FALL)
            conflict = ans != real_ans
            q = (en_q if lang == "en" else de_q).format(v=r.randint(*rng), fq=fe if lang == "en" else fd)
        elif rng is None:  # the sun rule: either side is a conflict with reality
            side_en, side_de = r.choice([("east", "Osten"), ("west", "Westen")])
            ans, conflict = side_en == "west", True
            q = (en_q if lang == "en" else de_q).format(v=side_en if lang == "en" else side_de)
        else:
            vals = [v for v in range(rng[0], rng[1] + 1) if v != 40 and (world(v) != real(v), world(v)) == want]
            if not vals:  # this combination cannot occur for the rule: any value
                vals = [v for v in range(rng[0], rng[1] + 1) if v != 40]
            v = r.choice(vals)
            ans, conflict = world(v), world(v) != real(v)
            q = (en_q if lang == "en" else de_q).format(v=v)
        yes, no = ("yes", "no") if lang == "en" else ("ja", "nein")
        items.append({"id": str(i), "lang": lang, "rule": k, "conflict": conflict, "gold": yes if ans else no,
                      "messages": [{"role": "user", "content": (en_r if lang == "en" else de_r) + " " + q
                                    + ("\nAnswer yes or no." if lang == "en" else "\nAntworte mit ja oder nein.")
                                    + "\n" + ASK[lang]}]})
    return items


def score_counterfactual_world(item, reply):
    yn = [w for w in re.findall(r"[a-z]+", (final_answer(reply["content"]) or "").lower()) if w in ("yes", "no", "ja", "nein")]
    ok = yn[:1] == [item["gold"]]
    c = "conflict" if item["conflict"] else "agree"
    return {"correct": ok, "score": float(ok), "pred": yn[:1], "gold": item["gold"],
            "detail": {f"ok_rule{item['rule']}": float(ok), f"ok_{c}": float(ok), f"ok_{c}_rule{item['rule']}": float(ok)}}


# ---------------------------------------------------------------- fallacy_spot (name the reasoning flaw in a generated argument)
# Arguments about invented proposals, each built on one fallacy, plus a sound control argument. The model names the
# flaw from a fixed list (or "none"). Topics are invented, so knowing the debate does not help.
FS_KINDS = ["ad_hominem", "false_dilemma", "hasty_generalization", "post_hoc", "slippery_slope", "none"]
FS_LABEL = {"en": {"ad_hominem": "attack on the person", "false_dilemma": "false either-or",
                   "hasty_generalization": "hasty generalization", "post_hoc": "false cause (after, therefore because)",
                   "slippery_slope": "slippery slope", "none": "no flaw"},
            "de": {"ad_hominem": "Angriff auf die Person", "false_dilemma": "falsches Entweder-oder",
                   "hasty_generalization": "vorschnelle Verallgemeinerung",
                   "post_hoc": "Scheinkausalität (danach, also deshalb)", "slippery_slope": "Dammbruchargument",
                   "none": "kein Fehler"}}
FS_TEXT = common.OwnSet("tasks_novel.FS_TEXT")  # private: flozi00/weakbench-data
FS_TOPIC = {"en": ["Zorvel parking rule", "Brimak market fee", "Telo night bus", "Quanda street lighting plan"],
            "de": ["Zorvel-Parkregel", "Brimak-Marktgebühr", "Telo-Nachtbuslinie", "Quanda-Beleuchtungsregel"]}


FS_TEXT2 = common.OwnSet("tasks_novel.FS_TEXT2")  # private: flozi00/weakbench-data


def load_fallacy_spot(lang, n, seed):
    items = []
    for i in range(n or 120):
        r = rng_of("fallacy_spot", i)
        kind = FS_KINDS[i % len(FS_KINDS)]
        labels = list(FS_KINDS)
        r.shuffle(labels)
        arg = (FS_TEXT, FS_TEXT2)[(i // len(FS_KINDS)) % 2][lang][kind].format(x=r.choice(FS_TOPIC[lang]), p=r.choice(["Ms Arden", "Mr Kolb"] if lang == "en"
                                                                                   else ["Frau Arden", "Herr Kolb"]))
        opts = "\n".join(f"{'ABCDEF'[j]}) {FS_LABEL[lang][k]}" for j, k in enumerate(labels))
        q = ("Which reasoning flaw does this argument contain?" if lang == "en"
             else "Welchen Denkfehler enthält dieses Argument?")
        items.append({"id": str(i), "lang": lang, "kind": kind, "gold": "ABCDEF"[labels.index(kind)],
                      "letters": {k: "ABCDEF"[j] for j, k in enumerate(labels)},
                      "messages": [{"role": "user", "content": f"{arg}\n\n{q}\n{opts}\n" + ASK[lang]}]})
    return items


def score_fallacy_spot(item, reply):
    line = (final_answer(reply["content"]) or "").strip()
    m = re.match(r"\W*([A-F])(?:[).:]|\s*$)", line) or re.search(r"\b([A-F])\)", line)
    pred = m.group(1) if m else None
    if pred is None:  # the label text instead of the letter
        hit = [k for k, lab in FS_LABEL[item["lang"]].items() if lab.split(" (")[0].lower() in line.lower()]
        pred = item["letters"][hit[0]] if len(hit) == 1 else None
    ok = pred == item["gold"]
    return {"correct": ok, "score": float(ok), "pred": pred, "gold": item["gold"], "detail": {f"ok_{item['kind']}": float(ok)}}


# ---------------------------------------------------------------- arith_length (where exact arithmetic breaks, by length)
# Exact addition and multiplication of n-digit numbers, n = 2..8, no calculator. The per-length sub-metrics show the
# length at which a model stops being exact: a generalisation cliff, not a single score.
AL_LEN = (2, 3, 4, 5, 6, 7, 8)


def load_arith_length(lang, n, seed):
    items = []
    for i in range(n or 140):
        r = rng_of("arith_length", i)
        d = AL_LEN[i % len(AL_LEN)]
        op = ("+", "×")[(i // len(AL_LEN)) % 2]
        a, b = (r.randint(10 ** (d - 1), 10 ** d - 1) for _ in range(2))
        gold = a + b if op == "+" else a * b
        q = (f"Compute exactly: {a} {op} {b}" if lang == "en" else f"Berechne exakt: {a} {op} {b}")
        items.append({"id": str(i), "lang": lang, "digits": d, "op": "add" if op == "+" else "mul", "gold": gold,
                      "messages": [{"role": "user", "content": q + "\n" + ASK[lang]}]})
    return items


def score_arith_length(item, reply):
    line = final_answer(reply["content"]) or ""
    digits = re.sub(r"[\s.,'’ ]", "", re.sub(r"[.,]0+\b", "", line.split("=")[-1]))  # "494.0" is 494
    m = re.search(r"\d+", digits)
    ok = bool(m) and int(m.group()) == item["gold"]
    return {"correct": ok, "score": float(ok), "pred": m.group() if m else None, "gold": item["gold"],
            "detail": {f"ok_{item['op']}{item['digits']}": float(ok), f"ok_d{item['digits']}": float(ok)}}


# ---------------------------------------------------------------- nesting_depth (where composition breaks, by depth)
# Two kinds: evaluate a nested expression of small numbers at depth 1..6, and decide whether a bracket string of
# depth 2..7 over ()[]{} is balanced (one swapped or dropped bracket in half of them). Per-depth sub-metrics.
ND_DEPTH = (1, 2, 3, 4, 5, 6)


def nd_expr(r, d):
    if d == 0:
        return str(r.randint(1, 9))
    op = r.choice("+-*")
    left, right = nd_expr(r, d - 1), str(r.randint(1, 9))
    if r.random() < 0.5:
        left, right = right, left
    return f"({left} {op} {right})"


def nd_brackets(r, d):
    """A balanced string of depth d with siblings at random levels ("([]{})"), not only mirrored chains."""
    p = r.choice(["()", "[]", "{}"])
    if d == 1:
        return p
    inner = nd_brackets(r, d - 1)
    if r.random() < 0.5:
        sib = nd_brackets(r, r.randint(1, max(1, d - 2)))
        inner = inner + sib if r.random() < 0.5 else sib + inner
    return p[0] + inner + p[1]


def load_nesting_depth(lang, n, seed):
    items = []
    for i in range(n or 144):
        r = rng_of("nesting_depth", i)
        kind = ("eval", "balance")[i % 2]
        d = ND_DEPTH[(i // 2) % len(ND_DEPTH)]
        if kind == "eval":
            e = nd_expr(r, d)
            gold = str(eval(e))  # noqa: S307 - our own generated digits and + - * only
            q = (f"Evaluate: {e}" if lang == "en" else f"Berechne: {e}")
        else:
            s = nd_brackets(r, d + 1)
            ok = (i // 12) % 2 == 0
            if not ok:
                k = r.randrange(len(s))
                swap = {"(": "[", "[": "{", "{": "(", ")": "]", "]": "}", "}": ")"}
                if r.random() < 0.6:  # a closer of the wrong type
                    s = s[:k] + swap[s[k]] + s[k + 1:]
                else:  # two neighbours exchanged: same length and counts, so length gives nothing away
                    k = min(k, len(s) - 2)
                    s = s[:k] + s[k + 1] + s[k] + s[k + 2:]
            gold = ("yes" if lang == "en" else "ja") if nd_balanced(s) else ("no" if lang == "en" else "nein")
            q = (f"Is this bracket string balanced (every bracket closed by its matching type, properly nested)? "
                 f"Answer yes or no.\n{s}" if lang == "en" else
                 f"Ist diese Klammerfolge korrekt (jede Klammer vom passenden Typ geschlossen, richtig verschachtelt)? "
                 f"Antworte mit ja oder nein.\n{s}")
        items.append({"id": str(i), "lang": lang, "kind": kind, "depth": d, "gold": gold,
                      "messages": [{"role": "user", "content": q + "\n" + ASK[lang]}]})
    return items


def nd_balanced(s):
    stack, match = [], {")": "(", "]": "[", "}": "{"}
    for c in s:
        if c in "([{":
            stack.append(c)
        elif not stack or stack.pop() != match[c]:
            return False
    return not stack


def score_nesting_depth(item, reply):
    line = (final_answer(reply["content"]) or "").lower()
    if item["kind"] == "eval":
        line = re.sub(r"\bminus\s+", "-", line.replace("−", "-"))
        firsts = [re.findall(r"-?\d+", seg)[:1] for seg in line.split("=")]  # "-2 = 3 - 5" and "3 - 5 = -2"
        ok = [item["gold"]] in firsts
    else:
        yn = [w for w in re.findall(r"[a-z]+", line) if w in ("yes", "no", "ja", "nein")]
        ok = yn[:1] == [item["gold"]]
    return {"correct": ok, "score": float(ok), "pred": line[:30], "gold": item["gold"],
            "detail": {f"ok_{item['kind']}_d{item['depth']}": float(ok), f"ok_{item['kind']}": float(ok)}}


# ---------------------------------------------------------------- grid_transform (mental rotation / reflection of a letter grid)
# A 3x3 or 4x4 grid of distinct letters, 1-3 operations (rotate 90° clockwise, rotate 90° counter-clockwise, mirror
# left-right, mirror top-bottom), then: what is row k? Per number-of-operations sub-metric.
GT_OPS = {"cw": lambda g: [list(r) for r in zip(*g[::-1])], "ccw": lambda g: [list(r) for r in zip(*g)][::-1],
          "lr": lambda g: [r[::-1] for r in g], "tb": lambda g: g[::-1]}
GT_NAME = {"en": {"cw": "rotate 90° clockwise", "ccw": "rotate 90° counter-clockwise", "lr": "mirror left-right (swap left and right)",
                  "tb": "mirror top-bottom (swap top and bottom)"},
           "de": {"cw": "um 90° im Uhrzeigersinn drehen", "ccw": "um 90° gegen den Uhrzeigersinn drehen",
                  "lr": "links-rechts spiegeln (links und rechts vertauschen)",
                  "tb": "oben-unten spiegeln (oben und unten vertauschen)"}}


def load_grid_transform(lang, n, seed):
    items = []
    for i in range(n or 150):
        r = rng_of("grid_transform", i)
        size = 3 if i % 2 == 0 else 4
        letters = r.sample("ABCDEFGHJKLMNPQRSTUVWXYZ", size * size)
        g = [letters[k * size:(k + 1) * size] for k in range(size)]
        nops = 1 + (i // 2) % 3
        while True:  # no sequence that collapses: every step yields a grid not seen before
            ops = [r.choice(list(GT_OPS)) for _ in range(nops)]
            states, out = [g], g
            for o in ops:
                out = GT_OPS[o](out)
                states.append(out)
            if len({str(x) for x in states}) == len(states):
                break
        k = r.randrange(size)
        grid = "\n".join(" ".join(row) for row in g)
        steps = ", ".join(GT_NAME[lang][o] for o in ops)
        q = (f"Grid:\n{grid}\n\nApply in order: {steps}. What is row {k + 1} (from the top) afterwards? Answer with "
             "the letters left to right." if lang == "en" else
             f"Raster:\n{grid}\n\nWende nacheinander an: {steps}. Wie lautet danach Zeile {k + 1} (von oben)? Antworte "
             "mit den Buchstaben von links nach rechts.")
        items.append({"id": str(i), "lang": lang, "ops": len(ops), "gold": "".join(out[k]), "size": size,
                      "messages": [{"role": "user", "content": q + "\n" + ASK[lang]}]})
    return items


def score_grid_transform(item, reply):
    line = (final_answer(reply["content"]) or "").upper()
    n = item.get("size", len(item["gold"]))
    runs = re.findall(r"(?:\b[A-Z]\b[\s,;|>–-]*){%d}" % n, line)  # the last run of n single letters ("I think: C A D")
    words = re.findall(r"\b[A-Z]{%d}\b" % n, line)
    got = re.sub(r"[^A-Z]", "", runs[-1]) if runs else (words[-1] if words else "")
    ok = got == item["gold"]
    return {"correct": ok, "score": float(ok), "pred": got, "gold": item["gold"], "detail": {f"ok_ops{item['ops']}": float(ok)}}


# ---------------------------------------------------------------- blocks_plan (PlanBench-style blocksworld, checked by simulation)
# Blocks on a table in stacks; the model writes moves "move X onto Y" / "move X onto table" (only a clear block, onto a
# clear block or the table). A simulator checks every move and the goal; any valid plan counts, `optimal` is reported.
BP_NAMES = "ABCDEFG"


def bp_random_state(r, blocks):
    stacks = []
    for b in r.sample(blocks, len(blocks)):
        if stacks and r.random() < 0.6:
            r.choice(stacks).append(b)
        else:
            stacks.append([b])
    return stacks


def bp_desc(stacks, lang):
    parts = []
    for st in stacks:
        parts.append(" on ".join(reversed(st)) + " on the table" if lang == "en" else
                     " auf ".join(reversed(st)) + " auf dem Tisch")
    return "; ".join(parts)


def bp_canon(stacks):
    return sorted(tuple(s) for s in stacks)


def bp_optimal(start, goal):
    """Breadth-first search over states: the shortest plan length (tiny: at most 6 blocks)."""
    from collections import deque
    start, goal = tuple(bp_canon(start)), tuple(bp_canon(goal))
    seen, q = {start}, deque([(start, 0)])
    while q:
        st, d = q.popleft()
        if st == goal:
            return d
        stacks = [list(x) for x in st]
        for i, src in enumerate(stacks):
            b = src[-1]
            targets = [j for j in range(len(stacks)) if j != i] + ([None] if len(src) > 1 else [])
            for j in targets:
                new = [list(x) for x in stacks]
                new[i].pop()
                if j is None:
                    new.append([b])
                else:
                    new[j].append(b)
                key = tuple(bp_canon([x for x in new if x]))
                if key not in seen:
                    seen.add(key)
                    q.append((key, d + 1))
    return None


def load_blocks_plan(lang, n, seed):
    items = []
    for i in range(n or 120):
        r = rng_of("blocks_plan", i)
        k = 3 + i % 4
        blocks = list(BP_NAMES[:k])
        start = bp_random_state(r, blocks)
        goal = bp_random_state(r, blocks)
        while bp_canon(goal) == bp_canon(start):
            goal = bp_random_state(r, blocks)
        q = (f"Blocks world (each stack listed top to bottom). Now: {bp_desc(start, 'en')}. Goal: {bp_desc(goal, 'en')}. You may move one clear block "
             "(nothing on it) at a time onto the table or onto another clear block. Write the plan as lines "
             "\"move X onto Y\" or \"move X onto table\", nothing else." if lang == "en" else
             f"Klötzchenwelt (jeder Stapel von oben nach unten). Jetzt: {bp_desc(start, 'de')}. Ziel: {bp_desc(goal, 'de')}. Du darfst immer einen freien "
             "Klotz (nichts liegt darauf) auf den Tisch oder auf einen anderen freien Klotz legen. Schreibe den Plan als "
             "Zeilen \"move X onto Y\" oder \"move X onto table\", sonst nichts.")
        items.append({"id": str(i), "lang": lang, "blocks": k, "start": start, "goal": goal,
                      "optimal": bp_optimal(start, goal), "gold": None, "messages": [{"role": "user", "content": q}]})
    return items


def bp_run(start, text):
    """Simulate the moves; returns (final stacks or None on an illegal move, number of moves)."""
    stacks = [list(s) for s in start]
    line_re = re.compile(r"^\W*(?:\d+[.)]\s*)?move\s+([A-G])\s+(?:onto|on|to)\s+(?:the\s+)?(table|[A-G])\b\W*$", re.I)
    runs, cur = [], []
    for ln in text.splitlines():  # only lines that are a move; the last unbroken run is the plan
        m = line_re.match(ln.strip())
        if m:
            cur.append(m.groups())
        elif ln.strip():
            if cur:
                runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    moves = runs[-1] if runs else []
    for b, t in moves:
        b, t = b.upper(), t.upper()
        src = next((s for s in stacks if s and s[-1] == b), None)
        if src is None:
            return None, len(moves)
        if t == "TABLE":
            if len(src) > 1:  # already on the table: a no-op that only costs optimality
                src.pop()
                stacks.append([b])
        else:
            dst = next((s for s in stacks if s and s[-1] == t), None)
            if dst is None or dst is src:
                return None, len(moves)
            src.pop()
            dst.append(b)
        stacks = [s for s in stacks if s]
    return stacks, len(moves)


def score_blocks_plan(item, reply):
    final, n = bp_run(item["start"], reply["content"] or "")
    ok = final is not None and n > 0 and bp_canon(final) == bp_canon(item["goal"])
    return {"correct": ok, "score": float(ok), "pred": n, "gold": item["optimal"],
            "detail": {f"ok_b{item['blocks']}": float(ok), "illegal": float(final is None),
                       "optimal": float(ok and n == item["optimal"])}}


# ---------------------------------------------------------------- judge_ability (the model as a judge: which solution is right?)
# Two worked solutions to the same generated problem: one correct, one with a single slip carried on to the end, the
# wrong one sometimes longer and more confident. Each pair is asked twice with A/B swapped ("grouped": a judge that
# flips with the order is unreliable). `chose_longer` exposes length bias.
JA_TEXT = {"en": ("A club buys {a} boxes of {b} balls. {c} balls are lost. The rest is split equally among {d} teams. "
                  "How many balls does each team get?", "Which solution is correct? Answer A or B."),
           "de": ("Ein Verein kauft {a} Kartons mit je {b} Bällen. {c} Bälle gehen verloren. Der Rest wird gleichmäßig auf "
                  "{d} Teams verteilt. Wie viele Bälle bekommt jedes Team?", "Welche Lösung ist richtig? Antworte mit A oder B.")}
JA_FLUFF = {"en": "Let me carefully double-check every step. All steps are verified, so the answer is certainly correct.",
            "de": "Ich prüfe jeden Schritt sorgfältig nach. Alle Schritte sind geprüft, das Ergebnis ist also sicher richtig."}


def ja_solution(a, b, c, d, slip_step, slip, lang, fluff):
    v1 = a * b + (slip if slip_step == 1 else 0)
    v2 = v1 - c + (slip if slip_step == 2 else 0)
    v3 = v2 // d + (slip if slip_step == 3 else 0)
    div = ":" if lang == "de" else "÷"
    lines = [f"{a} × {b} = {v1}", f"{v1} − {c} = {v2}", f"{v2} {div} {d} = {v3}",
             ("Antwort: " if lang == "de" else "Answer: ") + str(v3)]
    return ("\n".join(lines) + ("\n" + JA_FLUFF[lang] if fluff else "")), v3


def load_judge_ability(lang, n, seed):
    items = []
    for i in range(n or 160):
        g, swap = divmod(i, 2)
        r = rng_of("judge_ability", g)
        d = r.choice([3, 4, 5, 6])
        a, b = r.randint(4, 12), r.choice([10, 12, 20, 24])
        c = (a * b) % d + d * r.randint(1, 3)
        k = 1 + g % 3
        slip = d * r.choice([-2, -1, 1, 2]) if k < 3 else r.choice([-1, 1])
        # the confident sentence sits on the wrong solution, on the right one, or on neither, equally often: no payoff
        fluff = ("wrong", "right", "none")[g % 3]
        fluff_wrong = fluff == "wrong"
        right, _ = ja_solution(a, b, c, d, 0, 0, lang, fluff == "right")
        wrong, _ = ja_solution(a, b, c, d, k, slip, lang, fluff_wrong)
        first_right = ((g // 3) % 2 == 0) != bool(swap)
        A, B = (right, wrong) if first_right else (wrong, right)
        q, ask = JA_TEXT[lang]
        content = f"{q.format(a=a, b=b, c=c, d=d)}\n\nA:\n{A}\n\nB:\n{B}\n\n{ask}\n" + ASK[lang]
        items.append({"id": f"{g}-{swap}", "lang": lang, "gold": "A" if first_right else "B",
                      "longer": "A" if len(A) > len(B) else "B", "fluff_wrong": fluff_wrong, "fluff": fluff,
                      "messages": [{"role": "user", "content": content}]})
    return items


def score_judge_ability(item, reply):
    line = (final_answer(reply["content"]) or "").strip()
    letters = re.findall(r"\b([AB])\b", line)
    pred = None
    if len(set(letters)) == 1:
        pred = letters[0]
    elif letters:  # "A is wrong, B is correct": the letter next to the verdict word
        m = re.search(r"\b([AB])\b[^AB]{0,25}\b(correct|right|richtig|stimmt)", line, re.I) or \
            re.search(r"\b(correct|right|richtig)\b[^AB]{0,15}\b([AB])\b", line, re.I)
        if m:
            pred = m.group(1) if m.group(1) in "AB" else m.group(2)
        elif re.search(r"\bso\s+([AB])\b\W*$|\balso\s+([AB])\b\W*$", line):
            pred = letters[-1]
    ok = pred == item["gold"]
    d = {f"ok_fluff_{item.get('fluff', 'wrong' if item['fluff_wrong'] else 'none')}": float(ok)}
    if pred:
        d["chose_longer"] = float(pred == item["longer"])
    return {"correct": ok, "score": float(ok), "pred": pred, "gold": item["gold"], "detail": d}


# ---------------------------------------------------------------- regex_synth (write a regex from examples, checked by running it)
# A hidden pattern built from 2-3 parts (digits, letters, a literal separator, an optional part) generates matching
# and near-miss strings. The model writes one Python regex; it must fullmatch every positive (shown and held out)
# and no negative (shown and held out). Held-out strings test generalisation, not memorising the list.
RS_PARTS = [("[0-9]{%d}", "%d digits", lambda r, n: "".join(r.choice("0123456789") for _ in range(n))),
            ("[A-Z]{%d}", "%d capital letters", lambda r, n: "".join(r.choice("ABCDEFGHKMNPRSTXZ") for _ in range(n))),
            ("[a-z]{%d}", "%d small letters", lambda r, n: "".join(r.choice("abcdefghkmnprstxz") for _ in range(n)))]
RS_SEP = ["-", "_", ".", "/", ":"]


def rs_make(r):
    k = r.choice([2, 3])
    parts = [(RS_PARTS[r.randrange(3)], r.randint(2, 4)) for _ in range(k)]
    sep = r.choice(RS_SEP)
    pat = re.escape(sep).join(p[0] % n for p, n in parts)

    def gen(rr):
        return sep.join(p[2](rr, n) for p, n in parts)
    return pat, gen, sep, parts


def rs_negative(r, s, sep):
    """A near miss: one character class wrong, one character too many or too few, or another separator."""
    kind = r.randrange(4)
    i = r.randrange(len(s))
    if kind == 0 and s[i] != sep:
        if s[i].isalpha() and r.random() < 0.5:
            return s[:i] + s[i].swapcase() + s[i + 1:]  # wrong letter case: [A-Za-z] must not pass
        swap = "x" if s[i].isdigit() else ("7" if s[i].isalpha() else "x")
        return s[:i] + swap + s[i + 1:]
    if kind == 1:
        return s[:i] + s[i] + s[i:]
    if kind == 2 and len(s) > 3:
        return s[:i] + s[i + 1:]
    other = r.choice([x for x in RS_SEP if x != sep])
    return s.replace(sep, other)


def load_regex_synth(lang, n, seed):
    items = []
    for i in range(n or 80):
        r = rng_of("regex_synth", i)
        pat, gen, sep, parts = rs_make(r)
        pos = list(dict.fromkeys(gen(r) for _ in range(12)))
        negs = list(dict.fromkeys(x for x in (rs_negative(r, gen(r), sep) for _ in range(30))
                                  if not re.fullmatch(pat, x)))[:12]
        shown_p, held_p, shown_n, held_n = pos[:5], pos[5:], negs[:5], negs[5:]
        q = (("Write one Python regular expression that matches (with re.fullmatch) all of these:\n" if lang == "en" else
              "Schreibe einen regulären Python-Ausdruck, der (mit re.fullmatch) alle diese Zeichenketten trifft:\n")
             + "\n".join(shown_p) + ("\n\nand none of these:\n" if lang == "en" else "\n\nund keine von diesen:\n")
             + "\n".join(shown_n) + ("\n\nIt must generalise to new strings of the same kind. Give the regex alone in a "
                                     "code block." if lang == "en" else "\n\nEr muss auf neue Zeichenketten derselben Art "
                                     "verallgemeinern. Gib nur den Ausdruck in einem Codeblock an."))
        items.append({"id": str(i), "lang": lang, "gold": pat, "pos": pos, "neg": negs, "shown": 5,
                      "messages": [{"role": "user", "content": q}]})
    return items


def _rx_ok(pat, pos):
    try:
        return all(re.fullmatch(pat, s) for s in pos)
    except re.error:
        return False


def score_regex_synth(item, reply):
    text = reply["content"] or ""
    # a language tag only counts when a newline follows it ("```python\n"); "```AB_CD```" is the regex itself
    blocks = [b for _, b in re.findall(r"```([a-zA-Z]*[ \t]*\n)?(.*?)```", text, re.S)]
    spans = re.findall(r"`([^`\n]+)`", re.sub(r"```.*?```", "", text, flags=re.S))
    pool = blocks[-1:] or spans[::-1] or ([text.strip().splitlines()[-1]] if text.strip() else [])

    def extract(chunk):  # the quoted pattern of `pattern = r"..."` / re.compile(r"..."), else the first code line
        q = re.search(r"""r?(["'])(.+?)\1""", chunk)
        if q:
            return q.group(2)
        lines = [ln.strip() for ln in chunk.strip().splitlines() if ln.strip() and not ln.strip().startswith("import ")]
        return lines[0] if lines else ""
    cands = [extract(c) for c in pool]
    fits = [c for c in cands if _rx_ok(c, item["pos"][:5])]
    cand = (fits or cands or [""])[0]
    try:
        rx = re.compile(cand)
        hit = lambda s: rx.fullmatch(s) is not None  # noqa: E731
        shown_ok = all(hit(s) for s in item["pos"][:5]) and not any(hit(s) for s in item["neg"][:5])
        held_ok = all(hit(s) for s in item["pos"][5:]) and not any(hit(s) for s in item["neg"][5:])
    except re.error:
        shown_ok = held_ok = False
    ok = shown_ok and held_ok
    return {"correct": ok, "score": float(ok), "pred": cand[:60], "gold": item["gold"],
            "detail": {"fits_shown": float(shown_ok), "generalises": float(ok), "overfit": float(shown_ok and not held_ok)}}


# ---------------------------------------------------------------- continuity_error (find the sentence that contradicts the story)
# A generated 8-10 sentence story establishes facts (a coat's colour, who owns the dog, a time, a count); one later
# sentence contradicts one of them, or none does (control). Answer: the number of the contradicting sentence, or 0.
CE_FACTS = {
    "en": [("{a} owned only one coat, a {c1} one.", "{a} buttoned up the {c2} coat.", "{a} buttoned up the {c1} coat.",
            ["red", "green", "grey", "yellow"]),
           ("The dog belonged to {b}.", "{b} had never owned a dog.", "{a} took {b}'s dog for a walk.", None),
           ("The train left at {t1} o'clock.", "At {t2} o'clock the train was still waiting, as it always left at {t2}.",
            "Shortly after {t1} o'clock the train was gone.", None),
           ("{b} had baked {n1} rolls.", "All {n2} rolls {b} had baked were eaten.", "All {n1} rolls {b} had baked were eaten.",
            None)],
    "de": [("{a} besaß nur einen einzigen Mantel, einen {c1}.", "{a} knöpfte den {c2} Mantel zu.",
            "{a} knöpfte den {c1} Mantel zu.",
            ["roten", "grünen", "grauen", "gelben"]),
           ("Der Hund gehörte {b}.", "{b} hatte noch nie einen Hund gehabt.", "{a} ging mit {b_gen} Hund spazieren.", None),
           ("Der Zug fuhr um {t1} Uhr ab.", "Um {t2} Uhr wartete der Zug noch, denn er fuhr immer um {t2} Uhr ab.",
            "Kurz nach {t1} Uhr war der Zug weg.", None),
           ("{b} hatte {n1} Brötchen gebacken.", "Alle {n2} Brötchen, die {b} gebacken hatte, wurden gegessen.",
            "Alle {n1} Brötchen, die {b} gebacken hatte, wurden gegessen.", None)]}
CE_FILL = {"en": ["The sky was clear.", "Birds sang in the trees.", "The street was quiet.", "A bus drove past.",
                  "Someone laughed in the distance.", "The bakery smelled of bread.", "A cyclist rang a bell.",
                  "A dog barked twice.", "A cat slept on a wall.", "Two children argued about a ball."],
           "de": ["Der Himmel war klar.", "In den Bäumen sangen Vögel.", "Die Straße war ruhig.", "Ein Bus fuhr vorbei.",
                  "In der Ferne lachte jemand.", "Die Bäckerei duftete nach Brot.", "Ein Radfahrer klingelte.",
                  "Ein Hund bellte zweimal.", "Eine Katze schlief auf einer Mauer.",
                  "Zwei Kinder stritten um einen Ball."]}
# consistent sentences that touch the same fact in a way that looks suspicious but is not a contradiction
CE_NEAR = {"en": ["{a} took off the {c1} coat and hung it up.", "{b} fed the dog before breakfast.",
                  "At ten to {t1} the train was still waiting.", "Only {n3} of the rolls were eaten; the rest were frozen."],
           "de": ["{a} zog den {c1} Mantel aus und hängte ihn auf.", "{b} fütterte den Hund vor dem Frühstück.",
                  "Um zehn vor {t1} wartete der Zug noch.", "Nur {n3} der Brötchen wurden gegessen, der Rest wurde eingefroren."]}


def load_continuity_error(lang, n, seed):
    items = []
    for i in range(n or 120):
        r = rng_of("continuity_error", i)
        a, b = r.sample(["Lena", "Tom", "Mira", "Jonas"], 2)
        f = CE_FACTS[lang][i % 4]
        cols = f[3]
        c1, c2 = r.sample(cols, 2) if cols else ("", "")
        t1 = r.randint(6, 9)
        n1 = r.randint(6, 12)
        vals = {"a": a, "b": b, "c1": c1, "c2": c2, "t1": t1, "t2": t1 + 2, "n1": n1, "n2": n1 + r.choice([-3, 3]),
                "b_gen": b + ("'" if b.endswith("s") else "s"), "n3": n1 - 2}
        contra = (i // 4) % 3 != 0  # a third are clean controls
        fact = i % 4
        late = (f[1] if contra else (f[2] if (i // 12) % 2 else CE_NEAR[lang][fact])).format(**vals)
        decoy = r.random() < 0.5 if contra else (i // 12) % 2 == 0
        sents = [f[0].format(**vals)] + r.sample(CE_FILL[lang], 7 - decoy)  # the same length with or without decoy
        at = r.randint(3, 7 - decoy)  # the late sentence lands somewhere after a few fillers
        sents.insert(at, late)
        if decoy and contra:  # a consistent near-miss elsewhere as a decoy
            sents.insert(r.randint(1, at - 1), CE_NEAR[lang][fact].format(**vals))
        elif decoy:  # controls whose late sentence is already the near-miss keep one more filler
            sents.insert(r.randint(1, len(sents) - 1), r.choice([x for x in CE_FILL[lang] if x not in sents]))
        gold = sents.index(late) + 1 if contra else 0
        story = " ".join(f"({k + 1}) {x}" for k, x in enumerate(sents))
        q = ("Does any sentence contradict an earlier one? Answer with its number, or 0 if none does." if lang == "en" else
             "Widerspricht ein Satz einem früheren? Antworte mit seiner Nummer oder mit 0, wenn keiner das tut.")
        items.append({"id": str(i), "lang": lang, "gold": gold, "fact": i % 4,
                      "messages": [{"role": "user", "content": story + "\n\n" + q + "\n" + ASK[lang]}]})
    return items


def score_continuity_error(item, reply):
    v = last_int(final_answer(reply["content"]) or "")
    ok = v == item["gold"]
    return {"correct": ok, "score": float(ok), "pred": v, "gold": item["gold"],
            "detail": {"ok_clean" if item["gold"] == 0 else f"ok_fact{item['fact']}": float(ok)}}


# ---------------------------------------------------------------- number_compare (9.11 vs 9.9 and its relatives)
# Which of two numbers is larger, where surface cues mislead: more decimal digits but smaller (9.11 < 9.9), larger
# denominator (3/7 vs 3/5), negatives (-3.2 vs -3.15), and a control without a trap. German items use decimal commas.
NC_KINDS = ("decimals", "fractions", "negatives", "control")


def load_number_compare(lang, n, seed):
    items = []
    for i in range(n or 160):
        r = rng_of("number_compare", i)
        kind = NC_KINDS[i % 4]
        if kind == "decimals":
            w = r.randint(1, 20)
            a = f"{w}.{r.randint(10, 49)}"   # more digits, smaller
            b = f"{w}.{r.randint(5, 9)}"
            big = b
        elif kind == "fractions":
            num = r.randint(2, 5)
            d1, d2 = sorted(r.sample(range(num + 1, num + 9), 2))
            a, b = f"{num}/{d2}", f"{num}/{d1}"  # the bigger denominator is the smaller number
            big = b
        elif kind == "negatives":
            w = r.randint(1, 9)
            a, b = f"-{w}.{r.randint(2, 9)}", f"-{w}.{r.randint(10, 19)}"  # -3.2 < -3.15
            big = b
        else:
            x, y = sorted(r.sample(range(10, 99), 2))
            a, b = f"{x // 10}.{x % 10}", f"{y // 10}.{y % 10}"
            big = b if float(b) > float(a) else a
        if r.random() < 0.5:
            a, b = b, a
        fmt = (lambda t: t.replace(".", ",")) if lang == "de" else (lambda t: t)
        q = (f"Which number is larger: {fmt(a)} or {fmt(b)}? Answer with the number." if lang == "en" else
             f"Welche Zahl ist größer: {fmt(a)} oder {fmt(b)}? Antworte mit der Zahl.")
        items.append({"id": str(i), "lang": lang, "kind": kind, "gold": fmt(big), "other": fmt(a if big == b else b),
                      "messages": [{"role": "user", "content": q + "\n" + ASK[lang]}]})
    return items


def score_number_compare(item, reply):
    line = (final_answer(reply["content"]) or "").replace(" ", "").replace("−", "-").replace(",", ".")
    g, o = item["gold"].replace(",", "."), item["other"].replace(",", ".")  # "9.9" is fine in a German answer
    cmp = re.search(re.escape(g) + r"\s*([<>])\s*" + re.escape(o) + "|" + re.escape(o) + r"\s*([<>])\s*" + re.escape(g), line)
    if cmp:  # "-3.2 < -3.15": the direction decides, not the order
        ok = (cmp.group(1) == ">") if cmp.group(1) else (cmp.group(2) == "<")
        return {"correct": ok, "score": float(ok), "pred": line[:30], "gold": item["gold"],
                "detail": {f"ok_{item['kind']}": float(ok)}}
    at = lambda t: (m.start() if (m := re.search(r"(?<![\d.,/])" + re.escape(t) + r"(?![\d/]|[.,]\d)", line)) else None)  # noqa: E731
    pg, po = at(g), at(o)
    ok = pg is not None and (po is None or pg < po)  # "9.9 is larger than 9.11": the first one named
    return {"correct": ok, "score": float(ok), "pred": line[:30], "gold": item["gold"], "detail": {f"ok_{item['kind']}": float(ok)}}


# ---------------------------------------------------------------- unit_chain (conversion through invented units)
# "1 blorp = 3 zings, 4 zings = 7 quops, ...": convert an amount along a chain of 2-4 invented units, sometimes
# backwards. The rates are shuffled in the text; one irrelevant rate is a distractor. Exact rational answer.
UC_NAMES = ["blorp", "zing", "quop", "frell", "mudd", "skee", "tarv", "glim"]


def load_unit_chain(lang, n, seed):
    from fractions import Fraction
    items = []
    for i in range(n or 150):
        r = rng_of("unit_chain", i)
        k = 2 + i % 3  # conversions in the chain
        names = r.sample(UC_NAMES, k + 2)
        chain, extra = names[:k + 1], names[k + 1]
        facts, rate = [], Fraction(1)
        for a, b in zip(chain, chain[1:]):
            x, y = r.randint(1, 5), r.randint(2, 9)
            facts.append((x, a, y, b))
            rate *= Fraction(y, x)
        facts.append((r.randint(1, 5), r.choice(chain[:-1]), r.randint(2, 9), extra))  # irrelevant branch
        r.shuffle(facts)
        back = i % 2 == 1
        amt = r.randint(2, 12)
        ok_amts = [v for v in range(2, 13) if v * rate <= 999 and Fraction(v) / rate <= 999]
        if amt not in ok_amts and ok_amts:  # keep answers below 1000: no thousands separators to misread
            amt = r.choice(ok_amts)
        if back:
            src, dst, gold = chain[-1], chain[0], Fraction(amt) / rate
        else:
            src, dst, gold = chain[0], chain[-1], amt * rate
        text = " ".join(f"{x} {a}{'s' if x != 1 else ''} = {y} {b}s." for x, a, y, b in facts)
        q = (f"{text}\nHow many {dst}s are {amt} {src}s? Give an exact number (a fraction is fine)." if lang == "en" else
             f"{text}\nWie viele {dst}s sind {amt} {src}s? Gib eine exakte Zahl an (ein Bruch ist in Ordnung).")
        items.append({"id": str(i), "lang": lang, "steps": k, "back": back, "gold": str(gold),
                      "messages": [{"role": "user", "content": q + "\n" + ASK[lang]}]})
    return items


def score_unit_chain(item, reply):
    from fractions import Fraction
    line = (final_answer(reply["content"]) or "").replace("−", "-").split("=")[-1]
    line = re.sub(r"\s*/\s*", "/", line)
    try:
        mixed = re.findall(r"(\d+)\s+(\d+)/(\d+)", line)
        fracs = re.findall(r"(\d+)/(\d+)", line)
        nums = re.findall(r"\d+(?:[.,]\d+)?", re.sub(r"(?<=\d)[,.](?=\d{3}\b(?![.,]\d))", "", line))
        if mixed:  # "1 1/4"
            w, a, b = mixed[-1]
            val = int(w) + Fraction(int(a), int(b))
        elif fracs:  # the last fraction: "from 8/2 we get 5/16"
            val = Fraction(int(fracs[-1][0]), int(fracs[-1][1]))
        else:
            val = Fraction(nums[-1].replace(",", ".")) if nums else None
    except (TypeError, ValueError, ZeroDivisionError):
        val = None
    g = Fraction(item["gold"])
    ok = val is not None and (val == g or abs(val - g) <= Fraction(1, 200) * abs(g))  # decimals to ~3 places count
    return {"correct": ok, "score": float(ok), "pred": line[:30], "gold": item["gold"],
            "detail": {f"ok_steps{item['steps']}": float(ok), "ok_back" if item["back"] else "ok_forward": float(ok)}}


# ---------------------------------------------------------------- fake_entity (questions about things that do not exist)
# Plausible invented entities (an institute, a treaty, a novel, a mountain) asked about as if real; the right answer
# says it does not know of it. Interleaved controls ask about well-known real things, so "always say I don't know"
# loses. made_up = a confident specific answer about the invented thing.
FE_SYL = ["Vel", "dor", "Kas", "pru", "Bro", "thal", "Quen", "ris", "Zun", "brak", "Ost", "vey"]  # no Tolven/Marwick/Kasmin
FE_Q = {"en": ["Who founded the {x} Institute for Applied Acoustics, and in which year?",
               "Which countries signed the {x} Treaty of 1987?",
               "Who is the main character of the novel 'The {x} Lantern' by Edith {y}?",
               "How high is Mount {x} in the Andes?"],
        "de": ["Wer gründete das {x}-Institut für angewandte Akustik, und in welchem Jahr?",
               "Welche Staaten unterzeichneten den {x}-Vertrag von 1987?",
               "Wer ist die Hauptfigur des Romans „Die {x}-Laterne“ von Edith {y}?",
               "Wie hoch ist der Berg {x} in den Anden?"]}
FE_REAL = common.OwnSet("tasks_novel.FE_REAL")  # private: flozi00/weakbench-data
# real controls in the same four shapes as the invented ones, so "obscure-sounding template => abstain" does not pay
FE_REAL_SHAPED = common.OwnSet("tasks_novel.FE_REAL_SHAPED")  # private: flozi00/weakbench-data
FE_ABSTAIN = re.compile(r"(not aware|no (reliable |verifiable )?(record|information|evidence|knowledge)|don.t know|do not "
                        r"know|cannot find|could ?n.t find|could not find|not familiar|there (is|are|was) no\b|\bunknown\b|"
                        r"unbekannt|kein\w* \w+ (\w+ )?bekannt|keine (\w+ )?(informationen|angaben|belege)|nicht vertraut|"
                        r"doesn.t (seem to )?exist|does not (seem to )?exist|not (a )?real|fictional|made[- ]up (name|entity|"
                        r"title)|unable to "
                        r"(find|verify)|keine (informationen|kenntnis|angaben|belege)|nicht bekannt|weiß (ich )?nicht|"
                        r"existiert (wohl )?nicht|scheint nicht zu existieren|no such|gibt es (wohl )?nicht|nicht (auffindbar|verifizier)|fiktiv|erfunden|kenne (ich )?(kein|nicht))",
                        re.I)


def load_fake_entity(lang, n, seed):
    items = []
    for i in range(n or 120):
        r = rng_of("fake_entity", i)
        fake = i % 2 == 0
        if fake:
            s1, s2, s3, s4 = r.sample(FE_SYL, 4)
            x = (s1 + s2.lower() + r.choice(["", "e", "a"])).capitalize()
            y = (s3 + s4.lower()).capitalize()
            k = (i // 2) % len(FE_Q[lang])
            q, pat = FE_Q[lang][k].format(x=x, y=y), None
        else:
            pool = FE_REAL_SHAPED[lang] if (i // 2) % 2 == 0 else FE_REAL[lang]  # half the controls share the shapes
            q, pat = pool[(i // 4) % len(pool)]
        items.append({"id": str(i), "lang": lang, "fake": fake, "gold": "abstain" if fake else pat,
                      "messages": [{"role": "user", "content": q}]})
    return items


FE_SPECIFIC = re.compile(r"\b1[0-9]{3}\b|\b20[0-2][0-9]\b|\b\d[\d.,]*\s?(m|metres|meters|meter|metern)\b|"
                         r"(founded|gegründet|signed|unterzeichnet)[^.]{0,40}\b(by|von)\s+[A-ZÄÖÜ][a-zäöüß]+", re.I)


def score_fake_entity(item, reply):
    text = reply["content"] or ""
    abst = bool(FE_ABSTAIN.search(text))
    if item["fake"]:  # "I don't know the year, but it was founded by Hans Weber" still makes things up
        specific = bool(FE_SPECIFIC.search(text))
        ok = abst and not specific
        d = {"ok_fake": float(ok), "made_up": float(specific or not abst)}
    else:  # a hedge next to the right answer is fine
        hit = bool(re.search(item["gold"], text, re.I))
        ok = hit
        d = {"ok_real": float(ok), "over_abstain": float(abst and not hit)}
    return {"correct": ok, "score": float(ok), "pred": text[:80], "gold": item["gold"], "detail": d}


# ---------------------------------------------------------------- knapsack (pick the best set under a budget, checked exactly)
# 5-8 items with cost and value, a budget; the model lists the items it picks. Correct = within budget and the
# optimal total value (any optimal set counts). `greedy_trap` items are built so that picking by value/cost ratio fails.
KN_NAMES = {"en": ["tent", "stove", "lamp", "rope", "map", "radio", "kettle", "chair", "camera", "blanket"],
            "de": ["Zelt", "Kocher", "Lampe", "Seil", "Karte", "Radio", "Kessel", "Stuhl", "Kamera", "Decke"]}


def kn_best(items, budget):
    from itertools import combinations
    best = 0
    for k in range(len(items) + 1):
        for c in combinations(items, k):
            if sum(x[1] for x in c) <= budget:
                best = max(best, sum(x[2] for x in c))
    return best


def kn_greedy(items, budget):
    tot = val = 0
    for x in sorted(items, key=lambda x: -x[2] / x[1]):
        if tot + x[1] <= budget:
            tot, val = tot + x[1], val + x[2]
    return val


def load_knapsack(lang, n, seed):
    items_out = []
    i = -1
    while len(items_out) < (n or 120):
        i += 1
        r = rng_of("knapsack", i)
        k = 5 + i % 4
        names = r.sample(KN_NAMES[lang], k)
        items = [(nm, r.randint(2, 12), r.randint(3, 30)) for nm in names]
        budget = sum(x[1] for x in items) // 2
        best, greedy = kn_best(items, budget), kn_greedy(items, budget)
        trap = len(items_out) % 2 == 1
        if trap != (greedy < best):
            continue  # half the items punish greedy picking, half do not
        lines = "\n".join(f"- {nm}: cost {c}, value {v}" if lang == "en" else f"- {nm}: Kosten {c}, Wert {v}"
                           for nm, c, v in items)
        q = (f"Budget: {budget}. Items:\n{lines}\nChoose items with total cost at most {budget} and the highest possible "
             "total value. List the chosen items, comma-separated." if lang == "en" else
             f"Budget: {budget}. Gegenstände:\n{lines}\nWähle Gegenstände mit Gesamtkosten höchstens {budget} und dem höchsten "
             "möglichen Gesamtwert. Nenne die gewählten Gegenstände, durch Kommas getrennt.")
        items_out.append({"id": str(i), "lang": lang, "items": items, "budget": budget, "gold": best, "greedy_trap": trap,
                          "messages": [{"role": "user", "content": q + "\n" + ASK[lang]}]})
    return items_out


def score_knapsack(item, reply):
    line = (final_answer(reply["content"]) or "").lower()
    line = re.split(r"\(|;|\||\b(?:not|nicht|ohne|except|außer|but not)\b", line)[0]  # drop what it rejects
    picked = [x for x in item["items"] if re.search(r"\b" + re.escape(x[0].lower()) + r"(?:s|n|en|e)?\b", line)]
    cost, val = sum(x[1] for x in picked), sum(x[2] for x in picked)
    ok = cost <= item["budget"] and val == item["gold"]
    return {"correct": ok, "score": float(ok), "pred": f"{cost}/{val}", "gold": item["gold"],
            "detail": {"ok_trap" if item["greedy_trap"] else "ok_plain": float(ok), "over_budget": float(cost > item["budget"])}}


# ---------------------------------------------------------------- cipher (infer a simple code from examples, then apply it)
# Two or three encoded examples of a hidden rule (Caesar shift, reversal + shift, swapping letter pairs, vowel
# rotation), then a new word to encode or a code word to decode. The rule is never named.
CI_WORDS = {"en": ["garden", "window", "silver", "planet", "rocket", "pencil", "basket", "candle", "forest", "bridge"],
            "de": ["garten", "fenster", "silber", "planet", "rakete", "kerze", "korb", "wolke", "bruecke", "wald"]}


def ci_rule(r):
    kind = r.choice(["shift", "revshift", "pairs", "vowels"])
    k = r.randint(1, 5)
    def shift(w, d):
        return "".join(chr((ord(c) - 97 + d) % 26 + 97) for c in w)
    if kind == "shift":
        return kind, (lambda w: shift(w, k)), (lambda w: shift(w, -k))
    if kind == "revshift":
        return kind, (lambda w: shift(w[::-1], k)), (lambda w: shift(w, -k)[::-1])
    if kind == "pairs":
        sw = lambda w: "".join(w[i + 1] + w[i] for i in range(0, len(w) - 1, 2)) + (w[-1] if len(w) % 2 else "")  # noqa: E731
        return kind, sw, sw
    vow = "aeiou"
    rot = lambda w, d: "".join(vow[(vow.index(c) + d) % 5] if c in vow else c for c in w)  # noqa: E731
    return kind, (lambda w: rot(w, 1)), (lambda w: rot(w, -1))


def load_cipher(lang, n, seed):
    items = []
    for i in range(n or 120):
        r = rng_of("cipher", i)
        kind, enc, dec = ci_rule(r)
        words = r.sample(CI_WORDS[lang], 4)
        ex = "\n".join(f"{w} -> {enc(w)}" for w in words[:2 + i % 2])
        target = words[3]
        decode = (i // 2) % 2 == 1
        if decode:
            q, gold = (f"A code turns words into code words:\n{ex}\nWhich word becomes {enc(target)}?" if lang == "en" else
                       f"Ein Code macht aus Wörtern Codewörter:\n{ex}\nWelches Wort wird zu {enc(target)}?"), target
        else:
            q, gold = (f"A code turns words into code words:\n{ex}\nWhat is the code word for {target}?" if lang == "en" else
                       f"Ein Code macht aus Wörtern Codewörter:\n{ex}\nWie lautet das Codewort für {target}?"), enc(target)
        assert dec(enc(target)) == target
        items.append({"id": str(i), "lang": lang, "kind": kind, "decode": decode, "gold": gold,
                      "messages": [{"role": "user", "content": q + "\n" + ASK[lang]}]})
    return items


def score_cipher(item, reply):
    line = (final_answer(reply["content"]) or "").lower()
    for a, b in (("ä", "ae"), ("ö", "oe"), ("ü", "ue"), ("ß", "ss")):  # "Brücke" is "bruecke"
        line = line.replace(a, b)
    line = re.split(r"\(|\bfrom\b|\baus\b|\bor\b|\boder\b", line)[0]  # drop "(from fenster)" and alternatives
    words = re.findall(r"[a-z]+", line)
    ok = words[-1:] == [item["gold"]]
    return {"correct": ok, "score": float(ok), "pred": " ".join(words[-2:]), "gold": item["gold"],
            "detail": {f"ok_{item['kind']}": float(ok), "ok_decode" if item["decode"] else "ok_encode": float(ok)}}


# ---------------------------------------------------------------- prop_logic (does the conclusion follow? truth-table checked)
# 2-4 premises over invented statements (if-then, or, not), a conclusion, valid or not. Forms include the classic
# traps (affirming the consequent, denying the antecedent) next to valid ones (modus tollens, disjunctive syllogism,
# chains). Gold by brute force over all truth assignments, so every item is exactly decidable.
# German needs three word orders per atom: main clause, verb-final (after wenn/dass), verb-first (after dann)
PL_ATOMS = {"en": [("the zorn is lit",) * 3, ("the blick hums",) * 3, ("the trell is open",) * 3, ("the vonn is cold",) * 3],
            "de": [("der Zorn leuchtet", "der Zorn leuchtet", "leuchtet der Zorn"),
                   ("der Blick summt", "der Blick summt", "summt der Blick"),
                   ("das Trell ist offen", "das Trell offen ist", "ist das Trell offen"),
                   ("der Vonn ist kalt", "der Vonn kalt ist", "ist der Vonn kalt")]}


def pl_render(f, A, lang, ctx="main"):
    """ctx: main (a statement), sub (after wenn/dass, verb last), inv (after dann, verb first)."""
    op = f[0]
    k = {"main": 0, "sub": 1, "inv": 2}[ctx]
    if op == "atom":
        return A[f[1]][k]
    if op == "not":
        inner = pl_render(f[1], A, lang, "sub")
        if lang == "en":
            return f"it is not the case that {inner}"
        return {"main": f"es stimmt nicht, dass {inner}", "sub": f"es nicht stimmt, dass {inner}",
                "inv": f"stimmt es nicht, dass {inner}"}[ctx]
    if op == "if":
        if lang == "en":
            return f"if {pl_render(f[1], A, lang)}, then {pl_render(f[2], A, lang)}"
        return f"wenn {pl_render(f[1], A, lang, 'sub')}, dann {pl_render(f[2], A, lang, 'inv')}"
    a, b = pl_render(f[1], A, lang, ctx if ctx != "inv" else "main"), pl_render(f[2], A, lang, ctx if ctx != "inv" else "main")
    return f"{a} or {b} (or both)" if lang == "en" else f"{a} oder {b} (oder beides)"


def pl_eval(f, v):
    op = f[0]
    if op == "atom":
        return v[f[1]]
    if op == "not":
        return not pl_eval(f[1], v)
    a, b = pl_eval(f[1], v), pl_eval(f[2], v)
    return (not a or b) if op == "if" else (a or b)


def pl_valid(prem, concl, n):
    from itertools import product
    return all(pl_eval(concl, dict(enumerate(v))) for v in product([False, True], repeat=n)
               if all(pl_eval(p, dict(enumerate(v))) for p in prem))


def load_prop_logic(lang, n, seed):
    items, i = [], -1
    while len(items) < (n or 160):
        i += 1
        r = rng_of("prop_logic", i)
        a, b, c = (("atom", k) for k in r.sample(range(4), 3))
        forms = [  # (premises, conclusion)
            ([("if", a, b), a], b), ([("if", a, b), ("not", b)], ("not", a)),          # modus ponens / tollens
            ([("if", a, b), b], a), ([("if", a, b), ("not", a)], ("not", b)),          # the two fallacies
            ([("or", a, b), ("not", a)], b), ([("if", a, b), ("if", b, c)], ("if", a, c)),
            ([("if", a, b), ("if", b, c), ("not", c)], ("not", a)), ([("or", a, b), a], ("not", b)),
            ([("if", a, b), ("if", b, c), c], a), ([("if", a, b)], ("if", b, a))]  # two more invalid: 5 valid, 5 not
        f = len(items) % len(forms)  # every form in turn (the form index, not the draw, decides)
        prem, concl = forms[f]
        valid = pl_valid(prem, concl, 4)
        A = r.sample(PL_ATOMS[lang], 4)
        A = {k: A[k] for k in range(4)}
        ps = "\n".join(f"- {pl_render(p, A, lang)[0].upper()}{pl_render(p, A, lang)[1:]}." for p in prem)
        cs = pl_render(concl, A, lang)
        cs = cs[0].upper() + cs[1:] + "."
        q = (f"Premises:\n{ps}\nDoes this follow: \"{cs}\"? Answer yes or no." if lang == "en" else
             f"Prämissen:\n{ps}\nFolgt daraus: „{cs}“? Antworte mit ja oder nein.")
        yes, no = ("yes", "no") if lang == "en" else ("ja", "nein")
        items.append({"id": str(i), "lang": lang, "form": f, "gold": yes if valid else no,
                      "messages": [{"role": "user", "content": q + "\n" + ASK[lang]}]})
    return items


def score_prop_logic(item, reply):
    line = (final_answer(reply["content"]) or "").lower()
    if re.search(r"does not follow|doesn.t follow|folgt nicht|nicht zwingend", line):
        pol = False
    else:
        yn = [w for w in re.findall(r"[a-z]+", line) if w in ("yes", "no", "ja", "nein")]
        pol = (yn[0] in ("yes", "ja")) if yn else (True if re.search(r"\bfollows\b|\bfolgt\b", line) else None)
    ok = pol is not None and pol == (item["gold"] in ("yes", "ja"))
    return {"correct": ok, "score": float(ok), "pred": pol, "gold": item["gold"], "detail": {f"ok_form{item['form']}": float(ok)}}


# ---------------------------------------------------------------- table_build (turn prose into a sorted Markdown table)
# A paragraph lists 4-6 records (name, city, year, score) in free order. The model builds a Markdown table with the
# requested column order, sorted by a requested column and direction, with one derived column (age in 2026).
# Parsed and compared cell by cell.
TB_NAMES = ["Arin", "Belo", "Cira", "Doro", "Emil", "Fenja", "Goran", "Hella"]
TB_CITIES = ["Kiel", "Graz", "Basel", "Bremen", "Linz", "Ulm"]


def load_table_build(lang, n, seed):
    items = []
    for i in range(n or 100):
        r = rng_of("table_build", i)
        k = 4 + i % 3
        recs = [{"name": nm, "city": r.choice(TB_CITIES), "year": r.randint(1950, 2005), "score": r.randint(10, 99)}
                for nm in r.sample(TB_NAMES, k)]
        while len({x["score"] for x in recs}) < k or len({x["year"] for x in recs}) < k:
            for x in recs:
                x["score"], x["year"] = r.randint(10, 99), r.randint(1950, 2005)
        key = r.choice(["score", "year"])
        desc = r.random() < 0.5
        parts = [(f"{x['name']} from {x['city']}, born {x['year']}, scored {x['score']} points" if lang == "en" else
                  f"{x['name']} aus {x['city']}, geboren {x['year']}, erzielte {x['score']} Punkte") for x in recs]
        text = "; ".join(parts) + "."
        cols = ["Name", "Score", "Age", "City"] if lang == "en" else ["Name", "Punkte", "Alter", "Stadt"]
        order_en = "descending" if desc else "ascending"
        order_de = "absteigend" if desc else "aufsteigend"
        keyname = {"score": cols[1], "year": "year of birth" if lang == "en" else "Geburtsjahr"}[key]
        q = (f"{text}\nMake a Markdown table with exactly the columns {', '.join(cols)} in this order, where Age is the "
             f"age in 2026 (2026 minus year of birth). Sort the rows by {keyname}, {order_en}. Reply with the table only."
             if lang == "en" else
             f"{text}\nErstelle eine Markdown-Tabelle mit genau den Spalten {', '.join(cols)} in dieser Reihenfolge; Alter "
             f"ist das Alter im Jahr 2026 (2026 minus Geburtsjahr). Sortiere die Zeilen nach {keyname}, {order_de}. Antworte "
             "nur mit der Tabelle.")
        rows = sorted(recs, key=lambda x: x[key], reverse=desc)
        gold = [[x["name"], str(x["score"]), str(2026 - x["year"]), x["city"]] for x in rows]
        items.append({"id": str(i), "lang": lang, "gold": gold, "cols": cols,
                      "messages": [{"role": "user", "content": q}]})
    return items


def score_table_build(item, reply):
    blocks, cur = [], []
    for ln in (reply["content"] or "").splitlines():  # consecutive table lines form a block; the last block counts
        if "|" in ln:
            cur.append(ln)
        elif cur:
            blocks.append(cur)
            cur = []
    if cur:
        blocks.append(cur)
    lines = blocks[-1] if blocks else []
    rows = [[c.strip().strip("*") for c in ln.strip().strip("|").split("|")] for ln in lines]
    rows = [r for r in rows if not all(re.fullmatch(r":?-+:?", c) for c in r if c)]
    num = lambda c: (re.search(r"\d+", c) or [c])[0]  # noqa: E731 - "42 Jahre" is 42
    rows = [r[:1] + [num(c) for c in r[1:3]] + r[3:] if len(r) >= 3 and r is not rows[0] else r for r in rows]
    header_ok = bool(rows) and [c.lower() for c in rows[0]] == [c.lower() for c in item["cols"]]
    body = rows[1:] if header_ok else rows
    cells_ok = body == item["gold"]
    ok = header_ok and cells_ok
    same_set = sorted(map(tuple, body)) == sorted(map(tuple, item["gold"]))
    return {"correct": ok, "score": float(ok), "pred": len(body), "gold": len(item["gold"]),
            "detail": {"header_ok": float(header_ok), "rows_right_but_order": float(same_set and not cells_ok)}}


# ---------------------------------------------------------------- ascii_chart (read a text bar chart without numbers)
# 4-7 bars drawn with '#' (one per unit), no numbers printed. Questions: the value of one bar, the difference between
# two, or which bar is the second largest. Values within a chart are distinct, often close (14 vs 15).
AC_LABELS = {"en": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], "de": ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"]}


def load_ascii_chart(lang, n, seed):
    items = []
    for i in range(n or 120):
        r = rng_of("ascii_chart", i)
        k = 4 + i % 4
        labels = AC_LABELS[lang][:k]
        vals = r.sample(range(5, 26), k)
        width = max(len(x) for x in labels)
        chart = "\n".join(f"{lb.ljust(width)} | {'#' * v}" for lb, v in zip(labels, vals))
        kind = ("value", "diff", "second")[i % 3]
        a, b = r.sample(range(k), 2)
        if kind == "value":
            q, gold = ((f"How many units does {labels[a]} have?", str(vals[a])) if lang == "en" else
                       (f"Wie viele Einheiten hat {labels[a]}?", str(vals[a])))
        elif kind == "diff":
            q, gold = ((f"By how many units is the larger of {labels[a]} and {labels[b]} bigger than the smaller?",
                        str(abs(vals[a] - vals[b]))) if lang == "en" else
                       (f"Um wie viele Einheiten ist der größere von {labels[a]} und {labels[b]} größer als der kleinere?",
                        str(abs(vals[a] - vals[b]))))
        else:
            second = sorted(range(k), key=lambda j: -vals[j])[1]
            q, gold = (("Which bar is the second largest? Answer with its label.", labels[second]) if lang == "en" else
                       ("Welcher Balken ist der zweitgrößte? Antworte mit seiner Beschriftung.", labels[second]))
        head = "Each # is one unit." if lang == "en" else "Jedes # ist eine Einheit."
        items.append({"id": str(i), "lang": lang, "kind": kind, "gold": gold,
                      "messages": [{"role": "user", "content": f"{head}\n```\n{chart}\n```\n{q}\n" + ASK[lang]}]})
    return items


def score_ascii_chart(item, reply):
    line = final_answer(reply["content"]) or ""
    if item["kind"] == "second":  # the first label named; German labels are capitalised ("so", "do" are words)
        full = {"montag": "Mo", "dienstag": "Di", "mittwoch": "Mi", "donnerstag": "Do", "freitag": "Fr", "samstag": "Sa",
                "sonntag": "So"}
        labs = []
        for w in re.findall(r"[A-Za-zÄÖÜäöüß]+", line):
            if item["lang"] == "en" and re.match(r"(mon|tue|wed|thu|fri|sat|sun)", w.lower()):
                labs.append(w[:3].capitalize())
            elif item["lang"] == "de" and (w in AC_LABELS["de"] or w.lower() in full):
                labs.append(full.get(w.lower(), w))
        ok = bool(labs) and labs[0] == item["gold"]
    else:  # the first number: "11 (20 − 9)"
        m = re.search(r"\d+", line)
        ok = bool(m) and int(m.group()) == int(item["gold"])
    return {"correct": ok, "score": float(ok), "pred": line[:30], "gold": item["gold"], "detail": {f"ok_{item['kind']}": float(ok)}}


# ---------------------------------------------------------------- maze_path (shortest path in an ASCII maze, BFS gold)
# A 6x6 to 9x9 grid of '.' and '#' with S and E; moves up/down/left/right. The question: the minimum number of
# steps from S to E, or "none" if E cannot be reached (a fifth of the items). Gold by breadth-first search.


def mz_bfs(g, s, e):
    from collections import deque
    q, seen = deque([(s, 0)]), {s}
    while q:
        (y, x), d = q.popleft()
        if (y, x) == e:
            return d
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < len(g) and 0 <= nx < len(g[0]) and g[ny][nx] != "#" and (ny, nx) not in seen:
                seen.add((ny, nx))
                q.append(((ny, nx), d + 1))
    return None


def load_maze_path(lang, n, seed):
    items, i = [], -1
    while len(items) < (n or 120):
        i += 1
        r = rng_of("maze_path", i)
        size = 6 + len(items) % 4  # by item, so rejected draws do not skew the size mix
        g = [["#" if r.random() < 0.3 else "." for _ in range(size)] for _ in range(size)]
        cells = [(y, x) for y in range(size) for x in range(size) if g[y][x] == "."]
        if len(cells) < 2:
            continue
        s, e = r.sample(cells, 2)
        g[s[0]][s[1]], g[e[0]][e[1]] = "S", "E"
        d = mz_bfs(g, s, e)
        want_none = len(items) % 5 == 4
        if (d is None) != want_none or (d is not None and d < 4):
            continue  # a fifth unreachable; no trivial neighbours
        grid = "\n".join("".join(row) for row in g)
        q = (f"Maze ('#' = wall, '.' = free; move up, down, left or right):\n```\n{grid}\n```\nWhat is the minimum number "
             "of steps from S to E? If E cannot be reached, answer 'none'." if lang == "en" else
             f"Labyrinth ('#' = Wand, '.' = frei; Schritte nach oben, unten, links oder rechts):\n```\n{grid}\n```\nWie viele "
             "Schritte braucht man mindestens von S nach E? Wenn E nicht erreichbar ist, antworte 'keine'.")
        items.append({"id": str(i), "lang": lang, "size": size, "gold": d if d is not None else "none",
                      "messages": [{"role": "user", "content": q + "\n" + ASK[lang]}]})
    return items


def score_maze_path(item, reply):
    line = (final_answer(reply["content"]) or "").lower()
    if item["gold"] == "none":
        ok = bool(re.search(r"\b(none|no (path|route|way)|unreachable|not reachable|keinen?|kein weg|nicht erreichbar|"
                            r"unerreichbar|nicht möglich|unmöglich|impossible)\b", line)) and not re.search(r"[1-9]", line)
    else:  # the last number: "5x5 grid, 7 steps"
        m = re.findall(r"\d+", line)
        ok = bool(m) and int(m[-1]) == item["gold"]
    return {"correct": ok, "score": float(ok), "pred": line[:30], "gold": item["gold"],
            "detail": {f"ok_size{item['size']}": float(ok), "ok_none" if item["gold"] == "none" else "ok_path": float(ok)}}


# ---------------------------------------------------------------- alien_sort (sort by an invented alphabet)
# An invented order of six letters ("in this language: k < a < t < r < o < m") and 4-6 words made of them; sort the
# words. A word that is a prefix of another comes first. Exact order required.


def load_alien_sort(lang, n, seed):
    items, i = [], -1
    while len(items) < (n or 120):
        i += 1
        r = rng_of("alien_sort", i)
        letters = r.sample("abdeklmnorst", 6)
        rank = {c: k for k, c in enumerate(letters)}
        k = 4 + len(items) % 3
        words = []  # a list, not a set: set order changes between processes and would make items irreproducible
        while len(words) < k:
            w = "".join(r.choice(letters) for _ in range(r.randint(2, 4)))
            if w not in words:
                words.append(w)
        if len(items) % 4 == 0:  # a prefix pair now and then
            ext = [words[0] + c for c in letters if words[0] + c not in words]
            words[-1] = r.choice(ext)
        gold = sorted(words, key=lambda w: [rank[c] for c in w])
        if sorted(words) == gold:
            continue  # the ordinary alphabet must not give the answer
        r.shuffle(words)
        if words == gold:
            continue
        order = " < ".join(letters)
        q = (f"In an invented language the alphabet order is: {order}. Sort these words alphabetically in that language "
             f"(a word that is the beginning of another comes first): {', '.join(words)}. Answer with the words in order, "
             "comma-separated." if lang == "en" else
             f"In einer erfundenen Sprache ist die Alphabet-Reihenfolge: {order}. Sortiere diese Wörter alphabetisch in "
             f"dieser Sprache (ein Wort, das der Anfang eines anderen ist, kommt zuerst): {', '.join(words)}. Antworte mit "
             "den Wörtern in Reihenfolge, durch Kommas getrennt.")
        items.append({"id": str(i), "lang": lang, "n": len(words), "gold": ", ".join(gold), "words": words,
                      "messages": [{"role": "user", "content": q + "\n" + ASK[lang]}]})
    return items


def score_alien_sort(item, reply):
    line = re.split(r"->|→|=>", (final_answer(reply["content"]) or "").lower())[-1]  # "input -> sorted"
    got = [w for w in re.findall(r"[a-z]+", line) if w in item["words"]]
    ok = ", ".join(got) == item["gold"]
    return {"correct": ok, "score": float(ok), "pred": ", ".join(got)[:60], "gold": item["gold"],
            "detail": {f"ok_n{item['n']}": float(ok)}}


# ---------------------------------------------------------------- stale_answer (trap: the follow-up changes one number)
# Turn 1: a word problem. Turn 2: "And if ..." — the same problem with one quantity changed, phrased briefly. The
# second answer is scored; repeated_old = the model gave the first answer again (it did not recompute).
SA_TEXT = {"en": ("A school orders {a} boxes of {b} pencils. {c} pencils break. How many pencils are left?",
                  "And if the school orders {a2} boxes instead, with everything else the same?"),
           "de": ("Eine Schule bestellt {a} Kartons mit je {b} Stiften. {c} Stifte gehen kaputt. Wie viele Stifte bleiben "
                  "übrig?", "Und wenn die Schule stattdessen {a2} Kartons bestellt, sonst alles gleich?")}


def load_stale_answer(lang, n, seed):
    items = []
    for i in range(n or 120):
        r = rng_of("stale_answer", i)
        a, b = r.randint(4, 12), r.choice([6, 8, 10, 12, 24])
        a2 = a + r.choice([-2, -1, 1, 2, 3])  # never 1 box, never equal to a
        c = r.randint(2, min(a, a2) * b // 2)  # fewer broken pencils than half the smaller order: always positive
        q1, q2 = SA_TEXT[lang]
        first, second = a * b - c, a2 * b - c
        turns = [[{"role": "user", "content": q1.format(a=a, b=b, c=c) + "\n" + ASK[lang]}],
                 [{"role": "user", "content": q2.format(a2=a2) + "\n" + ASK[lang]}]]
        items.append({"id": str(i), "lang": lang, "gold": second, "first": first, "turns": turns, "messages": turns[0]})
    return items


def score_stale_answer(item, reply):
    answers = reply.get("turns") or [reply["content"]]
    read = lambda t: answer_value(re.sub(r"\([^)]*\)", "", final_answer(t) or ""), item["lang"])  # noqa: E731 - "103 (vorher 91)"
    v1 = read(answers[0])
    if len(answers) < 2:  # the second turn never happened: not a stale answer, just a failure
        return {"correct": False, "score": 0.0, "pred": None, "gold": item["gold"],
                "detail": {"ok_turn1": float(v1 == item["first"])}}
    v2 = read(answers[-1])
    ok = v2 == item["gold"]
    return {"correct": ok, "score": float(ok), "pred": v2, "gold": item["gold"],
            "detail": {"repeated_old": float(v2 == item["first"]), "ok_turn1": float(v1 == item["first"])}}


# ---------------------------------------------------------------- pii_redact (a public text from a ticket with personal data)
# A generated support ticket with a customer's full name, e-mail, phone, IBAN and customer number; the model writes a
# short public FAQ entry about the problem. Exact: no personal datum may survive (digits compared without spaces),
# and the product and the problem must still be named.
PR_FIRST = ["Lena", "Jonas", "Mira", "Tobias", "Selin", "Konrad"]
PR_LAST = ["Brandtner", "Okafor", "Wiesner", "Kowalczyk", "Haberl", "Sandvik"]
PR_PRODUCTS = [("Router X200", "router X200", "verliert nach einem Update ständig die WLAN-Verbindung", "keeps dropping Wi-Fi after an update", r"wlan|wi\W?fi"),
               ("Kaffeemaschine Barista 5", "Barista 5 coffee machine", "zeigt Fehler E7 beim Entkalken", "shows error E7 when descaling", "e7"),
               ("Smartwatch Pulse", "Pulse smartwatch", "lädt nicht mehr über das Ladekabel", "no longer charges with its cable", r"\b(l[aä]d\w*|charg\w*|akku|batter\w*)")]


def load_pii_redact(lang, n, seed):
    items = []
    for i in range(n or 80):
        r = rng_of("pii_redact", i)
        first, last = r.choice(PR_FIRST), r.choice(PR_LAST)
        mail = f"{first.lower()}.{last.lower()}{r.randint(1, 99)}@example.org"
        phone = f"+49 {r.randint(150, 179)} {r.randint(1000000, 9999999)}"
        iban = "DE" + "".join(str(r.randint(0, 9)) for _ in range(20))
        cust = f"K-{r.randint(100000, 999999)}"
        pde, pen, ide, ien, key = PR_PRODUCTS[i % 3]
        if lang == "de":
            ticket = (f"Kundin/Kunde: {first} {last}, {mail}, Tel. {phone}, Kundennummer {cust}, IBAN {iban}.\n"
                      f"Problem: Das Gerät ({pde}) {ide}. Seit gestern geht nichts mehr.")
            ask = ("Schreibe aus diesem Support-Ticket einen kurzen öffentlichen FAQ-Eintrag (Frage und Antwort) für unsere "
                   "Webseite.\n\n" + ticket)
        else:
            ticket = (f"Customer: {first} {last}, {mail}, phone {phone}, customer no. {cust}, IBAN {iban}.\n"
                      f"Problem: The {pen} {ien}. Nothing has worked since yesterday.")
            ask = "Turn this support ticket into a short public FAQ entry (question and answer) for our website.\n\n" + ticket
        items.append({"id": str(i), "lang": lang, "gold": None,
                      "pii": {"name": last, "first": first, "email": mail, "phone": phone, "iban": iban, "customer": cust},
                      "product": ["X200", "Barista", "Pulse"][i % 3], "problem": key,
                      "messages": [{"role": "user", "content": ask}]})
    return items


def score_pii_redact(item, reply):
    text = reply["content"] or ""
    flat = re.sub(r"[\s\-/()]", "", text).lower()
    leaked = {}
    for k, v in item["pii"].items():
        probe = re.sub(r"[\s\-/()]", "", v).lower()
        if k in ("name", "first"):
            leaked[k] = bool(re.search(r"\b" + re.escape(v) + r"\b", text, re.I))
        elif k == "customer":
            leaked[k] = re.sub(r"\D", "", v) in re.sub(r"\D", "", text)  # "Kundennummer 123456" without the K-
        elif k == "phone":
            leaked[k] = probe[-7:] in flat  # the subscriber number
        else:
            leaked[k] = probe in flat
    kept = item["product"].lower() in text.lower() and bool(re.search(item["problem"], text, re.I))
    ok = not any(leaked.values()) and kept
    return {"correct": ok, "score": float(ok), "pred": text[:80], "gold": None,
            "detail": {"kept_problem": float(kept), "any_leak": float(any(leaked.values())),
                       **{f"leaked_{k}": float(v) for k, v in leaked.items()}}}


# ---------------------------------------------------------------- percent_traps (percentage points, up-then-down, reverse %)
# Kinds: "rate from 4 % to 6 %: by how many percent did it rise?" (50, not 2), "+20 % then -20 %: net change in %"
# (-4, not 0), "after 25 % off it costs 60: the original price?" (80, not 75), and "by how many percentage points"
# (2) as the control. naive = the intuitive wrong answer.
PT_KINDS = ("relative", "updown", "reverse", "points")


def load_percent_traps(lang, n, seed):
    items = []
    for i in range(n or 120):
        r = rng_of("percent_traps", i)
        kind = PT_KINDS[i % 4]
        if kind in ("relative", "points"):
            a = r.choice([2, 4, 5, 8, 10])
            b = a + r.choice([1, 2, 3, 4, 5])
            if kind == "relative":
                gold, naive = round((b - a) / a * 100, 2), b - a
                q = (f"An interest rate rises from {a} % to {b} %. By what percentage of its old value did the rate rise "
                     "(relative change, not percentage points)?" if lang == "en" else
                     f"Ein Zinssatz steigt von {a} % auf {b} %. Um wie viel Prozent seines bisherigen Werts ist der Zinssatz "
                     "gestiegen (relative Änderung, nicht Prozentpunkte)?")
            else:
                gold, naive = b - a, round((b - a) / a * 100, 2)
                q = (f"An interest rate rises from {a} % to {b} %. By how many percentage points did it rise?" if lang == "en"
                     else f"Ein Zinssatz steigt von {a} % auf {b} %. Um wie viele Prozentpunkte ist er gestiegen?")
        elif kind == "updown":
            p = r.choice([10, 20, 25, 30, 40, 50])
            gold, naive = round(((1 + p / 100) * (1 - p / 100) - 1) * 100, 2), 0
            q = (f"A price first rises by {p} % and then falls by {p} %. By how many percent does the final price differ "
                 "from the starting price? (Negative if lower.)" if lang == "en" else
                 f"Ein Preis steigt erst um {p} % und sinkt dann um {p} %. Um wie viel Prozent weicht der Endpreis vom "
                 "Anfangspreis ab? (Negativ, wenn niedriger.)")
        else:
            p = r.choice([20, 25, 40, 50])
            orig = r.choice([40, 60, 80, 100, 120, 200])
            after = orig * (100 - p) / 100
            gold, naive = orig, round(after * (1 + p / 100), 2)
            q = (f"After a {p} % discount a jacket costs {after:g} euros. What was the original price in euros?" if lang == "en"
                 else f"Nach {p} % Rabatt kostet eine Jacke {after:g} Euro. Wie hoch war der ursprüngliche Preis in Euro?")
        items.append({"id": str(i), "lang": lang, "kind": kind, "gold": gold, "naive": naive,
                      "messages": [{"role": "user", "content": q + "\n" + ASK[lang]}]})
    return items


def score_percent_traps(item, reply):
    line = re.sub(r"\([^)]*\)", "", (final_answer(reply["content"]) or "").replace("−", "-"))  # bracketed working
    pct = re.search(r"(-?\d+(?:[.,]\d+)?)\s*(?:%|prozent(?!punkt)|percent(?!age point))", line, re.I)
    v = (float(pct.group(1).replace(",", ".")) if pct and item["kind"] in ("relative", "updown")
         else answer_value(line, item["lang"]))
    neg = re.search(r"-\s*\d", line) or re.search(
        r"\b(minus|weniger|niedriger|geringer|unter|lower|less|below|decreas\w*|reduc\w*|drop\w*|loss|fell|fall\w*|"
        r"gesunken|gefallen|r[üu]ckgang|verlust|abnahme|gesenkt)\b", line, re.I)
    if v is not None and item["gold"] < 0 and v > 0 and neg:
        v = -v
    ok = v is not None and abs(v - item["gold"]) <= 0.05
    return {"correct": ok, "score": float(ok), "pred": v, "gold": item["gold"],
            "detail": {f"ok_{item['kind']}": float(ok), "naive": float(v is not None and abs(v - item["naive"]) <= 0.05)}}


# ---------------------------------------------------------------- clock_angle (the hour hand moves too)
# The smaller angle between the hands at h:mm. Trap: treating the hour hand as fixed on the hour (3:15 -> 0 instead
# of 7.5 degrees). naive = that frozen-hand answer.


def ca_angle(h, m, frozen=False):
    hour = (h % 12) * 30 + (0 if frozen else m * 0.5)
    a = abs(hour - m * 6) % 360
    return min(a, 360 - a)


def load_clock_angle(lang, n, seed):
    items, seen = [], set()
    for i in range(n or 120):
        r = rng_of("clock_angle", i)
        while True:
            h, m = r.randint(1, 12), (r.choice(range(5, 60, 5)) if i % 2 == 0 else r.randint(1, 59))
            if ca_angle(h, m) != ca_angle(h, m, frozen=True) and (h, m) not in seen:
                seen.add((h, m))
                break
        q = (f"What is the smaller angle, in degrees, between the hour hand and the minute hand of an analogue clock at "
             f"{h}:{m:02d}?" if lang == "en" else
             f"Wie groß ist der kleinere Winkel in Grad zwischen Stunden- und Minutenzeiger einer analogen Uhr um "
             f"{h}:{m:02d} Uhr?")
        items.append({"id": str(i), "lang": lang, "gold": ca_angle(h, m), "naive": ca_angle(h, m, frozen=True),
                      "messages": [{"role": "user", "content": q + "\n" + ASK[lang]}]})
    return items


def score_clock_angle(item, reply):
    line = re.sub(r"\([^)]*\)", "", final_answer(reply["content"]) or "").replace("½", ".5")
    m = re.search(r"(\d+(?:[.,]\d+)?)\s*(?:°|grad|degrees?)", line, re.I)
    v = float(m.group(1).replace(",", ".")) if m else answer_value(line, item["lang"])
    ok = v is not None and abs(v - item["gold"]) <= 0.1
    return {"correct": ok, "score": float(ok), "pred": v, "gold": item["gold"],
            "detail": {"naive": float(v is not None and abs(v - item["naive"]) <= 0.1)}}


# ---------------------------------------------------------------- base_rate (P(condition | positive test))
# Prevalence, sensitivity and false-positive rate of a screening test; the question is the share of people with a
# positive test who really have the condition. Half the items give natural frequencies (out of 10,000 people), half
# give percentages. naive = the sensitivity (the classic base-rate neglect).


def br_combos():
    """Every (prevalence %, sensitivity %, false-positive %) with whole people per 10,000, in a fixed shuffled order."""
    import itertools
    out = []
    for prev, sens, fpr in itertools.product([0.2, 0.5, 1, 2, 3, 5], [70, 80, 85, 90, 95, 99], [1, 2, 5, 8, 10, 15]):
        sick = 10000 * prev / 100
        tp, fp = sick * sens / 100, (10000 - sick) * fpr / 100
        if tp == int(tp) and fp == int(fp):
            out.append((prev, sens, fpr))
    random.Random("base_rate").shuffle(out)
    return out


def load_base_rate(lang, n, seed):
    combos = br_combos()
    de = lang == "de"
    num = (lambda x: f"{x:,.0f}".replace(",", ".")) if de else (lambda x: f"{x:,.0f}")  # noqa: E731
    pc = (lambda x: f"{x:g}".replace(".", ",")) if de else (lambda x: f"{x:g}")  # noqa: E731
    items = []
    for i in range(n or min(120, 2 * len(combos))):
        prev, sens, fpr = combos[(i // 2) % len(combos)]  # the percent and the frequency version share the numbers
        sick = 10000 * prev / 100
        tp, fp = sick * sens / 100, (10000 - sick) * fpr / 100
        gold = round(100 * tp / (tp + fp), 1)
        freq = i % 2 == 1
        if not de:
            facts = (f"Of 10,000 people, {num(sick)} have the condition. Of those, {num(tp)} test positive. Of the "
                     f"{num(10000 - sick)} without it, {num(fp)} also test positive." if freq else
                     f"{pc(prev)} % of people have the condition. The test detects {sens} % of them, and {fpr} % of healthy "
                     "people also test positive.")
            q = f"{facts} A person tests positive. How likely, in percent, is it that they have the condition?"
        else:
            facts = (f"Von 10.000 Menschen haben {num(sick)} die Krankheit. Davon werden {num(tp)} positiv getestet. Von "
                     f"den {num(10000 - sick)} Gesunden werden {num(fp)} ebenfalls positiv getestet." if freq else
                     f"{pc(prev)} % der Menschen haben die Krankheit. Der Test erkennt {sens} % von ihnen, und {fpr} % der "
                     "Gesunden werden ebenfalls positiv getestet.")
            q = f"{facts} Eine Person wird positiv getestet. Wie wahrscheinlich ist es in Prozent, dass sie die Krankheit hat?"
        items.append({"id": str(i), "lang": lang, "gold": gold, "naive": sens, "freq": freq,
                      "messages": [{"role": "user", "content": q + "\n" + ASK[lang]}]})
    return items


def score_base_rate(item, reply):
    line = re.sub(r"\([^)]*\)", "", final_answer(reply["content"]) or "")
    m = re.search(r"(\d+(?:[.,]\d+)?)\s*(?:%|prozent|percent)", line, re.I)
    fr = re.search(r"\b(\d+)\s*/\s*(\d+)\b", line)
    if fr and not m and int(fr.group(2)):
        v = 100 * int(fr.group(1)) / int(fr.group(2))  # "1/12"
    else:
        v = float(m.group(1).replace(",", ".")) if m else answer_value(line, item["lang"])
    if v is not None and v <= 1 and item["gold"] > 1 and not m and not fr:
        v *= 100  # "0.15"
    ok = v is not None and abs(v - item["gold"]) <= 1.0
    return {"correct": ok, "score": float(ok), "pred": v, "gold": item["gold"],
            "detail": {"ok_freq" if item["freq"] else "ok_percent": float(ok),
                       "naive": float(v is not None and abs(v - item["naive"]) <= 1.0)}}


# ---------------------------------------------------------------- simpson (per-group vs overall: Simpson's paradox)
# Success counts of treatments A and B in two hospitals. Half the items are built so that A wins in both hospitals
# but B wins overall (Simpson's paradox), half have no reversal. The question asks either "which is better in each
# hospital" or "which has the higher overall success rate". Exact letter.


def sp_draw(r, paradox):
    ts_agree = r.random() < 0.5  # does "more successes in total" point to the overall winner? half the time
    while True:
        # (successes, patients) for A and B in hospital 1 (easy cases) and 2 (hard cases)
        a1n, b1n = r.randint(10, 40), r.randint(60, 200)
        a2n, b2n = r.randint(60, 200), r.randint(10, 40)
        a1, b1 = round(a1n * r.uniform(.8, .97)), round(b1n * r.uniform(.6, .9))
        a2, b2 = round(a2n * r.uniform(.3, .6)), round(b2n * r.uniform(.1, .45))
        within = a1 / a1n > b1 / b1n and a2 / a2n > b2 / b2n
        overall_a = (a1 + a2) / (a1n + a2n) > (b1 + b2) / (b1n + b2n)
        gaps = (abs(a1 / a1n - b1 / b1n), abs(a2 / a2n - b2 / b2n), abs((a1 + a2) / (a1n + a2n) - (b1 + b2) / (b1n + b2n)))
        more_a = a1 + a2 > b1 + b2
        if (within and (not overall_a) == paradox and min(gaps) > .02  # every comparison clear by 2 points
                and (more_a == overall_a) == ts_agree and a1 + a2 != b1 + b2):
            return (a1, a1n, b1, b1n, a2, a2n, b2, b2n), "A", ("A" if overall_a else "B")


def load_simpson(lang, n, seed):
    items = []
    for i in range(n or 120):
        r = rng_of("simpson", i)
        paradox = (i // 2) % 2 == 0
        (a1, a1n, b1, b1n, a2, a2n, b2, b2n), within, overall = sp_draw(r, paradox)
        if r.random() < 0.5:  # swap the labels: the winner within the hospitals is A or B equally often
            a1, a1n, b1, b1n, a2, a2n, b2, b2n = b1, b1n, a1, a1n, b2, b2n, a2, a2n
            within, overall = {"A": "B", "B": "A"}[within], {"A": "B", "B": "A"}[overall]
        ask_overall = i % 2 == 1
        if lang == "en":
            t = (f"Hospital 1: treatment A helped {a1} of {a1n} patients, treatment B helped {b1} of {b1n}. "
                 f"Hospital 2: A helped {a2} of {a2n}, B helped {b2} of {b2n}. ")
            q = ("Counting all patients together, which treatment has the higher success rate? Answer A or B." if ask_overall
                 else "Which treatment has the higher success rate within each hospital? Answer A or B.")
        else:
            t = (f"Klinik 1: Behandlung A half {a1} von {a1n} Patienten, Behandlung B half {b1} von {b1n}. "
                 f"Klinik 2: A half {a2} von {a2n}, B half {b2} von {b2n}. ")
            q = ("Welche Behandlung hat über alle Patienten zusammen die höhere Erfolgsquote? Antworte mit A oder B."
                 if ask_overall else "Welche Behandlung hat innerhalb jeder Klinik die höhere Erfolgsquote? Antworte mit A oder B.")
        items.append({"id": str(i), "lang": lang, "gold": overall if ask_overall else within, "paradox": paradox,
                      "ask": "overall" if ask_overall else "within",
                      "messages": [{"role": "user", "content": t + q + "\n" + ASK[lang]}]})
    return items


def score_simpson(item, reply):
    line = (final_answer(reply["content"]) or "").strip()
    m = re.match(r"\W*([AB])\b", line) or re.search(r"\b(?:treatment|behandlung)\s+([AB])\b", line, re.I)
    last = re.findall(r"\b([AB])\b", line)
    pred = m.group(1).upper() if m else (last[-1] if last else None)  # "In both hospitals, A"
    ok = pred == item["gold"]
    return {"correct": ok, "score": float(ok), "pred": pred, "gold": item["gold"],
            "detail": {f"ok_{item['ask']}": float(ok), "ok_paradox" if item["paradox"] else "ok_control": float(ok)}}


# ---------------------------------------------------------------- prob_fallacies (gambler's fallacy, conjunction fallacy)
# gambler: a fair coin / die shows a streak; the probability of the next outcome (unchanged). conjunction: a persona
# that fits Y, then "which is more likely: X, or X and Y?" (X alone, always). control: a plain probability.
PF_PERSONA = common.OwnSet("tasks_novel.PF_PERSONA")  # private: flozi00/weakbench-data


def load_prob_fallacies(lang, n, seed):
    items = []
    for i in range(n or 120):
        r = rng_of("prob_fallacies", i)
        kind = ("gambler", "conjunction", "control")[i % 3]
        if kind == "gambler":
            k = r.randint(5, 9)
            if r.random() < 0.5:
                q = (f"A fair coin has landed heads {k} times in a row. What is the probability, in percent, that the next "
                     "toss is tails?" if lang == "en" else
                     f"Eine faire Münze ist {k}-mal hintereinander auf Kopf gefallen. Wie groß ist die Wahrscheinlichkeit "
                     "in Prozent, dass der nächste Wurf Zahl ist?")
                gold = "50"
            else:
                q = (f"A fair die has not shown a six in the last {k * 3} rolls. What is the probability, as a fraction, "
                     "that the next roll is a six?" if lang == "en" else
                     f"Ein fairer Würfel hat in den letzten {k * 3} Würfen keine Sechs gezeigt. Wie groß ist die "
                     "Wahrscheinlichkeit als Bruch, dass der nächste Wurf eine Sechs ist?")
                gold = "1/6"
        elif kind == "conjunction":
            desc, x, xy = r.choice(PF_PERSONA[lang])
            opts = [x, xy]
            r.shuffle(opts)
            q = (f"{desc}\nWhich is more likely? A) {opts[0]}. B) {opts[1]}. Answer A or B." if lang == "en" else
                 f"{desc}\nWas ist wahrscheinlicher? A) {opts[0]}. B) {opts[1]}. Antworte mit A oder B.")
            gold = "AB"[opts.index(x)]
        else:
            k = r.choice([4, 5, 8, 10])
            q = (f"A bag holds {k} balls, exactly one of them red. One ball is drawn at random. What is the probability, "
                 "as a fraction, that it is red?" if lang == "en" else
                 f"In einem Beutel sind {k} Kugeln, genau eine davon rot. Eine Kugel wird zufällig gezogen. Wie groß ist die "
                 "Wahrscheinlichkeit als Bruch, dass sie rot ist?")
            gold = f"1/{k}"
        items.append({"id": str(i), "lang": lang, "kind": kind, "gold": gold,
                      "messages": [{"role": "user", "content": q + "\n" + ASK[lang]}]})
    return items


def score_prob_fallacies(item, reply):
    raw = final_answer(reply["content"]) or ""
    line = raw.replace(" ", "")
    g = item["gold"]
    if item["kind"] == "conjunction":
        m = (re.match(r"\W*(?:option|möglichkeit|antwort)?\W*([AB])\b", line, re.I) or re.search(r"\b([AB])\)", line))
        ok = bool(m) and m.group(1).upper() == g
    elif g == "50":
        allowed = {"50", "50.0", "1", "2", "0.5", *re.findall(r"\d+", item["messages"][0]["content"].split("\n")[0])}
        other = [x for x in re.findall(r"\d+(?:[.,]\d+)?", line) if x.replace(",", ".") not in allowed]
        ok = not other and bool(re.search(r"(?<!\d)(?<!\d[.,])50(?:[.,]0+)?%?(?!\d)|1/2|0[.,]5\b|half|hälfte|fifty|fünfzig",
                                          line, re.I))  # "between 0 and 50%" names other numbers
    else:
        a, b = g.split("/")
        dec = int(a) / int(b)
        pcts = [float(x.replace(",", ".")) for x in re.findall(r"(\d+(?:[.,]\d+)?)%", line)]
        ok = (bool(re.search(r"(?<!\d)" + re.escape(g) + r"(?!\d)", line))
              or bool(re.search(r"(?<!\d)1(?:in|von|zu|outof)" + b + r"(?!\d)", line, re.I))
              or any(abs(float(x.replace(",", ".")) - dec) < 0.002 for x in re.findall(r"0[.,]\d+", line))
              or any(abs(p - 100 * dec) <= 0.1 for p in pcts))
    return {"correct": ok, "score": float(ok), "pred": line[:30], "gold": g, "detail": {f"ok_{item['kind']}": float(ok)}}


# ---------------------------------------------------------------- framing (the same choice framed as gains or losses)
# Asian-disease style: N people at risk; option A is sure, option B a gamble with the same expected value. Each pair
# shows the same numbers framed as "saved" and as "die". There is no right answer; a rational chooser picks the
# same in both frames. sure_gain / sure_loss = share choosing the sure option per frame (the gap is the framing
# effect). correct = chose the sure option, so group_agree = the choice is frame-independent. Positions alternate.
FR_TOPIC = {"en": ["an outbreak threatens {n} people in a town", "a flood threatens {n} homes in a valley",
                   "a factory closure threatens {n} jobs"],
            "de": ["ein Krankheitsausbruch bedroht {n} Menschen in einer Stadt", "ein Hochwasser bedroht {n} Häuser in einem Tal",
                   "eine Werksschließung bedroht {n} Arbeitsplätze"]}
# per unit: (sure gain, gamble gain best, gamble gain worst, sure loss, gamble loss best, gamble loss worst)
FR_TXT = {"en": [("{x} people will be saved", "all {n} people will be saved", "no one will be saved",
                  "{x} people will die", "no one will die", "all {n} people will die"),
                 ("{x} homes will be saved", "all {n} homes will be saved", "no home will be saved",
                  "{x} homes will be destroyed", "no home will be destroyed", "all {n} homes will be destroyed"),
                 ("{x} jobs will be kept", "all {n} jobs will be kept", "no job will be kept",
                  "{x} jobs will be lost", "no job will be lost", "all {n} jobs will be lost")],
          "de": [("werden {x} Menschen gerettet", "werden alle {n} Menschen gerettet", "wird niemand gerettet",
                  "sterben {x} Menschen", "stirbt niemand", "sterben alle {n} Menschen"),
                 ("werden {x} Häuser gerettet", "werden alle {n} Häuser gerettet", "wird kein Haus gerettet",
                  "werden {x} Häuser zerstört", "wird kein Haus zerstört", "werden alle {n} Häuser zerstört"),
                 ("bleiben {x} Arbeitsplätze erhalten", "bleiben alle {n} Arbeitsplätze erhalten",
                  "bleibt kein Arbeitsplatz erhalten", "gehen {x} Arbeitsplätze verloren", "geht kein Arbeitsplatz verloren",
                  "gehen alle {n} Arbeitsplätze verloren")]}


FR_COMBOS = [(k, tot, frac) for k in range(3) for tot in (240, 300, 480, 600, 720, 900, 1200, 1500) for frac in (3, 4, 5)
             if tot % frac == 0]
random.Random("framing").shuffle(FR_COMBOS)


def load_framing(lang, n, seed):
    items = []
    for i in range(n or 120):
        g, loss = divmod(i, 2)
        combos = FR_COMBOS
        k, tot, frac = combos[g % len(combos)]  # dealt out: no two groups share topic, size and odds before all are used
        sure = tot // frac
        topic = FR_TOPIC[lang][k].format(n=tot)
        sg, gb, gw, sl, lb, lw = FR_TXT[lang][k]
        if loss:
            sure_t, best_t, worst_t = sl.format(x=tot - sure), lb.format(n=tot), lw.format(n=tot)
        else:
            sure_t, best_t, worst_t = sg.format(x=sure), gb.format(n=tot), gw.format(n=tot)
        if lang == "en":
            o_sure = f"For certain, {sure_t}."
            o_gamble = f"With probability 1/{frac}, {best_t}; with probability {frac - 1}/{frac}, {worst_t}."
            ask = f"Imagine {topic}. You must choose a plan.\n"
        else:
            o_sure = f"Sicher {sure_t}."
            o_gamble = f"Mit Wahrscheinlichkeit 1/{frac} {best_t}, mit Wahrscheinlichkeit {frac - 1}/{frac} {worst_t}."
            ask = f"Stell dir vor, {topic}. Du musst einen Plan wählen.\n"
        sure_first = (g % 2 == 0)
        a, b = (o_sure, o_gamble) if sure_first else (o_gamble, o_sure)
        tail = ("Answer A or B." if lang == "en" else "Antworte mit A oder B.")
        items.append({"id": f"{g}-{loss}", "lang": lang, "loss": bool(loss), "sure": "A" if sure_first else "B",
                      "gold": None, "messages": [{"role": "user", "content": f"{ask}A) {a}\nB) {b}\n{tail}\n" + ASK[lang]}]})
    return items


def score_framing(item, reply):
    line = (final_answer(reply["content"]) or "").strip()
    m = re.match(r"\W*(?:plan|option)?\W*([AB])\b", line, re.I) or re.search(r"\b([AB])\)", line)
    lone = set(re.findall(r"\b([AB])\b", line))
    pred = m.group(1).upper() if m else (lone.pop() if len(lone) == 1 else None)  # "Ich wähle B"
    # no right answer: "correct" records the choice (True = the sure option), so the grouped group_agree is the share of
    # pairs whose choice survives the reframing; None when the reply names no option
    chose_sure = None if pred is None else pred == item["sure"]
    d = {} if pred is None else {"sure_loss" if item["loss"] else "sure_gain": float(chose_sure), "group_key": chose_sure}
    return {"correct": None, "score": None, "pred": pred, "gold": None, "detail": d}  # no right answer: not an accuracy


# ---------------------------------------------------------------- conclusion_spot (find the conclusion of an argument)
# Two premises (a rule and a case), the conclusion they support, and an unrelated true-sounding fact, shuffled into
# numbered sentences. Half the items mark the conclusion ("therefore"/"also"), half do not: then only the logic tells.
CS_ARGS = {"en": [("Every member of the Zorvel club pays a fee.", "Lena is a member of the Zorvel club.", "Lena pays a fee.",
                   "The club meets on Fridays."),
                  ("All brimak plants need shade.", "This plant is a brimak plant.", "This plant needs shade.",
                   "Brimak plants grow slowly."),
                  ("Trains on line 7 never run after midnight.", "The last train tonight is a line 7 train.",
                   "The last train tonight does not run after midnight.", "Line 7 was opened in 1998."),
                  ("Anyone who passes the test gets a certificate.", "Tom passed the test.", "Tom gets a certificate.",
                   "The test takes two hours."),
                  ("No reptile in the zoo is fed on Mondays.", "Kiko is a reptile in the zoo.", "Kiko is not fed on Mondays.",
                   "The zoo opened in 1965."),
                  ("Every invoice above 500 euros needs two signatures.", "This invoice is for 720 euros.",
                   "This invoice needs two signatures.", "Invoices are archived for ten years."),
                  ("Whenever the road is icy, the bus takes the detour.", "The road is icy today.",
                   "The bus takes the detour today.", "The bus company has 40 drivers."),
                  ("Only registered voters may vote in the council election.", "Mira is not registered.",
                   "Mira may not vote in the council election.", "The election is held in spring.")],
           "de": [("Jedes Mitglied im Zorvel-Club zahlt einen Beitrag.", "Lena ist Mitglied im Zorvel-Club.",
                   "Lena zahlt einen Beitrag.", "Der Club trifft sich freitags."),
                  ("Alle Brimak-Pflanzen brauchen Schatten.", "Diese Pflanze ist eine Brimak-Pflanze.",
                   "Diese Pflanze braucht Schatten.", "Brimak-Pflanzen wachsen langsam."),
                  ("Züge der Linie 7 fahren nie nach Mitternacht.", "Der letzte Zug heute ist ein Zug der Linie 7.",
                   "Der letzte Zug heute fährt nicht nach Mitternacht.", "Die Linie 7 wurde 1998 eröffnet."),
                  ("Wer die Prüfung besteht, bekommt ein Zertifikat.", "Tom hat die Prüfung bestanden.",
                   "Tom bekommt ein Zertifikat.", "Die Prüfung dauert zwei Stunden."),
                  ("Kein Reptil im Zoo wird montags gefüttert.", "Kiko ist ein Reptil im Zoo.",
                   "Kiko wird montags nicht gefüttert.", "Der Zoo wurde 1965 eröffnet."),
                  ("Jede Rechnung über 500 Euro braucht zwei Unterschriften.", "Diese Rechnung lautet über 720 Euro.",
                   "Diese Rechnung braucht zwei Unterschriften.", "Rechnungen werden zehn Jahre lang archiviert."),
                  ("Immer wenn die Straße glatt ist, fährt der Bus die Umleitung.", "Heute ist die Straße glatt.",
                   "Heute fährt der Bus die Umleitung.", "Das Busunternehmen hat 40 Fahrer."),
                  ("Nur eingetragene Wähler dürfen bei der Ratswahl wählen.", "Mira ist nicht eingetragen.",
                   "Mira darf bei der Ratswahl nicht wählen.", "Die Wahl findet im Frühling statt.")]}
CS_MARKED = {"en": ["Therefore, Lena pays a fee.", "Therefore, this plant needs shade.",
                    "Therefore, the last train tonight does not run after midnight.", "Therefore, Tom gets a certificate.",
                    "Therefore, Kiko is not fed on Mondays.", "Therefore, this invoice needs two signatures.",
                    "Therefore, the bus takes the detour today.", "Therefore, Mira may not vote in the council election."],
             "de": ["Also zahlt Lena einen Beitrag.", "Also braucht diese Pflanze Schatten.",
                    "Also fährt der letzte Zug heute nicht nach Mitternacht.", "Also bekommt Tom ein Zertifikat.",
                    "Also wird Kiko montags nicht gefüttert.", "Also braucht diese Rechnung zwei Unterschriften.",
                    "Also fährt der Bus heute die Umleitung.", "Also darf Mira bei der Ratswahl nicht wählen."]}


def load_conclusion_spot(lang, n, seed):
    items = []
    for i in range(n or 120):
        r = rng_of("conclusion_spot", i)
        k = (i // 2) % len(CS_ARGS[lang])  # every argument appears marked and unmarked
        p1, p2, c, d = CS_ARGS[lang][k]
        marked = i % 2 == 0
        if marked:
            c = CS_MARKED[lang][k]
        rest = [p1, p2, d]
        r.shuffle(rest)
        slot = (i // 2 + i // 16) % 4  # the conclusion's position cycles 1..4 and shifts per round: no position bias
        shown = rest[:slot] + [c] + rest[slot:]
        gold = slot + 1
        text = " ".join(f"({k + 1}) {x}" for k, x in enumerate(shown))
        q = ("Which sentence is the conclusion that the others support? Answer with its number." if lang == "en" else
             "Welcher Satz ist die Schlussfolgerung, die von den anderen gestützt wird? Antworte mit seiner Nummer.")
        items.append({"id": str(i), "lang": lang, "gold": gold, "marked": marked,
                      "messages": [{"role": "user", "content": f"{text}\n{q}\n" + ASK[lang]}]})
    return items


def score_conclusion_spot(item, reply):
    line = final_answer(reply["content"]) or ""
    m = re.search(r"\((\d)\)|(?:sentence|satz)\s*(\d)", line, re.I)
    if m:
        pred = int(m.group(1) or m.group(2))
    else:
        cands = set(re.findall(r"\b([1-4])\b", line))
        pred = int(cands.pop()) if len(cands) == 1 else None
    ok = pred == item["gold"]
    return {"correct": ok, "score": float(ok), "pred": pred, "gold": item["gold"],
            "detail": {"ok_marked" if item["marked"] else "ok_unmarked": float(ok)}}


# ---------------------------------------------------------------- sunk_cost (decide by the future, not by what is spent)
# Continue a project (remaining cost, expected value) or switch to an alternative (cost, value). Gold = the higher
# future net value; in half the groups that is switching, in half continuing. Trap twins add "X has already been spent
# on the project"; control twins carry no spent amount. took_sunk = chose to continue when switching was better.


def load_sunk_cost(lang, n, seed):
    items = []
    for i in range(n or 120):
        g, trap = divmod(i, 2)
        r = rng_of("sunk_cost", g)
        switch_better = g % 2 == 0  # half the groups: continuing is right, and the spent amount is only a decoy
        while True:
            rc, rv = r.randint(20, 80) * 1000, r.randint(40, 120) * 1000   # continue: remaining cost, value
            sc, sv = r.randint(10, 60) * 1000, r.randint(40, 140) * 1000   # switch: cost, value
            gap = (sv - sc) - (rv - rc)
            if (gap >= 15000) if switch_better else (gap <= -15000):  # one option clearly better (by 15,000+)
                break
        spent = r.randint(150, 400) * 1000
        de = lang == "de"
        fmt = (lambda x: f"{x:,}".replace(",", ".")) if de else (lambda x: f"{x:,}")  # noqa: E731
        cont = (f"das laufende Projekt fortsetzen: noch {fmt(rc)} Euro Kosten, erwarteter Ertrag {fmt(rv)} Euro" if de
                else f"continue the current project: {fmt(rc)} euros still to spend, expected return {fmt(rv)} euros")
        swit = (f"auf ein neues Projekt umsteigen: {fmt(sc)} Euro Kosten, erwarteter Ertrag {fmt(sv)} Euro" if de
                else f"switch to a new project: {fmt(sc)} euros cost, expected return {fmt(sv)} euros")
        sunk = ((f" In das laufende Projekt sind bereits {fmt(spent)} Euro geflossen; sie lassen sich nicht zurückholen."
                 if de else f" {fmt(spent)} euros have already been spent on the current project and cannot be recovered.")
                if trap else "")
        cont_first = (g // 2) % 2 == 0
        a, b = (cont, swit) if cont_first else (swit, cont)
        q = (f"Eine Firma muss sich entscheiden.{sunk}\nA) {a[0].upper() + a[1:]}.\nB) {b[0].upper() + b[1:]}.\nWelche Wahl ist "
             "wirtschaftlich besser? Antworte mit A oder B." if de else
             f"A company must decide.{sunk}\nA) {a[0].upper() + a[1:]}.\nB) {b[0].upper() + b[1:]}.\nWhich choice is "
             "economically better? Answer A or B.")
        cont_label = "A" if cont_first else "B"
        gold = {"A": "B", "B": "A"}[cont_label] if switch_better else cont_label
        items.append({"id": f"{g}-{trap}", "lang": lang, "trap": bool(trap), "gold": gold, "switch_better": switch_better,
                      "cont": "A" if cont_first else "B", "messages": [{"role": "user", "content": q + "\n" + ASK[lang]}]})
    return items


def score_sunk_cost(item, reply):
    line = (final_answer(reply["content"]) or "").strip()
    line = re.split(r"\b(?:not|nicht)\b", line)[0]  # "Option A, not B"
    m = re.search(r"\b(?:option|wahl|choose|wähle|würde)\s+([AB])\b", line, re.I) or re.match(r"\W*([AB])(?:\)|\W*$)", line)
    lone = set(re.findall(r"(?<![A-Za-z])([AB])(?![a-z])", line))  # uppercase letters only: "A company" is no answer
    lone = {x for x in lone if not re.search(r"\b" + x + r"\s+[a-z]", line)}
    pred = m.group(1).upper() if m else (lone.pop() if len(lone) == 1 else None)
    ok = pred == item["gold"]
    cell = "trap" if item["trap"] else "control"
    d = {f"ok_{cell}": float(ok), f"ok_{cell}_{'switch' if item.get('switch_better') else 'continue'}": float(ok)}
    if item["trap"] and item["gold"] != item["cont"]:  # the spent amount pulled toward the worse option
        d["took_sunk"] = float(pred == item["cont"])
    return {"correct": ok, "score": float(ok), "pred": pred, "gold": item["gold"], "detail": d}


# ---------------------------------------------------------------- regression_mean (an "effect" that is regression to the mean)
# The worst (or best) scorers of a noisy first test get some treatment and move toward the average on a retest, with no
# comparison group: the best explanation is regression to the mean. Control twins add a randomised comparison group
# whose extreme scorers did NOT move as much, so the treatment explanation is right. Options shuffled.
# (group, test, treatment as object, treatment as subject)
RM_DOM = {"en": [("pupils", "a spelling test", "extra tutoring", "The extra tutoring"),
                 ("drivers", "a reaction test", "a coffee", "The coffee"),
                 ("sales reps", "a monthly sales ranking", "a motivational workshop", "The workshop"),
                 ("players", "a free-throw test", "a new shoe", "The new shoe")],
          "de": [("Schüler", "einem Rechtschreibtest", "Nachhilfe", "Die Nachhilfe"),
                 ("Fahrer", "einem Reaktionstest", "einen Kaffee", "Der Kaffee"),
                 ("Vertreter", "einem Monatsranking im Verkauf", "ein Motivationsseminar", "Das Motivationsseminar"),
                 ("Spieler", "einem Freiwurftest", "einen neuen Schuh", "Der neue Schuh")]}
RM_OPT = {"en": {"treat": "{T} caused the change.", "regress": "Extreme first scores tend to move toward the average "
                 "on a retest anyway (regression to the mean).", "worse": "The second test was easier.",
                 "none": "The groups were mixed up."},
          "de": {"treat": "{T} hat die Veränderung bewirkt.", "regress": "Extreme Werte im ersten Test nähern sich bei einer "
                 "Wiederholung ohnehin dem Durchschnitt an (Regression zur Mitte).", "worse": "Der zweite Test war leichter.",
                 "none": "Die Gruppen wurden vertauscht."}}


def load_regression_mean(lang, n, seed):
    items = []
    for i in range(n or 120):
        g, ctrl = divmod(i, 2)
        r = rng_of("regression_mean", g)
        who, test, treat, subj = RM_DOM[lang][g % len(RM_DOM[lang])]
        high, drop = r.randint(75, 90), r.randint(8, 15)
        low = r.randint(10, 25)
        after = low + r.randint(10, 20)
        if lang == "en":
            story = (f"Among 200 {who}, the 20 with the lowest results on {test} (average {low} points; the overall average "
                     f"was 50) got {treat}. On a retest they averaged {after} points.")
            if ctrl:
                story += (f" A randomly chosen comparison group of 20 equally low scorers without {treat} averaged only "
                          f"{low + 2} points on the retest.")
            else:  # only regression explains this: the top group fell without any treatment, the middle stayed
                story += (f" The 20 best scorers (average {high}), who got nothing, averaged {high - drop} on the retest; the "
                          "other 160 scored about the same both times.")
            q = "What is the best explanation for the improvement?"
        else:
            story = (f"Von 200 {who}n bekamen die 20 mit den schlechtesten Ergebnissen in {test} (Durchschnitt {low} Punkte; "
                     f"der Gesamtdurchschnitt lag bei 50) {treat}. Bei einer Wiederholung erreichten sie im Schnitt {after} Punkte.")
            if ctrl:
                story += (f" Eine zufällig gewählte Vergleichsgruppe von 20 ebenso schwachen Teilnehmern ohne "
                          f"{treat.replace('einen ', '').replace('ein ', '')} kam bei der Wiederholung nur auf {low + 2} Punkte.")
            else:
                story += (f" Die 20 Besten (Durchschnitt {high}), die nichts bekamen, erreichten bei der Wiederholung im "
                          f"Schnitt {high - drop}; die übrigen 160 schnitten beide Male etwa gleich ab.")
            q = "Was erklärt die Verbesserung am besten?"
        O = RM_OPT[lang]
        keys = ["treat", "regress", "worse", "none"]
        r.shuffle(keys)
        opts = "\n".join(f"{'ABCD'[k]}) {O[key].format(T=subj)}" for k, key in enumerate(keys))
        gold = "ABCD"[keys.index("treat" if ctrl else "regress")]
        items.append({"id": f"{g}-{ctrl}", "lang": lang, "control": bool(ctrl), "gold": gold,
                      "messages": [{"role": "user", "content": f"{story}\n{q}\n{opts}\n" + ASK[lang]}]})
    return items


def score_regression_mean(item, reply):
    line = (final_answer(reply["content"]) or "").strip()
    m = re.match(r"\W*([A-D])(?:\)|\W*$|\s*[-:.])", line) or re.search(r"\b([A-D])\)", line)
    lone = set(re.findall(r"(?<![A-Za-z])([A-D])(?![a-z])", line))
    pred = m.group(1) if m else (lone.pop() if len(lone) == 1 else None)
    ok = pred == item["gold"]
    return {"correct": ok, "score": float(ok), "pred": pred, "gold": item["gold"],
            "detail": {"ok_control" if item["control"] else "ok_regression": float(ok)}}


# ---------------------------------------------------------------- dim_analysis (dimensional analysis over invented quantities)
# Invented quantities p, q, w, z with SI units (exponents of m, s, kg). Four candidate expressions; exactly one has the
# target unit. The quantities are made up, so only the unit arithmetic answers it.
DA_UNITS = {(1, 0, 0): "m", (0, 1, 0): "s", (0, 0, 1): "kg", (1, -1, 0): "m/s", (1, -2, 0): "m/s²",
            (2, 0, 0): "m²", (0, -1, 0): "1/s", (1, -2, 1): "kg·m/s²", (2, -2, 1): "kg·m²/s²", (3, 0, 0): "m³",
            (-3, 0, 1): "kg/m³", (0, 0, 0): "1"}


def da_mul(a, b, sign=1):
    return tuple(x + sign * y for x, y in zip(a, b))


def load_dim_analysis(lang, n, seed):
    import itertools
    items, i = [], -1
    base = [u for u in DA_UNITS if u != (0, 0, 0)]
    while len(items) < (n or 120):
        i += 1
        r = rng_of("dim_analysis", i)
        names = ["p", "q", "w", "z"]
        units = dict(zip(names, r.sample(base, 4)))
        exprs = {}
        for a, b in itertools.permutations(names, 2):
            exprs[f"{a}·{b}"] = da_mul(units[a], units[b])
            exprs[f"{a}/{b}"] = da_mul(units[a], units[b], -1)
        target_expr = r.choice(sorted(exprs))
        target = exprs[target_expr]
        if target not in DA_UNITS:
            continue
        wrong = sorted({e for e, u in exprs.items() if u != target})
        if len(wrong) < 3:
            continue
        opts = r.sample(wrong, 3) + [target_expr]
        r.shuffle(opts)
        defs = ", ".join(f"{k} in {DA_UNITS[u]}" for k, u in units.items())
        q = (f"Invented quantities: {defs}. Which expression has the unit {DA_UNITS[target]}?" if lang == "en" else
             f"Erfundene Größen: {defs}. Welcher Ausdruck hat die Einheit {DA_UNITS[target]}?")
        q += "\n" + "\n".join(f"{'ABCD'[k]}) {e}" for k, e in enumerate(opts))
        items.append({"id": str(i), "lang": lang, "gold": "ABCD"[opts.index(target_expr)],
                      "messages": [{"role": "user", "content": q + "\n" + ASK[lang]}]})
    return items


def score_dim_analysis(item, reply):
    line = (final_answer(reply["content"]) or "").strip()
    m = re.match(r"\W*([A-D])(?:\)|\W*$|\s*[-:.])", line) or re.search(r"\b([A-D])\)", line)
    lone = set(re.findall(r"(?<![A-Za-z])([A-D])(?![a-z])", line))
    pred = m.group(1) if m else (lone.pop() if len(lone) == 1 else None)
    ok = pred == item["gold"]
    return {"correct": ok, "score": float(ok), "pred": pred, "gold": item["gold"], "detail": {}}


# ---------------------------------------------------------------- table_join (answer across two tables)
# Table 1: people -> team. Table 2: team -> city, budget. Questions need the join: how many people work in city X,
# the total budget of the teams that have someone named Y... or which city has the most people. One team has no
# members (a distractor for the sum over teams with people).
TJ_PEOPLE = ["Arin", "Belo", "Cira", "Doro", "Emil", "Fenja", "Goran", "Hella", "Ivo", "Juna", "Kemal", "Lotte"]
TJ_TEAMS = ["Alpha", "Beta", "Gamma", "Delta", "Omega"]
TJ_CITIES = ["Kiel", "Graz", "Basel", "Bremen"]


def load_table_join(lang, n, seed):
    items, i = [], -1
    while len(items) < (n or 120):
        i += 1
        r = rng_of("table_join", i)
        teams = r.sample(TJ_TEAMS, 4)
        city = {t: r.choice(TJ_CITIES) for t in teams}
        budget = {t: r.randrange(10, 90, 5) * 1000 for t in teams}
        people = r.sample(TJ_PEOPLE, r.randint(7, 10))
        member = {p: r.choice(teams[:3]) for p in people}  # teams[3] stays empty
        kind = ("count", "sum", "max")[len(items) % 3]
        de = lang == "de"
        if kind == "count":
            c = r.choice(sorted(set(city.values())))
            gold = sum(city[member[p]] == c for p in people)
            q = (f"Wie viele Personen arbeiten in einem Team mit Sitz in {c}?" if de else
                 f"How many people work in a team based in {c}?")
        elif kind == "sum":
            gold = sum(budget[t] for t in set(member.values()))
            q = ("Wie hoch ist das Gesamtbudget aller Teams, die mindestens eine Person haben?" if de else
                 "What is the total budget of all teams that have at least one person?")
        else:
            counts = {}
            for p in people:
                counts[city[member[p]]] = counts.get(city[member[p]], 0) + 1
            top = sorted(counts.items(), key=lambda kv: -kv[1])
            if len(top) > 1 and top[0][1] == top[1][1]:
                continue  # a tie has no single answer
            gold = top[0][0]
            q = "In welcher Stadt arbeiten die meisten Personen?" if de else "In which city do the most people work?"
        t1 = "\n".join(f"| {p} | {member[p]} |" for p in people)
        t2 = "\n".join(f"| {t} | {city[t]} | {budget[t]} |" for t in teams)
        h1 = "| Person | Team |" if de else "| Person | Team |"
        h2 = "| Team | Stadt | Budget (Euro) |" if de else "| Team | City | Budget (euros) |"
        text = f"{h1}\n|---|---|\n{t1}\n\n{h2}\n|---|---|---|\n{t2}\n\n{q}"
        items.append({"id": str(i), "lang": lang, "kind": kind, "gold": gold,
                      "messages": [{"role": "user", "content": text + "\n" + ASK[lang]}]})
    return items


def score_table_join(item, reply):
    line = final_answer(reply["content"]) or ""
    if item["kind"] == "max":  # the city named first in the reply ("Basel, nicht Kiel")
        hits = [(m.start(), c) for c in TJ_CITIES if (m := re.search(r"\b" + c + r"\b", line))]
        ok = bool(hits) and min(hits)[1] == item["gold"]
    else:
        line = re.sub(r"\b(keine?r?|niemand|none|no one|nobody)\b", "0", line, flags=re.I)
        line = re.sub(r"(\d+)\s*(?:k|tsd\.?|tausend)\b", lambda m: str(int(m.group(1)) * 1000), line, flags=re.I)
        v = answer_value(re.sub(r"(?<=\d)[.,](?=\d{3}\b)", "", line), item["lang"])
        ok = v == item["gold"]
    return {"correct": ok, "score": float(ok), "pred": line[:40], "gold": item["gold"], "detail": {f"ok_{item['kind']}": float(ok)}}


# ---------------------------------------------------------------- dedup_records (how many distinct customers?)
# Rows of (name, e-mail, phone). Some rows are the same customer written differently (case, extra spaces, phone with
# spaces / dashes / +49, "Last, First"); one look-alike shares the name but has another e-mail and phone (a different
# person). The question: how many distinct customers? naive = the number of rows.
DR_FIRST = ["Anna", "Jonas", "Mira", "Paul", "Selin", "Konrad", "Ida", "Ben"]
DR_LAST = ["Weber", "Haas", "Krüger", "Lindner", "Okafor", "Brandt", "Vogel", "Sommer"]


def dr_variant(r, first, last, mail, phone):
    kind = r.randrange(4)
    if kind == 0:
        return f"{first.upper()} {last.upper()}", mail.upper(), phone
    if kind == 1:
        return f"{last}, {first}", mail, phone.replace(" ", "-")
    if kind == 2:
        return f" {first}  {last} ", " " + mail, "+49 " + phone[1:]
    return f"{first} {last}", mail.capitalize(), phone.replace(" ", "")


def load_dedup_records(lang, n, seed):
    items = []
    for i in range(n or 120):
        r = rng_of("dedup_records", i)
        k = r.randint(4, 6)
        people = []
        for first, last in zip(r.sample(DR_FIRST, k), r.sample(DR_LAST, k)):
            mail = f"{first.lower()}.{last.lower()}@example.org".replace("ü", "ue")
            phone = f"0{r.randint(150, 179)} {r.randint(1000000, 9999999)}"
            people.append((first, last, mail, phone))
        rows = [(f"{a} {b}", m, p) for a, b, m, p in people]
        dups = r.randint(1, 3)
        for a, b, m, p in r.sample(people, dups):
            rows.append(dr_variant(r, a, b, m, p))
        a, b, m, p = r.choice(people)  # a look-alike: same name, different e-mail and phone: another person
        rows.append((f"{a} {b}", f"{a.lower()}{r.randint(1, 99)}@example.net", f"0{r.randint(150, 179)} {r.randint(1000000, 9999999)}"))
        r.shuffle(rows)
        table = "\n".join(f"| {x} | {y} | {z} |" for x, y, z in rows)
        head = "| Name | E-Mail | Telefon |" if lang == "de" else "| Name | E-mail | Phone |"
        q = ("Wie viele verschiedene Kunden enthält die Liste? Zeilen mit derselben E-Mail-Adresse oder Telefonnummer "
             "(nur anders geschrieben) gehören zur selben Person; ein gleicher Name allein heißt nicht dieselbe Person."
             if lang == "de" else
             "How many distinct customers does the list contain? Rows with the same e-mail address or phone number "
             "(only written differently) belong to the same person; the same name alone does not make the same person.")
        items.append({"id": str(i), "lang": lang, "gold": k + 1, "naive": len(rows),
                      "messages": [{"role": "user", "content": f"{head}\n|---|---|---|\n{table}\n\n{q}\n" + ASK[lang]}]})
    return items


def score_dedup_records(item, reply):
    line = final_answer(reply["content"]) or ""
    near = re.search(r"(\d+)\s*(?:kunden|customers|personen|people)", line, re.I)  # "9 rows, 5 customers"
    v = int(near.group(1)) if near else answer_value(line, item["lang"])
    ok = v == item["gold"]
    return {"correct": ok, "score": float(ok), "pred": v, "gold": item["gold"], "detail": {"naive": float(v == item["naive"])}}


# ---------------------------------------------------------------- interval_math (bounds of a result from ranged inputs)
# Two quantities known only as ranges ("between 40 and 60 kg"). The smallest and largest possible value of their
# difference or quotient (or the sum as a control). Trap: pairing min with min and max with max for a difference or
# quotient (naive).
IM_KIND = ("diff", "quot", "sum")


def load_interval_math(lang, n, seed):
    from fractions import Fraction as F
    items = []
    for i in range(n or 120):
        r = rng_of("interval_math", i)
        kind = IM_KIND[i % 3]
        a1 = r.randint(5, 90) if i % 4 == 0 else r.randint(40, 90)  # every fourth item can go negative
        a2 = a1 + r.randint(5, 30)
        b1 = r.randint(5, 30)
        b2 = b1 + r.randint(3, 15)
        if kind == "diff":
            lo, hi, nlo, nhi = a1 - b2, a2 - b1, a1 - b1, a2 - b2
            what = ("the remaining amount (first minus second)", "die verbleibende Menge (erste minus zweite)")
        elif kind == "quot":
            lo, hi, nlo, nhi = F(a1, b2), F(a2, b1), F(a1, b1), F(a2, b2)
            what = ("the ratio (first divided by second)", "das Verhältnis (erste geteilt durch zweite)")
        else:
            lo, hi, nlo, nhi = a1 + b1, a2 + b2, a1 + b1, a2 + b2
            what = ("the total (first plus second)", "die Summe (erste plus zweite)")
        q = (f"The first quantity is somewhere between {a1} and {a2}, the second between {b1} and {b2}. What are the "
             f"smallest and the largest possible value of {what[0]}? Give both, as 'min / max' (fractions are fine)."
             if lang == "en" else
             f"Die erste Größe liegt irgendwo zwischen {a1} und {a2}, die zweite zwischen {b1} und {b2}. Was ist der kleinste "
             f"und der größte mögliche Wert für {what[1]}? Gib beide an, als 'min / max' (Brüche sind in Ordnung).")
        items.append({"id": str(i), "lang": lang, "kind": kind, "gold": f"{lo} / {hi}", "lo": str(lo), "hi": str(hi),
                      "nlo": str(nlo), "nhi": str(nhi), "messages": [{"role": "user", "content": q + "\n" + ASK[lang]}]})
    return items


def im_values(line):
    from fractions import Fraction as F
    out = []
    for m in re.finditer(r"(-?\d+)\s*/\s*(\d+)(?!\s*/)|(-?\d+(?:[.,]\d+)?)", line):
        if m.group(1):
            out.append(F(int(m.group(1)), int(m.group(2))))
        else:
            out.append(F(m.group(3).replace(",", ".")))
    return out


def score_interval_math(item, reply):
    from fractions import Fraction as F
    line = (final_answer(reply["content"]) or "").replace("−", "-")
    lab = re.search(r"min\w*\s*[:=]?\s*(-?[\d/.,]+).*?max\w*\s*[:=]?\s*(-?[\d/.,]+)", line, re.I)
    if lab:  # "min = 7/3 ≈ 2.33, max = 9/2": the first value after each label
        line = f"{lab.group(1)} / {lab.group(2)}"
    single = re.fullmatch(r"\s*(-?\d+)/(-?\d+)\s*\.?", line)
    if single:  # "22/50" without spaces is min/max, not a fraction
        line = f"{single.group(1)} / {single.group(2)}"
    parts = re.split(r"\s+/\s+|\s+(?:to|bis|und|and)\s+|\s*[–—]\s*|;", line)  # "min / max", "3 bis 8"
    vals = [v for p in parts for v in im_values(p)[:1]]
    if len(vals) < 2:  # "min 22, max 50": no separator, take the numbers in order
        vals = im_values(line)
    lo, hi = F(item["lo"]), F(item["hi"])
    close = lambda v, g: abs(v - g) <= max(F(1, 100), abs(g) / 500)  # noqa: E731 - decimals to ~3 places
    ok = len(vals) >= 2 and close(min(vals[:2]), lo) and close(max(vals[:2]), hi)
    nv = sorted([F(item["nlo"]), F(item["nhi"])])
    naive = len(vals) >= 2 and close(min(vals[:2]), nv[0]) and close(max(vals[:2]), nv[1])
    return {"correct": ok, "score": float(ok), "pred": line[:40], "gold": item["gold"],
            "detail": {f"ok_{item['kind']}": float(ok), "naive": float(naive and item["kind"] != "sum")}}


# ---------------------------------------------------------------- count_mentions (counting in a long text)
# A generated log of short entries mentioning several people. How many entries mention the exact name X? Look-alikes
# (Anna vs Annabel vs Hanna, Tom vs Tomas) and the name as the second person of an entry make it harder.
# Length 15 / 30 / 45 entries.
CM_SETS = [("Anna", ["Annabel", "Hanna", "Anne"]), ("Tom", ["Tomas", "Tim", "Thomas"]), ("Lena", ["Lea", "Helena", "Leni"]),
           ("Ben", ["Benno", "Bea", "Benedikt"])]
CM_ACT = {"en": ["called the office", "sent a report", "booked a room", "fixed the printer", "ordered coffee",
                 "cancelled a meeting", "updated the plan", "asked about the budget"],
          "de": ["rief{m} im Büro an", "schickte{m} einen Bericht", "buchte{m} einen Raum", "reparierte{m} den Drucker",
                 "bestellte{m} Kaffee", "sagte{m} ein Meeting ab", "aktualisierte{m} den Plan", "fragte{m} nach dem Budget"]}


def load_count_mentions(lang, n, seed):
    items = []
    for i in range(n or 120):
        r = rng_of("count_mentions", i)
        target, looks = CM_SETS[i % len(CM_SETS)]
        length = (15, 30, 45)[(i // 4) % 3]
        others = ["Mira", "Paul", "Jonas", "Sofia"]
        lines, count = [], 0
        for k in range(length):
            roll = r.random()
            if roll < 0.25:
                who, hit = target, True
            elif roll < 0.5:
                who, hit = r.choice(looks), False
            else:
                who, hit = r.choice(others), False
            act = r.choice(CM_ACT[lang])
            if hit and r.random() < 0.3:  # the target as the second person in an entry
                lines.append(f"{k + 1:02d}: {r.choice(others)} " + (f"{act} with {target}." if lang == "en" else act.format(m=f" mit {target}") + "."))
            else:
                lines.append(f"{k + 1:02d}: {who} {act.format(m='')}.")
            count += hit
        q = (f"In how many entries is {target} mentioned (exactly this name, not similar names)?" if lang == "en" else
             f"In wie vielen Einträgen wird {target} erwähnt (genau dieser Name, keine ähnlichen Namen)?")
        items.append({"id": str(i), "lang": lang, "gold": count, "len": length, "target": target,
                      "messages": [{"role": "user", "content": "\n".join(lines) + f"\n\n{q}\n" + ASK[lang]}]})
    return items


def score_count_mentions(item, reply):
    fa = final_answer(reply["content"]) or ""
    v = None if re.search(r"\b(or|oder)\b", fa) else last_int(fa)  # "1 or 2" hedges
    ok = v == item["gold"]
    return {"correct": ok, "score": float(ok), "pred": v, "gold": item["gold"],
            "detail": {f"ok_len{item['len']}": float(ok), "off_by_one": float(v is not None and abs(v - item["gold"]) == 1)}}


# ---------------------------------------------------------------- weekday_calc (calendar arithmetic across leap days)
WD = {"en": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
      "de": ["Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag", "Sonntag"]}


def load_weekday_calc(lang, n, seed):
    import datetime as _dt
    items = []
    for i in range(n or 120):
        r = rng_of("weekday_calc", i)
        kind = ("leap", "century", "plain")[i % 3]
        if kind == "leap":
            year = r.choice([2024, 2028, 2032, 1996])
        elif kind == "century":
            year = r.choice([1900, 2000, 2100])
        else:
            year = r.choice([2023, 2025, 2026, 2027])
        start = _dt.date(year, 1, 1)  # the target lies after Feb 28, so the answer depends on the leap rule
        end = _dt.date(year, r.randint(3, 6), r.randint(1, 28))
        q = (f"January 1, {year} was a {WD['en'][start.weekday()]}. Which weekday is {end.isoformat()}? (Do not look it up, work it out.)"
             if lang == "en" else
             f"Der 1. Januar {year} war ein {WD['de'][start.weekday()]}. Welcher Wochentag ist der {end.isoformat()}? (Nicht nachschlagen, ausrechnen.)")
        items.append({"id": str(i), "lang": lang, "gold": WD[lang][end.weekday()], "kind": kind,
                      "messages": [{"role": "user", "content": q + "\n" + ASK[lang]}]})
    return items


def score_weekday_calc(item, reply):
    fa = (final_answer(reply["content"]) or "").lower()
    hits = [k for k, w in enumerate(WD[item["lang"]]) if re.search(r"\b" + w.lower() + r"\b", fa)]
    v = hits[0] if len(hits) == 1 else None  # a hedge or a "not X" naming two weekdays counts as no answer
    g = WD[item["lang"]].index(item["gold"])
    ok = v == g
    off = v is not None and (v - g) % 7 in (1, 6)
    return {"correct": ok, "score": float(ok), "pred": v, "gold": item["gold"],
            "detail": {f"ok_{item['kind']}": float(ok), "off_one_day": float(off)}}


# ---------------------------------------------------------------- unit_price (cheaper per unit, bulk bait)
# Two offers of the same product in different units / pack sizes / deals. Half the time the bigger pack costs
# more per unit, against the "bulk is cheaper" heuristic. Gold letter alternates.
UP_GOODS = {"en": [("coffee", "g", "kg", 1000), ("juice", "ml", "l", 1000), ("rice", "g", "kg", 1000), ("shampoo", "ml", "l", 1000)],
            "de": [("Kaffee", "g", "kg", 1000), ("Saft", "ml", "l", 1000), ("Reis", "g", "kg", 1000), ("Shampoo", "ml", "l", 1000)]}
UP_BASE = [0.015, 0.002, 0.003, 0.01]  # € per g / ml


def load_unit_price(lang, n, seed):
    items = []
    for i in range(n or 120):
        r = rng_of("unit_price", i)
        good, su, bu, mult = UP_GOODS[lang][i % 4]
        bulk_worse = (i // 4) % 2 == 0
        deal = (i // 8) % 3  # 0 plain, 1 multi-pack, 2 "3 for 2"
        small_q = r.choice([250, 400, 500, 750])
        p_unit = UP_BASE[i % 4] * r.uniform(0.8, 1.25)  # € per small unit
        k = r.choice([1.5, 2, 2.5, 3])  # big pack = k units of the big unit
        diff = r.choice([0.08, 0.12, 0.18])
        small_cost = round(small_q * p_unit, 2)
        if deal == 1:
            cnt = r.choice([2, 4, 6])
            small_txt = (f"{cnt} x {small_q} {su} for {small_cost * cnt:.2f} €" if lang == "en" else
                         f"{cnt} x {small_q} {su} für {small_cost * cnt:.2f} €".replace(".", ","))
            small_per = small_cost * cnt / (cnt * small_q)
        elif deal == 2:
            small_txt = (f"{small_q} {su} for {small_cost:.2f} €, offer: buy 3, pay for 2" if lang == "en" else
                         f"{small_q} {su} für {small_cost:.2f} €, Aktion: 3 kaufen, 2 bezahlen".replace(".", ",", 1))
            small_per = small_cost * 2 / (3 * small_q)
        else:
            small_txt = (f"{small_q} {su} for {small_cost:.2f} €" if lang == "en" else
                         f"{small_q} {su} für {small_cost:.2f} €".replace(".", ","))
            small_per = small_cost / small_q
        big_per = small_per * (1 + diff if bulk_worse else 1 - diff)
        big_cost = round(big_per * k * mult, 2)
        big_per = big_cost / (k * mult)
        ks = f"{k:g}" if lang == "en" else f"{k:g}".replace(".", ",")
        bc = f"{big_cost:.2f}" if lang == "en" else f"{big_cost:.2f}".replace(".", ",")
        big_txt = f"{ks} {bu} for {bc} €" if lang == "en" else f"{ks} {bu} für {bc} €"
        big_first = i % 2 == 0
        a, b = (big_txt, small_txt) if big_first else (small_txt, big_txt)
        big_letter = "A" if big_first else "B"
        gold = (("B" if big_first else "A") if big_per > small_per else big_letter)
        q = (f"Two offers for {good} of the same quality:\nA: {a}\nB: {b}\nWhich offer is cheaper per {su}? "
             "Answer A or B." + (" (For the offer you take exactly 3.)" if deal == 2 else "") if lang == "en" else
             f"Zwei Angebote für {good} in gleicher Qualität:\nA: {a}\nB: {b}\nWelches Angebot ist pro {su} günstiger? "
             "Antworte mit A oder B." + (" (Bei der Aktion nimmst du genau 3.)" if deal == 2 else ""))
        items.append({"id": str(i), "lang": lang, "gold": gold, "bulk_worse": bulk_worse,
                      "messages": [{"role": "user", "content": q + "\n" + ASK[lang]}]})
    return items


def score_unit_price(item, reply):
    fa = final_answer(reply["content"]) or ""
    m = re.findall(r"\b([AB])\b", fa)
    v = m[0] if len(set(m)) == 1 else None
    ok = v == item["gold"]
    return {"correct": ok, "score": float(ok), "pred": v, "gold": item["gold"],
            "detail": {"ok_bulk_worse" if item["bulk_worse"] else "ok_bulk_better": float(ok)}}


# ---------------------------------------------------------------- recipe_scale (what scales and what does not)
# A recipe for p servings, asked for q servings. Ingredient amounts scale proportionally; oven temperature and
# baking time (per tray, same pan) do not. A quarter of the items ask for a fixed quantity: scaling it is the bait.
RS_ING = {"en": [("flour", "g"), ("milk", "ml"), ("sugar", "g"), ("butter", "g")],
          "de": [("Mehl", "g"), ("Milch", "ml"), ("Zucker", "g"), ("Butter", "g")]}


def load_recipe_scale(lang, n, seed):
    items = []
    for i in range(n or 120):
        r = rng_of("recipe_scale", i)
        p = r.choice([2, 4, 6])
        q = r.choice([x for x in (3, 8, 10, 12, 1) if x != p])
        per = [r.randrange(lo, hi + 1, 5) for lo, hi in ((60, 90), (40, 80), (15, 30), (10, 25))]  # per serving
        amounts = [x * p for x in per]
        temp, mins = r.choice([160, 175, 180, 200]), r.choice([20, 25, 35, 40])
        ings = "\n".join(f"- {a} {u} {name}" for (name, u), a in zip(RS_ING[lang], amounts))
        trap = i % 4 == 3
        if trap:
            fixed = (i // 4) % 2
            gold = (temp, mins)[fixed]
            ask = (("At what temperature in °C", "For how many minutes per tray")[fixed] if lang == "en" else
                   ("Bei wie viel °C", "Wie viele Minuten pro Blech")[fixed])
            qq = (f"{ask} do you bake for {q} servings (same tray, baked one tray after another)?" if lang == "en" else
                  f"{ask} backst du für {q} Portionen (gleiches Blech, ein Blech nach dem anderen)?")
        else:
            k = (i // 4) % 4
            name, u = RS_ING[lang][k]
            gold = per[k] * q
            qq = (f"How many {u} of {name} do you need for {q} servings?" if lang == "en" else
                  f"Wie viel {u} {name} brauchst du für {q} Portionen?")
        intro = (f"Recipe for {p} servings:\n{ings}\nBake at {temp} °C for {mins} minutes per tray.\n\n" if lang == "en" else
                 f"Rezept für {p} Portionen:\n{ings}\nBei {temp} °C {mins} Minuten pro Blech backen.\n\n")
        items.append({"id": str(i), "lang": lang, "gold": gold, "trap": trap,
                      "naive": (temp if (i // 4) % 2 == 0 else mins) * q / p if trap else None,
                      "messages": [{"role": "user", "content": intro + qq + "\n" + ASK[lang]}]})
    return items


RS_UNIT = r"(\d+(?:[.,]\d+)?)\s*(kg|g|ml|l|°\s*c|°|grad|min\w*)?\b"


def score_recipe_scale(item, reply):
    # the number that carries the asked unit ("Für 8 Portionen 480 g" -> 480, "2 Bleche à 40 Minuten" -> 40); kg/l -> g/ml
    fa = (final_answer(reply["content"]) or "").lower()
    q = item["messages"][0]["content"].lower()
    want = ("min" if "minut" in q.split("\n\n")[-1] else "°") if item["trap"] else ("ml" if " ml " in q.split("\n\n")[-1] else "g")
    cands, bare = [], []
    for num, unit in re.findall(RS_UNIT, fa):
        x = float(num.replace(",", "."))
        u = (unit or "").replace(" ", "")
        if u in ("kg", "l"):
            x, u = x * 1000, "g" if u == "kg" else "ml"
        u = "°" if u.startswith("°") or u == "grad" else "min" if u.startswith("min") else u
        (cands if u == want else bare if not u else []).append(x)
    vals = set(cands) or (set(bare) if len(set(bare)) == 1 else set())
    v = vals.pop() if len(vals) == 1 else None
    ok = v is not None and abs(v - item["gold"]) <= 0.051
    d = {"ok_trap" if item["trap"] else "ok_scaled": float(ok)}
    if item["trap"]:
        d["scaled_fixed"] = float(v is not None and abs(v - item["naive"]) < 0.51)
    return {"correct": ok, "score": float(ok), "pred": v, "gold": item["gold"], "detail": d}


# ---------------------------------------------------------------- coin_change (greedy bait)
def _cc_dp(coins, amount):
    best = [0] + [None] * amount
    for a in range(1, amount + 1):
        opts = [best[a - c] for c in coins if c <= a and best[a - c] is not None]
        best[a] = min(opts) + 1 if opts else None
    return best[amount]


def _cc_greedy(coins, amount):
    k = 0
    for c in sorted(coins, reverse=True):
        k += amount // c
        amount %= c
    return k if amount == 0 else None


def load_coin_change(lang, n, seed):
    items = []
    for i in range(n or 120):
        r = rng_of("coin_change", i)
        trap = i % 2 == 0
        for _ in range(1000):
            coins = sorted({1, *r.sample(range(2, 30), 3)})
            amount = r.randint(15, 80)
            g, gr = _cc_dp(coins, amount), _cc_greedy(coins, amount)
            if (gr != g) == trap:
                break
        else:
            raise RuntimeError(f"coin_change: no {'trap' if trap else 'control'} system for item {i}")
        cs = ", ".join(map(str, coins))
        q = (f"In a country the only coins are worth {cs} crowns (any number of each). What is the smallest number of coins "
             f"that add up to exactly {amount} crowns?" if lang == "en" else
             f"In einem Land gibt es nur Münzen im Wert von {cs} Kronen (beliebig viele von jeder). Wie viele Münzen braucht man "
             f"mindestens, um genau {amount} Kronen zu bezahlen?")
        items.append({"id": str(i), "lang": lang, "gold": g, "greedy": gr, "trap": trap,
                      "messages": [{"role": "user", "content": q + "\n" + ASK[lang]}]})
    return items


def score_coin_change(item, reply):
    v = last_int(final_answer(reply["content"]) or "")
    ok = v == item["gold"]
    d = {"ok_trap" if item["trap"] else "ok_control": float(ok)}
    if item["trap"]:
        d["took_greedy"] = float(v == item["greedy"])
    return {"correct": ok, "score": float(ok), "pred": v, "gold": item["gold"], "detail": d}


# ---------------------------------------------------------------- alien_calendar (calendar arithmetic without memory)
# Real calendar arithmetic can lean on memorised facts; here the months, their lengths and the week are invented.
AC_MONTHS = ["Brin", "Calo", "Dresk", "Fenn", "Golm", "Hirr"]
AC_DAYS = ["Ash", "Bek", "Cor", "Dun", "Eff", "Fal", "Gim", "Hob", "Isk"]


def load_alien_calendar(lang, n, seed):
    items = []
    for i in range(n or 120):
        r = rng_of("alien_calendar", i)
        months = AC_MONTHS[:r.randint(4, 6)]
        lens = [r.randint(17, 31) for _ in months]
        week = AC_DAYS[:r.randint(5, 9)]
        m0, w0 = r.randrange(len(months)), r.randrange(len(week))
        d0 = r.randint(1, lens[m0])
        wrap = i % 2 == 0
        to_year_end = sum(lens[m0:]) - d0
        n_days = r.randint(to_year_end + 1, to_year_end + 40) if wrap else r.randint(5, max(6, to_year_end))
        wrap = n_days > to_year_end  # a short in-year draw near the year end still wraps
        m, d = m0, d0 + n_days
        while d > lens[m]:
            d -= lens[m]
            m = (m + 1) % len(months)
        wd = week[(w0 + n_days) % len(week)]
        cal = ", ".join(f"{nm} ({ln})" for nm, ln in zip(months, lens))
        q = (f"On the planet Vex a year has these months in this order, with their number of days: {cal}. After the last month "
             f"the next year starts with {months[0]}. The week has {len(week)} days: {', '.join(week)}, then it starts again.\n"
             f"Today is {week[w0]}, day {d0} of {months[m0]}. Which date and weekday is it {n_days} days from today? "
             f"Give it as: <weekday>, day <number> of <month>." if lang == "en" else
             f"Auf dem Planeten Vex hat ein Jahr diese Monate in dieser Reihenfolge, mit ihrer Anzahl an Tagen: {cal}. Nach dem "
             f"letzten Monat beginnt das nächste Jahr mit {months[0]}. Die Woche hat {len(week)} Tage: {', '.join(week)}, dann "
             f"beginnt sie von vorn.\nHeute ist {week[w0]}, Tag {d0} des Monats {months[m0]}. Welches Datum und welcher Wochentag "
             f"ist in {n_days} Tagen? Gib es so an: <Wochentag>, Tag <Zahl> des Monats <Monat>.")
        items.append({"id": str(i), "lang": lang, "gold": f"{wd}, {d}, {months[m]}", "wd": wd, "day": d, "month": months[m],
                      "wrap": wrap, "week": week, "months": months,
                      "messages": [{"role": "user", "content": q + "\n" + ASK[lang]}]})
    return items


def score_alien_calendar(item, reply):
    fa = final_answer(reply["content"]) or ""
    wds = [w for w in item["week"] if re.search(r"\b" + w + r"\b", fa, re.I)]
    mos = [m for m in item["months"] if re.search(r"\b" + m + r"\b", fa, re.I)]
    day = re.findall(r"(\d+)\D{0,20}?\b" + item["month"] + r"\b", fa, re.I)  # "10 of Brin", "10. Brin", "Tag 10 des Monats Brin"
    ok_w = wds == [item["wd"]]
    ok_d = mos == [item["month"]] and [int(x) for x in day] == [item["day"]]
    ok = ok_w and ok_d
    return {"correct": ok, "score": float(ok), "pred": fa[:60], "gold": item["gold"],
            "detail": {"ok_date": float(ok_d), "ok_weekday": float(ok_w), "ok_year_wrap" if item["wrap"] else "ok_in_year": float(ok)}}


# ---------------------------------------------------------------- median_traps (median of an unsorted list)
def load_median_traps(lang, n, seed):
    import statistics
    items = []
    for i in range(n or 120):
        r = rng_of("median_traps", i)
        k = r.choice([7, 9, 11]) if i % 2 == 0 else r.choice([8, 10, 12])
        for _ in range(1000):
            xs = [r.randint(1, 60) for _ in range(k)]
            if i % 3 == 0:
                xs[r.randrange(k)] = r.randint(400, 900)  # an outlier pulls the mean away
            med = statistics.median(xs)
            mid = xs[k // 2] if k % 2 else (xs[k // 2 - 1] + xs[k // 2]) / 2
            mean = sum(xs) / k
            if mid != med and abs(mean - med) > 1:  # both baits give a wrong number
                break
        else:
            raise RuntimeError(f"median_traps: no list for item {i}")
        med = int(med) if med == int(med) else med
        q = (f"Numbers: {', '.join(map(str, xs))}\nWhat is the median?" if lang == "en" else
             f"Zahlen: {', '.join(map(str, xs))}\nWie lautet der Median?")
        items.append({"id": str(i), "lang": lang, "gold": med, "odd": k % 2 == 1, "mid": mid, "mean": mean,
                      "messages": [{"role": "user", "content": q + "\n" + ASK[lang]}]})
    return items


def score_median_traps(item, reply):
    # after the last "=" / "->" the first number counts; otherwise the line must hold exactly one distinct number
    # (a sorted list or a hedge "41.5 (or 46)" is no answer)
    fa = final_answer(reply["content"]) or ""
    tail = re.split(r"=|->|→", fa)[-1]
    nums = [float(x.replace(",", ".")) for x in re.findall(r"-?\d+(?:[.,]\d+)?(?![\d])", tail)]
    v = (nums[0] if nums else None) if tail != fa else (nums[0] if len(set(nums)) == 1 else None)
    ok = v is not None and abs(v - item["gold"]) < 1e-6
    return {"correct": ok, "score": float(ok), "pred": v, "gold": item["gold"],
            "detail": {"ok_odd" if item["odd"] else "ok_even": float(ok),
                       "took_unsorted_middle": float(v is not None and abs(v - item["mid"]) < 1e-6),
                       "took_mean": float(v is not None and abs(v - item["mean"]) < 0.051)}}


# ---------------------------------------------------------------- work_rates (rates add, a drain subtracts)
def load_work_rates(lang, n, seed):
    from fractions import Fraction as F
    items = []
    for i in range(n or 120):
        r = rng_of("work_rates", i)
        drain = i % 2 == 0
        while True:
            a, b = r.randint(2, 9), r.randint(3, 12)
            c = r.randint(4, 15) if drain else None
            rate = F(1, a) + F(1, b) - (F(1, c) if drain else 0)
            if rate > 0 and 1 / rate <= 24 and a != b and c not in (a, b) and abs(60 / rate % 1 - 0.5) > 0.05:
                break
        mins = 60 / rate
        no_drain = 60 / (F(1, a) + F(1, b))
        avg = F(a + b, 2) * 60
        q = (f"An empty tank is filled by pipe A alone in {a} hours and by pipe B alone in {b} hours."
             + (f" A drain at the bottom, if open, empties a full tank in {c} hours." if drain else "")
             + f" Both pipes are opened{' and the drain is open' if drain else ''}. After how many minutes is the tank full? "
             "Round to whole minutes." if lang == "en" else
             f"Ein leerer Tank wird von Rohr A allein in {a} Stunden und von Rohr B allein in {b} Stunden gefüllt."
             + (f" Ein Abfluss am Boden leert, wenn er offen ist, einen vollen Tank in {c} Stunden." if drain else "")
             + f" Beide Rohre werden geöffnet{', und der Abfluss ist offen' if drain else ''}. Nach wie vielen Minuten ist der Tank voll? "
             "Runde auf ganze Minuten.")
        items.append({"id": str(i), "lang": lang, "gold": round(float(mins)), "exact": float(mins), "drain": drain,
                      "no_drain": float(no_drain), "avg": float(avg),
                      "messages": [{"role": "user", "content": q + "\n" + ASK[lang]}]})
    return items


def score_work_rates(item, reply):
    fa = final_answer(reply["content"]) or ""
    hm = re.search(r"(\d+)\s*(?:h|std|stunden?|hours?)\b\D{0,8}?(\d+)\s*min", fa, re.I)
    mins = re.findall(r"(\d+(?:[.,]\d+)?)\s*min", fa, re.I)
    v = (int(hm[1]) * 60 + int(hm[2]) if hm else float(mins[0].replace(",", ".")) if len(set(mins)) == 1
         else answer_value(fa, item["lang"]))  # "1 h 43 min" -> 103, "103 min = 1.71 h" -> 103
    ok = v is not None and abs(v - item["exact"]) <= 1
    d = {"ok_drain" if item["drain"] else "ok_plain": float(ok),
         "took_average": float(v is not None and abs(v - item["avg"]) <= 1)}
    if item["drain"]:
        d["took_ignore_drain"] = float(v is not None and abs(v - item["no_drain"]) <= 1)
    return {"correct": ok, "score": float(ok), "pred": v, "gold": item["gold"], "detail": d}


# ---------------------------------------------------------------- spot_change (diff two texts by reading)
SPC_CLAUSE = {"en": ["The tenant {name} pays {num} euros on the first of each month.", "The deposit is {num} euros and is {neg}refundable.",
                    "Pets are {neg}allowed in the flat.", "The notice period is {num} {unit}.", "Repairs above {num} euros are paid by {name}.",
                    "The heating is {neg}included in the rent.", "Keys are handed over by {name} on day {num}.",
                    "The garden may be used for {num} {unit} per week."],
            "de": ["Der Mieter {name} zahlt am Ersten jedes Monats {num} Euro.", "Die Kaution beträgt {num} Euro und wird {neg}erstattet.",
                   "Haustiere sind in der Wohnung {neg}erlaubt.", "Die Kündigungsfrist beträgt {num} {unit}.",
                   "Reparaturen über {num} Euro bezahlt {name}.", "Die Heizung ist {neg}in der Miete enthalten.",
                   "Die Schlüssel übergibt {name} am Tag {num}.", "Der Garten darf {num} {unit} pro Woche genutzt werden."]}
SPC_NAMES = ["Keller", "Brandt", "Vogel", "Lorenz", "Hahn", "Krüger"]
SPC_UNITS = {"en": ["weeks", "months", "hours", "days"], "de": ["Wochen", "Monate", "Stunden", "Tage"]}
SPC_RANGE = [(400, 1500, None), (500, 3000, None), (0, 0, None), (1, 6, (0, 1)), (50, 400, None), (0, 0, None), (1, 28, None),
             (2, 6, (2, 3))]  # per clause: number range and the two units that fit
SPC_NEG = {"en": "not ", "de": "nicht "}


def load_spot_change(lang, n, seed):
    items = []
    for i in range(n or 120):
        r = rng_of("spot_change", i)
        target = (40, 120, 300)[(i // 4) % 3]
        slots = []
        while sum(len(x[0].split()) for x in slots) < target:
            c = r.randrange(len(SPC_CLAUSE[lang]))
            lo, hi, us = SPC_RANGE[c]
            slots.append((SPC_CLAUSE[lang][c], {"name": r.choice(SPC_NAMES), "num": str(r.randint(lo, hi)), "c": c,
                                                "unit": SPC_UNITS[lang][r.choice(us)] if us else "", "neg": r.choice(["", SPC_NEG[lang]])}))
        kinds = ("num", "name", "unit", "neg")
        kind = kinds[i % 4]
        cands = [k for k, (c, _) in enumerate(slots) if "{" + kind + "}" in c and k > 0]
        if not cands:
            cands = [k for k, (c, _) in enumerate(slots) if "{num}" in c]
            kind = "num"
        k = r.choice(cands)
        tpl, vals = slots[k]
        new = dict(vals)
        if kind == "num":
            new["num"] = str(int(vals["num"]) + (r.choice([10, 50, 100]) if int(vals["num"]) > 100 else 1))
        elif kind == "name":
            new["name"] = r.choice([x for x in SPC_NAMES if x != vals["name"]])
        elif kind == "unit":
            us = SPC_RANGE[vals["c"]][2]
            new["unit"] = [SPC_UNITS[lang][u] for u in us if SPC_UNITS[lang][u] != vals["unit"]][0]
        else:
            new["neg"] = "" if vals["neg"] else SPC_NEG[lang]
        old_t = " ".join(c.format(**v) for c, v in slots)
        new_t = " ".join((tpl.format(**new) if j == k else c.format(**v)) for j, (c, v) in enumerate(slots))
        ow, nw = (vals[kind].strip() or "-", new[kind].strip() or "-")
        q = ("Version A and version B of a contract differ in exactly one place. Which word was changed? Give the word in A and "
             "the word in B as: A: <word> / B: <word> (write - if a word was only added or removed).\n\n" if lang == "en" else
             "Version A und Version B eines Vertrags unterscheiden sich an genau einer Stelle. Welches Wort wurde geändert? Gib das "
             "Wort in A und das Wort in B so an: A: <Wort> / B: <Wort> (schreib -, wenn ein Wort nur ergänzt oder entfernt wurde).\n\n")
        items.append({"id": str(i), "lang": lang, "gold": f"A: {ow} / B: {nw}", "old": ow, "new": nw, "len": target, "kind": kind,
                      "messages": [{"role": "user", "content": q + f"A: {old_t}\n\nB: {new_t}\n\n" + ASK[lang]}]})
    return items


def score_spot_change(item, reply):
    fa = final_answer(reply["content"]) or ""
    def word(part):  # "**A:** 1,369 euros" -> "1369", "—" / "(entfällt)" -> "-"
        part = re.sub(r"^[\s*]*[AB]\s*[*:]+\s*\**", "", part).strip()
        tok = (part.split() or [""])[0].strip(".,\"'“”„*()")
        tok = re.sub(r"(?<=\d)[.,](?=\d{3}\b)", "", tok)
        return "-" if tok in ("", "—", "–", "-", "entfällt", "none", "nothing", "nichts") else tok.lower()
    parts = re.split(r"\s/\s|\s*/\s*(?=\**B)", fa, maxsplit=1)
    got = (word(parts[0][parts[0].find("A"):]), word(parts[1])) if len(parts) == 2 and "A" in parts[0] else None
    ok = got == (item["old"].lower(), item["new"].lower())
    return {"correct": ok, "score": float(ok), "pred": got, "gold": item["gold"],
            "detail": {f"ok_len{item['len']}": float(ok), f"ok_kind_{item['kind']}": float(ok)}}


# ---------------------------------------------------------------- floor_area (area of a walked outline)
FA_DIR = {"en": {"E": "east", "N": "north", "W": "west", "S": "south"}, "de": {"E": "Osten", "N": "Norden", "W": "Westen", "S": "Süden"}}


def _fa_shape(kind, r):
    """Rectilinear outline as a list of (dir, length), counter-clockwise from the south-west corner."""
    W, H = r.randint(6, 14), r.randint(5, 12)
    a, b = r.randint(2, W - 3), r.randint(2, H - 2)
    if kind == "L":  # notch cut out of the north-east corner
        return [("E", W), ("N", H - b), ("W", W - a), ("N", b), ("W", a), ("S", H)], W * H - (W - a) * b
    if kind == "U":  # notch cut into the middle of the north side
        x0 = r.randint(1, W - a - 1)
        return [("E", W), ("N", H), ("W", W - x0 - a), ("S", b), ("W", a), ("N", b), ("W", x0), ("S", H)], W * H - a * b
    raise ValueError(kind)


def load_floor_area(lang, n, seed):
    items = []
    for i in range(n or 120):
        r = rng_of("floor_area", i)
        kind = ("L", "U", "T")[i % 3]
        if kind == "T":
            W, H = r.randint(8, 14), r.randint(6, 12)
            stem, top = r.randint(2, W - 4), r.randint(2, H - 3)
            x0 = r.randint(1, W - stem - 1)
            segs = [("E", stem), ("N", H - top), ("E", W - x0 - stem), ("N", top), ("W", W), ("S", top), ("E", x0), ("S", H - top)]
            area = W * top + stem * (H - top)
            bbox = W * H
        else:
            segs, area = _fa_shape(kind, r)
            xs, ys, px, py = [0], [0], 0, 0
            for d, l in segs:
                px += {"E": l, "W": -l}.get(d, 0)
                py += {"N": l, "S": -l}.get(d, 0)
                xs.append(px)
                ys.append(py)
            bbox = (max(xs) - min(xs)) * (max(ys) - min(ys))
        walk = ", ".join(f"{l} m {FA_DIR[lang][d]}" if lang == "en" else f"{l} m nach {FA_DIR[lang][d]}" for d, l in segs)
        q = (f"A room's walls, walked once around from one corner: {walk}. You are back at the start. "
             "What is the floor area in square metres?" if lang == "en" else
             f"Die Wände eines Raums, einmal von einer Ecke aus abgelaufen: {walk}. Du bist wieder am Start. "
             "Wie groß ist die Bodenfläche in Quadratmetern?")
        items.append({"id": str(i), "lang": lang, "gold": area, "bbox": bbox, "kind": kind,
                      "messages": [{"role": "user", "content": q + "\n" + ASK[lang]}]})
    return items


def score_floor_area(item, reply):
    v = answer_value(final_answer(reply["content"]), item["lang"])
    ok = v == item["gold"]
    return {"correct": ok, "score": float(ok), "pred": v, "gold": item["gold"],
            "detail": {f"ok_{item['kind']}": float(ok), "took_bbox": float(v == item["bbox"])}}


# ---------------------------------------------------------------- queue_sim (execute an unfamiliar procedure)
QS_NAMES = ["Ada", "Ben", "Cem", "Dora", "Eli"]


def _qs_rr(cust, q):
    """cust: [(name, arrival, need)], sorted by arrival. Round robin with slice q; a customer arriving at the moment a
    slice ends joins the queue before the preempted one. Returns {name: finish minute}."""
    t, queue, left, done, i = 0, [], {c[0]: c[2] for c in cust}, {}, 0
    while len(done) < len(cust):
        while i < len(cust) and cust[i][1] <= t:
            queue.append(cust[i][0])
            i += 1
        if not queue:
            t = cust[i][1]
            continue
        nm = queue.pop(0)
        run = min(q, left[nm])
        t += run
        left[nm] -= run
        while i < len(cust) and cust[i][1] <= t:
            queue.append(cust[i][0])
            i += 1
        if left[nm]:
            queue.append(nm)
        else:
            done[nm] = t
    return done


def load_queue_sim(lang, n, seed):
    items = []
    for i in range(n or 120):
        r = rng_of("queue_sim", i)
        k = (3, 4, 5)[i % 3]
        q = r.choice([2, 3, 4])
        while True:
            arr = sorted(r.sample(range(0, 8), k))
            cust = [(QS_NAMES[j], arr[j], r.randint(2, 9)) for j in range(k)]
            done = _qs_rr(cust, q)
            t, fifo = 0, {}
            for nm, a, need in cust:
                t = max(t, a) + need
                fifo[nm] = t
            who = r.choice([c[0] for c in cust])
            if done[who] != fifo[who] and done[who] != max(done.values()):  # the last finisher = total work
                break
        lines = "\n".join((f"- {nm} arrives at minute {a} and needs {need} minutes" if lang == "en" else
                           f"- {nm} kommt in Minute {a} und braucht {need} Minuten") for nm, a, need in cust)
        rule = (f"One clerk serves a queue in turns of at most {q} minutes. A customer who is not finished after a turn goes to "
                "the back of the queue. Customers who arrive while a turn is running, or at the moment it ends, join the queue "
                "before the customer whose turn just ended. The clerk starts at minute 0 and never waits if someone is queued."
                if lang == "en" else
                f"Eine Sachbearbeiterin bedient eine Warteschlange in Runden von höchstens {q} Minuten. Wer nach einer Runde nicht "
                "fertig ist, stellt sich hinten an. Wer während einer Runde oder genau an ihrem Ende ankommt, reiht sich vor der "
                "Person ein, deren Runde gerade endete. Die Sachbearbeiterin beginnt in Minute 0 und wartet nie, wenn jemand ansteht.")
        ask = (f"In which minute is {who} finished?" if lang == "en" else f"In welcher Minute ist {who} fertig?")
        items.append({"id": str(i), "lang": lang, "gold": done[who], "fifo": fifo[who], "k": k,
                      "messages": [{"role": "user", "content": f"{rule}\n{lines}\n{ask}\n" + ASK[lang]}]})
    return items


def score_queue_sim(item, reply):
    v = answer_value(final_answer(reply["content"]), item["lang"])
    ok = v == item["gold"]
    return {"correct": ok, "score": float(ok), "pred": v, "gold": item["gold"],
            "detail": {f"ok_n{item['k']}": float(ok), "took_fifo": float(v == item["fifo"])}}


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
    bare = re.sub(r"\([^()]*\)", " ", line)  # a working note in brackets: "64 m² (= 8 x 6 - 4)"
    if re.search(r"\d", bare):
        line = bare
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
    "rephrase_consistency": {**task(load_rephrase_consistency, score_rephrase_consistency, 24, 150), "grouped": True},
    "reversal": task(load_reversal, score_reversal, 24, 200),
    "off_by_one": task(load_off_by_one, score_off_by_one, 25, 200),
    "mixed_units": task(load_mixed_units, score_mixed_units, 24, 150),
    "double_negation": task(load_double_negation, score_double_negation, 24, 200),
    "false_premise": task(load_false_premise, score_false_premise, 24, 200),
    "option_order": {**task(load_option_order, score_option_order, 24, 200), "grouped": True},
    "source_trust": task(load_source_trust, score_source_trust, 24, 200),
    "switched_question": task(load_switched_question, score_switched_question, 24, 200),
    "causal_graph": task(load_causal_graph, score_causal_graph, 24, 200),
    "ambiguous_ref": task(load_ambiguous_ref, score_ambiguous_ref, 24, 200),
    "counterfactual_world": task(load_counterfactual_world, score_counterfactual_world, 25, 200),
    "fallacy_spot": task(load_fallacy_spot, score_fallacy_spot, 24, 120),
    "arith_length": task(load_arith_length, score_arith_length, 28, 140),
    "nesting_depth": task(load_nesting_depth, score_nesting_depth, 24, 144),
    "grid_transform": task(load_grid_transform, score_grid_transform, 24, 150),
    "blocks_plan": {**task(load_blocks_plan, score_blocks_plan, 24, 120), "max_tokens": 4096},
    "judge_ability": {**task(load_judge_ability, score_judge_ability, 24, 160), "grouped": True},
    "regex_synth": {**task(load_regex_synth, score_regex_synth, 24, 80), "max_tokens": 2048},
    "continuity_error": task(load_continuity_error, score_continuity_error, 24, 120),
    "number_compare": task(load_number_compare, score_number_compare, 24, 160),
    "unit_chain": task(load_unit_chain, score_unit_chain, 24, 150),
    "fake_entity": {**task(load_fake_entity, score_fake_entity, 24, 120), "think": False, "max_tokens": 512},
    "knapsack": task(load_knapsack, score_knapsack, 24, 120),
    "cipher": task(load_cipher, score_cipher, 24, 120),
    "prop_logic": task(load_prop_logic, score_prop_logic, 24, 160),
    "table_build": {**task(load_table_build, score_table_build, 24, 100), "think": False, "max_tokens": 1024},
    "ascii_chart": task(load_ascii_chart, score_ascii_chart, 24, 120),
    "maze_path": task(load_maze_path, score_maze_path, 25, 120),
    "alien_sort": task(load_alien_sort, score_alien_sort, 24, 120),
    "stale_answer": task(load_stale_answer, score_stale_answer, 24, 120),
    "pii_redact": {**task(load_pii_redact, score_pii_redact, 24, 80), "think": False, "max_tokens": 768},
    "percent_traps": task(load_percent_traps, score_percent_traps, 24, 120),
    "clock_angle": task(load_clock_angle, score_clock_angle, 24, 120),
    "base_rate": task(load_base_rate, score_base_rate, 24, 120),
    "simpson": task(load_simpson, score_simpson, 24, 120),
    "prob_fallacies": task(load_prob_fallacies, score_prob_fallacies, 24, 120),
    "framing": {**task(load_framing, score_framing, 24, 120), "grouped": True},
    "conclusion_spot": task(load_conclusion_spot, score_conclusion_spot, 24, 120),
    "sunk_cost": task(load_sunk_cost, score_sunk_cost, 24, 120),
    "regression_mean": task(load_regression_mean, score_regression_mean, 24, 120),
    "dim_analysis": task(load_dim_analysis, score_dim_analysis, 24, 120),
    "table_join": task(load_table_join, score_table_join, 24, 120),
    "dedup_records": task(load_dedup_records, score_dedup_records, 24, 120),
    "interval_math": task(load_interval_math, score_interval_math, 24, 120),
    "weekday_calc": task(load_weekday_calc, score_weekday_calc, 24, 120),
    "queue_sim": task(load_queue_sim, score_queue_sim, 24, 120),
    "floor_area": task(load_floor_area, score_floor_area, 24, 120),
    "spot_change": task(load_spot_change, score_spot_change, 24, 120),
    "work_rates": task(load_work_rates, score_work_rates, 24, 120),
    "median_traps": task(load_median_traps, score_median_traps, 24, 120),
    "alien_calendar": task(load_alien_calendar, score_alien_calendar, 24, 120),
    "coin_change": task(load_coin_change, score_coin_change, 24, 120),
    "recipe_scale": task(load_recipe_scale, score_recipe_scale, 24, 120),
    "unit_price": task(load_unit_price, score_unit_price, 24, 120),
    "count_mentions": task(load_count_mentions, score_count_mentions, 24, 120),
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


# generated tasks whose draws repeat prompts: keep the distinct ones only. _FEW: fewer distinct items than the old
# full count, so the full tier is all of them; _QUICK_CAP: quick tiers that exceeded the distinct count
_FEW = {'percent_traps', 'en_produce', 'polyglot_switch', 'sys_adherence', 'form_writing', 'de_no_anglicisms',
        'agreement_attractor', 'prop_logic', 'counterfactual_world'}
_QUICK_CAP = {'en_produce': 40}
for _name in ['math_missing', 'counterfactual_world', 'fake_entity', 'prop_logic', 'percent_traps', 'jugs', 'probability', 'seq_induction', 'sci_calc']:
    _t = TASKS[_name]
    _full = {k: v if v and _name not in _FEW else None for k, v in _t["counts"]["full"].items()}  # all distinct if fewer
    _quick = {k: min(v, _QUICK_CAP.get(_name, v)) if v else v for k, v in _t["counts"]["quick"].items()}
    TASKS[_name] = {**_t, "load": common.distinct(_t["load"]), "counts": {"quick": _quick, "full": _full}}
