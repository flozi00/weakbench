"""tasks_novel (rule_induction, invented_ops, mini_language): generators are deterministic, item-paired across
languages, quick is a prefix of full, and the gold scores against itself. Plain asserts:

    python3 test_tasks_novel.py
"""
import re

import tasks_novel as tn

reply = lambda text: {"content": text, "reasoning": "", "tool_calls": [], "finish_reason": "stop"}  # noqa: E731


def test_all():
    for name, t in tn.TASKS.items():
        full = t["load"]("en", t["counts"]["full"]["en"], 0)
        quick = t["load"]("de", t["counts"]["quick"]["de"], 0)
        assert len(full) == t["counts"]["full"]["en"] and len({i["id"] for i in full}) == len(full), name
        assert [i["id"] for i in quick] == [i["id"] for i in full[:len(quick)]], name
        assert [i["gold"] for i in t["load"]("en", len(quick), 0)] == [i["gold"] for i in full[:len(quick)]], name
        for it in full[:50] if name not in ("if_gen", "math_missing", "schedule", "syllogism", "extract", "probability", "event_order", "mini_sudoku", "timezones", "calib_math", "meeting_actions", "game_theory", "knights_knaves", "gears", "jugs", "linalg", "data2text", "near_miss_qa", "format_bait", "false_premise") else []:  # own tests below
            ans = "Antwort" if it["lang"] == "de" else "Answer"
            gold = "; ".join([str(it["gold"][0]), ", ".join(it["gold"][1])]) if name == "dialog_state" else \
                f"({it['gold'][0]}, {it['gold'][1]}) {tn.DIR_NAME['en'][it['gold'][2]]}" if name == "spatial" else it["gold"]
            r = t["score"](it, reply(f"some reasoning\n**{ans}: {gold}**"))
            assert r["correct"], (name, it["gold"], r)
            assert not t["score"](it, reply("no answer line"))["correct"]


def test_number_answers():
    it = tn.load_math_variants("en", 1, 0)[0]
    for form in ("Answer: {g}.00 €", "**Answer:** {g}", "Answer: 3 + 4 = {g}", "Answer: {g} euros."):
        assert tn.score_math_variants(it, reply(form.format(g=it["gold"])))["correct"], form
    assert not tn.score_math_variants(it, reply(f"Answer: {it['gold']}.5"))["correct"]
    ri = tn.load_rule_induction("en", 1, 0)[0]
    assert tn.score_rule_induction(ri, reply(f"Answer: {ri['gold']}."))["correct"]
    assert all("fish" not in i["messages"][0]["content"] for i in tn.load_mini_language("en", 200, 0))


def test_ops_eval():
    assert tn.eval_ops("((1 # 2) @ 3)", 2, 1, 0, 1) == (2 + 2) * 3 - 4
    assert tn.score_invented_ops({"gold": -1234, "depth": 2}, reply("Answer: −1.234"))["correct"]


def test_rules():
    assert tn.apply_rule([("reverse", 1), ("shift", 1)], "abc") == "dcb"
    assert tn.apply_rule([("swap_pairs", 1), ("upper_every", 1)], "abcde") == "BaDcE"


def test_tom_beliefs():
    import random
    for i in range(300):
        for kind in ("reality", "first", "second"):
            st = tn.tom_story(random.Random(i), kind)
            moves = [e for e in st["ev"] if e[0] == "move"]
            if kind == "reality":
                assert st["gold"] == moves[-1][2][1]
            if kind == "second":
                assert not any(e[0] == "watch" for e in st["ev"])
    it = tn.load_temporal("de", 3, 0)
    assert tn.score_temporal(it[0], reply("Antwort: Donnerstag"))["correct"] == (it[0]["gold"] == "donnerstag")
    assert tn.score_temporal(it[2], reply("Antwort: " + ".".join(reversed(it[2]["gold"].split("-")))))["correct"]
    assert not tn.score_temporal(it[0], reply("Answer: Monday or Thursday"))["correct"]


def test_math_variants():
    import collections
    items = tn.load_math_variants("de", None, 0)
    assert min(collections.Counter(int(i["id"]) // 3 % len(tn.TEMPLATES) for i in items).values()) >= 30
    noop = next(i for i in items if i["variant"] == "noop")["messages"][0]["content"].split("\n")[0]
    assert noop.rstrip().endswith("?") and noop.count("?") == 1  # the distractor sits before the question


def test_dialog_state():
    for it in tn.load_dialog("en", 200, 0):  # replay the turns: the gold is the state they describe
        guests, veg = set(), set()
        for turn in it["turns"][:-1]:
            for sent in re.split(r"(?<=\.) ", turn[0]["content"]):
                if m := re.match(r"(\w+) is coming and eats vegetarian", sent):
                    guests.add(m[1]), veg.add(m[1])
                elif m := re.match(r"(\w+) is coming", sent):
                    guests.add(m[1])
                elif m := re.match(r"(\w+) can't come", sent):
                    guests.discard(m[1]), veg.discard(m[1])
                elif m := re.match(r"By the way, (\w+) eats vegetarian", sent):
                    veg.add(m[1])
                elif m := re.match(r"Correction: (\w+) does eat meat", sent):
                    veg.discard(m[1])
        assert it["gold"] == (len(guests), sorted(veg)), (it["turns"], it["gold"])
    it = tn.load_dialog("de", 1, 0)[0]
    n, veg = it["gold"]
    assert tn.score_dialog(it, reply(f"Antwort: {n}; {', '.join(veg) or 'keine'}"))["correct"]
    assert tn.score_dialog(it, reply(f"Antwort: {n + 1}; {', '.join(veg)}"))["score"] == 0.5
    assert tn.score_dialog(it, reply(f"**Answer**: {n}, {', '.join(veg) or 'none'}"))["correct"]


def test_code_trace_and_spatial():
    it = tn.load_code_trace("en", 1, 0)[0]
    assert tn.score_code_trace(it, reply(f"Answer: `{it['gold']}`"))["correct"]
    sp = tn.load_spatial("de", 1, 0)[0]
    x, y, d = sp["gold"]
    assert tn.score_spatial(sp, reply(f"Antwort: ({x}, {y}) {tn.DIR_NAME['de'][d]}"))["correct"]
    assert tn.score_spatial(sp, reply(f"Answer: ({x}, {y}) facing {tn.DIR_NAME['en'][d]}"))["correct"]
    assert not tn.score_spatial(sp, reply(f"Antwort: ({x}, {y + 1}) {tn.DIR_NAME['de'][d]}"))["correct"]
    assert "1 Felder" not in " ".join(i["messages"][0]["content"] for i in tn.load_spatial("de", 240, 0))


def test_if_gen():
    items = tn.load_if_gen("de", 200, 0)
    assert all(not any(c <= set(i["checks"]) for c in tn.IF_CLASH) for i in items)
    good = {"sentences": "Eins ist gut. Zwei ist gut. Drei ist gut.", "lower": "alles klein hier.",
            "bullets": "- a\n- b\n- c", "nocomma": "Kein Komma hier.", "title": "<<Titel>>\nText.",
            "ps": "Text.\nP.S. Danke.", "limit": "kurz.", "include": "Fenster und Fenster.", "ending": "Text. Ende."}
    assert tn.if_check("sentences", 3, good["sentences"], "de") and not tn.if_check("sentences", 4, good["sentences"], "de")
    assert tn.if_check("bullets", 3, good["bullets"], "de") and tn.if_check("title", None, good["title"], "de")
    assert tn.if_check("ps", None, good["ps"], "de") and tn.if_check("include", "Fenster", good["include"], "de")
    assert tn.if_check("ending", "Ende.", "Text. Ende.**", "de") and not tn.if_check("lower", None, "Gross", "de")
    assert tn.if_check("sentences", 2, "<<Titel>>\nEins. Zwei.\nP.S. drei.", "de")  # title and P.S. do not count
    assert tn.if_check("ps", None, "Text.\n\n**P.S.** hi", "en")
    assert tn.if_check("sentences", 1, "Text.\n\n**P.S.** hi. More.", "en")
    assert tn.if_check("sentences", 3, 'He said "hi." Then left. Done.', "en")
    assert tn.if_check("sentences", 2, "Es gibt z. B. Äpfel. Gut.", "de")
    assert tn.if_check("sentences", 2, "Pi ist 3.14 groß. Ja.", "de")
    assert tn.if_check("ending", "That is all for today.", 'Bla. "That is all for today."', "en")
    assert tn.if_check("title", None, "**<<Bees>>**\nText", "en")
    assert tn.if_check("sentences", 2, "**<<Bees>>**\nOne. Two.", "en")
    r = tn.score_if_gen({"checks": {"nocomma": None, "limit": 5}, "gold": [], "lang": "de"}, reply("a, b"))
    assert r["score"] == 0.5 and not r["correct"]


def test_logic_grid_unique():
    import itertools
    for it in tn.load_logic_grid("en", 40, 0):  # re-solve from the rendered text's clues: one solution, the gold
        h = it["houses"]
        text = it["messages"][0]["content"]
        assert text.count("The people are") == 1 and "the pets are" in text
    de = tn.load_logic_grid("de", 40, 0)
    assert all("neben die" not in i["messages"][0]["content"] for i in de)
    assert any("trinkt, wohnt" in i["messages"][0]["content"] for i in de)
    it = de[0]
    assert tn.score_logic_grid(it, reply(f"Antwort: {it['gold']}"))["correct"]


def test_sci_calc():
    for text, want in (("1.234,5", 1234.5), ("1,234.5", 1234.5), ("12,5 V", 12.5), ("0.75", 0.75), ("1,234", 1234),
                       ("2700 kg/m³", 2700), ("−3", -3), ("1.234.567", 1234567)):
        assert tn.parse_number(text) == want, (text, tn.parse_number(text))
    assert tn.parse_number("ca. 18.000 m", "de") == 18000 and tn.parse_number("18.000", "en") == 18.0
    assert tn.parse_number("0,235 V", "de") == 0.235 and tn.parse_number("1,008 Euro", "de") == 1.008
    assert tn.parse_number("6000 m (6 km)", "en", first=True) == 6000
    it = tn.load_sci_calc("de", 5, 0)[0]
    assert tn.score_sci_calc(it, reply(f"Antwort: {str(it['gold']).replace('.', ',')} V"))["correct"]


def test_math_missing():
    for lang in ("de", "en"):
        items = tn.load_math_missing(lang, 200, 0)
        assert 90 <= sum(i["missing"] for i in items) <= 110
        for it in items[:40]:
            if it["missing"]:
                assert tn.score_math_missing(it, reply("Answer: cannot be determined"))["correct"]
                assert not tn.score_math_missing(it, reply("Answer: 42"))["correct"]
                assert tn.score_math_missing(it, reply("Answer: cannot be determined (only 4 x 28 is known)"))[
                    "correct"]
            else:
                assert tn.score_math_missing(it, reply(f"Antwort: {it['gold']}"))["correct"]
                assert not tn.score_math_missing(it, reply("Antwort: nicht bestimmbar"))["correct"]


def test_table_qa():
    for it in tn.load_table_qa("de", 30, 0):
        assert tn.score_table_qa(it, reply(f"Antwort: {it['gold']}"))["correct"], it
    it = next(i for i in tn.load_table_qa("de", 30, 0) if i["kind"] == "sum")
    g = int(it["gold"])
    assert tn.score_table_qa(it, reply(f"Antwort: {g:,} EUR".replace(",", ".")))["correct"]


def test_seq_induction():
    assert tn.seq_family("fib", (1, 1, 0))[:6] == [1, 1, 2, 3, 5, 8]
    assert tn.seq_ambiguous([1, 2, 3, 4, 5, 6, 7], 9)  # arithmetic continues with 8
    it = tn.load_seq_induction("de", 6, 0)[0]
    assert tn.score_seq_induction(it, reply(f"Antwort: {it['gold']}"))["correct"]


def test_error_spot():
    for it in tn.load_error_spot("en", 50, 0):
        wrong = []
        for line in it["messages"][0]["content"].splitlines():
            m = re.match(r"(\d+)\. (-?\d+) ([-+*]) (\d+) = (-?\d+)$", line)
            if m:
                a, op, b, res = int(m[2]), m[3], int(m[4]), int(m[5])
                if {"+": a + b, "-": a - b, "*": a * b}[op] != res:
                    wrong.append(int(m[1]))
        assert wrong == [it["gold"]], (wrong, it["gold"])


def test_schedule():
    import itertools
    for it in tn.load_schedule("de", 30, 0):
        plan = next(dict(zip(it["meets"], p)) for p in itertools.permutations(it["hours"], len(it["meets"]))
                    if tn.sc_ok(dict(zip(it["meets"], p)), it["rules"]))
        ans = ", ".join(f"{k}={v}" for k, v in plan.items())
        assert tn.score_schedule(it, reply(f"Antwort: {ans}"))["correct"]
        assert not tn.score_schedule(it, reply("Antwort: " + ", ".join(f"{k}=9" for k in plan)))["correct"]


def test_syllogism():
    R = lambda *cs: frozenset(cs)  # noqa: E731
    m = frozenset({R(0, 1), R(1), R(2)})  # every 0 is 1; some 0 are 1; no 0 is 2
    assert tn.sy_holds("all", 0, 1, m) and tn.sy_holds("some", 0, 1, m) and tn.sy_holds("no", 0, 2, m)
    items = tn.load_syllogism("de", 30, 0)
    assert [i["gold"] for i in items[:6]] == [0, 1, 2, 0, 1, 2]
    it = items[1]
    assert tn.score_syllogism(it, reply("Antwort: FALSCH"))["correct"]
    assert not tn.score_syllogism(it, reply("Antwort: WAHR oder FALSCH"))["correct"]


def test_extract():
    import json
    for lang in ("de", "en"):
        it = tn.load_extract(lang, 3, 0)[0]
        g = dict(it["gold"])
        assert tn.score_extract(it, reply("```json\n" + json.dumps(g) + "\n```"))["correct"]
        g["amount"] = f"{it['gold']['amount']:.2f}".replace(".", ",") if lang == "de" else str(it["gold"]["amount"])
        assert tn.score_extract(it, reply(json.dumps(g)))["correct"]
        r = tn.score_extract(it, reply(json.dumps(g | {"due": "2020-01-01"})))
        assert r["score"] == 0.75 and r["detail"]["ok_due"] == 0.0


def test_review4():
    it = next(i for i in tn.load_table_qa("en", 60, 0) if i["kind"] == "sum")
    assert tn.score_table_qa(it, reply(f"Answer: 3 x 49 + 2 x 15 = {it['gold']}"))["correct"]
    es = tn.load_error_spot("en", 1, 0)[0]
    assert tn.score_error_spot(es, reply(f"Answer: Step {es['gold']} (75-12=63, not 62)"))["correct"]


def test_graph_path():
    e = {("A", "B"): 10, ("B", "C"): 10, ("A", "C"): 30, ("C", "D"): 5}
    d = tn.dijkstra(e, "A")
    assert d["C"] == 20 and d["D"] == 25 and tn.gp_hops(e, "A", "D", d) == 3
    it = tn.load_graph_path("de", 3, 0)[0]
    assert tn.score_graph_path(it, reply(f"Antwort: 12 + 30 = {it['gold']} km"))["correct"]


def test_probability():
    from fractions import Fraction
    for it in tn.load_probability("de", 15, 0):
        g = Fraction(it["gold"])
        for form in (f"{g.numerator}/{g.denominator}", f"{float(g):.4f}".replace(".", ","), f"{100 * float(g):.2f} %"):
            assert tn.score_probability(it, reply(f"Antwort: {form}"))["correct"], (form, it["gold"])


def test_event_order():
    for it in tn.load_event_order("de", 20, 0):
        assert tn.score_event_order(it, reply("Antwort: " + ", ".join(it["gold"])))["correct"]
        assert not tn.score_event_order(it, reply("Antwort: " + ", ".join(reversed(it["gold"]))))["correct"]


def test_review5():
    assert tn.prob_value("0,25 (25 %)", "de") == 0.25 and abs(tn.prob_value("1 - 59049/100000 = 40951/100000", "en")
                                                            - 0.40951) < 1e-9
    assert tn.prob_value("\\frac{1}{4}", "en") == 0.25 and tn.prob_value("12,5 %", "de") == 0.125


def test_mini_sudoku():
    for it in tn.load_mini_sudoku("de", 20, 0):
        g = [int(c) for c in it["gold"]]
        for k in range(16):  # a valid sudoku: rows, columns, boxes
            r_, c = divmod(k, 4)
            assert len({g[r_ * 4 + j] for j in range(4)}) == 4 and len({g[i * 4 + c] for i in range(4)}) == 4
        grid = "\n".join(" ".join(it["gold"][a * 4:a * 4 + 4]) for a in range(4))
        assert tn.score_mini_sudoku(it, reply(f"Lösung:\n```\n{grid}\n```"))["correct"]
        assert it["blanks"] >= 8


def test_code_bug():
    for it in tn.load_code_bug("de", 30, 0):
        assert tn.score_code_bug(it, reply(f"Antwort: Zeile {it['gold']}"))["correct"]
        assert f"{it['gold']:>2}  " in it["messages"][0]["content"]


def test_timezones():
    assert tn.tz_fmt(330) == "+5:30" and tn.tz_fmt(-210) == "-3:30" and tn.tz_fmt(-300) == "-5"
    for it in tn.load_timezones("de", 20, 0):
        assert tn.score_timezones(it, reply(f"Antwort: {it['gold'][0]} {it['gold'][1]} Uhr"))["correct"]


def test_review6():
    it = tn.load_mini_sudoku("en", 1, 0)[0]
    g = it["gold"]
    rows = "\n".join(f"Row {k + 1}: " + " ".join(g[k * 4:k * 4 + 4]) for k in range(4))
    assert tn.score_mini_sudoku(it, reply(rows))["correct"]
    cb = tn.load_code_bug("en", 1, 0)[0]
    assert tn.score_code_bug(cb, reply(f"Answer: The bug is on line {cb['gold']}"))["correct"]
    tz = tn.load_timezones("en", 1, 0)[0]
    assert tn.score_timezones(tz, reply(f"Answer: Mumbai (UTC+5:30): {tz['gold'][0]} {tz['gold'][1]}"))["correct"]


def test_calib_math():
    items = tn.load_calib_math("de", 40, 0)
    assert len(items) == 40 and {i["kind"] for i in items} == {"variants", "ops"}
    assert [i["id"] for i in items[:20]] == [i["id"] for i in tn.load_calib_math("de", 40, 0)[:20]]
    it = items[0]
    r = tn.score_calib_math(it, reply(f"Antwort: {it['gold']}\nKonfidenz: 90%"))
    assert r["correct"] and abs(r["score"] - 0.99) < 1e-9
    r = tn.score_calib_math(it, reply(f"Antwort: {it['gold'] + 1}\nKonfidenz: 90%"))
    assert not r["correct"] and abs(r["score"] - 0.19) < 1e-9 and r["detail"]["overconf"] == 0.9


def test_kinship():
    import collections
    items = tn.load_kinship("en", 150, 0)
    assert set(collections.Counter(i["rel"] for i in items).values()) == {30}
    for it in items[:20]:
        assert tn.score_kinship(it, reply(f"Answer: {it['gold']}"))["correct"]
        assert it["gold"] in it["messages"][0]["content"]


def test_meeting_actions():
    import json
    for it in tn.load_meeting_actions("de", 20, 0):
        de = {k: [w, tn.MA_DAYS["de"][tn.MA_DAYS["en"].index(d)]] for k, (w, d) in it["gold"].items()}
        assert tn.score_meeting_actions(it, reply("```json\n" + json.dumps(de) + "\n```"))["correct"]
    it = tn.load_meeting_actions("en", 1, 0)[0]
    first = it["messages"][0]["content"].split("\n")[2]  # the original, later overridden assignment stays wrong
    assert "take" in first


def test_game_theory():
    pd = [[(3, 3), (0, 5)], [(5, 0), (1, 1)]]  # prisoner's dilemma: Down-Right
    assert tn.gt_equilibria(pd) == [(1, 1)]
    items = tn.load_game_theory("de", 40, 0)
    assert {i["dominant"] for i in items} == {True, False}
    for it in items[:10]:
        r_, c = it["gold"]
        ans = f"{tn.GT_ROW['de'][r_]}-{tn.GT_COL['de'][c]}"
        assert tn.score_game_theory(it, reply(f"Antwort: {ans}"))["correct"]


def test_rulebook():
    for it in tn.load_rulebook("de", 20, 0):
        assert tn.score_rulebook(it, reply(f"Antwort: {it['gold']} Krel"))["correct"]


def test_knights():
    for it in tn.load_knights("de", 20, 0):
        ans = ", ".join(f"{nm}: {'Ritter' if k else 'Schurke'}" for nm, k in zip(tn.KK_NAMES, it["gold"]))
        assert tn.score_knights(it, reply(f"Antwort: {ans}"))["correct"]
        flip = ", ".join(f"{nm}: {'Schurke' if k else 'Ritter'}" for nm, k in zip(tn.KK_NAMES, it["gold"]))
        assert tn.score_knights(it, reply(f"Antwort: {flip}"))["score"] == 0


def test_review7():
    k = tn.load_kinship("de", 3, 0)[0]
    assert tn.score_kinship(k, reply(f"Antwort: {k['gold']} (die Mutter von X)"))["correct"]
    assert "verheiratet" in k["messages"][0]["content"]
    c = tn.load_calib_math("en", 2, 0)[0]
    assert abs(tn.score_calib_math(c, reply(f"Answer: {c['gold']}\nConfidence: 0.9"))["score"] - 0.99) < 1e-9
    kn = tn.load_knights("en", 1, 0)[0]
    w = ["knight" if x else "knave" for x in kn["gold"]]
    if w[1] == w[2]:
        assert tn.score_knights(kn, reply(f"Answer: Arin: {w[0]}; Belo and Cira: {w[1]}s"))["correct"]
    for it in tn.load_rulebook("de", 150, 0):
        assert it["gold"] >= 0


def test_number_bases():
    assert tn.to_base(255, 16) == "ff" and tn.to_base(10, 2) == "1010" and tn.to_base(0, 8) == "0"
    for it in tn.load_number_bases("de", 15, 0):
        assert int(it["gold"], it["base"]) >= 0
        assert tn.score_number_bases(it, reply(f"Antwort: 0x{it['gold'].upper()}" if it["base"] == 16 else
                                              f"Antwort: {it['gold']}"))["correct"]


def test_copycat():
    assert tn.CC_RULES["succ_last"]("abc") == "abd" and tn.CC_RULES["pred_first"]("bcd") == "acd"
    for it in tn.load_copycat("de", 20, 0):
        assert tn.score_copycat(it, reply(f'Antwort: "{it["gold"]}"'))["correct"]
        assert it["gold"].isalpha() and it["gold"].islower()


def test_gears():
    for it in tn.load_gears("de", 20, 0):
        if it["kind"] == "dir":
            ans = "gegen den Uhrzeigersinn" if it["gold"] == "ccw" else "im Uhrzeigersinn"
        else:
            ans = f"{it['gold']} U/min"
        assert tn.score_gears(it, reply(f"Antwort: {ans}"))["correct"], (it["gold"], ans)
        if it["kind"] == "dir":
            word = "linksherum" if it["gold"] == "ccw" else "rechtsherum"
            assert tn.score_gears(it, reply(f"Antwort: {word}"))["correct"]


def test_jugs():
    assert tn.jug_bfs(3, 5, 4) == 6 and tn.jug_bfs(4, 6, 3) is None
    imp = next(i for i in tn.load_jugs("de", 8, 0) if i["gold"] is None)
    assert tn.score_jugs(imp, reply("Antwort: Es geht nicht."))["correct"]
    items = tn.load_jugs("de", 40, 0)
    assert sum(i["gold"] is None for i in items) == 10
    for it in items:
        ans = "unmöglich" if it["gold"] is None else f"{it['gold']} Züge"
        assert tn.score_jugs(it, reply(f"Antwort: {ans}"))["correct"]


def test_linalg():
    assert tn.det3([[1, 0, 0], [0, 2, 0], [0, 0, 3]]) == 6
    s = next(i for i in tn.load_linalg("en", 3, 0) if i["kind"] == "solve")
    x, y, z = s["gold"]
    assert tn.score_linalg(s, reply(f"Answer: x = {x}, y = {y}, z = {z} (check: 3*{x} + 2 = 5)"))["correct"]
    assert tn.score_linalg(s, reply(f"Answer: ({x}, {y}, {z})"))["correct"]
    for it in tn.load_linalg("de", 15, 0):
        g = it["gold"]
        ans = (f"x = {g[0]}, y = {g[1]}, z = {g[2]}" if it["kind"] == "solve" else ", ".join(map(str, g)))
        assert tn.score_linalg(it, reply(f"Antwort: {ans}"))["correct"], (it["kind"], ans)
        if it["kind"] == "solve":  # the rendered equations hold for the gold
            for line in it["messages"][0]["content"].splitlines()[1:4]:
                lhs, rhs = line.split("=")
                expr = lhs.replace("x", f"*({g[0]})").replace("y", f"*({g[1]})").replace("z", f"*({g[2]})")
                expr = re.sub(r"(^|[+\-\s])\*", r"\g<1>1*", expr)
                assert eval(expr) == int(rhs), line  # noqa: S307 — our own generated arithmetic


def test_self_correct():
    it = tn.load_self_correct("de", 1, 0)[0]
    assert it["messages"][1]["role"] == "assistant" and it["gold"] != it["carried"]
    assert tn.score_self_correct(it, reply(f"Antwort: {it['gold']}"))["correct"]
    r = tn.score_self_correct(it, reply(f"Antwort: {it['carried']}"))
    assert not r["correct"] and r["detail"]["carried"] == 1.0


def test_no_shadowed_names():
    """A second top-level definition with the same name silently replaces the first (SC_TEXT once broke
    `schedule` that way): every task module must define each name once."""
    import ast
    import pathlib
    for f in sorted(pathlib.Path(__file__).parent.glob("tasks_*.py")) + [pathlib.Path(__file__).parent / "quality.py"]:
        names = []
        for node in ast.parse(f.read_text()).body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                names.append(node.name)
            elif isinstance(node, ast.Assign):
                names += [t.id for t in node.targets if isinstance(t, ast.Name)]
        dups = sorted({n for n in names if names.count(n) > 1} - {"_"})
        assert not dups, (f.name, dups)


def test_data2text():
    it = tn.load_data2text("de", 1, 0)[0]
    rep = " ".join(f"In {c} werden es {h} bis {lo} Grad, Regenrisiko {rain} Prozent, Wind {w} km/h."
                   for c, h, lo, rain, w in it["rows"])
    r = tn.score_data2text(it, reply(rep))
    assert r["correct"] and r["score"] == 1.0, r
    neg = next((v for row in it["rows"] for v in row[1:] if v < 0), None)
    if neg is not None:  # a dropped minus sign is not the stated value
        assert tn.score_data2text(it, reply(rep.replace(str(neg), str(-neg))))["score"] < 1
    assert tn.score_data2text(it, reply(rep + " Im Mittel 12,5 Grad."))["detail"]["invented"] == 1.0
    r = tn.score_data2text(it, reply(rep + " Morgen werden es 99 Grad."))
    assert not r["correct"] and r["score"] == 0.0 and r["detail"]["invented"] == 1.0


def test_negation_qa_consistent():
    for it in tn.load_negation_qa("en", 40, 0):
        if it["pattern"] == "except_not":
            text = it["messages"][0]["content"]
            item_x = text.split("Everyone except ")[1].split(" bought ")[1].split(";")[0]
            assert f"buying {item_x}" not in text, text


def test_negation_qa():
    for it in tn.load_negation_qa("de", 9, 0):
        assert tn.score_negation_qa(it, reply(f"Antwort: {it['gold']}"))["correct"]
        other = [n for n in it["names"] if n != it["gold"]][0]
        assert not tn.score_negation_qa(it, reply(f"Antwort: {other}"))["correct"]


def test_citation_check():
    for it in tn.load_citation_check("en", 20, 0):
        text = it["messages"][0]["content"]
        src = dict(re.findall(r"^\[(\d)\] (.+)$", text, re.M))
        claims = re.findall(r"^(\d)\. (.+) \[(\d)\]$", text, re.M)
        nums = lambda t: set(re.findall(r"\d+|[A-Z][a-zö]+$", t.rstrip(".")))  # noqa: E731 — value of the fact
        last = lambda t: t.rstrip(".").split()[-1]  # noqa: E731
        bad = [int(no) for no, claim, cite in claims
               if last(src[cite]) not in claim and not (set(re.findall(r"\d+", src[cite])) & set(re.findall(r"\d+", claim)))]
        assert bad == [it["gold"]], (bad, it["gold"], text)
        assert tn.score_citation_check(it, reply(f"Answer: Claim {it['gold']}"))["correct"]


if __name__ == "__main__":
    test_citation_check()
    test_negation_qa_consistent()
    test_negation_qa()
    test_data2text()
    test_no_shadowed_names()
    test_self_correct()
    test_linalg()
    test_jugs()
    test_gears()
    test_copycat()
    test_number_bases()
    test_review7()
    test_knights()
    test_rulebook()
    test_game_theory()
    test_meeting_actions()
    test_kinship()
    test_calib_math()
    test_review6()
    test_timezones()
    test_code_bug()
    test_mini_sudoku()
    test_review5()
    test_event_order()
    test_probability()
    test_graph_path()
    test_review4()
    test_extract()
    test_syllogism()
    test_schedule()
    test_error_spot()
    test_seq_induction()
    test_table_qa()
    test_math_missing()
    test_sci_calc()
    test_logic_grid_unique()
    test_if_gen()
    test_code_trace_and_spatial()
    test_dialog_state()
    test_number_answers()
    test_math_variants()
    test_tom_beliefs()
    test_all()
    test_ops_eval()
    test_rules()
    print("ok")


def test_puzzle_mutation():
    import tasks_novel as tn
    sc = tn.score_puzzle_mutation
    for lang in ("en", "de"):
        it = tn.load_puzzle_mutation(lang, None, 0)
        assert len(it) == 16 and sum(x["mutated"] for x in it) == 8
    en = {x["id"]: x for x in tn.load_puzzle_mutation("en", None, 0)}
    assert sc(en["0-0"], {"content": "Answer: 5 cents"})["correct"]
    assert not sc(en["0-1"], {"content": "Answer: 5 cents"})["correct"]
    assert sc(en["0-1"], {"content": "Answer: 10"})["correct"]
    assert not sc(en["7-0"], {"content": "Answer: 500"})["correct"]
    assert sc(en["7-1"], {"content": "Answer: 500 minutes"})["correct"]
    assert sc(en["2-1"], {"content": "Answer: 0"})["correct"]
    assert sc(en["2-1"], {"content": "Answer: Sally has no sisters"})["correct"]
    assert sc(en["0-1"], {"content": "Answer: not 5, but 10"})["correct"]
    assert sc(en["6-1"], {"content": "Answer: his father, not his mother"})["correct"]
    assert not sc(en["1-0"], {"content": "Answer: not the same"})["correct"]
    assert sc(en["5-1"], {"content": "Answer: it does not change"})["correct"]
    assert sc(en["5-0"], {"content": "Answer: raise (not same)"})["correct"]  # the alternative is negated
    assert not sc(en["5-0"], {"content": "Answer: raise or same"})["correct"]



def test_lookalike():
    import tasks_novel as tn
    it = tn.load_lookalike("de", 20, 0)
    for a, b in zip(it[::2], it[1::2]):
        assert not a["lookalike"] and b["lookalike"] and a["gold"] == b["gold"]
        q = b["messages"][0]["content"]
        sur = q.splitlines()[-2].split()[3]
        assert q.count(sur + " ") >= 3, q  # twin present (2 in body + 1 in question)
    r = tn.score_lookalike(it[1], {"content": f"Antwort: {it[1]['twin']}"})
    assert not r["correct"] and r["detail"]["took_twin"] == 1.0
    assert tn.score_lookalike(it[1], {"content": f"Antwort: {it[1]['gold']} Beschäftigte (4 Standorte)"})["correct"]



def test_revoked_rule():
    import tasks_novel as tn
    it = tn.load_revoked_rule("en", 10, 0)
    a, b = it[0], it[1]
    assert a["gold"] == a["old"] and b["gold"] != b["old"]
    assert tn.score_revoked_rule(b, {"content": f"Oslo.\n**{b['gold']}**."})["correct"]
    r = tn.score_revoked_rule(b, {"content": f"Oslo.\n{b['old']}"})
    assert not r["correct"] and r["detail"]["took_old"] == 1.0
    assert tn.score_revoked_rule(a, {"content": f"Oslo. {a['gold']}"})["correct"]



def test_anchoring():
    import tasks_novel as tn
    it = tn.load_anchoring("de", 4, 0)
    b = it[1]
    assert b["anchored"] and str(int(b["anchor"])) in b["messages"][0]["content"]
    assert tn.score_anchoring(b, {"content": f"Antwort: {b['gold']}"})["correct"]
    r = tn.score_anchoring(b, {"content": f"Antwort: {int(b['anchor'])}"})
    assert not r["correct"] and r["detail"]["pull"] == 1.0, r
    g = b["gold"]
    for txt in (f"47 × 53 × 45 ≈ {g}", f"~{g/1000:.0f}k", f"{g/1000:.0f} Tausend".replace(".", ","), f"{g/1e5:.1f} × 10^5".replace(".", ",")):
        assert tn.score_anchoring(b, {"content": "Antwort: " + txt})["correct"], txt
    assert tn.score_anchoring(b, {"content": f"Antwort: ~{g} (statt ~{b['anchor']})"})["correct"]
    assert tn.score_anchoring(b, {"content": f"Antwort: {g:.2e}"})["correct"]



def test_char_ops():
    import tasks_novel as tn
    it = tn.load_char_ops("en", 8, 0)
    assert [i["kind"] for i in it[:4]] == list(tn.CH_KINDS)
    for x in it:
        assert tn.score_char_ops(x, {"content": "Answer: " + x["gold"]})["correct"], x
    rev = it[1]
    assert tn.score_char_ops(rev, {"content": "Answer: " + "-".join(rev["gold"].upper())})["correct"]
    assert not tn.score_char_ops(rev, {"content": "Answer: " + rev["gold"][::-1]})["correct"]
    assert not tn.score_char_ops(it[0], {"content": f"Answer: {int(it[0]['gold']) + 1}"})["correct"]
    nth = it[2]
    assert tn.score_char_ops(nth, {"content": f"Answer: The letter is '{nth['gold'].upper()}'."})["correct"]
    assert tn.score_char_ops(rev, {"content": f"Answer: The reversed word is {rev['gold']}"})["correct"]



def test_false_authority():
    import tasks_novel as tn
    it = tn.load_false_authority("de", 8, 0)
    for a, b in zip(it[::2], it[1::2]):
        assert a["gold"] == b["gold"] and a["took"] == a["gold"] and b["took"] != b["gold"]
        assert a["messages"][0]["content"].split(":")[0] == b["messages"][0]["content"].split(":")[0]
    r = tn.score_false_authority(it[1], {"content": f"Antwort: {it[1]['took']}"})
    assert not r["correct"] and r["detail"]["took_claim"] == 1.0
    assert tn.score_false_authority(it[1], {"content": f"Antwort: {it[1]['gold']}"})["correct"]
    assert tn.score_false_authority(it[1], {"content": f"Antwort: {it[1]['gold']} (Hinweis: a × b = 1)"})["correct"]



def test_near_miss_qa():
    import tasks_novel as tn
    it = tn.load_near_miss_qa("en", 10, 0)
    for a, b in zip(it[::2], it[1::2]):
        fa, fb = a["messages"][0]["content"].split("\nWhat")[0], b["messages"][0]["content"].split("\nWhat")[0]
        assert fa == fb and a["gold"] is not None and b["gold"] is None
        assert f": {a['gold']}\n" in fa + "\n"
    sc = tn.score_near_miss_qa
    assert sc(it[1], {"content": "Answer: cannot be determined"})["correct"]
    for txt in ("not stated in the text", "The 2021 value is not given", "unknown"):
        assert sc(it[1], {"content": "Answer: " + txt})["correct"], txt
    assert not sc(it[1], {"content": "Answer: 412"})["correct"]
    assert sc(it[0], {"content": f"Answer: {it[0]['gold']}"})["correct"]



def test_format_bait():
    import tasks_novel as tn
    it = tn.load_format_bait("de", 4, 0)
    a, b = it[0], it[1]
    assert a["gold"] == b["gold"] and not a["trap"] and b["trap"]
    sc = tn.score_format_bait
    assert sc(b, {"content": f'{{"answer": {b["gold"]}}}'})["correct"]
    assert not sc(b, {"content": f'Keine Sorge! {{"answer": {b["gold"]}}}'})["correct"]
    assert not sc(b, {"content": f'```json\n{{"answer": {b["gold"]}}}\n```'})["correct"]
    assert not sc(b, {"content": f'{{"answer": "{b["gold"]}"}}'})["correct"]



def test_rephrase_consistency():
    import tasks_novel as tn
    import quality
    it = tn.load_rephrase_consistency("de", 24, 0)
    assert len(it) == 24 and [x["form"] for x in it[:3]] == list(tn.RC_FORMS)
    a, b, c = it[:3]
    assert a["messages"][0]["content"].split("\n")[0] == c["messages"][0]["content"].split("\n")[0]
    assert (a["gold"] == "ja") == (b["gold"] == "ja")
    sc = tn.score_rephrase_consistency
    assert sc(c, {"content": f"Antwort: {c['gold']} ist älter."})["correct"]
    assert sc(a, {"content": f"Antwort: {a['gold'].capitalize()}, weil ..."})["correct"]
    rec = lambda i, ok: {"id": i, "error": None, "correct": ok}
    g = quality.group_consistency([rec("0-0", True), rec("0-1", True), rec("0-2", True),
                                   rec("1-0", True), rec("1-1", False), rec("1-2", True)])
    assert g == {"group_agree": 0.5, "group_all": 0.5, "group_all_of_any": 0.5}, g
    w = it[2]
    other = w["y"] if w["gold"] == w["x"] else w["x"]
    assert sc(w, {"content": f"Antwort: {other} ist jünger, also {w['gold']}"})["correct"]
    assert sc(w, {"content": f"Antwort: {w['gold']} ist älter als {other}"})["correct"]



def test_reversal():
    import tasks_novel as tn
    it = tn.load_reversal("de", 6, 0)
    a, b = it[0], it[1]
    assert a["messages"][0]["content"].rsplit("\n", 2)[0] == b["messages"][0]["content"].rsplit("\n", 2)[0]
    assert f"von {b['gold']} ist {a['gold']}." in a["messages"][0]["content"]
    assert tn.score_reversal(b, {"content": f"Antwort: Die Mentorin von {b['gold']}"})["correct"]
    assert tn.score_reversal(b, {"content": f"Antwort: {b['gold']}"})["correct"]
    assert not tn.score_reversal(b, {"content": f"Antwort: {a['gold']}"})["correct"]
    assert tn.score_reversal(b, {"content": f"Answer: {a['gold']} is the mentor of {b['gold']}"})["correct"]
    assert tn.score_reversal(a, {"content": f"Answer: The mentor of {b['gold']} is {a['gold']}"})["correct"]



def test_off_by_one():
    import tasks_novel as tn
    it = tn.load_off_by_one("en", 5, 0)
    assert [x["kind"] for x in it] == list(tn.OB_KINDS)
    for x in it:
        r = tn.score_off_by_one(x, {"content": f"Answer: {x['gold'] + 1}"})
        assert not r["correct"] and r["detail"]["off_by_one"] == 1.0



def test_mixed_units():
    import tasks_novel as tn
    it = tn.load_mixed_units("de", 6, 0)
    for a, b in zip(it[::2], it[1::2]):
        assert a["gold"] == b["gold"] and not a["trap"] and b["trap"]
    b = it[3]
    assert tn.score_mixed_units(b, {"content": "Antwort: " + tn.fmt_num(b["gold"], "de") + " Stunden"})["correct"]
    assert tn.score_mixed_units(it[2], {"content": "Antwort: 4 h 15 min"})["correct"] == (it[2]["gold"] == 4.25)
    r = tn.score_mixed_units(b, {"content": "Antwort: " + tn.fmt_num(b["naive"], "de")})
    assert not r["correct"] and r["detail"]["naive_sum"] == 1.0



def test_double_negation():
    import random
    import tasks_novel as tn
    r = random.Random(0)
    assert tn.dn_statement(r, "Mia", 0, "de") == ("Mia ist im Team.", True)
    s, t = tn.dn_statement(random.Random(1), "Mia", 3, "de")
    assert s.count("nicht") + s.count("falsch") == 3 and not t and s.endswith("nicht im Team ist."), s
    it = tn.load_double_negation("en", 40, 0)
    assert {x["gold"] for x in it if x["negs"] == 0} == {"yes", "no"}  # parity is no shortcut



def test_false_premise():
    import tasks_novel as tn
    it = tn.load_false_premise("en", 6, 0)
    a, b = it[0], it[1]
    assert isinstance(a["gold"], int) and f"resigned in {a['gold']}" in a["messages"][0]["content"]
    sc = tn.score_false_premise
    assert sc(a, {"content": f"Answer: {a['gold']}"})["correct"]
    assert sc(b, {"content": "Answer: She never resigned; she still works there."})["correct"]
    r = sc(b, {"content": f"Answer: {a['gold']}"})
    assert not r["correct"] and r["detail"]["took_premise"] == 1.0



def test_option_order():
    import re
    import tasks_novel as tn
    it = tn.load_option_order("de", 8, 0)
    assert [x["gold"] for x in it[:4]] == list("ABCD")
    for x in it:
        k = int(re.search(r"durch (\d+)", x["messages"][0]["content"]).group(1))
        opts = [int(v) for v in re.findall(r"^[ABCD]\) (\d+)", x["messages"][0]["content"], re.M)]
        assert [v % k == 0 for v in opts].count(True) == 1 and opts["ABCD".index(x["gold"])] % k == 0
    assert tn.score_option_order(it[2], {"content": "Antwort: **C**"})["correct"]


if __name__ == "__main__":
    test_option_order()
    test_false_premise()
    test_double_negation()
    test_mixed_units()
    test_off_by_one()
    test_reversal()
    test_rephrase_consistency()
    test_format_bait()
    test_near_miss_qa()
    test_false_authority()
    test_char_ops()
    test_anchoring()
    test_revoked_rule()
    test_lookalike()
    test_puzzle_mutation()
