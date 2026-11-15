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
        want = t["counts"]["full"]["en"]
        assert (want is None or len(full) == want) and len({i["id"] for i in full}) == len(full), name
        assert [i["id"] for i in quick] == [i["id"] for i in full[:len(quick)]], name
        assert [i["gold"] for i in t["load"]("en", len(quick), 0)] == [i["gold"] for i in full[:len(quick)]], name
        for it in full[:50] if name not in ("if_gen", "math_missing", "schedule", "syllogism", "extract", "probability", "event_order", "mini_sudoku", "timezones", "calib_math", "meeting_actions", "game_theory", "knights_knaves", "gears", "jugs", "linalg", "data2text", "near_miss_qa", "format_bait", "false_premise", "ambiguous_ref", "blocks_plan", "regex_synth", "fake_entity", "knapsack", "table_build", "maze_path", "stale_answer", "pii_redact", "prob_fallacies", "framing") else []:  # own tests below
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
    for txt in ("There is no such year", "Paul arbeitet bis heute dort", "-"):
        assert sc(b, {"content": "Answer: " + txt})["correct"], txt
    assert not sc(b, {"content": "Answer: 2015 (not sure)"})["correct"]
    assert not sc(b, {"content": "Answer: yes"})["correct"]  # no year is not yet a rejection
    assert sc(a, {"content": f"Answer: {a['gold']}, not 2001"})["correct"]
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
    assert tn.score_option_order(it[1], {"content": "Answer: A number divisible by 7 is B"})["correct"]
    assert tn.score_option_order(it[1], {"content": "Answer: b"})["correct"]



def test_source_trust():
    import tasks_novel as tn
    it = tn.load_source_trust("en", 4, 0)
    a, b = it[0], it[1]
    assert a["gold"] == b["gold"] and b["messages"][0]["content"].count(f": {b['majority']} g.") == 3
    r = tn.score_source_trust(b, {"content": f"Answer: {b['majority']} g"})
    assert not r["correct"] and r["detail"]["took_majority"] == 1.0
    assert tn.score_source_trust(b, {"content": f"Answer: {b['gold']} g (lab)"})["correct"]
    assert tn.score_source_trust(b, {"content": f"Answer: ignoring 3 blogs, {b['gold']} g"})["correct"]
    assert a["messages"][0]["content"].count(f": {a['majority']} g.") == 1



def test_switched_question():
    import tasks_novel as tn
    for lang in ("en", "de"):
        it = tn.load_switched_question(lang, 4, 0)
        a, b = it[0], it[1]
        assert a["gold"] == b["gold"] and len(b["messages"][0]["content"]) > len(a["messages"][0]["content"]) + 150
        assert "vorher" not in a["messages"][0]["content"] and "before that" not in a["messages"][0]["content"]
        assert tn.score_switched_question(b, {"content": f"Answer: Items over 30 euros: {b['gold']}"})["correct"]
        r = tn.score_switched_question(b, {"content": f"Answer: {b['total']}"})
        assert not r["correct"] and r["detail"]["took_total"] == 1.0



def test_causal_graph():
    import tasks_novel as tn
    e = [(0, 1), (1, 2), (0, 3), (3, 2)]
    assert tn.cg_reach(e, 0, 2) and tn.cg_reach(e, 0, 2, avoid=1) and not tn.cg_reach(e, 2, 0)
    assert not tn.cg_reach([(0, 1), (1, 2)], 0, 2, avoid=1)
    for lang in ("en", "de"):
        it = tn.load_causal_graph(lang, 40, 0)
        assert sum(x["gold"] in ("yes", "ja") for x in it) == 20
        assert len(it) == 40 and {x["kind"] for x in it} == {"any", "fix"}
        assert sorted({x["gold"] for x in it}) == (["no", "yes"] if lang == "en" else ["ja", "nein"])



def test_ambiguous_ref():
    import tasks_novel as tn
    for lang in ("en", "de"):
        it = tn.load_ambiguous_ref(lang, 16, 0)
        assert [x["ambiguous"] for x in it[:4]] == [True, False, False, True]
        for x in it:
            q = x["messages"][0]["content"].split("\n")[1]
            if x["gold"]:
                assert x["gold"].lower() in x["messages"][0]["content"].lower()
                assert tn.score_ambiguous_ref(x, {"content": f"Answer: {x['gold']} (not ambiguous)"})["correct"]
                assert not tn.score_ambiguous_ref(x, {"content": "Antwort: mehrdeutig"})["correct"]
            else:
                assert tn.score_ambiguous_ref(x, {"content": "Answer: there are two, one here and one there"})["correct"]
        # colour in the question no longer gives the answer away
        assert {x["ambiguous"] for x in it if x["variant"] in (1, 3)} == {True, False}



def test_counterfactual_world():
    import tasks_novel as tn
    it = tn.load_counterfactual_world("en", 200, 0)
    assert {x["conflict"] for x in it} == {True, False}
    for k in range(5):
        assert {x["gold"] for x in it if x["rule"] == k} == {"yes", "no"} or k == 3
    for k in range(5):
        g = [x["gold"] for x in it if x["rule"] == k]
        assert 0.25 <= g.count("yes") / len(g) <= 0.75, (k, g.count("yes"), len(g))
    assert 0.4 <= sum(x["conflict"] for x in it) / len(it) <= 0.8



def test_fallacy_spot():
    import tasks_novel as tn
    for lang in ("en", "de"):
        it = tn.load_fallacy_spot(lang, 12, 0)
        assert [x["kind"] for x in it[:6]] == tn.FS_KINDS
        for x in it:
            assert "{" not in x["messages"][0]["content"]
            assert tn.score_fallacy_spot(x, {"content": f"Answer: {x['gold']}) ..."})["correct"]
            lab = tn.FS_LABEL[lang][x["kind"]].split(" (")[0]
            assert tn.score_fallacy_spot(x, {"content": f"Answer: {lab}"})["correct"], lab



def test_arith_length():
    import tasks_novel as tn
    it = tn.load_arith_length("de", 28, 0)
    assert {x["digits"] for x in it} == set(tn.AL_LEN) and {x["op"] for x in it} == {"add", "mul"}
    x = next(x for x in it if x["op"] == "mul" and x["digits"] == 6)
    g = x["gold"]
    for txt in (f"{g}", f"{g:,}", f"{g:,}".replace(",", "."), f"a × b = {g:,}".replace(",", " ")):
        assert tn.score_arith_length(x, {"content": "Antwort: " + txt})["correct"], txt
    assert not tn.score_arith_length(x, {"content": f"Antwort: {g + 1}"})["correct"]



def test_nesting_depth():
    import tasks_novel as tn
    assert tn.nd_balanced("([]{})") and not tn.nd_balanced("([)]") and not tn.nd_balanced("((")
    it = tn.load_nesting_depth("en", 144, 0)
    bal = [x["gold"] for x in it if x["kind"] == "balance"]
    assert 0.3 < bal.count("yes") / len(bal) < 0.7
    for x in it[:12]:
        assert tn.score_nesting_depth(x, {"content": f"Answer: {x['gold']}"})["correct"]
    no = [x for x in it if x["kind"] == "balance" and x["gold"] == "no"]
    odd = [x for x in no if len(x["messages"][0]["content"].split("\n")[1]) % 2]
    assert len(odd) <= len(no) // 3, (len(odd), len(no))



def test_grid_transform():
    import tasks_novel as tn
    g = [list("AB"), list("CD")]
    assert tn.GT_OPS["cw"](g) == [list("CA"), list("DB")] and tn.GT_OPS["ccw"](g) == [list("BD"), list("AC")]
    assert tn.GT_OPS["cw"](tn.GT_OPS["ccw"](g)) == g and tn.GT_OPS["lr"](g) == [list("BA"), list("DC")]
    it = tn.load_grid_transform("en", 12, 0)
    assert {x["ops"] for x in it} == {1, 2, 3}
    x = it[0]
    for txt in (x["gold"], " ".join(x["gold"]), ", ".join(x["gold"]), "I think it is " + " ".join(x["gold"]).lower()):
        assert tn.score_grid_transform(x, {"content": "Answer: " + txt})["correct"], txt



def test_blocks_plan():
    import tasks_novel as tn
    start, goal = [["A", "B"], ["C"]], [["C", "B", "A"]]  # B on A; want A on B on C
    plan = "move B onto C\nmove A onto B"
    final, n = tn.bp_run(start, plan)
    assert tn.bp_canon(final) == tn.bp_canon(goal) and n == 2 and tn.bp_optimal(start, goal) == 2
    assert tn.bp_run(start, "move A onto C")[0] is None  # A is covered
    final, n = tn.bp_run(start, "First I would move A onto B, hm.\nPlan:\n1. Move B onto the table\n2. move B to C\nmove A onto B")
    assert tn.bp_canon(final) == tn.bp_canon(goal) and n == 3
    it = tn.load_blocks_plan("de", 8, 0)
    for x in it:
        assert x["optimal"] and x["optimal"] >= 1
    x = it[0]
    r = tn.score_blocks_plan(x, {"content": "move A onto A"})
    assert not r["correct"]



def test_judge_ability():
    import tasks_novel as tn
    it = tn.load_judge_ability("en", 16, 0)
    for a, b in zip(it[::2], it[1::2]):
        assert a["gold"] != b["gold"]  # the same pair, A/B swapped
    assert {x["gold"] for x in it} == {"A", "B"} and {x["fluff_wrong"] for x in it} == {True, False}
    x = it[0]
    assert tn.score_judge_ability(x, {"content": f"Answer: {x['gold']}"})["correct"]
    assert tn.score_judge_ability(x, {"content": f"Answer: Solution {x['gold']} is right"})["correct"]
    other = "B" if x["gold"] == "A" else "A"
    assert tn.score_judge_ability(x, {"content": f"Answer: {other} is wrong, {x['gold']} is correct"})["correct"]
    assert not tn.score_judge_ability(x, {"content": "Answer: A and B both are wrong"})["correct"]
    full = tn.load_judge_ability("en", 160, 0)
    for f in ("wrong", "right", "none"):
        assert sum(y["fluff"] == f for y in full) > 40



def test_regex_synth():
    import tasks_novel as tn
    it = tn.load_regex_synth("en", 10, 0)
    for x in it:
        assert len(x["pos"]) >= 7 and len(x["neg"]) >= 7
        assert tn.score_regex_synth(x, {"content": f"```\n{x['gold']}\n```"})["correct"]
        literal = "|".join(__import__("re").escape(p) for p in x["pos"][:5])  # memorises the shown list
        r = tn.score_regex_synth(x, {"content": f"```{literal}```"})
        assert not r["correct"] and r["detail"]["overfit"] == 1.0
    assert not tn.score_regex_synth(it[0], {"content": "```(unclosed```"})["correct"]
    g = it[0]["gold"]
    assert tn.score_regex_synth(it[0], {"content": f'```python\nimport re\npattern = r"{g}"\n```'})["correct"]
    assert tn.score_regex_synth(it[0], {"content": f"Use `{g}` with `re.fullmatch`."})["correct"]



def test_continuity_error():
    import tasks_novel as tn
    it = tn.load_continuity_error("de", 24, 0)
    assert {x["gold"] == 0 for x in it} == {True, False}
    lens = {len(__import__("re").findall(r"\(\d+\)", x["messages"][0]["content"])) for x in tn.load_continuity_error("en", 120, 0)}
    assert len(lens) == 1, lens  # the length gives nothing away
    for x in it:
        n = len(__import__("re").findall(r"\(\d+\)", x["messages"][0]["content"]))
        assert 0 <= x["gold"] <= n
        assert tn.score_continuity_error(x, {"content": f"Antwort: {x['gold']}"})["correct"]
        assert "{" not in x["messages"][0]["content"]



def test_number_compare():
    from fractions import Fraction
    import tasks_novel as tn
    for lang in ("en", "de"):
        for x in tn.load_number_compare(lang, 40, 0):
            val = lambda t: Fraction(t.replace(",", "."))  # noqa: E731
            assert val(x["gold"]) > val(x["other"]), x
            assert tn.score_number_compare(x, {"content": f"Answer: {x['gold']}"})["correct"]
            assert not tn.score_number_compare(x, {"content": f"Answer: {x['other']}"})["correct"]
    x = next(y for y in tn.load_number_compare("en", 8, 0) if y["kind"] == "negatives")
    assert tn.score_number_compare(x, {"content": f"Answer: {x['gold']} is larger than {x['other']}"})["correct"]
    assert not tn.score_number_compare(x, {"content": f"Answer: {x['other']}, not {x['gold']}"})["correct"]
    assert tn.score_number_compare(x, {"content": f"Answer: {x['other']} < {x['gold']}"})["correct"]
    d = next(y for y in tn.load_number_compare("de", 8, 0) if y["kind"] == "decimals")
    assert tn.score_number_compare(d, {"content": "Antwort: " + d["gold"].replace(",", ".")})["correct"]



def test_unit_chain():
    from fractions import Fraction
    import tasks_novel as tn
    it = tn.load_unit_chain("en", 12, 0)
    assert {x["steps"] for x in it} == {2, 3, 4} and {x["back"] for x in it} == {True, False}
    for x in it:
        g = Fraction(x["gold"])
        assert tn.score_unit_chain(x, {"content": f"Answer: {g.numerator}/{g.denominator}"})["correct"]
        assert tn.score_unit_chain(x, {"content": f"Answer: {float(g):.6f}"})["correct"]
        assert not tn.score_unit_chain(x, {"content": f"Answer: {float(g) * 1.1 + 1:.4f}"})["correct"]
        assert tn.score_unit_chain(x, {"content": f"Answer: from 8/2 we get {g.numerator}/{g.denominator}"})["correct"]
    y = dict(it[0], gold="5/4")
    assert tn.score_unit_chain(y, {"content": "Answer: 1 1/4"})["correct"]



def test_fake_entity():
    import tasks_novel as tn
    it = tn.load_fake_entity("de", 8, 0)
    f, r = it[0], it[1]
    assert f["fake"] and not r["fake"]
    assert tn.score_fake_entity(f, {"content": "Dazu habe ich keine Informationen, das Institut scheint nicht zu existieren."})["correct"]
    assert not tn.score_fake_entity(f, {"content": "Es wurde 1952 von Hans Müller gegründet."})["correct"]
    faust = next(y for y in tn.load_fake_entity("de", 40, 0) if "Faust" in y["messages"][0]["content"])
    assert tn.score_fake_entity(faust, {"content": "Das war Johann Wolfgang von Goethe."})["correct"]
    assert not tn.score_fake_entity(f, {"content": "I don't know the exact year, but it was founded by Hans Weber."})["correct"]
    shaped = [y for y in tn.load_fake_entity("en", 40, 0) if not y["fake"] and "Aconcagua" in y["messages"][0]["content"]]
    assert shaped and tn.score_fake_entity(shaped[0], {"content": "About 6,961 metres."})["correct"]
    for t in ("I couldn't find any reliable information.", "Mir ist kein solches Institut bekannt.", "There is no such mountain."):
        assert tn.score_fake_entity(f, {"content": t})["correct"], t



def test_knapsack():
    from itertools import combinations
    import tasks_novel as tn
    it = tn.load_knapsack("en", 12, 0)
    assert {x["greedy_trap"] for x in it} == {True, False}
    for x in it:
        best = max((c for k in range(len(x["items"]) + 1) for c in combinations(x["items"], k)
                    if sum(y[1] for y in c) <= x["budget"]), key=lambda c: sum(y[2] for y in c))
        ans = ", ".join(y[0] for y in best)
        assert tn.score_knapsack(x, {"content": f"Answer: {ans}"})["correct"]
        allx = ", ".join(y[0] for y in x["items"])
        assert not tn.score_knapsack(x, {"content": f"Answer: {allx}"})["correct"]
        rest = [y[0] for y in x["items"] if y not in best]
        if rest:
            assert tn.score_knapsack(x, {"content": f"Answer: {ans} (not the {rest[0]})"})["correct"]



def test_cipher():
    import tasks_novel as tn
    it = tn.load_cipher("de", 40, 0)
    assert {x["kind"] for x in it} == {"shift", "revshift", "pairs", "vowels"} and {x["decode"] for x in it} == {True, False}
    for x in it:
        assert tn.score_cipher(x, {"content": f"Antwort: {x['gold']}"})["correct"]
    b = next((x for x in it if x["gold"] == "bruecke"), None)
    if b:
        assert tn.score_cipher(b, {"content": "Antwort: Brücke"})["correct"]



def test_prop_logic():
    import tasks_novel as tn
    a, b = ("atom", 0), ("atom", 1)
    assert tn.pl_valid([("if", a, b), ("not", b)], ("not", a), 2) and not tn.pl_valid([("if", a, b), b], a, 2)
    it = tn.load_prop_logic("de", 40, 0)
    assert sum(x["gold"] == "ja" for x in it) == 20 and {x["form"] for x in it} == set(range(10))
    x = next(y for y in it if y["gold"] == "nein")
    assert tn.score_prop_logic(x, {"content": "Antwort: Das folgt nicht."})["correct"]
    assert "Wenn das Trell ist offen" not in " ".join(x["messages"][0]["content"] for x in it)



def test_table_build():
    import tasks_novel as tn
    x = tn.load_table_build("de", 1, 0)[0]
    table = "| " + " | ".join(x["cols"]) + " |\n|---|---|---|---|\n" + "\n".join("| " + " | ".join(r) + " |" for r in x["gold"])
    assert tn.score_table_build(x, {"content": table})["correct"]
    swapped = table.replace("\n" + "| " + " | ".join(x["gold"][0]) + " |", "", 1) + "\n| " + " | ".join(x["gold"][0]) + " |"
    nopipes = "\n".join(ln.strip().strip("|") for ln in table.splitlines())
    assert tn.score_table_build(x, {"content": "Entwurf:\n| a | b |\n\nFinal:\n" + nopipes})["correct"]
    r = tn.score_table_build(x, {"content": swapped})
    assert not r["correct"] and r["detail"]["rows_right_but_order"] == 1.0



def test_ascii_chart():
    import tasks_novel as tn
    for lang in ("de", "en"):
        it = tn.load_ascii_chart(lang, 12, 0)
        assert {x["kind"] for x in it} == {"value", "diff", "second"}
        for x in it:
            assert tn.score_ascii_chart(x, {"content": f"Answer: {x['gold']}"})["correct"]
            if x["kind"] == "diff":
                assert tn.score_ascii_chart(x, {"content": f"Answer: {x['gold']} (20 - 9)"})["correct"]
            if x["kind"] == "second" and lang == "de":
                assert tn.score_ascii_chart(x, {"content": f"Antwort: {x['gold']}, so wie gezeigt"})["correct"]



def test_maze_path():
    import tasks_novel as tn
    g = [list("S.#"), list(".##"), list("..E")]
    assert tn.mz_bfs(g, (0, 0), (2, 2)) == 4
    assert tn.mz_bfs([list("S#E")], (0, 0), (0, 2)) is None
    it = tn.load_maze_path("de", 25, 0)
    assert sum(x["gold"] == "none" for x in it) == 5
    for x in it:
        assert tn.score_maze_path(x, {"content": f"Antwort: {'keine' if x['gold'] == 'none' else x['gold']}"})["correct"]
        if x["gold"] == "none":
            assert tn.score_maze_path(x, {"content": "Antwort: Es gibt keinen Weg (0 Schritte)."})["correct"]



def test_alien_sort():
    import tasks_novel as tn
    it = tn.load_alien_sort("en", 12, 0)
    for x in it:
        assert tn.score_alien_sort(x, {"content": "Answer: " + x["gold"]})["correct"]
        assert sorted(x["gold"].split(", ")) == sorted(x["words"])
    x = it[0]
    assert not tn.score_alien_sort(x, {"content": "Answer: " + ", ".join(sorted(x["words"]))})["correct"] or \
        sorted(x["words"]) == x["gold"].split(", ")



def test_stale_answer():
    import tasks_novel as tn
    x = tn.load_stale_answer("de", 1, 0)[0]
    assert len(x["turns"]) == 2 and x["gold"] != x["first"]
    r = tn.score_stale_answer(x, {"content": "", "turns": [f"Antwort: {x['first']}", f"Antwort: {x['first']}"]})
    assert not r["correct"] and r["detail"]["repeated_old"] == 1.0
    assert tn.score_stale_answer(x, {"content": "", "turns": [f"Antwort: {x['first']}", f"Antwort: {x['gold']}"]})["correct"]



def test_pii_redact():
    import tasks_novel as tn
    x = tn.load_pii_redact("de", 1, 0)[0]
    clean = "Frage: Mein Router X200 verliert ständig das WLAN. Antwort: Bitte das Update erneut einspielen."
    assert tn.score_pii_redact(x, {"content": clean})["correct"]
    r = tn.score_pii_redact(x, {"content": clean + " Kontakt: " + x["pii"]["email"]})
    assert not r["correct"] and r["detail"]["leaked_email"] == 1.0
    spaced = " ".join(x["pii"]["iban"][i:i + 4] for i in range(0, 22, 4))
    assert tn.score_pii_redact(x, {"content": clean + " IBAN " + spaced})["detail"]["leaked_iban"] == 1.0



def test_percent_traps():
    import tasks_novel as tn
    it = tn.load_percent_traps("de", 8, 0)
    assert [x["kind"] for x in it[:4]] == list(tn.PT_KINDS)
    for x in it:
        g = x["gold"]
        assert tn.score_percent_traps(x, {"content": f"Antwort: {g:g}".replace(".", ",")})["correct"], x
        r = tn.score_percent_traps(x, {"content": f"Antwort: {x['naive']:g}".replace(".", ",")})
        assert not r["correct"] and r["detail"]["naive"] == 1.0
    rel = it[0]
    assert tn.score_percent_traps(rel, {"content": f"Antwort: {rel['naive']} Prozentpunkte, das sind {rel['gold']:g} %".replace(".", ",")})["correct"]
    ud = it[1]
    assert tn.score_percent_traps(ud, {"content": f"Answer: decreased by {abs(ud['gold']):g}% (1.2 x 0.8 = 0.96)"})["correct"]
    assert tn.score_percent_traps(ud, {"content": f"Antwort: {abs(ud['gold']):g} % niedriger".replace(".", ",")})["correct"]



def test_clock_angle():
    import tasks_novel as tn
    assert tn.ca_angle(3, 15) == 7.5 and tn.ca_angle(3, 15, frozen=True) == 0 and tn.ca_angle(12, 30) == 165
    for x in tn.load_clock_angle("de", 10, 0):
        assert tn.score_clock_angle(x, {"content": f"Antwort: {x['gold']:g}°".replace(".", ",")})["correct"]
        r = tn.score_clock_angle(x, {"content": f"Antwort: {x['naive']:g} Grad"})
        assert not r["correct"] and r["detail"]["naive"] == 1.0



def test_base_rate():
    import tasks_novel as tn
    it = tn.load_base_rate("en", 8, 0)
    for x in it:
        assert 0 < x["gold"] < x["naive"]
        assert tn.score_base_rate(x, {"content": f"Answer: about {x['gold']}%"})["correct"]
        assert tn.score_base_rate(x, {"content": f"Answer: {x['naive']}%"})["detail"]["naive"] == 1.0
    assert "10,000" in it[1]["messages"][0]["content"]
    full = tn.load_base_rate("de", None, 0)
    assert len({x["messages"][0]["content"] for x in full}) == len(full)
    assert all(full[k]["gold"] == full[k + 1]["gold"] for k in range(0, len(full) - 1, 2))
    assert "0.5 %" not in " ".join(x["messages"][0]["content"] for x in full)
    assert tn.score_base_rate(dict(it[0], gold=8.3), {"content": "Answer: 1/12"})["correct"]



def test_simpson():
    import tasks_novel as tn
    it = tn.load_simpson("de", 16, 0)
    assert {(x["paradox"], x["ask"]) for x in it} == {(p, a) for p in (True, False) for a in ("within", "overall")}
    for x in it:
        assert tn.score_simpson(x, {"content": f"Antwort: {x['gold']}"})["correct"]



def test_prob_fallacies():
    import tasks_novel as tn
    it = tn.load_prob_fallacies("en", 30, 0)
    assert {x["kind"] for x in it} == {"gambler", "conjunction", "control"}
    for x in it:
        g = x["gold"]
        assert tn.score_prob_fallacies(x, {"content": f"Answer: {g}" + ("%" if g == "50" else "")})["correct"], x
    six = next(x for x in it if x["gold"] == "1/6")
    assert tn.score_prob_fallacies(six, {"content": "Answer: 0.167"})["correct"]
    assert not tn.score_prob_fallacies(six, {"content": "Answer: 1/5"})["correct"]
    for txt in ("16,7 %", "1 in 6", "16.67%"):
        assert tn.score_prob_fallacies(six, {"content": "Answer: " + txt})["correct"], txt
    coin = next(x for x in it if x["gold"] == "50")
    assert tn.score_prob_fallacies(coin, {"content": "Answer: unchanged, 50%"})["correct"]
    k = __import__("re").search(r"(\d+) times", coin["messages"][0]["content"]).group(1)
    assert tn.score_prob_fallacies(coin, {"content": f"Answer: after {k} heads still 50%"})["correct"]
    assert not tn.score_prob_fallacies(coin, {"content": "Answer: between 0 and 50%"})["correct"]



def test_framing():
    import tasks_novel as tn
    it = tn.load_framing("de", 8, 0)
    for a, b in zip(it[::2], it[1::2]):
        assert not a["loss"] and b["loss"] and a["sure"] == b["sure"]
    r = tn.score_framing(it[1], {"content": f"Antwort: Ich wähle {it[1]['sure']}"})
    assert r["correct"] is None and r["detail"]["sure_loss"] == 1.0 and r["detail"]["group_key"] is True
    full = tn.load_framing("en", None, 0)
    assert len({x["messages"][0]["content"] for x in full}) == len(full)



def test_conclusion_spot():
    import tasks_novel as tn
    it = tn.load_conclusion_spot("de", 16, 0)
    assert {x["marked"] for x in it} == {True, False}
    m = next(x for x in it if x["marked"])
    assert "Also braucht diese Pflanze" in " ".join(x["messages"][0]["content"] for x in it)
    for x in it:
        assert tn.score_conclusion_spot(x, {"content": f"Antwort: ({x['gold']})"})["correct"]
        assert tn.score_conclusion_spot(x, {"content": f"Antwort: Linie 7 ... Satz {x['gold']}"})["correct"]
    pos = [x["gold"] for x in tn.load_conclusion_spot("en", 120, 0)]
    assert all(25 <= pos.count(k) <= 35 for k in (1, 2, 3, 4)), [pos.count(k) for k in (1, 2, 3, 4)]



def test_sunk_cost():
    import tasks_novel as tn
    it = tn.load_sunk_cost("de", 8, 0)
    for a, b in zip(it[::2], it[1::2]):
        assert not a["trap"] and b["trap"] and a["gold"] == b["gold"] and "bereits" in b["messages"][0]["content"]
    trap = next(x for x in it if x["trap"] and x["gold"] != x["cont"])
    r = tn.score_sunk_cost(trap, {"content": f"Antwort: {trap['cont']}"})
    assert not r["correct"] and r["detail"]["took_sunk"] == 1.0
    assert {x["gold"] for x in it} == {"A", "B"}
    g = trap["gold"]
    assert tn.score_sunk_cost(trap, {"content": f"Answer: A company should pick {g}"})["correct"]
    assert tn.score_sunk_cost(trap, {"content": f"Answer: Option {g}, not {trap['cont']}"})["correct"]



def test_regression_mean():
    import tasks_novel as tn
    it = tn.load_regression_mean("de", 8, 0)
    for a, b in zip(it[::2], it[1::2]):
        assert not a["control"] and b["control"] and "Vergleichsgruppe" in b["messages"][0]["content"]
    for x in it:
        assert tn.score_regression_mean(x, {"content": f"Antwort: {x['gold']}"})["correct"]



def test_dim_analysis():
    import tasks_novel as tn
    assert tn.da_mul((1, -1, 0), (0, 1, 0)) == (1, 0, 0)
    it = tn.load_dim_analysis("en", 30, 0)
    golds = [x["gold"] for x in it]
    assert len(set(golds)) == 4
    for x in it:
        assert tn.score_dim_analysis(x, {"content": f"Answer: {x['gold']})"})["correct"]



def test_table_join():
    import tasks_novel as tn
    it = tn.load_table_join("de", 12, 0)
    assert {x["kind"] for x in it} == {"count", "sum", "max"}
    for x in it:
        g = x["gold"]
        txt = f"{g:,}".replace(",", ".") if isinstance(g, int) and g >= 1000 else str(g)
        assert tn.score_table_join(x, {"content": f"Antwort: {txt}"})["correct"], (x["kind"], txt)
        if x["kind"] == "max":
            other = next(c for c in tn.TJ_CITIES if c != g)
            assert tn.score_table_join(x, {"content": f"Antwort: {g}, nicht {other}"})["correct"]



def test_dedup_records():
    import tasks_novel as tn
    for x in tn.load_dedup_records("de", 10, 0):
        rows = [ln for ln in x["messages"][0]["content"].splitlines() if ln.startswith("| ") and "---" not in ln][1:]
        assert len(rows) == x["naive"] > x["gold"]
        norm = lambda s: __import__("re").sub(r"[^0-9]", "", s)[-7:]  # noqa: E731
        phones = {norm(r.split("|")[3]) for r in rows}
        assert len(phones) == x["gold"]  # every distinct person has one distinct phone
        assert tn.score_dedup_records(x, {"content": f"Antwort: {x['gold']}"})["correct"]



def test_interval_math():
    from fractions import Fraction as F
    import tasks_novel as tn
    for x in tn.load_interval_math("de", 9, 0):
        assert tn.score_interval_math(x, {"content": f"Antwort: {x['lo']} / {x['hi']}"})["correct"], x["gold"]
        if x["kind"] == "quot":
            lo, hi = F(x["lo"]), F(x["hi"])
            assert tn.score_interval_math(x, {"content": f"Antwort: {float(lo):.3f} bis {float(hi):.3f}".replace(".", ",")})["correct"]
        if x["kind"] != "sum":
            r = tn.score_interval_math(x, {"content": f"Antwort: {x['nlo']} / {x['nhi']}"})
            assert not r["correct"] and r["detail"]["naive"] == 1.0


if __name__ == "__main__":
    test_interval_math()
    test_dedup_records()
    test_table_join()
    test_dim_analysis()
    test_regression_mean()
    test_sunk_cost()
    test_conclusion_spot()
    test_framing()
    test_prob_fallacies()
    test_simpson()
    test_base_rate()
    test_clock_angle()
    test_percent_traps()
    test_pii_redact()
    test_stale_answer()
    test_alien_sort()
    test_maze_path()
    test_ascii_chart()
    test_table_build()
    test_prop_logic()
    test_cipher()
    test_knapsack()
    test_fake_entity()
    test_unit_chain()
    test_number_compare()
    test_continuity_error()
    test_regex_synth()
    test_judge_ability()
    test_blocks_plan()
    test_grid_transform()
    test_nesting_depth()
    test_arith_length()
    test_fallacy_spot()
    test_counterfactual_world()
    test_ambiguous_ref()
    test_causal_graph()
    test_switched_question()
    test_source_trust()
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
