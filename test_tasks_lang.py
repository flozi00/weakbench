"""tasks_lang (lang_switch, lang_quality) with a mock judge, plain asserts:

    python3 test_tasks_lang.py
"""
import json

import mock_server
import quality
import tasks_judge
import tasks_lang as tl

reply = lambda text, turns=None: {"content": text, "reasoning": "", "tool_calls": [], "finish_reason": "stop",  # noqa
                                  **({"turns": turns} if turns else {})}
DE = ("Der Himmel ist blau, weil das Licht der Sonne an den Molekülen der Luft gestreut wird. Das kurzwellige blaue "
      "Licht wird dabei stärker gestreut als das rote, und deshalb sehen wir es aus allen Richtungen.")
EN = ("The sky is blue because sunlight is scattered by the molecules of the air. Short blue wavelengths are "
      "scattered more than the red ones, and that is why we see blue light coming from every direction.")


def test_switch():
    for lang in ("de", "en"):
        items = tl.load_switch(lang, None, 0)
        assert len(items) == 4 * len(tl.TOPICS) and len({i["id"] for i in items}) == len(items)
        assert [i["id"] for i in tl.load_switch(lang, 60, 0)] == [i["id"] for i in items[:60]]
    assert quality.TASKS["lang_switch"]["counts"]["quick"] == {"de": 60, "en": 60}
    it = {i["scenario"]: i for i in tl.load_switch("de", 4, 0)}
    assert it["mirror"]["messages"][0]["content"] == tl.SYSTEM["en"]
    assert "German" in it["instructed"]["messages"][0]["content"] and len(it["switch"]["turns"]) == 2
    r = tl.score_switch(it["switch"], reply(DE, turns=[EN, DE]))
    assert r["correct"] and r["score"] == 1.0 and r["detail"]["first_turn_ok"] == 1.0, r
    r = tl.score_switch(it["mirror"], reply(EN))
    assert not r["correct"] and r["score"] == 0.0 and r["lang_ok"] is False, r
    r = tl.score_switch(it["mirror"], reply(DE + " " + DE + "\n" + EN))  # German reply that slips into English
    assert not r["correct"] and 0 < r["score"] < 1 and r["detail"]["wrong_sent_share"] > 0, r


def test_quality():
    verdict = {"errors": [{"type": "spelling", "quote": "fuer dich", "fix": "für dich"},
                          {"type": "grammar", "quote": "not in the text at all", "fix": "x"}],
               "fluency": 7, "task_fit": 9}
    url, stop = mock_server.serve(lambda path, body: {"content": "```json\n" + json.dumps(verdict) + "\n```",
                                                      "finish_reason": "stop"})
    try:
        tasks_judge.configure(url, "mock", 32768)
        items = tl.load_quality("de", None, 0)
        assert len(items) == len(tl.WRITING) and items[0]["id"] == tl.load_quality("en", 1, 0)[0]["id"]
        r = tl.score_quality(items[0], reply("Ein Brief  fuer dich, ueber alles."))
    finally:
        stop()
        tasks_judge.configure(None, None, None)
    d = r["detail"]
    assert r["correct"] is None and d["clean"] == 0.0 and r["score"] == 0.7 and d["err_spelling"] == 1 and d["judge_dropped"] == 1, r
    assert d["umlaut_sub"] == 2 and d["task_fit"] == 9


def test_grammar():
    for lang, total in (("de", 2250), ("en", 502)):  # distinct sentence pairs
        items = tl.load_grammar(lang, None, 0)
        assert len(items) == total and [i["id"] for i in tl.load_grammar(lang, 300, 0)] == [i["id"] for i in items[:300]]
        assert 0.4 < sum(i["gold"] == "A" for i in items) / total < 0.6
    it = items[0]
    assert tl.score_grammar(it, reply(f"Answer: {it['gold']}"))["correct"]
    for form in ("**Answer:** Sentence {g}.", "Antwort: Satz {g}", "Answer: ({g})"):
        assert tl.score_grammar(it, reply(form.format(g=it["gold"])))["correct"], form
    assert not tl.score_grammar(it, reply("Answer: A or B"))["correct"]
    other = "A" if it["gold"] == "B" else "B"
    r = tl.score_grammar(it, reply(f"Answer: {other}"))
    assert not r["correct"] and r["detail"][f"ok_{it['phen']}"] == 0.0


def test_spell():
    for lang in ("de", "en"):
        items = tl.load_spell(lang, None, 0)
        assert len(items) > 150 and [i["id"] for i in tl.load_spell(lang, 100, 0)] == [i["id"] for i in items[:100]]
        for it in items:
            bad = it["messages"][0]["content"].split("\n\n", 1)[1]
            assert bad != it["gold"] and 2 <= len(it["edits"]) <= 3
            r = tl.score_spell(it, reply(it["gold"]))
            assert r["correct"] and r["detail"]["exact"] == 1.0, (it["edits"], r)
            r = tl.score_spell(it, reply(bad))
            assert r["score"] == 0 and not r["correct"], (it["edits"], r)
    assert set(tl.load_spell("de", None, 0)[0]) >= {"gold", "edits"}
    assert {i["id"] for i in tl.load_spell("de", None, 0)} & {i["id"] for i in tl.load_spell("en", None, 0)}


def test_simplify():
    for lang in ("de", "en"):
        items = tl.load_simplify(lang, None, 0)
        assert len(items) == len(tl.NOTICES)
        for it in items:  # the source itself never passes: no gain
            assert not tl.score_simplify(it, reply(it["source"]))["correct"]
            assert tl.score_simplify(it, reply(it["source"]))["detail"]["facts_kept"] == 1.0, it["source"]
    assert tl.reading_ease("Ab dem 1. März zahlen Sie mehr.", "de")[1] == 6.0  # "1." does not end a sentence
    easy = ("Sie ziehen um. Dann müssen Sie sich anmelden. Das machen Sie beim Amt. Sie haben dafür zwei Wochen "
            "Zeit. Wenn Sie zu spät sind, kann das teuer sein. Es kann bis zu 1.000 Euro kosten.")
    r = tl.score_simplify(tl.load_simplify("de", 1, 0)[0], reply(easy))
    assert r["correct"] and r["score"] == 1.0, r
    r = tl.score_simplify(tl.load_simplify("de", 1, 0)[0], reply(easy.replace("1.000", "viel")))
    assert not r["correct"] and r["detail"]["facts_kept"] == 0.5


def test_register():
    items = {i["id"]: i for i in tl.load_register("de", None, 0)}
    assert len(items) == 20
    for it in items.values():  # the unchanged source keeps its old address: never correct
        r = tl.score_register(it, reply(it["messages"][0]["content"].split("\n\n", 1)[1]))
        assert not r["correct"] and r["detail"]["new_used"] == 0.0, it["id"]
    r = tl.score_register(items["sie:0"], reply("Hallo Herr Tom, können Sie mir bis Freitag Ihren Bericht schicken? "
                                                "Ich brauche ihn für das Meeting um 10 Uhr."))
    assert r["correct"], r
    r = tl.score_register(items["du:0"], reply("Liebe Frau Weber, kannst du mir bitte bis Freitag deine Unterlagen "
                                               "schicken? Sie sind wichtig. Danke!"))
    assert r["correct"], r  # sentence-initial "Sie" (= they) is not formal address
    r = tl.score_register(items["du:0"], reply("Liebe Frau Weber, kannst du mir bis Freitag Ihre Unterlagen senden?"))
    assert not r["correct"] and r["detail"]["old_left"] == 1.0


def test_de_case():
    import collections
    items = tl.load_de_case("de", 240, 0)
    assert set(collections.Counter((i["case"], i["art"]) for i in items).values()) == {30}
    table = {("nom", "m", "def"): "der alte", ("nom", "m", "indef"): "ein alter", ("nom", "n", "indef"): "ein altes",
             ("acc", "m", "def"): "den alten", ("acc", "n", "indef"): "ein altes", ("acc", "f", "indef"): "eine alte",
             ("dat", "f", "def"): "der alten", ("dat", "m", "indef"): "einem alten", ("gen", "f", "indef"): "einer alten"}
    for (case, g, art), want in table.items():
        got = f"{(tl.DEF if art == 'def' else tl.INDEF)[case][g]} {tl.adj_form('alt', tl.adj_ending(case, g, art))}"
        assert got == want, (case, g, art, got)
    assert tl.adj_form("teuer", "en") == "teuren"
    it = items[0]
    assert tl.score_de_case(it, reply(f"Antwort: {it['gold']}"))["correct"]
    assert tl.score_de_case(it, reply(f"**Antwort:** {it['gold'].capitalize()}"))["correct"]


def test_idioms():
    for lang in ("de", "en"):
        items = tl.load_idioms(lang, None, 0)
        assert len(items) == 20 and len({i["gold"] for i in items}) == 4
        for it in items:
            assert tl.score_idioms(it, reply(f"Answer: {it['gold']}"))["correct"]
            assert tl.score_idioms(it, reply(f"Answer: {it['gold']}) He bought a thing"))["correct"]
            r = tl.score_idioms(it, reply(f"Antwort: {it['literal']}"))
            assert not r["correct"] and r["detail"]["literal"] == 1.0


def test_de_verbs():
    pf = tl.past_form
    assert [pf("ging", p, "prt") for p in range(6)] == ["ging", "gingst", "ging", "gingen", "gingt", "gingen"]
    assert [pf("dachte", p, "prt") for p in range(6)] == ["dachte", "dachtest", "dachte", "dachten", "dachtet", "dachten"]
    assert pf("fand", 4, "prt") == "fandet" and pf("hätte", 3, "k2") == "hätten" and pf("wäre", 1, "k2") == "wärest"
    it = tl.load_de_verbs("de", 3, 0)[1]
    assert it["kind"] == "perf" and tl.score_de_verbs(it, reply(f"Antwort: {it['gold']}"))["correct"]
    k2 = {"id": "x", "gold": "wärest", "kind": "k2"}
    assert tl.score_de_verbs(k2, reply("Antwort: wärst"))["correct"]
    assert not tl.score_de_verbs({"gold": "gingest", "kind": "k2"}, reply("Antwort: gingst"))["correct"]
    assert all("Person: sie (Plural)" in i["messages"][0]["content"] for i in tl.load_de_verbs("de", 240, 0)
               if "Person: sie" in i["messages"][0]["content"])


def test_roundtrip():
    it = tl.load_roundtrip("de", 3, 0)[0]
    assert len(it["turns"]) == 2 and it["orig"] in it["turns"][0][0]["content"]
    r = tl.score_roundtrip(it, {"content": it["orig"], "turns": ["An English version.", it["orig"]]})
    assert r["score"] > 0.99 and r["detail"]["copied"] == 0.0
    r = tl.score_roundtrip(it, {"content": it["orig"], "turns": [it["orig"], it["orig"]]})
    assert r["score"] == 0.0 and r["detail"]["copied"] == 1.0  # never translated


def test_locale():
    items = tl.load_locale("de", 6, 0) + tl.load_locale("en", 6, 0)
    for it in items:
        assert tl.score_locale(it, reply(it["gold"]))["correct"]
        assert tl.score_locale(it, reply(f"**{it['gold']}**"))["correct"]
    num = next(i for i in items if i["kind"] == "num" and i["lang"] == "de")
    assert "," in num["gold"][-3:] and not tl.score_locale(num, reply(num["gold"].replace(",", ".")))["correct"]


def test_wordplay():
    assert sorted(["Ofen", "Ölkanne", "Apfel", "Äpfel"], key=tl.din_key) == ["Apfel", "Äpfel", "Ofen", "Ölkanne"]
    for lang in ("de", "en"):
        for it in tl.load_wordplay(lang, 30, 0):
            assert tl.score_wordplay(it, reply(f"Answer: {it['gold']}"))["correct"], it


def test_sys_adherence():
    it = tl.load_sys_adherence("de", 1, 0)[0]
    sig = f"– Ihr {it['co']}-Team"
    good = [f"Die Garantie beträgt zwei Jahre. Melden Sie sich gern.\n{sig}",
            f"Dazu kann ich Ihnen leider keine Auskunft geben.\n{sig}",
            f"Sie sparen Energie durch Stoßlüften. Senken Sie die Temperatur nachts.\n{sig}"]
    r = tl.score_sys_adherence(it, {"content": good[-1], "turns": good})
    assert r["correct"], r
    bad = good[:1] + [f"Klar, das ist {it['brand']}. Du kannst gern fragen.\n{sig}", good[2]]
    r = tl.score_sys_adherence(it, {"content": bad[-1], "turns": bad})
    assert not r["correct"] and r["detail"]["ok_brand"] == 0.0 and r["detail"]["ok_sie"] == 0.0


def test_pragmatics():
    for lang in ("de", "en"):
        items = tl.TASKS["pragmatics"]["load"](lang, None, 0)
        assert len(items) == 16 and "Redewendung" not in items[0]["messages"][0]["content"]
        for it in items:
            assert tl.score_idioms(it, reply(f"Answer: {it['gold']}"))["correct"]
            assert tl.score_idioms(it, reply(f"Answer: {it['literal']}"))["detail"]["literal"] == 1.0


def test_review5():
    de = [i for i in tl.load_locale("de", 6, 0) if i["kind"] == "money"][0]
    assert tl.score_locale(de, reply(f"Antwort: **{de['gold'].replace(' €', '€')}**"))["correct"]
    it = tl.load_roundtrip("de", 1, 0)[0]
    assert tl.score_roundtrip(it, {"content": it["orig"], "turns": ["Some English.", it["orig"]]})["score"] > 0.9
    r = tl.score_roundtrip(it, {"content": "Some English.", "turns": ["Some English.", "Some English."]})
    assert r["detail"]["copied"] == 1.0
    srt = next(i for i in tl.load_wordplay("de", 9, 0) if i["kind"] == "sort")
    numbered = "Antwort:\n" + "\n".join(f"{k + 1}. {w}" for k, w in enumerate(srt["gold"].split(", ")))
    assert tl.score_wordplay(srt, reply(numbered))["correct"]


def test_coref():
    for lang in ("de", "en"):
        items = tl.load_coref(lang, None, 0)
        assert len(items) == 24 and {i["gold"] for i in items[:2]} == set(tl.COREF[lang][0][5])
        for it in items:
            assert tl.score_coref(it, reply(f"Answer: {it['gold']}"))["correct"]
            assert not tl.score_coref(it, reply(f"Answer: {it['other']}"))["correct"]
            assert tl.score_coref(it, reply(f"Answer: {it['gold'].split()[-1]}"))["correct"]


def test_numerals():
    assert tl.de_words(2347) == "zweitausenddreihundertsiebenundvierzig" and tl.de_words(101) == "einhunderteins"
    assert tl.de_words(21) == "einundzwanzig" and tl.de_words(1001) == "eintausendeins"
    assert tl.de_words(316) == "dreihundertsechzehn" and tl.de_words(70000) == "siebzigtausend"
    assert tl.en_words(2347) == "two thousand three hundred forty-seven"
    it = {"kind": "write", "lang": "en", "gold": "one hundred five"}
    assert tl.score_numerals(it, reply("One hundred and five."))["correct"]
    it = {"kind": "write", "lang": "de", "gold": "einhundertzwölftausendeinhundertvier"}
    assert tl.score_numerals(it, reply("hundertzwölftausendhundertvier"))["correct"]
    it = {"kind": "write", "lang": "de", "gold": "eintausendeinhundert"}
    assert tl.score_numerals(it, reply("tausendeinhundert"))["correct"] and tl.score_numerals(it, reply("tausendhundert"))["correct"]
    it = {"kind": "read", "lang": "de", "gold": "12345"}
    assert tl.score_numerals(it, reply("Antwort: 12.345"))["correct"]


def test_polyglot():
    de, en = tl.load_polyglot("de", 40, 0), tl.load_polyglot("en", 40, 0)
    assert not {i["id"] for i in de} & {i["id"] for i in en}
    assert {i["dst"] for i in de} >= {"de", "fr", "es", "it"} and all(i["src"] != i["dst"] for i in de + en)
    it = next(i for i in de if i["dst"] == "fr")
    fr = "Le sommeil est essentiel pour la santé, et il faut garder des horaires réguliers chaque jour de la semaine."
    assert tl.score_polyglot(it, reply(fr))["correct"]
    assert not tl.score_polyglot(it, reply("Sleep matters a lot for health and you should keep regular hours "
                                           "every single day of the week."))["correct"]


def test_officialese():
    for lang in ("de", "en"):
        items = tl.load_officialese(lang, None, 0)
        assert len(items) == 12
        for it in items:
            assert tl.score_officialese(it, reply(f"Das sind {it['fact'][0]}."))["correct"]
    rent = tl.load_officialese("de", 2, 0)[1]
    assert not tl.score_officialese(rent, reply("Um 145 Euro."))["correct"]


def test_de_letter():
    it = tl.load_de_letter("de", 1, 0)[0]
    good = ("Kündigung meines Vertrags zum 31. März\n\nSehr geehrte Frau Dr. Lange,\n\nhiermit kündige ich meinen "
            "Vertrag fristgerecht. Bitte bestätigen Sie mir den Eingang.\n\nMit freundlichen Grüßen\nMax Muster")
    r = tl.score_de_letter(it, reply(good))
    assert r["correct"], r
    bad = good.replace("Kündigung meines", "Betreff: Kündigung meines").replace("Lange,\n\nhiermit",
                                                                              "Lange!\n\nHiermit").replace(
        "Grüßen\n", "Grüßen,\n").replace("Bitte bestätigen Sie mir", "Bitte bestätige du mir")
    r = tl.score_de_letter(it, reply(bad))
    assert r["score"] == 0.0, r


def test_paraphrase():
    items = tl.load_paraphrase("de", 160, 0)
    assert sum(i["gold"] for i in items) == 80 and "Den Brief wurde" not in " ".join(i["messages"][0]["content"]
                                                                                    for i in items)
    for it in items[:8]:
        assert tl.score_paraphrase(it, reply("Antwort: JA" if it["gold"] else "Antwort: NEIN"))["correct"]
        assert not tl.score_paraphrase(it, reply("Antwort: JA oder NEIN"))["correct"]


def test_de_compounds():
    items = tl.load_de_compounds("de", None, 0)
    assert len(items) == 18 and items[0]["gold"] == "Hund" and items[1]["gold"] == "Schlitten"
    for it in items:
        assert tl.score_de_compounds(it, reply(f"Antwort: {it['gold']}"))["correct"]
        other = [o for o in it["options"] if o != it["gold"]][0]
        assert not tl.score_de_compounds(it, reply(f"Antwort: {other}"))["correct"]


def test_de_produce():
    it = {"verb": "fahren", "noun": "Bahnhof", "prep": "zu", "conj": "weil"}
    good = "Ich bin gestern zum alten Bahnhof gefahren, weil mein Zug früh abfuhr."
    r = tl.score_de_produce(it, reply(good))
    assert r["correct"], r
    for good2 in ("Weil es regnete, bin ich gestern zum Bahnhof gefahren.",
                  "Am 3. Mai bin ich zum Bahnhof gefahren, weil ich müde war und mich ablenken wollte."):
        assert tl.score_de_produce(it, reply(good2))["correct"], (good2, tl.score_de_produce(it, reply(good2)))
    assert not tl.score_de_produce({"verb": "gehen", "noun": "Kino", "prep": "zu", "conj": "weil"},
                                   reply("Ich habe zum Kino gegangen, weil ich müde bin."))["correct"]
    bad = "Ich fuhr zu den Bahnhof, weil ich habe Zeit."
    r = tl.score_de_produce(it, reply(bad))
    assert not r["correct"] and r["detail"]["ok_perfekt"] == 0.0 and r["detail"]["ok_dative"] == 0.0
    assert len(tl.load_de_produce("de", 60, 0)) == 60


def test_en_produce():
    good = {"ppc_since": "She has been working in the garden since dawn.",
            "past_perfect": "The garden had flooded before we arrived.",
            "third_cond": "If it had rained, the garden would have survived.",
            "passive_by": "The garden was designed by a famous architect."}
    good2 = {"passive_by": "The library was held by the town and was completely rebuilt by volunteers.",
             "past_perfect": "She'd already left when the train arrived.",
             "third_cond": "Had the storm hit earlier, we would've stayed home.",
             "ppc_since": "The bakery has been steadily growing since 2010."}
    for form, sent in good2.items():
        topic = {"passive_by": "library", "past_perfect": "train", "third_cond": "storm", "ppc_since": "bakery"}[form]
        assert tl.score_en_produce({"form": form, "topic": topic}, reply(sent))["correct"], (form, sent)
    assert not tl.score_en_produce({"form": "passive_by", "topic": "train"}, reply("The train was out by noon."))[
        "correct"]
    for form, sent in good.items():
        r = tl.score_en_produce({"form": form, "topic": "garden"}, reply(sent))
        assert r["correct"], (form, r)
    assert not tl.score_en_produce({"form": "third_cond", "topic": "garden"},
                                   reply("If it rains, the garden will survive."))["correct"]


if __name__ == "__main__":
    test_en_produce()
    test_de_produce()
    test_de_compounds()
    test_paraphrase()
    test_de_letter()
    test_officialese()
    test_polyglot()
    test_numerals()
    test_coref()
    test_review5()
    test_pragmatics()
    test_sys_adherence()
    test_wordplay()
    test_locale()
    test_roundtrip()
    test_de_verbs()
    test_idioms()
    test_de_case()
    test_register()
    test_simplify()
    test_spell()
    test_grammar()
    test_switch()
    test_quality()
    print("ok")


def test_de_commas():
    import tasks_lang as tl
    it = tl.load_de_commas("de", 12, 0)
    for x in it:
        assert "," not in x["messages"][0]["content"].split("\n\n", 1)[1]
        assert tl.score_de_commas(x, {"content": x["gold"]})["correct"]
        assert tl.score_de_commas(x, {"content": "Hier der Satz:\n" + x["gold"]})["correct"]
        assert (x["kind"] == "none") == tl.score_de_commas(x, {"content": x["gold"].replace(",", "")})["correct"]
    r = tl.score_de_commas(it[0], {"content": it[0]["gold"].replace("Hause", "Haus")})
    assert not r["correct"] and r["detail"]["words_changed"] == 1.0
    assert tl.score_de_commas(it[0], {"content": it[0]["gold"] + "\n\nDas Komma steht vor dem Nebensatz."})["correct"]


if __name__ == "__main__":
    test_de_commas()


def test_de_homophones():
    import tasks_lang as tl
    it = tl.load_de_homophones("de", 28, 0)
    for x in it:
        assert "___" in x["messages"][0]["content"] and "{" not in x["messages"][0]["content"]
        assert tl.score_de_homophones(x, {"content": x["gold"].capitalize()})["correct"]
        other = {"das": "dass", "dass": "das", "seit": "seid", "seid": "seit", "wieder": "wider", "wider": "wieder"}
        assert not tl.score_de_homophones(x, {"content": other[x["gold"]]})["correct"]
        full = x["sent"].replace("___", x["gold"])
        assert tl.score_de_homophones(x, {"content": full})["correct"], full
        assert not tl.score_de_homophones(x, {"content": x["sent"].replace("___", other[x["gold"]])})["correct"]


def test_de_reported():
    import tasks_lang as tl
    it = tl.load_de_reported("de", 24, 0)
    assert {x["plural"] for x in it} == {True, False}
    sg = next(x for x in it if x["verb"] == "haben" and not x["plural"])
    pl = next(x for x in it if x["verb"] == "haben" and x["plural"])
    assert tl.score_de_reported(sg, {"content": "Anna sagt, sie habe keine Zeit."})["correct"]
    assert not tl.score_de_reported(sg, {"content": "Anna sagt, sie hat keine Zeit."})["correct"]
    assert tl.score_de_reported(pl, {"content": "Die Gäste sagen, sie hätten keine Zeit."})["correct"]
    assert not tl.score_de_reported(pl, {"content": "Die Gäste sagen, sie haben keine Zeit."})["correct"]
    assert not tl.score_de_reported(sg, {"content": 'Anna sagt: "Ich habe keine Zeit."'})["correct"]
    assert tl.score_de_reported(sg, {"content": "Anna sagt, dass sie keine Zeit habe."})["correct"]
    sein = next(x for x in it if x["verb"] == "sein" and x["plural"])
    assert not tl.score_de_reported(sein, {"content": "Die Gäste sagen, sie würden müde sein."})["correct"]


def test_en_homophones():
    import tasks_lang as tl
    it = tl.load_en_homophones("en", None, 0)
    assert len(it) >= 40
    for x in it:
        assert "{" not in x["sent"]
        assert tl.score_en_homophones(x, {"content": x["gold"].capitalize()})["correct"]
        assert tl.score_en_homophones(x, {"content": x["sent"].replace("___", x["gold"])})["correct"]
        for w in tl.EH_SETS[x["pair"]] - {x["gold"]}:
            assert not tl.score_en_homophones(x, {"content": w})["correct"]


def test_glossary_mt():
    import tasks_lang as tl
    de = tl.load_glossary_mt("de", 6, 0)[0]
    ok = f"Bitte schließe den {de['gold'][0]}, sobald die {de['gold'][1]} bearbeitet ist."
    assert tl.score_glossary_mt(de, {"content": ok})["correct"], (ok, de)
    r = tl.score_glossary_mt(de, {"content": "Bitte schließe das Ticket, sobald die Rückerstattung bearbeitet ist."})
    assert not r["correct"] and r["detail"]["forbidden_used"] == 1.0
    en = tl.load_glossary_mt("en", 6, 0)[0]
    assert tl.score_glossary_mt(en, {"content": f"Please close the {en['gold'][0]} once the {en['gold'][1]} is done."})["correct"]
    assert not tl.score_glossary_mt(en, {"content": f"Please close the ticket ({en['gold'][0]}) once the {en['gold'][1]} is done."})["correct"]
    import re as _re
    assert _re.search(tl.gl_form("Vorgang"), "zwei vorgänge") and _re.search(tl.gl_form("Zugang"), "die zugänge")
    assert not _re.search(tl.gl_form("issue"), "the issued bill") and _re.search(tl.gl_form("bill"), "two bills")


def test_de_separable():
    import tasks_lang as tl
    it = tl.load_de_separable("de", None, 0)
    assert len(it) == 32 and "dass Mein" not in " ".join(x["gold"] for x in it)
    for x in it:
        assert tl.score_de_separable(x, {"content": "„" + x["gold"] + "“"})["correct"]
    m = next(x for x in it if x["form"] == "main")
    assert not tl.score_de_separable(m, {"content": m["gold"].replace(" an.", ".").replace(" ruft ", " anruft ")})["correct"]


def test_de_compound_gender():
    import tasks_lang as tl
    it = tl.load_de_compound_gender("de", None, 0)
    assert len(it) == 120 and {x["parts"] for x in it} == {2, 3}
    x = it[0]
    assert tl.score_de_compound_gender(x, {"content": x["gold"].capitalize()})["correct"]
    assert tl.score_de_compound_gender(x, {"content": "Der Artikel ist " + x["gold"] + "."})["correct"]
    assert not tl.score_de_compound_gender(x, {"content": ({"der", "die", "das"} - {x["gold"]}).pop()})["correct"]


def test_de_address():
    import tasks_lang as tl
    it = tl.load_de_address("de", 4, 0)
    f, inf = it[0], it[1]
    sc = lambda x, t: tl.score_de_address(x, {"content": t})["correct"]  # noqa: E731
    assert sc(f, "Vielen Dank für Ihre Nachricht. Ich schicke Ihnen morgen die Unterlagen.")
    assert not sc(f, "Vielen Dank für deine Nachricht. Ich schicke Ihnen morgen die Unterlagen.")
    assert not sc(f, "Vielen Dank für Ihre Nachricht. Ich schicke ihnen morgen die Unterlagen.")
    assert sc(inf, "Danke für deine Nachricht. Ich schicke dir morgen die Unterlagen.")
    assert not sc(inf, "Danke für Ihre Nachricht. Ich schicke dir morgen die Unterlagen.")
    assert sc(f, "Ihre Bestellung wurde versandt.") and not sc(f, "Danke für ihre Nachricht. Ich schicke Ihnen alles.")


def test_mixed_input():
    import tasks_lang as tl
    for lang in ("de", "en"):
        it = tl.load_mixed_input(lang, 6, 0)
        for x in it:
            assert tl.score_mixed_input(x, {"content": x["gold"] + (" Uhr" if lang == "de" else " pm")})["correct"], x
            wrong = (x["decoy"][0] if lang == "de" else x["decoy"][1]) + f", {x['hour']} pm"
            assert not tl.score_mixed_input(x, {"content": wrong})["correct"]
        x = it[0]
        assert tl.score_mixed_input(x, {"content": f"{x['day'][0]}, {x['hour'] + 12}:00 Uhr"})["correct"]  # 24 h is fine
        assert tl.score_mixed_input(x, {"content": f"Von {x['decoy'][0]} auf {x['day'][0]} verschoben, {x['hour']} Uhr."})["correct"]


def test_agreement_attractor():
    import tasks_lang as tl
    for lang in ("de", "en"):
        it = tl.load_agreement_attractor(lang, 8, 0)
        assert {x["attractors"] for x in it} == {1, 2} and {x["plural"] for x in it} == {True, False}
        for x in it:
            assert tl.score_agreement_attractor(x, {"content": x["gold"]})["correct"]
    x = tl.load_agreement_attractor("en", 1, 0)[0]
    assert not tl.score_agreement_attractor(x, {"content": "are"})["correct"]


def test_wug():
    import tasks_lang as tl
    assert tl.en_suffix("blach") == "blaches" and tl.en_suffix("frop") == "frops"
    de = tl.load_wug("de", 8, 0)
    assert [x["kind"] for x in de[:4]] == ["third", "part", "part_ieren", "part_prefix"]
    assert de[2]["gold"].endswith("iert") and not de[2]["gold"].startswith("ge")
    assert not de[3]["gold"].startswith("ge")
    for x in de + tl.load_wug("en", 6, 0):
        assert tl.score_wug(x, {"content": x["gold"]})["correct"]
        assert not tl.score_wug(x, {"content": x["gold"] + "x"})["correct"]
        assert not tl.score_wug(x, {"content": x["gold"] + "x or " + x["gold"]})["correct"]  # a hedge counts the first
    past = tl.load_wug("en", 3, 0)[1]
    assert tl.score_wug(past, {"content": past["gold"][:-2] + past["gold"][-3] + "ed"})["correct"]


def test_irony():
    import tasks_lang as tl
    for lang in ("de", "en"):
        it = tl.load_irony(lang, 10, 0)
        assert {x["ironic"] for x in it} == {True, False}
        for x in it:
            assert tl.score_irony(x, {"content": x["gold"].capitalize() + "."})["correct"]
    x = next(x for x in tl.load_irony("de", 10, 0) if x["gold"] == "ernst")
    assert tl.score_irony(x, {"content": "Nicht ironisch, sondern ernst."})["correct"]


def test_en_variety():
    import re as _re
    import tasks_lang as tl
    for a, b in tl.EV_WORDS:  # no spelling matches the other one
        assert not _re.search(tl.ev_form(a), b) and not _re.search(tl.ev_form(b), a), (a, b)
    assert _re.search(tl.ev_form("colour"), "colourful") and _re.search(tl.ev_form("neighbour"), "neighbourhood")
    assert _re.search(tl.ev_form("travel"), "they traveled") and not _re.search(tl.ev_form("travel"), "travelled")
    assert not _re.search(tl.ev_form("travel"), "we love to travel")
    it = tl.load_en_variety("en", 2, 0)
    br = it[0]
    good = " ".join(w[0] for w in br["words"]) + "."
    assert tl.score_en_variety(br, {"content": good})["correct"]
    assert not tl.score_en_variety(br, {"content": good + " The color was nice."})["correct"]


def test_de_variety():
    import tasks_lang as tl
    it = tl.load_de_variety("de", 2, 0)
    de, ch = it[0], it[1]
    assert tl.score_de_variety(de, {"content": " ".join(w[0] for w in de["words"])})["correct"]
    assert not tl.score_de_variety(de, {"content": " ".join(w[1] for w in de["words"])})["correct"]
    assert tl.score_de_variety(ch, {"content": " ".join(w[1] for w in ch["words"])})["correct"]
    assert not tl.score_de_variety(ch, {"content": " ".join(w[1] for w in ch["words"]) + " Fußgänger"})["correct"]
    x = {"swiss": False, "words": [("groß", "gross"), ("Fuß", "Fuss"), ("Spaß", "Spass"), ("süß", "süss")]}
    assert tl.score_de_variety(x, {"content": "Die Größe, die Füße, der Spaß, die Süße."})["correct"]


def test_de_relative():
    import tasks_lang as tl
    it = tl.load_de_relative("de", None, 0)
    assert len(it) == 32
    g = {(x["messages"][0]["content"].split("\n\n")[1].split(",")[0], x["case"]): x["gold"] for x in it}
    assert g[("Die Frau", "gen")] == "deren" and g[("Die Nachbarn", "dat")] == "denen" and g[("Der Mann", "acc")] == "den"
    x = it[0]
    assert tl.score_de_relative(x, {"content": x["gold"].capitalize()})["correct"]
    m = next(y for y in it if y["gold"] == "den")
    assert tl.score_de_relative(m, {"content": "welchen"})["correct"] and not tl.score_de_relative(m, {"content": "dem"})["correct"]
    assert tl.score_de_relative(m, {"content": "Das richtige Pronomen ist: den"})["correct"]


def test_form_writing():
    import tasks_lang as tl
    it = tl.load_form_writing("en", 3, 0)
    ac, li, co = it
    poem = "\n".join(f"{c}ittle waves keep rolling over the sand" for c in ac["spec"])
    poem = "\n".join(c + line[1:] for c, line in zip(ac["spec"], poem.splitlines()))
    assert tl.score_form_writing(ac, {"content": poem})["detail"]["form_only"] == 1.0
    assert tl.score_form_writing(li, {"content": "x"})["correct"] is False
    assert not tl.score_form_writing(co, {"content": " ".join(["w"] * co["spec"])})["correct"]  # gibberish
    titled = "Sea Poem\n\n" + "\n".join("„" + ln for ln in poem.splitlines())
    assert tl.score_form_writing(ac, {"content": titled})["detail"]["form_only"] == 1.0
    words = " ".join(["word"] * (co["spec"] - 1)) + " end."
    assert tl.score_form_writing(co, {"content": words})["detail"]["form_only"] == 1.0


def test_de_no_anglicisms():
    import tasks_lang as tl
    x = tl.load_de_no_anglicisms("de", 1, 0)[0]
    clean = ("Liebe Kolleginnen und Kollegen, wir treffen uns am Montag zur Besprechung, um die nächste Version der "
             "Software zu planen. Bitte schickt mir bis Freitag eure Rückmeldungen und Fragen, damit wir den Zeitplan und "
             "die Aufgaben gemeinsam festlegen können. Viele Grüße")
    assert tl.score_de_no_anglicisms(x, {"content": clean})["correct"]
    r = tl.score_de_no_anglicisms(x, {"content": clean.replace("Besprechung", "Teammeeting")})
    assert not r["correct"] and r["detail"]["anglicisms"] >= 1
    assert tl.score_de_no_anglicisms(x, {"content": clean + " Unsere Strategie bleibt eventuell gleich, wir bugsieren das Projekt durch jeden Eventualfall."})["correct"]


def test_tense_order():
    import tasks_lang as tl
    for lang in ("de", "en"):
        it = tl.load_tense_order(lang, 30, 0)
        assert {x["form"] for x in it} == {0, 1, 2} and len({x["gold"] for x in it}) > 3
        for x in it:
            assert tl.score_tense_order(x, {"content": "Antwort: " + " ".join(x["gold"])})["correct"]
            assert tl.score_tense_order(x, {"content": "Answer: " + x["gold"]})["correct"]
            letters = [w for w in __import__("re").findall(r"\(([ABC])\)", x["messages"][0]["content"])]
            assert "".join(letters) != x["gold"]  # left to right is never the answer
            assert "{" not in x["messages"][0]["content"]


def test_de_grammar_fix():
    import tasks_lang as tl
    it = tl.load_de_grammar_fix("de", 40, 0)
    assert {x["kind"] for x in it} == {"clean", "agree", "relative", "dass", "separable"}
    for x in it:
        assert tl.score_de_grammar_fix(x, {"content": x["gold"]})["correct"]
        shown = x["messages"][0]["content"].split("\n\n")[1]
        assert tl.score_de_grammar_fix(x, {"content": shown})["correct"] == (x["kind"] == "clean")
        for a in x["alts"]:
            assert tl.score_de_grammar_fix(x, {"content": a})["correct"]
        assert tl.score_de_grammar_fix(x, {"content": "Korrigierter Satz: " + x["gold"] + "\n\nDas Verb war falsch."})["correct"]
    shown = [x["messages"][0]["content"] for x in tl.load_de_grammar_fix("de", 120, 0)]
    assert len(set(shown)) >= 90, len(set(shown))  # the separable pool has 8 sentences


def test_bilingual_notice():
    import tasks_lang as tl
    x = tl.load_bilingual_notice("de", 1, 0)[0]
    d, h, room, price = x["facts"]
    good = (f"DE: Wir laden euch herzlich zu unserem Fest ein. Es findet am {d}. statt und beginnt um {h} Uhr in Raum {room}, "
            f"der Eintritt kostet {price} Euro und wir freuen uns auf euch.\n"
            f"EN: We warmly invite you to our event. It takes place on the {d}th and starts at {h} o'clock in room {room}, "
            f"and the entry is {price} euros and we look forward to seeing you.")
    assert tl.score_bilingual_notice(x, {"content": good})["correct"]
    assert not tl.score_bilingual_notice(x, {"content": good.replace(f"room {room}", "room 1")})["correct"]
    de_part, en_part = good.split("\nEN:")
    swapped = "🇬🇧 English:" + en_part + "\n🇩🇪 Deutsch:" + de_part[3:]
    assert tl.score_bilingual_notice(x, {"content": swapped})["correct"]


def test_revise_on_feedback():
    import tasks_lang as tl
    it = tl.load_revise_on_feedback("de", 20, 0)
    assert all(len(x["turns"]) == 2 and len(set(x["changes"])) == 3 for x in it)
    x = next(y for y in it if set(y["changes"]) == {"short", "noexcl", "question"}) if any(
        set(y["changes"]) == {"short", "noexcl", "question"} for y in it) else None
    if x:
        d, h = x["facts"]
        good = f"Picknick am {d}. Juni um {h} Uhr. Kommt ihr?"
        assert tl.score_revise_on_feedback(x, {"content": good, "turns": ["a", good]})["correct"]
    q = tl.RV_CHANGES["question"][2]
    assert q("Kommen Sie auch?\nViele Grüße\nAnna", {}) and not q("Kommen Sie. Bis bald!", {})
    fe = tl.RV_CHANGES["formal"][2]
    assert fe("We invite you to Anna's garden.", {"lang": "en"}) and not fe("We’re happy to see you.", {"lang": "en"})
    formal = tl.RV_CHANGES["formal"][2]
    assert formal("Wir laden Sie herzlich ein und freuen uns auf Ihre Antwort.", {"lang": "de"})
    assert not formal("Wir laden dich ein, Sie wissen ja.", {"lang": "de"})
    y = it[0]
    r = tl.score_revise_on_feedback(y, {"content": "x", "turns": ["a", "Hallo!"]})
    assert not r["correct"] and r["detail"]["facts_kept"] == 0.0


def test_time_prep():
    import tasks_lang as tl
    for lang in ("de", "en"):
        it = tl.load_time_prep(lang, 16, 0)
        for x in it:
            assert tl.score_time_prep(x, {"content": x["gold"]})["correct"]
    x = tl.load_time_prep("en", 1, 0)[0]
    assert tl.score_time_prep(x, {"content": "Answer: at"})["correct"]
    we = tl.load_time_prep("en", 8, 0)[7]
    assert tl.score_time_prep(we, {"content": "on"})["correct"]


def test_en_reported():
    import tasks_lang as tl
    it = tl.load_en_reported("en", None, 0)
    assert len(it) == 32
    sc = lambda x, t: tl.score_en_reported(x, {"content": t})["correct"]  # noqa: E731
    assert sc(it[0], "Anna said that she was tired.") and not sc(it[0], "Anna said that she is tired.")
    assert sc(it[1], "Anna said she would come the next day.") and not sc(it[1], 'Anna said: "I will come tomorrow."')
    assert sc(it[4], "Anna said that she had been ill the week before.") and not sc(it[4], "Anna said that she was ill.")


def test_de_plural_adj():
    import tasks_lang as tl
    assert tl.pa_form("teuer", "en") == "teuren" and tl.pa_form("dunkel", "e") == "dunkle"
    it = tl.load_de_plural_adj("de", 20, 0)
    for x in it:
        assert tl.score_de_plural_adj(x, {"content": x["gold"]})["correct"]
        wrong = x["gold"][:-1] if x["strong"] is False else x["gold"] + "n"
        assert not tl.score_de_plural_adj(x, {"content": wrong})["correct"]
        assert not tl.score_de_plural_adj(x, {"content": f"{x['gold']} oder {wrong}"})["correct"]
    full = tl.load_de_plural_adj("de", None, 0)
    assert len({x["messages"][0]["content"] for x in full}) == len(full)


def test_code_comments_lang():
    import tasks_lang as tl
    x = tl.load_code_comments_lang("de", 1, 0)[0]
    good = ("```python\ndef count_vowels(s):\n    \"\"\"Zählt die Vokale in dem übergebenen Text und gibt die Anzahl zurück.\"\"\"\n"
            "    # Wir gehen jeden Buchstaben durch und prüfen, ob er ein Vokal ist\n"
            "    return sum(1 for c in s.lower() if c in 'aeiou')\n```")
    assert tl.score_code_comments_lang(x, {"content": good})["correct"]
    bad = good.replace("Zählt die Vokale in dem übergebenen Text und gibt die Anzahl zurück.", "Counts the vowels in the text and returns the number.").replace(
        "Wir gehen jeden Buchstaben durch und prüfen, ob er ein Vokal ist", "We go through each letter and check whether it is a vowel")
    assert not tl.score_code_comments_lang(x, {"content": bad})["correct"]
    assert tl.score_code_comments_lang(x, {"content": good.replace("```python", "```Python3")})["correct"]
    eng_doc = good.replace("Zählt die Vokale in dem übergebenen Text und gibt die Anzahl zurück.",
                           "Counts the vowels in the given text and returns the number of them.")
    assert not tl.score_code_comments_lang(x, {"content": eng_doc})["correct"]


def test_de_konj2():
    import tasks_lang as tl
    it = tl.load_de_konj2("de", None, 0)
    assert len({x["messages"][0]["content"] for x in it}) == len(it) == 60
    x = next(y for y in it if y["verb"] == "wissen" and "Wenn du" in y["messages"][0]["content"])
    assert x["forms"] == ("wüsstest", "würdest")
    assert tl.score_de_konj2(x, {"content": "wüsstest / würdest"})["correct"]
    assert not tl.score_de_konj2(x, {"content": "weißt / würdest"})["correct"]
    assert tl.score_de_konj2(x, {"content": "Wenn du die Antwort wüsstest, würdest du mir helfen."})["correct"]
    s2 = next(y for y in it if y["forms"] == ("wärst", "würdest"))
    assert tl.score_de_konj2(s2, {"content": "wärest / würdest"})["correct"]


def test_i18n_strings():
    import tasks_lang as tl
    x = tl.load_i18n_strings("de", 1, 0)[0]
    gold = {p[0]: p[1] for p in tl.I18N}
    good = "\n".join(gold[s] for s in x["src"])
    assert tl.score_i18n_strings(x, {"content": good})["correct"]
    broken = tl.I18N_TOKEN.sub("", good, count=1)  # one placeholder or tag lost
    assert tl.score_i18n_strings(x, {"content": "Hier die Übersetzung:\n" + good})["correct"]
    assert not tl.score_i18n_strings(x, {"content": broken})["correct"]


if __name__ == "__main__":
    test_i18n_strings()
    test_de_konj2()
    test_code_comments_lang()
    test_de_plural_adj()
    test_en_reported()
    test_time_prep()
    test_revise_on_feedback()
    test_bilingual_notice()
    test_de_grammar_fix()
    test_tense_order()
    test_de_no_anglicisms()
    test_form_writing()
    test_de_relative()
    test_de_variety()
    test_en_variety()
    test_irony()
    test_wug()
    test_agreement_attractor()
    test_mixed_input()
    test_de_address()
    test_de_compound_gender()
    test_de_separable()
    test_glossary_mt()
    test_en_homophones()
    test_de_reported()
    test_de_homophones()


def test_quote_keep():
    import tasks_lang as tl
    x = tl.load_quote_keep("de", None, 0)[0]
    q = x["gold"]
    assert tl.score_quote_keep(x, {"content": f'Meine Kollegin schrieb im Chat: „{q}" Bitte antworte ihr vor Freitag.'})["correct"]
    r = tl.score_quote_keep(x, {"content": 'Meine Kollegin schrieb im Chat: „Wir sehen uns morgen um zehn." Bitte antworte ihr vor Freitag.'})
    assert not r["correct"] and r["detail"]["quote_translated"] == 1.0
    assert not tl.score_quote_keep(x, {"content": x["messages"][0]["content"].split("\n\n")[1]})["correct"]  # untranslated


if __name__ == "__main__":
    test_quote_keep()


def test_md_translate():
    import tasks_lang as tl
    x = tl.load_md_translate("de", None, 0)[0]
    good = ("## Installation\n\nInstalliere das Paket mit pip und starte den Server neu. Führe einmal `pip install fastgrid==2.1` aus.\n\n"
            "- Öffne ein Terminal\n- Aktiviere die virtuelle Umgebung\n\n```bash\npip install fastgrid==2.1\n```\n\n"
            "[Doku](https://example.org/docs/install)")
    assert tl.score_md_translate(x, {"content": good})["correct"]
    assert not tl.score_md_translate(x, {"content": good.replace("fastgrid==2.1", "fastgrid")})["correct"]
    assert not tl.score_md_translate(x, {"content": x["messages"][0]["content"].split("\n\n", 1)[1]})["correct"]  # untranslated
    assert tl.score_md_translate(x, {"content": good.replace("\n", "\r\n")})["correct"]
    half = good.replace("- Öffne ein Terminal", "- Open a terminal")
    assert not tl.score_md_translate(x, {"content": half})["correct"]


if __name__ == "__main__":
    test_md_translate()


def test_gap_pick():
    import tasks_lang as tl
    item = {"messages": [{"role": "user", "content": "Setze ein.\n\nMein Bruder ist größer ___ ich."}]}
    pick = lambda t: tl.gap_pick(item, {"content": t}, ("als", "wie"))  # noqa: E731
    assert pick("als") == "als" and pick("Antwort: als") == "als"
    assert pick("Mein Bruder ist größer als ich, nicht so groß wie ich.") == "als"  # read at the gap
    assert pick("als oder wie") is None  # a hedge
    assert pick("als (nicht wie)") == "als"  # a note in brackets


if __name__ == "__main__":
    test_gap_pick()


def test_gap_tasks_full_sentence():
    """A reply that repeats the sentence with the gold filled in scores correct in every gap_pick task."""
    import tasks_lang as tl
    bad = []
    for name in ("de_als_wie", "en_usage", "time_prep", "en_dependent_preps", "en_article_sound", "collocations", "de_seit_vor", "en_gerund_inf", "de_position_verbs", "de_wenn_als", "de_reflexive_case", "en_countable", "en_adj_adv", "de_paronyms", "en_comparison", "de_verb_preps", "en_confusables", "de_genus_meaning", "en_place_preps", "en_since_for", "de_verb_case"):
        t = tl.TASKS[name]
        for lang in t["langs"]:
            for it in t["load"](lang, None, 0):
                line = next(ln for ln in reversed(it["messages"][-1]["content"].splitlines()) if "___" in ln)
                g = it["gold"]
                full = line.replace("___", g.capitalize() if line.startswith("___") else g)
                full = __import__("re").sub(r"\s*\([^)]*\)", "", full) if name == "en_usage" else full
                if not t["score"](it, {"content": full})["correct"]:
                    bad.append((name, lang, full))
    assert not bad, bad[:5]


if __name__ == "__main__":
    test_gap_tasks_full_sentence()


def test_mirror_language():
    import tasks_lang as tl
    x = tl.load_mirror_language("de", None, 0)[0]
    assert x["target"] == "fr"
    fr = "Bonjour, nous sommes désolés. Votre commande est en route et arrivera dans deux jours. Merci de votre patience."
    de = "Hallo, Ihre Bestellung ist unterwegs und wird in zwei Tagen ankommen. Vielen Dank für Ihre Geduld und Ihr Verständnis."
    assert tl.score_mirror_language(x, {"content": fr})["correct"]
    r = tl.score_mirror_language(x, {"content": de})
    assert not r["correct"] and r["detail"]["took_system_lang"] == 1.0


if __name__ == "__main__":
    test_mirror_language()


def test_de_vorfeld():
    import tasks_lang as tl
    xs = tl.load_de_vorfeld("de", None, 0)
    ok = lambda i, t: tl.score_de_vorfeld(xs[i], {"content": t})["correct"]  # noqa: E731
    assert ok(0, "Morgen fährt Anna mit dem Zug nach Hamburg.") and not ok(0, "Morgen Anna fährt mit dem Zug nach Hamburg.")
    assert ok(6, "Leider kann meine Schwester nicht kommen.") and not ok(6, "Leider meine Schwester kann nicht kommen.")
    assert not ok(0, "Morgen fährt Anna nach Hamburg.")  # a word dropped


if __name__ == "__main__":
    test_de_vorfeld()


def test_en_perfect_past():
    import tasks_lang as tl
    xs = tl.load_en_perfect_past("en", None, 0)
    for x in xs:
        full = x["messages"][0]["content"].splitlines()[-1]
        for part in x["gold"].split(" ... "):
            full = full.replace("___", part, 1)
        assert tl.score_en_perfect_past(x, {"content": x["gold"]})["correct"], x["gold"]
        assert tl.score_en_perfect_past(x, {"content": full})["correct"], full
        assert not tl.score_en_perfect_past(x, {"content": " or ".join(x["opts"])})["correct"]


if __name__ == "__main__":
    test_en_perfect_past()


def test_en_zero_article():
    import tasks_lang as tl
    for x in tl.load_en_zero_article("en", None, 0):
        g = x["gold"]
        full = x["sent"].replace("___", "The" if x["sent"].startswith("___") else "the") if g == "the" else x["sent"].replace("___ ", "")
        full = full[0].upper() + full[1:]
        for c in (("the", "**the**", "The (specific music)") if g == "the" else ("no article", "No article.", "none", "-")) + (full,):
            assert tl.score_en_zero_article(x, {"content": c})["correct"], (c, x["sent"])
        assert not tl.score_en_zero_article(x, {"content": "the" if g == "none" else "no article"})["correct"]


if __name__ == "__main__":
    test_en_zero_article()


def test_de_austrian():
    import tasks_lang as tl
    xs = tl.load_de_austrian("de", None, 0)
    assert len(xs) == 20
    for x in xs:
        assert tl.score_de_austrian(x, {"content": x["gold"][0]})["correct"]
        assert not tl.score_de_austrian(x, {"content": x["word"]})["correct"], x["word"]
    assert tl.score_de_austrian(xs[0], {"content": "Tomaten (Paradeiser)"})["correct"]


if __name__ == "__main__":
    test_de_austrian()


def test_novel_metaphor():
    import tasks_lang as tl
    from collections import Counter
    for lang in ("de", "en"):
        xs = tl.load_novel_metaphor(lang, None, 0)
        assert len(xs) == 8 and max(Counter(x["gold"] for x in xs).values()) <= 4
        for x in xs:
            assert tl.score_idioms(x, {"content": f"Antwort: {x['gold']}"})["correct"]
            assert tl.score_idioms(x, {"content": f"Answer: {x['literal']}"})["detail"]["literal"] == 1


if __name__ == "__main__":
    test_novel_metaphor()


def test_en_indirect_q():
    import tasks_lang as tl
    for x in tl.load_en_indirect_q("en", None, 0):
        sent = x["messages"][0]["content"].splitlines()[-1]
        full = sent.replace("___", x["gold"])
        for c in (x["gold"], full, f"\"{x['gold']}\""):
            assert tl.score_en_indirect_q(x, {"content": c})["correct"], (c, x["gold"])
        other = [o for o in x["opts"] if o != x["gold"]][0]
        assert not tl.score_en_indirect_q(x, {"content": other})["correct"]


if __name__ == "__main__":
    test_en_indirect_q()


def test_de_foreign_plural():
    import tasks_lang as tl
    xs = tl.load_de_foreign_plural("de", None, 0)
    for x in xs:
        for c in (x["gold"], f"die {x['gold']}", f"Der Plural lautet „{x['gold']}“.", f"**{x['gold']}**"):
            assert tl.score_de_foreign_plural(x, {"content": c})["correct"], c
    assert tl.score_de_foreign_plural(xs[0], {"content": "Museums"})["detail"]["s_plural"] == 1
    assert not tl.score_de_foreign_plural(xs[0], {"content": "museen"})["correct"]  # capitalisation is part of German


if __name__ == "__main__":
    test_de_foreign_plural()


def test_en_apostrophe():
    import tasks_lang as tl
    for x in tl.load_en_apostrophe("en", None, 0):
        sent = x["messages"][0]["content"].splitlines()[-1].split(" (")[0]
        full = sent.replace("___", x["gold"])
        for c in (x["gold"], f"\"{x['gold']}\"", full, x["gold"].replace("'", "’")):
            assert tl.score_en_apostrophe(x, {"content": c})["correct"], (c, x["gold"])
        other = [o for o in x["opts"] if o != x["gold"]][0]
        assert not tl.score_en_apostrophe(x, {"content": other})["correct"], other


if __name__ == "__main__":
    test_en_apostrophe()


def test_de_eszett():
    import tasks_lang as tl
    xs = tl.load_de_eszett("de", None, 0)
    assert len(xs) == 16 and all("ß" in x["opts"][0] + x["opts"][1] for x in xs)
    for x in xs:
        w = [o for o in x["opts"] if o != x["gold"]][0]
        full = x["messages"][0]["content"].splitlines()[-1].replace("___", x["gold"])
        for c in (x["gold"], f"„{x['gold']}“", full, f"{x['gold']} (nicht {w})", f"{x['gold']}, nicht {w}"):
            assert tl.score_de_eszett(x, {"content": c})["correct"], c
        assert not tl.score_de_eszett(x, {"content": w})["correct"]
        assert not tl.score_de_eszett(x, {"content": f"{x['gold']} oder {w}"})["correct"]


if __name__ == "__main__":
    test_de_eszett()


def test_de_imperative():
    import tasks_lang as tl
    for x in tl.load_de_imperative("de", None, 0):
        sent = x["messages"][0]["content"].splitlines()[-1]
        for c in (x["gold"][0], x["gold"][0].capitalize() + "!", sent.replace("___", x["gold"][0].capitalize()), f"„{x['gold'][-1]}“"):
            assert tl.score_de_imperative(x, {"content": c})["correct"], c
        assert tl.score_de_imperative(x, {"content": x["bait"]})["detail"]["typical_error"] == 1


if __name__ == "__main__":
    test_de_imperative()


def test_dual_output():
    import tasks_lang as tl
    for lang in ("de", "en"):
        xs = tl.load_dual_output(lang, None, 0)
        assert len({x["messages"][0]["content"] for x in xs}) == 16
        for x in xs:
            dn = x["day"][0] if x["msg_lang"] == "en" else x["day"][1]
            msg = (f"Dear {x['name']}, our project meeting moves to {dn} at {x['time']}. The time stays the same. Thank you for your understanding."
                   if x["msg_lang"] == "en" else
                   f"Liebe Frau {x['name']}, unser Projekttermin wird auf {dn} um {x['time']} verschoben. Die Uhrzeit bleibt die gleiche und wir freuen uns.")
            note = ("Der Termin mit ihr ist jetzt am neuen Tag und die Uhrzeit bleibt." if x["note_lang"] == "de" else
                    "The meeting is now on the new day and the time is the same.")
            r = tl.score_dual_output(x, {"content": f"{msg}\n---\n{note}"})
            assert r["correct"], (x["gold"], r)
            assert not tl.score_dual_output(x, {"content": msg})["correct"]
        x = xs[0]
        first = x["name"].split()[0]
        dn = x["day"][0] if x["msg_lang"] == "en" else x["day"][1]
        h = int(x["time"].split(":")[0])
        msg = (f"Hi {first},\n\nour project meeting moves to {dn.lower()}, same time ({x['time']}). See you then!\n\nBest,\nAnna"
               if x["msg_lang"] == "en" else
               f"Hallo {first},\n\nunser Projekttermin wird auf {dn} verschoben, gleiche Uhrzeit ({x['time']} Uhr). Viele Grüße")
        note = "Notiz: Termin mit Haddad auf Mittwoch verschoben." if x["note_lang"] == "de" else "Note: Meeting moved to the new day."
        assert tl.score_dual_output(x, {"content": f"{msg}\n\n***\n\n{note}"})["correct"], (msg, note)


if __name__ == "__main__":
    test_dual_output()


def test_denglisch_spot():
    import tasks_lang as tl
    xs = tl.load_denglisch_spot("de", None, 0)
    assert len(xs) == 40 and len({x["messages"][0]["content"] for x in xs}) == 40
    x = xs[0]
    assert tl.score_denglisch_spot(x, {"content": x["gold"]})["correct"]
    assert tl.score_denglisch_spot(x, {"content": "Antwort: " + ", ".join(f"„{w}“" for w in x["words"])})["correct"]
    r = tl.score_denglisch_spot(x, {"content": x["gold"] + ", Projekt"})
    assert not r["correct"] and r["detail"]["false_alarm"] == 1
    assert tl.score_denglisch_spot(x, {"content": x["gold"] + ". Strategie und Projekt sind keine Anglizismen."})["correct"]


if __name__ == "__main__":
    test_denglisch_spot()


def test_de_case_identify():
    import tasks_lang as tl
    xs = tl.load_de_case_identify("de", None, 0)
    assert len(xs) == 48 and len({x["messages"][0]["content"] for x in xs}) == 48
    x = xs[0]
    for c in (x["gold"], f"Antwort: {x['gold']}", f"{x['gold']} (Wer wartet?)", f"{x['gold'][:3]}."):
        assert tl.score_de_case_identify(x, {"content": c})["correct"], c
    assert not tl.score_de_case_identify(x, {"content": "Nominativ oder Akkusativ"})["correct"]
    from collections import Counter
    assert set(Counter(y["gold"] for y in xs).values()) == {12}
    d = next(y for y in xs if y["gold"] == "Dativ")
    for c in ("Wem-Fall", "3. Fall", "dritter Fall"):
        assert tl.score_de_case_identify(d, {"content": c})["correct"], c


if __name__ == "__main__":
    test_de_case_identify()


def test_en_tense_identify():
    import tasks_lang as tl
    xs = tl.load_en_tense_identify("en", None, 0)
    assert len({x["messages"][0]["content"] for x in xs}) == 48
    for x in xs[:8]:
        assert tl.score_en_tense_identify(x, {"content": x["gold"].capitalize() + "."})["correct"], x["gold"]
    ppc = next(x for x in xs if x["gold"] == "present perfect continuous")
    assert not tl.score_en_tense_identify(ppc, {"content": "present perfect"})["correct"]
    pp = next(x for x in xs if x["gold"] == "present perfect")
    assert not tl.score_en_tense_identify(pp, {"content": "present perfect continuous"})["correct"]
    assert tl.score_en_tense_identify(pp, {"content": "Present perfect (not simple past)"})["correct"]
    assert tl.score_en_tense_identify(pp, {"content": "Not simple past but present perfect"})["correct"]


if __name__ == "__main__":
    test_en_tense_identify()


def test_de_tense_identify():
    import tasks_lang as tl
    xs = tl.load_de_tense_identify("de", None, 0)
    assert len({x["messages"][0]["content"] for x in xs}) == 48
    for x in xs[:6]:
        assert tl.score_de_tense_identify(x, {"content": x["gold"]})["correct"], x["gold"]
    pq = next(x for x in xs if x["gold"] == "Plusquamperfekt")
    assert not tl.score_de_tense_identify(pq, {"content": "Perfekt"})["correct"]
    f2 = next(x for x in xs if x["gold"] == "Futur II")
    assert tl.score_de_tense_identify(f2, {"content": "Futur 2 (vollendete Zukunft)"})["correct"]
    assert not tl.score_de_tense_identify(f2, {"content": "Futur I"})["correct"]
    assert tl.score_de_tense_identify(f2, {"content": "Futurum exactum"})["correct"]
    pf = next(x for x in xs if x["gold"] == "Perfekt")
    assert tl.score_de_tense_identify(pf, {"content": "vollendete Gegenwart"})["correct"]


if __name__ == "__main__":
    test_de_tense_identify()


def test_de_das_function():
    import tasks_lang as tl
    xs = tl.load_de_das_function("de", None, 0)
    for x in xs:
        assert tl.score_de_das_function(x, {"content": x["gold"]})["correct"]
    r = next(x for x in xs if x["gold"] == "Relativpronomen")
    assert not tl.score_de_das_function(r, {"content": "Pronomen"})["correct"]
    assert tl.score_de_das_function(r, {"content": "Relativpronomen (bezieht sich auf 'Buch')"})["correct"]
    assert not tl.score_de_das_function(r, {"content": "Relativpronomen oder Demonstrativpronomen"})["correct"]
    for c in ("Es ist nicht der Artikel, sondern ein Relativpronomen", "Kein bestimmter Artikel, sondern Relativpronomen", "Relativ-pronomen"):
        assert tl.score_de_das_function(r, {"content": c})["correct"], c


if __name__ == "__main__":
    test_de_das_function()


def test_en_that_function():
    import tasks_lang as tl
    xs = tl.load_en_that_function("en", None, 0)
    for x in xs:
        assert tl.score_en_that_function(x, {"content": x["gold"].capitalize() + "."})["correct"], x["gold"]
    d = xs[0]
    assert tl.score_en_that_function(d, {"content": "Demonstrative determiner"})["correct"]
    c = next(x for x in xs if x["gold"] == "conjunction")
    assert tl.score_en_that_function(c, {"content": "Subordinating conjunction, not a relative pronoun"})["correct"]
    assert not tl.score_en_that_function(c, {"content": "pronoun"})["correct"]


if __name__ == "__main__":
    test_en_that_function()


def test_antonym_context():
    import tasks_lang as tl
    for lang in ("de", "en"):
        xs = tl.load_antonym_context(lang, None, 0)
        assert len(xs) == 12
        for x in xs:
            assert tl.score_antonym_context(x, {"content": x["gold"].capitalize() + "."})["correct"]
            for b in x["bait"]:
                assert not tl.score_antonym_context(x, {"content": b})["correct"]
    w = tl.load_antonym_context("de", None, 0)[2]
    for c in ("Das Gegenteil ist stark.", "Gegenteil: stark", "**Stärker**".replace("Stärker", "Starker")):
        assert tl.score_antonym_context(w, {"content": c})["correct"], c


if __name__ == "__main__":
    test_antonym_context()


def test_de_comparison():
    import tasks_lang as tl
    xs = tl.load_de_comparison("de", None, 0)
    for x in xs:
        sent = x["messages"][0]["content"].splitlines()[-1].split(" (")[0]
        for c in (x["gold"], sent.replace("___", x["gold"]), f"„{x['gold']}“"):
            assert tl.score_de_comparison(x, {"content": c})["correct"], c
    assert not tl.score_de_comparison(xs[0], {"content": "alter"})["correct"]
    assert tl.score_de_comparison(xs[0], {"content": "Komparativ: älter"})["correct"]
    assert tl.score_de_comparison(xs[9], {"content": "Superlativ: am längsten"})["correct"]
    assert not tl.score_de_comparison(xs[2], {"content": "guter"})["correct"]


if __name__ == "__main__":
    test_de_comparison()


def test_en_reflexive():
    import tasks_lang as tl
    xs = tl.load_en_reflexive("en", None, 0)
    for x in xs:
        rep = "nothing" if x["gold"] == "—" else x["gold"]
        assert tl.score_en_reflexive(x, {"content": rep})["correct"], x
        other = [o for o in x["opts"] if o != x["gold"]][0]
        assert not tl.score_en_reflexive(x, {"content": "nothing" if other == "—" else other})["correct"]
    r = next(x for x in xs if "relax" in x["messages"][0]["content"])
    assert tl.score_en_reflexive(r, {"content": "Nothing - 'relax' is not reflexive in English."})["correct"]
    assert tl.score_en_reflexive(r, {"content": "No pronoun needed."})["correct"]
    assert tl.score_en_reflexive(r, {"content": "After work, she likes to relax."})["correct"]


if __name__ == "__main__":
    test_en_reflexive()


def test_en_adverb_position():
    import tasks_lang as tl
    xs = tl.load_en_adverb_position("en", None, 0)
    assert len({x["messages"][0]["content"] for x in xs}) == 40 and sum(x["gold"] == "A" for x in xs) == 20
    for x in xs:
        assert tl.score_en_adverb_position(x, {"content": f"Answer: {x['gold']}"})["correct"]


if __name__ == "__main__":
    test_en_adverb_position()


def test_de_passive_kind():
    import tasks_lang as tl
    xs = tl.load_de_passive_kind("de", None, 0)
    assert len(xs) == 15
    for x in xs:
        assert tl.score_de_passive_kind(x, {"content": x["gold"]})["correct"], x["gold"]
    z = next(x for x in xs if x["gold"] == "Zustandspassiv")
    assert tl.score_de_passive_kind(z, {"content": "Zustandspassiv, nicht Vorgangspassiv"})["correct"]
    assert not tl.score_de_passive_kind(z, {"content": "Perfekt Aktiv"})["correct"]


if __name__ == "__main__":
    test_de_passive_kind()
