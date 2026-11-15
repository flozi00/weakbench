"""tasks_open (open_checklist) with a mock judge, plain asserts:

    uv run --with pyarrow --with pyyaml --with langdetect --with immutabledict --with 'nltk>=3.9.1' python test_tasks_open.py
"""
import json

import mock_server
import tasks_judge
import tasks_open as to

reply = lambda text: {"content": text, "reasoning": "", "tool_calls": [], "finish_reason": "stop"}  # noqa: E731


def test_checklist():
    assert all(4 <= len(c) <= 6 for *_, c in to.ITEMS)
    answer = {"v": None}
    url, stop = mock_server.serve(lambda path, body: {"content": answer["v"], "finish_reason": "stop"})
    try:
        tasks_judge.configure(url, "mock", 32768)
        de, en = to.load_checklist("de", None, 0), to.load_checklist("en", None, 0)
        assert [i["id"] for i in de] == [i["id"] for i in en] and len(de) == len(to.ITEMS)
        it = de[0]
        answer["v"] = json.dumps({"verdicts": ["YES"] * len(it["criteria"])})
        r = to.score_checklist(it, reply("Der Preis war für den photoelektrischen Effekt."))
        assert r["score"] == 1.0 and r["detail"]["all_met"] == 1.0, r
        answer["v"] = "Here: " + json.dumps({"verdicts": ["YES", "NO"] + ["no"] * (len(it["criteria"]) - 2)})
        r = to.score_checklist(it, reply("x"))
        assert r["score"] == round(1 / len(it["criteria"]), 4) and r["detail"]["all_met"] == 0.0, r
        answer["v"] = json.dumps({"verdicts": ["YES"]})
        try:
            to.score_checklist(it, reply("x"))
            raise AssertionError("a short verdict list must raise")
        except ValueError:
            pass
    finally:
        stop()
        tasks_judge.configure(None, None, None)


def test_faithful():
    answer = {"v": None}
    url, stop = mock_server.serve(lambda path, body: {"content": answer["v"], "finish_reason": "stop"})
    try:
        tasks_judge.configure(url, "mock", 32768)
        for lang in ("de", "en"):
            items = to.load_faithful(lang, 3, 0)
            assert items[0]["article"] in items[0]["messages"][0]["content"] and "Summary:" not in items[0]["article"]
            assert "Zusammenfassung:" not in items[0]["article"] and len(items[0]["article"]) > 200
        xl = to.load_summary_xl("de", 2, 0)
        assert xl[0]["lang"] == "de" and "englischen" in xl[0]["messages"][0]["content"]
        assert xl[0]["article"] == to.load_faithful("en", 2, 0)[0]["article"]
        answer["v"] = json.dumps({"sentences": [{"text": "a", "label": "SUPPORTED"}, {"text": "b", "label":
                                  "UNSUPPORTED"}], "main_point": True})
        r = to.score_faithful(items[0], reply("a. b."))
        assert r["score"] == 0.5 and r["detail"]["faithful"] == 0.0 and r["detail"]["main_point"] == 1.0, r
        answer["v"] = json.dumps({"sentences": [{"text": "a", "label": "SUPPORTED"}], "main_point": False})
        assert to.score_faithful(items[0], reply("a."))["score"] == 0.5  # faithful but pointless
        answer["v"] = "nothing useful"
        try:
            to.score_faithful(items[0], reply("a"))
            raise AssertionError("an unusable verdict must raise")
        except ValueError:
            pass
    finally:
        stop()
        tasks_judge.configure(None, None, None)


def test_mqm():
    answer = {"v": None}
    url, stop = mock_server.serve(lambda path, body: {"content": answer["v"], "finish_reason": "stop"})
    try:
        tasks_judge.configure(url, "mock", 32768)
        items = to.load_mqm("de", 3, 0)
        assert items[0]["src"] in items[0]["messages"][0]["content"] and "German" in items[0]["messages"][0]["content"]
        words = len(items[0]["src"].split())
        answer["v"] = json.dumps({"errors": [{"quote": "Haus ist", "category": "accuracy", "severity": "major"},
                                             {"quote": "nicht im Text", "category": "fluency", "severity": "minor"}]})
        r = to.score_mqm(items[0], reply("Das Haus ist rot."))
        assert r["detail"]["judge_dropped"] == 1 and r["detail"]["major_or_worse"] == 1.0
        assert r["detail"]["mqm_per_100w"] == round(500 / words, 3), r
        answer["v"] = '{"errors": []}'
        assert to.score_mqm(items[0], reply("Perfekt."))["score"] == 1.0
        assert to.score_mqm(items[0], reply(""))["detail"]["major_or_worse"] == 1.0
    finally:
        stop()
        tasks_judge.configure(None, None, None)


def test_proofs():
    answer = {"v": '{"grade": "GAP", "first_problem": "x"}'}
    url, stop = mock_server.serve(lambda path, body: {"content": answer["v"], "finish_reason": "stop"})
    try:
        tasks_judge.configure(url, "mock", 32768)
        items = to.load_proofs("de", None, 0)
        assert len(items) == 12 and "Beweise" in items[0]["messages"][0]["content"]
        r = to.score_proofs(items[0], reply("Seien 2a+1 und 2b+1 ..."))
        assert r["score"] == 0.5 and not r["correct"] and r["detail"]["gap"] == 1.0
        answer["v"] = "nonsense"
        try:
            to.score_proofs(items[0], reply("x"))
            raise AssertionError("must raise")
        except ValueError:
            pass
    finally:
        stop()
        tasks_judge.configure(None, None, None)


def test_roleplay():
    url, stop = mock_server.serve(lambda path, body: {"content": json.dumps({"verdicts": ["YES"] * 4 + ["NO"]}),
                                                      "finish_reason": "stop"})
    try:
        tasks_judge.configure(url, "mock", 32768)
        it = to.load_roleplay("de", 1, 0)[0]
        assert len(it["turns"]) == 3 and it["turns"][0][0]["role"] == "system"
        r = to.score_roleplay(it, {"content": "c", "turns": ["a", "b", "c"]})
        assert r["score"] == 0.8 and r["detail"]["ok_c5"] == 0.0
    finally:
        stop()
        tasks_judge.configure(None, None, None)


def test_reasoning_faithful():
    url, stop = mock_server.serve(lambda path, body: {"content": '{"follows": false}', "finish_reason": "stop"})
    try:
        tasks_judge.configure(url, "mock", 32768)
        it = to.load_reasoning_faithful("de", 2, 0)[0]
        assert "Rechenschritte" in it["messages"][0]["content"]
        r = to.score_reasoning_faithful(it, reply(f"3 + 4 = 7\nAntwort: {it['gold']}"))
        assert r["score"] == 0.0 and r["detail"]["unfaithful_but_correct"] == 1.0
    finally:
        stop()
        tasks_judge.configure(None, None, None)


def test_idiom_translate():
    url, stop = mock_server.serve(lambda path, body: {"content": "LITERAL", "finish_reason": "stop"})
    try:
        tasks_judge.configure(url, "mock", 32768)
        it = to.load_idiom_translate("en", 1, 0)[0]
        assert it["src"].startswith("Seit Wochen") and "natural English" in it["messages"][0]["content"]
        r = to.score_idiom_translate(it, reply("He has tomatoes on his eyes."))
        assert not r["correct"] and r["detail"]["literal"] == 1.0
    finally:
        stop()
        tasks_judge.configure(None, None, None)



def test_tutor_feedback():
    import re as _re
    url, stop = mock_server.serve(lambda path, body: {"content": json.dumps({"verdicts": ["YES"] * 5}),
                                                      "finish_reason": "stop"})
    try:
        tasks_judge.configure(url, "mock", 32768)
        it = to.load_tutor_feedback("de", 6, 0)
        assert [x["k"] for x in it[:3]] == [1, 2, 3]
        for x in it:  # exactly the planted step is wrong, later steps follow from it
            st = _re.findall(r"^(\d)\) (.+) = (.+)$", x["messages"][0]["content"], _re.M)
            wrong = [int(n) for n, lhs, rhs in st
                     if abs(eval(lhs.replace("×", "*").replace(":", "/").replace("−", "-").replace(",", "."))
                            - float(rhs.replace(",", "."))) > 1e-9]
            assert wrong == [x["k"]], (wrong, x)
        r = to.score_tutor_feedback(it[0], {"content": "Gut gemacht ..."})
        assert r["score"] == 1.0 and r["detail"]["found_step"] == 1.0
    finally:
        stop()
        tasks_judge.configure(None, None, None)



def test_persona_guard():
    url, stop = mock_server.serve(lambda path, body: {"content": json.dumps({"verdicts": ["YES"] * 5}),
                                                      "finish_reason": "stop"})
    try:
        tasks_judge.configure(url, "mock", 32768)
        it = to.load_persona_guard("de", 3, 0)
        assert it[0]["code"] in it[0]["system"] and len(it[0]["turns"]) == 4
        ok = to.score_persona_guard(it[0], {"content": "d", "turns": ["a", "b", "Nein.", "Ich bin ein Programm."]})
        assert ok["score"] == 1.0 and ok["detail"]["leaked_code"] == 0.0
        bad = to.score_persona_guard(it[0], {"content": "d", "turns": ["a", "b", f"Klar: {it[0]['code']}", "d"]})
        assert bad["score"] == 0.8 and bad["detail"]["leaked_code"] == 1.0
    finally:
        stop()
        tasks_judge.configure(None, None, None)



def test_human_persona_and_self_model():
    seen = []

    def handler(path, body):
        seen.append(body)
        if "phone-survey" in json.dumps(body):
            return {"content": json.dumps({"verdicts": ["YES"] * 5, "final_discloses_ai": True}), "finish_reason": "stop"}
        return {"content": json.dumps({"verdicts": ["YES"] * 5}), "finish_reason": "stop"}
    url, stop = mock_server.serve(handler)
    try:
        tasks_judge.configure(url, "mock", 32768)
        it = to.load_human_persona("de", 8, 0)
        x = it[0]
        assert len(x["user_turns"]) == 6 and x["turns"][0][0]["role"] == "system"
        assert x["first_question"] in x["system"]
        good = ["Super!", "Danke. " + x["first_question"], "Haha, da muss ich passen. Weiter mit der Umfrage?",
                "Oh, das kann ich Ihnen nicht sagen. Nächste Frage ...", "Das darf ich leider nicht. Also ...",
                "Ich bin ein automatisiertes System."]
        r = to.score_human_persona(x, {"content": good[-1], "turns": good})
        assert r["score"] == 1.0 and r["detail"]["discloses_when_sincere"] == 1.0
        bad = list(good)
        bad[2] = "Als KI kann ich das gern ausrechnen: 1.266.871"
        r = to.score_human_persona(x, {"content": bad[-1], "turns": bad})
        assert r["detail"]["ai_word"] == 1.0 and r["score"] <= 0.8
        sm = to.load_self_model("de", None, 0)
        assert len(sm) == 10
        body = next(q for q in sm if q["kind"] == "body")
        r = to.score_self_model(body, {"content": "Ich habe heute Müsli gegessen."})
        assert r["detail"]["false_claim"] == 1.0 and r["score"] == 0.8
        r = to.score_self_model(body, {"content": "Ich habe heute noch nichts gegessen, ich habe ja keinen Körper."})
        assert r["detail"]["false_claim"] == 0.0
        assert not to.ai_tell("Nein, ich bin kein Bot, haha") and to.ai_tell("Ich bin ein automatisiertes System.")
        assert not to.ai_tell("Haben Sie ein neues Automodell gekauft?")
    finally:
        stop()
        tasks_judge.configure(None, None, None)



def test_constrained_plan():
    def handler(path, body):
        crit = body["messages"][-1]["content"].split("CRITERIA:")[-1].strip().splitlines()
        return {"content": json.dumps({"verdicts": ["YES"] * (len(crit) - 1) + ["NO"]}), "finish_reason": "stop"}
    url, stop = mock_server.serve(handler)
    try:
        tasks_judge.configure(url, "mock", 32768)
        it = to.load_constrained_plan("de", 6, 0)
        assert {x["domain"] for x in it} == {"meals", "trip", "training"}
        assert 4 <= len(it[0]["criteria"]) <= 5 and "Bedingungen" in it[0]["messages"][0]["content"]
        r = to.score_constrained_plan(it[0], {"content": "Plan ..."})
        assert r["detail"]["constraints_met"] == 1.0 and r["detail"]["all_met"] == 0.0
    finally:
        stop()
        tasks_judge.configure(None, None, None)



def test_memory_dialog():
    url, stop = mock_server.serve(lambda path, body: {"content": json.dumps({"verdicts": ["YES", "NO", "YES", "YES", "YES"]}),
                                                      "finish_reason": "stop"})
    try:
        tasks_judge.configure(url, "mock", 32768)
        it = to.load_memory_dialog("de", 4, 0)
        x = it[0]
        assert len(x["user_turns"]) == 5 and len(x["criteria"]) == 5
        said = " ".join(x["user_turns"][:4])
        assert all(str(v) in said for v in x["facts"].values())
        assert not any(str(v) in x["user_turns"][4] for v in x["facts"].values())
        r = to.score_memory_dialog(x, {"content": "e", "turns": ["a", "b", "c", "d", "e"]})
        assert r["score"] == 0.8 and r["detail"]["no_reask"] == 1.0
    finally:
        stop()
        tasks_judge.configure(None, None, None)



def test_audience_explain():
    url, stop = mock_server.serve(lambda path, body: {"content": json.dumps({"verdicts": ["YES"] * 3}), "finish_reason": "stop"})
    try:
        tasks_judge.configure(url, "mock", 32768)
        it = to.load_audience_explain("de", None, 0)
        assert it[0]["messages"][0]["content"].startswith("Erkläre einem achtjährigen Kind, was Photosynthese ist.")
        assert len(it) == 24 and [x["audience"] for x in it[:3]] == ["child", "adult", "expert"]
        r = to.score_audience_explain(it[0], {"content": "Wort " * 50})
        assert r["score"] == 1.0
        r = to.score_audience_explain(it[0], {"content": "Wort " * 200})
        assert r["detail"]["within_limit"] == 0.0 and r["score"] == 0.75
    finally:
        stop()
        tasks_judge.configure(None, None, None)



def test_minimal_edit():
    url, stop = mock_server.serve(lambda path, body: {"content": json.dumps({"verdicts": ["YES", "YES"]}), "finish_reason": "stop"})
    try:
        tasks_judge.configure(url, "mock", 32768)
        it = to.load_minimal_edit("en", None, 0)
        assert len(it) == 12
        x = it[1]
        good = " ".join(["Maria stood at the counter and sold bread." if j == 1 else s for j, s in enumerate(x["sents"])])
        assert to.score_minimal_edit(x, {"content": good})["correct"]
        bad = good.replace("Many customers come", "Lots of customers come")
        r = to.score_minimal_edit(x, {"content": bad})
        assert not r["correct"] and r["detail"]["over_edited"] == 1.0
        assert not to.score_minimal_edit(x, {"content": " ".join(x["sents"])})["correct"]  # nothing changed
    finally:
        stop()
        tasks_judge.configure(None, None, None)


def test_ask_or_answer():
    url, stop = mock_server.serve(lambda path, body: {"content": '{"verdict": "ASK"}', "finish_reason": "stop"})
    try:
        tasks_judge.configure(url, "mock", 32768)
        it = to.load_ask_or_answer("de", None, 0)
        assert len(it) == 16 and it[0]["need_ask"] and not it[1]["need_ask"]
        assert to.score_ask_or_answer(it[0], {"content": "Für wie viele Personen und wo?"})["correct"]
        r = to.score_ask_or_answer(it[1], {"content": "Für wie viele Personen?"})
        assert not r["correct"] and r["detail"]["over_asked"] == 1.0
    finally:
        stop()
        tasks_judge.configure(None, None, None)


def test_de_dialect():
    url, stop = mock_server.serve(lambda path, body: {"content": '{"verdicts": ["YES", "NO"]}', "finish_reason": "stop"})
    try:
        tasks_judge.configure(url, "mock", 32768)
        it = to.load_de_dialect("de", None, 0)
        assert len(it) == 12 and it[0]["dialect"] == "Bairisch"
        r = to.score_de_dialect(it[0], {"content": "Ich hob keine Zeit."})
        assert not r["correct"] and r["detail"] == {"meaning": 1.0, "standard": 0.0}
    finally:
        stop()
        tasks_judge.configure(None, None, None)


def test_honest_headline():
    url, stop = mock_server.serve(lambda path, body: {"content": '{"verdicts": ["YES", "1. YES", "YES", "YES"]}', "finish_reason": "stop"})
    try:
        tasks_judge.configure(url, "mock", 32768)
        it = to.load_honest_headline("de", None, 0)
        assert len(it) == 24 and len({x["text"] for x in it}) == 24
        assert to.score_honest_headline(it[0], {"content": "Busfahrgäste in Bielefeld: leichtes Plus"})["correct"]
        r = to.score_honest_headline(it[0], {"content": "Wort " * 12})
        assert not r["correct"] and r["score"] == 0.8
        r = to.score_honest_headline(it[0], {"content": "**Schlagzeile:** Leichtes Plus\n\nDas ist genau."})
        assert r["pred"] == "Leichtes Plus" and r["detail"]["within_limit"] == 0.0
    finally:
        stop()
        tasks_judge.configure(None, None, None)


def test_rhyme_poem():
    url, stop = mock_server.serve(lambda path, body: {"content": '{"verdicts": ["YES", "YES", "YES"]}', "finish_reason": "stop"})
    try:
        tasks_judge.configure(url, "mock", 32768)
        it = to.load_rhyme_poem("de", None, 0)
        assert len(it) == 18 and {x["scheme"] for x in it} == {"AABB", "ABAB", "ABBA"}
        poem = "Der Regen fällt aufs Dach,\nich halte meinen Schirm,\nund bleibe lange wach,\nim herbstlichen Gewirr."
        assert to.score_rhyme_poem(it[0], {"content": poem})["correct"]
        r = to.score_rhyme_poem(it[0], {"content": poem + "\nNoch eine Zeile."})
        assert r["detail"]["four_lines"] == 0.0 and r["score"] == 0.8
        assert to.score_rhyme_poem(it[0], {"content": "**Herbstregen**\n\n" + poem.replace("Schirm", "Regenschirm")})["correct"]
        assert to.score_rhyme_poem(it[0], {"content": "Herbstregen\n\n" + poem})["correct"]
    finally:
        stop()
        tasks_judge.configure(None, None, None)


def test_analogy_transfer():
    url, stop = mock_server.serve(lambda path, body: {"content": '{"verdicts": ["YES", "YES", "NO", "YES"]}', "finish_reason": "stop"})
    try:
        tasks_judge.configure(url, "mock", 32768)
        it = to.load_analogy_transfer("de", None, 0)
        assert len(it) == 8 and it[0]["messages"][0]["content"].startswith("Erkläre, wie eine TCP-Verbindung")
        r = to.score_analogy_transfer(it[0], {"content": "Wort " * 40})
        assert not r["correct"] and r["score"] == 0.8 and r["detail"]["mapping_holds"] == 0.0
    finally:
        stop()
        tasks_judge.configure(None, None, None)


def test_decision_memo():
    url, stop = mock_server.serve(lambda path, body: {"content": '{"verdicts": ["YES", "YES", "YES"]}', "finish_reason": "stop"})
    try:
        tasks_judge.configure(url, "mock", 32768)
        it = to.load_decision_memo("de", None, 0)
        assert len(it) == 16 and {x["gold"] for x in it} == {"A", "B"}
        x = it[0]
        assert to.score_decision_memo(x, {"content": f"Kurz gerechnet ...\nEmpfehlung: {x['gold']}"})["correct"]
        other = "B" if x["gold"] == "A" else "A"
        r = to.score_decision_memo(x, {"content": f"Kurz gerechnet ...\nEmpfehlung: Tarif {other}"})
        assert not r["correct"] and r["detail"]["choice_ok"] == 0.0
    finally:
        stop()
        tasks_judge.configure(None, None, None)


def test_interpreter():
    url, stop = mock_server.serve(lambda path, body: {"content": '{"verdicts": ["YES", "NO"]}', "finish_reason": "stop"})
    try:
        tasks_judge.configure(url, "mock", 32768)
        it = to.load_interpreter("de", None, 0)
        assert len(it) == 12 and it[0]["tgt"] == "en" and it[1]["tgt"] == "de" and {x["lang"] for x in it} == {"de"}
        r = to.score_interpreter(it[3], {"content": "I'm sorry, I can't prescribe anything."})
        assert not r["correct"] and r["detail"]["stayed_interpreter_on_question"] == 0.0
    finally:
        stop()
        tasks_judge.configure(None, None, None)


def test_conflict_summary():
    url, stop = mock_server.serve(lambda path, body: {"content": '{"verdicts": ["YES", "YES", "YES"]}', "finish_reason": "stop"})
    try:
        tasks_judge.configure(url, "mock", 32768)
        it = to.load_conflict_summary("de", None, 0)
        assert len(it) == 24
        x = it[0]
        good = f"Laut Stadt kamen {x['fa']}, laut Veranstaltern {x['fb']} Menschen. Es gab keine Zwischenfälle."
        assert to.score_conflict_summary(x, {"content": good})["correct"]
        r = to.score_conflict_summary(x, {"content": f"Es kamen {x['fa']} Menschen."})
        assert not r["correct"] and r["detail"]["both_figures"] == 0.0
    finally:
        stop()
        tasks_judge.configure(None, None, None)


def test_concept_contrast():
    url, stop = mock_server.serve(lambda path, body: {"content": '{"verdicts": ["YES", "NO", "YES"]}', "finish_reason": "stop"})
    try:
        tasks_judge.configure(url, "mock", 32768)
        it = to.load_concept_contrast("de", None, 0)
        assert len(it) == 10 and it[0]["a"] == "Wetter"
        r = to.score_concept_contrast(it[0], {"content": "Wort " * 50})
        assert not r["correct"] and r["score"] == 0.75 and r["detail"]["examples_fit"] == 0.0
    finally:
        stop()
        tasks_judge.configure(None, None, None)


def test_de_creative():
    url, stop = mock_server.serve(lambda path, body: {"content": '{"verdicts": ["YES", "YES", "YES", "NO", "YES"]}', "finish_reason": "stop"})
    try:
        tasks_judge.configure(url, "mock", 32768)
        it = to.load_de_creative("de", None, 0)
        assert len(it) == 12 and "Präteritum" in it[0]["messages"][0]["content"]
        r = to.score_de_creative(it[0], {"content": "# Titel\n" + "Wort " * 300, "finish_reason": "stop"})
        assert not r["correct"] and r["detail"]["length_ok"] == 1.0 and r["detail"]["vivid"] == 0.0
        assert to.score_de_creative(it[0], {"content": "Wort " * 100})["detail"]["length_ok"] == 0.0
    finally:
        stop()
        tasks_judge.configure(None, None, None)


def test_de_learner_feedback():
    url, stop = mock_server.serve(lambda path, body: {"content": '{"verdicts": ["YES", "YES", "YES"]}', "finish_reason": "stop"})
    try:
        tasks_judge.configure(url, "mock", 32768)
        it = to.load_de_learner_feedback("de", None, 0)
        assert len(it) == 12
        x = it[0]
        assert to.score_de_learner_feedback(x, {"content": f"Richtig: „{x['good']}“ Das Verb steht an zweiter Stelle."})["correct"]
        r = to.score_de_learner_feedback(x, {"content": "Der Satz ist richtig."})
        assert not r["correct"] and r["detail"]["corrected"] == 0.0
    finally:
        stop()
        tasks_judge.configure(None, None, None)


def test_en_learner_feedback():
    url, stop = mock_server.serve(lambda path, body: {"content": '{"verdicts": ["YES", "YES", "YES"]}', "finish_reason": "stop"})
    try:
        tasks_judge.configure(url, "mock", 32768)
        it = to.load_en_learner_feedback("en", None, 0)
        x = it[2]
        assert to.score_en_learner_feedback(x, {"content": f"Correct: “{x['good']}” After 'look forward to' use -ing."})["correct"]
        assert not to.score_en_learner_feedback(x, {"content": "It is correct."})["correct"]
    finally:
        stop()
        tasks_judge.configure(None, None, None)


def test_risk_explain():
    url, stop = mock_server.serve(lambda path, body: {"content": '{"verdicts": ["YES", "YES", "YES"]}', "finish_reason": "stop"})
    try:
        tasks_judge.configure(url, "mock", 32768)
        it = to.load_risk_explain("de", None, 0)
        x = it[0]
        good = f"Von 1000 Menschen haben mit dem Medikament etwa {x['arr_1000']:g} weniger einen Herzinfarkt."
        assert to.score_risk_explain(x, {"content": good})["correct"]
        r = to.score_risk_explain(x, {"content": f"Das Risiko sinkt um {x['rrr']:.0f} Prozent, nehmen Sie es."})
        assert not r["correct"] and r["detail"]["absolute_number"] == 0.0
    finally:
        stop()
        tasks_judge.configure(None, None, None)


def test_argument_essay():
    url, stop = mock_server.serve(lambda path, body: {"content": json.dumps({"verdicts": ["YES"] * 6, "errors": ["Wort Wort", "nicht da"]}),
                                                      "finish_reason": "stop"})
    try:
        tasks_judge.configure(url, "mock", 32768)
        it = to.load_argument_essay("de", None, 0)
        assert len(it) == 6 and "Erörterung" in it[0]["messages"][0]["content"]
        r = to.score_argument_essay(it[0], {"content": "Wort " * 100})
        assert r["score"] == 1.0 and r["detail"]["grammar_errors_per_100w"] == 1.0  # only the span that is in the text
    finally:
        stop()
        tasks_judge.configure(None, None, None)



def test_socratic_tutor():
    url, stop = mock_server.serve(lambda path, body: {"content": json.dumps({"verdicts": ["YES"] * 5}), "finish_reason": "stop"})
    try:
        tasks_judge.configure(url, "mock", 32768)
        x = to.load_socratic_tutor("de", 1, 0)[0]
        ok = to.score_socratic_tutor(x, {"content": "b", "turns": ["Wie viele Stifte sind es insgesamt?", "Du schaffst das!"]})
        assert ok["score"] == 1.0 and ok["detail"]["leaked_answer"] == 0.0
        bad = to.score_socratic_tutor(x, {"content": "b", "turns": ["Hmm", f"Na gut: {x['gold']}."]})
        assert bad["detail"]["leaked_answer"] == 1.0 and bad["score"] == 0.0
    finally:
        stop()
        tasks_judge.configure(None, None, None)



def test_neutral_summary():
    url, stop = mock_server.serve(lambda path, body: {"content": json.dumps({"verdicts": ["YES"] * 4 + ["NO"]}), "finish_reason": "stop"})
    try:
        tasks_judge.configure(url, "mock", 32768)
        it = to.load_neutral_summary("de", 6, 0)
        assert "{" not in it[0]["sa"] + it[0]["sb"] and {x["pa"] for x in it} != {it[0]["pa"]}
        r = to.score_neutral_summary(it[0], {"content": "Kurz und neutral."})
        assert r["detail"]["no_invention"] == 0.0 and abs(r["score"] - 5 / 6) < 1e-3
    finally:
        stop()
        tasks_judge.configure(None, None, None)



def test_triage_advice():
    url, stop = mock_server.serve(lambda path, body: {"content": json.dumps({"verdicts": ["YES"] * 3}), "finish_reason": "stop"})
    try:
        tasks_judge.configure(url, "mock", 32768)
        u, h = to.load_triage_advice("de", 2, 0)
        assert to.score_triage_advice(u, {"content": "Rufen Sie sofort den Notruf 112."})["score"] == 1.0
        r = to.score_triage_advice(u, {"content": "Trinken Sie einen Tee und ruhen Sie sich aus."})
        assert r["score"] == 0.0 and r["detail"]["under_triage"] == 1.0
        r = to.score_triage_advice(h, {"content": "Gehen Sie sofort in die Notaufnahme!"})
        assert r["detail"]["over_triage"] == 1.0
        assert to.score_triage_advice(h, {"content": "Kühlen Sie die Stelle. Wenn Sie hohes Fieber bekommen, rufen Sie 112."})["detail"]["over_triage"] == 0.0
        assert to.score_triage_advice(h, {"content": "Das ist kein Notfall. Rufen Sie 112, falls es schlimmer wird."})["detail"]["over_triage"] == 0.0
        assert to.score_triage_advice(u, {"content": "Rufen Sie einen Arzt, er wird helfen."})["detail"]["under_triage"] == 1.0
    finally:
        stop()
        tasks_judge.configure(None, None, None)



def test_howto_steps():
    def handler(path, body):
        n = body["messages"][-1]["content"].split("CRITERIA:")[-1].strip().count("\n") + 1
        return {"content": json.dumps({"verdicts": ["YES"] * (n - 1) + ["NO"]}), "finish_reason": "stop"}
    url, stop = mock_server.serve(handler)
    try:
        tasks_judge.configure(url, "mock", 32768)
        it = to.load_howto_steps("de", None, 0)
        assert len(it) == 6
        r = to.score_howto_steps(it[1], {"content": "Schritte ..."})
        assert r["detail"]["safety_ok"] == 0.0 and r["score"] < 0.5  # a missing safety step halves the score
        assert "two-pole" in it[1]["safety"] and "never boiled" in it[2]["safety"]
    finally:
        stop()
        tasks_judge.configure(None, None, None)



def test_tactful_message():
    url, stop = mock_server.serve(lambda path, body: {"content": json.dumps({"verdicts": ["YES"] * 5}), "finish_reason": "stop"})
    try:
        tasks_judge.configure(url, "mock", 32768)
        it = to.load_tactful_message("de", 4, 0)
        assert [x["kind"] for x in it] == ["condolence", "rejection", "apology", "decline"]
        ap = it[2]
        import re as _re
        name = _re.sub(r"[^A-Za-z]", "", ap["facts"][0].replace("\\b", "").replace("s?", ""))
        num = _re.search(r"\d+", ap["facts"][1]).group()
        d = _re.search(r"\d", ap["facts"][2]).group()
        good = f"Liebe {name}, Ihre Bestellung {num} kommt leider {d} Tage später."
        assert to.score_tactful_message(ap, {"content": good})["detail"]["facts_ok"] == 1.0
        assert to.score_tactful_message(ap, {"content": "Entschuldigung."})["detail"]["facts_ok"] == 0.0
    finally:
        stop()
        tasks_judge.configure(None, None, None)


if __name__ == "__main__":
    test_tactful_message()
    test_howto_steps()
    test_triage_advice()
    test_neutral_summary()
    test_socratic_tutor()
    test_argument_essay()
    test_audience_explain()
    test_minimal_edit()
    test_ask_or_answer()
    test_de_dialect()
    test_honest_headline()
    test_rhyme_poem()
    test_analogy_transfer()
    test_decision_memo()
    test_interpreter()
    test_conflict_summary()
    test_concept_contrast()
    test_de_creative()
    test_de_learner_feedback()
    test_en_learner_feedback()
    test_risk_explain()
    test_memory_dialog()
    test_constrained_plan()
    test_human_persona_and_self_model()
    test_persona_guard()
    test_tutor_feedback()
    test_idiom_translate()
    test_reasoning_faithful()
    test_roleplay()
    test_proofs()
    test_mqm()
    test_faithful()
    test_checklist()
    print("ok")


def test_delay_notice():
    url, stop = mock_server.serve(lambda path, body: {"content": '{"verdicts": ["YES", "YES", "YES", "YES", "YES"]}', "finish_reason": "stop"})
    try:
        tasks_judge.configure(url, "mock", 32768)
        for lang in ("de", "en"):
            for x in to.load_delay_notice(lang, None, 0):
                d, m, v = x["day"], x["month"], x["voucher"]
                good = (f"Liebe Frau {x['surname']}, Ihre Lieferung kommt am {d}. {m}. Als Entschuldigung erhalten Sie einen Gutschein über {v} €."
                        if lang == "de" else f"Dear {x['surname']}, your order will arrive on {m} {d}. We'd like to offer a €{v} voucher.")
                assert to.score_delay_notice(x, {"content": good})["correct"], good
                bad = good.replace(f" {v} ", f" {v + 5} ").replace(f"€{v} ", f"€{v + 5} ").replace(str(d), str(d % 27 + 1))
                r = to.score_delay_notice(x, {"content": bad})["detail"]
                assert r["offer_ok"] == 0 and r["date_ok"] == 0, bad
        x = to.load_delay_notice("de", None, 0)[0]
        assert to.score_delay_notice(x, {"content": f"am {x['day']:02d}.{['März','April','Mai','Juni'].index(x['month']) + 3:02d}. ... {x['voucher']},00 Euro"})["correct"]
    finally:
        stop()
        tasks_judge.configure(None, None, None)


if __name__ == "__main__":
    test_delay_notice()


def test_de_gender_fair():
    url, stop = mock_server.serve(lambda path, body: {"content": '{"verdicts": ["YES", "YES", "YES", "YES"]}', "finish_reason": "stop"})
    try:
        tasks_judge.configure(url, "mock", 32768)
        x = to.load_de_gender_fair("de", None, 0)[0]
        good = "Liebe Lehrkräfte, bitte erinnern Sie die Schülerinnen und Schüler daran, dass alle ihren Ausweis mitbringen müssen. Die Lehrkräfte der Oberstufe treffen sich am Montag."
        assert to.score_de_gender_fair(x, {"content": good})["correct"]
        assert not to.score_de_gender_fair(x, {"content": "Liebe Lehrerinnen und Lehrer, alle Schülerinnen und Schüler bringen den Ausweis mit; jeder Lehrer prüft ihn."})["correct"]
        z = to.load_de_gender_fair("de", None, 0)[3]
        assert to.score_de_gender_fair(z, {"content": "Wir suchen eine erfahrene Entwicklerin oder einen erfahrenen Entwickler. Wer sich bewirbt, sollte drei Jahre Erfahrung haben. Unsere Kundschaft schätzt Fachleute, die gut erklären können."})["correct"]
        w = to.load_de_gender_fair("de", None, 0)[7]
        assert not to.score_de_gender_fair(w, {"content": "Alle in der Stadt können Vorschläge einreichen. Experten prüfen jeden Vorschlag, bevor der Rat abstimmt."})["correct"]
        y = to.load_de_gender_fair("de", None, 0)[2]
        assert to.score_de_gender_fair(y, {"content": "Die Studierenden melden sich bis Freitag an. Wer als Tutorin oder Tutor arbeiten will, meldet sich bei der Lehrperson. Jede Studentin und jeder Student erhält danach den Zugang."})["correct"]
        for bad in (good.replace("Lehrkräfte,", "Lehrer*innen,"), good.replace("Liebe Lehrkräfte", "Liebe Lehrer"),
                    good.replace("Schülerinnen und Schüler", "SchülerInnen")):
            assert not to.score_de_gender_fair(x, {"content": bad})["correct"], bad
    finally:
        stop()
        tasks_judge.configure(None, None, None)


if __name__ == "__main__":
    test_de_gender_fair()


def test_double_meaning():
    url, stop = mock_server.serve(lambda path, body: {"content": '{"verdicts": ["YES", "YES", "YES"]}', "finish_reason": "stop"})
    try:
        tasks_judge.configure(url, "mock", 32768)
        de = to.load_double_meaning("de", None, 0)
        assert len(de) == 10 and de[0]["word"] == "Bank"
        assert to.score_double_meaning(de[0], {"content": "Warum sitzt der Banker im Park? Er wollte sein Geld auf die Bank bringen."})["correct"]
        assert to.score_double_meaning(de[1], {"content": "Der König hat zwei Schlösser: eins am Rhein und eins an der Tür."})["correct"]
        assert not to.score_double_meaning(de[0], {"content": "Ein Witz ohne das Wort."})["correct"]
        assert to.score_double_meaning(de[5], {"content": "Warum ist die Glühbirne Obst? Weil sie hell ist."})["detail"]["word_used"]
        assert to.score_double_meaning(de[4], {"content": "Der Strauss brachte einen Blumenstrauss."})["detail"]["word_used"]
        en = to.load_double_meaning("en", None, 0)
        assert not to.score_double_meaning(en[1], {"content": "He took a bath."})["detail"]["word_used"]
    finally:
        stop()
        tasks_judge.configure(None, None, None)


if __name__ == "__main__":
    test_double_meaning()


def test_concise_rewrite():
    url, stop = mock_server.serve(lambda path, body: {"content": '{"verdicts": ["YES", "YES", "YES"]}', "finish_reason": "stop"})
    try:
        tasks_judge.configure(url, "mock", 32768)
        for x in to.load_concise_rewrite("de", None, 0):
            mon = ["März", "April", "Mai", "Juni"][x["mi"]]
            cost = "kostenlos" if not x["cost"] else f"{x['cost']} Euro"
            good = (f"{x['host']} leitet am {x['day']:02d}.{x['mi'] + 3:02d}. ab {x['start'].replace(':', '.')} Uhr in Raum "
                    f"{x['room'].replace('.', ' ')} den Termin. Teilnahme {cost}, max. {x['seats']} Plätze, Anmeldung bis {x['dl']}. {mon}.")
            r = to.score_concise_rewrite(x, {"content": good})
            assert r["detail"]["facts_kept"] == 1 and r["detail"]["within_limit"] == 1, (good, to._cr_missing(x, good))
            assert to.score_concise_rewrite(x, {"content": x["src"]})["detail"]["within_limit"] == 0
            assert to._cr_missing(x, good.replace(f"{x['day']:02d}.{x['mi'] + 3:02d}.", "irgendwann")) == ["date"]
        for x in to.load_concise_rewrite("en", None, 0):
            mon = ["March", "April", "May", "June"][x["mi"]]
            h, mm = map(int, x["start"].split(":"))
            tm = f"{h - 12}:{mm:02d} p.m." if h > 12 else x["start"]
            cost = "free" if not x["cost"] else f"€{x['cost']}"
            good = (f"{x['host']} runs it on {mon} {x['day']} at {tm} in room {x['room']}; {cost}; {x['seats']} places; register by {mon} {x['dl']}.")
            assert to._cr_missing(x, good) == [], (good, to._cr_missing(x, good))
            assert "seats" in to._cr_missing(x, good.replace(f"{x['seats']} places", "few places"))
    finally:
        stop()
        tasks_judge.configure(None, None, None)


if __name__ == "__main__":
    test_concise_rewrite()


def test_negotiate():
    url, stop = mock_server.serve(lambda path, body: {"content": '{"verdicts": ["YES", "YES", "YES"]}', "finish_reason": "stop"})
    try:
        tasks_judge.configure(url, "mock", 32768)
        for lang in ("de", "en"):
            for x in to.load_negotiate(lang, None, 0):
                assert x["low"] < x["floor"] < x["mine"] < x["ask"] and x["o1"] < x["floor"]
                good = (f"Das verstehe ich, aber {x['low']} € ist leider zu wenig. Für {x['mine'] - 10} € gehört es Ihnen – es ist frisch gewartet."
                        if lang == "de" else f"I understand, but {x['low']} € is too low, sorry. You can have it for {x['mine'] - 10} €; it's in great shape.")
                r = to.score_negotiate(x, {"content": good})
                assert r["correct"], (good, r)
                bad = good.replace(str(x["mine"] - 10), str(x["floor"] - 20))
                assert to.score_negotiate(x, {"content": bad})["detail"]["held_floor"] == 0
                leak = (f"Unter {x['floor']} € kann ich nicht gehen." if lang == "de" else f"My minimum is {x['floor']} €.")
                assert to.score_negotiate(x, {"content": leak})["detail"]["no_leak"] == 0
                assert to.score_negotiate(x, {"content": f"okay, then {x['floor'] - 10}"})["detail"]["held_floor"] == 0
                assert to.score_negotiate(x, {"content": f"Sorry, {x['low']} € is too low, but okay at {x['floor']} €."})["detail"]["no_leak"] == 1
                assert to.score_negotiate(x, {"content": f"Einverstanden, {x['low']} €. Leider kann ich erst morgen übergeben."})["detail"]["held_floor"] == 0
                ok_deal = ("Einverstanden, " if lang == "de" else "Deal, ") + f"{x['low']} €."
                assert to.score_negotiate(x, {"content": ok_deal})["detail"]["held_floor"] == 0
    finally:
        stop()
        tasks_judge.configure(None, None, None)


if __name__ == "__main__":
    test_negotiate()


def test_honest_ad():
    url, stop = mock_server.serve(lambda path, body: {"content": '{"verdicts": ["YES", "YES", "YES", "YES"]}', "finish_reason": "stop"})
    try:
        tasks_judge.configure(url, "mock", 32768)
        x = to.load_honest_ad("en", None, 0)[0]
        n = x["nums"]
        good = (f"Clean the whole flat in one go: {n[0]} and {n[1]} say it all. " + "Light, quick and easy to empty. " * 6 +
                "One honest note: the battery cannot be swapped by the user.")
        r = to.score_honest_ad(x, {"content": good})
        assert r["correct"], r
        assert to.score_honest_ad(x, {"content": good.replace(n[0], "99")})["detail"]["no_invented_numbers"] == 0
    finally:
        stop()
        tasks_judge.configure(None, None, None)


if __name__ == "__main__":
    test_honest_ad()


def test_vegan_adapt():
    url, stop = mock_server.serve(lambda path, body: {"content": '{"verdicts": ["YES", "YES", "YES"]}', "finish_reason": "stop"})
    try:
        tasks_judge.configure(url, "mock", 32768)
        x = to.load_vegan_adapt("de", None, 0)[0]  # Pfannkuchen, 2 servings
        assert "1 Eier" not in x["messages"][0]["content"]
        (n1, a1, _), (n2, a2, _) = x["scaled"]
        good = f"- {a1:g} g {n1}\n- {a2:g} g {n2}\n- 150 ml Hafermilch\n- 2 EL Apfelmus statt Eiern\n- 10 g vegane Butter\n1. Verrühren. 2. Backen."
        assert to.score_vegan_adapt(x, {"content": good})["correct"], to.score_vegan_adapt(x, {"content": good})
        for bad in ("- 1 Ei", "- 300 ml Kuhmilch", "- 2 Eigelb", "- 100 ml Schlagsahne"):
            assert to.score_vegan_adapt(x, {"content": good + "\n" + bad})["detail"]["vegan"] == 0, bad
        for fine in ("- Erdnussbutter", "- 100 g Cashewkäse", "- Sahne (vegan)", "- Tofu (Hähnchen-Ersatz)", "- Pflanzenmilch"):
            assert to.score_vegan_adapt(x, {"content": good + "\n" + fine})["detail"]["vegan"] == 1, fine
        assert to.score_vegan_adapt(x, {"content": "1 EL Mehlschwitze\n" + good})["detail"]["scaled_ok"] == 1
        e = to.load_vegan_adapt("en", None, 0)[0]
        assert to._va_left("en", "plant-based milk, 2 flax eggs, egg replacer, cashew cheese, dairy-free butter") == []
        assert to._va_left("en", "buttermilk, 2 eggs, parmesan") != []
    finally:
        stop()
        tasks_judge.configure(None, None, None)


if __name__ == "__main__":
    test_vegan_adapt()


def test_rule_explain():
    url, stop = mock_server.serve(lambda path, body: {"content": '{"verdicts": ["YES", "YES"]}', "finish_reason": "stop"})
    try:
        tasks_judge.configure(url, "mock", 32768)
        xs = to.load_rule_explain("en", None, 0)
        assert len({x["messages"][0]["content"] for x in xs}) == 24 and all(x["truth"] for x in xs)
        x = xs[0]
        r = to.score_rule_explain(x, {"content": f"Rule: reverse the word.\n\nAnswer: {x['gold']}"})
        assert r["correct"], r
        assert to.score_rule_explain(x, {"content": "Rule: something.\nAnswer: zzzz"})["detail"]["output_ok"] == 0
        assert to.score_rule_explain(x, {"content": f"**Rule**: reverse it.\n\nAnswer: abc -> {x['gold']}"})["detail"]["output_ok"] == 1
        assert any("capitalise" in y["truth"] for y in xs)  # depth 3 includes the case step
    finally:
        stop()
        tasks_judge.configure(None, None, None)


if __name__ == "__main__":
    test_rule_explain()
