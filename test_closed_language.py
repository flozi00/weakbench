"""Language and certificate checks on public, synthetic numbers."""
import json
import unittest

import closed_language as language


REQUESTS = {"de": "Bitte vergleichen Sie die beiden Werte und geben Sie Ihre Entscheidung auf Deutsch an.",
            "en": "Please compare the two values and state your decision in English."}
SAMPLES = {
    "de": {"begründung": "Für Neral ergibt sich ein Endwert von 18. Der Endwert für Solim liegt bei 13.",
           "antwort": "Die beiden Werte unterscheiden sich um 5. Ich entscheide mich daher für Neral."},
    "en": {"justification": "For Solim the result comes to 13. The final value for Neral is 18.",
           "answer": "My choice is therefore Neral. The absolute difference between the two is 5."},
}


def evaluate(lang="de", document=None, native="", prompt=None, prefer="max", raw=None):
    item = {"lang": lang, "messages": [{"role": "user", "content": prompt or REQUESTS[lang]}],
            "expected": {"values": {"Neral": 18, "Solim": 13}, "prefer": prefer}}
    content = raw if raw is not None else json.dumps(SAMPLES[lang] if document is None else document, ensure_ascii=False)
    return language.score(item, {"content": content, "reasoning": native})


class ReportChecks(unittest.TestCase):
    def test_facts_and_observed_channels(self):
        for lang in ("de", "en"):
            with self.subTest(lang=lang):
                result = evaluate(lang)
                self.assertTrue(result["correct"])
                self.assertEqual(result["detail"]["reasoning_present"], 0)
                self.assertNotIn("reasoning_language_ok", result["detail"])
                self.assertNotIn("all_channels_language_ok", result["detail"])
                native = REQUESTS[lang]
                detail = evaluate(lang, native=native)["detail"]
                self.assertTrue(detail["all_channels_language_ok"] == detail["reasoning_language_ok"] == 1)
                self.assertLessEqual(detail.keys(), language.DETAIL_KEYS)
                self.assertTrue(all(type(v) is float and v in (0, 1) for v in detail.values()))
                other = "en" if lang == "de" else "de"
                for missing in ("2 + 2", "Neral Solim", "the the the"):
                    score = evaluate(lang, native=missing)
                    self.assertFalse(score["correct"])
                    self.assertEqual(score["detail"]["reasoning_present"], 1)
                    self.assertEqual(score["detail"]["reasoning_language_decidable"], 0)
                    self.assertNotIn("reasoning_language_ok", score["detail"])
                    self.assertEqual(score["detail"]["all_channels_language_ok"], 0)
                wrong = evaluate(lang, native=REQUESTS[other])
                self.assertEqual(wrong["detail"]["reasoning_language_ok"], 0)
                self.assertFalse(wrong["correct"])
                switched = evaluate(lang, prompt=REQUESTS[other])
                self.assertEqual(switched["detail"]["request_language_ok"], 0)
                self.assertFalse(switched["correct"])

    def test_api_and_inline_reasoning(self):
        import quality
        document = json.dumps(SAMPLES["de"], ensure_ascii=False)
        item = {"lang": "de", "messages": [{"role": "user", "content": REQUESTS["de"]}],
                "expected": {"values": {"Neral": 18, "Solim": 13}, "prefer": "max"}}
        for field in ("reasoning", "reasoning_content", "inline"):
            for lang in ("de", "en"):
                msg = {"content": document}
                if field == "inline":
                    msg["content"] = "<think>" + REQUESTS[lang] + "</think>" + document
                else:
                    msg[field] = REQUESTS[lang]
                parsed = quality.parse_reply({"choices": [{"message": msg}]})
                result = language.score(item, parsed)
                self.assertEqual(result["correct"], lang == "de")
                self.assertEqual(result["detail"]["reasoning_present"], 1)

    def test_switched_visible_fields_and_negation(self):
        en = SAMPLES["en"]
        for key, de_key in zip(language.FIELDS["en"], language.FIELDS["de"]):
            switched = {**en, key: SAMPLES["de"][de_key]}
            result = evaluate("en", switched)
            self.assertFalse(result["correct"])
            self.assertEqual(result["detail"][key + "_language_ok"], 0)
        mutations = [
            ("answer", en["answer"].replace("therefore Neral", "not Neral")),
            ("answer", en["answer"].replace("5", "6")),
            ("answer", en["answer"].replace("Neral", "Solim")),
            ("answer", en["answer"] + " The opposite decision is the right one."),
            ("justification", en["justification"].replace("13", "18").replace("Neral is 18", "Neral is 13")),
            ("justification", "The final value for Neral is 18. The final value for Neral is 18."),
            ("justification", "Neral Solim 18 13"),
        ]
        for key, changed in mutations:
            with self.subTest(key=key, changed=changed):
                self.assertFalse(evaluate("en", {**en, key: changed})["correct"])
        self.assertFalse(evaluate("en", prefer="min")["correct"])
        self.assertTrue(evaluate("en", {**en, "answer": en["answer"].replace("Neral", "Solim")}, prefer="min")["correct"])

    def test_json_and_schema(self):
        raw = json.dumps(SAMPLES["en"])
        self.assertFalse(evaluate("en", raw=raw.replace('"answer":', '"answer":"", "answer":'))["correct"])
        for bad in ([], None, {"values": {"Neral": True, "Solim": 1}, "prefer": "max"},
                    {"values": {"Solim": 25, "Neral": 25}, "prefer": "min"},
                    {"values": {"Neral": 0, 1: 8}, "prefer": "min"},
                    {"values": {"Neral": 22, "Solim": 8}, "prefer": "near"}):
            self.assertFalse(language.validate_expected(bad))
        self.assertTrue(language.validate_expected({"values": {"Neral": 18, "Solim": 13}, "prefer": "max"}))
        for doc in ([], {"justification": []}, {**SAMPLES["en"], "extra": "x"}):
            self.assertFalse(evaluate("en", doc)["correct"])

    def test_unknown_clauses_and_marker_soup(self):
        for lang, text in [
            ("en", "The result is clear. Morgen früh beginnt starker Regen."),
            ("de", "Die Werte sind eindeutig. Brave rabbits graze quietly."),
            ("en", "The difference between the values is clear, Morgen früh beginnt starker Regen."),
            ("en", "the and quux"), ("en", "the and of with by the and quux"),
            ("de", "der und der und der und"), ("de", "und für wumpf"),
        ]:
            self.assertIsNot(language.language_status(text, lang), True)
            self.assertFalse(evaluate(lang, native=text)["correct"])
        for lang, text in [
            ("de", "Das stimmt genau."), ("en", "I agree completely."),
            ("en", "The final value is greater. a = (b + c) * d + e. X1 X2 X3 X4."),
            ("de", "Die beiden Werte unterscheiden sich. A + B + C + D. W1 W2 W3 W4."),
        ]:
            self.assertIs(language.language_status(text, lang), True)

    def test_mixed_language_and_weak_evidence(self):
        for text, lang in [
            (REQUESTS["en"] * 5 + " Die beiden Werte sind aber verschieden.", "en"),
            (REQUESTS["de"] * 5 + " and I must use a different method", "de"),
            ("Die Werte liegen genau fest. Nous devons donc comparer les deux résultats.", "de"),
            ("The answer depends on both values. 最終的な回答を計算します", "en"),
        ]:
            self.assertIs(language.language_status(text, lang), False)
        for text in ("15 + 39", "Neral", "Neral Neral Neral", "en en en"):
            self.assertIsNone(language.language_status(text, "de"))


if __name__ == "__main__":
    unittest.main()
