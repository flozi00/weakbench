"""Public synthetic examples for the bilingual sentence-fusion scorer."""

import unittest

import closed_fusion


def fixture():
    return {
        "names": {"Ida": "f", "Otto": "m", "Elin": "f", "Kuno": "m"},
        "lexicon": {
            "spot": {
                "en": ["spot", "spotted", "spotted"],
                "de": ["sichtete", "gesichtet"],
            },
            "find": {"en": ["find", "found", "found"], "de": ["fand", "gefunden"]},
        },
        "events": [["Ida", "spot", "Otto", True], ["Otto", "find", "Elin", False]],
        "relation": "before",
    }


def evaluate(lang, relation, text, *, reverse=False):
    data = fixture()
    data["relation"] = relation
    if reverse:
        data["events"].reverse()
    return closed_fusion.score({"lang": lang, "expected": data}, {"content": text})


class SentenceFusionTests(unittest.TestCase):
    def assert_accepted(self, lang, relation, sentences, *, reverse=False):
        for text in sentences:
            with self.subTest(lang=lang, relation=relation, text=text, reverse=reverse):
                result = evaluate(lang, relation, text, reverse=reverse)
                self.assertIs(result["correct"], True)
                self.assertEqual(result["score"], 1)
                self.assertEqual(result["detail"]["fusion_target_grammar"], 1)
                self.assertEqual(result["detail"]["fusion_events_fraction"], 1)

    def test_before_and_after_in_both_clause_orders(self):
        self.assert_accepted(
            "en",
            "before",
            (
                "Ida did not spot Otto before Otto found Elin.",
                "Before Otto found Elin, Ida did not spot Otto.",
                "Otto found Elin after Ida did not spot Otto.",
                "After Ida hadn't spotted Otto, Otto found Elin.",
            ),
        )
        self.assert_accepted(
            "de",
            "before",
            (
                "Ida sichtete Otto nicht, bevor Otto Elin fand.",
                "Bevor Otto Elin fand, sichtete Ida Otto nicht.",
                "Otto fand Elin, nachdem Ida Otto nicht gesichtet hatte.",
                "Nachdem Ida Otto nicht gesichtet hatte, fand Otto Elin.",
            ),
        )

    def test_causal_connectors_in_both_directions(self):
        english = (
            "Otto found Elin because Ida did not spot Otto.",
            "Because Ida did not spot Otto, Otto found Elin.",
            "Ida did not spot Otto, so Otto found Elin.",
        )
        german = (
            "Otto fand Elin, weil Ida Otto nicht sichtete.",
            "Weil Ida Otto nicht sichtete, fand Otto Elin.",
            "Ida sichtete Otto nicht, deshalb fand Otto Elin.",
        )
        for lang, forward, backward in (
            (
                "en",
                english,
                (
                    "Ida did not spot Otto because Otto found Elin.",
                    "Because Otto found Elin, Ida did not spot Otto.",
                    "Otto found Elin, so Ida did not spot Otto.",
                ),
            ),
            (
                "de",
                german,
                (
                    "Ida sichtete Otto nicht, weil Otto Elin fand.",
                    "Weil Otto Elin fand, sichtete Ida Otto nicht.",
                    "Otto fand Elin, deshalb sichtete Ida Otto nicht.",
                ),
            ),
        ):
            self.assert_accepted(lang, "cause", forward)
            self.assert_accepted(lang, "cause", backward, reverse=True)
            for sentence in backward:
                with self.subTest(lang=lang, text=sentence):
                    result = evaluate(lang, "cause", sentence)
                    self.assertFalse(result["correct"])
                    self.assertEqual(result["detail"]["fusion_events_fraction"], 1)
                    self.assertEqual(result["detail"]["fusion_relation_kind"], 1)
                    self.assertEqual(result["detail"]["fusion_relation_direction"], 0)

    def test_additive_coordination_and_relative_case(self):
        for reverse in (False, True):
            self.assert_accepted(
                "de",
                "also",
                (
                    "Ida sichtete Otto nicht und Otto fand Elin.",
                    "Otto fand Elin und Ida sichtete Otto nicht.",
                    "Otto, den Ida nicht sichtete, fand Elin.",
                    "Ida sichtete Otto, der Elin fand, nicht.",
                ),
                reverse=reverse,
            )
            self.assert_accepted(
                "en",
                "also",
                (
                    "Ida did not spot Otto and Otto found Elin.",
                    "Otto found Elin and Ida did not spot Otto.",
                    "Otto, whom Ida did not spot, found Elin.",
                    "Otto, who Ida did not spot, found Elin.",
                    "Ida did not spot Otto, who found Elin.",
                ),
                reverse=reverse,
            )

    def test_changed_polarity_roles_and_attachment(self):
        cases = (
            ("en", "Ida spotted Otto and Otto found Elin."),
            ("en", "Otto did not spot Ida and Otto found Elin."),
            ("en", "Ida did not spot Otto, whom Elin found."),
            ("en", "Ida did not spot Kuno, who found Elin."),
            ("de", "Ida sichtete Otto und Otto fand Elin."),
            ("de", "Otto sichtete Ida nicht und Otto fand Elin."),
            ("de", "Ida sichtete Otto, den Elin fand, nicht."),
        )
        for lang, sentence in cases:
            with self.subTest(lang=lang, text=sentence):
                result = evaluate(lang, "also", sentence)
                self.assertFalse(result["correct"])
                self.assertEqual(result["detail"]["fusion_target_grammar"], 1)
                self.assertLess(result["detail"]["fusion_events_fraction"], 1)

    def test_incorrect_relation_with_intact_events(self):
        for relation, sentence, direction in (
            ("before", "Before Ida did not spot Otto, Otto found Elin.", 0),
            ("cause", "After Ida had not spotted Otto, Otto found Elin.", None),
            ("also", "Because Ida did not spot Otto, Otto found Elin.", None),
        ):
            with self.subTest(relation=relation, text=sentence):
                result = evaluate("en", relation, sentence)
                self.assertFalse(result["correct"])
                detail = result["detail"]
                self.assertEqual(detail["fusion_events_fraction"], 1)
                self.assertEqual(detail.get("fusion_relation_direction"), direction)
                self.assertEqual(detail["fusion_relation_kind"], float(direction == 0))

    def test_unlicensed_word_order_tense_and_relative(self):
        for lang, relation, sentence in (
            ("de", "before", "Bevor Otto Elin fand, Ida sichtete Otto nicht."),
            ("de", "before", "Bevor Otto fand Elin, sichtete Ida Otto nicht."),
            ("de", "before", "Nachdem Ida Otto nicht sichtete, fand Otto Elin."),
            ("de", "before", "Nachdem Ida Otto nicht gesichtet hat, fand Otto Elin."),
            ("de", "cause", "Ida sichtete Otto nicht, deshalb Otto fand Elin."),
            ("de", "also", "Ida sichtete Otto nicht, der Elin fand."),
            ("de", "also", "Otto, der Ida sichtete nicht, fand Elin."),
            ("en", "also", "Ida did not spot Otto, whom found Elin."),
        ):
            with self.subTest(lang=lang, relation=relation, text=sentence):
                result = evaluate(lang, relation, sentence)
                self.assertFalse(result["correct"])
                self.assertEqual(result["detail"]["fusion_target_grammar"], 0)

    def test_foreign_language_diagnostic(self):
        for lang, foreign in (
            ("en", "Ida sichtete Otto, der Elin fand, nicht."),
            ("de", "Ida did not spot Otto, who found Elin."),
        ):
            with self.subTest(lang=lang):
                result = evaluate(lang, "also", foreign)
                self.assertFalse(result["correct"])
                self.assertEqual(result["detail"]["fusion_target_grammar"], 0)
                self.assertEqual(result["detail"]["fusion_other_language"], 1)

    def test_extra_material_and_missing_text(self):
        for lang, sentence, extra, preface in (
            (
                "de",
                "Ida sichtete Otto nicht und Otto fand Elin",
                "Elin fand Kuno",
                "Die Antwort ist",
            ),
            (
                "en",
                "Ida did not spot Otto and Otto found Elin",
                "Elin found Kuno",
                "The answer is",
            ),
        ):
            join = " und " if lang == "de" else " and "
            for answer in (
                sentence + ". " + extra + ".",
                sentence + join + extra + ".",
                preface + ": " + sentence + ".",
                "",
                None,
            ):
                with self.subTest(lang=lang, answer=answer):
                    result = evaluate(lang, "also", answer)
                    self.assertFalse(result["correct"])
                    self.assertEqual(result["detail"]["fusion_target_grammar"], 0)

    def test_english_relative_can_attach_to_a_female_name(self):
        data = fixture()
        data["events"] = [
            ["Ida", "spot", "Otto", False],
            ["Elin", "find", "Ida", False],
        ]
        data["relation"] = "also"
        reply = {"content": "Ida, whom Elin found, spotted Otto."}
        self.assertTrue(
            closed_fusion.score({"lang": "en", "expected": data}, reply)["correct"]
        )
        # Both the subject and the object case of feminine singular are 'die'.
        reply = {"content": "Ida, die Elin fand, sichtete Otto."}
        self.assertFalse(
            closed_fusion.score({"lang": "de", "expected": data}, reply)["correct"]
        )
        reply = {"content": "Ida sichtete Otto und Elin fand Ida."}
        self.assertTrue(
            closed_fusion.score({"lang": "de", "expected": data}, reply)["correct"]
        )

    def test_invalid_expected(self):
        data = fixture()
        self.assertTrue(closed_fusion.validate_expected(data))
        first, second = data["events"]
        for invalid in (
            None,
            {},
            {**data, "extraneous": 1},
            {**data, "relation": "after"},
            {**data, "names": {"Ida": "x", "Otto": "m"}},
            {
                **data,
                "lexicon": {"spot": {"en": ["spot"], "de": ["sichtete", "gesichtet"]}},
            },
            {**data, "events": [first, first]},
            {**data, "events": [first]},
            {**data, "events": [first, ["Otto", "find", "Unknown", False]]},
            {**data, "events": [first, ["Otto", "find", "Otto", False]]},
            {**data, "events": [first, ["Otto", "find", "Elin", 0]]},
            {**data, "events": [["Ida", "missing", "Otto", False], second]},
        ):
            with self.subTest(invalid=invalid):
                self.assertFalse(closed_fusion.validate_expected(invalid))


if __name__ == "__main__":
    unittest.main()
