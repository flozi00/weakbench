import copy
import unittest

import closed_reference as reference


def fixture(lang):
    return {
        "words": ["warm", "leise", "weich"] if lang == "de" else ["warm", "quiet", "soft"],
        "views": [
            {"target": 1, "cards": [[0, 2], [0, 1, 2]]},
            {"target": 0, "cards": [[0, 1, 2], [1, 2]]},
        ],
    }


class ReferringExpressionTests(unittest.TestCase):
    def test_each_listener_and_both_orders(self):
        for lang, first, both, reverse, longer in (
            ("de", "Die Karte ist leise.", "Die Karte ist warm und leise.", "Die Karte ist leise und warm.",
             "Die Karte ist leise und warm und weich."),
            ("en", "The card is quiet.", "The card is warm and quiet.", "The card is quiet and warm.",
             "The card is quiet and warm and soft."),
        ):
            item = {"expected": fixture(lang), "lang": lang}
            def grade(s):
                return reference.score(item, {"content": s})
            self.assertTrue(reference.validate_expected(item["expected"]))
            self.assertEqual(reference.optimum(item["expected"]["views"]), 2)
            self.assertTrue(grade(both)["correct"])
            self.assertTrue(grade(reverse)["correct"])
            self.assertFalse(grade(first)["correct"])
            self.assertEqual(grade(first)["detail"]["first_listener_unique"], 1)
            self.assertEqual(grade(first)["detail"]["second_listener_unique"], 0)
            self.assertFalse(grade(longer)["correct"])
            self.assertEqual(grade(longer)["detail"]["both_listeners_unique"], 1)
            self.assertFalse(grade("No shared description.")["correct"])

    def test_impossible_in_joint_view_and_language_form(self):
        # Each recipient could identify the target separately using a different attribute.
        spec = fixture("en")
        spec["views"] = [{"target": 0, "cards": [[0, 2], [2]]}, {"target": 0, "cards": [[1, 2], [2]]}]
        item = {"expected": spec, "lang": "en"}
        self.assertTrue(reference.score(item, {"content": "NO SHARED DESCRIPTION."})["correct"])
        for s in ("Keine gemeinsame Beschreibung.", "The card is quiet.", "The card is soft and soft.",
                  "The card is soft. Other words.", "", [], None):
            self.assertFalse(reference.score(item, {"content": s})["correct"])
        self.assertNotIn("both_listeners_unique", reference.score(item, {"content": ""})["detail"])

    def test_equal_minima_are_accepted(self):
        spec = fixture("en")
        spec["views"][0]["cards"][0] = []
        spec["views"][1]["cards"][1] = []
        for word in spec["words"]:
            self.assertTrue(reference.score({"expected": spec, "lang": "en"},
                                            {"content": "The card is " + word})["correct"])

    def test_schema_bounds(self):
        for invalid in (None, False, -1, 4):
            spec = fixture("en")
            spec["views"][0]["target"] = invalid
            self.assertFalse(reference.validate_expected(spec))
        spec = fixture("en")
        for bad in (["duplicate", "duplicate"], ["bad word", "fine"], [None, "word"]):
            self.assertFalse(reference.validate_expected({**spec, "words": bad}))
        copy_spec = copy.deepcopy(spec)
        copy_spec["views"][0]["cards"][0] = [True]
        self.assertFalse(reference.validate_expected(copy_spec))


if __name__ == "__main__":
    unittest.main()
