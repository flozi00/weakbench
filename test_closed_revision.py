"""Public toy catalogue and hand-derived response examples. No release cases."""

from copy import deepcopy
import json
import unittest

import closed_revision as scorer


def fixture():
    return {
        "specimen": "Prop",
        "channels": [["Left", "Ix"], ["Right", "Ox"]],
        "cards": [
            ["Apron", 1, "00"],
            ["Brick", 0, "01"],
            ["Cedar", 0, "10"],
            ["Daisy", 1, "11"],
            ["Elm", 0, "20"],
        ],
        "turns": [
            {
                "demand": None,
                "records": [
                    ["Pin", "Prop", "Left", 2, 0, "Ix"],
                    ["Tab", "Prop", "Right", 3, 0, "Ox"],
                ],
            },
            {
                "demand": "yes",
                "records": [
                    ["Ink", "Prop", "Left", 5, 1, "Ix"],
                    ["Slip", "Prop", "Left", 9, 0, "Ox"],
                ],
            },
            {"demand": "no", "records": [["Tag", "Prop", "Right", 8, 1, "Ox"]]},
        ],
    }


def examples(lang="en"):
    k, m, n = scorer.FIELDS[lang]
    return [
        {k: scorer.WORDS[lang][v], m: [c], n: proof}
        for v, c, proof in (
            ("yes", "Apron", ["Tab", "Pin"]),
            ("no", "Cedar", ["Ink", "Tab"]),
            ("yes", "Daisy", ["Ink", "Tag"]),
        )
    ]


def grade(inputs, lang="en", expected=None):
    return scorer.score(
        {"expected": expected or fixture(), "lang": lang},
        {"turns": [json.dumps(a) if isinstance(a, dict) else a for a in inputs]},
    )


class RevisionTest(unittest.TestCase):
    def test_grounded_both_languages(self):
        for lang in ("de", "en"):
            r = grade(examples(lang), lang)
            self.assertEqual(
                (r["correct"], r["score"], r["detail"]["revision_all_turns_correct"]),
                (True, 1.0, 1.0),
            )
            self.assertNotIn("revision_t2_grounded_recovery", r["detail"])
            self.assertEqual(
                grade(examples(lang), "de" if lang == "en" else "en")["score"], 0.0
            )

    def test_late_recovery_survives_initial_error(self):
        a = examples()
        a[0]["verdict"] = "open"
        r = grade(a)
        self.assertEqual(
            (r["correct"], r["score"], r["detail"]["revision_t2_grounded_recovery"]),
            (True, 2 / 3, 1.0),
        )
        self.assertEqual(r["detail"]["revision_all_turns_correct"], 0.0)
        a[1] = None
        r = grade(a)
        self.assertEqual(
            (r["correct"], r["score"], r["detail"]["revision_t3_grounded_recovery"]),
            (True, 1 / 3, 1.0),
        )

    def test_bad_last_turn_keeps_earlier_credit(self):
        a = examples()
        a[2]["candidates"] = ["Cedar"]
        r = grade(a)
        self.assertFalse(r["correct"])
        self.assertEqual(r["score"], 2 / 3)
        self.assertEqual(r["detail"]["revision_t3_verdict"], 1.0)

    def test_copy_echo_and_hold_are_distinct(self):
        a = examples()
        a[1] = deepcopy(a[0])
        a[1]["evidence"].reverse()
        d = grade(a)["detail"]
        self.assertEqual(
            [
                d["revision_t2_" + s]
                for s in (
                    "stale_verdict",
                    "wrong_pressure_echo",
                    "copied_obsolete_report",
                    "inactive_citation",
                )
            ],
            [1.0] * 4,
        )
        self.assertEqual(d["revision_t2_invented_citation"], 0.0)
        a = examples()
        a[1]["verdict"] = "yes"
        d = grade(a)["detail"]
        self.assertEqual(d["revision_t2_copied_obsolete_report"], 0.0)
        self.assertEqual(d["revision_t2_wrong_pressure_echo"], 1.0)
        self.assertEqual(d["revision_t2_grounding"], 1.0)

    def test_citations_exposed_at_different_turns(self):
        a = examples()
        a[1]["evidence"] = ["Tag"]
        d = grade(a)["detail"]
        self.assertEqual(
            (d["revision_t2_invented_citation"], d["revision_t2_inactive_citation"]),
            (1.0, 0.0),
        )
        for ignored in ("Slip", "Pin"):
            a[1]["evidence"] = [ignored]
            d = grade(a)["detail"]
            self.assertEqual(
                (
                    d["revision_t2_invented_citation"],
                    d["revision_t2_inactive_citation"],
                ),
                (0.0, 1.0),
            )

    def test_unconfirmed_editions_do_not_displace_confirmed_reading(self):
        for change in ("wrong stamp", "other specimen", "older printed last"):
            spec = fixture()
            last = spec["turns"][2]["records"][0]
            if change == "wrong stamp":
                last[-1] = "Ix"
            elif change == "other specimen":
                last[1] = "Other"
            else:
                last[3] = 1
            a = examples()
            a[2] = a[1]
            r = grade(a, expected=spec)
            self.assertEqual(r["score"], 1.0)
            self.assertNotIn("revision_t3_copied_obsolete_report", r["detail"])

    def test_supersession_can_restore_excluded_design(self):
        spec = fixture()
        spec["turns"][2]["records"] = [["New", "Prop", "Left", 6, 0, "Ix"]]
        a = examples()
        a[2] = {"verdict": "yes", "candidates": ["Apron"], "evidence": ["New", "Tab"]}
        self.assertEqual(grade(a, expected=spec)["score"], 1.0)
        a[2] = a[0]
        self.assertFalse(grade(a, expected=spec)["correct"])

    def test_inclusion_minimal_certificates_can_have_different_sizes(self):
        spec = fixture()
        spec["channels"] = [["X", "Ax"], ["Y", "Ay"], ["Z", "Az"]]
        spec["cards"] = [
            ["A", 0, "000"],
            ["B", 1, "111"],
            ["C", 1, "110"],
            ["D", 0, "101"],
        ]
        for turn in spec["turns"]:
            turn["records"] = []
        spec["turns"][0]["records"] = [
            [c + "a", "Prop", c, 1, 0, "A" + c.lower()] for c in ("X", "Y", "Z")
        ]
        self.assertTrue(scorer.validate_expected(spec))
        for proof, good in (
            (["Xa"], True),
            (["Ya", "Za"], True),
            (["Xa", "Ya"], False),
            (["Ya"], False),
            ([], False),
        ):
            answer = {"verdict": "no", "candidates": ["A"], "evidence": proof}
            self.assertEqual(grade([answer] * 3, expected=spec)["correct"], good)

    def test_open_verdict_and_empty_certificate(self):
        spec = fixture()
        for turn in spec["turns"]:
            turn["records"] = []
        a = {
            "verdict": "open",
            "candidates": ["Elm", "Daisy", "Cedar", "Brick", "Apron"],
            "evidence": [],
        }
        self.assertEqual(grade([a] * 3, expected=spec)["score"], 1.0)
        a["verdict"] = "no"
        self.assertEqual(grade([a] * 3, expected=spec)["score"], 0.0)

    def test_malformed_answers_and_ambiguous_specs(self):
        for bad in (
            "text",
            '{"verdict":"no","verdict":"yes","candidates":["Daisy"],"evidence":["Ink","Tag"]}',
            dict(examples()[2], extra=1),
            dict(examples()[2], evidence=["Ink", "Ink", "Tag"]),
            dict(examples()[2], candidates=["Daisy", "Daisy"]),
            dict(examples()[2], evidence=[{}, 2]),
            {"verdict": "yes"},
        ):
            self.assertFalse(grade([*examples()[:2], bad])["correct"])
        spec = fixture()
        self.assertTrue(scorer.validate_expected(spec))
        for position, value in ((0, "Pin"), (3, 2), (4, True), (2, "Missing")):
            bad = deepcopy(spec)
            bad["turns"][1]["records"][0][position] = value
            self.assertFalse(scorer.validate_expected(bad))
        spec["turns"][2]["records"][0][4] = 2
        self.assertFalse(scorer.validate_expected(spec))


if __name__ == "__main__":
    unittest.main()
