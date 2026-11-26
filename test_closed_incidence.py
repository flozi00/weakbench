"""Public synthetic regression fixtures for exact competing failure probabilities."""

from fractions import Fraction
import json
import unittest

import closed_incidence as c


def fixture():
    return {
        "fleet": [
            ["r", 0, 2, "jam"],
            ["s", 0, 2, "signal"],
            ["u", 0, 2, "censor"],
            ["v", 0, 3, "jam"],
            ["w", 2, 4, "censor"],
        ],
        "times": [2, 3],
    }


def accepted():
    return {
        "risk": [4, 2],
        "operating": ["1/2", "1/4"],
        "jam": ["1/4", "1/2"],
        "signal": ["1/4", "1/4"],
    }


def mark(answer):
    return c.score({"expected": fixture()}, {"content": json.dumps(answer)})


class IncidenceTests(unittest.TestCase):
    def test_all_boundaries_and_absorbing_competing_failures(self):
        self.assertTrue(mark(accepted())["correct"])
        p = c.estimate(fixture()["fleet"], [2, 3])
        self.assertEqual(p["risk"], [4, 2])
        self.assertEqual(p["operating"], [Fraction(1, 2), Fraction(1, 4)])

    def test_rational_equivalence(self):
        answers = accepted()
        answers["jam"] = ["02/08", "50/100"]
        self.assertTrue(mark(answers)["correct"])
        for value in (0, "0", "000/7"):
            self.assertEqual(c.probability(value), 0)
        for value in (1, "1", "421/421"):
            self.assertEqual(c.probability(value), 1)

    def test_reject_wrong_or_ambiguous_numbers(self):
        for invalid in (
            True,
            0.5,
            2,
            -1,
            "0.25",
            "0/0",
            "1/0",
            "5/4",
            "-1/4",
            None,
            [1, 4],
            {"a": 1},
        ):
            with self.subTest(invalid=invalid):
                answers = accepted()
                answers["jam"] = [invalid, "1/2"]
                result = mark(answers)
                self.assertFalse(result["correct"])
                self.assertEqual(result["detail"]["incidence_schema"], 0.0)
                self.assertEqual(result["detail"]["jam_incidence_correct"], 0.5)

    def test_mass_balance_cannot_replace_cause_specific_values(self):
        answers = accepted()
        answers["jam"], answers["signal"] = answers["signal"], answers["jam"]
        result = mark(answers)
        self.assertFalse(result["correct"])
        self.assertEqual(result["detail"]["incidence_mass_balanced"], 1)

    def test_censor_between_first_and_second_failure(self):
        records = [["a", 0, 1, "signal"], ["b", 0, 2, "censor"], ["c", 0, 3, "jam"]]
        p = c.estimate(records, [2, 3])
        self.assertEqual(p["operating"], [Fraction(2, 3), 0])
        self.assertEqual(p["jam"], [0, Fraction(2, 3)])
        self.assertEqual(p["signal"], [Fraction(1, 3)] * 2)

    def test_ensure_all_times_and_fields(self):
        answers = accepted()
        for body in ("null", "[]", '{"risk":[],"risk":[]}', "NaN"):
            self.assertFalse(
                c.score({"expected": fixture()}, {"content": body})["correct"]
            )
        for mutation in (
            {k: v for k, v in answers.items() if k != "operating"},
            {**answers, "leak": 0},
            {**answers, "risk": [True, 2]},
            {**answers, "risk": [4]},
            {**answers, "risk": ["4", 2]},
        ):
            self.assertFalse(mark(mutation)["correct"])

    def test_invalid_records(self):
        baseline = fixture()
        self.assertTrue(c.validate_expected(baseline))
        for data in (
            None,
            {},
            {**baseline, "times": [2, 2]},
            {**baseline, "times": [1, True]},
            {**baseline, "times": [1, 8]},
            {**baseline, "fleet": []},
            {**baseline, "fleet": [["x", 0, 2, "censor"], ["x", 0, 3, "jam"]]},
            {**baseline, "fleet": [["x", 1, 3, "censor"]]},
            *[
                {**baseline, "fleet": [r]}
                for r in (
                    ["x", 1, 1, "jam"],
                    ["x", 0, 4, "other"],
                    ["x", False, 4, "jam"],
                    ["x", 0, 4.0, "jam"],
                    ["bad id", 0, 4, "jam"],
                    ["x", 0, 4, []],
                    ["x", 0, 4],
                )
            ],
        ):
            with self.subTest(fields=type(data).__name__):
                self.assertFalse(c.validate_expected(data))


if __name__ == "__main__":
    unittest.main()
