import json
import unittest
import closed_probes as probes


def fixture():
    return {
        "alphabet": 2,
        "maxlen": 2,
        "base": ["ident"],
        "variants": [
            ["V21", ["first0"], ["drop1"]],
            ["V42", ["first1"], ["drop1"]],
            ["V87", ["pal"], ["rev"]],
        ],
    }


def witness():
    return {
        "tests": [{"input": [1], "expect": [1]}, {"input": [0], "expect": [0]}],
        "equivalent": ["V87"],
    }


class ProbeTests(unittest.TestCase):
    def grade(self, obj, spec=None):
        return probes.score(
            {"expected": spec or fixture()}, {"content": json.dumps(obj)}
        )

    def test_minimal_suite_has_genuine_independent_tests(self):
        self.assertTrue(probes.validate_expected(fixture()))
        self.assertTrue(self.grade(witness())["correct"])
        redundant = witness()
        redundant["tests"][0] = {"input": [1, 1], "expect": [1, 1]}
        result = self.grade(redundant)
        self.assertFalse(result["correct"])
        self.assertEqual(result["detail"]["probe_minimum_count"], 1)
        self.assertEqual(result["detail"]["probe_minimum_total_length"], 0)
        bad = witness()
        bad["tests"][0]["expect"] = []
        self.assertEqual(self.grade(bad)["detail"]["probe_distinguished"], 0.5)

    def test_real_equivalence_and_empty_set(self):
        s = fixture()
        s["variants"] = s["variants"][-1:]
        self.assertTrue(self.grade({"tests": [], "equivalent": ["V87"]}, s)["correct"])
        self.assertFalse(self.grade(dict(witness(), equivalent=[]))["correct"])

    def test_bad_values_and_ambiguous_duplicates(self):
        for tests in (
            [{"input": [True], "expect": [0]}],
            witness()["tests"] * 2,
            [{"input": [], "expect": 0}],
        ):
            self.assertFalse(self.grade(dict(witness(), tests=tests))["correct"])
        self.assertFalse(
            probes.validate_expected(
                dict(fixture(), variants=[["V02", ["always"], ["rev"]]])
            )
        )
        self.assertFalse(
            probes.score(
                {"expected": fixture()},
                {"content": '{"tests":[],"tests":[],"equivalent":[]}'},
            )["correct"]
        )


if __name__ == "__main__":
    unittest.main()
