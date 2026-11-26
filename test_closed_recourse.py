"""Independent small calibration fixture with ties, an unreachable target and a free solution."""

import json
import unittest
from copy import deepcopy
import closed_recourse as verifier


def fixture():
    return {
        "environment": ["U", "V"],
        "modules": [
            ["X", "U", "V", ["0011", "0111"]],
            ["Y", "V", "U", ["0011", "0111"]],
            ["Z", "X", "Y", ["0001", "0111"]],
        ],
        "calibration": [["01", {}, "011"], ["10", {}, "101"], ["00", {"Y": 1}, "011"]],
        "settings": ["00", "01", "10", "11"],
        "controls": [["X", 2, 1], ["Y", 2, 1]],
        "goal": {"Z": 1},
    }


def witness():
    return {
        "mechanisms": {"X": "0011", "Y": "0011", "Z": "0111"},
        "cost": 1,
        "pins": {"Y": 1},
    }


def result(spec, ans):
    return verifier.score({"expected": spec}, {"content": json.dumps(ans)})


class RecourseTests(unittest.TestCase):
    def test_override_only_propagates_along_an_input_edge(self):
        spec = fixture()
        # Under U=0 and V=0, fixing X=1 affects Z (a child) but not Y (a later sibling).
        spec["settings"] = ["00"]
        spec["goal"] = {"X": 1, "Y": 0, "Z": 1}
        self.assertEqual(verifier.coverage(spec, witness()["mechanisms"], {"X": 1}), 1)
        self.assertEqual(verifier.coverage(spec, witness()["mechanisms"], {}), 0)
        spec["goal"]["Y"] = 1
        self.assertEqual(verifier.coverage(spec, witness()["mechanisms"], {"X": 1}), 0)
        self.assertEqual(
            verifier.coverage(spec, witness()["mechanisms"], {"X": 1, "Y": 1}), 1
        )

    def test_robust_alternatives(self):
        spec = fixture()
        self.assertTrue(verifier.validate_expected(spec))
        for label in ("X", "Y"):
            self.assertTrue(result(spec, dict(witness(), pins={label: 1}))["correct"])
        none = result(spec, dict(witness(), pins={}, cost=0))
        self.assertFalse(none["correct"])
        self.assertEqual(none["detail"]["recourse_coverage"], 0.75)
        expensive = result(spec, dict(witness(), pins={"Y": 1, "X": 1}, cost=2))
        self.assertEqual(expensive["detail"]["recourse_coverage"], 1)
        self.assertEqual(expensive["detail"]["recourse_optimal"], 0)

    def test_zero_is_not_impossible(self):
        spec = dict(fixture(), settings=["01", "10"])
        self.assertTrue(result(spec, dict(witness(), cost=0, pins={}))["correct"])
        self.assertFalse(result(spec, dict(witness(), cost=None, pins=None))["correct"])
        bad = dict(fixture(), goal={"Z": 0}, controls=[["X", None, 1], ["Y", None, 1]])
        impossible = dict(witness(), cost=None, pins=None)
        r = result(bad, impossible)
        self.assertTrue(r["correct"])
        self.assertNotIn("recourse_optimal", r["detail"])
        self.assertFalse(result(bad, dict(witness(), cost=0, pins={}))["correct"])
        self.assertFalse(result(spec, dict(witness(), cost=None, pins={}))["correct"])

    def test_malformed_answers_and_unidentified_modules(self):
        spec = fixture()
        for ans in (
            dict(witness(), cost=True),
            dict(witness(), cost=7),
            dict(witness(), pins={"Y": True}),
            dict(witness(), pins={"Z": 1}),
            dict(witness(), mechanisms={"X": "0011"}),
            dict(witness(), unknown=0),
        ):
            self.assertFalse(result(spec, ans)["correct"])
        unknown = deepcopy(spec)
        unknown["calibration"] = unknown["calibration"][:1]
        self.assertFalse(verifier.validate_expected(unknown))
        wrong = deepcopy(spec)
        wrong["calibration"][-1][1] = {"Y": 0}
        self.assertFalse(verifier.validate_expected(wrong))
        loop = deepcopy(spec)
        loop["modules"][0][1] = "Z"
        self.assertFalse(verifier.validate_expected(loop))
        for raw in ("null", "[]", '{"cost":1,"cost":0}'):
            self.assertFalse(
                verifier.score({"expected": spec}, {"content": raw})["correct"]
            )


if __name__ == "__main__":
    unittest.main()
