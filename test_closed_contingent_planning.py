"""Synthetic controls for legal branching, impossibility and truthful worst-case costs."""

from copy import deepcopy
import json
import unittest

import closed_contingent_planning as planning


def fixture(lang=None):
    return {
        "worlds": ["one", "two", "three"],
        "limit": 6,
        "checks": [{"id": "look", "time": 2, "signals": ["x", "y", "y"]}],
        "services": [
            {"id": "a", "times": [4, None, None]},
            {"id": "b", "times": [None, 1, 3]},
            {"id": "c", "times": [8, 8, 8]},
        ],
    }


def witness(lang):
    de = lang == "de"
    leaf = "abfertigen" if de else "dispatch"
    inspect, branches = ("prüfen", "zweige") if de else ("inspect", "branches")
    return {
        "dauer" if de else "time": 6,
        "plan": {inspect: "look", branches: {"x": {leaf: "a"}, "y": {leaf: "b"}}},
    }


def grade(spec, ans, lang="en"):
    return planning.score(
        {"expected": spec, "lang": lang}, {"content": json.dumps(ans)}
    )


class Policies(unittest.TestCase):
    def test_two_languages_and_alternative_optimal_implementations(self):
        s = fixture()
        self.assertTrue(planning.validate_expected(s))
        self.assertEqual(planning.optimum(s), 6)
        for lang in ("de", "en"):
            self.assertTrue(grade(s, witness(lang), lang)["correct"])
        # A redundant inspection on a noncritical branch is globally optimal.
        s["checks"].append({"id": "repeat", "time": 1, "signals": ["m", "m", "m"]})
        ans = witness("en")
        ans["plan"]["branches"]["y"] = {
            "inspect": "repeat",
            "branches": {"m": {"dispatch": "b"}},
        }
        self.assertTrue(grade(s, ans)["correct"])

    def test_aliased_worlds_and_deadline(self):
        s = fixture()
        self.assertFalse(grade(s, {"plan": None, "time": None})["correct"])
        s["limit"] = 5
        self.assertTrue(grade(s, {"plan": None, "time": None})["correct"])
        self.assertEqual(
            grade(s, witness("en"))["detail"]["contingent_deadline_met"], 0
        )
        s["checks"][0]["signals"] = ["z", "z", "z"]
        s["services"].pop()
        self.assertIsNone(planning.optimum(s))
        self.assertTrue(grade(s, {"plan": None, "time": None})["correct"])

    def test_imagined_observation_and_missing_branch(self):
        s = fixture()
        extra = witness("en")
        extra["plan"]["branches"]["three"] = {"dispatch": "b"}
        self.assertFalse(grade(s, extra)["correct"])
        self.assertEqual(
            grade(s, extra)["detail"]["contingent_observation_branches"], 0
        )
        missing = witness("en")
        del missing["plan"]["branches"]["x"]
        result = grade(s, missing)
        self.assertEqual(result["detail"]["contingent_world_coverage"], 2 / 3)
        self.assertEqual(result["detail"]["contingent_all_worlds"], 0)
        repeated = witness("en")
        repeated["plan"]["branches"]["x"] = {
            "inspect": "look",
            "branches": {"x": {"dispatch": "a"}},
        }
        self.assertFalse(grade(s, repeated)["correct"])

    def test_legality_cost_optimum_and_feasibility_are_independent(self):
        s = fixture()
        s["limit"] = 9
        slow = grade(s, {"time": 8, "plan": {"dispatch": "c"}})
        self.assertEqual(slow["detail"]["contingent_deadline_met"], 1)
        self.assertEqual(slow["detail"]["contingent_optimal_policy"], 0)
        ans = witness("en")
        ans["time"] = 5
        under = grade(s, ans)
        self.assertFalse(under["correct"])
        self.assertEqual(under["detail"]["contingent_optimal_policy"], 1)
        self.assertEqual(under["detail"]["contingent_cost_report_correct"], 0)
        ans["time"] = True
        self.assertFalse(grade(s, ans)["correct"])
        wrong = grade(s, {"time": 4, "plan": {"dispatch": "a"}})
        self.assertEqual(wrong["detail"]["contingent_world_coverage"], 1 / 3)

    def test_reject_ambiguous_or_invalid_data(self):
        s = fixture()
        for key, value in (
            ("limit", True),
            ("worlds", ["one"] * 3),
            ("checks", []),
            ("services", [{}]),
        ):
            self.assertFalse(planning.validate_expected({**s, key: value}))
        for key, val in (("time", 0), ("signals", [None, "a", "b"]), ("id", "a")):
            t = deepcopy(s)
            t["checks"][0][key] = val
            self.assertFalse(planning.validate_expected(t))
        r = planning.score(
            {"expected": s, "lang": "en"},
            {"content": '{"time":null,"time":6,"plan":null}'},
        )
        self.assertFalse(r["correct"])


if __name__ == "__main__":
    unittest.main()
