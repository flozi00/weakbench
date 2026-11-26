"""Public toy calendar covering certificate acceptance and each scheduling constraint."""
from copy import deepcopy
import json
import unittest

import closed_schedule


def job(duration, resources, after=(), release=0, deadline=7):
    return dict(duration=duration, resources=list(resources), after=list(after), release=release, deadline=deadline)


class ScheduleTests(unittest.TestCase):
    def setUp(self):
        self.instance = {
            "tasks": {"a": job(1, ["x"]), "b": job(1, ["y"], ["a"]),
                      "c": job(1, ["x", "y"], ["b"], release=3), "d": job(1, ["z"], deadline=3)},
            "closures": {"x": [[2, 3]], "y": [[2, 3]], "z": []}, "optimum": 4,
        }
        self.good = dict(a=0, b=1, c=3, d=0)

    def grade(self, answer):
        return closed_schedule.score({"expected": self.instance}, {"content": answer})

    def test_alternative_optima_and_feasible_but_slow(self):
        for start in (0, 1, 2):
            self.assertTrue(self.grade(json.dumps({**self.good, "d": start}))["correct"])
        late = self.grade(json.dumps({**self.good, "c": 4}))
        self.assertFalse(late["correct"])
        self.assertEqual(late["detail"]["schedule_feasible"], 1.0)

    def test_malformed_and_infeasible_certificates(self):
        baseline = json.dumps(self.good)
        for answer in ('{}', '[]', 'bad', baseline[:-1] + ', "a":0}', baseline[:-1] + ', "extra":2}',
                       *(json.dumps({**self.good, "a": x}) for x in (True, 0.0, "0", -1)),
                       *(json.dumps({**self.good, **change}) for change in
                         ({"b": 0}, {"b": 2}, {"c": 1}, {"d": 3}, {"a": 1, "b": 3}))):
            self.assertFalse(self.grade(answer)["correct"], answer)
        # Only y is occupied here. A multi-resource job must check every resource.
        self.instance["tasks"]["b"]["after"] = []
        self.assertFalse(self.grade(json.dumps({**self.good, "b": 3}))["correct"])

    def test_reject_invalid_instance(self):
        self.assertTrue(closed_schedule.validate_expected(self.instance))
        for key, value in (("duration", True), ("release", 3.0), ("deadline", 0), ("after", ["unknown"]),
                           ("after", ["c"]), ("resources", ["missing"]), ("resources", ["x", "x"])):
            instance = deepcopy(self.instance)
            instance["tasks"]["a"][key] = value
            self.assertFalse(closed_schedule.validate_expected(instance), key)
        for key, value in (("closures", {"x": [[4, 1]]}), ("optimum", False), ("tasks", {})):
            instance = {**self.instance, key: value}
            self.assertFalse(closed_schedule.validate_expected(instance), key)


if __name__ == "__main__":
    unittest.main()
