"""Public artificial machines for exact streaming-equivalence checks."""

import copy
import json
from itertools import product
import unittest

import closed_transducer as checker


class TransducerTests(unittest.TestCase):
    def setUp(self):
        # Emit a marker on every fourth a. b resets the cycle.
        self.inputs = ["a", "b"]
        self.reference = {
            "start": "0",
            "states": {
                str(i): {"a": [str((i + 1) % 4), "yes" if i == 3 else "no"], "b": ["0", "no"]}
                for i in range(4)
            },
        }
        self.expected = {
            "inputs": self.inputs,
            "outputs": ["yes", "no"],
            "max_states": 5,
            "reference": self.reference,
        }

    def grade(self, machine):
        return checker.score({"expected": self.expected}, {"content": json.dumps(machine)})

    def test_equivalent_relabelled_and_split_state(self):
        names = {str(i): s for i, s in enumerate(["start here", "ß", "x/y", "done!"])}
        states = {
            names[key]: {token: [names[nxt], mark] for token, (nxt, mark) in row.items()}
            for key, row in self.reference["states"].items()
        }
        # Both copies of the initial state can be reached, and are observationally equivalent.
        states["spare"] = copy.deepcopy(states[names["0"]])
        states[names["2"]]["b"][0] = "spare"
        result = self.grade({"start": names["0"], "states": states})
        self.assertTrue(result["correct"])
        self.assertEqual(set(result["detail"]), checker.DETAIL_KEYS)

    def test_deep_error_and_unreachable(self):
        wrong = copy.deepcopy(self.reference)
        wrong["states"]["3"]["a"][1] = "no"
        self.assertFalse(self.grade(wrong)["correct"])
        clone = copy.deepcopy(self.reference)
        clone["states"]["off"] = {"a": ["off", "yes"], "b": ["off", "yes"]}
        self.assertTrue(self.grade(clone)["correct"])
        del clone["states"]["off"]["a"]
        self.assertFalse(self.grade(clone)["detail"]["transducer_total"])

    def test_invalid_tables_and_bounds(self):
        for replacement in ([False, "no"], ["missing", "no"], ["0", "other"], ["0", "yes", 1], {"q": "0"}):
            obj = copy.deepcopy(self.reference)
            obj["states"]["0"]["a"] = replacement
            self.assertFalse(self.grade(obj)["correct"])
        for machine in (
            [],
            {},
            {"start": "0", "states": {}},
            {"start": "other", "states": self.reference["states"]},
        ):
            self.assertFalse(self.grade(machine)["correct"])
        excessive = copy.deepcopy(self.reference)
        excessive["states"].update({k: self.reference["states"]["0"] for k in ("a", "b")})
        self.assertFalse(self.grade(excessive)["detail"]["transducer_within_budget"])
        double = json.dumps(self.reference).replace('"start": "0"', '"start": "3", "start": "0"')
        self.assertFalse(
            checker.score({"expected": self.expected}, {"content": double})["detail"]["valid_json"]
        )

    def test_every_two_state_unary_pair(self):
        # Four steps cover the four pairs in the product of two two-state machines.
        # Exhaust all 16 unary machines and compare their literal four-step outputs.
        machines = []
        for first, second in product(product(("p", "q"), ("0", "1")), repeat=2):
            m = {"start": "p", "states": {"p": {"a": list(first)}, "q": {"a": list(second)}}}
            state = "p"
            trace = []
            for _ in range(4):
                state, out = m["states"][state]["a"]
                trace.append(out)
            machines.append((m, trace))
        for (left, lt), (right, rt) in product(machines, repeat=2):
            self.assertEqual(checker.equivalent(left, right, ["a"]), lt == rt)

    def test_expected_schema(self):
        self.assertTrue(checker.validate_expected(self.expected))
        for update in (
            {"max_states": True},
            {"max_states": 0},
            {"inputs": ["a", "a"]},
            {"outputs": [[]]},
            {"max_states": 3},
        ):
            self.assertFalse(checker.validate_expected({**self.expected, **update}))


if __name__ == "__main__":
    unittest.main()
