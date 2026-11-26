"""Synthetic public circuits for checking the bounded compare-exchange interpreter."""
from itertools import product
import json
import unittest

import closed_fabric


class FabricTests(unittest.TestCase):
    def setUp(self):
        self.spec = {"wires": 3,
                     "ticks": [[[0, 1], [1, 2]], [[0, 2], [1, 2], [0, 1]], [[0, 1], [1, 2]]],
                     "budget": 3, "targets": [[0, 1], [1, 2]]}
        self.bubble = [[[0, 1]], [[1, 2]], [[0, 1]]]
        self.other = [[[1, 2]], [[0, 1]], [[1, 2]]]

    def grade(self, program, spec=None):
        return closed_fabric.score({"expected": self.spec if spec is None else spec},
                                   {"content": json.dumps(program)})

    def test_multiple_programs_and_wrong_orientation(self):
        self.assertTrue(closed_fabric.validate_expected(self.spec))
        for program in (self.bubble, self.other):
            verdict = self.grade({"stages": program})
            self.assertTrue(verdict["correct"])
            self.assertEqual(set(verdict["detail"]), closed_fabric.DETAIL_KEYS)
            self.assertTrue(all(verdict["detail"].values()))
        bad = [[[0, 1]], [[1, 2]], [[1, 0]]]
        result = self.grade({"stages": bad})["detail"]
        self.assertEqual(result["fabric_wiring"], 1.0)
        self.assertLess(result["fabric_input_coverage"], 1.0)
        self.assertEqual(result["fabric_all_inputs"], 0.0)

    def test_one_missed_input(self):
        spec = {**self.spec, "targets": [[0, 1]]}
        verdict = self.grade({"stages": [[[0, 1]], [], []]}, spec)
        self.assertFalse(verdict["correct"])
        self.assertEqual(verdict["detail"]["fabric_input_coverage"], 7/8)

    def test_wiring_and_schedule_limits(self):
        spec = {"wires": 4, "ticks": [[[0, 1], [1, 2], [2, 3]]], "budget": 1, "targets": [[0, 1]]}
        for ops in ([[1, 0], [1, 2]], [[0, 2]]):
            detail = self.grade({"stages": [ops]}, spec)["detail"]
            self.assertEqual(detail["fabric_program"], 1)
            self.assertEqual(detail["fabric_wiring"], 0)
        detail = self.grade({"stages": [[[0, 1], [2, 3]]]}, spec)["detail"]
        self.assertEqual(detail["fabric_budget"], 0)
        self.assertEqual(detail["fabric_wiring"], 1)

    def test_grammar_and_data_limits(self):
        for program in (None, {"stages": False}, {"stages": []}, {"stages": self.bubble, "x": 1},
                        {"stages": [[[True, 2]], [], []]}, {"stages":[[[0, 0]], [], []]},
                        {"stages": [[[0, 3]], [], []]}, {"stages": [[[0, 1, 2]], [], []]}):
            self.assertFalse(self.grade(program)["detail"]["fabric_program"])
        for content in ('{"stages":[],"stages":[]}', '['*1100+']'*1100, ' ' * 16385):
            result = closed_fabric.score({"expected": self.spec}, {"content": content})
            self.assertFalse(result["correct"])
        for fields in ({"wires": True}, {"ticks": [[]]*13}, {"ticks": [[[1, 0]]]},
                       {"ticks": [[[0, 1], [0, 1]]]}, {"targets": [[1, 2], [2, 2]]},
                       {"targets": [[0, 1], [0, 2]]}, {"targets": []}, {"budget": 4}):
            self.assertFalse(closed_fabric.validate_expected({**self.spec, **fields}))

    def test_binary_packing_against_all_small_programs(self):
        gates = [[]] + [[[a, b]] for a in range(3) for b in range(3) if a != b]
        inputs = tuple(product((0, 1), repeat=3))
        for program in product(gates, repeat=3):
            outputs = []
            for bits in inputs:
                values = list(bits)
                for stage in program:
                    if stage:
                        a, b = stage[0]
                        values[a], values[b] = sorted([values[a], values[b]])
                outputs.append(values)
            for wire, rank in product(range(3), range(1, 4)):
                count = sum(out[wire] == sorted(bits)[rank-1] for bits, out in zip(inputs, outputs))
                self.assertEqual(closed_fabric.passing_inputs(3, program, [[wire, rank]]), count)


if __name__ == "__main__":
    unittest.main()
