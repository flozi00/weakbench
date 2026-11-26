"""Public artificial networks for the flow/cut certificate verifier."""

import json
from itertools import product
import unittest
import closed_flow


class FlowTests(unittest.TestCase):
    def setUp(self):
        self.network = {
            "nodes": ["s", "a", "b", "x", "m", "t"],
            "source": "s",
            "sink": "t",
            "arcs": {
                "a": ["s", "x", 3],
                "b": ["x", "a", 4],
                "c": ["x", "b", 4],
                "d": ["a", "m", 4],
                "e": ["b", "m", 4],
                "f": ["m", "t", 3],
            },
        }
        self.base = {
            "value": 3,
            "flow": {"a": 3, "b": 3, "c": 0, "d": 3, "e": 0, "f": 3},
            "source_side": ["s"],
        }

    def grade(self, doc):
        return closed_flow.score({"expected": self.network}, {"content": json.dumps(doc)})

    def test_alternative_flows_and_cuts(self):
        for left in range(4):
            for side in (["s"], ["m", "x", "b", "a", "s"]):
                result = self.grade(
                    {
                        "value": 3,
                        "flow": {"a": 3, "b": left, "c": 3 - left, "d": left, "e": 3 - left, "f": 3},
                        "source_side": side,
                    }
                )
                self.assertTrue(result["correct"])
                self.assertEqual(set(result["detail"]), closed_flow.DETAIL_KEYS)

    def test_bad_certificates(self):
        for field, value in (
            ("value", True),
            ("value", 3.0),
            ("value", 2.9999999999999),
            ("source_side", ["t", "s"]),
            ("source_side", ["s", "extra"]),
            ("source_side", ["s", "s"]),
            ("source_side", ["s", []]),
            ("source_side", ["s", False]),
            ("source_side", ["s", 4]),
            ("source_side", ["s", "x"]),
            ("flow", {**self.base["flow"], "a": True}),
            ("flow", {**self.base["flow"], "a": 3.00000001}),
            ("flow", {**self.base["flow"], "b": 2}),
            ("flow", {**self.base["flow"], "c": -1}),
            ("flow", {}),
        ):
            self.assertFalse(self.grade({**self.base, field: value})["correct"])
        too_small = {**self.base, "value": 2, "flow": {k: min(v, 2) for k, v in self.base["flow"].items()}}
        check = self.grade(too_small)
        self.assertTrue(check["detail"]["flow_conservation"])
        self.assertFalse(check["detail"]["flow_cut_equal"])
        raw = json.dumps(self.base).replace('"value": 3', '"value": 1, "value": 3')
        self.assertFalse(closed_flow.score({"expected": self.network}, {"content": raw})["correct"])

    def test_directed_cut(self):
        self.network["arcs"]["return"] = ["t", "s", 13]
        self.network["arcs"]["bridge"] = ["x", "m", 5]
        self.assertTrue(
            self.grade({**self.base, "flow": {**self.base["flow"], "return": 0, "bridge": 0}})["correct"]
        )

    def test_isolated_vertices_and_zero_flow(self):
        self.network["nodes"].append("island")
        self.assertTrue(closed_flow.validate_expected(self.network))
        for side in (["s"], ["island", "s"]):
            self.assertTrue(self.grade({**self.base, "source_side": side})["correct"])
        self.network = {"nodes": ["s", "t", "island"], "source": "s", "sink": "t",
                        "arcs": {"reverse": ["t", "s", 8]}}
        for side in (["s"], ["island", "s"]):
            self.assertTrue(self.grade({"value": 0, "flow": {"reverse": 0}, "source_side": side})["correct"])

    def test_exhaustive_public_network(self):
        # Enumerate every candidate edge vector, including reverse and parallel edges,
        # and find the optimum by direct optimization before trying either possible cut.
        edges = {"first": ["s", "a", 1], "second": ["s", "a", 2], "out": ["a", "t", 2],
                 "return": ["a", "s", 1], "back": ["t", "a", 1]}
        self.network = {"nodes": ["s", "a", "t"], "source": "s", "sink": "t", "arcs": edges}
        assignments = list(product(*(range(cap + 1) for _, _, cap in edges.values())))
        feasible = {v: v[2] - v[4] for v in assignments if v[0] + v[1] + v[4] == v[2] + v[3]}
        optimum = max(feasible.values())
        for vector in assignments:
            for side, cut_value in ((["s"], 3), (["s", "a"], 2)):
                for claimed in range(4):
                    result = self.grade({"value": claimed, "flow": dict(zip(edges, vector)), "source_side": side})
                    self.assertEqual(result["correct"], feasible.get(vector) == optimum == cut_value == claimed)

    def test_schema(self):
        self.assertTrue(closed_flow.validate_expected(self.network))
        for change in (
            {"nodes": ["s", []]},
            {"source": 1},
            {"arcs": {"k": [["s"], "t", 2]}},
            {"sink": "s"},
            {"arcs": {"k": ["s", "t", False]}},
            {"arcs": {"k": ["s", "s", 2]}},
        ):
            self.assertFalse(closed_flow.validate_expected({**self.network, **change}))


if __name__ == "__main__":
    unittest.main()
