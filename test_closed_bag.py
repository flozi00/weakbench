"""Public synthetic fixtures cover null handling, duplicate counts and executable-query boundaries."""

import base64
import json
import struct
import unittest

import closed_bag as verifier


def witness():
    return {
        "join": "inner",
        "on": [["eq", "L.g", "R.k"]],
        "where": [],
        "group": [],
        "measures": [["count", "*"]],
        "having": [],
    }


def fixture():
    # Room codes are known; every matching pair contributes even when lines are duplicated.
    index = bytearray()
    outputs = []
    for l, r in verifier.universe():
        answer = [
            [sum(1 for _, room in l for key, _ in r if key is not None and key == room)]
        ]
        if answer not in outputs:
            outputs.append(answer)
        index.extend(struct.pack(">H", outputs.index(answer)))
    return {"outputs": outputs, "index": base64.b64encode(index).decode()}


class BagTests(unittest.TestCase):
    def test_known_values_and_equivalent_queries(self):
        spec = fixture()
        self.assertTrue(verifier.validate_expected(spec))
        for q in (
            witness(),
            dict(
                witness(),
                on=[],
                where=[["eq", "L.g", "R.k"]],
                measures=[["count", "L.g"]],
            ),
        ):
            result = verifier.score({"expected": spec}, {"content": json.dumps(q)})
            self.assertTrue(result["correct"])
            self.assertEqual(set(result["detail"]), set(verifier.DETAIL_KEYS))
        wrong = dict(witness(), join="left")
        result = verifier.score({"expected": spec}, {"content": json.dumps(wrong)})
        self.assertFalse(result["correct"])
        self.assertGreater(result["detail"]["bag_database_accuracy"], 0)
        self.assertLess(result["detail"]["bag_unmatched_accuracy"], 1)

    def test_statements_and_shapes(self):
        base = witness()
        for field, bad in (
            ("on", [["eq", "L.g", True]]),
            ("where", [["not supplied"]]),
            ("group", ["R.k", "R.k"]),
            ("group", ["R.k", "L.g", "R.v"]),
            ("measures", [["count", "L.missing"]]),
            ("having", [["gt", 1, 0]]),
            ("join", "other"),
            ("on", [["known", ["L.g"]]]),
            ("having", [["eq", True, 2]]),
        ):
            self.assertFalse(verifier.valid_plan(dict(base, **{field: bad})))
        for raw in (
            "SELECT 1",
            "[null]",
            json.dumps(dict(base, other=0)),
            '{"join":"left","join":"inner"}',
            " " * 8200,
        ):
            self.assertFalse(
                verifier.score({"expected": fixture()}, {"content": raw})["correct"]
            )
        spec = fixture()
        self.assertFalse(
            verifier.validate_expected(dict(spec, index="?" * len(spec["index"])))
        )
        self.assertFalse(
            verifier.validate_expected(dict(spec, outputs=[*spec["outputs"], [[3]]]))
        )

    def test_always_empty_groups(self):
        spec = {
            "outputs": [[]],
            "index": base64.b64encode(bytes(2 * len(verifier.universe()))).decode(),
        }
        plan = dict(witness(), where=[["ne", 0, 0]], group=["L.g"])
        self.assertTrue(verifier.validate_expected(spec))
        self.assertTrue(
            verifier.score({"expected": spec}, {"content": json.dumps(plan)})["correct"]
        )

    def test_exact_join_nulls_and_aggregates(self):
        plan = dict(
            witness(),
            join="left",
            on=[["eq", "L.k", "R.k"]],
            group=["L.g"],
            measures=[["count", "R.k"], ["count", "R.v"], ["sum", "R.v"]],
        )
        left = ((None, 0), (0, 1))
        right = ((None, 1), (0, None))
        self.assertEqual(
            verifier.evaluate(plan, left, right), [[0, 0, 0, None], [1, 1, 0, None]]
        )
        nullable = dict(plan, on=[["same", "L.k", "R.k"]])
        self.assertEqual(
            verifier.evaluate(nullable, left, right), [[0, 0, 1, 1], [1, 1, 0, None]]
        )
        on = dict(plan, on=[*plan["on"], ["eq", "R.v", 1]])
        where = dict(plan, where=[["eq", "R.v", 1]])
        self.assertEqual(
            verifier.evaluate(on, left, right), [[0, 0, 0, None], [1, 0, 0, None]]
        )
        self.assertEqual(verifier.evaluate(where, left, right), [])
        totals = dict(
            plan, group=[], measures=[["sum0", "R.v"], ["sum", "R.v"], ["count", "*"]]
        )
        self.assertEqual(verifier.evaluate(totals, (), ()), [[0, None, 0]])
        ties = dict(
            plan, measures=[["count", "*"], ["distinct", "R.v"], ["max", "R.v"]]
        )
        self.assertEqual(
            verifier.evaluate(ties, ((0, 0), (0, 0)), ((0, 1), (0, 1))), [[0, 4, 1, 1]]
        )


if __name__ == "__main__":
    unittest.main()
