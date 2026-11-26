"""Synthetic point masks with two disjoint regions, including half-open edges."""
import json
import unittest

import closed_grounding as grounding


class GroundingTests(unittest.TestCase):
    spec = {"size": [20, 10], "regions": [[2, 2, 8, 5], [10, 2, 18, 5]]}

    def check(self, text, target=None):
        return grounding.score({"expected": self.spec if target is None else target}, {"content": text})

    def test_every_pixel_including_edges(self):
        self.assertTrue(grounding.validate_expected(self.spec))
        for y in range(10):
            for x in range(20):
                v = self.check(json.dumps({"point": [x, y]}))
                inside = y in range(2, 5) and x in (*range(2,8), *range(10,18))
                self.assertEqual(v["correct"], inside)
                self.assertEqual(v["detail"]["actionable_point_hit"], float(inside))
                self.assertEqual(v["detail"]["actionability_correct"], 1.0)

    def test_unavailable_and_malformed(self):
        empty = {"size": [20, 10], "regions": []}
        self.assertTrue(grounding.validate_expected(empty))
        v = self.check('{"point":null}', empty)
        self.assertTrue(v["correct"])
        self.assertNotIn("actionable_point_hit", v["detail"])
        self.assertFalse(self.check('{"point":null}')["correct"])
        self.assertFalse(self.check('{"point":[2,2]}', empty)["correct"])
        for wrong in ('{"point":[true,2]}','{"point":[3.0,2]}','{"point":[20,0]}',
                      '{"point":[2,-1]}','{"point":[2,3],"why":null}','{"point":null,"point":[2,2]}',
                      'null','outside','{"point":false}'):
            self.assertEqual(self.check(wrong)["detail"]["point_form_ok"], 0.0)
        for invalid in ({"size":[0,5],"regions":[]},{"size":[10,10],"regions":[[0,0,11,1]]},
                        {"size":[10,10],"regions":[[0,0,1,False]]}):
            self.assertFalse(grounding.validate_expected(invalid))


if __name__ == "__main__":
    unittest.main()
