import json
import unittest
from closed_halfplanes import score, validate_expected


def fixture():
    return {"rows": [[-2, 0, 0], [0, -1, 0], [1, 1, 1]], "claim": [1, 2, 2]}


def witness():
    return {"status": "entailed", "point": [0, 1], "weights": ["1/2", 0, 2]}


class CertificateTests(unittest.TestCase):
    def test_equivalent_proofs_and_alternatives(self):
        for point in ([0, 1], ["1/3", "2/3"]):
            answer = dict(witness(), point=point, weights=["2/4", 0, 2])
            self.assertTrue(self.check(fixture(), answer))
        self.assertTrue(
            self.check(
                dict(fixture(), claim=[1, 2, 1]), {"status": "refuted", "point": [0, 1]}
            )
        )
        self.assertTrue(
            self.check(
                {"rows": [[1, 0, 0], [-3, 0, -1]], "claim": [1, 2, 1]},
                {"status": "inconsistent", "weights": [3, 1]},
            )
        )

    def check(self, spec, answer):
        self.assertTrue(validate_expected(spec))
        return score({"expected": spec}, {"content": json.dumps(answer)})["correct"]

    def test_equalities_and_false_certificates(self):
        for answer in (
            dict(witness(), point=[-1, 1]),
            dict(witness(), weights=[-1, 0, 2]),
            dict(witness(), weights=["1/0", 0, 2]),
            dict(witness(), weights=[True, 0, 2]),
            {"status": "refuted", "point": [0, 1]},
            {"status": "inconsistent", "weights": [0, 0, 0]},
            {"status": "entailed", "weights": ["1/2", 0, 2]},
        ):
            self.assertFalse(self.check(fixture(), answer))
        self.assertFalse(validate_expected(dict(fixture(), claim=[0, 0, 1])))


if __name__ == "__main__":
    unittest.main()
