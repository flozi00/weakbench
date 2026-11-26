import copy
import json
import unittest

import closed_commitments as scorer


def fixture(weights=None):
    # Synthetic example in which a group forced out of x can displace one or two others.
    original = [[0, 1], [1, 5], [2, 3], [3, 5], [4]]
    updated = [[1, 2], *original[1:]]
    final = [*updated[:4], [0, 4]]
    return {
        "parties": ["aa", "bb", "cc", "dd", "ee"],
        "slots": ["x", "y", "z", "u", "v", "w"],
        "weights": weights or [4, 2, 1, 1, 5],
        "stages": [original, updated, final],
    }


def text(expected, positions):
    return json.dumps(
        {p: expected["slots"][s] for p, s in zip(expected["parties"], positions)}
    )


def grade(expected, *path):
    turns = [x if isinstance(x, str) else text(expected, x) for x in path]
    return scorer.score(
        {"expected": expected}, {"content": turns[-1] if turns else "", "turns": turns}
    )


class CommitmentTests(unittest.TestCase):
    start = (0, 1, 2, 3, 4)
    short = (1, 5, 2, 3, 4)
    long = (2, 1, 3, 5, 4)

    def test_secondary_objective_and_full(self):
        e = fixture()
        self.assertTrue(scorer.validate_expected(e))
        self.assertTrue(grade(e, self.start, self.short, self.short)["correct"])
        losing = grade(e, self.start, self.long, self.long)
        self.assertEqual(losing["detail"]["repair1_feasible"], 1.0)
        self.assertEqual(losing["detail"]["repair1_optimal"], 0.0)
        self.assertEqual(losing["detail"]["repair2_optimal"], 1.0)
        self.assertEqual(losing["detail"]["full"], 0.0)

    def test_weight_precedes_number_moved(self):
        e = fixture([4, 9, 1, 1, 5])
        self.assertTrue(grade(e, self.start, self.long, self.long)["correct"])
        self.assertFalse(grade(e, self.start, self.short, self.short)["correct"])

    def test_own_history_and_all_tied_alternatives(self):
        e = fixture()
        base = [[0, 1], [0, 1], [2], [3], [4]]
        middle = [*base[:2], [2, 5], *base[3:]]
        e["stages"] = [base, middle, [*middle[:3], [0, 3], middle[4]]]
        swapped = (1, 0, 2, 3, 4)
        self.assertTrue(grade(e, self.start, self.start, self.start)["correct"])
        self.assertTrue(grade(e, swapped, swapped, swapped)["correct"])
        self.assertFalse(grade(e, swapped, swapped, self.start)["correct"])
        self.assertFalse(grade(e, swapped, self.start, self.start)["correct"])
        e = fixture([4, 1, 1, 1, 5])
        for stage in e["stages"]:
            stage[2] = [2, 5]
        alternative = (2, 1, 5, 3, 4)
        self.assertTrue(grade(e, self.start, self.short, self.short)["correct"])
        self.assertTrue(grade(e, self.start, alternative, alternative)["correct"])

    def test_invalid_turns_do_not_use_last_content_as_history(self):
        e = fixture()
        for reply in (
            {"content": text(e, self.short)},
            {"turns": []},
            {"turns": ["{}"] * 4},
        ):
            self.assertFalse(scorer.score({"expected": e}, reply)["correct"])
        wrong = grade(e, "{}", self.short, self.short)
        self.assertEqual(wrong["detail"]["repair1_feasible"], 1.0)
        self.assertEqual(wrong["detail"]["repair1_optimal"], 0.0)
        self.assertEqual(wrong["detail"]["full"], 0.0)
        for bad in ('{"aa":"x","aa":"x"}', "[1]", '{"aa":true}', "other text"):
            self.assertFalse(grade(e, self.start, bad, self.short)["correct"])
        collision = grade(e, self.start, (1, 1, 2, 3, 4), self.short)
        self.assertFalse(collision["detail"]["repair1_feasible"])
        self.assertFalse(collision["detail"]["repair2_optimal"])

    def test_schema(self):
        e = fixture()
        for field, value in (
            ("slots", ["x"] * 6),
            ("weights", [1, 2, 3, True, 5]),
            ("parties", ["aa"] * 5),
            ("stages", []),
        ):
            self.assertFalse(scorer.validate_expected({**e, field: value}))
        for values in ([True], [8], [1, 1], [], [{}]):
            clone = copy.deepcopy(e)
            clone["stages"][1][0] = values
            self.assertFalse(scorer.validate_expected(clone))
        clone = copy.deepcopy(e)
        clone["stages"][1] = [[1]] * 5
        self.assertFalse(scorer.validate_expected(clone))


if __name__ == "__main__":
    unittest.main()
