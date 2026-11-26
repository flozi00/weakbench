"""Small arbitrary public score with no overlap with private exercise designs."""

import json
import unittest
import closed_counterpoint as music


def fixture():
    return {
        "bass": [68, 67, 70, 68],
        "draft": [75, 77, 78, 80],
        "allowed": [[75, 80], [75, 79], [77, 78, 79], [80, 83]],
    }


def witness():
    return [75, 79, 78, 80]


class CounterpointTests(unittest.TestCase):
    def grade(self, spec, answer):
        return music.score({"expected": spec}, {"content": json.dumps(answer)})

    def test_tied_and_secondary_objectives(self):
        s = fixture()
        self.assertTrue(music.validate_expected(s))
        for p in (75, 79):
            self.assertTrue(self.grade(s, [75, p, 78, 80])["correct"])
        s["allowed"][1].append(76)
        result = self.grade(s, witness())
        self.assertEqual(result["detail"]["counterpoint_fewest_edits"], 1)
        self.assertEqual(result["detail"]["counterpoint_least_displacement"], 0)
        self.assertTrue(self.grade(s, [75, 76, 78, 80])["correct"])

    def test_perfect_families_and_motion_direction(self):
        # Equal fifths rising by one are forbidden; an oblique perfect arrival is fine.
        self.assertFalse(music.legal_transition([60, 61, 60, 59], 1, None, 67, 68)[1])
        self.assertTrue(music.legal_transition([60, 61, 60, 59], 1, None, 68, 68)[1])
        # A minuscule bass shift into an octave may coexist with a stationary upper note.
        self.assertTrue(music.legal_transition([60, 61, 60, 59], 1, None, 73, 73)[1])
        self.assertTrue(music.legal_transition([60, 62, 60, 59], 1, None, 76, 74)[1])
        self.assertFalse(music.legal_transition([60, 62, 60, 59], 1, None, 71, 74)[1])

    def test_leap_recovery_and_cadence(self):
        b = [55, 57, 58, 56]
        self.assertFalse(music.legal_transition(b, 2, 73, 78, 78)[0])
        self.assertTrue(music.legal_transition(b, 2, 73, 78, 76)[0])
        self.assertTrue(music.legal_transition(b, 3, 65, 67, 68)[2])
        self.assertFalse(music.legal_transition(b, 3, 65, 69, 68)[2])
        self.assertFalse(music.harmonic(b, 3, 63))

    def test_invalid_syntax_and_release(self):
        s = fixture()
        self.assertFalse(self.grade(s, [True, 79, 78, 80])["correct"])
        self.assertFalse(self.grade(s, {"upper": witness()})["correct"])
        self.assertFalse(
            self.grade(s, [75, 76, 78, 80])["detail"]["counterpoint_palette"]
        )
        s["allowed"][-1] = [70]
        self.assertFalse(music.validate_expected(s))


if __name__ == "__main__":
    unittest.main()
