"""Invented compact public fixture, with no private room names, routes or stories."""

import unittest
import closed_backstage as check


def fixture(two=True):
    return {
        "cast": ["Joe", "May", "Max"],
        "start": "Mint",
        "final": "Snow",
        "endings": 2 if two else 1,
        "scenes": [
            {"moves": [["Mint", "Dune"]], "seeker": [], "helper": [], "card": ["Dune"]},
            {
                "moves": [["Dune", "Snow"]],
                "seeker": [],
                "helper": ["Snow"],
                "card": ["Snow"],
            },
        ],
    }


def witness(lang, two=True):
    fixture(two)

    def line(action, u, v="", variant=0):
        return check.FORMS[lang][action][variant].format(
            m="Joe", s="May", h="Max", u=u, v=v
        )

    start = "\n".join((line("move", "Mint", "Dune"), line("move", "Dune", "Snow", 1)))
    search = line("search", "Mint")
    find = line("find", "Snow")
    first = "\n".join(
        (search, line("note", "Mint", "Dune"), line("note", "Dune", "Snow", 1), find)
    )
    second = "\n".join((search, line("ask", "Mint", "Snow"), find))
    return "\n\n".join((start, first, second) if two else (start, second))


class MiniatureTests(unittest.TestCase):
    def check(self, text, spec=None, lang="en"):
        spec = fixture() if spec is None else spec
        return check.score({"expected": spec, "lang": lang}, {"content": text})

    def test_two_endings(self):
        for lang in ("en", "de"):
            self.assertTrue(self.check(witness(lang), lang=lang)["correct"])
            a, b, c = witness(lang).split("\n\n")
            self.assertTrue(self.check("\n\n".join((a, c, b)), lang=lang)["correct"])
            self.assertFalse(self.check("\n\n".join((a, b, b)), lang=lang)["correct"])
            self.assertFalse(
                self.check(witness(lang) + "\nInvented extra.", lang=lang)["correct"]
            )
        self.assertFalse(self.check(witness("de"))["correct"])

    def test_observation_and_card_are_causal(self):
        text = witness("en")
        for field, value in (("seeker", ["Dune"]), ("card", [])):
            s = fixture()
            s["scenes"][0][field] = value
            self.assertFalse(self.check(text, s)["correct"])
        s = fixture()
        s["scenes"][-1]["helper"] = []
        self.assertFalse(self.check(text, s)["correct"])

    def test_source_word_alone_cannot_create_an_alternative(self):
        s = fixture()
        s["scenes"][0]["helper"] = ["Dune"]
        s["scenes"][1]["helper"] = []
        opening, notes, _ = witness("en").split("\n\n")
        address = notes.replace(
            "The card in locker Mint sent May to locker Dune.",
            "At locker Mint, Max directed May to locker Dune.",
        )
        result = self.check("\n\n".join((opening, notes, address)), s)
        self.assertEqual(result["detail"]["story_evidence"], 1)
        self.assertEqual(result["detail"]["story_alternate_endings"], 0)

    def test_one_ending_and_forms(self):
        result = self.check(witness("de", False), fixture(False), "de")
        self.assertTrue(result["correct"])
        self.assertNotIn("story_alternate_endings", result["detail"])
        self.assertFalse(check.validate_expected(dict(fixture(), endings=True)))
        self.assertTrue(check.validate_expected(fixture()))


if __name__ == "__main__":
    unittest.main()
