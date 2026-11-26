"""Public synthetic council records. None are held-out questions."""

import copy
import json
import unittest

import closed_quorum as quorum


def fixture(lang):
    literals = (
        [
            "A sign reads “first” here.",
            "A sign reads “second” here.",
            "A sign reads “third” here.",
            "A sign reads “fourth” here.",
        ]
        if lang == "de"
        else [
            "Hier steht „erste Tafel“.",
            "Hier steht „zweite Tafel“.",
            "Hier steht „dritte Tafel“.",
            "Hier steht „vierte Tafel“.",
        ]
    )
    return {
        "ballots": [
            ["Ivo", "yes"],
            ["Ela", "Ivo"],
            ["Eda", "Eda"],
            ["Mio", "abstain"],
            ["Mia", "no"],
        ],
        "quorum": 3,
        "sources": [
            [kind, key, text]
            for kind, key, text in zip(
                quorum.OUTCOMES, ("Xx", "Xy", "Xz", "Xw"), literals
            )
        ],
    }


def sample_prompt(lang, spec):
    saying = (
        "Prüfe die Stimmen der Mitglieder und wähle für die Entscheidung den passenden Beleg."
        if lang == "de"
        else "Check the members' votes and choose the source that supports the result."
    )
    return (
        saying
        + "\n"
        + quorum.SECTIONS[lang]
        + "\n"
        + "\n".join(k + " | " + v for _, k, v in spec["sources"])
        + "\n\n"
        + saying
    )


def answer(lang, spec):
    _, key, quote = spec["sources"][0]
    return {
        "parts": [
            {"prose": quorum.tally_forms(lang, (2, 1, 2))[0]},
            {"source": key, "verbatim": quote},
            {"prose": quorum.DECISIONS[lang]["adopt"][0]},
        ]
    }


class AuthenticatedMinuteTests(unittest.TestCase):
    def grade(self, lang, wire, reasoning=None):
        spec = fixture(lang)
        item = {
            "expected": spec,
            "lang": lang,
            "messages": [{"role": "user", "content": sample_prompt(lang, spec)}],
        }
        return quorum.score(item, {"content": wire, "reasoning": reasoning})

    def test_delegation_cycles_quorum_and_ties(self):
        for lang in ("de", "en"):
            spec = fixture(lang)
            self.assertTrue(quorum.validate_expected(spec))
            self.assertEqual(quorum.tally(spec), ((2, 1, 2), "adopt"))
            self.assertEqual(quorum.tally(dict(spec, quorum=4)), ((2, 1, 2), "quorum"))
            spec["ballots"] = [
                ["Ivo", "Ela"],
                ["Ela", "Ivo"],
                ["Mia", "Ela"],
                ["Ina", "yes"],
                ["Udo", "no"],
            ]
            self.assertEqual(quorum.tally(dict(spec, quorum=2)), ((1, 1, 3), "tie"))
            spec["ballots"][0][1] = "no"
            self.assertEqual(quorum.tally(spec), ((1, 4, 0), "reject"))

    def test_content_and_all_observable_channels(self):
        phrases = {
            "en": "We count the votes before we compare the results with the threshold.",
            "de": "Wir zählen die Stimmen und vergleichen das Ergebnis mit dem geforderten Wert.",
        }
        for lang in phrases:
            wire = json.dumps(answer(lang, fixture(lang)), ensure_ascii=False)
            base = self.grade(lang, wire)
            self.assertTrue(base["correct"])
            self.assertEqual(base["detail"]["all_channels_verified"], 0)
            self.assertNotIn("reasoning_language_ok", base["detail"])
            self.assertEqual(
                self.grade(lang, wire, phrases[lang])["detail"][
                    "correct_with_all_channels"
                ],
                1,
            )
            for invalid in (phrases["de" if lang == "en" else "en"], "2 + 3 = 5"):
                bad = self.grade(lang, wire, invalid)
                self.assertFalse(bad["correct"])
                self.assertEqual(bad["detail"]["correct_with_all_channels"], 0)

    def test_source_authentication_and_scope(self):
        for lang in ("de", "en"):
            spec = fixture(lang)
            reference = answer(lang, spec)
            for mode in (
                "wrong_id",
                "short",
                "foreign_in_prose",
                "nested",
                "extra_part",
                "typographic",
            ):
                candidate = copy.deepcopy(reference)
                parts = candidate["parts"]
                if mode == "wrong_id":
                    parts[1]["source"] = spec["sources"][1][1]
                if mode == "short":
                    parts[1]["verbatim"] = parts[1]["verbatim"][:-1]
                if mode == "foreign_in_prose":
                    parts[2]["prose"] += " <q>" + parts[1]["verbatim"] + "</q>"
                if mode == "nested":
                    parts[1]["prose"] = parts[0]["prose"]
                if mode == "extra_part":
                    parts.append({"verbatim": parts[1]["verbatim"]})
                if mode == "typographic":
                    parts[1]["verbatim"] = (
                        parts[1]["verbatim"].replace("“", '"').replace("„", '"')
                    )
                with self.subTest(lang=lang, mode=mode):
                    self.assertFalse(self.grade(lang, json.dumps(candidate))["correct"])
            wire = json.dumps(reference)
            self.assertFalse(
                self.grade(lang, wire.replace('"source":', '"source":"Xz", "source":'))[
                    "correct"
                ]
            )
            prompt = sample_prompt(lang, spec)
            self.assertTrue(quorum.request_language_ok(prompt, lang, spec))
            for bad in (
                prompt.replace("Xy | ", "Xz | "),
                prompt.replace("Xy | ", "Xy | <q>"),
                prompt + prompt,
            ):
                self.assertFalse(quorum.request_language_ok(bad, lang, spec))

    def test_expected_validation_and_partial_counts(self):
        spec = fixture("en")
        for broken in (
            dict(spec, quorum=True),
            dict(spec, quorum=0),
            dict(spec, ballots=[["Ivo", "absent"], ["Ela", "no"]]),
            dict(spec, sources=spec["sources"][:-1]),
            dict(spec, ballots=[["Ivo", "yes"], ["Ivo", "no"]]),
        ):
            self.assertFalse(quorum.validate_expected(broken))
        candidate = answer("en", spec)
        candidate["parts"][0]["prose"] = quorum.tally_forms("en", (3, 1, 2))[1]
        marked = self.grade("en", json.dumps(candidate))
        self.assertFalse(marked["correct"])
        self.assertEqual(marked["detail"]["tally_fields_correct"], 2 / 3)


if __name__ == "__main__":
    unittest.main()
