"""Synthetic dates for relay score and language diagnostics; no released dialogue data."""

import copy
import json
import unittest

import closed_relay as c


def fixture():
    return {
        "target": "Beta",
        "window": [5, 7],
        "initial": {
            "K1": ["", "Alfa", 2, 4],
            "K2": ["Alfa", "Beta", 2, 3],
            "K3": ["", "Beta", 1, 12],
        },
        "updates": [
            {"K1": ["", "Alfa", 3, 3], "K4": ["", "Beta", 1, 11]},
            {"K2": ["Alfa", "Beta", 6, 7], "K5": ["", "Beta", 9, 12]},
        ],
    }


REQ = {
    "de": "Prüfe die Angaben und erkläre, warum der Termin im Zeitraum liegt oder außerhalb liegen kann.",
    "en": "Check the records and explain which date is possible for the milestone.",
}
TRACE = {
    "de": "Ich prüfe die Angaben und erkläre, warum sich dieses Ergebnis aus den Regeln ergibt.",
    "en": "I check the information and explain why the rules give this result.",
}


def synthetic_report(lang, i):
    lo, hi = ((4, 7), (5, 6), (9, 10))[i]
    if lang == "de":
        facts = f"Aus K1 und K2 ergibt sich Woche {lo} als frühestmöglicher Termin. Aus K2 und K1 ergibt sich Woche {hi} als spätestmöglicher Termin."
        verdict = (
            "Der Termin kann innerhalb oder außerhalb des Prüfzeitraums liegen.",
            "Der Termin liegt sicher im Prüfzeitraum.",
            "Der Termin liegt sicher außerhalb des Prüfzeitraums.",
        )[i]
        return {"begründung": facts, "antwort": verdict}
    facts = f"From K1 and K2 the earliest possible week is {lo}. From K2 and K1 the latest possible week is {hi}."
    verdict = (
        "The date can fall inside or outside the review window.",
        "The date is certainly inside the review window.",
        "The date is certainly outside the review window.",
    )[i]
    return {"justification": facts, "answer": verdict}


def wire_fixture(first):
    """Return the public synthetic expectation, three localized user prompts and three response objects."""
    langs = c.turn_languages(first)
    initial = {
        "de": "Gesucht ist der Termin von Beta. Der Prüfzeitraum umfasst die Wochen 5 bis 7. Alle Grenzen zählen mit. K1 datiert Alfa auf Woche 2 bis 4. Nach K2 liegt Beta um 2 bis 3 Wochen später als Alfa. K3 datiert Beta auf Woche 1 bis 12.",
        "en": "The target is the date of Beta. The review window covers weeks 5 through 7. Every bound is inclusive. K1 dates Alfa to weeks 2 through 4. Under K2 the week for Beta is 2 to 3 weeks later than the week for Alfa. K3 dates Beta to weeks 1 through 12.",
    }
    second = {
        "de": "Berichtige K1 auf Woche 3 für Alfa. Aus dem neuen Beleg K4 ergibt sich für Beta der Bereich von Woche 1 bis Woche 11.",
        "en": "Correct K1 to week 3 for Alfa. The new record K4 dates Beta to a week between 1 and 11.",
    }
    third = {
        "de": "Berichtige K2. Der Abstand von Alfa zu Beta beträgt jetzt 6 bis 7 Wochen. Aus dem neuen Beleg K5 ergibt sich für Beta der Bereich von Woche 9 bis Woche 12.",
        "en": "Correct K2. The interval from Alfa to Beta is now 6 to 7 weeks. The new record K5 dates Beta to a week between 9 and 12.",
    }
    documents = [
        REQ[turn_lang] + " " + table[turn_lang]
        for turn_lang, table in zip(langs, (initial, second, third))
    ]
    return (
        fixture(),
        documents,
        [synthetic_report(turn_lang, i) for i, turn_lang in enumerate(langs)],
    )


def item_and_reports(first):
    langs = c.turn_languages(first)
    expected, prompts, answers = wire_fixture(first)
    item = {
        "lang": first,
        "expected": expected,
        "turns": [[{"role": "user", "content": p}] for p in prompts],
    }
    reports = [
        {"content": json.dumps(a, ensure_ascii=False), "reasoning": TRACE[k]}
        for k, a in zip(langs, answers)
    ]
    return item, reports


class RelayTests(unittest.TestCase):
    def test_three_visible_reports_in_both_directions(self):
        self.assertTrue(c.validate_expected(fixture()))
        for first in ("de", "en"):
            item, reports = item_and_reports(first)
            good = c.score(item, {"observed_turns": reports})
            self.assertTrue(good["correct"])
            self.assertEqual(good["detail"]["all_channels_verified"], 1)
            self.assertEqual(
                good,
                c.finish([c.grade_turn(item, i, r) for i, r in enumerate(reports)]),
            )
            self.assertFalse(
                c.score(item, {"turns": [r["content"] for r in reports]})["correct"]
            )
            final_only = c.score(item, {"content": reports[-1]["content"]})
            self.assertFalse(final_only["correct"])

    def test_correct_with_all_channels_requires_content_and_evidence_at_every_turn(self):
        for first in ("de", "en"):
            item, reports = item_and_reports(first)
            good = c.score(item, {"observed_turns": reports})
            self.assertTrue(good["correct"])
            self.assertEqual(good["detail"]["correct_with_all_channels"], 1)
            for index in range(3):
                missing = copy.deepcopy(reports)
                del missing[index]["reasoning"]
                result = c.score(item, {"observed_turns": missing})
                self.assertTrue(result["correct"])
                self.assertEqual(result["detail"]["correct_with_all_channels"], 0)

                false_fact = copy.deepcopy(reports)
                lang = c.turn_languages(first)[index]
                answer = json.loads(false_fact[index]["content"])
                field = "begründung" if lang == "de" else "justification"
                answer[field] = answer[field].replace(str((4, 5, 9)[index]), "44", 1)
                false_fact[index]["content"] = json.dumps(answer)
                result = c.score(item, {"observed_turns": false_fact})
                self.assertFalse(result["correct"])
                self.assertEqual(result["detail"]["all_channels_verified"], 1)
                self.assertEqual(result["detail"]["correct_with_all_channels"], 0)
            self.assertEqual(c.score(item, {})["detail"]["correct_with_all_channels"], 0)

    def test_absent_undecidable_and_switched_native_traces(self):
        item, reports = item_and_reports("de")
        no_native = copy.deepcopy(reports)
        del no_native[0]["reasoning"]
        valid = c.score(item, {"observed_turns": no_native})
        self.assertTrue(valid["correct"])
        self.assertEqual(valid["detail"]["all_channels_verified"], 0)
        self.assertEqual(valid["detail"]["reasoning_present_fraction"], 2 / 3)
        self.assertNotIn("turn1_reasoning_language_ok", valid["detail"])
        for value, decidable in (("2 + 3 = 5", 0), (TRACE["en"], 1)):
            wrong = copy.deepcopy(reports)
            wrong[0]["reasoning"] = value
            output = c.score(item, {"observed_turns": wrong})
            self.assertFalse(output["correct"])
            self.assertEqual(output["detail"]["visible_reports_correct"], 1)
            self.assertEqual(
                output["detail"]["turn1_reasoning_language_decidable"], decidable
            )
            self.assertEqual(output["detail"]["turn3_turn_correct"], 1)

    def test_correct_final_does_not_hide_wrong_past(self):
        item, reports = item_and_reports("en")
        wrong = copy.deepcopy(reports)
        fields = synthetic_report("de", 1)
        fields["begründung"] = synthetic_report("en", 1)["justification"]
        wrong[1]["content"] = json.dumps(fields)
        detail = c.score(item, {"observed_turns": wrong})["detail"]
        self.assertFalse(detail["full"])
        self.assertEqual(detail["turn2_justification_language_ok"], 0)
        self.assertEqual(detail["turn2_answer_language_ok"], 1)
        self.assertEqual(detail["turn3_turn_correct"], 1)
        invalid = copy.deepcopy(item)
        invalid["turns"][1][0]["content"] = REQ["en"]
        detail = c.score(invalid, {"observed_turns": reports})["detail"]
        self.assertEqual(detail["visible_reports_correct"], 1)
        self.assertEqual(detail["all_requests_language_ok"], 0)
        self.assertFalse(detail["full"])

    def test_alternate_necessary_witnesses_and_padding(self):
        item, reports = item_and_reports("en")
        fields = synthetic_report("en", 2)
        alternate = fields["justification"].replace("From K1 and K2", "From K5")
        for text, passed in (
            (alternate, True),
            (alternate.replace("From K5", "From K5 and K1"), False),
            (alternate.replace("week is 9", "week is 8"), False),
        ):
            third = {
                **reports[2],
                "content": json.dumps({**fields, "justification": text}),
            }
            output = c.score(item, {"observed_turns": [*reports[:2], third]})
            self.assertIs(output["correct"], passed)
        cut = c.score(item, {"observed_turns": reports[:2]})["detail"]
        self.assertEqual(cut["turns_observed"], 2 / 3)
        self.assertNotIn("turn3_reasoning_present", cut)
        self.assertFalse(cut["full"])

    def test_bad_instance(self):
        e = fixture()
        for key, value in (
            ("updates", []),
            ("window", [True, 5]),
            ("target", "missing!!!"),
        ):
            self.assertFalse(c.validate_expected({**e, key: value}))
        e["updates"][1]["K5"] = ["", "Beta", 30, 33]
        self.assertFalse(c.validate_expected(e))


if __name__ == "__main__":
    unittest.main()
