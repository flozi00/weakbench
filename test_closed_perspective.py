"""Public synthetic, case-independent tests for localized perspective reports."""

import json
import unittest

import closed_language as language
import closed_perspective as perspective


# Public synthetic record, separate from any generated benchmark scenario.
FACTS = dict(
    speaker="Avena",
    listener="Borim",
    town="Corva",
    spoken="2000-02-29",
    giver="Borim",
    receiver="Avena",
    event="2000-03-01",
)
REQUESTS = {
    "de": "Bitte beantworte die Frage auf Deutsch und begründe deine Entscheidung.",
    "en": "Please give your answer in English and explain the reason for your decision.",
}


def render(lang, *, facts=None, alt=0):
    v = facts or FACTS
    utterances = [
        perspective.FORMS[lang][key][alt].format(**v) + "."
        for key in ("people", "setting", "transfer", "event")
    ]
    why, answer = language.FIELDS[lang]
    return {why: " ".join(utterances[:2][::-1]), answer: " ".join(utterances[2:][::-1])}


def grade(lang, doc=None, reasoning=None, raw=None, prompt=None):
    row = {
        "lang": lang,
        "expected": FACTS,
        "messages": [
            {
                "role": "user",
                "content": prompt if prompt is not None else REQUESTS[lang],
            }
        ],
    }
    reply = {
        "content": raw
        if raw is not None
        else json.dumps(render(lang) if doc is None else doc, ensure_ascii=False)
    }
    if reasoning is not None:
        reply["reasoning"] = reasoning
    return perspective.score(row, reply)


class PerspectiveTests(unittest.TestCase):
    def test_public_schema(self):
        self.assertTrue(perspective.validate_expected(FACTS))
        for change in [
            dict(spoken="2000-02-30"),
            dict(spoken="20000229"),
            dict(event="2000-03-04"),
            dict(event=False),
            dict(town=None),
            dict(town="Avena"),
            dict(listener="Avena"),
            dict(receiver="Corva"),
            dict(extra=2),
        ]:
            self.assertFalse(perspective.validate_expected({**FACTS, **change}))
        for nonobject in (None, [], 42):
            self.assertFalse(perspective.validate_expected(nonobject))

    def test_german_and_english_variants(self):
        for lang in REQUESTS:
            for variant in (0, 1):
                result = grade(lang, render(lang, alt=variant))
                self.assertTrue(result["correct"])
                self.assertEqual(result["detail"]["justification_facts_ok"], 1)
                self.assertEqual(result["detail"]["reasoning_present"], 0)
                self.assertNotIn("reasoning_language_ok", result["detail"])
                self.assertNotIn("all_channels_language_ok", result["detail"])

    def test_facts_graded_by_role_and_field(self):
        for lang in REQUESTS:
            why, answer = language.FIELDS[lang]
            for field, role, changes in [
                (
                    why,
                    "justification",
                    dict(speaker=FACTS["listener"], listener=FACTS["speaker"]),
                ),
                (why, "justification", dict(spoken=FACTS["event"])),
                (why, "justification", dict(town="Terna")),
                (
                    answer,
                    "answer",
                    dict(giver=FACTS["receiver"], receiver=FACTS["giver"]),
                ),
                (answer, "answer", dict(event=FACTS["spoken"])),
                (answer, "answer", dict(town="Terna")),
            ]:
                with self.subTest(lang=lang, change=changes):
                    original = render(lang)
                    bad = render(lang, facts={**FACTS, **changes})
                    result = grade(lang, {**original, field: bad[field]})
                    self.assertFalse(result["correct"])
                    self.assertEqual(result["detail"][role + "_facts_ok"], 0)
                    self.assertEqual(
                        result["detail"][
                            "answer_facts_ok"
                            if role == "justification"
                            else "justification_facts_ok"
                        ],
                        1,
                    )
                    self.assertEqual(result["detail"]["all_language_ok"], 1)

    def test_wrong_or_extra_assertion_fails(self):
        for lang in REQUESTS:
            why, answer = language.FIELDS[lang]
            original = render(lang)
            first = original[why].split(".")[0] + "."
            for wrong in [
                first,
                first + " " + first,
                original[why] + " " + first,
                original[why]
                + " "
                + ("Das stimmt nicht." if lang == "de" else "That is incorrect."),
            ]:
                result = grade(lang, {**original, why: wrong})
                self.assertFalse(result["correct"])
                self.assertEqual(result["detail"]["justification_facts_ok"], 0)

    def test_switch_and_observability_denominators(self):
        for lang, other in [("de", "en"), ("en", "de")]:
            for same, f in zip(language.FIELDS[lang], language.FIELDS[other]):
                result = grade(lang, {**render(lang), same: render(other)[f]})
                self.assertFalse(result["correct"])
                self.assertEqual(result["detail"]["all_language_ok"], 0)
            good = grade(lang, reasoning=REQUESTS[lang])
            self.assertTrue(good["correct"])
            self.assertEqual(good["detail"]["all_channels_language_ok"], 1)
            bad = grade(lang, reasoning=REQUESTS[other])
            self.assertFalse(bad["correct"])
            self.assertEqual(bad["detail"]["reasoning_language_ok"], 0)
            unclear = grade(lang, reasoning="A0 + B1 = 9")
            self.assertFalse(unclear["correct"])
            self.assertEqual(unclear["detail"]["reasoning_language_decidable"], 0)
            self.assertNotIn("reasoning_language_ok", unclear["detail"])
            self.assertEqual(
                grade(lang, prompt=REQUESTS[other])["detail"]["request_language_ok"], 0
            )

    def test_schema_strictness(self):
        for lang in REQUESTS:
            why, answer = language.FIELDS[lang]
            for raw in (
                "text",
                json.dumps({why: 4, answer: 42}),
                json.dumps({why: "s"}),
                json.dumps({**render(lang), "extra": "s"}),
                "{" + json.dumps(why) + ':"a",' + json.dumps(why) + ':"b"}',
            ):
                result = grade(lang, raw=raw)
                self.assertFalse(result["correct"])
                self.assertIsNone(result["lang_ok"])
                self.assertNotIn("justification_language_ok", result["detail"])
                self.assertNotIn("answer_language_ok", result["detail"])


if __name__ == "__main__":
    unittest.main()
