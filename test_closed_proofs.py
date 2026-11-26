import copy
import json
import unittest

import closed_proofs


def wire_fixture(lang):
    """Public mock theory, localized request and one correct response."""
    spec = {
        "sources": {"F1": "Ada/Ray", "F2": "Bea/Rim"},
        "rules": {"R1": {"if": ["Ada/Ray", "Bea/Rim"], "then": "Ada/Sol"}},
        "goals": ["Ada/Sol", "Ada/Nix"],
    }
    if lang == "de":
        request = "Prüfe die Quellen und erkläre, welches Merkmal sich für Ada aus der Regel ergibt."
        response = {
            "begründung": "Die Quelle F2 belegt die Aussage Bea/Rim. Die Quelle F1 belegt die Aussage Ada/Ray. "
                          "Aus Ada/Ray und Bea/Rim ergibt sich nach Regel R1 die Aussage Ada/Sol.",
            "antwort": "Damit ist für den Fund Ada das Merkmal Sol belegt.",
        }
    else:
        request = "Check the two sources and explain which mark the rule establishes for Ada."
        response = {
            "justification": "The source F2 establishes the claim Bea/Rim. The source F1 establishes the claim Ada/Ray. "
                             "From Ada/Ray and Bea/Rim it follows by rule R1 that Ada/Sol holds.",
            "answer": "The record therefore establishes the mark Sol for the find Ada.",
        }
    facts = "F1 = Ada/Ray\nF2 = Bea/Rim\nR1 = Ada/Ray + Bea/Rim -> Ada/Sol\nAda/Sol | Ada/Nix"
    return spec, request + "\n" + facts, response


class ProofTests(unittest.TestCase):
    def setUp(self):
        self.expected = {
            "sources": {"F1": "Ava/Lux", "F2": "Bea/Mir", "F3": "Cia/Bel"},
            "rules": {
                "R1": {"if": ["Ava/Lux", "Bea/Mir"], "then": "Cia/Nox"},
                "R2": {"if": ["Cia/Bel"], "then": "Cia/Nox"},
                "R3": {"if": ["Cia/Nox"], "then": "Ava/Tal"},
                "R4": {"if": ["Ava/Zel"], "then": "Bea/Vim"},
                "R5": {"if": ["Bea/Vim"], "then": "Ava/Zel"},
            },
            "goals": ["Ava/Tal", "Ava/Zel"],
        }
        self.item = {
            "lang": "en",
            "expected": self.expected,
            "messages": [
                {
                    "role": "user",
                    "content": "Check the sources and tell me which of the two marks follows from the rules.",
                }
            ],
        }
        self.fact1 = "The source F1 establishes the claim Ava/Lux."
        self.fact2 = "The source F2 establishes the claim Bea/Mir."
        self.fact3 = "The source F3 establishes the claim Cia/Bel."
        self.rule1 = (
            "From Bea/Mir and Ava/Lux it follows by rule R1 that Cia/Nox holds."
        )
        self.rule2 = "From Cia/Bel it follows by rule R2 that Cia/Nox holds."
        self.finish = "From Cia/Nox it follows by rule R3 that Ava/Tal holds."
        self.answer = "The record therefore establishes the mark Tal for the find Ava."

    def grade(self, steps, answer=None, native=""):
        body = json.dumps(
            {
                "justification": " ".join(steps),
                "answer": self.answer if answer is None else answer,
            }
        )
        return closed_proofs.score(self.item, {"content": body, "reasoning": native})

    def test_alternative_rules_and_topological_orders(self):
        for steps in (
            [self.fact1, self.fact2, self.rule1, self.finish],
            [self.fact2, self.fact1, self.rule1, self.finish],
            [self.fact3, self.rule2, self.finish],
        ):
            result = self.grade(steps)
            self.assertTrue(result["correct"])
            self.assertFalse(result["detail"]["all_channels_verified"])
            self.assertNotIn("reasoning_language_ok", result["detail"])

    def test_unfounded_correct_guess_reversal_and_unused_fact(self):
        for bad in (
            [],
            [self.finish],
            [self.fact3, self.rule2, self.finish, self.fact1],
            [
                self.fact1,
                self.fact2,
                "From Cia/Nox it follows by rule R1 that Ava/Lux holds.",
                self.finish,
            ],
            [
                self.fact1,
                "From Ava/Lux it follows by rule R1 that Cia/Nox holds.",
                self.finish,
            ],
            [
                "From Bea/Vim it follows by rule R5 that Ava/Zel holds.",
                "From Ava/Zel it follows by rule R4 that Bea/Vim holds.",
            ],
        ):
            result = self.grade(bad)
            self.assertFalse(result["correct"])
            self.assertTrue(result["detail"]["answer_facts_ok"])
        padded = self.grade([self.fact1, self.fact3, self.rule2, self.finish])
        self.assertTrue(padded["detail"]["proof_steps_supported"])
        self.assertFalse(padded["detail"]["proof_steps_relevant"])

    def test_request_visible_answer_and_optional_native_language(self):
        steps = [self.fact3, self.rule2, self.finish]
        wrong = self.grade(["Die Quelle F3 belegt die Aussage Cia/Bel.",
                            "Aus Cia/Bel ergibt sich nach Regel R2 die Aussage Cia/Nox.",
                            "Aus Cia/Nox ergibt sich nach Regel R3 die Aussage Ava/Tal."])
        self.assertFalse(wrong["correct"])
        self.assertEqual(wrong["detail"]["justification_language_ok"], 0)
        self.assertEqual(wrong["detail"]["answer_facts_ok"], 1)
        self.assertFalse(
            self.grade(
                steps, answer="Damit ist für den Fund Ava das Merkmal Tal belegt."
            )["correct"]
        )
        self.assertFalse(
            self.grade(
                steps,
                native="Ich prüfe die Daten und begründe die Entscheidung auf Deutsch.",
            )["correct"]
        )
        self.assertTrue(
            self.grade(
                steps,
                native="I check the information and explain why the rules give this result.",
            )["detail"]["all_channels_verified"]
        )
        math = self.grade(steps, native="1 + 1 = 2")
        self.assertFalse(math["correct"])
        self.assertTrue(math["detail"]["reasoning_present"])
        self.assertNotIn("reasoning_language_ok", math["detail"])

    def test_a_fact_can_support_two_branches_but_cannot_be_double_counted(self):
        self.expected["rules"].update({
            "R6": {"if": ["Cia/Bel"], "then": "Bea/Duo"},
            "R7": {"if": ["Cia/Nox", "Bea/Duo"], "then": "Ava/Tal"},
            "R8": {"if": ["Cia/Nox"], "then": "Cia/Nox"},
        })
        branch = "From Cia/Bel it follows by rule R6 that Bea/Duo holds."
        merge = "From Bea/Duo and Cia/Nox it follows by rule R7 that Ava/Tal holds."
        self.assertTrue(self.grade([self.fact3, self.rule2, branch, merge])["correct"])
        self.assertFalse(self.grade([self.fact3, self.rule2, self.fact3, branch, merge])["correct"])
        self_loop = "From Cia/Nox it follows by rule R8 that Cia/Nox holds."
        self.assertFalse(self.grade([self_loop, self.finish])["correct"])
        self.assertFalse(self.grade([self.fact3, self.rule2, self_loop, self.finish])["correct"])

    def test_schema_rejects_malformed_and_nonexclusive_goals(self):
        self.assertTrue(closed_proofs.validate_expected(self.expected))
        for key, value in (
            ("sources", {}),
            ("goals", ["Ava/Tal", "Ava/Tal"]),
            ("rules", {"R1": {"if": ["bad"], "then": "Cia/Nox"}}),
        ):
            self.assertFalse(
                closed_proofs.validate_expected({**self.expected, key: value})
            )
        both = copy.deepcopy(self.expected)
        both["rules"]["R6"] = {"if": ["Bea/Mir"], "then": "Ava/Zel"}
        self.assertFalse(closed_proofs.validate_expected(both))


if __name__ == "__main__":
    unittest.main()
