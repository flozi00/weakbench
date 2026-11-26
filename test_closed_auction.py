"""Regression tests with hand-computed synthetic XOR auctions."""
import json
import unittest

from closed_auction import score, validate_expected


def offers():
    # A+B can make 13 either way. Without A the maximum is 6, without B
    # it is 10. Dropping only the winning offer preserves a 13-valued award.
    return [
        ["Ax", "A", ["x"], 10],
        ["Ay", "A", ["y"], 7],
        ["Bx", "B", ["x"], 6],
        ["By", "B", ["y"], 3],
        ["Cx", "C", ["x"], 2],
    ]


def grade(accepted, payments, bids=None):
    item = {"expected": {"bids": offers() if bids is None else bids}}
    return score(item, {"content": json.dumps({"accepted": accepted, "payments": payments})})


class AuctionScoreTests(unittest.TestCase):
    def assert_rejected(self, result):
        self.assertIs(result["correct"], False)
        self.assertEqual(result["score"], 0.0)

    def test_tied_allocations_charge_same_winners_different_prices(self):
        first = {"A": 3, "B": 0, "C": 0}
        second = {"A": 0, "B": 3, "C": 0}
        for accepted, prices, other_prices in [
            (["Ax", "By"], first, second),
            (["Ay", "Bx"], second, first),
        ]:
            with self.subTest(accepted=accepted):
                result = grade(accepted, prices)
                self.assertIs(result["correct"], True)
                self.assertEqual(result["score"], 1.0)
                self.assertEqual(result["detail"], {
                    "valid_json": 1.0,
                    "auction_form_ok": 1.0,
                    "auction_feasible": 1.0,
                    "auction_optimal": 1.0,
                    "auction_payments_correct": 1.0,
                    "auction_winner_payment_fraction": 1.0,
                })
                swapped = grade(accepted, other_prices)
                self.assert_rejected(swapped)
                self.assertEqual(swapped["detail"]["auction_optimal"], 1.0)
                self.assertEqual(swapped["detail"]["auction_winner_payment_fraction"], 0.0)

    def test_counterfactual_removes_every_offer_by_winner(self):
        # Removing only Ax would leave Ay+Bx, erroneously pricing A at 10.
        wrong = grade(["Ax", "By"], {"A": 10, "B": 0, "C": 0})
        self.assert_rejected(wrong)
        self.assertEqual(wrong["detail"]["auction_winner_payment_fraction"], 0.5)

    def test_loser_must_be_present_at_zero(self):
        for payments, form_ok in [
            ({"A": 3, "B": 0, "C": 1}, 1.0),
            ({"A": 3, "B": 0}, 0.0),
            ({"A": 3, "B": 0, "C": 0, "extra": 0}, 0.0),
        ]:
            with self.subTest(payments=payments):
                result = grade(["Ax", "By"], payments)
                self.assert_rejected(result)
                self.assertEqual(result["detail"]["auction_form_ok"], form_ok)
                self.assertEqual(result["detail"]["auction_winner_payment_fraction"], 1.0)
                self.assertEqual(result["detail"]["auction_payments_correct"], 0.0)

    def test_xor_and_overlapping_bundles_are_infeasible(self):
        bids = offers() + [["Dxy", "D", ["x", "y"], 1]]
        for accepted in [["Ax", "Ay"], ["Ax", "Bx"], ["Dxy", "By"], ["Dxy", "Cx"]]:
            with self.subTest(accepted=accepted):
                result = grade(accepted, dict.fromkeys("ABCD", 0), bids)
                self.assert_rejected(result)
                self.assertEqual(result["detail"]["auction_form_ok"], 1.0)
                self.assertEqual(result["detail"]["auction_feasible"], 0.0)
                self.assertEqual(result["detail"]["auction_optimal"], 0.0)

    def test_bid_bundle_selection_and_payment_order(self):
        # A's bundle beats the two singletons by 1 and pays their sum.
        bids = [
            ["Pair", "A", ["x", "y"], 10],
            ["SingleX", "B", ["x"], 4],
            ["SingleY", "C", ["y"], 5],
            ["Other", "D", ["z"], 2],
        ]
        payments = {"A": 9, "B": 0, "C": 0, "D": 0}
        reference = grade(["Pair", "Other"], payments, bids)
        self.assertIs(reference["correct"], True)
        bids[0][2].reverse()
        result = grade(["Other", "Pair"], dict(reversed(list(payments.items()))), bids[::-1])
        self.assertEqual(reference, result)

    def test_suboptimal_awards_receive_zero(self):
        for accepted in [[], ["Ax"], ["Ay", "Cx"]]:
            with self.subTest(accepted=accepted):
                result = grade(accepted, {"A": 0, "B": 0, "C": 0})
                self.assert_rejected(result)
                self.assertEqual(result["detail"]["auction_form_ok"], 1.0)
                self.assertEqual(result["detail"]["auction_feasible"], 1.0)
                self.assertEqual(result["detail"]["auction_optimal"], 0.0)
                self.assertEqual(result["detail"]["auction_payments_correct"], 0.0)

    def test_json_parse_and_form_are_distinct(self):
        cases = [
            (None, 0.0),
            ("{", 0.0),
            ('{"accepted":[],"accepted":[],"payments":{}}', 0.0),
            ('{"accepted":[],"payments":{"A":0,"A":0}}', 0.0),
            ('{"accepted":[],"payments":{"A":NaN}}', 0.0),
            ("null", 1.0),
            ("[]", 1.0),
            ('{"accepted":[]}', 1.0),
            ('{"accepted":[],"payments":{},"other":true}', 1.0),
        ]
        for content, valid in cases:
            with self.subTest(content=content):
                result = score({"expected": {"bids": offers()}}, {"content": content})
                self.assert_rejected(result)
                self.assertEqual(result["detail"]["valid_json"], valid)
                self.assertEqual(result["detail"]["auction_form_ok"], 0.0)

    def test_invalid_offer_selections(self):
        for selected in ["Ax", ["Ax", "Ax"], ["missing"], [True], [1.0], [["Ax"]]]:
            with self.subTest(selected=selected):
                result = grade(selected, {"A": 3, "B": 0, "C": 0})
                self.assert_rejected(result)
                self.assertEqual(result["detail"]["valid_json"], 1.0)
                self.assertEqual(result["detail"]["auction_form_ok"], 0.0)

    def test_payment_types_and_nonnegative_amounts(self):
        for payments in [
            {"A": 3.0, "B": 0, "C": 0},
            {"A": 3, "B": False, "C": 0},
            {"A": 3, "B": 0, "C": False},
            {"A": -1, "B": 0, "C": 0},
            {"A": "3", "B": 0, "C": 0},
            [],
        ]:
            with self.subTest(payments=payments):
                result = grade(["Ax", "By"], payments)
                self.assert_rejected(result)
                self.assertEqual(result["detail"]["valid_json"], 1.0)
                self.assertEqual(result["detail"]["auction_optimal"], 1.0)
                self.assertEqual(result["detail"]["auction_form_ok"], 0.0)


class AuctionExpectedTests(unittest.TestCase):
    def test_valid_expected_including_upper_bounds(self):
        self.assertTrue(validate_expected({"bids": offers()}))
        bids = [[f"O{i}", f"B{i % 8}", [f"G{i % 10}"], 1_000_000] for i in range(24)]
        self.assertTrue(validate_expected({"bids": bids}))

    def test_invalid_expected_structure_and_limits(self):
        for expected in [
            None, [], {}, {"bids": []}, {"bids": "no"}, {"bids": offers(), "secret": 1},
            {"bids": [["O", "B"]]}, {"bids": [offers()[0], offers()[0]]},
            {"bids": [[f"O{i}", "B", ["G"], 1] for i in range(25)]},
            {"bids": [[f"O{i}", f"B{i}", ["G"], 1] for i in range(9)]},
            {"bids": [[f"O{i}", "B", [f"G{i}"], 1] for i in range(11)]},
        ]:
            with self.subTest(expected=expected):
                self.assertIs(validate_expected(expected), False)

    def test_invalid_bid_fields(self):
        for row in [
            ["1", "B", ["G"], 1], ["O", "A" * 25, ["G"], 1], ["O", False, ["G"], 1],
            ["O", "B", [], 1], ["O", "B", "G", 1], ["O", "B", ["G", "G"], 1],
            ["O", "B", [1], 1], ["O", "B", ["G-x"], 1], ["O", "B", [{}], 1],
            *[["O", "B", ["G"], v] for v in (0, -1, 1_000_001, True, 1.0, "1")],
        ]:
            with self.subTest(row=row):
                self.assertIs(validate_expected({"bids": [row]}), False)


if __name__ == "__main__":
    unittest.main()
