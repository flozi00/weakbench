"""Grade a submitted XOR package allocation and its Clarke-pivot payments."""
from collections import defaultdict
import re

DETAIL_KEYS = (
    "valid_json", "auction_form_ok", "auction_feasible",
    "auction_optimal", "auction_payments_correct", "auction_winner_payment_fraction",
)


def identifier(value):
    return isinstance(value, str) and re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,23}", value) is not None


def validate_expected(expected):
    if not isinstance(expected, dict) or set(expected) != {"bids"}:
        return False
    bids = expected["bids"]
    if not isinstance(bids, list) or not 1 <= len(bids) <= 24:
        return False
    offers, participants, goods = set(), set(), set()
    for row in bids:
        if not isinstance(row, list) or len(row) != 4:
            return False
        offer, bidder, bundle, value = row
        if not identifier(offer) or offer in offers or not identifier(bidder):
            return False
        if (not isinstance(bundle, list) or not 1 <= len(bundle) <= 10
                or not all(map(identifier, bundle)) or len(set(bundle)) != len(bundle)):
            return False
        if type(value) is not int or not 1 <= value <= 1_000_000:
            return False
        offers.add(offer)
        participants.add(bidder)
        goods.update(bundle)
    return len(participants) <= 8 and len(goods) <= 10


def maximum(grouped, exclude=None):
    """After each bidder, keep the best welfare for every occupied goods mask."""
    best = {0: 0}
    for bidder, choices in grouped.items():
        if bidder == exclude:
            continue
        after = best.copy()
        for mask, value in choices:
            for occupied, total in best.items():
                if not occupied & mask:
                    union = occupied | mask
                    after[union] = max(after.get(union, 0), total + value)
        best = after
    return max(best.values())


def score(item, reply):
    from hidden import strict_json

    try:
        answer = strict_json(reply.get("content"))
        parsed = True
    except (TypeError, ValueError, RecursionError):
        answer, parsed = None, False
    detail = dict.fromkeys(DETAIL_KEYS[:-1], 0.0)
    detail["valid_json"] = float(parsed)
    shape = isinstance(answer, dict) and set(answer) == {"accepted", "payments"}
    if not shape:
        return {"correct": False, "score": 0.0, "detail": detail}

    ballots = item["expected"]["bids"]
    goods = sorted({good for _, _, bundle, _ in ballots for good in bundle})
    bits = {good: 1 << i for i, good in enumerate(goods)}
    offers, grouped = {}, defaultdict(list)
    for offer, bidder, bundle, value in ballots:
        mask = sum(bits[good] for good in bundle)
        offers[offer] = (bidder, mask, value)
        grouped[bidder].append((mask, value))

    selected, payments = answer["accepted"], answer["payments"]
    selection_ok = (isinstance(selected, list) and len(selected) <= len(offers)
                    and all(isinstance(o, str) and o in offers for o in selected)
                    and len(selected) == len(set(selected)))
    money_ok = (isinstance(payments, dict) and payments.keys() == grouped.keys()
                and all(type(n) is int and n >= 0 for n in payments.values()))
    detail["auction_form_ok"] = float(selection_ok and money_ok)
    if not selection_ok:
        return {"correct": False, "score": 0.0, "detail": detail}

    awards, occupied = {}, 0
    for offer in selected:
        bidder, mask, value = offers[offer]
        if bidder in awards or occupied & mask:
            return {"correct": False, "score": 0.0, "detail": detail}
        awards[bidder] = value
        occupied |= mask
    detail["auction_feasible"] = 1.0
    welfare = sum(awards.values())
    if welfare != maximum(grouped):
        return {"correct": False, "score": 0.0, "detail": detail}
    detail["auction_optimal"] = 1.0

    winner_prices = {b: maximum(grouped, exclude=b) - (welfare - value) for b, value in awards.items()}
    def price_match(bidder, value):
        return isinstance(payments, dict) and type(payments.get(bidder)) is int and payments[bidder] == value

    detail["auction_winner_payment_fraction"] = sum(price_match(b, p) for b, p in winner_prices.items()) / len(awards)
    correct = money_ok and all(price_match(b, winner_prices.get(b, 0)) for b in grouped)
    detail["auction_payments_correct"] = float(correct)
    return {"correct": correct, "score": float(correct), "detail": detail}
