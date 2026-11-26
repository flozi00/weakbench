"""Replay a localized referring expression in both recipients' views."""
import itertools
import re

FORMS = {
    "de": ("die karte ist ", " und ", "keine gemeinsame beschreibung"),
    "en": ("the card is ", " and ", "no shared description"),
}
DETAIL_KEYS = ("reference_form_ok", "first_listener_unique", "second_listener_unique",
               "both_listeners_unique", "reference_minimal", "impossibility_correct")


def validate_expected(data):
    if not isinstance(data, dict) or set(data) != {"words", "views"}:
        return False
    words, views = data["words"], data["views"]
    if (not isinstance(words, list) or not 2 <= len(words) <= 10
            or any(not isinstance(w, str) or not re.fullmatch(r"[^\W\d_]{2,24}", w) or w != w.casefold() for w in words)
            or len(set(words)) != len(words)):
        return False
    if not isinstance(views, list) or len(views) != 2:
        return False
    for view in views:
        if not isinstance(view, dict) or set(view) != {"cards", "target"}:
            return False
        cards, target = view["cards"], view["target"]
        if (not isinstance(cards, list) or not 2 <= len(cards) <= 12
                or type(target) is not int or not 0 <= target < len(cards)):
            return False
        for card in cards:
            if (not isinstance(card, list) or len(card) > len(words)
                    or any(type(i) is not int or not 0 <= i < len(words) for i in card)
                    or len(set(card)) != len(card)):
                return False
    return True


def unique(view, selection):
    candidates = [i for i, card in enumerate(view["cards"]) if selection.issubset(card)]
    return candidates == [view["target"]]


def optimum(views):
    shared = set.intersection(*(set(view["cards"][view["target"]]) for view in views))
    for size in range(1, len(shared) + 1):
        for choice in itertools.combinations(shared, size):
            if all(unique(view, set(choice)) for view in views):
                return size
    return None


def interpret(text, lang, words):
    if not isinstance(text, str) or len(text) > 512:
        return None
    sentence = " ".join(text.casefold().split()).removesuffix(".")
    prefix, join, impossible = FORMS[lang]
    if sentence == impossible:
        return ()
    if not sentence.startswith(prefix):
        return None
    terms = sentence[len(prefix):].split(join)
    if not set(terms) <= set(words) or len(terms) != len(set(terms)):
        return None
    return tuple(words.index(term) for term in terms)


def score(item, reply):
    expected = item["expected"]
    views = expected["views"]
    choice = interpret(reply.get("content"), item["lang"], expected["words"])
    minimum = optimum(views)
    matches = [bool(choice) and unique(view, set(choice)) for view in views]
    minimal = all(matches) and len(choice) == minimum
    abstained = choice == () and minimum is None
    correct = minimal or abstained
    detail = {"reference_form_ok": float(choice is not None)}
    if minimum is None:
        detail["impossibility_correct"] = float(abstained)
    else:
        detail.update(first_listener_unique=float(matches[0]), second_listener_unique=float(matches[1]),
                      both_listeners_unique=float(all(matches)), reference_minimal=float(minimal))
    return {"correct": correct, "score": float(correct), "detail": detail}
