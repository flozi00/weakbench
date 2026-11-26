"""Parse two controlled natural-language clauses into events and a discourse relation."""

from collections import Counter
import re
import unicodedata

DETAIL_KEYS = (
    "fusion_target_grammar",
    "fusion_other_language",
    "fusion_events_fraction",
    "fusion_relation_kind",
    "fusion_relation_direction",
)
WORDS = re.compile(r"[^\W\d_]+", re.UNICODE)


def validate_expected(data):
    if not isinstance(data, dict) or set(data) != {
        "names",
        "lexicon",
        "events",
        "relation",
    }:
        return False
    names, lexicon, events = data["names"], data["lexicon"], data["events"]
    if not isinstance(names, dict) or not 2 <= len(names) <= 8:
        return False
    if not all(
        isinstance(n, str)
        and re.fullmatch(r"[A-Z][a-z]{0,19}", n)
        and isinstance(g, str)
        and g in ("f", "m")
        for n, g in names.items()
    ):
        return False
    if not isinstance(lexicon, dict) or not 1 <= len(lexicon) <= 4:
        return False
    for key, entry in lexicon.items():
        if not isinstance(key, str) or not re.fullmatch(r"[a-z]{1,24}", key):
            return False
        if not isinstance(entry, dict) or set(entry) != {"en", "de"}:
            return False
        for lang, n in (("en", 3), ("de", 2)):
            forms = entry[lang]
            if not isinstance(forms, list) or len(forms) != n:
                return False
            if not all(
                isinstance(f, str)
                and 1 <= len(f) <= 32
                and f.islower()
                and WORDS.fullmatch(f)
                for f in forms
            ):
                return False
    for lang, count in (("en", 3), ("de", 2)):
        for index in range(count):
            inflected = {entry[lang][index].casefold() for entry in lexicon.values()}
            if len(inflected) != len(lexicon):
                return False
    if not isinstance(events, list) or len(events) != 2 or events[0] == events[1]:
        return False
    for ev in events:
        if (
            not isinstance(ev, list)
            or len(ev) != 4
            or not all(isinstance(x, str) for x in ev[:3])
        ):
            return False
        s, v, o, neg = ev
        if (
            s not in names
            or o not in names
            or s == o
            or v not in lexicon
            or type(neg) is not bool
        ):
            return False
    return data["relation"] in ("before", "cause", "also")


def normalized(text):
    if not isinstance(text, str) or len(text) > 900:
        return None
    text = (
        unicodedata.normalize("NFC", text).strip().casefold().removesuffix(".").strip()
    )
    for contraction, expanded in (("didn't", "did not"), ("hadn't", "had not")):
        text = re.sub(rf"\b{contraction}\b", expanded, text.replace("’", "'"))
    if not re.fullmatch(r"[^\W\d_]+(?:[\s,]+[^\W\d_]+)*", text):
        return None
    return [p.split() for p in text.split(",")]


def meaning(relation, first, second):
    if relation == "also":
        first, second = sorted((first, second))
    return relation, first, second


class Reader:
    def __init__(self, lang, vocabulary):
        self.lang = lang
        self.names = {n.casefold(): n for n in vocabulary["names"]}
        self.genders = vocabulary["names"]
        self.lexicon = {
            k: [v.casefold() for v in forms[lang]]
            for k, forms in vocabulary["lexicon"].items()
        }

    def clause(self, tokens, order="main", perfect=False):
        if len(tokens) < 3:
            return set()
        if self.lang == "en":
            subj, obj = tokens[0], tokens[-1]
            vp = tokens[1:-1]
            options = [(False, 1, vp)] if len(vp) == 1 else []
            if vp[:2] == ["did", "not"]:
                options.append((True, 0, vp[2:]))
            if perfect and vp[:1] == ["had"]:
                negative = vp[1:2] == ["not"]
                options.append((negative, 2, vp[2:] if negative else vp[1:]))
        elif order in ("main", "inverted"):
            subj, finite = (
                (tokens[0], tokens[1]) if order == "main" else (tokens[1], tokens[0])
            )
            obj = tokens[2]
            tail = tokens[3:]
            options = [(bool(tail), 0, [finite])] if tail in ([], ["nicht"]) else []
        else:
            subj, obj = tokens[:2]
            vp = tokens[2:]
            negative = vp[:1] == ["nicht"]
            if negative:
                vp = vp[1:]
            options = (
                [(negative, 1, vp[:-1])] if perfect and vp[-1:] == ["hatte"] else []
            )
            if not perfect:
                options.append((negative, 0, vp))
        if subj not in self.names or obj not in self.names or subj == obj:
            return set()
        return {
            (self.names[subj], key, self.names[obj], negative)
            for negative, index, vp in options
            for key, forms in self.lexicon.items()
            if vp == [forms[index]]
        }

    def linked(self, conj, dependent, main, front):
        inverse = conj in ("after", "nachdem")
        cause = conj in ("because", "weil")
        subs = self.clause(dependent, "subordinate", perfect=inverse)
        mains = self.clause(main, "inverted" if front and self.lang == "de" else "main")
        for sub in subs:
            for host in mains:
                yield (
                    meaning("cause" if cause else "before", sub, host)
                    if inverse or cause
                    else meaning("before", host, sub)
                )

    def relative(self, words, anchor):
        if len(words) < 3 or anchor not in self.names:
            return set()
        if self.lang == "de" and self.genders[self.names[anchor]] != "m":
            return set()
        pron, *tail = words
        results = set()
        if pron == ("who" if self.lang == "en" else "der"):
            results.update(self.clause([anchor, *tail], "subordinate"))
        if pron in (("who", "whom") if self.lang == "en" else ("den",)):
            completed = (
                [*tail, anchor] if self.lang == "en" else [tail[0], anchor, *tail[1:]]
            )
            results.update(self.clause(completed, "subordinate"))
        return results

    def parse(self, text):
        pieces = normalized(text)
        if not pieces or len(pieces) > 3 or not all(pieces):
            return set()
        results = set()
        connectors = {
            "en": ("before", "after", "because"),
            "de": ("bevor", "nachdem", "weil"),
        }[self.lang]
        and_word, so_word = ("and", "so") if self.lang == "en" else ("und", "deshalb")
        if len(pieces) == 1:
            words = pieces[0]
            for i, w in enumerate(words):
                if self.lang == "en" and w in connectors and i > 0:
                    results.update(self.linked(w, words[i + 1 :], words[:i], False))
                if w == and_word:
                    results.update(
                        meaning("also", a, b)
                        for a in self.clause(words[:i])
                        for b in self.clause(words[i + 1 :])
                    )
        if len(pieces) == 2:
            left, right = pieces
            if left[0] in connectors:
                results.update(self.linked(left[0], left[1:], right, True))
            if right[0] in connectors:
                results.update(self.linked(right[0], right[1:], left, False))
            if right[0] == and_word:
                results.update(
                    meaning("also", a, b)
                    for a in self.clause(left)
                    for b in self.clause(right[1:])
                )
            if right[0] == so_word:
                style = "inverted" if self.lang == "de" else "main"
                results.update(
                    meaning("cause", a, b)
                    for a in self.clause(left)
                    for b in self.clause(right[1:], style)
                )
            for a in self.clause(left):
                if left[-1] == a[2].casefold():
                    results.update(
                        meaning("also", a, b) for b in self.relative(right, left[-1])
                    )
        if len(pieces) == 3:
            head, relative, tail = pieces
            for host in self.clause(head + tail):
                s, _, o, _ = host
                if head == [s.casefold()] or (
                    self.lang == "de" and tail == ["nicht"] and head[-1] == o.casefold()
                ):
                    results.update(
                        meaning("also", host, b)
                        for b in self.relative(relative, head[-1])
                    )
        return results


def score(item, reply):
    data, lang = item["expected"], item["lang"]
    submitted = reply.get("content")
    parses = Reader(lang, data).parse(submitted)
    foreign = (
        Reader("de" if lang == "en" else "en", data).parse(submitted)
        if not parses
        else set()
    )
    detail = {
        "fusion_target_grammar": float(len(parses) == 1),
        "fusion_other_language": float(bool(foreign)),
        "fusion_events_fraction": 0.0,
        "fusion_relation_kind": 0.0,
    }
    correct = False
    if len(parses) == 1:
        actual = next(iter(parses))
        expected = meaning(data["relation"], *map(tuple, data["events"]))
        matched = sum((Counter(actual[1:]) & Counter(expected[1:])).values())
        detail["fusion_events_fraction"] = matched / 2
        detail["fusion_relation_kind"] = float(actual[0] == expected[0])
        correct = actual == expected
        if matched == 2 and actual[0] == expected[0] and actual[0] != "also":
            detail["fusion_relation_direction"] = float(correct)
    return {"correct": correct, "score": float(correct), "detail": detail}
