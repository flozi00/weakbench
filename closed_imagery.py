"""Verify constrained poetic comparisons, their grammatical forms, and their image chain."""
import re
import unicodedata

DETAIL_KEYS = ("verse_length", "verse_grammar", "verse_shared_traits", "verse_cross_domain",
               "verse_linked", "verse_anchors", "verse_fresh_images", "verse_fresh_traits",
               "verse_varied_forms", "verse_drafts", "verse_distinct_drafts")


def _phrase(value):
    return (isinstance(value, str) and 0 < len(value) <= 64
            and re.fullmatch(r"[^\W\d_]+(?:[ '’\-][^\W\d_]+)*", value) is not None)


def validate_expected(spec):
    if not isinstance(spec, dict) or set(spec) != {"language", "images", "traits", "anchors", "length", "turn", "drafts"}:
        return False
    images, traits, anchors = spec["images"], spec["traits"], spec["anchors"]
    length, turn = spec["length"], spec["turn"]
    if (spec["language"] not in ("de", "en") or type(length) is not int or not 3 <= length <= 5
            or type(spec["drafts"]) is not int or spec["drafts"] not in (1, 2)
            or type(turn) is not int or not 0 < turn < length
            or not isinstance(traits, list) or not length <= len(traits) <= 12
            or not isinstance(images, list) or not length + 1 <= len(images) <= 16):
        return False
    if any(not isinstance(t, list) or len(t) != 3 or not all(_phrase(w) for w in t) for t in traits):
        return False
    if any(len(set(col)) != len(col) for col in zip(*traits)):
        return False
    for image in images:
        if not isinstance(image, list) or len(image) != 5:
            return False
        name, oblique, number, domain, cues = image
        if (not all(_phrase(w) for w in (name, oblique, domain)) or number not in ("sg", "pl")
                or not isinstance(cues, list) or not cues or len(cues) > len(traits)
                or any(type(c) is not int or not 0 <= c < len(traits) for c in cues)
                or len(set(cues)) != len(cues)):
            return False
    for col in (0, 1):
        if len({image[col] for image in images}) != len(images):
            return False
    return (isinstance(anchors, list) and len(anchors) == 3
            and all(type(i) is int and 0 <= i < len(images) for i in anchors)
            and len(set(anchors)) == 3)


def _initial(text):
    return text[:1].upper() + text[1:]


def _lines(text):
    if not isinstance(text, str) or len(text) > 2048:
        return []
    return [" ".join(line.split()) for line in unicodedata.normalize("NFC", text).strip().splitlines()]


def _read_line(line, spec, names, obliques):
    if not line.endswith("."):
        return None
    body = line[:-1]
    lang = spec["language"]
    for subject, (name, _, number, _, _) in enumerate(spec["images"]):
        plural = number == "pl"
        for cue, trait in enumerate(spec["traits"]):
            verb, image = trait[int(plural)], trait[2]
            prefix = f"{_initial(name)} {verb} {'wie' if lang == 'de' else 'like'} "
            if lang == "de":
                alt = f"{_initial(image)} {'gleichen' if plural else 'gleicht'} {name} "
            else:
                alt = f"{_initial(name)} {'share' if plural else 'shares'} {image} with "
            for form, opening, endings in ((0, prefix, names), (1, alt, obliques)):
                if body.startswith(opening):
                    target = endings.get(body[len(opening):])
                    if target is not None:
                        return subject, target, cue, form
    return None


def _check_stanza(spec, lines):
    images = spec["images"]
    names = {im[0]: i for i, im in enumerate(images)}
    obliques = {im[1]: i for i, im in enumerate(images)}
    length = spec["length"]
    detail = dict.fromkeys(DETAIL_KEYS[:-2], 0.0)
    signature = None
    detail["verse_length"] = float(len(lines) == length)
    if len(lines) == length:
        parsed = [_read_line(line, spec, names, obliques) for line in lines]
        detail["verse_grammar"] = sum(edge is not None for edge in parsed) / length
        if all(edge is not None for edge in parsed):
            route = [parsed[0][0], *(edge[1] for edge in parsed)]
            signature = frozenset(route)
            detail.update(
                verse_shared_traits=sum(c in images[a][4] and c in images[b][4] for a,b,c,_ in parsed) / length,
                verse_cross_domain=sum(images[a][3] != images[b][3] for a,b,_,_ in parsed) / length,
                verse_linked=sum(first[1] == second[0] for first,second in zip(parsed, parsed[1:])) / (length-1),
                verse_anchors=float([route[0], route[spec["turn"]], route[-1]] == spec["anchors"]),
                verse_fresh_images=float(len(set(route)) == len(route)),
                verse_fresh_traits=float(len({edge[2] for edge in parsed}) == length),
                verse_varied_forms=float(len({edge[3] for edge in parsed}) == 2),
            )
    return detail, signature


def score(item, reply):
    spec = item["expected"]
    count = spec["drafts"]
    # Only the harder briefs ask for a second version. Keep that metric's denominator explicit.
    keys = DETAIL_KEYS if count == 2 else DETAIL_KEYS[:-1]
    detail = dict.fromkeys(keys, 0.0)
    lines = _lines(reply.get("content"))
    stanzas = [[]]
    for line in lines:
        if line:
            stanzas[-1].append(line)
        else:
            stanzas.append([])
    if len(stanzas) == count and all(stanzas) and item["lang"] == spec["language"]:
        detail["verse_drafts"] = 1.0
        evaluations = [_check_stanza(spec, stanza) for stanza in stanzas]
        for key in DETAIL_KEYS[:-2]:
            detail[key] = sum(result[key] for result, _ in evaluations) / count
        if count == 2:
            first, second = (signature for _, signature in evaluations)
            detail["verse_distinct_drafts"] = float(first is not None and second is not None and first != second)
    correct = all(value == 1.0 for value in detail.values())
    return {"correct": correct, "score": float(correct), "detail": detail}
