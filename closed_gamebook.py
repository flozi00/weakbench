"""Check a finite gamebook as a reader, using visible choice words and all possible reading histories."""

from collections import Counter
import re
import unicodedata

DETAIL_KEYS = (
    "valid_json",
    "request_language_ok",
    "reasoning_present",
    "reasoning_language_decidable",
    "reasoning_language_ok",
    "book_all_choice_languages_verified",
    "all_channels_verified",
    "correct_with_all_channels",
    "book_card_schema",
    "book_prose_form",
    "book_choice_labels",
    "book_reachable",
    "book_reader_promises",
    "book_link_promises",
    "book_motif_continuity",
    "book_required_callbacks",
    "book_language_decidable",
    "book_language_match",
    "book_all_prose_verified",
    "book_choice_language_decidable",
    "book_choice_language_match",
    "book_playable",
)


def identifier_character(char):
    category = unicodedata.category(char)
    return category[0] in "LMN" or category in ("Pc", "Cf")


def words(text):
    """Do not split letters from neighbouring numbers, marks or identifier connectors."""
    text = unicodedata.normalize("NFC", text).casefold()
    out = []
    current = []
    apostrophes = "'’ʼ"
    for pos, char in enumerate(text):
        if char in apostrophes:
            joins = bool(
                current
                and pos + 1 < len(text)
                and text[pos + 1] not in apostrophes
                and identifier_character(text[pos + 1])
            )
        else:
            joins = identifier_character(char)
        if joins:
            current.append(char)
        elif current:
            if any(c.isalpha() for c in current):
                out.append("".join(current))
            current = []
    if current and any(c.isalpha() for c in current):
        out.append("".join(current))
    return out


FORMS = {
    "de": (
        "Ich folge dem Hinweis «{cue}» und komme zum Zeichen «{motif}».",
        "Ich wähle die Tür «{cue}» und sehe dort das Zeichen «{motif}».",
    ),
    "en": (
        "I follow the cue “{cue}” and reach the sign marked “{motif}”.",
        "I choose the door marked “{cue}” and it leads to the sign for “{motif}”.",
    ),
}
QUOTES = str.maketrans(dict.fromkeys("“”«»", '"'))


def caption_form(text, lang, cue, motif):
    def norm(value):
        return " ".join(
            unicodedata.normalize("NFC", value).casefold().translate(QUOTES).split()
        )

    for n, form in enumerate(FORMS[lang]):
        if norm(text) == norm(form.format(cue=cue, motif=motif)):
            return n
    return None


def table(spec, lang):
    names = (
        ("Schichten und Motive", "Hinweispaare", "Leserwünsche", "Rückbezüge", "Ende")
        if lang == "de"
        else ("Layers and motifs", "Cue pairs", "Promised readings", "Callbacks", "End")
    )
    lines = [names[0]]
    for layer in spec["layers"]:
        lines.append(" | ".join(k + " " + spec["motifs"][k] for k in layer))
    lines.extend((names[1], *(" | ".join(pair) for pair in spec["cues"]), names[2]))
    for moves, ending in spec["routes"]:
        lines.append(
            " ".join(pair[i] for pair, i in zip(spec["cues"], moves)) + " -> " + ending
        )
    lines.extend(
        (names[3], *(" <- ".join(pair) for pair in spec["callbacks"]), names[4])
    )
    return "\n".join(lines)


def request_language_ok(prompt, lang, spec):
    from closed_language import language_status

    if not isinstance(prompt, str) or lang not in FORMS:
        return False
    suffix = "\n" + table(spec, lang)
    return (
        prompt.endswith(suffix)
        and language_status(prompt[: -len(suffix)], lang) is True
    )


def valid_id(k):
    return (
        isinstance(k, str) and re.fullmatch("[A-Za-z][A-Za-z0-9]{0,15}", k) is not None
    )


def word(k):
    return (
        isinstance(k, str)
        and 1 < len(k) <= 24
        and k.isalpha()
        and k == unicodedata.normalize("NFC", k)
    )


def validate_expected(s):
    if not isinstance(s, dict) or set(s) != {
        "layers",
        "motifs",
        "cues",
        "routes",
        "callbacks",
    }:
        return False
    layers, motifs, cues, routes, callbacks = (
        s[k] for k in ("layers", "motifs", "cues", "routes", "callbacks")
    )
    if not isinstance(layers, list) or not 3 <= len(layers) <= 5:
        return False
    if any(
        not isinstance(layer, list)
        or len(layer) != (1 if i == 0 else 2)
        or not all(valid_id(k) for k in layer)
        for i, layer in enumerate(layers)
    ):
        return False
    keys = sum(layers, [])
    if (
        len(set(keys)) != len(keys)
        or not isinstance(motifs, dict)
        or set(motifs) != set(keys)
    ):
        return False
    if not all(word(v) for v in motifs.values()) or len(
        {v.casefold() for v in motifs.values()}
    ) != len(keys):
        return False
    depth = len(layers) - 1
    if (
        not isinstance(cues, list)
        or len(cues) != depth
        or not all(
            isinstance(pair, list) and len(pair) == 2 and all(word(w) for w in pair)
            for pair in cues
        )
    ):
        return False
    cue_words = [w.casefold() for pair in cues for w in pair]
    if len(set(cue_words)) != len(cue_words) or set(cue_words) & {
        w.casefold() for w in motifs.values()
    }:
        return False
    if not isinstance(routes, list) or not 2 <= len(routes) <= 2**depth:
        return False
    seen = set()
    for r in routes:
        if not isinstance(r, list) or len(r) != 2:
            return False
        moves, end = r
        if (
            not isinstance(moves, list)
            or len(moves) != depth
            or not all(type(i) is int and i in (0, 1) for i in moves)
            or not valid_id(end)
            or end not in layers[-1]
            or tuple(moves) in seen
        ):
            return False
        seen.add(tuple(moves))
    if not isinstance(callbacks, list) or not 1 <= len(callbacks) <= 12:
        return False
    level = {k: i for i, layer in enumerate(layers) for k in layer}
    for pair in callbacks:
        if (
            not isinstance(pair, list)
            or len(pair) != 2
            or not all(valid_id(k) and k in level for k in pair)
            or not 0 < level[pair[1]] < level[pair[0]]
        ):
            return False
    return len(set(map(tuple, callbacks))) == len(callbacks)


def score(item, reply):
    from hidden import strict_json
    from closed_language import language_status

    # Language match has observations only when some free paragraphs can be classified.
    detail = dict.fromkeys(
        (
            k
            for k in DETAIL_KEYS
            if k
            not in (
                "book_language_match",
                "book_choice_language_match",
                "reasoning_language_decidable",
                "reasoning_language_ok",
            )
        ),
        0.0,
    )

    spec = item["expected"]
    lang = item["lang"]
    messages = item.get("messages", [])
    prompt = (
        messages[0].get("content")
        if messages and isinstance(messages[0], dict)
        else None
    )
    request_ok = request_language_ok(prompt, lang, spec)
    detail["request_language_ok"] = float(request_ok)
    segments = reply.get("reasoning_parts")
    if segments is None:
        segments = [reply.get("reasoning"), reply.get("reasoning_content")]
    traces = (
        [part for part in segments if isinstance(part, str) and part.strip()]
        if isinstance(segments, (list, tuple))
        else []
    )

    present = bool(traces)
    detail["reasoning_present"] = float(present)
    states = [language_status(v, lang) for v in dict.fromkeys(traces)]
    native = (
        False
        if False in states
        else True
        if states and all(flag is True for flag in states)
        else None
    )
    if present:
        detail["reasoning_language_decidable"] = float(native is not None)
        if native is not None:
            detail["reasoning_language_ok"] = float(native)

    def finish(ok=False):
        ok = ok and request_ok and (not present or native is True)
        detail["correct_with_all_channels"] = float(
            bool(ok and detail["all_channels_verified"])
        )
        return {"correct": bool(ok), "score": float(bool(ok)), "detail": detail}

    try:
        text = reply["content"]
        if not isinstance(text, str) or len(text) > 30000:
            return finish()
        answer = strict_json(text)
    except (ValueError, KeyError, TypeError, RecursionError):
        return finish()
    detail["valid_json"] = 1.0
    layers = spec["layers"]
    keys = sum(layers, [])
    if (
        not isinstance(answer, dict)
        or set(answer) != {"cards"}
        or not isinstance(answer["cards"], dict)
        or set(answer["cards"]) != set(keys)
    ):
        return finish()
    cards = answer["cards"]
    motifs = {k: w.casefold() for k, w in spec["motifs"].items()}
    options = {}
    references = {}
    lex = {}
    prose = []
    lang_flags = []
    choice_flags = []
    signposts = []
    basic_format = True
    for step, layer in enumerate(layers):
        final = step == len(layers) - 1
        for k in layer:
            card = cards[k]
            if (
                not isinstance(card, dict)
                or set(card) != {"text", "links"}
                or not isinstance(card["text"], str)
                or not isinstance(card["links"], list)
                or len(card["links"]) != (0 if final else 2)
            ):
                return finish()
            body = card["text"].strip()
            tokens = words(body)
            lex[k] = set(tokens) - set(motifs.values())
            prose.append(
                6 <= len(tokens) <= 60
                and len(set(tokens)) >= 5
                and body.endswith((".", "!", "?", "…"))
            )
            lang_flags.append(language_status(body, item["lang"]))
            references[k] = {s for s, w in motifs.items() if w in tokens}
            options[k] = {}
            forms = set()
            for link in card["links"]:
                if (
                    not isinstance(link, dict)
                    or set(link) != {"label", "to"}
                    or not isinstance(link["label"], str)
                    or not valid_id(link["to"])
                ):
                    return finish()
                choice_flags.append(language_status(link["label"], item["lang"]))
                tokens = words(link["label"])
                choices = [
                    i
                    for i, w in enumerate(spec["cues"][step])
                    if w.casefold() in tokens
                ]
                if (
                    not 2 <= len(tokens) <= 15
                    or len(choices) != 1
                    or choices[0] in options[k]
                    or tokens.count(spec["cues"][step][choices[0]].casefold()) != 1
                    or link["to"] not in layers[step + 1]
                ):
                    basic_format = False
                    continue
                form = caption_form(
                    link["label"],
                    lang,
                    spec["cues"][step][choices[0]],
                    spec["motifs"][link["to"]],
                )
                forms.add(form)
                options[k][choices[0]] = link["to"]
                next_named = {n for n in layers[step + 1] if motifs[n] in tokens}
                signposts.append(form is not None and next_named == {link["to"]})
            if not final and forms != {0, 1}:
                basic_format = False
    detail["book_card_schema"] = 1.0
    freq = Counter(w for ws in lex.values() for w in ws)
    # Each paragraph must have one word of its own besides the required motif names.
    detail["book_prose_form"] = float(
        all(prose) and all(any(freq[w] == 1 for w in ws) for ws in lex.values())
    )
    detail["book_choice_labels"] = float(basic_format)
    decided = [f for f in lang_flags if f is not None]
    detail["book_language_decidable"] = len(decided) / len(keys)
    if decided:
        detail["book_language_match"] = sum(decided) / len(decided)
    detail["book_all_prose_verified"] = float(all(v is True for v in lang_flags))
    seen_labels = [flag for flag in choice_flags if flag is not None]
    detail["book_choice_language_decidable"] = len(seen_labels) / len(choice_flags)
    if seen_labels:
        detail["book_choice_language_match"] = sum(seen_labels) / len(seen_labels)
    detail["book_all_choice_languages_verified"] = float(
        all(flag is True for flag in choice_flags)
    )
    detail["all_channels_verified"] = float(
        request_ok
        and native is True
        and detail["book_all_prose_verified"] == 1
        and detail["book_all_choice_languages_verified"] == 1
    )
    detail["book_link_promises"] = (
        sum(signposts) / len(choice_flags) if choice_flags else 0.0
    )
    if not basic_format:
        return finish()
    # Keep separate histories after a merge; a memory of just one branch is insufficient.
    paths = {(): [layers[0][0]]}
    visited = {k: [] for k in keys}
    visited[layers[0][0]].append(set(paths[()]))
    for _ in spec["cues"]:
        extended = {}
        for prefix, history in paths.items():
            for choice, dest in options[history[-1]].items():
                line = [*history, dest]
                visited[dest].append(set(line))
                extended[prefix + (choice,)] = line
        paths = extended
    detail["book_reachable"] = sum(bool(v) for v in visited.values()) / len(keys)
    detail["book_reader_promises"] = sum(
        paths[tuple(moves)][-1] == end for moves, end in spec["routes"]
    ) / len(spec["routes"])
    detail["book_motif_continuity"] = sum(
        bool(visited[k])
        and k in references[k]
        and all(references[k] <= history for history in visited[k])
        for k in keys
    ) / len(keys)
    detail["book_required_callbacks"] = sum(
        prior in references[k] for k, prior in spec["callbacks"]
    ) / len(spec["callbacks"])
    playable = all(
        detail[k] == 1
        for k in (
            "book_choice_labels",
            "book_reachable",
            "book_reader_promises",
            "book_link_promises",
            "book_motif_continuity",
            "book_required_callbacks",
        )
    )
    detail["book_playable"] = float(playable)
    return finish(
        playable
        and detail["book_prose_form"] == 1
        and False not in lang_flags
        and False not in choice_flags
    )
