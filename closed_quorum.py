"""Grade delegated votes and an authenticated verbatim citation in a bilingual minute."""

import re

import closed_language

OUTCOMES = ("adopt", "reject", "tie", "quorum")
DETAIL_KEYS = (
    "valid_json",
    "minutes_form",
    "tally_fields_correct",
    "decision_correct",
    "source_id_correct",
    "source_text_exact",
    "request_language_ok",
    "justification_language_ok",
    "answer_language_ok",
    "reasoning_present",
    "reasoning_language_decidable",
    "reasoning_language_ok",
    "all_channels_verified",
    "correct_with_all_channels",
)
DECISIONS = {
    "de": {
        "adopt": ("Der Antrag wird angenommen.", "Die Vorlage erhält die Mehrheit."),
        "reject": (
            "Der Antrag wird abgelehnt.",
            "Die Vorlage verliert die Abstimmung.",
        ),
        "tie": (
            "Die Abstimmung endet unentschieden.",
            "Es gibt keine Stimmenmehrheit.",
        ),
        "quorum": ("Das Quorum wurde verfehlt.", "Die Beteiligung reicht nicht aus."),
    },
    "en": {
        "adopt": ("The proposal is adopted.", "The motion gains a majority."),
        "reject": ("The proposal is rejected.", "The motion loses the vote."),
        "tie": (
            "The vote ends in a tie.",
            "Neither side has the votes for a majority.",
        ),
        "quorum": (
            "Participation falls short of the quorum.",
            "Participation is too low.",
        ),
    },
}
SECTIONS = {"de": "Belege:", "en": "Sources:"}
VOTES = ("yes", "no", "abstain")


def identifier(value):
    return (
        isinstance(value, str)
        and re.fullmatch(r"[A-Z][a-zA-Z0-9]{0,19}", value) is not None
    )


def validate_expected(spec):
    if not isinstance(spec, dict) or set(spec) != {"ballots", "quorum", "sources"}:
        return False
    ballots, limit, sources = spec["ballots"], spec["quorum"], spec["sources"]
    if not (
        isinstance(ballots, list)
        and 2 <= len(ballots) <= 40
        and type(limit) is int
        and 1 <= limit <= len(ballots)
    ):
        return False
    if any(
        not isinstance(row, list) or len(row) != 2 or not identifier(row[0])
        for row in ballots
    ):
        return False
    names = {name for name, _ in ballots}
    if len(names) != len(ballots) or any(
        not isinstance(v, str) or v not in names and v not in VOTES for _, v in ballots
    ):
        return False
    if not isinstance(sources, list) or len(sources) != 4:
        return False
    for row in sources:
        if not isinstance(row, list) or len(row) != 3:
            return False
        outcome, key, text = row
        if (
            outcome not in OUTCOMES
            or not identifier(key)
            or not isinstance(text, str)
            or not 10 <= len(text) <= 240
            or any(not c.isprintable() for c in text)
            or text != text.strip()
        ):
            return False
    return all(len({row[i] for row in sources}) == 4 for i in range(3))


def request_language_ok(prompt, lang, spec):
    """Only the four exact authored source lines can be removed from the language probe."""
    if not isinstance(prompt, str) or lang not in SECTIONS:
        return False
    marker = "\n" + SECTIONS[lang] + "\n"
    if prompt.count(marker) != 1:
        return False
    before, after = prompt.split(marker)
    data, separator, rest = after.partition("\n\n")
    expected = [key + " | " + literal for _, key, literal in spec["sources"]]
    return (
        bool(separator)
        and data.splitlines() == expected
        and closed_language.request_language_ok(before + "\n" + rest, lang)
    )


def tally(spec):
    votes = dict(spec["ballots"])
    totals = dict.fromkeys(VOTES, 0)
    for person in votes:
        current, visited = person, set()
        while current in votes and current not in visited:
            visited.add(current)
            current = votes[current]
        totals[current if current in VOTES else "abstain"] += 1
    yes, no = totals["yes"], totals["no"]
    if yes + no < spec["quorum"]:
        outcome = "quorum"
    elif yes == no:
        outcome = "tie"
    else:
        outcome = "adopt" if yes > no else "reject"
    return (yes, no, totals["abstain"]), outcome


def tally_forms(lang, counts):
    y, n, u = counts
    if lang == "de":
        yes = "Ja-Stimme" if y == 1 else "Ja-Stimmen"
        no = "Nein-Stimme" if n == 1 else "Nein-Stimmen"
        unused = "ungenutzte Stimme" if u == 1 else "ungenutzte Stimmen"
        return (
            f"Die Auszählung ergibt {y} {yes}, {n} {no} und {u} {unused}.",
            f"Für Ja zähle ich {y}, für Nein {n} und als ungenutzt {u}.",
        )

    def noun(k):
        return "vote" if k == 1 else "votes"

    return (
        f"The count gives {y} yes {noun(y)}, {n} no {noun(n)} and {u} unused {noun(u)}.",
        f"I count {y} for yes, {n} for no and {u} as unused.",
    )


def read_tally(text, lang):
    if not isinstance(text, str) or len(text) > 180:
        return None
    values = re.findall(r"\b[0-9]{1,2}\b", text)
    if len(values) != 3:
        return None
    counts = tuple(map(int, values))
    return counts if text in tally_forms(lang, counts) else None


def score(item, reply):
    from hidden import strict_json

    spec, lang = item["expected"], item["lang"]
    counts, outcome = tally(spec)
    _, source_id, literal = next(row for row in spec["sources"] if row[0] == outcome)
    prompt = item["messages"][0]["content"]
    native = reply.get("reasoning")
    present = isinstance(native, str) and bool(native.strip())
    request_ok = request_language_ok(prompt, lang, spec)
    checks = dict.fromkeys(
        (
            "valid_json",
            "minutes_form",
            "tally_fields_correct",
            "decision_correct",
            "source_id_correct",
            "source_text_exact",
            "justification_language_ok",
            "answer_language_ok",
        ),
        0.0,
    )
    checks.update(
        request_language_ok=float(request_ok), reasoning_present=float(present)
    )
    native_ok = closed_language.language_status(native, lang) if present else None
    if present:
        checks["reasoning_language_decidable"] = float(native_ok is not None)
        if native_ok is not None:
            checks["reasoning_language_ok"] = float(native_ok)
    try:
        value = strict_json(reply.get("content"))
        checks["valid_json"] = 1.0
    except (TypeError, ValueError, RecursionError):
        value = None
    if isinstance(value, dict) and set(value) == {"parts"}:
        parts = value["parts"]
        if (
            isinstance(parts, list)
            and len(parts) == 3
            and all(isinstance(part, dict) for part in parts)
            and set(parts[0]) == set(parts[2]) == {"prose"}
            and set(parts[1]) == {"source", "verbatim"}
            and all(isinstance(v, str) for part in parts for v in part.values())
        ):
            checks["minutes_form"] = 1.0
            observed = read_tally(parts[0]["prose"], lang)
            recognized = next(
                (
                    key
                    for key, forms in DECISIONS[lang].items()
                    if parts[2]["prose"] in forms
                ),
                None,
            )
            checks["justification_language_ok"] = float(observed is not None)
            checks["answer_language_ok"] = float(recognized is not None)
            checks["decision_correct"] = float(recognized == outcome)
            if observed is not None:
                checks["tally_fields_correct"] = (
                    sum(a == b for a, b in zip(observed, counts)) / 3
                )
            checks["source_id_correct"] = float(parts[1]["source"] == source_id)
            checks["source_text_exact"] = float(parts[1]["verbatim"] == literal)
    surface = bool(
        checks["justification_language_ok"]
        and checks["answer_language_ok"]
        and request_ok
    )
    facts = all(
        checks[k] == 1
        for k in (
            "tally_fields_correct",
            "decision_correct",
            "source_id_correct",
            "source_text_exact",
        )
    )
    correct = surface and facts and (not present or native_ok is True)
    checks["all_channels_verified"] = float(
        surface
        and native_ok is True
        and checks["source_text_exact"] == 1
        and checks["source_id_correct"] == 1
    )
    checks["correct_with_all_channels"] = float(
        correct and checks["all_channels_verified"] == 1
    )
    return {"correct": correct, "score": float(correct), "detail": checks}
