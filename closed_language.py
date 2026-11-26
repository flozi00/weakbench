"""Grade short localized decision reports and the language of any exposed reasoning.

The response contract uses controlled natural language. Every accepted sentence
asserts one measurable fact. The separate DE/EN language probe uses lexical
evidence and reports inconclusive text explicitly.
"""
import functools
import re
import unicodedata


# The wire format is public. Instances, source events and computed values are private.
FIELDS = {"de": ("begründung", "antwort"), "en": ("justification", "answer")}
FORMS = {
    "de": {
        "value": ("Für {name} ergibt sich ein Endwert von {value}",
                  "Der Endwert für {name} liegt bei {value}"),
        "choice": ("Ich entscheide mich daher für {name}", "Meine Wahl fällt damit auf {name}"),
        "margin": ("Der absolute Unterschied beträgt hier {value}", "Die beiden Werte unterscheiden sich um {value}"),
    },
    "en": {
        "value": ("The final value for {name} is {value}", "For {name} the result comes to {value}"),
        "choice": ("I therefore choose {name} as my recommendation", "My choice is therefore {name}"),
        "margin": ("The absolute difference between the two is {value}", "The two values differ by {value}"),
    },
}
DETAIL_KEYS = frozenset(("valid_json", "report_fields_present", "request_language_ok", "justification_language_ok", "answer_language_ok",
                         "justification_facts_ok", "answer_facts_ok", "reasoning_present", "reasoning_language_decidable",
                         "reasoning_language_ok", "all_language_ok", "all_channels_language_ok"))

# The shared marker lists are intentionally conservative. These extra frequent words
# support short sentences without giving a name, digit or single token a language.
ADDITIONAL = {
    "de": frozenset("daher damit weil dabei dann beide beiden muss müssen wählen "
                    "hier liegt welche welcher werte erst zuerst somit falls du jedes zwei "
                    "bleibt deren also insgesamt bitte ergibt genau berechne "
                    "ergebnis jetzt nachdem beträgt etwas ergibt sich meine "
                    "mich entscheide empfehle zwischen unterschiedliche prüfen passt stimmt besser klar".split()),
    "en": frozenset("i my therefore thus because since first second must need "
                    "here result final choose choice values between both two each "
                    "only after before total so let's use calculate "
                    "remaining given now need difference compute please "
                    "equals enough less more recommend checking agree prefer works clear".split()),
}
# English and German share a few words (e.g. "also" and "so"); remove them
# explicitly, as a one-sided lexical vote for either language is misleading.
SHARED = {"also", "so", "war", "am", "an", "in", "will", "was"}
# Common words specific to other frequently encountered Latin-script languages.
# Multiple distinct markers in one clause are evidence of an observable switch.
OTHER_MARKERS = frozenset(
    "avec cette dans donc elle est lequel leur mais même nous parce pour sont une vous puisque "
    "aunque cual cuando desde entonces esta esto hay las los para pero porque también una usted "
    "anche degli della delle dello gli quindi sono sulla perché "
    "aan bij deze een geen het zijn niet wij omdat moet kunnen".split()
)


@functools.cache
def markers():
    # The Console imports the catalogue without starting a benchmark. Only a
    # live language check should import the runner's networking module.
    import common
    return {"de": (common.DE_WORDS | ADDITIONAL["de"]) - SHARED,
            "en": (common.EN_WORDS | ADDITIONAL["en"]) - SHARED}


WORDS = re.compile(r"[^\W\d_]+", re.UNICODE)
CLAUSE_BREAK = re.compile(r"[.!?,;:\n]+")
SEPARATOR = re.compile(r"[.;\n]+")


# Pure function-word sequences and repeated short keyword lists cannot establish
# an intelligible language sample. A few ordinary predicates also occur in short
# identifiable statements such as "Das stimmt genau" or "This works well".
PREDICATES = frozenset("choose recommend prefer agree works clear result difference compare computed correct "
                      "passt stimmt klar wähle empfehle vergleiche berechnet ergebnis unterschied unterscheiden korrekt".split())


def is_notation(clause):
    tokens = clause.split()
    if tokens and all(re.fullmatch(r"(?:[A-Z]|[A-Z]+[_0-9][A-Z_0-9]*|[0-9]+)", w) for w in tokens):
        return True
    # Identifiers joined by math operators (including local variable names).
    return (bool(re.search(r"[+=*/×÷−<>-]", clause))
            and not re.search(r"[^\W\d_]{2,}\s+[^\W\d_]{2,}", clause))


def language_status(text, lang):
    """True for target-language evidence, False for an observable switch, else None.

    This is a DE/EN lexical test. A single shared name, a number or an equation
    cannot establish the language. Opposite-language clauses are checked before
    counting the passage as a whole so padding cannot hide a switched sentence.
    """
    if lang not in FIELDS or not isinstance(text, str):
        return None
    vocabulary = markers()
    target = vocabulary[lang]
    other = vocabulary["en" if lang == "de" else "de"]
    normalized = unicodedata.normalize("NFC", text)
    words = []
    undecided_clause = False
    for clause in CLAUSE_BREAK.split(normalized):
        if is_notation(clause):
            continue
        clause_words = WORDS.findall(clause.casefold())
        words.extend(clause_words)
        distinct = set(clause_words)
        if len(distinct & OTHER_MARKERS) >= 2:
            return False
        # A full non-Latin phrase mixed into Latin prose should not disappear
        # from the language check. Isolated symbols in equations are common.
        if any(sum("LATIN" not in unicodedata.name(ch, "") for ch in word) >= 4 for word in clause_words):
            return False
        if len(clause_words) >= 4 and not distinct & target:
            undecided_clause = True
        # Catch an unpunctuated switch in a long sentence.
        streak = set()
        for w in clause_words:
            if w in target:
                streak.clear()
            elif w in other:
                streak.add(w)
                if len(streak) >= 2:
                    return False

    unique = set(words)
    if len(unique) < 3:
        return None
    opposite = sum(w in other for w in words)
    wanted = sum(w in target for w in words)
    if len(unique & other) >= 2 and (opposite >= wanted or opposite >= 3 and opposite * 4 > wanted):
        return False
    if undecided_clause or len(unique & target) < 2 or wanted * 10 < len(words):
        return None
    # Distinct function words plus a single arbitrary token (e.g. "the and quux")
    # give too little evidence of prose. Short idiomatic claims with an identifiable
    # predicate still have sufficient signal.
    if not unique & PREDICATES and len(unique - target - other - SHARED) < 2:
        return None
    return True


def request_language_ok(prompt, lang):
    return language_status(prompt, lang) is True


def validate_expected(expected):
    if type(expected) is not dict or set(expected) != {"values", "prefer"} or expected["prefer"] not in ("min", "max"):
        return False
    values = expected["values"]
    return (type(values) is dict and len(values) == 2 and
            all(isinstance(name, str) and re.fullmatch(r"[A-Z][a-z]{2,15}", name) for name in values) and
            all(type(v) is int and 0 <= v <= 1_000_000 for v in values.values()) and
            len(set(values.values())) == 2)


def clauses(text):
    # Full-sentence matching rejects extra/negated assertions and swapped bindings.
    normalized = unicodedata.normalize("NFC", text).casefold()
    return [" ".join(part.split()) for part in SEPARATOR.split(normalized) if part.strip()]


def checked_statements(text, alternatives):
    """Accept any sentence order, and require each factual role exactly once."""
    sentences = clauses(text)
    return (len(sentences) == len(alternatives) and
            all(sum(sentence in allowed for sentence in sentences) == 1 for allowed in alternatives))


def score_report(item, reply, evidence, conclusion):
    """Score two controlled-language fields against sets of permissible assertions."""
    from hidden import strict_json

    lang = item["lang"]
    prompt = item["messages"][0]["content"]
    try:
        parsed = strict_json(reply.get("content", ""))
        valid_json = True
    except (TypeError, ValueError, RecursionError):
        parsed, valid_json = None, False
    reason_key, answer_key = FIELDS[lang]
    has_fields = (type(parsed) is dict and set(parsed) == {reason_key, answer_key} and
                  all(type(v) is str for v in parsed.values()))
    reason, answer = (parsed[reason_key], parsed[answer_key]) if has_fields else ("", "")

    request_ok = request_language_ok(prompt, lang)
    reason_ok = language_status(reason, lang) is True
    answer_ok = language_status(answer, lang) is True
    evidence_ok = checked_statements(reason, evidence)
    conclusion_ok = checked_statements(answer, conclusion)

    native = reply.get("reasoning") or ""
    present = isinstance(native, str) and bool(native.strip())
    all_ok = request_ok and reason_ok and answer_ok
    detail = {"valid_json": float(valid_json), "report_fields_present": float(has_fields),
              "request_language_ok": float(request_ok),
              "justification_facts_ok": float(evidence_ok), "answer_facts_ok": float(conclusion_ok),
              "reasoning_present": float(present)}
    if has_fields:
        detail["justification_language_ok"] = float(reason_ok)
        detail["answer_language_ok"] = float(answer_ok)
    if present:
        native_ok = language_status(native, lang)
        detail["reasoning_language_decidable"] = float(native_ok is not None)
        if native_ok is not None:
            detail["reasoning_language_ok"] = float(native_ok)
        all_ok = all_ok and native_ok is True
        detail["all_channels_language_ok"] = float(all_ok)
    detail["all_language_ok"] = float(all_ok)
    correct = all_ok and evidence_ok and conclusion_ok
    return {"correct": correct, "score": float(correct), "lang_ok": answer_ok if has_fields else None, "detail": detail}


def score(item, reply):
    lang = item["lang"]
    target = item["expected"]
    values = target["values"]
    best = (max if target["prefer"] == "max" else min)(values, key=values.get)
    margin = max(values.values()) - min(values.values())
    forms = FORMS[lang]
    evidence = [
        {form.format(name=name, value=v).casefold() for form in forms["value"]} for name, v in values.items()
    ]
    conclusion = [{form.format(name=best).casefold() for form in forms["choice"]},
                  {form.format(value=margin).casefold() for form in forms["margin"]}]
    return score_report(item, reply, evidence, conclusion)
