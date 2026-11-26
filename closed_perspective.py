"""Grade controlled bilingual reports about indexicals in quoted conversations."""

from datetime import date
import re

import closed_language as language

DETAIL_KEYS = language.DETAIL_KEYS
FIELDS = language.FIELDS
FORMS = {
    "en": {
        "people": (
            "In these words {speaker} is speaking to {listener}",
            "These are the words of {speaker} addressed to {listener}",
        ),
        "setting": (
            "The conversation takes place in {town} on {spoken}",
            "The words are spoken on {spoken} in {town}",
        ),
        "transfer": (
            "The person who gives the item is {giver} and the person who receives it is {receiver}",
            "The item goes from {giver} to {receiver}",
        ),
        "event": (
            "The exchange takes place in {town} on {event}",
            "The date of the exchange is {event} and its location is {town}",
        ),
    },
    "de": {
        "people": (
            "Bei diesen Worten spricht {speaker} zu {listener}",
            "Diese Worte von {speaker} richten sich an {listener}",
        ),
        "setting": (
            "Das Gespräch findet am {spoken} in {town} statt",
            "Die Worte fallen in {town} am {spoken}",
        ),
        "transfer": (
            "Der Gegenstand geht von {giver} an {receiver}",
            "{giver} gibt den Gegenstand an {receiver} weiter",
        ),
        "event": (
            "Die Übergabe findet am {event} in {town} statt",
            "Der Ort der Übergabe ist {town} und ihr Datum ist {event}",
        ),
    },
}


def validate_expected(v):
    if type(v) is not dict or set(v) != {
        "speaker",
        "listener",
        "town",
        "spoken",
        "giver",
        "receiver",
        "event",
    }:
        return False
    if not all(
        type(v[k]) is str and re.fullmatch(r"[A-Z][a-z]{2,18}", v[k])
        for k in ("speaker", "listener", "town", "giver", "receiver")
    ):
        return False
    if (
        v["speaker"] == v["listener"]
        or {v["speaker"], v["listener"]} != {v["giver"], v["receiver"]}
        or v["town"] in (v["speaker"], v["listener"])
    ):
        return False
    try:
        dates = [date.fromisoformat(v[k]) for k in ("spoken", "event")]
    except (TypeError, ValueError):
        return False
    return (
        all(v[k] == d.isoformat() for k, d in zip(("spoken", "event"), dates))
        and 0 <= (dates[1] - dates[0]).days <= 2
    )


def score(item, reply):
    values = item["expected"]
    patterns = [
        {form.format(**values).casefold() for form in FORMS[item["lang"]][role]}
        for role in ("people", "setting", "transfer", "event")
    ]
    return language.score_report(item, reply, patterns[:2], patterns[2:])
