"""Reproducible prefill probes separate from the private scored cases.

Token IDs in drift artifacts can be decoded. These texts are public technical
fixtures. Prose and code use separate sources to expose tokenizer differences.
"""
from pathlib import Path
import random

SIZE = 6500
_WORDS = {
    "de": (("Garten", "Lesesaal", "Archiv", "Bahnhof", "Markt", "Werkstatt", "Ufer", "Museum"),
           ("Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag"),
           ("blaue", "graue", "rote", "gelbe", "schmale", "breite", "glatte")),
    "en": (("garden", "reading room", "archive", "station", "market", "workshop", "riverbank", "museum"),
           ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"),
           ("blue", "grey", "red", "yellow", "narrow", "wide", "smooth")),
}
_PARAGRAPHS = {
    "de": (
        "Am {day} stand auf dem Plan nur ein Wort: {place}. Als die Gruppe kurz nach {hour} Uhr ankam, hatte der Regen aufgehört. Die {adj} Kiste war noch verschlossen. Niemand wusste, wer den Ersatzschlüssel verwahrte.",
        "Im Register erscheint der Eintrag „{place}“ zweimal. Die erste Aufnahme stammt von {year}, die zweite ist undatiert. Erst auf der Rückseite sieht man die {adj} Markierung. Sie fehlt auf der Kopie, die am {day} in der Post lag.",
        "Für den Abschnitt {place} fehlen noch {n} Karten. Bitte lassen Sie die Lieferung am {day} bis {hour} Uhr stehen. Den übrigen Bestand haben wir bereits gezählt und im Protokoll für {year} vermerkt.",
        "»Wirklich erst am {day}?« Sie faltete den Plan auseinander und zeigte auf das {adj} Band. Neben dem Wort {place} hatte jemand mit Bleistift eine kleine {n} notiert. Was sie bedeutete, blieb offen.",
        "Seit {year} sammelt der Verein Aufnahmen aus dem Viertel. Unter dem Stichwort {place} verzeichnet der Katalog {n} Stück. Die jüngste Aufnahme entstand an einem {day}. Man hört Schritte, dann für ein paar Sekunden nur Wind.",
    ),
    "en": (
        "The plan for {day} contained a single word: {place}. When the group arrived shortly after {hour}, the rain had stopped. The {adj} box was still shut. Nobody could remember who held the spare key.",
        "The word “{place}” occurs twice in the register. One photograph dates to {year}; the other has no date. A {adj} mark on the back is missing from the copy that arrived in the post on {day}.",
        "We still need {n} cards for the {place} section. Please leave the delivery in place until {hour} on {day}. We have counted the other boxes and entered them in the records for {year}.",
        "“As late as {day}?” She unfolded the plan and pointed to the {adj} strip. Somebody had written a small {n} in pencil beside the word {place}. Nobody offered an explanation.",
        "The society has collected recordings of the district since {year}. The catalogue lists {n} entries under {place}. The most recent one was recorded on a {day}. Footsteps give way to several seconds of wind.",
    ),
}


def texts(lang, n):
    if lang == "code":
        root = Path(__file__).resolve().parent
        source = "\n\n".join((root / name).read_text(encoding="utf-8") for name in ("common.py", "speed.py"))
        # Fixed starts, independent of n: quick and full share a prefix.
        starts = ((i * 7109) % (len(source) - SIZE + 1) for i in range(n))
        return [source[start:start + SIZE] for start in starts]
    places, days, adjectives = _WORDS[lang]
    docs = []
    for i in range(n):
        rng = random.Random(f"weakbench-drift-1/{lang}/{i}")
        paragraphs = []
        length = 0
        while length < SIZE:
            paragraph = rng.choice(_PARAGRAPHS[lang]).format(
                day=rng.choice(days), place=rng.choice(places), adj=rng.choice(adjectives),
                hour=rng.randint(7, 19), n=rng.randint(8, 64), year=rng.randint(1965, 2015))
            paragraphs.append(paragraph)
            length += len(paragraph) + 2
        docs.append("\n\n".join(paragraphs)[:SIZE])
    return docs
