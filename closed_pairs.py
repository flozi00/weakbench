"""Paired accuracy from redacted item verdicts of a frozen private release."""
from hashlib import sha256

METRICS = ("contrast_both_correct", "bilingual_item_correct", "bilingual_contrast_correct")


def item_id(revision, family, ordinal):
    return sha256(f"{revision}/{family}/{ordinal}".encode()).hexdigest()[:32]


def tally(groups, verdicts):
    passed = scored = 0
    for group in groups:
        if all(key in verdicts for key in group):
            scored += 1
            passed += all(verdicts[key] for key in group)
    return {"passed": passed, "scored": scored} if scored else None


def checks(cells, revision, counts, paired):
    """Only complete groups of error-free boolean verdicts have an eligible denominator."""
    result = {}
    for family in {task for task, _ in cells} & counts.keys():
        ids = [item_id(revision, family, i) for i in range(counts[family])]
        verdicts = {(lang, str(row["id"])): row["correct"] for lang in ("de", "en")
                    for row in cells.get((family, lang), ()) if not row["error"] and type(row["correct"]) is bool}
        both_languages = tally([(("de", uid), ("en", uid)) for uid in ids], verdicts)
        twins = list(zip(ids[::2], ids[1::2])) if family in paired else []
        both_pairs = tally([tuple((lang, uid) for lang in ("de", "en") for uid in pair) for pair in twins], verdicts)
        for lang in ("de", "en"):
            same_language = tally([((lang, first), (lang, second)) for first, second in twins], verdicts)
            summaries = dict(zip(METRICS, (same_language, both_languages, both_pairs)))
            if any(summaries.values()):
                result[family, lang] = {metric: summary for metric, summary in summaries.items() if summary}
    return result
