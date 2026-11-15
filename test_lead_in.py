"""Every generated task whose gold is a short value must accept it behind a lead-in ("Answer: The answer is 42." /
"Antwort: Die Lösung lautet Museen."): scorers that read the first word once saw "the" or "die". Full-sentence answers
(grammar fixes, rewrites) are exempt: there the sentence itself is the answer."""
import importlib
import sys

sys.path.insert(0, ".")
SENTENCE_TASKS = {"de_separable", "de_grammar_fix", "en_grammar_fix", "de_direct_speech"}


def test_lead_in():
    bad = []
    for mod in ("tasks_novel", "tasks_lang"):
        for name, t in importlib.import_module(mod).TASKS.items():
            if t.get("needs_judge") or name in SENTENCE_TASKS:
                continue
            for lang in t["langs"][:2]:
                for it in t["load"](lang, t["counts"]["quick"][lang] or 8, 0)[:8]:
                    g = it.get("gold")
                    if not isinstance(g, (str, int, float)) or isinstance(g, bool):
                        continue
                    gs = f"{g:g}" if isinstance(g, float) else str(g)
                    gs = gs.replace(".", ",") if isinstance(g, float) and lang == "de" else gs
                    plain, lead = (f"Antwort: {gs}", f"Antwort: Die Lösung lautet {gs}.") if lang == "de" else \
                        (f"Answer: {gs}", f"Answer: The answer is {gs}.")
                    rep = lambda c: {"content": c, "reasoning": "", "tool_calls": [], "finish_reason": "stop"}  # noqa: E731
                    if t["score"](it, rep(plain))["correct"] and not t["score"](it, rep(lead))["correct"]:
                        bad.append((name, lang, gs))
    assert not bad, bad[:10]


def test_article_frames():
    import tasks_lang as tl
    x = next(y for y in tl.load_de_genus_meaning("de", None, 0) if y["gold"] == "die")
    for c in ("Der richtige Artikel ist „die“.", "Der passende Artikel ist die.", "Der gesuchte Artikel: die", "Die korrekte Antwort: die"):
        assert tl.score_de_genus_meaning(x, {"content": c})["correct"], c


if __name__ == "__main__":
    test_lead_in()
    test_article_frames()
