"""lang_id on the real pinned Belebele data (test_quality.py fakes dataset loads, so this lives apart)."""
import sys

sys.path.insert(0, ".")
import quality  # noqa: E402


def test_lang_id():
    xs = quality.load_lang_id("en", 40, 0)
    assert len(xs) == 40 and len({x["code"] for x in xs}) == 20 and len({x["messages"][0]["content"] for x in xs}) == 40
    x = next(y for y in xs if y["code"] == "slk_Latn")
    assert quality.score_lang_id(x, {"content": "Answer: Slovak"})["correct"]
    assert not quality.score_lang_id(x, {"content": "Answer: Czech or Slovak"})["correct"]
    assert not quality.score_lang_id(x, {"content": "Answer: Czech"})["correct"]
    assert quality.score_lang_id(x, {"content": "Answer: It's Slovak, not Czech"})["correct"]
    assert min(len(y["messages"][0]["content"]) for y in xs) > 250


if __name__ == "__main__":
    test_lang_id()


def test_xl_faithful():
    for lang in ("de", "en"):
        xs = quality.load_xl_faithful(lang, None, 0)
        assert len(xs) == 80 and sum(x["changed"] for x in xs) == 40 and len({x["messages"][0]["content"] for x in xs}) == 80
        assert not any("COVID-2" in x["messages"][0]["content"] for x in xs)
        x = xs[1]
        assert quality.score_xl_faithful(x, {"content": f"Answer: {x['gold']}"})["correct"]


if __name__ == "__main__":
    test_xl_faithful()
