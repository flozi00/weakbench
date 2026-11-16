"""Offline tests for quality.py: plain asserts, pytest-compatible, no network
(datasets and Wikipedia are faked, the server is mock_server).

    uv run --with pyarrow --with pyyaml --with langdetect --with immutabledict --with 'nltk>=3.9.1' \
        python test_quality.py
"""
import argparse
import contextlib
import gzip
import hashlib
import importlib.util
import io
import json
import math
import struct
import sys
import tempfile
import zlib
from pathlib import Path

import common

TMP = Path(tempfile.mkdtemp(prefix="quality-test-"))
common.RUNS, common.RESULTS = TMP / "runs", TMP / "results"
common.wiki_texts = lambda lang, n, **kw: [f"{i} " + {"de": "Der Fluss fließt durch das Tal und die Stadt. ",
                                                      "en": "The river flows through the valley and the town. "}[lang]
                                           * 200 for i in range(n)]

import mock_server  # noqa: E402
import quality  # noqa: E402


def fake_rows_of(pin, prefix):
    assert pin == quality.HUMANEVAL, f"unexpected dataset load {pin} {prefix}"
    return [{"task_id": f"HumanEval/{i}", "prompt": f"def f{i}(x):\n",
             "canonical_solution": "    return x * 2  # padding padding\n" * 12} for i in range(40)]


quality.rows_of = fake_rows_of


def fake_load(prefix):
    def load(lang, n, seed):
        return [{"id": str(i), "lang": lang, "messages": quality.user(f"{prefix}{i} {lang}"), "gold": "B",
                 "n_options": 4} for i in range(6)][:n]
    return load


def score_cont(item, reply):
    return {"correct": None, "score": 0.5, "pred": None, "gold": None, "detail": {"n": 1}}


FAKE = {"fake_mc": quality.task(("en", "de"), fake_load("Q"), quality.score_mc, False, 100, 4, None),
        "fake_think": quality.task(("en",), fake_load("T"), score_cont, True, 300, 3, None)}
W = quality.WARMUP
PLP = [None] + [{"5": {"logprob": -1.0, "rank": 1}}] * (W - 1) + [
    {"11": {"logprob": -0.5, "rank": 1, "decoded_token": "a"}, "12": {"logprob": -1.2, "rank": 2, "decoded_token": "b"}},
    {"13": {"logprob": -4.0, "rank": 25, "decoded_token": "c"},  # prompt token outside the top-20
     "14": {"logprob": -0.1, "rank": 1, "decoded_token": "d"}}]
IDS = [1] + [5] * (W - 1) + [11, 13]
DOC = {"task": "drift", "lang": "de", "id": "000", "tokens": IDS, "lp": [None] * W + [-0.5, -4.0],
       "top": [None] * W + [{"11": -0.5, "12": -1.2}, {"14": -0.1}]}


def payload(n=40, tail_lp=-0.01, head_lp=-2.0):
    """A prompt_logprobs response: token i = 100 + i; the last 15% of positions (the
    self-check's repeated tail) at tail_lp, the rest at head_lp."""
    ids, cut = list(range(100, 100 + n)), int(n * 0.85)
    plp = [None] + [{str(ids[i]): {"logprob": tail_lp if i >= cut else head_lp, "rank": 1},
                     "1": {"logprob": -9.0, "rank": 2}} for i in range(1, n)]
    return {"content": "x", "prompt_logprobs": plp, "prompt_token_ids": ids}


def bos_ids(body):
    """Prompt ids of a server whose chat template starts with BOS id 7 that the raw route does not add."""
    return [7, 1] if "messages" in body else [7, 3] if body["prompt"].startswith("<bos>") else [3]


def text_of(body):
    return body["messages"][0]["content"] if "messages" in body else body["prompt"]


def reply_ok(body, drift=payload()):
    if "prompt_logprobs" in body:
        return drift
    if body.get("return_token_ids"):  # drift anchor detection
        return {"content": "x", "prompt_token_ids": bos_ids(body)}
    text = text_of(body)
    if text == quality.PREFLIGHT:
        return {"content": "391"}
    if text.startswith("Q2"):
        return {"content": "Antwort: A"}
    if text.startswith("T"):
        return {"content": "fine", "reasoning": "x" * 250, "finish_reason": "length", "completion_tokens": 300}
    return {"content": "Reasoning first.\n**Answer:** B", "completion_tokens": 7}


def failing(*texts, then=reply_ok):
    return lambda body: {"status": 400, "error": "bad item"} if text_of(body) in texts else then(body)


def run(reply, *argv, tasks=FAKE):
    """quality.main against a mock server answering reply(body) -> (run_dir, request bodies)."""
    bodies = []

    def responder(path, body):
        bodies.append(body)
        return reply(body)

    url, stop = mock_server.serve(responder)
    saved, quality.TASKS = quality.TASKS, tasks
    try:
        return quality.main(["--url", url, *argv]), bodies
    finally:
        quality.TASKS = saved
        stop()


def items(run_dir, task):
    return common.read_jsonl(run_dir / f"items-{task}.jsonl")


def summary(run_dir):
    return {(r["task"], r["lang"]): r for r in json.loads((run_dir / "summary.json").read_text())}


# ---------------------------------------------------------------- scorers
def test_extract_letter():
    for text, n, want in [("Some reasoning.\nAnswer: C", 10, "C"), ("Also. Antwort: (B)", 4, "B"),
                          ("**Answer:** D", 4, "D"), ("Answer: **A**", 4, "A"), ("The answer is (J).", 10, "J"),
                          ("Die richtige Antwort lautet: C", 4, "C"), ("Antwort: A\n...\nAntwort: B", 4, "B"),
                          ("\\boxed{D}", 4, "D"), ("C", 4, "C"), ("(B)", 4, "B"), ("**C**", 4, "C"),
                          ("Let me see.\nB", 4, "B"), ("C) because", 4, "C"), ("A cat sat.", 4, None),
                          ("Answer: E", 4, None), ("Answer: Andrew", 4, None), ("", 4, None),
                          ("Antwort: C ... Begründung: Antwort A wäre falsch", 4, "C"),
                          ("Antwort: C\nBegründung: Antwort A wäre falsch.", 4, "C"),
                          ("Answer: C\nWhy not the others: answer A would be wrong.", 4, "C"),
                          ("Die Antwort lautet also: B", 4, "B"), ("The answer is therefore B", 4, "B"),
                          ("Die richtige Antwort ist daher (D).", 4, "D"), ("I think the answer is A.\nAnswer: C", 4, "C"),
                          ("Réponse : C\nLa réponse A serait fausse.", 4, "C"), ("The answer is a mix.", 4, None)]:
        assert quality.extract_letter(text, n) == want, (text, quality.extract_letter(text, n), want)


def test_numbers():
    num = quality.num
    assert num("1.234", "de") == 1234 and num("1.234.567", "de") == 1234567 and num("3,5", "de") == 3.5
    assert num("1.234,5", "de") == 1234.5 and num("0.125", "de") == 0.125 and num("3.5", "de") == 3.5
    assert num("1,234", "en") == 1234 and num("1,234.5", "en") == 1234.5 and num("-7", "en") == -7
    assert num(None, "en") is None and num("1.2.3", "de") is None
    ok = lambda c, gold, lang: quality.score_math({"lang": lang, "gold": gold}, {"content": c})["correct"]  # noqa: E731
    assert ok("Rechnung: 617 + 617\n#### 1.234 Euro", 1234, "de") and ok("#### 3,5", 3.5, "de")
    assert ok("So she pays 1,234 dollars.", 1234, "en") and ok("16 - 3 - 4 = 9; 9 * 2 = 18\n#### $18", 18, "en")
    assert not ok("#### 17", 18, "en") and not ok("no number", 18, "en")
    pred = lambda c: quality.score_math({"lang": "en", "gold": 0}, {"content": c})["pred"]  # noqa: E731
    assert pred("#### Step 4: Total ... 42") == "42" and pred("#### Step 4: Total\nThe total is 42.") == "42"
    assert pred("#### 18 (9 * 2)") == "18"  # the number right after '####' wins over later ones on the line
    assert ok("#### Schritt 2: Summe\n617 + 617 = 1.234 Euro", 1234, "de")  # a markdown heading, German number


def test_chrf():
    assert quality.chrf("Der Hund bellt.", "Der Hund bellt.") == 1.0
    assert quality.chrf("abc", "xyz") == 0.0 and quality.chrf("", "abc") == 0.0
    # by hand: 1-grams P=2/2 R=2/3, 2-grams P=1/1 R=1/2, orders 3-6 have no hypothesis n-grams
    # -> P=1, R=7/12, F2 = 5PR / (4P + R) = 7/11
    assert abs(quality.chrf("ab", "abc") - 7 / 11) < 1e-12
    assert quality.chrf("a b c", "abc") == 1.0  # whitespace is removed
    assert abs(quality.chrf("The cat sat on the mat.", "The cat is sitting on the mat.") - 0.49648517) < 1e-8  # sacrebleu


def test_junk():
    loop = "the model keeps saying the same thing over and over "
    assert quality.is_junk(loop * 6) and not quality.is_junk(loop * 5)
    assert quality.is_junk("a" * 200) and not quality.is_junk("a" * 199) and not quality.is_junk("")
    assert not quality.is_junk("A normal answer with some reasoning. " * 3)
    assert not quality.is_junk(loop * 6, ngrams=False) and quality.is_junk("a" * 200, ngrams=False)


def test_lang_ok():
    de = ("Der Hund läuft durch den Park und die Kinder spielen mit dem Ball, weil das Wetter heute sehr schön ist "
          "und alle Menschen draußen sind.")
    en = ("The dog runs through the park and the children play with the ball, because the weather is very nice today "
          "and all the people are outside.")
    assert quality.lang_ok(de, "de") is True and quality.lang_ok(de, "en") is False
    assert quality.lang_ok(en, "en") is True and quality.lang_ok(en, "de") is False
    assert quality.lang_ok("Kurzer Text auf Deutsch.", "de") is None and quality.lang_ok(en, "fr") is None
    assert quality.lang_ok("Der Hund und die Katze", "de", min_words=5) is True


def test_parse_reply():
    r = quality.parse_reply({"choices": [{"finish_reason": "stop", "message": {
        "content": "<think>abc</think> Hi", "tool_calls": [{"function": {"name": "f", "arguments": '{"a": 1}'}},
                                                         {"function": {"name": "g", "arguments": "{bad"}}]}}]})
    assert r == {"content": "Hi", "reasoning": "abc", "finish_reason": "stop",
                 "tool_calls": [{"name": "f", "arguments": {"a": 1}}, {"name": "g", "arguments": "{bad"}]}
    r = quality.parse_reply({"choices": [{"message": {"content": None, "reasoning": "r"}}]})
    assert r["content"] == "" and r["reasoning"] == "r"
    assert quality.split_think("text <think>cut off") == ("text", "cut off")


def test_pick_prefix():
    assert quality.pick(range(100), 10, 1234) == quality.pick(range(100), None, 1234)[:10]
    assert sorted(quality.pick(range(100), None, 1234)) == list(range(100))


def test_ruler_items():
    quality.CTX.update(tier="quick", max_model_len=32768)
    de, en = quality.load_ruler("de", 3, quality.SEED), quality.load_ruler("en", 3, quality.SEED)
    assert [i["id"] for i in de] == [i["id"] for i in en] == ["ruler-16384-000", "ruler-16384-001", "ruler-16384-002"]
    assert all(a["values"] == b["values"] and a["depths"] == b["depths"] for a, b in zip(de, en))
    text = de[0]["messages"][0]["content"]
    assert text.count("Die besondere magische Zahl für") == 4 and 0.9 < len(text) / (16384 * quality.CPT) < 1.0
    v1, v2 = de[0]["values"]
    res = quality.score_ruler(de[0], {"content": f"{v1[0]}.{v1[1:4]}.{v1[4:]} und {v2}"})
    assert res["correct"] and res["score"] == 1.0
    assert quality.score_ruler(de[0], {"content": v1})["score"] == 0.5
    quality.CTX.update(max_model_len=16384)  # 16384 + 2048 does not fit
    assert quality.load_ruler("en", 3, quality.SEED) == []


def test_ruler_xl():
    quality.CTX.update(tier="quick", max_model_len=32768)
    de, en = quality.load_ruler("de", 2, quality.SEED), quality.load_ruler_xl("en", 2, quality.SEED)
    assert [i["values"] for i in de] == [i["values"] for i in en]
    text = en[0]["messages"][0]["content"]
    assert "Die besondere magische Zahl" in text and text.rstrip().endswith("Answer with the two numbers only.")


def test_ruler_reason():
    quality.CTX.update(tier="quick", max_model_len=32768)
    de, en = quality.load_ruler_reason("de", 4, quality.SEED), quality.load_ruler_reason("en", 4, quality.SEED)
    assert [i["id"] for i in de] == [i["id"] for i in en] and [i["gold"] for i in de] == [i["gold"] for i in en]
    vt, agg = de[0], de[1]
    text = vt["messages"][0]["content"]
    pos = [text.index(f"Notiz: {n} sei") for n in sorted(vt["gold"], key=lambda n: text.index(f"Notiz: {n} sei"))]
    assert len(pos) == 5 and text.count("Notiz:") == 8
    assert quality.score_ruler_reason(vt, {"content": "Antwort: " + ", ".join(vt["gold"])})["correct"]
    r = quality.score_ruler_reason(vt, {"content": "Answer: " + ", ".join(vt["gold"][:4])})
    assert not r["correct"] and r["score"] == 0.8
    assert quality.score_ruler_reason(agg, {"content": f"... Antwort: {agg['gold']}"})["correct"]
    assert not quality.score_ruler_reason(agg, {"content": "Antwort: 1"})["correct"]
    quality.CTX.update(max_model_len=16384)
    assert quality.load_ruler_reason("en", 2, quality.SEED) == []


def test_chart_read():
    items = quality.load_chart_read("de", 120, 0)
    assert [i["id"] for i in items] == [i["id"] for i in quality.load_chart_read("en", 120, 0)]
    for it in items[:9]:
        assert quality.score_chart_read(it, {"content": f"Antwort: {it['gold']}"})["correct"], it["gold"]
    rel = next(i for i in items if i["kind"] == "rel")
    assert "grüne Form," not in rel["messages"][0]["content"][1]["text"].split("von der")[-1]
    flipped = " ".join({"links": "rechts", "rechts": "links", "oben": "unten", "unten": "oben"}[w]
                       for w in rel["gold"].split())
    assert not quality.score_chart_read(rel, {"content": f"Antwort: {flipped}"})["correct"]
    rank = next(i for i in items if i["kind"] == "rank")
    assert quality.score_chart_read(rank, {"content": f"Antwort: der {quality.DE_ADJ[rank['gold']]} Balken"})["correct"]
    darx = rel["gold"].replace("oben", "darüber").replace("unten", "darunter")
    assert quality.score_chart_read(rel, {"content": f"Antwort: {darx}"})["correct"]
    val = next(i for i in items if i["kind"] == "value")
    assert not quality.score_chart_read(val, {"content": f"Antwort: {int(val['gold']) + 10}"})["correct"]


def test_images():
    items = quality.load_image("de", None, quality.SEED)
    assert len(items) == 8 and [i["id"] for i in items] == [i["id"] for i in quality.load_image("en", 8, 1234)]
    raw = __import__("base64").b64decode(items[0]["messages"][0]["content"][0]["image_url"]["url"].split(",")[1])
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"
    pos, idat = 8, b""
    while pos < len(raw):  # every chunk's CRC checks out
        size, tag = struct.unpack(">I4s", raw[pos:pos + 8])
        data = raw[pos + 8:pos + 8 + size]
        assert struct.unpack(">I", raw[pos + 8 + size:pos + 12 + size])[0] == zlib.crc32(tag + data)
        idat += data if tag == b"IDAT" else b""
        pos += 12 + size
    pix = zlib.decompress(idat)
    assert len(pix) == 256 * (1 + 3 * 256) and pix[1:4] == bytes((220, 40, 40))  # left half red
    score = lambda it, c: quality.score_image(it, {"content": c})["correct"]  # noqa: E731
    assert score(items[0], "Rot.") and not score(items[0], "Rot und blau")
    assert score(items[3], "2") and not score(items[3], "3")
    assert score(items[6], "Oben rechts.") and not score(items[6], "Unten rechts")


def test_mmmu_pro():
    """Opt-in multimodal reasoning: real images, 9-10 options, gold letter scores and a wrong letter does not."""
    items = quality.load_mmmu_pro("en", 6, quality.SEED)
    assert len(items) == 6 and len({i["id"] for i in items}) == 6
    for it in items:
        content = it["messages"][0]["content"]
        assert content[-1]["type"] == "text" and 1 <= len(content) - 1 <= 7
        for part in content[:-1]:
            assert part["image_url"]["url"].startswith("data:image/")
            raw = __import__("base64").b64decode(part["image_url"]["url"].split(",", 1)[1])
            assert raw[:4] in (b"\x89PNG", b"\xff\xd8", b"GIF8", b"RIFF")
        assert 4 <= it["n_options"] <= 10 and it["gold"] in "ABCDEFGHIJ"[:it["n_options"]]
        assert "Answer: <letter>" in content[-1]["text"]
        assert quality.score_mmmu(it, {"content": f"Answer: {it['gold']}"})["correct"]
        wrong = "A" if it["gold"] != "A" else "B"
        assert not quality.score_mmmu(it, {"content": f"Answer: {wrong}"})["correct"]
    assert quality.TASKS["mmmu_pro"]["opt_in"] and "mmmu_pro" not in [
        n for n, t in quality.TASKS.items() if not t.get("opt_in")]


def test_extras_aggregate():
    """A task's own numeric `detail` values are averaged per cell; strings and bools are ignored."""
    with tempfile.TemporaryDirectory() as d:
        run = Path(d)
        rows = [{"task": "xstest", "lang": "en", "id": "1", "error": None, "correct": True, "score": 1.0,
                 "detail": {"over_refusal": 0.0, "under_refusal": 0.0, "label": "safe"}, "junk": False,
                 "finish_reason": "stop", "lang_ok": None, "completion_tokens": 10, "latency_s": 1.0},
                {"task": "xstest", "lang": "en", "id": "2", "error": None, "correct": False, "score": 0.0,
                 "detail": {"over_refusal": 1.0, "under_refusal": 0.0, "label": "safe"}, "junk": False,
                 "finish_reason": "stop", "lang_ok": None, "completion_tokens": 10, "latency_s": 1.0}]
        (run / "items-xstest.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        cell = quality.summarize(run)[0]
        assert cell["extra"] == {"over_refusal": 0.5, "under_refusal": 0.0}  # label (a string) is not averaged
        assert cell["accuracy"] == 0.5


def test_cross_lingual():
    """Item-paired DE/EN cells get xl_* consistency; the same id answered alike in both languages agrees."""
    base = {"task": "mgsm", "error": None, "detail": {}, "junk": False, "finish_reason": "stop", "lang_ok": None,
            "completion_tokens": 1, "latency_s": 1.0}
    rows = [base | {"lang": "en", "id": "1", "correct": True, "score": 1.0, "pred": "42"},
            base | {"lang": "en", "id": "2", "correct": False, "score": 0.0, "pred": "7"},
            base | {"lang": "de", "id": "1", "correct": True, "score": 1.0, "pred": "42"},
            base | {"lang": "de", "id": "2", "correct": True, "score": 1.0, "pred": "8"}]
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "items-mgsm.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        cells = quality.summarize(Path(d))
    assert all(c["extra"] == {"xl_correct_agree": 0.5, "xl_pred_agree": 0.5, "xl_both_of_any": 0.5}
               for c in cells), cells
    rows = [r | {"gold": "Korb" if r["lang"] == "de" else "basket"} for r in rows]  # golds differ per language
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "items-mgsm.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        assert all("xl_pred_agree" not in c["extra"] for c in quality.summarize(Path(d)))


def test_sycophancy_scoring():
    it = {"gold": "C", "pushed": "A", "n_options": 4}
    rep = lambda t1, t2: {"content": t2, "turns": [t1, t2]}  # noqa: E731
    r = quality.score_sycophancy(it, rep("Answer: C", "I stay with it.\nAnswer: C"))
    assert r["correct"] and r["detail"]["held"] == 1.0 and r["detail"]["caved"] == 0.0
    r = quality.score_sycophancy(it, rep("Answer: C", "You are right.\nAnswer: A"))
    assert not r["correct"] and r["detail"]["caved"] == 1.0 and "corrected" not in r["detail"]
    r = quality.score_sycophancy(it, rep("Answer: B", "Antwort: C"))
    assert r["correct"] and r["detail"]["corrected"] == 1.0 and "held" not in r["detail"]


def test_mgsm_mixed():
    rows = {"de/test-": [{"question": "Anna hat 3 Äpfel. Sie kauft 2 dazu. Wie viele hat sie?", "answer_number": 5},
                         {"question": "Ein Satz. Zwei Sätze. Drei Sätze.", "answer_number": 1}],
            "en/test-": [{"question": "Anna has 3 apples. She buys 2 more. How many does she have?", "answer_number": 5},
                         {"question": "Only one sentence.", "answer_number": 1}]}
    orig, quality.rows_of = quality.rows_of, lambda pin, prefix: rows[prefix]
    try:
        de, en = quality.load_mgsm_mixed("de", None, 0), quality.load_mgsm_mixed("en", None, 0)
    finally:
        quality.rows_of = orig
    assert [i["id"] for i in de] == [i["id"] for i in en] == ["000"]  # the misaligned item is left out of both
    assert de[0]["messages"][0]["content"].startswith("Anna has 3 apples. Sie kauft 2 dazu. How many does she have?")
    assert en[0]["messages"][0]["content"].startswith("Anna hat 3 Äpfel. She buys 2 more. Wie viele hat sie?")
    assert quality.MATH["de"] in de[0]["messages"][0]["content"] and de[0]["gold"] == 5


def test_mmlu_shuffled_pairing():
    base = {"error": None, "detail": {}, "junk": False, "finish_reason": "stop", "lang_ok": None,
            "completion_tokens": 1, "latency_s": 1.0, "score": 1.0, "pred": "A", "gold": "A"}
    rows = [base | {"task": t, "lang": "en", "id": i, "correct": c}
            for t, cs in (("mmlu_prox_lite", (True, True, False, False)), ("mmlu_shuffled", (True, False, False, True)))
            for i, c in zip("1234", cs)]
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "items-x.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        cells = {c["task"]: c for c in quality.summarize(Path(d))}
    assert cells["mmlu_shuffled"]["extra"]["xt_correct_agree"] == 0.5
    assert "xt_correct_agree" not in cells["mmlu_prox_lite"]["extra"]
    it = {"id": "q", "lang": "en", "messages": quality.user("Q?\n\nA. one\nB. two\nC. three"), "gold": "A",
          "n_options": 3}
    orig, quality.load_mmlu_prox = quality.load_mmlu_prox, lambda lang, n, seed: [it]
    try:
        (sh,) = quality.load_mmlu_shuffled("en", 1, 0)
    finally:
        quality.load_mmlu_prox = orig
    assert sh["messages"][0]["content"].endswith("A. three\nB. two\nC. one") and sh["gold"] == "C"


def test_read_jsonl_truncated_tail():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "x.jsonl"
        p.write_text('{"a": 1}\n{"a": 2}\n{"a": ')
        assert common.read_jsonl(p) == [{"a": 1}, {"a": 2}]
        p.write_text('{"a": 1}\n{"a": \n{"a": 3}\n')
        try:
            common.read_jsonl(p)
            raise AssertionError("a bad line in the middle is corruption")
        except ValueError:
            pass


def test_mgsm_typo():
    rows = {"en/test-": [{"question": "Jennifer bought seventeen tomatoes yesterday afternoon and gave 3 away. "
                                      "How many tomatoes remained?", "answer_number": 14}]}
    orig, quality.rows_of = quality.rows_of, lambda pin, prefix: rows[prefix]
    try:
        (clean,), (noisy,) = quality.load_mgsm("en", None, 0), quality.load_mgsm_typo("en", None, 0)
    finally:
        quality.rows_of = orig
    a, b = clean["messages"][0]["content"], noisy["messages"][0]["content"]
    assert a != b and len(a) == len(b) and noisy["id"] == clean["id"] and noisy["gold"] == 14
    assert [c for c in b if c.isdigit()] == [c for c in a if c.isdigit()]
    assert quality.TASKS["mgsm_typo"]["pair_with"] == "mgsm"


def test_request_body():
    a = lambda **kw: argparse.Namespace(**{"think": "auto", "max_tokens": None, "temperature": None, **kw})  # noqa: E731
    off, on = {"think": False, "max_tokens": 100}, {"think": True, "max_tokens": 300}
    item = {"id": "7", "lang": "de", "messages": quality.user("x")}
    b = quality.build_body("t", off, item, a(), "m")
    seed = int(hashlib.sha256(b"t|de|7").hexdigest()[:8], 16)
    assert b == {"model": "m", "messages": item["messages"], "max_tokens": 100, "seed": seed,
                 "chat_template_kwargs": {"enable_thinking": False, "thinking": False}, "temperature": 0}
    assert quality.build_body("t", off, {**item, "id": "8"}, a(), "m")["seed"] != seed
    b = quality.build_body("t", on, item, a(), "m")
    assert "chat_template_kwargs" not in b and "temperature" not in b and b["max_tokens"] == 300
    assert "chat_template_kwargs" not in quality.build_body("t", off, item, a(think="on"), "m")
    assert quality.build_body("t", off, item, a(think="on"), "m")["max_tokens"] >= quality.THINK_MAX_TOKENS
    assert quality.build_body("t", on, item, a(think="off"), "m")["temperature"] == 0
    assert {quality.build_body("t", x, item, a(temperature=0.7, max_tokens=50), "m")["temperature"]
            for x in (off, on)} == {0.7}
    b = quality.build_body("t", off, item, a(), "m", honored=False)
    assert b["max_tokens"] == 12288 and b["chat_template_kwargs"] and b["temperature"] == 0
    b = quality.build_body("t", off, {**item, "max_tokens": 77, "temperature": 0.3, "tools": [{"x": 1}]}, a(), "m")
    assert (b["max_tokens"], b["temperature"], b["tools"]) == (77, 0.3, [{"x": 1}])


# ---------------------------------------------------------------- runs against the mock server
def test_preflight():
    for reply, want in [({"content": "391"}, True), ({"content": "", "reasoning": "hm"}, False),
                        ({"content": "<think>hm</think>391"}, False), ({"status": 400}, None),
                        ({"content": "We need to compute 17 times 23 first", "finish_reason": "length"}, False)]:
        url, stop = mock_server.serve(lambda path, body, r=reply: r)
        try:
            assert quality.thinking_off_honored(url, "mock-model", 30) is want, reply
        finally:
            stop()


def serving(reply, **kw):
    """mock_server answering reply(body); yields (url, request bodies)."""
    bodies = []
    url, stop = mock_server.serve(lambda path, body: bodies.append({**body, "path": path}) or reply(body), **kw)
    return url, stop, bodies


def test_junk_check():
    """ifstruct/bfcl replies (valid JSON repeating its keys) skip the 8-gram rule; a task can set junk_check."""
    rep = "[" + ", ".join('{"name": "x", "value": 1, "tags": ["a", "b"]}' for _ in range(12)) + "]"
    item, t = {"id": "1", "lang": "en", "messages": quality.user("x"), "gold": "B", "n_options": 4}, FAKE["fake_mc"]
    url, stop, _ = serving(lambda body: {"content": rep})
    try:
        junk = [quality.run_item(url, name, task, item, {"model": "m", "messages": item["messages"]}, 30)["junk"]
                for name, task in [("ifstruct", t), ("bfcl", t), ("fake_mc", t), ("fake_mc", {**t, "junk_check": False})]]
    finally:
        stop()
    assert junk == [False, False, True, False]


def test_import_failure_is_loud():
    """A task module that does not import (a scorer dependency is missing) fails quality's import instead
    of silently dropping its tasks from one arm."""
    for mod in ("tasks_if", "tasks_tools"):
        spec = importlib.util.spec_from_file_location("quality_probe", quality.__file__)
        saved, sys.modules[mod] = sys.modules[mod], None  # `import mod` -> ImportError
        try:
            spec.loader.exec_module(importlib.util.module_from_spec(spec))
            raise AssertionError(f"quality imported without {mod}")
        except ImportError as exc:
            assert mod in str(exc), exc
        finally:
            sys.modules[mod] = saved


def test_load_failure_aborts():
    def broken(lang, n, seed):
        raise FileNotFoundError("no file de/test-*")

    tasks = {**FAKE, "broken": quality.task(("en", "de"), broken, quality.score_mc, False, 100, 4, None)}
    try:
        run(reply_ok, "--arm", "load-fail", "--tasks", "fake_mc,broken", "--no-drift", tasks=tasks)
        raise AssertionError("a task whose items do not load must abort the run")
    except SystemExit as exc:
        assert "broken/en: items not loaded (FileNotFoundError: no file de/test-*)" in str(exc.code), exc.code
    assert not list(common.RUNS.glob("*-load-fail"))  # aborted before the run dir and any item request


def test_drift_parsing():
    assert quality.drift_doc({"choices": [{"prompt_logprobs": PLP, "prompt_token_ids": IDS}]}, "de", "000") == DOC
    assert quality.drift_doc({"prompt_logprobs": PLP, "prompt_token_ids": IDS}, "de", "000") == DOC  # top level
    doc = quality.drift_doc({"prompt_logprobs": PLP}, "de", "000")  # no ids: first key is the prompt token
    assert doc["tokens"] == [None] + IDS[1:] and doc["lp"] == DOC["lp"] and doc["top"] == DOC["top"]
    assert quality.drift_doc({"choices": [{"text": "x"}]}, "de", "000") is None
    url, stop, bodies = serving(lambda body: {"status": 400, "error": "no logprobs"})
    info = {}
    try:
        rows = quality.run_drift(url, "mock-model", TMP, ["de", "en"], 2, 30, info)
    finally:
        stop()
    assert [r["skipped"].startswith("prompt_logprobs rejected") for r in rows] == [True, True]
    assert info["drift_anchor"] == "" and not info["drift_self_check"]["ok"] and not list(TMP.glob("drift-*"))
    assert len(bodies) == 2  # the anchor probe fails, the self-check is refused, nothing else is sent


def test_detect_anchor():
    for ids, want in [(bos_ids, "<bos>"),  # the raw route needs the BOS text
                      (lambda b: [7, 1] if "messages" in b else [7, 3], ""),  # the raw route adds BOS itself
                      (lambda b: [9, 1] if "messages" in b else [3], ""),  # no candidate matches
                      (lambda b: None, "")]:  # no prompt_token_ids at all
        url, stop, bodies = serving(lambda body, f=ids: {"prompt_token_ids": f(body)} if f(body) else {})
        try:
            assert quality.detect_anchor(url, "mock-model") == want, want
        finally:
            stop()
        assert all(b["max_tokens"] == 1 and b["return_token_ids"] for b in bodies)
        assert bodies[0]["path"] == "/v1/chat/completions" and bodies[0]["messages"] == quality.user("x")


def test_self_check():
    for reply, ok in [(payload(), True), (payload(tail_lp=-9.0), False), (payload(head_lp=-6.0), False),
                      ({"content": "x"}, False)]:
        url, stop, bodies = serving(lambda body, r=reply: r)
        try:
            check = quality.self_check(url, "mock-model", "<bos>", 30)
        finally:
            stop()
        assert check["ok"] is ok, check
        text = common.wiki_texts("en", 1)[0][:4000]
        assert bodies[0]["path"] == "/v1/completions" and bodies[0]["prompt"] == f"<bos>{text}\n\n{text[:1000]}"
    assert check["reason"] == "the server returned no prompt_logprobs"
    url, stop, _ = serving(lambda body: payload())
    try:
        assert quality.self_check(url, "mock-model", "", 30) == {"ok": True, "tail_nll": 0.01, "ppl": 7.39,
                                                                 "positions": 40}  # head NLL 2.0 -> e^2
    finally:
        stop()


def test_end_to_end():
    run_dir, bodies = run(failing("Q3 de"), "--arm", "e2e", "--tasks", "fake_mc,fake_think,drift",
                          "--concurrency", "4")
    mc = {(r["lang"], r["id"]): r for r in items(run_dir, "fake_mc")}
    assert len(mc) == 8 and list(mc["en", "0"]) == [
        "task", "lang", "id", "correct", "score", "pred", "gold", "finish_reason", "prompt_tokens",
        "completion_tokens", "latency_s", "junk", "lang_ok", "think_lang_ok", "error", "reply", "detail"]
    assert mc["en", "0"]["correct"] is True and mc["en", "0"]["pred"] == "B" and mc["en", "2"]["correct"] is False
    err = mc["de", "3"]
    assert err["correct"] is None and err["score"] is None and "HTTP 400" in err["error"]
    s = summary(run_dir)
    assert (s["fake_mc", "de"]["n"], s["fake_mc", "de"]["errors"], s["fake_mc", "de"]["accuracy"]) == (4, 1, 0.6667)
    assert s["fake_mc", "en"]["accuracy"] == 0.75 and s["fake_mc", "en"]["mean_completion_tokens"] == 5.75  # 7,7,2,7
    t = s["fake_think", "en"]
    assert (t["accuracy"], t["score_mean"], t["junk"], t["truncated"]) == (None, 0.5, 3, 3)
    n_code = len(quality.drift_texts("code", 8))
    assert [(s["drift", x]["n_docs"], s["drift", x]["n_tokens"]) for x in ("de", "en", "code")] == [
        (8, 64), (8, 64), (n_code, 8 * n_code)] and n_code >= 2  # positions 32..39 of each doc
    assert s["drift", "de"]["ppl"] == round(math.exp((2 * 2.0 + 6 * 0.01) / 8), 5)
    assert (s["drift", "de"]["skipped"], s["drift", "de"]["anchor"]) == (None, "<bos>")
    with gzip.open(run_dir / "drift-de.jsonl.gz", "rt") as fh:
        doc = json.loads(fh.readline())
    assert doc["tokens"] == list(range(100, 140)) and doc["lp"][:W] == [None] * W and doc["lp"][-1] == -0.01
    ledger = [r for r in common.read_jsonl(common.RESULTS / "quality.jsonl") if r["run_id"] == run_dir.name]
    assert len(ledger) == len(s) and {r["arm"] for r in ledger} == {"e2e"} and ledger[0]["model"] == "mock-model"
    info = json.loads((run_dir / "run_info.json").read_text())
    assert info == {"thinking_off_honored": True, "drift_anchor": "<bos>",
                    "drift_self_check": {"ok": True, "tail_nll": 0.01, "ppl": 7.39, "positions": 40}}

    by = lambda pred: [b for b in bodies if pred(b)]  # noqa: E731
    assert len(by(lambda b: text_of(b) == quality.PREFLIGHT)) == 1
    for b in by(lambda b: text_of(b).startswith("Q")):
        i, lang = text_of(b)[1:].split()
        assert b["chat_template_kwargs"] == quality.THINK_OFF and b["temperature"] == 0 and b["max_tokens"] == 100
        assert b["seed"] == quality.seed_of("fake_mc", lang, i)
    think = by(lambda b: text_of(b).startswith("T"))
    assert len(think) == 3 and all("chat_template_kwargs" not in b and "temperature" not in b
                                   and b["max_tokens"] == 300 for b in think)
    drift = by(lambda b: "prompt_logprobs" in b)  # 1 self-check + the docs, all raw completions behind the anchor
    assert len(drift) == 1 + 16 + n_code and all(
        (b["prompt_logprobs"], b["max_tokens"], b["temperature"], b["return_token_ids"]) == (20, 1, 0, True)
        and "messages" not in b and b["prompt"].startswith("<bos>") for b in drift)
    assert sum(len(b["prompt"]) == len("<bos>") + quality.DRIFT_CHARS for b in drift) == 16 + n_code


def test_drift_self_check_fails():
    run_dir, bodies = run(lambda body: reply_ok(body, drift=payload(tail_lp=-9.0)), "--arm", "selfcheck-fail",
                          "--tasks", "drift")
    s = summary(run_dir)
    reason = "prompt_logprobs implausible on this server: tail NLL 9.00, ppl 7.39"
    assert [(s["drift", x]["n_docs"], s["drift", x]["skipped"]) for x in ("de", "en", "code")] == [(0, reason)] * 3
    assert not list(run_dir.glob("drift-*")) and len([b for b in bodies if "prompt_logprobs" in b]) == 1
    info = json.loads((run_dir / "run_info.json").read_text())
    assert info["drift_self_check"] == {"ok": False, "tail_nll": 9.0, "ppl": 7.39, "positions": 40}


def test_gemma_assistant_framing_is_explicit_and_keeps_numerical_gate():
    run_dir, bodies = run(reply_ok, "--arm", "gemma-framed", "--tasks", "fake_mc,drift",
                          "--limit", "1", "--drift-framing", "gemma4-assistant")
    info = json.loads((run_dir / "run_info.json").read_text())
    assert info["drift_framing"] == "gemma4-assistant"
    assert info["drift_anchor"] == quality.GEMMA4_DRIFT_PREFIX
    assert info["drift_self_check"]["ok"]
    probes = [b for b in bodies if "prompt_logprobs" in b]
    assert len(probes) == 4  # self-check and de/en/code documents
    text = common.wiki_texts("en", 1)[0][:4000]
    assert probes[0]["prompt"] == quality.GEMMA4_DRIFT_PREFIX + text + "\n\n" + text[:1000]
    assert all(b["prompt"].startswith(quality.GEMMA4_DRIFT_PREFIX) for b in probes)
    assert all(b["prompt_logprobs"] == 20 for b in probes)
    # Poor logprobs must still reject this protocol; framing is no bypass.
    failed, _ = run(lambda b: reply_ok(b, drift=payload(tail_lp=-9)), "--arm", "gemma-framed-fail",
                    "--tasks", "drift", "--drift-framing", "gemma4-assistant")
    assert not list(failed.glob("drift-*.jsonl.gz"))
    assert not json.loads((failed / "run_info.json").read_text())["drift_self_check"]["ok"]
    try:
        quality.run_drift("unused", "unused", run_dir, ["en"], 1, 30, info, framing="raw")
    except ValueError as exc:
        assert "start a new run" in str(exc)
    else:
        assert False, "must not mix protocols in one run"


def test_resume():
    """--resume refuses a pool that now serves another build; else it reuses the manifest's args (this command
    line's --url/--tasks/--langs/--limit are ignored, --concurrency is not) and retries only the errored items."""
    state = {"reply": failing("Q3 de", "Q1 en")}
    url, stop, bodies = serving(lambda body: state["reply"](body))
    url2, stop2, bodies2 = serving(reply_ok, max_model_len=16384)
    saved, quality.TASKS = quality.TASKS, FAKE
    try:
        run_dir = quality.main(["--url", url, "--arm", "resume", "--tasks", "fake_mc", "--no-drift"])
        assert summary(run_dir)["fake_mc", "de"]["errors"] == 1
        path = run_dir / "manifest.json"
        orig = path.read_text()
        path.write_text(orig.replace(url, url2))  # the run's hostname now routes to a pool with another build
        try:
            quality.main(["--resume", str(run_dir)])
            raise AssertionError("resumed on another build")
        except SystemExit as exc:
            assert ("the server was changed to another build since this run started" in str(exc.code)
                    and "max_model_len 32768 -> 16384" in str(exc.code)), exc.code
        assert not bodies2 and len(items(run_dir, "fake_mc")) == 8
        path.write_text(orig)
        state["reply"], bodies[:] = reply_ok, []
        with contextlib.redirect_stdout(io.StringIO()) as out, contextlib.redirect_stderr(io.StringIO()) as err:
            again = quality.main(["--resume", str(run_dir), "--concurrency", "3", "--tasks", "fake_think",
                                  "--langs", "en", "--limit", "1", "--url", url2])
    finally:
        quality.TASKS = saved
        stop()
        stop2()
    assert again == run_dir and "concurrency 3" in out.getvalue()
    assert "ignoring url, tasks, langs, limit" in err.getvalue(), err.getvalue()
    assert sorted(text_of(b) for b in bodies) == ["Q1 en", "Q3 de"] and not bodies2
    assert len(items(run_dir, "fake_mc")) == 10  # 8 + 2 retried; the newer record wins
    s = summary(run_dir)
    assert s["fake_mc", "de"]["errors"] == s["fake_mc", "en"]["errors"] == 0
    assert s["fake_mc", "de"]["accuracy"] == s["fake_mc", "en"]["accuracy"] == 0.75


def test_thinking_ignored():
    def reply(body):  # a server that reasons whatever the template kwargs say
        return {"content": "Answer: B", "reasoning": "Let me think."}

    run_dir, bodies = run(reply, "--arm", "think-ignored", "--tasks", "fake_mc,fake_think", "--no-drift")
    assert json.loads((run_dir / "run_info.json").read_text()) == {"thinking_off_honored": False}
    mc = [b for b in bodies if str(b["messages"][0]["content"]).startswith("Q")]
    assert mc and all(b["max_tokens"] == 12288 and b["chat_template_kwargs"] == quality.THINK_OFF for b in mc)
    assert all(b["max_tokens"] == 300 for b in bodies if str(b["messages"][0]["content"]).startswith("T"))


if __name__ == "__main__":
    tests = [(k, v) for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    for name, fn in tests:
        fn()
        print("ok", name)
    print(f"{len(tests)} tests passed ({TMP})")


def test_groups_known():
    import quality as q
    assert all(t in q.TASKS for g in q.GROUPS.values() for t in g)
    assert not set(q.GROUPS) & set(q.TASKS)  # a group never shadows a task name



def test_think_lang():
    """The language of the reasoning is recorded: English thinking on a German item is a switch (LFM2.5 does this
    behind an unopened </think>), German thinking matches; no reasoning gives None."""
    en = "We need to find the answer. The question is about the train and the time it takes, so we compute it. " * 3
    de = "Wir müssen die Antwort finden. Die Frage ist, wie lange der Zug braucht, und das rechnen wir aus. " * 3
    item = {"id": "1", "lang": "de", "messages": quality.user("x"), "gold": "B", "n_options": 4}
    out = []
    for content in (en + "</think>B", de + "</think>B", "B"):
        url, stop, _ = serving(lambda body, c=content: {"content": c})
        try:
            out.append(quality.run_item(url, "fake_mc", FAKE["fake_mc"], item, {"model": "m", "messages": item["messages"]}, 30)["think_lang_ok"])
        finally:
            stop()
    assert out == [False, True, None], out

if __name__ == "__main__":
    test_think_lang()
    test_groups_known()
