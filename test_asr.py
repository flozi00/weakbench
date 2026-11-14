"""Hermetic tests for asr.py: normalization, WER, the dataset cache, both modes.

    python3 test_asr.py      (or: pytest test_asr.py)

No network: the dataset viewer and the model are stubbed on localhost.
"""
import base64
import json
import shutil
import sys
import tempfile
import threading
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import asr  # noqa: E402
import common  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="asr-test-"))
common.CACHE = TMP  # never write into the repo


# ---- scoring ---------------------------------------------------------------

def test_strip_accents():
    # NFKD handles the combining accents and the ligatures.
    assert asr._strip_accents("Café") == "Cafe"
    assert asr._strip_accents("señor") == "senor"
    assert asr._strip_accents("ﬁ") == "fi"
    # NFKD does NOT decompose these; unidecode transliterates them, so must we.
    assert asr._strip_accents("København") == "Kobenhavn"
    assert asr._strip_accents("Łódź") == "Lodz"
    assert asr._strip_accents("Þór") == "Thor"
    assert asr._strip_accents("Œuvre") == "OEuvre"
    assert asr._strip_accents("Đakovo") == "Dakovo"
    assert asr._strip_accents("ıstanbul") == "istanbul"


def test_normalize():
    # Case is folded (the deliberate difference from the card).
    assert asr.normalize("HALLO") == "hallo"
    assert asr.normalize("Hallo, Welt!") == "hallo welt"
    assert asr.normalize("  A   B  ") == "a b"
    assert asr.normalize("a\tb\nc") == "a b c"
    assert asr.normalize("") == "" and asr.normalize(None) == ""
    assert asr.normalize([]) == "" and asr.normalize(["nur der erste"]) == "nur der erste"
    assert asr.normalize(["erster", "zweiter"]) == "erster"  # the card takes text[0]
    assert asr.normalize(5) == "5"  # a non-string cell is coerced, not crashed on

    # Umlauts survive, lower and upper; accents do not (the card's unidecode).
    assert asr.normalize("Grüße aus München") == "grüsse aus münchen"
    assert asr.normalize("ÄÖÜ") == "äöü"
    assert asr.normalize("Café") == "cafe"
    assert asr.normalize("señor") == "senor"
    assert asr.normalize("ﬁ") == "fi"
    # ß is "ss" (unidecode's effect), and casefold folds a capital ẞ the same.
    assert asr.normalize("Straße") == "strasse"
    assert asr.normalize("STRASSE") == "strasse"
    assert asr.normalize("ẞ") == "ss"
    # The characters NFKD leaves alone (foreign names in MLS / CommonVoice).
    assert asr.normalize("København") == "kobenhavn"
    assert asr.normalize("Łódź") == "lodz"
    assert asr.normalize("Þór") == "thor"

    # The card's literal replacements: ",," and a double quote.
    assert asr.normalize("Hallo,, Welt") == "hallo welt"
    assert asr.normalize('er sagte "ja"') == "er sagte ja"
    # Bracketed noise, quotes and apostrophes go.
    assert asr.normalize("[lachen] hallo") == "hallo"
    assert asr.normalize("[a [b] c]") == "c"  # non-greedy, like the card
    assert asr.normalize("geht's") == "gehts"
    # Punctuation and hyphens go; \w keeps the underscore (the card's regex too).
    assert asr.normalize("a.b,c!") == "abc"
    assert asr.normalize("gut-fünf") == "gutfünf"
    assert asr.normalize("foo_bar") == "foo_bar"

    # Number words become digits; a standalone article stays a word.
    assert asr.normalize("zweiundzwanzig") == "22"
    assert asr.normalize("einhundertfünfundzwanzig") == "125"
    assert asr.normalize("zweitausenddreihundert") == "2300"
    assert asr.normalize("Das kostet fünf Euro") == "das kostet 5 euro"
    assert asr.normalize("5 Euro") == "5 euro"  # digits pass through
    assert asr.normalize("ein Haus") == "ein haus"
    for article in ("ein", "eine", "einen", "einem", "einer", "eines"):
        assert asr.normalize(article) == article
    # text2num's default threshold is 3.0: an isolated 1 or 2 stays a word.
    assert asr.normalize("zwei Söhne") == "zwei söhne"
    assert asr.normalize("ein bis zwei Tagen") == "ein bis zwei tagen"
    assert asr.normalize("erste") == "erste" and asr.normalize("zweite") == "zweite"
    # ... but a number in a group converts, and 3 and up always convert.
    assert asr.normalize("zwei drei") == "2 3"
    assert asr.normalize("zwei; drei") == "2 3"  # ';' does not break the group
    assert asr.normalize("zwei. Drei") == "zwei 3"  # '.' does
    assert asr.normalize("zwei oder drei") == "zwei oder 3"
    assert asr.normalize("zwei und drei") == "2 und 3"  # a list, not a compound
    assert asr.normalize("ein und zwanzig") == "21"  # a compound
    assert asr.normalize("zwanzig vier") == "20 4"  # no "und", so not 24
    assert asr.normalize("zwei hundert") == "200"
    assert asr.normalize("Sechs Stück") == "6 stück"
    assert asr.normalize("Hunderttausend") == "100000"
    assert asr.normalize("zwei Millionen") == "2000000"
    assert asr.normalize("eine Million") == "eine 1000000"
    assert asr.normalize("drei Millionen fünfhunderttausend") == "3500000"
    # Ordinals convert to their rank (the card's text_to_num appends a period,
    # which the punctuation step then removes).
    assert asr.normalize("dem fünften Kaufmann") == "dem 5 kaufmann"
    assert asr.normalize("die dritte Person") == "die 3 person"
    assert asr.normalize("vom Hundertsten aufs Tausendste") == "vom 100 aufs 1000"
    # Punctuation is read before the number words, so a number with a trailing
    # comma still converts (text2num is punctuation-aware).
    assert asr.normalize("fünf,") == "5"
    assert asr.normalize("(fünf)") == "5"


def test_normalize_fillers():
    # Non-lexical disfluencies are dropped from both sides.
    assert asr.normalize("Oh, das ist gut") == "das ist gut"
    assert asr.normalize("Mm. Jetzt sieht man") == "jetzt sieht man"
    assert asr.normalize("äh das ist") == "das ist"
    assert asr.normalize("hmm") == ""
    assert asr.normalize("Ähm, ja") == "ja"
    assert asr.normalize("Hmm hmm hmm") == ""
    # Real words that can also be discourse particles stay.
    assert asr.normalize("ja das ist gut") == "ja das ist gut"
    assert asr.normalize("nein") == "nein"
    assert asr.normalize("also gut") == "also gut"
    assert asr.normalize("Gena") == "gena"  # a misrecognition, not a filler


def test_de_number():
    assert asr._de_number("") is None and asr._de_number("foo") is None
    assert asr._de_number("hundert") == 100 and asr._de_number("tausend") == 1000
    assert asr._de_number("elf") == 11 and asr._de_number("zwanzig") == 20
    assert asr._de_number("einundzwanzig") == 21
    assert asr._de_number("fünfundzwanzig") == 25
    assert asr._de_number("neunundneunzig") == 99
    assert asr._de_number("einhundert") == 100
    assert asr._de_number("hundertfünf") == 105
    assert asr._de_number("zweihundert") == 200
    assert asr._de_number("sechshundertsechsundsechzig") == 666
    assert asr._de_number("eintausend") == 1000
    assert asr._de_number("tausendfünf") == 1005
    assert asr._de_number("zweiundzwanzigtausend") == 22000
    assert asr._de_number("und") is None
    # Scale words are numbers on their own, and compounds across tokens work.
    assert asr._de_number("Millionen") == 10 ** 6
    assert asr._de_number("Milliarden") == 10 ** 9
    # Ordinals, inflected or not (text2num lemmatizes -ten/-tes/-ter/-tem).
    assert asr._de_number("erste") == 1 and asr._de_number("zweite") == 2
    assert asr._de_number("dritte") == 3 and asr._de_number("dritten") == 3
    assert asr._de_number("fünften") == 5 and asr._de_number("fünftes") == 5
    assert asr._de_number("hundertsten") == 100
    assert asr._de_number("tausendste") == 1000
    # "eine" is an article, not a number word; "einmal" is not a number.
    assert asr._de_number("eine") is None and asr._de_number("einmal") is None
    # The text2num-rs German test vectors, so the port stays honest.
    assert asr._de_number("fünfundachtzig") == 85
    assert asr._de_number("einundachtzig") == 81
    assert asr._de_number("fünfzehn") == 15
    assert asr._de_number("zwei und vierzig") == 42
    assert asr._de_number("einhundertfünfzehn") == 115
    assert asr._de_number("einhundert fünfzehn") == 115
    assert asr._de_number("ein hundert fünfzehn") == 115
    assert asr._de_number("fünfundsiebzigtausend") == 75000
    assert asr._de_number("vierzehntausend") == 14000
    assert asr._de_number("eintausendneunhundertzwanzig") == 1920
    assert asr._de_number("neunzehnhundertdreiundsiebzig") == 1973
    assert asr._de_number(
        "dreiundfünfzig Milliarden zweihundertdreiundvierzigtausendsiebenhundertvierundzwanzig"
    ) == 53000243724
    assert asr._de_number(
        "einundfünfzig Millionen fünfhundertachtundsiebzigtausenddreihundertzwei"
    ) == 51578302


def test_alpha2digit():
    # The threshold only drops isolated 1 and 2; a group always converts.
    assert asr._alpha2digit("zwei") == "zwei"
    assert asr._alpha2digit("drei") == "3"
    assert asr._alpha2digit("zwei drei") == "2 3"
    assert asr._alpha2digit("ein Haus") == "ein Haus"
    assert asr._alpha2digit("einundzwanzig") == "21"
    assert asr._alpha2digit("kein Wort") == "kein Wort"
    # Punctuation is preserved and decides whether numbers are contiguous.
    assert asr._alpha2digit("zwei, drei") == "2, 3"
    assert asr._alpha2digit("zwei. Drei") == "zwei. 3"
    assert asr._alpha2digit("zwei; drei") == "2; 3"
    assert asr._alpha2digit("zwei oder drei") == "zwei oder 3"
    # Compounds across tokens, and the scale words.
    assert asr._alpha2digit("zwei hundert") == "200"
    assert asr._alpha2digit("ein und zwanzig") == "21"
    assert asr._alpha2digit("zwei Millionen") == "2000000"
    assert asr._alpha2digit("drei Millionen fünfhunderttausend") == "3500000"
    # Ordinals, with the period text2num appends.
    assert asr._alpha2digit("dem fünften Kaufmann") == "dem 5. Kaufmann"
    assert asr._alpha2digit("die dritte Person") == "die 3. Person"
    assert asr._alpha2digit("erste") == "erste"


def test_wer():
    assert asr.wer("a b c", "a b c") == {
        "errors": 0, "words": 3, "substitutions": 0, "deletions": 0, "insertions": 0, "rate": 0.0}
    assert asr.wer("a b c", "a x c")["substitutions"] == 1
    assert asr.wer("a b c", "a b")["deletions"] == 1
    assert asr.wer("a b", "a b c")["insertions"] == 1
    assert asr.wer("a b", "a b c")["rate"] == 0.5
    # Contiguous duplicates collapse on both sides (jiwer's standardize).
    assert asr.wer("a a b", "a b")["errors"] == 0
    assert asr.wer("a b", "a a b")["errors"] == 0
    # Normalization applies to both sides.
    assert asr.wer("Hallo, Welt!", "hallo welt")["errors"] == 0
    assert asr.wer("guten tag", "hallo welt")["rate"] == 1.0
    # A filler insertion is not an error; a real-word insertion is.
    assert asr.wer("das ist gut", "oh das ist gut")["errors"] == 0
    assert asr.wer("das ist gut", "gena das ist gut")["errors"] == 1
    # An empty reference has no rate (it is not a 0 % row).
    assert asr.wer("", "x")["rate"] is None and asr.wer("", "x")["words"] == 0


def test_summarize():
    items = [
        {"source": "tuda", "reference": "a b", "hypothesis": "a b", "wer": 0.0,
         "errors": 0, "words": 2, "error": None},
        {"source": "tuda", "reference": "a b", "hypothesis": "x y", "wer": 100.0,
         "errors": 2, "words": 2, "error": None},
        {"source": "cv", "reference": "a", "hypothesis": "", "wer": 100.0,
         "errors": 1, "words": 1, "error": None},
        {"source": "cv", "reference": "", "hypothesis": "", "wer": None,
         "errors": 0, "words": 0, "error": "boom"},
    ]
    summary = asr.summarize(items)
    assert summary["overall"] == {"wer": 60.0, "errors": 3, "words": 5, "items": 4,
                                  "failed": 1, "empty": 1}
    assert summary["by_source"]["tuda"] == {"wer": 50.0, "errors": 2, "words": 4, "items": 2,
                                            "failed": 0, "empty": 0}
    assert summary["by_source"]["cv"]["failed"] == 1
    assert asr.summarize([])["overall"]["wer"] is None


def test_wav_bytes():
    raw = asr._wav_bytes([0.0, 0.5, -0.5, 2.0, -2.0], 16000)
    with wave.open(__import__("io").BytesIO(raw)) as reader:
        assert (reader.getnchannels(), reader.getsampwidth(), reader.getframerate()) == (1, 2, 16000)
        assert reader.getnframes() == 5
    assert raw[:4] == b"RIFF" and raw[8:12] == b"WAVE" and len(raw) == 44 + 10


# ---- the dataset -----------------------------------------------------------

ROWS = [
    {"from": "tuda", "references": "Hallo, Welt!", "audio": {"array": [0.0] * 160, "sampling_rate": 16000}},
    {"from": "cv", "references": "guten tag", "audio": {"array": [0.0] * 160, "sampling_rate": 16000}},
]
calls = []


def viewer_rows(offset, length):
    calls.append((offset, length))
    return {"num_rows_total": len(ROWS),
            "rows": [{"row": ROWS[i]} for i in range(offset, min(offset + length, len(ROWS)))]}


def test_plan():
    saved = (asr._viewer_rows, asr._ROW_COUNT)
    asr._viewer_rows, asr._ROW_COUNT = viewer_rows, None
    try:
        assert asr.plan(2, 7) == [0, 1]  # fewer items than rows -> all of them
        assert asr.plan(1, 7) == asr.plan(1, 7)  # deterministic
        assert asr.plan(1, 7) == sorted(asr.plan(1, 7))
    finally:
        asr._viewer_rows, asr._ROW_COUNT = saved


def test_fetch_item_cached():
    saved = (asr._viewer_rows, asr._ROW_COUNT)
    asr._viewer_rows, asr._ROW_COUNT = viewer_rows, None
    calls.clear()
    try:
        item = asr.fetch_item(0)
        assert item["reference"] == "Hallo, Welt!" and item["source"] == "tuda"
        assert item["seconds"] == 0.01 and item["wav"][:4] == b"RIFF"
        assert calls == [(0, 1)]
        again = asr.fetch_item(0)  # from the cache: no second viewer call
        assert again == item and calls == [(0, 1)]
        assert (TMP / "asr" / asr.DATASET.replace("/", "__") / "0.wav").is_file()
    finally:
        asr._viewer_rows, asr._ROW_COUNT = saved


# ---- the model -------------------------------------------------------------

def serve(transcript="hallo welt"):
    seen = []

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _send(self, code, payload):
            raw = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("content-length") or 0))
            if self.path == "/v1/audio/transcriptions":
                seen.append(("asr", body))
                return self._send(200, {"text": transcript})
            if self.path == "/v1/chat/completions":
                seen.append(("llm", json.loads(body)))
                return self._send(200, {"choices": [{"message": {"content": transcript}}]})
            return self._send(404, {"error": "not found"})

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{server.server_port}", seen, server.shutdown


def test_transcribe_modes():
    url, seen, stop = serve()
    try:
        wav = asr._wav_bytes([0.0] * 160, 16000)
        text, latency = asr.transcribe(url, "m", "asr", wav)
        assert text == "hallo welt" and latency >= 0
        assert b'name="file"; filename="item.wav"' in seen[0][1] and b"RIFF" in seen[0][1]
        text, _ = asr.transcribe(url, "m", "llm", wav, "Transkribiere.")
        assert text == "hallo welt"
        part = seen[1][1]["messages"][0]["content"][1]
        assert part["type"] == "input_audio" and part["input_audio"]["format"] == "wav"
        assert base64.b64decode(part["input_audio"]["data"]) == wav
        try:
            asr.transcribe(url, "m", "nope", wav)
        except ValueError as exc:
            assert "unknown mode" in str(exc)
        else:
            raise AssertionError("an unknown mode must raise")
    finally:
        stop()


def test_clean_transcript():
    assert asr._clean_transcript("  Transkription: Hallo Welt.  ") == "Hallo Welt."
    assert asr._clean_transcript('"hallo"') == "hallo"
    assert asr._clean_transcript(None) == ""


def test_evaluate():
    url, seen, stop = serve()
    saved = (asr._viewer_rows, asr._ROW_COUNT)
    asr._viewer_rows, asr._ROW_COUNT = viewer_rows, None
    try:
        items = list(asr.evaluate(url, "m", "asr", [0, 1]))
        assert [i["row"] for i in items] == [0, 1]
        assert [i["wer"] for i in items] == [0.0, 100.0]
        assert all(i["error"] is None and i["latency_ms"] is not None for i in items)
        summary = asr.summarize(items)
        assert summary["overall"]["wer"] == 50.0 and summary["overall"]["words"] == 4
        # A dataset row that cannot be fetched is an item error, not a dead run.
        def broken(offset, length):
            raise OSError("viewer down")
        shutil.rmtree(TMP / "asr", ignore_errors=True)  # drop the cached rows
        asr._viewer_rows, asr._ROW_COUNT = broken, None
        items = list(asr.evaluate(url, "m", "asr", [0]))
        assert items[0]["error"].startswith("dataset:") and items[0]["wer"] is None
    finally:
        asr._viewer_rows, asr._ROW_COUNT = saved
        stop()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    print(f"all asr tests passed (artifacts in {TMP})")
