"""German ASR quality eval — word error rate on real speech.

One suite for every speech model this cluster serves:

  * an ASR server (Parakeet, or vLLM's Whisper) at
    ``POST /v1/audio/transcriptions`` (multipart, ``{"text": ...}`` back);
  * an audio-enabled LLM (Gemma 3n / Gemma 4 E4B, Qwen-Omni, …) at
    ``POST /v1/chat/completions`` with an OpenAI ``input_audio`` part.

The data is ``flozi00/asr-german-mixed-evals`` (the German mixed test set:
Tuda-De, multilingual LibriSpeech and CommonVoice 19), read over the public
Hugging Face dataset-viewer API — stdlib only, no ``datasets``/``pyarrow``.
The viewer returns each row's audio as a float array; it is converted to a
16 kHz mono PCM WAV here and cached under ``BENCH_CACHE/asr/``.

Scoring follows the dataset card's recipe (``normalize_text`` + jiwer's
``wer_standardize_contiguous``) as closely as the standard library allows:
umlauts are preserved, accents are stripped, German number words become
digits, punctuation and bracketed noise are removed, case is ignored, and
contiguous duplicate words are collapsed before the word-level Levenshtein
distance. Two deliberate differences from the card: case is folded (the card's
published table is only reproducible case-insensitively — whisper-large-v3
scores 16.8 % WER case-sensitively and 4.1 % case-insensitively on the same
100 rows), and non-lexical fillers (``äh``, ``ähm``, ``hm``, ``mm``, ``oh``, …)
are dropped from both sides, so a model is not penalized for emitting a
disfluency the reference does not carry. Real words that can also act as
discourse particles (``ja``, ``nein``, ``also``, ``halt``) are kept.

The number words are a port of text2num's German interpreter, including its
default threshold of 3.0 (an isolated ``zwei`` stays a word, ``zwei drei``
becomes ``2 3``) and its ordinals. On all 4,312 references of the dataset it
normalizes identically to the card's ``normalize_text``.

``unidecode`` is approximated by NFKD plus a transliteration table for the
Latin characters NFKD does not decompose (``ø``, ``æ``, ``ł``, ``þ``, …), which
the dataset's MLS and CommonVoice rows need for foreign names.

CLI (stdlib only, from a machine that reaches the model and Hugging Face):

    python3 asr.py --url https://asr.example.org \
        --model my-asr-model --mode asr --items 50

    python3 asr.py --url https://llm.example.org \
        --model my-audio-llm --mode llm --items 50

A server can run the same ``evaluate()`` in-process
(api/app.py), so a UI run and a CLI run score identically.
"""
import argparse
import array
import base64
import gzip
import io
import json
import queue
import random
import re
import sys
import threading
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import uuid
import wave
from itertools import groupby

import common

DATASET = "flozi00/asr-german-mixed-evals"
CONFIG = "default"
SPLIT = "train"
VIEWER = "https://datasets-server.huggingface.co"
# Fixed so two arms (A/B) score the same rows. The dataset is shuffled across
# its three sources, so a seeded sample is representative.
SEED = 20261020
DEFAULT_ITEMS = 50
MAX_ITEMS = 200
# The audio cache may be a small tmpfs (/dev/shm, ~64 MiB): keep it
# well under that. 50 items are ~18 MiB of 16 kHz PCM.
MAX_CACHE_BYTES = 32 << 20
DEFAULT_PROMPT = (
    "Transcribe the following German audio verbatim. Reply with the "
    "transcription only — no introduction, no commentary, no translation."
)

_UMLAUTS = (("ä", "ae"), ("ö", "oe"), ("ü", "ue"),
            ("Ä", "Ae"), ("Ö", "Oe"), ("Ü", "Ue"))
_ROW_COUNT = None


# ---- scoring ---------------------------------------------------------------

# unidecode's effect for the Latin characters NFKD does NOT decompose. The
# dataset's MLS / CommonVoice rows carry foreign names (København, Łódź, Þór),
# and unidecode turns those into ASCII while NFKD leaves them alone. Everything
# NFKD does handle (é, ñ, ﬁ) is left to _strip_accents.
_TRANSLIT = str.maketrans({
    "ø": "o", "Ø": "O", "æ": "ae", "Æ": "AE", "œ": "oe", "Œ": "OE",
    "ł": "l", "Ł": "L", "đ": "d", "Đ": "D", "ð": "d", "Ð": "D",
    "þ": "th", "Þ": "Th", "ı": "i", "ħ": "h", "Ħ": "H", "ŧ": "t",
    "Ŧ": "T", "ŋ": "n", "Ŋ": "N", "ĸ": "k", "ſ": "s",
})


def _strip_accents(text):
    """unidecode's effect for Latin scripts: é -> e, ñ -> n, ﬁ -> fi, ø -> o."""
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return stripped.translate(_TRANSLIT)


# Non-lexical German disfluencies. The references are read speech and carry
# none; a model that emits them is not saying a word the reference lacks, so
# they are dropped from both sides before scoring — the same convention the
# Whisper normalizer uses for "uh"/"um". Real words that can also act as
# discourse particles ("ja", "nein", "also", "halt") are deliberately NOT
# here: dropping them would hide a real substitution.
FILLERS = frozenset({
    "äh", "ähm", "ah", "ahm", "eh", "ehm", "hm", "hmm", "mm", "mhm",
    "öh", "öhm", "uh", "uhm", "ähem", "hmmm", "mhh", "ooh", "oh", "aha",
})


# ---- German number words ---------------------------------------------------
#
# The dataset card normalizes with ``text_to_num.alpha2digit(text, "de")``,
# whose default threshold is 3.0: an *isolated* cardinal or ordinal below 3
# stays a word ("zwei Söhne" keeps "zwei"), while a number in a group converts
# ("zwei drei" -> "2 3", "zwei Millionen" -> "2000000"). Reproducing that needs
# text2num's positional digit buffer and its contiguity tracker, so this is a
# port of the German interpreter (text2num-rs, MIT) rather than a word map.

_TENS_FLAG = 1


class _Digits:
    """text2num's DigitString: a positional digit buffer, most significant first."""

    def __init__(self):
        self.buf = []
        self.leading_zeroes = 0
        self.frozen = False
        self.flags = 0
        self.marker = None

    def reset(self):
        self.buf = []
        self.leading_zeroes = 0
        self.frozen = False
        self.flags = 0
        self.marker = None

    def __len__(self):
        return len(self.buf) + self.leading_zeroes

    def digits(self):
        return "0" * self.leading_zeroes + "".join(self.buf)

    def parse(self):
        text = self.digits()
        return int(text) if text else 0

    def parse_decimal(self):
        text = self.digits()
        return int(text) / (10 ** len(text)) if text else 0.0

    def is_empty(self):
        return not self.buf and self.leading_zeroes == 0

    def put(self, digits):
        if self.frozen:
            return False
        if not self.buf and digits == "0":
            self.leading_zeroes += 1
            return True
        if all(digit == "0" for digit in digits):
            return False
        positions = len(digits)
        if not self.buf:
            self.buf = list(digits)
            return True
        if len(self.buf) < positions:
            return False
        start = len(self.buf) - positions
        if all(digit == "0" for digit in self.buf[start:]):
            self.buf[start:] = list(digits)
            return True
        return False

    def put_digit_at(self, digit, position):
        if self.frozen or digit == "0":
            return False
        length = len(self.buf)
        if position >= length:
            new = ["0"] * (position + 1)
            new[0] = digit
            new[position + 1 - length:] = self.buf
            self.buf = new
            return True
        index = length - 1 - position
        if self.buf[index] == "0":
            self.buf[index] = digit
            return True
        return False

    def peek(self, positions):
        length = len(self.buf)
        count = min(length, positions)
        return "".join(self.buf[length - count:])

    def is_free(self, positions):
        return self.is_empty() or all(digit == "0" for digit in self.peek(positions))

    def is_range_free(self, start, end):
        if start >= len(self.buf):
            return True
        left = 0 if end >= len(self.buf) else len(self.buf) - end - 1
        return all(digit == "0" for digit in self.buf[left:len(self.buf) - start])

    def shift(self, positions):
        if self.frozen:
            return False
        if positions == 0:
            return True
        if not self.buf:
            self.buf = ["1"]
        length = len(self.buf)
        if length <= positions:
            self.buf.extend(["0"] * positions)
            return True
        padding = 0
        for digit in self.buf[length - positions:]:
            if digit != "0":
                break
            padding += 1
        if padding == positions:
            self.buf[length - 1] = "1"
            padding -= 1
        span = 2 * positions - padding
        if length >= span and all(digit == "0" for digit in self.buf[length - span:length - positions]):
            left = self.buf[length - span:length - positions]
            right = self.buf[length - positions + padding:]
            self.buf[length - span:length - positions] = right
            self.buf[length - positions + padding:] = left
            return True
        return False


# Longest first: text2num's WordSplitter matches leftmost-longest.
_SPLITTERS = (
    "billionste", "milliardste", "millionste", "tausendste", "hundertste",
    "billion", "milliarden", "milliarde", "millionen", "million",
    "tausend", "hundert", "und",
)
_ONES = {
    "null": "0", "ein": "1", "eins": "1", "erste": "1", "zwei": "2",
    "zwo": "2", "zweite": "2", "drei": "3", "dritte": "3", "vier": "4",
    "vierte": "4", "fünf": "5", "fünfte": "5", "sechs": "6", "sechste": "6",
    "sieben": "7", "siebte": "7", "acht": "8", "achte": "8", "neun": "9",
    "neunte": "9",
}
_TEENS = {
    "zehn": "10", "zehnte": "10", "elf": "11", "elfte": "11",
    "zwölf": "12", "zwölfte": "12", "dreizehn": "13", "dreizehnte": "13",
    "vierzehn": "14", "vierzehnte": "14", "fünfzehn": "15",
    "fünfzehnte": "15", "sechzehn": "16", "sechzehnte": "16",
    "siebzehn": "17", "siebzehnte": "17", "achtzehn": "18",
    "achtzehnte": "18", "neunzehn": "19", "neunzehnte": "19",
}
_TENS = {
    "zwanzig": "2", "zwanzigste": "2", "dreißig": "3", "dreissig": "3",
    "dreißigste": "3", "dreissigste": "3", "vierzig": "4", "vierzigste": "4",
    "fünfzig": "5", "fünfzigste": "5", "sechzig": "6", "sechzigste": "6",
    "siebzig": "7", "siebzigste": "7", "achtzig": "8", "achtzigste": "8",
    "neunzig": "9", "neunzigste": "9",
}
_SCALES = {
    "hundert": 2, "hundertste": 2, "tausend": 3, "tausendste": 3,
    "million": 6, "millionen": 6, "millionste": 6,
    "milliarde": 9, "milliarden": 9, "milliardste": 9,
    "billion": 12, "billionste": 12,
}
_DECIMAL_DIGITS = {
    "null": "0", "eins": "1", "zwei": "2", "drei": "3", "vier": "4",
    "fünf": "5", "sechs": "6", "sieben": "7", "acht": "8", "neun": "9",
}
# text2num's German INSIGNIFICANT set: words that link numbers instead of
# separating them ("zwei plus drei" vs "zwei Kühe und drei Hühner").
_INSIGNIFICANT = {
    "aber", "ah", "äh", "ähm", "also", "gut", "auch", "denn", "doch", "dort",
    "eben", "eh", "halt", "ja", "mal", "sehen", "naja", "nun", "ok", "schon",
    "so", "genau", "und", "noch",
}


def _lemmatize(word):
    """Strip an ordinal's inflection: fünften -> fünfte, drittes -> dritte."""
    if word.endswith(("tes", "ter", "ten", "tem")):
        return word.rstrip("snmr")
    return word


def _is_splittable(word):
    best = None
    for pattern in _SPLITTERS:
        pos = word.find(pattern)
        if pos == -1:
            continue
        if best is None or pos < best[0] or (pos == best[0] and len(pattern) > len(best[1])):
            best = (pos, pattern)
    if best is None:
        return False
    pos, pattern = best
    return pos > 0 or pos + len(pattern) < len(word)


def _split_compound(word):
    parts = []
    cursor = 0
    while cursor < len(word):
        match = None
        for pattern in _SPLITTERS:
            if word.startswith(pattern, cursor):
                match = pattern
                break
        if match is not None:
            parts.append(match)
            cursor += len(match)
            continue
        next_pos = len(word)
        for pattern in _SPLITTERS:
            pos = word.find(pattern, cursor)
            if pos != -1 and pos < next_pos:
                next_pos = pos
        if next_pos > cursor:
            parts.append(word[cursor:next_pos])
        cursor = next_pos
    return parts


def _apply_simple(word, digits):
    """One German number word on a _Digits. True = ok, None = joiner, False = no."""
    word = _lemmatize(word)
    blocked = digits.flags & _TENS_FLAG
    to_block = 0
    if word in _ONES:
        if word == "null":
            ok = digits.put("0")
        elif digits.is_free(2):
            to_block = _TENS_FLAG
            ok = digits.put(_ONES[word])
        else:
            ok = False
    elif word in _TEENS:
        ok = digits.put(_TEENS[word])
    elif word in _TENS:
        ok = digits.put_digit_at(_TENS[word], 1) if not blocked else False
    elif word in _SCALES:
        shift = _SCALES[word]
        if shift == 2:
            peek = digits.peek(2)
            ok = digits.shift(2) if (len(peek) == 1 or peek < "20") else False
        elif shift == 3:
            ok = digits.shift(3) if digits.is_range_free(3, 5) else False
        elif shift == 6:
            ok = digits.shift(6) if digits.is_range_free(6, 8) else False
        else:
            ok = digits.shift(shift)
    elif word == "und":
        digits.flags = 0
        return None
    else:
        digits.flags = 0
        return False
    if ok:
        digits.flags = to_block
        if word.endswith("te"):
            digits.marker = "ordinal"
            digits.frozen = True
        if word == "eins":
            digits.frozen = True
    else:
        digits.flags = 0
    return ok


def _exec_group(parts):
    digits = _Digits()
    incomplete = False
    for part in parts:
        status = _apply(part, digits)
        if status is None:
            incomplete = True
        elif status is False:
            return None
        else:
            incomplete = False
    return None if incomplete else digits


def _apply(word, digits):
    word = _lemmatize(word)
    if _is_splittable(word):
        result = _exec_group(_split_compound(word))
        if result is None:
            return False
        if 3 < len(result) <= 6 and not digits.is_range_free(3, 5):
            return False
        if not digits.put(result.digits()):
            return False
        if result.marker == "ordinal":
            digits.marker = "ordinal"
            digits.frozen = True
        return True
    return _apply_simple(word, digits)


def _de_number(word):
    """A German number word as an int, or None. Handles compounds and ordinals:
    zweiundzwanzig, einhundertfünfundzwanzig, fünften, Millionen."""
    word = _lemmatize(word.lower())
    if " " in word:
        digits = _exec_group(word.split())
    elif _is_splittable(word):
        digits = _exec_group(_split_compound(word))
    else:
        digits = _Digits()
        if _apply_simple(word, digits) is not True:
            return None
    return digits.parse() if digits is not None else None


def _apply_decimal(word, digits):
    digit = _DECIMAL_DIGITS.get(word)
    if digit is None:
        return False
    digits.buf.append(digit)
    return True


def _check_decimal_separator(word):
    if word == "komma":
        return ","
    if word == "punkt":
        return "."
    return None


class _Parser:
    """text2num's WordToDigitParser: one number at a time, integer and decimal."""

    def __init__(self):
        self.ds = _Digits()
        self.dec = _Digits()
        self.dec_sep = None

    def reset(self):
        self.ds.reset()
        self.dec.reset()
        self.dec_sep = None

    def push(self, word):
        if self.dec_sep is not None:
            return _apply_decimal(word, self.dec)
        status = _apply(word, self.ds)
        if status is False and not self.ds.is_empty():
            sep = _check_decimal_separator(word)
            if sep is not None:
                self.dec_sep = sep
                return None
        return status

    def has_number(self):
        return not self.ds.is_empty()

    def is_ordinal(self):
        return self.ds.marker == "ordinal"

    def string_and_value(self):
        if not self.dec.is_empty():
            digits = self.ds.digits() + self.dec_sep + self.dec.digits()
            value = self.ds.parse() + self.dec.parse_decimal()
        else:
            digits = self.ds.digits()
            if self.ds.marker == "ordinal":
                digits += "."
            value = float(self.ds.parse())
        self.reset()
        return digits, value


class _Tracker:
    """text2num's NumTracker: the threshold only drops *isolated* small numbers."""

    def __init__(self):
        self.matches = []
        self.on_hold = None
        self.last_kind = None
        self.match_start = 0
        self.match_end = 0

    def number_advanced(self, pos):
        if self.match_start == self.match_end:
            self.match_start = pos
        self.match_end = pos + 1

    def number_end(self, parser):
        is_ordinal = parser.is_ordinal()
        digits, value = parser.string_and_value()
        forget = (len(digits) == 1 or is_ordinal) and value < 3.0
        kind = "ordinal" if is_ordinal else "cardinal"
        if self.last_kind != kind:
            self.last_kind = None
        if self.last_kind is not None:
            if self.on_hold is not None:
                self.matches.append(self.on_hold)
                self.on_hold = None
            self.matches.append((self.match_start, self.match_end, digits))
        elif forget:
            self.on_hold = (self.match_start, self.match_end, digits)
        else:
            self.matches.append((self.match_start, self.match_end, digits))
            self.on_hold = None
        self.last_kind = kind
        self.match_start = self.match_end

    def sequence_breaker(self):
        self.last_kind = None


def _tokenize(text):
    """text2num's Tokenize: words are alphanumeric plus '-' and \"'\", the rest
    is separator runs (so punctuation stays attached to its neighbours)."""
    tokens = []
    cursor = 0
    while cursor < len(text):
        char = text[cursor]
        end = cursor + 1
        if char.isalnum():
            while end < len(text) and (text[end].isalnum() or text[end] in "-'"):
                end += 1
        else:
            while end < len(text) and not text[end].isalnum():
                end += 1
        tokens.append(text[cursor:end])
        cursor = end
    return tokens


def _outside_number(tracker, token):
    no_alpha = all(not char.isalpha() for char in token)
    if not ((no_alpha and token.strip() != ".") or token in _INSIGNIFICANT):
        tracker.sequence_breaker()


def _alpha2digit(text):
    """German number words to digits, with text2num's default threshold of 3.0."""
    tokens = _tokenize(text)
    lower = [token.lower() for token in tokens]
    parser = _Parser()
    tracker = _Tracker()
    for pos, token in enumerate(tokens):
        if token == "-" or token.isspace():
            continue
        word = lower[pos]
        status = parser.push(word)
        if status is True:
            tracker.number_advanced(pos)
        elif status is None:
            pass
        elif parser.has_number():
            tracker.number_end(parser)
            if parser.push(word) is True:
                tracker.number_advanced(pos)
            else:
                _outside_number(tracker, token)
        else:
            _outside_number(tracker, token)
    if parser.has_number():
        tracker.number_end(parser)
    for start, end, digits in reversed(tracker.matches):
        tokens[start:end] = [digits]
    return "".join(tokens)


def normalize(text):
    """The dataset card's normalize_text, stdlib-only and case-insensitive.

    Two deliberate deviations from the card: case is folded (the card's
    published table is only reproducible case-insensitively), and non-lexical
    fillers are dropped from both sides (see FILLERS). Everything else follows
    the card's order, including reading the number words before the
    punctuation is stripped (text2num is punctuation-aware).
    """
    if isinstance(text, list):
        text = text[0] if text else ""
    text = str(text or "")
    for umlaut, ascii_ in _UMLAUTS:
        text = text.replace(umlaut, f"__{ascii_}__")
    text = text.replace("ß", "ss").replace(",,", "").replace('"', "")
    text = _strip_accents(text)
    for umlaut, ascii_ in _UMLAUTS:
        text = text.replace(f"__{ascii_}__", umlaut)
    text = _alpha2digit(text)
    text = re.sub(r"\[.*?\]", "", text)          # [noise] annotations
    text = re.sub(r"['\"]", "", text)
    text = re.sub(r"[^\w\s]", "", text)          # punctuation and hyphens
    text = re.sub(r"\s+", " ", text.strip())
    words = [word for word in text.casefold().split() if word not in FILLERS]
    return " ".join(words)


def _collapse(words):
    """jiwer's wer_standardize_contiguous: one of each run of equal words."""
    return [word for word, _ in groupby(words)]


def wer(reference, hypothesis):
    """Word-level Levenshtein distance with S/D/I counts.

    ``rate`` is errors / reference words (insertions can push it above 1);
    None when the reference has no words after normalization.
    """
    ref = _collapse(normalize(reference).split())
    hyp = _collapse(normalize(hypothesis).split())
    if not ref:
        return {"errors": 0, "words": 0, "substitutions": 0, "deletions": 0,
                "insertions": 0, "rate": None}
    n, m = len(ref), len(hyp)
    distance = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        distance[i][0] = i
    for j in range(m + 1):
        distance[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            distance[i][j] = min(
                distance[i - 1][j] + 1,
                distance[i][j - 1] + 1,
                distance[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]),
            )
    i, j, substitutions, deletions, insertions = n, m, 0, 0, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1] \
                and distance[i][j] == distance[i - 1][j - 1]:
            i, j = i - 1, j - 1
        elif i > 0 and j > 0 and distance[i][j] == distance[i - 1][j - 1] + 1:
            substitutions, i, j = substitutions + 1, i - 1, j - 1
        elif i > 0 and distance[i][j] == distance[i - 1][j] + 1:
            deletions, i = deletions + 1, i - 1
        else:
            insertions, j = insertions + 1, j - 1
    return {"errors": distance[n][m], "words": n, "substitutions": substitutions,
            "deletions": deletions, "insertions": insertions,
            "rate": distance[n][m] / n}


# ---- dataset ---------------------------------------------------------------

def _viewer_rows(offset, length):
    """One page of the dataset viewer (stdlib; gzip when the server offers it)."""
    url = VIEWER + "/rows?" + urllib.parse.urlencode(
        {"dataset": DATASET, "config": CONFIG, "split": SPLIT,
         "offset": offset, "length": length})

    def fetch():
        request = urllib.request.Request(url, headers={"Accept-Encoding": "gzip"})
        with urllib.request.urlopen(request, timeout=90) as response:
            raw = response.read()
            if response.headers.get("Content-Encoding") == "gzip":
                raw = gzip.decompress(raw)
        return json.loads(raw)

    return common._retrying(fetch)


def row_count():
    """Rows the viewer serves (the first 5 GB of the dataset). Cached."""
    global _ROW_COUNT
    if _ROW_COUNT is None:
        page = _viewer_rows(0, 1)
        _ROW_COUNT = int(page.get("num_rows_total") or 0)
    return _ROW_COUNT


def plan(items=DEFAULT_ITEMS, seed=SEED):
    """The row indices one run scores: a seeded sample, sorted for a stable
    order. The same (items, seed) always picks the same rows."""
    total = row_count()
    if total <= 0:
        raise RuntimeError("the dataset viewer reports no rows")
    items = max(1, min(int(items), total))
    if items >= total:
        return list(range(total))
    return sorted(random.Random(seed).sample(range(total), items))


def _cache_root():
    return common.CACHE / "asr" / DATASET.replace("/", "__")


def _cache_put(path, data):
    """Best effort: a read-only or full cache never fails an eval."""
    try:
        _evict(path.parent)
        common.write_atomic(path, data)
    except OSError:
        pass


def _evict(root):
    try:
        files = sorted((p for p in root.iterdir() if p.is_file()),
                       key=lambda p: p.stat().st_mtime)
    except OSError:
        return
    total = sum(p.stat().st_size for p in files)
    while files and total > MAX_CACHE_BYTES:
        victim = files.pop(0)
        total -= victim.stat().st_size
        try:
            victim.unlink()
        except OSError:
            pass


def _wav_bytes(samples, rate):
    """A float array (-1..1) as a 16-bit mono PCM WAV."""
    pcm = array.array("h", (int(round(max(-1.0, min(1.0, float(s))) * 32767))
                            for s in samples))
    if sys.byteorder == "big":
        pcm.byteswap()
    out = io.BytesIO()
    with wave.open(out, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(rate)
        writer.writeframes(pcm.tobytes())
    return out.getvalue()


def fetch_item(index):
    """One (reference, source, WAV) item, cached as a small JSON + WAV pair."""
    meta_path = _cache_root() / f"{index}.json"
    wav_path = _cache_root() / f"{index}.wav"
    if meta_path.exists() and wav_path.exists():
        meta = json.loads(meta_path.read_text())
        return {**meta, "wav": wav_path.read_bytes()}
    page = _viewer_rows(index, 1)
    rows = page.get("rows") or []
    if not rows:
        raise RuntimeError(f"the dataset viewer returned no row {index}")
    entry = rows[0]
    if "audio" in (entry.get("truncated_cells") or []):
        raise RuntimeError(f"row {index}: the viewer truncated the audio")
    row = entry["row"]
    audio = row.get("audio") or {}
    samples = audio.get("array") or []
    rate = int(audio.get("sampling_rate") or 16000)
    if not samples:
        raise RuntimeError(f"row {index}: no audio")
    wav = _wav_bytes(samples, rate)
    meta = {
        "row": index,
        "source": str(row.get("from") or ""),
        "reference": str(row.get("references") or ""),
        "seconds": round(len(samples) / rate, 2),
    }
    _cache_put(meta_path, json.dumps(meta, ensure_ascii=False).encode())
    _cache_put(wav_path, wav)
    return {**meta, "wav": wav}


# ---- model calls -----------------------------------------------------------

def _clean_transcript(text):
    text = str(text or "").strip()
    text = re.sub(r"^(?:transkription|transcript|transcription)\s*:\s*", "",
                  text, flags=re.IGNORECASE)
    return text.strip().strip('"').strip("'").strip()


def _transcribe_asr(base, model, wav, timeout):
    """POST one WAV to an OpenAI-compatible /v1/audio/transcriptions."""
    boundary = uuid.uuid4().hex
    parts = []
    for name, value in (("model", model), ("response_format", "json")):
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"'
            f"\r\n\r\n{value}\r\n".encode())
    parts.append(
        (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
         'filename="item.wav"\r\nContent-Type: audio/wav\r\n\r\n').encode()
        + wav + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    request = urllib.request.Request(
        base.rstrip("/") + "/v1/audio/transcriptions",
        data=b"".join(parts), method="POST")
    request.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    request.add_header("Accept", "application/json")
    with urllib.request.urlopen(request, timeout=timeout, context=common.CTX) as response:
        raw = response.read()
    try:
        parsed = json.loads(raw)
    except ValueError:
        return raw.decode(errors="replace")
    if isinstance(parsed, dict):
        return str(parsed.get("text") or "")
    return str(parsed)


def _transcribe_llm(base, model, wav, prompt, timeout):
    """One chat completion with an OpenAI input_audio part."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "input_audio", "input_audio": {
                "data": base64.b64encode(wav).decode("ascii"), "format": "wav"}},
        ]}],
        "temperature": 0,
        "max_tokens": 512,
    }
    result = common.request(base, "/v1/chat/completions", body, timeout=timeout)
    choices = (result or {}).get("choices") or []
    if not choices:
        raise RuntimeError("the server returned no choices")
    content = (choices[0].get("message") or {}).get("content")
    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content
                          if isinstance(part, dict))
    return str(content or "")


def transcribe(base, model, mode, wav, prompt=DEFAULT_PROMPT, timeout=180):
    """(transcript, latency_ms) for one WAV. Raises on a request failure."""
    started = time.monotonic()
    if mode == "asr":
        text = _transcribe_asr(base, model, wav, timeout)
    elif mode == "llm":
        text = _transcribe_llm(base, model, wav, prompt, timeout)
    else:
        raise ValueError(f"unknown mode {mode!r} — asr or llm")
    return _clean_transcript(text), int((time.monotonic() - started) * 1000)


# ---- the run ---------------------------------------------------------------

def evaluate(base, model, mode, indices, prompt=DEFAULT_PROMPT, timeout=180):
    """Yield one result per row: reference, hypothesis, WER (percent), latency.

    Audio is fetched on a background thread one row ahead, so the dataset
    download overlaps the model call instead of doubling the wall time. A
    fetch failure is an item error, not a dead run.
    """
    pending = queue.Queue(maxsize=2)

    def fetch():
        for index in indices:
            try:
                item = fetch_item(index)
            except Exception as exc:  # noqa: BLE001 — one row must not kill the run
                item = {"row": index, "error": f"dataset: {exc}"}
            pending.put(item)
        pending.put(None)

    threading.Thread(target=fetch, name="asr-fetch", daemon=True).start()
    while True:
        item = pending.get()
        if item is None:
            return
        if item.get("error"):
            yield {"row": item["row"], "source": "", "reference": "",
                   "hypothesis": "", "wer": None, "errors": 0, "words": 0,
                   "latency_ms": None, "audio_seconds": None,
                   "error": item["error"]}
            continue
        try:
            hypothesis, latency = transcribe(base, model, mode, item["wav"],
                                             prompt, timeout)
            error = None
        except Exception as exc:  # noqa: BLE001 — one request must not kill the run
            hypothesis, latency, error = "", None, str(exc)
        score = wer(item["reference"], hypothesis) if error is None else {
            "errors": 0, "words": 0, "rate": None}
        yield {
            "row": item["row"],
            "source": item["source"],
            "reference": item["reference"],
            "hypothesis": hypothesis,
            "wer": None if score["rate"] is None else round(100 * score["rate"], 2),
            "errors": score["errors"],
            "words": score["words"],
            "latency_ms": latency,
            "audio_seconds": item["seconds"],
            "error": error,
        }


def summarize(items):
    """Overall and per-source WER (percent), errors and words pooled."""
    def score(rows):
        errors = sum(row["errors"] for row in rows)
        words = sum(row["words"] for row in rows)
        ok = [row for row in rows if row["error"] is None]
        return {
            "wer": round(100 * errors / words, 2) if words else None,
            "errors": errors,
            "words": words,
            "items": len(rows),
            "failed": len(rows) - len(ok),
            "empty": sum(1 for row in ok if not normalize(row["hypothesis"])),
        }

    by_source = {}
    for item in items:
        by_source.setdefault(item["source"] or "(unknown)", []).append(item)
    return {"overall": score(items),
            "by_source": {name: score(rows) for name, rows in sorted(by_source.items())}}


# ---- CLI -------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="German ASR word-error-rate eval (flozi00/asr-german-mixed-evals)")
    parser.add_argument("--url", required=True, help="model endpoint, e.g. https://asr.example.org")
    parser.add_argument("--model", default=None, help="served model id (default: /v1/models, else 'default')")
    parser.add_argument("--mode", choices=("asr", "llm"), default="asr",
                        help="asr = /v1/audio/transcriptions, llm = chat with input_audio")
    parser.add_argument("--items", type=int, default=DEFAULT_ITEMS,
                        help=f"rows to score, 1-{MAX_ITEMS} (default {DEFAULT_ITEMS})")
    parser.add_argument("--seed", type=int, default=SEED, help="row-sample seed")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="instruction for --mode llm")
    parser.add_argument("--timeout", type=float, default=180, help="seconds per request")
    parser.add_argument("--json", default=None, help="write per-item JSONL here (default: stdout)")
    args = parser.parse_args(argv)
    if not 1 <= args.items <= MAX_ITEMS:
        parser.error(f"--items must be 1-{MAX_ITEMS}")

    model = args.model
    if not model:
        try:
            model = common.server_info(args.url)["model"]
        except Exception:  # noqa: BLE001 — Parakeet has no /v1/models
            model = "default"
    indices = plan(args.items, args.seed)
    print(f"{len(indices)} rows from {DATASET} (seed {args.seed}), mode {args.mode}, "
          f"model {model}", file=sys.stderr, flush=True)
    out = open(args.json, "w") if args.json else sys.stdout
    items = []
    try:
        for item in evaluate(args.url, model, args.mode, indices, args.prompt, args.timeout):
            items.append(item)
            out.write(json.dumps(item, ensure_ascii=False) + "\n")
            out.flush()
    finally:
        if args.json:
            out.close()
    summary = summarize(items)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["overall"]["words"] else 1


if __name__ == "__main__":
    sys.exit(main())
