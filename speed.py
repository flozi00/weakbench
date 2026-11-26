#!/usr/bin/env python3
"""Speed suite: closed-loop streamed load over a scenario matrix (stdlib only).

A CELL is scenario x concurrency c x round: c workers send requests back to
back until --seconds pass; requests in flight then finish and are recorded,
and every worker completes at least one. Tokens are counted from per-chunk
token_ids (speculative decoding emits several per chunk) and each chunk's gap
is split evenly over its k tokens for ITL. Aggregate tok/s counts only tokens
that arrived in [start + ramp, stop]; spec-decode and prefix-cache rates are
/metrics deltas around the cell. --rounds repeats the whole matrix,
round-major, so a gate can compare ranges.

  speed.py --url https://llm.example.org --arm NAME [--model ID] [--conc 1,8,32]
           [--scenarios chat-en,chat-de,rag-8k,multiturn,tools,thinking]
           [--seconds 60] [--rounds 1] [--temperature 0.2] [--ramp 5] [--vision] [--dry-run]

Also available: rag-32k, edit, vision (vision only with --vision). Writes
runs/<stamp>-speed-<arm>/{manifest.json,requests.jsonl,summary.json} and
appends one ledger row per cell to results/speed.jsonl.
"""
import argparse
import base64
import functools
import itertools
import os
import random
import struct
import threading
import time
import zlib

import common

PATH = "/v1/chat/completions"
LANGS = ("de", "en")
TOPICS = {
    "en": ["volcanoes", "the history of tea", "sorting algorithms", "Roman roads", "coral reefs",
           "the printing press", "black holes", "sourdough baking", "the Silk Road", "photosynthesis",
           "jazz improvisation", "glaciers", "the immune system", "medieval castles", "solar power",
           "honeybees", "the transistor", "ocean tides", "the Hanseatic League", "public-key cryptography"],
    "de": ["Vulkane", "die Geschichte des Kaffees", "Sortieralgorithmen", "die Römerstraßen", "Korallenriffe",
           "den Buchdruck", "Schwarze Löcher", "das Sauerteigbrot", "die Seidenstraße", "die Photosynthese",
           "die Jazz-Improvisation", "Gletscher", "das Immunsystem", "mittelalterliche Burgen", "die Energiewende",
           "Honigbienen", "den Transistor", "die Gezeiten", "die Hanse", "die Public-Key-Kryptografie"],
}
ESSAY = {
    "en": "Request {nonce}. Write a detailed essay about {a} and {b}. Explain how each developed over time, "
          "compare them with concrete examples, discuss at least three surprising connections between the two, "
          "and close with what an expert would want a curious newcomer to understand first. "
          "Use section headings and full paragraphs.",
    "de": "Anfrage {nonce}. Schreibe einen ausführlichen Aufsatz über {a} und {b}. Erkläre, wie sich beide im "
          "Lauf der Zeit entwickelt haben, vergleiche sie anhand konkreter Beispiele, erörtere mindestens drei "
          "überraschende Verbindungen zwischen ihnen und schließe mit dem, was eine Expertin einem neugierigen "
          "Neuling zuerst vermitteln würde. Gliedere den Text mit Zwischenüberschriften und schreibe in ganzen "
          "Absätzen.",
}
SUMMARIZE = {"de": "Fasse den obigen Text in fünf Sätzen zusammen.",
             "en": "Summarize the text above in five sentences."}
FOLLOWUPS = ["Summarize the reference text in a few sentences.",
             "What are the three most important facts in the text?",
             "Explain the main topic to a twelve-year-old.",
             "Which statements in the text would you double-check, and why?",
             "Write three quiz questions about the text, with answers.",
             "What would a reader most likely want to know next?",
             "Rewrite your previous answer more concisely.",
             "Give a short timeline of the events or developments the text mentions."]
CITIES = ["Berlin", "Hamburg", "Frankfurt", "Stuttgart", "Paris", "Madrid", "London", "Amsterdam", "Oslo", "Dublin"]
ASKS = {  # one per tool, in TOOLS order
    "en": ["What's the weather in {city} right now? Do I need an umbrella?",
           "How much is {n} euros in US dollars today?",
           "Find me a flight from {city} to {city2} next Friday.",
           "Put 'Quarterly review' in my calendar for tomorrow at {h}:00, 45 minutes.",
           "Email jonas.weber@example.com with the subject 'Report' that it will be ready by {h}:00.",
           "What is SAP trading at right now?",
           "Remind me at {h}:30 to call the dentist.",
           "Has my order {n} shipped yet?"],
    "de": ["Wie ist das Wetter gerade in {city}? Brauche ich einen Schirm?",
           "Wie viel sind {n} Euro heute in US-Dollar?",
           "Such mir einen Flug von {city} nach {city2} für nächsten Freitag.",
           "Trag mir morgen um {h}:00 Uhr „Quartalsreview“ für 45 Minuten in den Kalender ein.",
           "Schreib jonas.weber@example.com mit dem Betreff „Bericht“, dass er bis {h}:00 Uhr fertig ist.",
           "Wo steht die SAP-Aktie gerade?",
           "Erinnere mich um {h}:30 Uhr daran, den Zahnarzt anzurufen.",
           "Ist meine Bestellung {n} schon verschickt?"],
}
MATH = {
    "en": ["A bakery bakes {a} loaves on Monday and {p}% more on Tuesday. On Wednesday it bakes half as many as on "
           "Monday and Tuesday together. A loaf sells for {c} euros and costs {d} euros to make, and every evening "
           "{e} loaves are left over and given away. What is the bakery's total profit over the three days? "
           "Solve it step by step.",
           "A freight train leaves Hamburg at {h}:00 at {a} km/h. A second train leaves the same station {d} hours "
           "later on a parallel track at {b} km/h. At what time does the second train catch up, how far from "
           "Hamburg are they then, and how much diesel has the second train used by then if it burns {f} litres "
           "per 100 km? Solve it step by step."],
    "de": ["Eine Bäckerei backt am Montag {a} Brote und am Dienstag {p} % mehr. Am Mittwoch backt sie halb so viele "
           "wie am Montag und Dienstag zusammen. Ein Brot wird für {c} Euro verkauft, die Herstellung kostet {d} "
           "Euro, und jeden Abend bleiben {e} Brote übrig, die verschenkt werden. Wie hoch ist der Gesamtgewinn der "
           "drei Tage? Löse die Aufgabe Schritt für Schritt.",
           "Ein Güterzug verlässt Hamburg um {h}:00 Uhr mit {a} km/h. Ein zweiter Zug fährt {d} Stunden später auf "
           "einem Parallelgleis mit {b} km/h los. Um wie viel Uhr holt der zweite Zug den ersten ein, wie weit sind "
           "beide dann von Hamburg entfernt, und wie viel Diesel hat der zweite Zug bis dahin verbraucht, wenn er "
           "{f} Liter pro 100 km braucht? Löse die Aufgabe Schritt für Schritt."],
}
EDIT = {"en": "Request {nonce}. Return the text below unchanged, except replace every 'and' with 'plus'. "
              "Output only the text.",
        "de": "Anfrage {nonce}. Gib den folgenden Text unverändert zurück, ersetze aber jedes „und“ durch „sowie“. "
              "Gib nur den Text aus."}
VISION_Q = "Describe this image: which colours appear, and where in the grid?"


def _tool(name, description, **params):
    """OpenAI function tool; params = {name: (json type, description)}, all required."""
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {
        "type": "object", "required": list(params),
        "properties": {k: {"type": t, "description": d} for k, (t, d) in params.items()}}}}


TOOLS = [
    _tool("get_weather", "Current weather and today's forecast for a city", city=("string", "City name, e.g. Berlin")),
    _tool("convert_currency", "Convert an amount between currencies at today's rate",
          amount=("number", "Amount to convert"), source=("string", "ISO code, e.g. EUR"),
          target=("string", "ISO code, e.g. USD")),
    _tool("search_flights", "Search flights between two cities on a date", origin=("string", "Departure city"),
          destination=("string", "Arrival city"), date=("string", "YYYY-MM-DD")),
    _tool("create_event", "Add an event to the user's calendar", title=("string", "Event title"),
          start=("string", "ISO 8601 start time"), minutes=("integer", "Duration in minutes")),
    _tool("send_email", "Send an email on the user's behalf", to=("string", "Recipient address"),
          subject=("string", "Subject line"), body=("string", "Message text")),
    _tool("get_stock_price", "Latest share price for a ticker symbol", symbol=("string", "Ticker, e.g. SAP")),
    _tool("set_reminder", "Remind the user of something at a given time", text=("string", "What to remind about"),
          time=("string", "ISO 8601 time")),
    _tool("order_status", "Shipping status of a web-shop order", order_id=("string", "Order number")),
]


@functools.cache
def corpus():
    """40 long Wikipedia articles per language, fetched once per process."""
    return {lang: common.wiki_texts(lang, 40) for lang in LANGS}


def passage(rng, lang, chars):
    """~chars of seeded Wikipedia prose: the language's articles shuffled,
    joined and cut at a random offset (word boundaries)."""
    texts = corpus()[lang][:]
    rng.shuffle(texts)
    joined = "\n\n".join(texts)
    joined *= chars // max(1, len(joined)) + 1  # a corpus shorter than chars repeats
    start = joined.find(" ", rng.randrange(len(joined) - chars)) + 1
    return joined[start:start + chars].rsplit(" ", 1)[0]


@functools.cache
def system_prompt():
    """~8k-token system prompt shared by every multiturn request (prefix-cached).
    Hybrid models (Qwen3.8 GDN) cache prefixes in 2880-token blocks: a 2k prompt
    never fills one, so it would measure no caching at all on them."""
    return (f"Session {SALT}. You are the research assistant of a public library. Answer questions about the reference text "
            "below accurately and concisely, and say so when the text does not contain the answer.\n\n"
            "Reference text:\n" + passage(random.Random("system-prompt"), "en", 28800))


# Per-process salt in every prompt: two runs against one live server never
# share a cached prefix (run A2 would otherwise reuse A1's KV). Within a run,
# shared prefixes (multiturn system prompt, tool schemas) stay shared.
SALT = os.urandom(4).hex()


def nonce(rng):
    return f"{rng.getrandbits(48):012x}"


def png(rng, size=448, grid=4):
    """Seeded size x size RGB PNG of grid x grid random-colour blocks."""
    side, colors = size // grid, [bytes(rng.randrange(256) for _ in range(3)) for _ in range(grid * grid)]
    raw = b"".join((b"\x00" + b"".join(c * side for c in colors[r * grid:(r + 1) * grid])) * side
                   for r in range(grid))

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


# Builders: (rng, worker, request index, session history) -> messages.
# Mixed-language scenarios alternate de/en per request.
def chat(lang):
    def build(rng, w, i, hist):
        a, b = rng.sample(TOPICS[lang], 2)
        return [{"role": "user", "content": ESSAY[lang].format(nonce=nonce(rng), a=a, b=b)}]
    return {"build": build, "max_tokens": 512, "ignore_eos": True}


def rag(tokens):
    """Nonce first line (no prefix reuse across requests) + ~tokens of Wikipedia
    (chars / 3.6) + a summary question."""
    def build(rng, w, i, hist):
        lang, tag = LANGS[(w + i) % 2], nonce(rng)
        text = passage(rng, lang, int(tokens * 3.6))
        return [{"role": "user", "content": f"Request {tag}\n\n{text}\n\n{SUMMARIZE[lang]}"}]
    return {"build": build, "max_tokens": 256, "ignore_eos": True, "ctx": tokens}


def session_turn(rng, w, i, hist):
    """Shared system prompt + this worker's session; the worker appends each reply."""
    if i % 6 == 0:  # a session lasts 6 turns
        hist.clear()
    hist.append({"role": "user", "content": rng.choice(FOLLOWUPS)})
    return [{"role": "system", "content": system_prompt()}, *hist]


def tool_request(rng, w, i, hist):
    city, city2 = rng.sample(CITIES, 2)
    ask = rng.choice(ASKS[LANGS[(w + i) % 2]])
    return [{"role": "system", "content": f"Session {SALT}."},  # salts the cached tool-schema prefix per run
            {"role": "user", "content": ask.format(city=city, city2=city2, n=rng.randint(12, 980),
                                                   h=rng.randint(8, 18))}]


def word_problem(rng, w, i, hist):
    a = 20 * rng.randint(2, 6)
    text = rng.choice(MATH[LANGS[(w + i) % 2]]).format(
        a=a, b=a + rng.randint(15, 60), c=rng.randint(4, 9), d=rng.randint(1, 3), e=rng.randint(3, 12),
        p=rng.choice((10, 20, 50)), h=rng.randint(6, 14), f=rng.randint(150, 400))
    return [{"role": "user", "content": text}]


def edit_request(rng, w, i, hist):
    lang = LANGS[(w + i) % 2]
    return [{"role": "user", "content": EDIT[lang].format(nonce=nonce(rng)) + "\n\n" + passage(rng, lang, 4320)}]


def image_request(rng, w, i, hist):
    url = "data:image/png;base64," + base64.b64encode(png(rng)).decode()
    return [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}},
                                         {"type": "text", "text": VISION_Q}]}]


def longform(lang):
    """The chat prompt with a 2048-token budget: sustained decode, the shape a long-form answer has.

    `chat` stops at 512 tokens, which is short enough that a decode-side regression hides behind the
    prefill. This is the cell that sees it — and the one a creative-writing or report workload lives in.
    """
    return {**chat(lang), "max_tokens": 2048}


SCENARIOS = {
    "chat-en": chat("en"),
    "chat-de": chat("de"),
    "longform-en": longform("en"),
    "longform-de": longform("de"),
    "rag-8k": rag(8000),
    "rag-32k": rag(32000),
    "multiturn": {"build": session_turn, "max_tokens": 256, "ignore_eos": True, "session": True},
    "tools": {"build": tool_request, "max_tokens": 256, "ignore_eos": False,
              "extra": {"tools": TOOLS, "tool_choice": "auto"}},
    "thinking": {"build": word_problem, "max_tokens": 2048, "ignore_eos": True},  # server-default thinking
    "edit": {"build": edit_request, "max_tokens": 1600, "ignore_eos": False},  # copy-heavy, natural stop
    "vision": {"build": image_request, "max_tokens": 128, "ignore_eos": True},  # only with --vision
}
DEFAULT = "chat-en,chat-de,longform-en,longform-de,rag-8k,multiturn,tools,thinking"
REQ_FIELDS = ("t_send", "ttft_s", "e2e_s", "n_out", "prompt_tokens", "decode_tps", "finish_reason",
              "tool_call", "error")


def body(model, name, rnd, w, i, hist, temperature):
    """Request body; the prompt is a pure function of (SALT, scenario, round,
    worker, index) plus, for multiturn, the session's earlier replies."""
    spec = SCENARIOS[name]
    rng = random.Random(f"{SALT}/{name}/{rnd}/{w}/{i}")
    # No per-request seed: vLLM samples seeded requests through a per-request
    # generator (a different sampling kernel) — prod traffic is unseeded.
    req = {"model": model, "messages": spec["build"](rng, w, i, hist), "max_tokens": spec["max_tokens"],
           "temperature": temperature, "stream": True,
           "stream_options": {"include_usage": True}, "return_token_ids": True,
           "ignore_eos": spec["ignore_eos"], **spec.get("extra", {})}
    if os.environ.get("BENCH_NO_TOKEN_IDS"):
        # SGLang rejects return_token_ids on streaming chat (HTTP 400): parse() then spreads
        # usage.completion_tokens over the chunks. Set it for BOTH sides of an A/B, or the
        # ITL percentiles of the arm with token_ids and the one without are not comparable.
        del req["return_token_ids"]
    return req


def parse(t_send, chunks):
    """(time, chunk) pairs of one streamed reply -> request record. Never
    raises: a failure mid-stream keeps what arrived and lands in "error"."""
    rec = {"t_send": t_send, "finish_reason": None, "tool_call": False, "error": None}
    marks, content, reasoning, usage = [], [], [], {}  # marks: (arrival, tokens in chunk)
    try:
        for t, chunk in chunks:
            if chunk.get("error") or chunk.get("object") == "error":  # vLLM reports mid-stream failures in-band
                raise RuntimeError(f"stream error: {chunk.get('error') or chunk.get('message')}")
            usage = chunk.get("usage") or usage
            for ch in chunk.get("choices") or ():
                d = ch.get("delta") or {}
                rec["finish_reason"] = ch.get("finish_reason") or rec["finish_reason"]
                rec["tool_call"] |= bool(d.get("tool_calls"))
                content.append(d.get("content") or "")
                reasoning.append(d.get("reasoning_content") or d.get("reasoning") or "")
                k = len(ch.get("token_ids") or ())
                if k or content[-1] or reasoning[-1] or d.get("tool_calls"):
                    marks.append((t, k))
    except Exception as exc:  # noqa: BLE001 — a failed request is a data point, never a crash
        rec["error"] = repr(exc)[:300]
    if rec["error"] is None and rec["finish_reason"] is None:  # cut by a proxy without an HTTP error
        rec["error"] = "stream ended without finish_reason"
    if not any(k for _, k in marks) and usage.get("completion_tokens"):
        # server without token_ids: spread usage.completion_tokens evenly over the content chunks
        q, r = divmod(usage["completion_tokens"], len(marks) or 1)
        marks = [(t, q + (j < r)) for j, (t, _) in enumerate(marks)]
    marks = [m for m in marks if m[1]]
    n = sum(k for _, k in marks)
    t0, t1 = (marks[0][0], marks[-1][0]) if marks else (None, None)
    rec.update(marks=marks, n_out=n, prompt_tokens=usage.get("prompt_tokens"),
               gaps=[g for (a, _), (b, k) in itertools.pairwise(marks) for g in [(b - a) / k] * k],
               ttft_s=None if t0 is None else t0 - t_send, e2e_s=None if t1 is None else t1 - t_send,
               # tokens after the first chunk over the time after it (spec decode can put several in it)
               decode_tps=(n - marks[0][1]) / (t1 - t0) if marks and t1 > t0 else None,
               text="".join(content) or "".join(reasoning))
    return rec


def one(url, req):
    return parse(time.monotonic(), common.stream(url, PATH, req))


def rounded(row):
    return {k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}


def summarize(recs, t_start, t_stop, ramp, t_end, tools):
    """Cell statistics; every key always present, None when there is no data."""
    ok = [r for r in recs if not r["error"]]
    lo, win = t_start + ramp, t_stop - t_start - ramp

    def q(xs, p):  # nearest rank on sorted values
        return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else None

    def ms(v):
        return None if v is None else v * 1e3

    def mean(xs):
        return sum(xs) / len(xs) if xs else None

    tps = sorted(r["decode_tps"] for r in ok if r["decode_tps"] is not None)
    ttft = sorted(r["ttft_s"] for r in ok if r["ttft_s"] is not None)
    e2e = sorted(r["e2e_s"] for r in ok if r["e2e_s"] is not None)
    gaps = sorted(g for r in ok for g in r["gaps"])
    return {"requests": len(ok), "errors": len(recs) - len(ok), "seconds": t_end - t_start,
            # windowed: tokens that arrived / prompts sent inside [start + ramp, stop]
            "agg_out_tps": sum(k for r in recs for t, k in r["marks"] if lo <= t <= t_stop) / win,
            "agg_in_tps": sum(r["prompt_tokens"] or 0 for r in recs if lo <= r["t_send"] <= t_stop) / win,
            "stream_tps_p50": q(tps, .5), "stream_tps_p10": q(tps, .1),
            "ttft_p50_ms": ms(q(ttft, .5)), "ttft_p90_ms": ms(q(ttft, .9)), "ttft_p99_ms": ms(q(ttft, .99)),
            "ttft_max_ms": ms(q(ttft, 1)), "tpot_p50_ms": 1e3 / q(tps, .5) if tps else None,
            "itl_p50_ms": ms(q(gaps, .5)), "itl_p99_ms": ms(q(gaps, .99)),
            "e2e_p50_s": q(e2e, .5), "e2e_p90_s": q(e2e, .9),
            "mean_out_tokens": mean([r["n_out"] for r in ok]),
            "mean_in_tokens": mean([r["prompt_tokens"] for r in ok if r["prompt_tokens"] is not None]),
            "tool_call_rate": mean([r["tool_call"] for r in ok]) if tools else None}


def spec_rates(m0, m1):
    """Spec-decode and prefix-cache rates from two /metrics scrapes (None when
    the counters are missing or did not move)."""
    def d(key):
        return m1[key] - m0[key] if key in m0 and key in m1 else None

    def ratio(a, b):
        return a / b if a is not None and b is not None and b > 0 else None

    acc, pre = d("spec_decode_num_accepted_tokens_total"), d("num_preemptions_total")
    per_draft = ratio(acc, d("spec_decode_num_drafts_total"))
    return {"spec_accept_rate": ratio(acc, d("spec_decode_num_draft_tokens_total")),
            "spec_mean_len": None if per_draft is None else 1 + per_draft,
            "prefix_hit_rate": ratio(d("prefix_cache_hits_total"), d("prefix_cache_queries_total")),
            "preemptions": None if pre is None else int(pre)}


def load(m):
    """Requests the server is running or queueing (None without /metrics)."""
    if "num_requests_running" not in m:
        return None
    return int(m["num_requests_running"] + m.get("num_requests_waiting", 0))


def run_cell(url, model, name, conc, rnd, seconds, ramp, temperature, out=None):
    """One cell: appends per-request rows to `out`, returns the summary row."""
    spec, recs, lock = SCENARIOS[name], [], threading.Lock()
    m0 = common.metrics(url)
    t_start = time.monotonic()
    t_stop = t_start + seconds
    inflight, foreign = [0], [load(m0)]  # load(m0): nothing of ours runs yet

    def probe():  # mid-cell: server load beyond our own in-flight requests
        time.sleep(seconds / 2)
        m, ours = common.metrics(url), inflight[0]
        if load(m) is not None:
            foreign.append(max(0, load(m) - ours))

    def worker(w):
        hist, i = [], 0
        while i == 0 or time.monotonic() < t_stop:  # >= 1 request each; in-flight ones finish
            with lock:
                inflight[0] += 1
            rec = one(url, body(model, name, rnd, w, i, hist, temperature))
            with lock:
                inflight[0] -= 1
            text = rec.pop("text")
            if spec.get("session"):
                hist.append({"role": "assistant", "content": text or "..."})
            with lock:
                recs.append(rec)
                if out:  # t_send in seconds since cell start
                    common.append_jsonl(out, rounded({"scenario": name, "conc": conc, "round": rnd,
                                                      **{k: rec[k] for k in REQ_FIELDS},
                                                      "t_send": rec["t_send"] - t_start}))
            if rec["error"]:  # never hammer a failing server in a tight loop
                time.sleep(max(0.0, min(1.0, t_stop - time.monotonic())))
            i += 1

    threads = [threading.Thread(target=worker, args=(w,), daemon=True) for w in range(conc)]
    threading.Thread(target=probe, daemon=True).start()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    t_end = time.monotonic()
    return rounded({"scenario": name, "conc": conc, "round": rnd,
                    **summarize(recs, t_start, t_stop, ramp, t_end, "tools" in spec.get("extra", {})),
                    **spec_rates(m0, common.metrics(url)),
                    "foreign_load": max((f for f in foreign if f is not None), default=None)})


def plan(names, conc, rounds, seconds, max_model_len):
    """Round-major cells, {skipped scenario: reason}, estimated minutes."""
    skipped = {n: f"max_model_len {max_model_len} < {SCENARIOS[n]['ctx'] + 1024} needed" for n in names
               if "ctx" in SCENARIOS[n] and max_model_len and max_model_len < SCENARIOS[n]["ctx"] + 1024}
    cells = [(r, n, c) for r in range(1, rounds + 1) for n in names if n not in skipped for c in conc]
    # ponytail: tail guess = one max_tokens request at 50 tok/s per cell; no server-speed model
    minutes = sum(seconds + SCENARIOS[n]["max_tokens"] / 50 for _, n, _ in cells) / 60
    return cells, skipped, minutes


def fmt(v):
    return "-" if v is None else f"{v:.2f}" if isinstance(v, float) else str(v)


COLS = {"scenario": "scenario", "conc": "c", "round": "round", "requests": "req", "errors": "err",
        "agg_out_tps": "out tok/s", "agg_in_tps": "in tok/s", "stream_tps_p50": "stream p50",
        "stream_tps_p10": "stream p10", "ttft_p50_ms": "ttft p50 ms", "ttft_p99_ms": "ttft p99 ms",
        "itl_p99_ms": "itl p99 ms", "e2e_p50_s": "e2e p50 s", "mean_out_tokens": "out tok",
        "spec_accept_rate": "accept", "spec_mean_len": "mean len", "prefix_hit_rate": "prefix hit",
        "tool_call_rate": "tool calls", "foreign_load": "foreign"}


def line(r):
    return "  ".join(f"{h} {fmt(r[k])}" for k, h in COLS.items())


def table(rows):
    head = ["| " + " | ".join(COLS.values()) + " |", "|" + "---|" * len(COLS)]
    return "\n".join(head + ["| " + " | ".join(fmt(r[k]) for k in COLS) + " |" for r in rows])


def warmups(model, cells, temperature):
    """One short unrecorded request body per scenario of `cells` (building them
    loads the Wikipedia corpus, so call this before creating a run dir)."""
    return {n: {**body(model, n, 0, 0, 0, [], temperature), "max_tokens": min(32, SCENARIOS[n]["max_tokens"])}
            for _, n, _ in cells}


def matrix(url, model, cells, seconds, ramp, temperature, out=None, warm=None):
    """Run planned cells (see plan()), yielding each cell's summary row as it
    finishes. The CLI and any server embedding this module
    all measure through this one loop."""
    warm = dict(warm or warmups(model, cells, temperature))
    for rnd, n, c in cells:
        if n in warm:  # one unrecorded warmup request per scenario, before its first cell
            rec = one(url, warm.pop(n))
            if rec["error"]:
                print(f"warmup {n} failed: {rec['error']}", flush=True)
        yield run_cell(url, model, n, c, rnd, seconds, ramp, temperature, out)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Closed-loop speed benchmark over a scenario matrix.")
    ap.add_argument("--url", required=True)
    ap.add_argument("--arm", required=True, help="label of this server config (run dir, ledger)")
    ap.add_argument("--model", help="model id (default: first of /v1/models)")
    ap.add_argument("--scenarios", default=DEFAULT, help=f"comma list of: {', '.join(SCENARIOS)}")
    ap.add_argument("--conc", default="1,8,32")
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--max-parallel", type=int, default=0,
                    help="cap the suite's own requests in flight on the model (0 = off; the nightly "
                         "leaderboard sets this so its traffic never crowds a pool)")
    ap.add_argument("--yield-above", type=int, default=0,
                    help="send nothing new while the model runs more than this many requests "
                         "(running + waiting, from /metrics; 0 = off; user traffic has the right of way)")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--ramp", type=float, help="seconds left out of aggregate tok/s (default min(5, seconds/10))")
    ap.add_argument("--vision", action="store_true", help="add the vision scenario (multimodal models)")
    ap.add_argument("--dry-run", action="store_true", help="GET /v1/models, print the plan, send no load")
    args = ap.parse_args(argv)
    names = [n for n in args.scenarios.split(",") if n]
    if args.vision and "vision" not in names:
        names.append("vision")
    unknown = [n for n in names if n not in SCENARIOS]
    if unknown or ("vision" in names and not args.vision):
        ap.error(f"unknown scenarios: {unknown}" if unknown else "the vision scenario needs --vision")
    conc = [int(c) for c in args.conc.split(",")]
    args.ramp = min(5.0, args.seconds / 10) if args.ramp is None else args.ramp
    if not 0 <= args.ramp < args.seconds:
        ap.error("--ramp must be in [0, --seconds)")
    common.throttle_configure(args.url, args.max_parallel, args.yield_above)

    data = (common.request(args.url, "/v1/models", timeout=30) or {}).get("data") or [{}]
    entry = next((m for m in data if m.get("id") == args.model), data[0])
    args.model, mml = args.model or entry.get("id"), entry.get("max_model_len")
    cells, skipped, minutes = plan(names, conc, args.rounds, args.seconds, mml)
    if args.max_parallel:  # a cell above the own cap would break the politeness rule
        over = sorted({c for _, _, c in cells if c > args.max_parallel})
        if over:
            cells = [c for c in cells if c[2] <= args.max_parallel]
            skipped[f"c{','.join(map(str, over))}"] = f"above --max-parallel {args.max_parallel}"
    print(f"{args.model} (max_model_len {mml}): {len(cells)} cells = "
          f"[{', '.join(dict.fromkeys(n for _, n, _ in cells))}] x c={conc} x {args.rounds} round(s), "
          f"~{minutes:.0f} min", flush=True)
    for n, why in skipped.items():
        print(f"skip {n}: {why}", flush=True)
    if args.dry_run or not cells:
        return None

    warm = warmups(args.model, cells, args.temperature)  # loads the corpus before the run dir exists
    run_dir, manifest = common.new_run("speed", args.arm, args.url, args.model, vars(args))
    rows = []
    for row in matrix(args.url, args.model, cells, args.seconds, args.ramp, args.temperature,
                      run_dir / "requests.jsonl", warm):
        rows.append(row)
        print(line(row), flush=True)
    print("\n" + table(rows))
    common.finish_run(run_dir, manifest, rows)
    print(f"\n-> {run_dir}")
    return run_dir


if __name__ == "__main__":
    main()
