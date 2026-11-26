"""Hermetic tests for speed.py: mock server, temp run dirs, offline corpus.

    python3 test_speed.py      (or: pytest test_speed.py)
"""
import os
import base64
import contextlib
import io
import json
import struct
import sys
import tempfile
import threading
import time
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import mock_server  # noqa: E402
import speed  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="speed-test-"))
common.RUNS, common.RESULTS = TMP / "runs", TMP / "results"  # never write into the repo
TEXT = "\n".join(f"Paragraph {i}: the miller and the baker crossed the river {i} near the bridge." for i in range(300))
common.wiki_texts = lambda lang, n, **kw: [f"[{lang}] {TEXT}"] * n  # no Hugging Face download
M = mock_server.MODEL
ROW = ("scenario", "conc", "round", "requests", "errors", "seconds", "agg_out_tps", "agg_in_tps",
       "stream_tps_p50", "stream_tps_p10", "ttft_p50_ms", "ttft_p90_ms", "ttft_p99_ms", "ttft_max_ms",
       "tpot_p50_ms", "itl_p50_ms", "itl_p99_ms", "e2e_p50_s", "e2e_p90_s", "mean_out_tokens",
       "mean_in_tokens", "tool_call_rate", "spec_accept_rate", "spec_mean_len", "prefix_hit_rate", "preemptions",
       "foreign_load")
REQ = ("scenario", "conc", "round", "t_send", "ttft_s", "e2e_s", "n_out", "prompt_tokens", "decode_tps",
       "finish_reason", "tool_call", "error")


def chunk(k, content="w", **extra):
    return {"choices": [{"index": 0, "delta": {"content": content}, "token_ids": list(range(k)), **extra}]}


def usage(prompt, completion):
    return {"choices": [], "usage": {"prompt_tokens": prompt, "completion_tokens": completion}}


@contextlib.contextmanager
def mock(responder, **kw):
    url, stop = mock_server.serve(responder, **kw)
    try:
        yield url
    finally:
        stop()


def test_parse_token_ids_math():
    r = speed.parse(10.0, iter([(10.2, {"choices": [{"delta": {"role": "assistant", "content": ""}}]}),
                                (10.5, chunk(1)), (10.6, chunk(2)), (10.9, chunk(3, finish_reason="length")),
                                (10.9, usage(42, 6))]))
    assert (r["n_out"], r["prompt_tokens"], r["finish_reason"], r["error"]) == (6, 42, "length", None)
    assert abs(r["ttft_s"] - 0.5) < 1e-9 and abs(r["e2e_s"] - 0.9) < 1e-9  # role chunk carries no token
    assert abs(r["decode_tps"] - 5 / 0.4) < 1e-9  # (n - first chunk's tokens) / (t_last - t_first)
    assert [round(g, 9) for g in r["gaps"]] == [0.05, 0.05, 0.1, 0.1, 0.1]  # chunk gap / k, repeated k


def test_parse_usage_fallback_tool_calls_and_errors():
    def plain(t, text, **extra):
        return t, {"choices": [{"delta": {"content": text}, **extra}]}
    r = speed.parse(0.0, iter([plain(1.0, "a"), plain(2.0, "b"), plain(3.0, "c", finish_reason="stop"),
                               (3.0, usage(5, 7))]))
    assert r["n_out"] == 7 and [k for _, k in r["marks"]] == [3, 2, 2] and r["decode_tps"] == 2.0  # (7 - 3) / 2 s
    assert r["error"] is None
    call = {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"name": "get_weather"}}]},
                         "token_ids": [1, 2], "finish_reason": "tool_calls"}]}
    assert speed.parse(0.0, iter([(0.1, call)]))["tool_call"] is True

    def cut():
        yield 0.5, chunk(2)
        raise ConnectionResetError("peer reset")
    r = speed.parse(0.0, cut())
    assert "peer reset" in r["error"] and r["n_out"] == 2 and r["ttft_s"] == 0.5
    r = speed.parse(0.0, iter([(0.5, chunk(2)), (0.6, {"error": {"message": "engine dead", "code": 500}})]))
    assert "engine dead" in r["error"] and r["n_out"] == 2  # in-band SSE error
    assert speed.parse(0.0, iter([(0.5, chunk(2))]))["error"] == "stream ended without finish_reason"


def test_window_and_percentiles():
    def rec(t_send, marks, prompt):
        chunks = [(t, chunk(k, finish_reason="length" if j == len(marks) - 1 else None))
                  for j, (t, k) in enumerate(marks)]
        return speed.parse(t_send, iter(chunks + [(marks[-1][0], usage(prompt, 0))]))
    recs = [rec(0.1, [(0.5, 2), (3.0, 4), (9.0, 2), (11.0, 5)], 100), rec(3.0, [(4.0, 1)], 50)]
    s = speed.summarize(recs, 0.0, 10.0, 2.0, 11.0, tools=False)
    assert tuple(s) == ROW[3:22]
    assert s["agg_out_tps"] == 7 / 8 and s["agg_in_tps"] == 50 / 8  # ramp [0, 2) and post-stop excluded
    assert (s["requests"], s["errors"], s["seconds"], s["tool_call_rate"]) == (2, 0, 11.0, None)
    assert s["ttft_p50_ms"] == 1000.0 and s["ttft_max_ms"] == 1000.0  # nearest rank over [0.4, 1.0] s
    assert abs(s["stream_tps_p50"] - 11 / 10.5) < 1e-9 and abs(s["tpot_p50_ms"] - 10500 / 11) < 1e-6  # 13 - 2 first
    assert s["mean_out_tokens"] == 7.0 and s["mean_in_tokens"] == 75.0


def test_mock_ttft_decode_itl():
    with mock(lambda p, b: {"content": "w " * 20, "completion_tokens": 20, "prompt_tokens": 33,
                            "delay_s": 0.2, "chunk_delay_s": 0.02}) as url:
        r = speed.one(url, speed.body(M, "chat-en", 1, 0, 0, [], 0.2))
    assert r["error"] is None and (r["n_out"], r["prompt_tokens"], r["finish_reason"]) == (20, 33, "stop")
    assert 0.2 <= r["ttft_s"] < 0.4  # delay_s before the first chunk
    assert 50 < r["decode_tps"] < 18 / 0.18 * 2  # 18 tokens after the first chunk over 9 gaps of 20 ms (loose)
    gaps = sorted(r["gaps"])
    assert len(gaps) == 18 and 0.008 < gaps[9] < 0.016  # 20 ms gap / 2 tokens


def test_stale_pool_retries_on_a_fresh_socket():
    """A c8 cell after a c32 cell found ~30 sockets the server had closed while
    idle; retrying on another pooled one failed the request twice in a row."""
    class Stale:
        def __init__(self):
            self.sock, self.timeout = type("Sock", (), {"settimeout": lambda self, t: None})(), None  # looks reused

        def request(self, *a, **kw):
            raise BrokenPipeError(32, "Broken pipe")

        def close(self):
            self.sock = None

    with mock(lambda p, b: {"content": "w " * 4, "completion_tokens": 4, "prompt_tokens": 5}) as url:
        u = common.urllib.parse.urlsplit(url)
        common._POOL[(u.scheme, u.hostname, u.port)] = [Stale() for _ in range(30)]
        r = speed.one(url, speed.body(M, "chat-en", 1, 0, 0, [], 0.2))
        assert r["error"] is None and r["n_out"] == 4, r
        assert not any(isinstance(c, Stale) for c in common._POOL.get((u.scheme, u.hostname, u.port), []))


def test_cell_metric_deltas():
    out = TMP / "cell.jsonl"
    with mock(lambda p, b: {"content": "w " * 16, "completion_tokens": 16, "chunk_delay_s": 0.005}) as url:
        row = speed.run_cell(url, M, "chat-en", 2, 1, 1.0, 0.1, 0.2, out)
    assert tuple(row) == ROW and row["errors"] == 0 and row["requests"] >= 2 and row["agg_out_tps"] > 0
    # mock counters: accepted = drafts = n//2 per request, draft tokens = 3 * drafts, hits = queries / 2
    assert abs(row["spec_accept_rate"] - 1 / 3) < 1e-3 and row["spec_mean_len"] == 2.0
    assert row["prefix_hit_rate"] == 0.5 and row["preemptions"] is None and row["tool_call_rate"] is None
    reqs = common.read_jsonl(out)
    assert len(reqs) == row["requests"] and all(tuple(r) == REQ for r in reqs) and {r["n_out"] for r in reqs} == {16}


def test_prompts_unique_and_shared_system_prompt():
    seen = []
    with mock(lambda p, b: seen.append(b) or {"content": "ok " * 8, "completion_tokens": 8}) as url:
        for name in ("chat-en", "chat-de", "rag-8k", "multiturn"):
            speed.run_cell(url, M, name, 3, 1, 0.5, 0.05, 0.2)
    users = [b["messages"][0]["content"] for b in seen if b["messages"][0]["role"] == "user"]
    assert len(users) > 30 and len({u[:64] for u in users}) == len(users)  # no shared prefix at all
    rag = [u for u in users if u.endswith(tuple(speed.SUMMARIZE.values()))]
    assert {u.endswith("zusammen.") for u in rag} == {True, False}  # de/en alternate
    assert all(abs(len(u) - 8000 * 3.6) < 200 for u in rag)
    sessions = [b["messages"] for b in seen if b["messages"][0]["role"] == "system"]
    assert len({m[0]["content"] for m in sessions}) == 1 and len(sessions[0][0]["content"]) > 28000  # ~8k tokens
    assert {len(m) for m in sessions} == {2, 4, 6, 8, 10, 12}  # grows a turn per request, restarts after 6
    assert all(m[-2]["role"] == "assistant" and m[-2]["content"] for m in sessions if len(m) > 2)


def test_no_token_ids_env():
    """SGLang answers HTTP 400 to return_token_ids on streaming chat: BENCH_NO_TOKEN_IDS drops
    exactly that field (the request is otherwise byte-identical)."""
    base = speed.body(M, "chat-en", 1, 0, 0, [], 0.2)
    assert base["return_token_ids"] is True
    os.environ["BENCH_NO_TOKEN_IDS"] = "1"
    try:
        off = speed.body(M, "chat-en", 1, 0, 0, [], 0.2)
    finally:
        del os.environ["BENCH_NO_TOKEN_IDS"]
    assert "return_token_ids" not in off and {**off, "return_token_ids": True} == base


def test_run_salt_and_foreign_load():
    a = speed.body(M, "rag-8k", 1, 0, 0, [], 0.2)
    salt, speed.SALT = speed.SALT, "0badc0de"
    try:  # another run (another salt) never repeats a prompt prefix of this one
        b = speed.body(M, "rag-8k", 1, 0, 0, [], 0.2)
        assert a["messages"][0]["content"][:64] != b["messages"][0]["content"][:64]
        assert speed.body(M, "tools", 1, 0, 0, [], 0.2)["messages"][0] == {"role": "system", "content": "Session 0badc0de."}
    finally:
        speed.SALT = salt
    import threading, urllib.request  # noqa: E401
    with mock(lambda p, b: {"content": "w " * 8, "completion_tokens": 8,
                            "delay_s": 1.5 if b.get("max_tokens") == 1 else 0}) as url:
        assert speed.run_cell(url, M, "chat-en", 2, 1, 0.6, 0.05, 0.2)["foreign_load"] == 0  # only our own load
        req = urllib.request.Request(url + "/v1/chat/completions", method="POST", headers={"content-type": "application/json"},
                                     data=json.dumps({"model": M, "messages": [], "max_tokens": 1}).encode())
        threading.Thread(target=lambda: urllib.request.urlopen(req).read(), daemon=True).start()
        import time; time.sleep(0.1)  # noqa: E702 — the foreign request is in flight
        assert speed.run_cell(url, M, "chat-en", 2, 1, 0.6, 0.05, 0.2)["foreign_load"] >= 1


def test_hf_retries():
    """Hugging Face 503s are retried with backoff; a 404 raises at once."""
    import urllib.error
    calls, sleeps = [], []
    sleep, common.time.sleep = common.time.sleep, sleeps.append
    try:
        def flaky():
            calls.append(1)
            if len(calls) < 3:
                raise urllib.error.HTTPError("u", 503, "busy", {}, None)
            return "ok"
        assert common._retrying(flaky) == "ok" and len(calls) == 3 and sleeps == [5, 15]
        def missing():
            raise urllib.error.HTTPError("u", 404, "nope", {}, None)
        try:
            common._retrying(missing)
            raise AssertionError("404 must raise")
        except urllib.error.HTTPError as exc:
            assert exc.code == 404 and sleeps == [5, 15]
    finally:
        common.time.sleep = sleep


def test_bodies_deterministic_tools_vision():
    assert speed.body(M, "rag-8k", 2, 3, 4, [], 0.2) == speed.body(M, "rag-8k", 2, 3, 4, [], 0.2)
    assert speed.body(M, "chat-en", 1, 0, 0, [], 0.2) != speed.body(M, "chat-en", 2, 0, 0, [], 0.2)
    t = speed.body(M, "tools", 1, 0, 0, [], 0.2)
    assert t["ignore_eos"] is False and len(t["tools"]) == 8 and t["tool_choice"] == "auto"
    assert speed.body(M, "thinking", 1, 0, 0, [], 0.2)["max_tokens"] == 2048
    url = speed.body(M, "vision", 1, 0, 0, [], 0.2)["messages"][0]["content"][0]["image_url"]["url"]
    png = base64.b64decode(url.split(",", 1)[1])
    assert png[:8] == b"\x89PNG\r\n\x1a\n" and struct.unpack(">II", png[16:24]) == (448, 448)
    n = struct.unpack(">I", png[33:37])[0]
    assert png[37:41] == b"IDAT" and len(zlib.decompress(png[41:41 + n])) == 448 * (1 + 448 * 3)


def test_errors_are_recorded_not_raised():
    out = TMP / "errors.jsonl"
    with mock(lambda p, b: {"status": 500}) as url:
        row = speed.run_cell(url, M, "chat-en", 2, 1, 0.3, 0.03, 0.2, out)
    assert row["requests"] == 0 and row["errors"] >= 2 and row["agg_out_tps"] == 0
    assert row["ttft_p50_ms"] is None and row["stream_tps_p50"] is None and row["mean_out_tokens"] is None
    assert all("HTTP 500" in r["error"] for r in common.read_jsonl(out))


def test_main_dry_run_skip_and_run():
    posts = []
    responder = lambda p, b: posts.append(b) or {"content": "w " * 6, "completion_tokens": 6,  # noqa: E731
                                                 "chunk_delay_s": 0.002}
    argv = ["--arm", "t", "--scenarios", "chat-en,tools,rag-32k", "--conc", "1,2", "--seconds", "0.4"]
    runs_before = set(common.RUNS.glob("*")) if common.RUNS.exists() else set()
    with mock(responder, max_model_len=16384) as url:
        with contextlib.redirect_stdout(io.StringIO()) as dry:
            assert speed.main(["--url", url, *argv, "--dry-run"]) is None
        assert "skip rag-32k: max_model_len 16384 < 33024 needed" in dry.getvalue() and " 4 cells " in dry.getvalue()
        assert not posts and (set(common.RUNS.glob("*")) if common.RUNS.exists() else set()) == runs_before
        with contextlib.redirect_stdout(io.StringIO()) as log:
            run_dir = speed.main(["--url", url, *argv, "--rounds", "2"])
    rows = json.loads((run_dir / "summary.json").read_text())
    assert [(r["round"], r["scenario"], r["conc"]) for r in rows] == [
        (rnd, s, c) for rnd in (1, 2) for s in ("chat-en", "tools") for c in (1, 2)]  # round-major, rag-32k skipped
    assert all(tuple(r) == ROW for r in rows) and all(r["errors"] == 0 for r in rows)
    assert [r["tool_call_rate"] for r in rows if r["scenario"] == "tools"] == [0.0] * 4  # mock streams no tool_calls
    assert all(r["tool_call_rate"] is None for r in rows if r["scenario"] == "chat-en")
    reqs = common.read_jsonl(run_dir / "requests.jsonl")
    assert all(tuple(r) == REQ for r in reqs) and len(reqs) == sum(r["requests"] for r in rows)
    assert len(posts) == len(reqs) + 2  # one unrecorded warmup per scenario
    ledger = [r for r in common.read_jsonl(common.RESULTS / "speed.jsonl") if r["run_id"] == run_dir.name]
    assert len(ledger) == 8 and ledger[0]["arm"] == "t" and ledger[0]["model"] == M
    assert "skip rag-32k" in log.getvalue() and "| scenario | c | round |" in log.getvalue()
    assert json.loads((run_dir / "manifest.json").read_text())["args"]["ramp"] == 0.04


def test_throttle_yields_to_user_traffic():
    """The politeness gate: a POST waits while the model runs
    more than --yield-above, and goes once the load is back under; GETs (the
    yield poll's own /metrics, /v1/models) never wait."""
    load = {"num_requests_running": 1.0, "num_requests_waiting": 1.0}
    real_metrics = common.metrics
    common.metrics = lambda url: dict(load)  # noqa: E731 — fake a busy model
    try:
        with mock(lambda p, b: {"content": "w", "completion_tokens": 1}) as url:
            common.throttle_configure(url, max_parallel=2, yield_above=2, poll_s=0.05)
            done = []
            worker = threading.Thread(target=lambda: done.append(
                common.request(url, "/v1/chat/completions", {"messages": []}, timeout=5)), daemon=True)
            worker.start()
            worker.join(0.4)
            assert not done and worker.is_alive()  # running + waiting is already at the ceiling
            t0 = time.monotonic()
            common.request(url, "/v1/models")  # a GET is not the model's business: no wait
            assert time.monotonic() - t0 < 0.3
            load["num_requests_waiting"] = 0.0
            worker.join(5)
            assert done, "the request must go once the model is back under the yield line"
    finally:
        common.throttle_configure("", 0, 0)
        common.metrics = real_metrics


def test_throttle_caps_own_requests():
    """At most --max-parallel of the suite's own requests are on the model at once."""
    live, peak, lock = [0], [0], threading.Lock()

    def responder(p, b):
        with lock:
            live[0] += 1
            peak[0] = max(peak[0], live[0])
        time.sleep(0.2)
        with lock:
            live[0] -= 1
        return {"content": "w", "completion_tokens": 1}

    try:
        with mock(responder) as url, mock(responder) as judge_url:
            common.throttle_configure(url, max_parallel=4, judge_url=judge_url)
            threads = [threading.Thread(target=common.request,
                                        args=(endpoint, "/v1/chat/completions", {"m": 1}),
                                        kwargs={"timeout": 5}, daemon=True) for endpoint in (url, judge_url) * 6]
            for t in threads:
                t.start()
            for t in threads:
                t.join(10)
            assert peak[0] == 4, peak[0]
    finally:
        common.throttle_configure("", 0, 0)


def test_main_dry_run_skips_cells_above_the_cap():
    try:
        argv = ["--arm", "t", "--scenarios", "chat-en", "--conc", "1,8", "--seconds", "0.4",
                "--max-parallel", "2"]
        with mock(lambda p, b: {"content": "w"}) as url:
            with contextlib.redirect_stdout(io.StringIO()) as dry:
                assert speed.main(["--url", url, *argv, "--dry-run"]) is None
        assert "skip c8: above --max-parallel 2" in dry.getvalue()
    finally:
        common.throttle_configure("", 0, 0)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    print(f"all speed tests passed (artifacts in {TMP})")
