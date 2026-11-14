"""tasks_rag (MultiHop-RAG + BM25) on the real pinned data, plain asserts. Stdlib only:

    python3 test_tasks_rag.py

The last test drives `quality.run_item` against a fake OpenAI server, so the tool loop itself is
exercised — a loop that never executes the tool would still pass every scoring test above it.
"""
import json
import threading
from collections import Counter
from http.server import BaseHTTPRequestHandler, HTTPServer

import quality
import tasks_rag as tr

reply = lambda text, **kw: {"content": text, "reasoning": "", "tool_calls": [], "finish_reason": "stop", **kw}  # noqa: E731


def test_counts_prefix_and_balance():
    full = tr.TASKS["search_rag"]["load"]("en", None, 0)
    quick = tr.TASKS["search_rag"]["load"]("en", 150, 0)
    assert len(full) == 2556 and len(quick) == 150
    assert [i["id"] for i in quick] == [i["id"] for i in full][:150]  # quick is a prefix of full
    assert Counter(i["question_type"] for i in full) == Counter(
        {"comparison_query": 856, "inference_query": 816, "temporal_query": 583, "null_query": 301})
    # round-robin, so even the 150-item quick tier covers all four types
    assert set(Counter(i["question_type"] for i in quick).values()) == {37, 38}
    assert {i["lang"] for i in full} == {"en"}
    for i in full[:20]:
        assert i["tools"] == tr.TOOLS and callable(i["tool_exec"]) and i["max_rounds"] == 4
        assert i["messages"][0]["role"] == "system" and "Insufficient information" in i["messages"][0]["content"]


def test_bm25_finds_the_gold_evidence():
    """The index must be good enough that a miss is the model's query, not the retriever."""
    full = tr.TASKS["search_rag"]["load"]("en", None, 0)
    checked = 0
    for i in full:
        if not i["evidence"]:
            continue
        got = {d["title"] for d in tr._index().search(i["messages"][1]["content"], 10)}
        assert got & set(i["evidence"]), i["id"]  # the question text alone already retrieves evidence
        checked += 1
        if checked == 40:
            break
    assert checked == 40


def test_tool_exec():
    out = tr._exec("search", json.dumps({"query": "Sam Bankman-Fried FTX trial", "k": 3}))
    assert out.count("[") >= 3 and "FTX" in out
    assert tr._exec("search", "not json").startswith("error:")
    assert tr._exec("lookup", "{}").startswith("error: unknown tool")
    assert tr._exec("search", json.dumps({"query": "zzzzqqqq xxyyzz"})) == "No articles matched that query."
    # k is clamped, and a non-numeric k falls back to 5 instead of raising
    assert tr._exec("search", json.dumps({"query": "Google", "k": 99})).count("[") == 10
    assert tr._exec("search", json.dumps({"query": "Google", "k": "many"})).count("[") == 5


def test_scoring():
    full = tr.TASKS["search_rag"]["load"]("en", None, 0)
    score = tr.TASKS["search_rag"]["score"]
    inf = next(i for i in full if i["question_type"] == "inference_query")
    assert score(inf, reply(f"The answer is {inf['gold']}."))["correct"]
    r = score(inf, reply("The answer is somebody else entirely."))
    assert not r["correct"] and r["score"] == -1.0 and r["detail"]["verdict"] == "INCORRECT"
    # whole-word only: a gold that is a substring of a longer word must not count
    assert not score(inf, reply(f"Not{inf['gold']}x"))["correct"]
    # null_query: abstaining is CORRECT, inventing is a hallucination
    null = next(i for i in full if i["question_type"] == "null_query")
    assert null["gold"] == "Insufficient information." and null["evidence"] == []
    r = score(null, reply("Insufficient information."))
    assert r["correct"] and r["score"] == 1.0 and r["detail"]["verdict"] == "CORRECT"
    r = score(null, reply("Answer: Google"))
    assert not r["correct"] and r["score"] == -1.0 and r["detail"]["verdict"] == "HALLUCINATED"
    # Yes/No gold: the polarity of the last sentence decides
    yes = next(i for i in full if i["gold"] == "Yes")
    assert score(yes, reply("Both articles report it. Answer: Yes"))["correct"]
    assert not score(yes, reply("No, they differ. Answer: No"))["correct"]
    no = next(i for i in full if i["gold"] == "no")
    assert score(no, reply("They do not. Answer: No"))["correct"]
    assert not score(no, reply("Yes, both do."))["correct"]


class _Fake(BaseHTTPRequestHandler):
    """Two rounds: a tool call, then a final answer. Records what the client sent."""

    seen = []
    always_tools = False  # when True, every round calls the tool, so the cap is hit

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Fake.seen.append(body)
        if (_Fake.always_tools and "Do not call any tool" not in json.dumps(body["messages"][-1])) or (
                not _Fake.always_tools and len(_Fake.seen) == 1):
            msg = {"role": "assistant", "content": "", "tool_calls": [
                {"id": "x", "type": "function",
                 "function": {"name": "search", "arguments": json.dumps({"query": "FTX trial", "k": 2})}}]}
            out = {"choices": [{"message": msg, "finish_reason": "tool_calls"}], "usage": {"prompt_tokens": 10,
                                                                                          "completion_tokens": 5}}
        else:
            out = {"choices": [{"message": {"role": "assistant", "content": "Answer: Sam Bankman-Fried"},
                                "finish_reason": "stop"}], "usage": {"prompt_tokens": 20, "completion_tokens": 7}}
        raw = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, format, *args):  # noqa: A002 — the base class's signature
        pass


def _run(item, port):
    body = quality.build_body("search_rag", tr.TASKS["search_rag"], item, _Args(), "m")
    return quality.run_item(f"http://127.0.0.1:{port}", "search_rag", tr.TASKS["search_rag"], item, body, 30)


def test_run_item_tool_loop():
    """The loop in quality.run_item: the tool really runs, its result reaches the model, usage sums."""
    _Fake.seen, _Fake.always_tools = [], False
    srv = HTTPServer(("127.0.0.1", 0), _Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        item = tr.TASKS["search_rag"]["load"]("en", 1, 0)[0]
        rec = _run(item, srv.server_port)
    finally:
        srv.shutdown()
    assert rec["error"] is None, rec["error"]
    assert len(_Fake.seen) == 2  # tool call, then the answer
    assert _Fake.seen[0]["tools"] == tr.TOOLS  # the schema went out with the request
    second = _Fake.seen[1]["messages"]
    assert second[-1]["role"] == "tool" and "FTX" in second[-1]["content"]  # the real BM25 result
    assert second[-2]["role"] == "assistant" and second[-2]["tool_calls"][0]["id"] == "call_0"
    assert rec["prompt_tokens"] == 30 and rec["completion_tokens"] == 12  # summed over both rounds
    assert rec["finish_reason"] == "stop"
    assert rec["detail"]["tool_rounds"] == 1 and rec["detail"]["tool_forced"] is False


def test_run_item_forces_an_answer_at_the_cap():
    """A model that never stops searching must still be scored on an answer, not on an empty reply."""
    _Fake.seen, _Fake.always_tools = [], True
    srv = HTTPServer(("127.0.0.1", 0), _Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        item = tr.TASKS["search_rag"]["load"]("en", 1, 0)[0]
        rec = _run(item, srv.server_port)
    finally:
        srv.shutdown()
        _Fake.always_tools = False
    assert rec["error"] is None, rec["error"]
    assert len(_Fake.seen) == item["max_rounds"] + 1  # the cap, then one forced answer
    nudge = _Fake.seen[-1]["messages"][-1]
    assert nudge["role"] == "user" and "Do not call any tool" in nudge["content"]  # the forced round
    assert rec["detail"]["tool_rounds"] == item["max_rounds"] and rec["detail"]["tool_forced"] is True
    assert rec["reply"]  # an answer, not the empty content of the last tool-call round


class _Args:
    think, max_tokens, temperature = "auto", None, None


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
