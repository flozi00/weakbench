"""A tiny OpenAI-compatible server for hermetic suite tests (stdlib only).

    url, stop = serve(responder)        # responder(path, body) -> dict
    ... point a suite at url ...
    stop()

The responder returns {"content", "reasoning", "tool_calls", "finish_reason",
"prompt_tokens", "completion_tokens", "prompt_logprobs", "prompt_token_ids",
"delay_s", "chunk_delay_s", "status"} (all optional). Chat/completions
answer streamed (2 token ids per chunk, usage on
the last chunk) or whole; /v1/models, /version and /metrics are canned, and
the spec-decode counters advance with every generated token;
vllm:num_requests_running is the live count of in-flight POSTs.
"""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL = "mock-model"


def serve(responder, max_model_len=32768):
    counters = {"gen": 0, "drafts": 0, "accepted": 0, "running": 0, "lock": threading.Lock()}

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"  # keep-alive + chunked SSE, like vLLM behind uvicorn

        def log_message(self, *a):  # keep test output clean
            pass

        def _chunk(self, data):
            self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
            self.wfile.flush()

        def _send(self, code, payload, ctype="application/json"):
            raw = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("content-type", ctype)
            self.send_header("content-length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            if self.path == "/v1/models":
                return self._send(200, {"data": [{"id": MODEL, "max_model_len": max_model_len}]})
            if self.path == "/version":
                return self._send(200, {"version": "mock-0.0"})
            if self.path == "/metrics":
                with counters["lock"]:
                    g, d, a, run = counters["gen"], counters["drafts"], counters["accepted"], counters["running"]
                text = (f'vllm:num_requests_running{{model_name="{MODEL}"}} {run}\n'
                        f'vllm:num_requests_waiting{{model_name="{MODEL}"}} 0\n'
                        f'vllm:generation_tokens_total{{model_name="{MODEL}"}} {g}\n'
                        f'vllm:spec_decode_num_drafts_total{{model_name="{MODEL}"}} {d}\n'
                        f'vllm:spec_decode_num_draft_tokens_total{{model_name="{MODEL}"}} {d * 3}\n'
                        f'vllm:spec_decode_num_accepted_tokens_total{{model_name="{MODEL}"}} {a}\n'
                        f'vllm:prefix_cache_queries_total{{model_name="{MODEL}"}} {g * 2}\n'
                        f'vllm:prefix_cache_hits_total{{model_name="{MODEL}"}} {g}\n')
                return self._send(200, text.encode(), "text/plain")
            self._send(404, {"error": "not found"})

        def do_POST(self):
            with counters["lock"]:
                counters["running"] += 1
            try:
                self._post()
            finally:
                with counters["lock"]:
                    counters["running"] -= 1

        def _post(self):
            body = json.loads(self.rfile.read(int(self.headers.get("content-length") or 0)) or b"{}")
            if self.path not in ("/v1/chat/completions", "/v1/completions"):
                return self._send(404, {"error": "not found"})
            r = responder(self.path, body) or {}
            if r.get("status"):
                return self._send(r["status"], {"error": r.get("error", "mock error")})
            time.sleep(r.get("delay_s", 0))
            content = r.get("content", "")
            n_out = r.get("completion_tokens") or max(1, len(content.split()))
            usage = {"prompt_tokens": r.get("prompt_tokens", 10), "completion_tokens": n_out,
                     "total_tokens": r.get("prompt_tokens", 10) + n_out}
            with counters["lock"]:
                counters["gen"] += n_out
                counters["drafts"] += n_out // 2
                counters["accepted"] += n_out // 2
            chat = self.path == "/v1/chat/completions"
            if not body.get("stream"):
                choice = {"index": 0, "finish_reason": r.get("finish_reason", "stop")}
                if chat:
                    choice["message"] = {"role": "assistant", "content": content,
                                         "reasoning_content": r.get("reasoning"),
                                         "tool_calls": r.get("tool_calls") or []}
                else:
                    choice["text"] = content
                payload = {"id": "mock", "model": MODEL, "choices": [choice], "usage": usage}
                for key in ("prompt_logprobs", "prompt_token_ids"):
                    if key in r:
                        payload[key] = r[key]
                return self._send(200, payload)
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.send_header("transfer-encoding", "chunked")
            self.end_headers()
            words = content.split(" ") if content else []
            ids = list(range(n_out))
            pieces = max(1, (n_out + 1) // 2)
            calls = r.get("tool_calls") or []
            for i in range(pieces):
                text = " ".join(words[i * len(words) // pieces:(i + 1) * len(words) // pieces])
                delta = {"content": text} if chat else None
                if chat and i == 0 and r.get("reasoning"):
                    delta["reasoning"] = r["reasoning"]  # vLLM streams it as delta.reasoning
                if chat and i == pieces - 1 and calls:
                    delta["tool_calls"] = [{"index": k, **c} for k, c in enumerate(calls)]
                ch = {"index": 0, "token_ids": ids[i * 2:i * 2 + 2],
                      **({"delta": delta} if chat else {"text": text})}
                if i == pieces - 1:
                    ch["finish_reason"] = r.get("finish_reason", "tool_calls" if calls else "stop")
                self._chunk(f"data: {json.dumps({'choices': [ch]})}\n\n".encode())
                time.sleep(r.get("chunk_delay_s", 0))
            self._chunk(f"data: {json.dumps({'choices': [], 'usage': usage})}\n\n".encode())
            self._chunk(b"data: [DONE]\n\n")
            self._chunk(b"")  # last chunk: the connection stays open for the next request

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{server.server_address[1]}", server.shutdown


if __name__ == "__main__":
    import urllib.request
    url, stop = serve(lambda path, body: {"content": "hello there world", "completion_tokens": 3})
    req = urllib.request.Request(url + "/v1/chat/completions", method="POST",
                                 data=json.dumps({"stream": True, "messages": []}).encode(),
                                 headers={"content-type": "application/json"})
    lines = [ln for ln in urllib.request.urlopen(req).read().decode().splitlines() if ln.startswith("data:")]
    assert lines[-1] == "data: [DONE]" and '"usage"' in lines[-2], lines
    assert json.loads(urllib.request.urlopen(url + "/v1/models").read())["data"][0]["id"] == MODEL
    stop()
    call = {"type": "function", "function": {"name": "f", "arguments": "{}"}}
    url, stop = serve(lambda path, body: {"content": "x", "tool_calls": [call], "reasoning": "hm"})
    req = urllib.request.Request(url + "/v1/chat/completions", method="POST",
                                 data=json.dumps({"stream": True, "messages": []}).encode(),
                                 headers={"content-type": "application/json"})
    chunks = [json.loads(ln[5:]) for ln in urllib.request.urlopen(req).read().decode().splitlines()
              if ln.startswith("data:") and not ln.endswith("[DONE]")]
    last = chunks[-2]["choices"][0]
    assert last["delta"]["tool_calls"][0]["function"]["name"] == "f" and last["finish_reason"] == "tool_calls"
    assert chunks[0]["choices"][0]["delta"]["reasoning"] == "hm"
    stop()
    print("mock_server self-check OK")
