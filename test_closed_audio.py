"""Synthetic PCM fixtures test the private audio path, including its actual HTTP wire format."""
import base64
from contextlib import ExitStack, redirect_stderr, redirect_stdout
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import io
import json
import os
from pathlib import Path
import struct
import tempfile
import threading
import unittest
from unittest.mock import patch
import wave

import check_response
import closed_audio
import common
import compare
import hidden
import quality


def wav(samples):
    stream = io.BytesIO()
    with wave.open(stream, "wb") as writer:
        writer.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        writer.writeframes(samples)
    return stream.getvalue()


def media(raw):
    return {"mime_type": "audio/wav", "data": base64.b64encode(raw).decode("ascii")}


def encode(rows):
    return b"\n".join(json.dumps(row).encode() for row in rows)


class PCMTests(unittest.TestCase):
    def test_bounds_and_encoding(self):
        raw = wav(bytes(2))
        self.assertEqual(len(raw), 46)
        for size in (2, 32_000, 480_000):
            closed_audio.validate_media(media(wav(bytes(size))))
        encoded = media(raw)["data"]
        # Alternate unused pad bits can decode to the same bytes.
        alias = encoded[:-3] + "B=="
        self.assertEqual(base64.b64decode(alias), raw)
        for m in (None, {}, {**media(raw), "url": "https://example.invalid/"},
                  {**media(raw), "mime_type": "audio/mpeg"},
                  *({"mime_type": "audio/wav", "data": s} for s in
                    (None, 1, "", alias, encoded + "\n", "data:audio/wav;base64," + encoded,
                     "https://example.invalid/test.wav", "Ä", "!" * 20))):
            with self.assertRaises(ValueError):
                closed_audio.validate_media(m)
        with patch.object(base64, "b64decode", side_effect=AssertionError), self.assertRaises(ValueError):
            closed_audio.validate_media({"mime_type": "audio/wav", "data": "A" * 640_061})

    def test_reject_other_codecs_containers_and_chunks(self):
        raw = wav(struct.pack("<hh", -120, 240))
        for bad in (b"", b"ID3", wav(b""), wav(b"\0\1\2"), wav(bytes(480_002)), raw[:-1], raw + b"xx",
                    raw[:36] + b"LIST" + (4).to_bytes(4, "little") + b"test" + raw[36:]):
            with self.assertRaises(ValueError):
                closed_audio.validate_media(media(bad))
        for offset, replacement in (
            (0, b"RIFX"), (4, struct.pack("<I", len(raw))), (8, b"AVI "), (12, b"fact"),
            (16, struct.pack("<I", 18)), (20, struct.pack("<H", 3)), (22, struct.pack("<H", 2)),
            (24, struct.pack("<II", 44_100, 88_200)), (28, struct.pack("<I", 16_000)),
            (32, struct.pack("<H", 1)), (34, struct.pack("<H", 8)), (36, b"LIST"),
            (40, struct.pack("<I", 2)),
        ):
            with self.subTest(offset=offset), self.assertRaises(ValueError):
                closed_audio.validate_media(media(raw[:offset] + replacement + raw[offset + len(replacement):]))


class RunnerTests(unittest.TestCase):
    def setUp(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        self.scratch = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        stack.enter_context(patch.dict(os.environ, {"BENCH_CACHE": str(self.scratch / "private-cache")}))
        stack.enter_context(patch.multiple(common, RUNS=self.scratch / "runs", RESULTS=self.scratch / "results"))
        stack.enter_context(patch.multiple(hidden, FULL_COUNTS={"snapshot_trace": 4}, PLAIN_TASKS=("snapshot_trace",),
                                          VISUAL_COUNTS={"interval_plot": 4}))
        self.rows = []
        for lang in hidden.LANGS:
            for family, count in (hidden.FULL_COUNTS | hidden.VISUAL_COUNTS | hidden.AUDIO_COUNTS).items():
                for n in range(count):
                    row = {"id": f"fake_{family}_{lang}_{n}", "language": lang, "family": family, "mode": "exact",
                           "prompt": f"synthetic-hidden-request-{family}-{lang}", "expected": {"synthetic-key": n}}
                    if family == "tap_timing":
                        row["media"] = media(wav(struct.pack("<hh", n, -n)))
                    if family == "interval_plot":
                        png = quality.png(2, 1, lambda x, y: (x, y, n))
                        row["media"] = {"mime_type": "image/png", "data": base64.b64encode(png).decode()}
                    self.rows.append(row)
        self.payload = encode(self.rows)
        stack.enter_context(patch.object(hidden, "DATA_SHA256", hashlib.sha256(self.payload).hexdigest()))
        self.cache = hidden.cache_path()
        self.cache.parent.mkdir(parents=True)
        self.cache.write_bytes(self.payload)

    def test_registration_and_release_checks(self):
        cells = hidden.release()
        task = hidden.tasks(cells, None)["tap_timing"]
        self.assertEqual(task["counts"], {"quick": {"en": 16, "de": 16}, "full": {"en": 32, "de": 32}})
        self.assertIs(task["score"], hidden.exact)
        self.assertTrue(task["think"])
        self.assertNotIn("tap_timing", (*hidden.PLAIN_TASKS, *hidden.FULL_COUNTS, *quality.TASKS))
        self.assertFalse(any("tap_timing" in c["tasks"] for c in hidden.CATEGORIES))
        a = task["load"]("en", 32, 79)
        b = task["load"]("de", 16, 79)
        self.assertEqual([it["id"] for it in a[:16]], [it["id"] for it in b])
        for en, de in zip(a, b):
            self.assertEqual(en["messages"][0]["content"][0], de["messages"][0]["content"][0])
        for first, second in zip(a[::2], a[1::2]):
            left, right = (it["messages"][0]["content"] for it in (first, second))
            self.assertEqual(left[1], right[1])
            self.assertNotEqual(left[0], right[0])
            self.assertEqual(first["expected"]["synthetic-key"] + 1, second["expected"]["synthetic-key"])
        sources = {family: next(r for r in self.rows if r["family"] == family)
                   for family in ("tap_timing", "interval_plot", "snapshot_trace")}
        audio = sources["tap_timing"]
        image = sources["interval_plot"]
        text = sources["snapshot_trace"]
        for row, edited in ((audio, {k: v for k, v in audio.items() if k != "media"}),
                            (audio, {**audio, "media": image["media"]}), (image, {**image, "media": audio["media"]}),
                            (text, {**text, "media": audio["media"]})):
            payload = encode([edited if r is row else r for r in self.rows])
            with patch.object(hidden, "DATA_SHA256", hashlib.sha256(payload).hexdigest()), self.assertRaisesRegex(
                    ValueError, "^invalid private release row [0-9]+$"):
                hidden.decode(payload)
        with self.assertRaisesRegex(ValueError, "checksum"):
            hidden.decode(self.payload + b"\n")

    def test_selection_and_legacy_resume_defaults(self):
        argv = ["--url", "http://mock", "--arm", "selection", "--no-drift", "--dry-run"]
        out = io.StringIO()
        with redirect_stdout(out), redirect_stderr(out), patch.object(common, "request", side_effect=AssertionError):
            for flags in (["--tasks", "tap_timing"], ["--audio"], ["--audio", "--tasks", "snapshot_trace"],
                          ["--suite", "diagnostics", "--audio", "--tasks", "tap_timing"],
                          ["--suite", "diagnostics", "--audio", "--tasks", "image", "--image"]):
                with self.assertRaises(SystemExit) as err:
                    quality.main([*argv, *flags])
                self.assertEqual(err.exception.code, 2)
            with patch.object(common, "server_info", return_value={"model": "local", "max_model_len": 4096}):
                for flags, names in (
                    ([], ["snapshot_trace"]),
                    (["--image"], ["snapshot_trace", "image"]),
                    (["--audio", "--image", "--tasks", "tap_timing,interval_plot"], ["tap_timing", "interval_plot"]),
                ):
                    out.seek(0)
                    out.truncate()
                    quality.main([*argv, *flags])
                    found = {line.split("|")[1].strip() for line in out.getvalue().splitlines() if "| en |" in line}
                    self.assertEqual(found, set(names))
                prior = self.scratch / "legacy"
                prior.mkdir()
                prior_info = {"args": {"suite": "closed", "tasks": "snapshot_trace", "url": "http://mock",
                                      "arm": "legacy", "no_drift": True},
                              "server": {"model": "local", "max_model_len": 4096}}
                (prior / "manifest.json").write_text(json.dumps(prior_info))
                out.seek(0)
                out.truncate()
                quality.main(["--resume", str(prior), "--audio", "--dry-run"])
                self.assertIn("| snapshot_trace | en |", out.getvalue())
                self.assertNotIn("| tap_timing |", out.getvalue())
            self.assertFalse(common.RUNS.exists())
            self.cache.write_bytes(self.payload + b"\n")
            with self.assertRaises(SystemExit):
                quality.main([*argv, "--audio", "--tasks", "tap_timing"])
        self.assertIn("checksum mismatch", out.getvalue())

    def test_loopback_resume_redaction_and_offline_grading(self):
        calls = []
        types = []
        fail_once = []
        echo = "SYNTHETIC_PRIVATE_ECHO"
        attempts = []
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def send_json(self, status, obj):
                blob = json.dumps(obj).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(blob)))
                self.end_headers()
                self.wfile.write(blob)

            def do_GET(self):
                if self.path == "/v1/models":
                    self.send_json(200, {"data": [{"id": "local", "max_model_len": 4096}]})
                elif self.path == "/version":
                    self.send_json(200, {"version": "mock"})
                else:
                    self.send_json(404, {})

            def do_POST(self):
                attempts.append(self.path)
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                types.append(self.headers.get_content_type())
                calls.append(body)
                parts = body["messages"][0]["content"]
                raw = base64.b64decode(parts[0]["input_audio"]["data"], validate=True)
                with wave.open(io.BytesIO(raw)) as reader:
                    n, _ = struct.unpack("<hh", reader.readframes(2))
                if len(calls) == 2:
                    fail_once.append(n)
                    self.send_json(400, {"error": echo + parts[0]["input_audio"]["data"]})
                else:
                    self.send_json(200, {"choices": [{"message": {"content": json.dumps({"synthetic-key": n}),
                                                                 "reasoning_content": echo},
                                                     "finish_reason": echo}],
                                         "usage": {"prompt_tokens": echo, "completion_tokens": 5}})

        with HTTPServer(("127.0.0.1", 0), Handler) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            self.addCleanup(thread.join, 1)
            args = ["--audio", "--tasks", "tap_timing", "--arm", "offline", "--langs", "de",
                    "--url", f"http://127.0.0.1:{server.server_port}", "--limit", "2",
                    "--no-drift", "--concurrency", "1", "--max-parallel", "1"]
            out = io.StringIO()
            try:
                with redirect_stdout(out), redirect_stderr(out):
                    with self.assertRaisesRegex(SystemExit, "unscored"):
                        quality.main(args)
                    (run,) = list(common.RUNS.iterdir())
                    self.assertTrue(json.loads((run / "manifest.json").read_text())["args"]["audio"])
                    self.assertEqual(quality.main(["--resume", str(run), "--concurrency", "1"]), run)
                    self.assertEqual(quality.main(["--resume", str(run), "--concurrency", "1"]), run)
            finally:
                server.shutdown()
        self.assertEqual(types, ["application/json"] * 3)
        self.assertEqual(attempts, ["/v1/chat/completions"] * 3)
        self.assertEqual(len(fail_once), 1)
        self.assertEqual(calls[1], calls[2])
        self.assertNotEqual(calls[0]["messages"], calls[1]["messages"])
        for body in calls:
            self.assertEqual(len(body["messages"]), 1)
            user, = body["messages"]
            self.assertEqual(user["role"], "user")
            self.assertEqual(len(user["content"]), 2)
            audio_part, prompt_part = user["content"]
            self.assertEqual(set(audio_part), {"type", "input_audio"})
            self.assertEqual(audio_part["type"], "input_audio")
            self.assertEqual(set(audio_part["input_audio"]), {"data", "format"})
            self.assertEqual(audio_part["input_audio"]["format"], "wav")
            self.assertEqual(prompt_part, {"type": "text", "text": "synthetic-hidden-request-tap_timing-de"})
        self.assertEqual(json.loads((run / "run_info.json").read_text())["expected_items"], 2)
        records = common.read_jsonl(run / "items-tap_timing.jsonl")
        self.assertEqual([r["correct"] for r in records], [True, None, True])
        for r in records:
            self.assertIsNone(r["reply"])
            self.assertIsNone(r["pred"])
            self.assertIsNone(r["gold"])
            self.assertIsNone(r["finish_reason"])
            self.assertIsNone(r["prompt_tokens"])
        self.assertEqual(records[1]["error"], "HTTPError")
        summary, = json.loads((run / "summary.json").read_text())
        self.assertEqual(summary["errors"], 0)
        self.assertEqual(summary["extra"]["answer_components_correct"], 1.0)
        expected = hidden.load(hidden.release(), "tap_timing", "de", 1, quality.SEED)[0]
        answer = self.scratch / "response.json"
        answer.write_text(json.dumps({"choices": [{"message": {"content": json.dumps(expected["expected"])}}]}))
        with redirect_stdout(out), patch.object(common, "request", side_effect=AssertionError):
            code = check_response.main(["--task", "tap_timing", "--lang", "de", "--id", expected["id"],
                                        "--response", str(answer)])
        self.assertEqual(code, 0)
        public = out.getvalue() + "".join(p.read_text() for root in (common.RUNS, common.RESULTS)
                                         for p in root.rglob("*") if p.is_file())
        for canary in (echo, "synthetic-hidden-request", "synthetic-key", "input_audio",
                       *(p["messages"][0]["content"][0]["input_audio"]["data"] for p in calls)):
            self.assertNotIn(canary, public)
        manifest = json.loads((run / "manifest.json").read_text())
        modified = {**manifest, "args": {**manifest["args"], "audio": False}}
        by_id = {(r["task"], r["lang"], r["id"]): r for r in records if not r["error"]}
        self.assertEqual(compare.hidden_protocol([(run, manifest), (run, manifest)], [by_id, by_id]), [])
        self.assertIn("hidden benchmark argument audio differs",
                      compare.hidden_protocol([(run, manifest), (run, modified)], [by_id, by_id]))


if __name__ == "__main__":
    unittest.main()
