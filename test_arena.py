"""Synthetic local persistence, fixture, artifact and resume tests with fake API responses."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, redirect_stdout
import fcntl
import hashlib
import io
import json
import os
import select
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
from threading import Event, Lock, Thread
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import arena


def case(id="q1", kind="text", prompt="Describe a fictional stage.", rubric=None):
    return {
        "id": id,
        "kind": kind,
        "language": "en",
        "prompt": prompt,
        "rubric": rubric or ["clarity"],
    }


def completion(content, finish="stop", **extra):
    return {
        "choices": [
            {
                "message": {"role": "assistant", "content": content, **extra},
                "finish_reason": finish,
            }
        ]
    }


class ArenaTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db = Path(self.tmp.name) / "season.sqlite3"
        self.a = arena.Arena(self.db)
        self.a.register_cases([case()])
        self.a1 = self.a.register_model("atlas", "r1")
        self.a2 = self.a.register_model("birch", "r2")
        self.j = self.a.register_judge("referee", "j1")
        tls = patch.object(arena.common, "verify_tls")
        tls.start()
        self.addCleanup(tls.stop)

    @staticmethod
    def transport(url, path, body, **args):
        assert (
            path == "/v1/chat/completions"
            and args["retries"] == 0
            and args["reject_redirects"] is True
        )
        if body["messages"][0]["role"] == "user":
            return completion(
                "entry-" + body["model"] + " " + body["messages"][0]["content"]
            )
        data = json.loads(body["messages"][1]["content"])
        return completion(
            "A" if data["A"] > data["B"] else "B" if data["A"] < data["B"] else "TIE"
        )

    def gen(self):
        with patch.object(arena.common, "request", side_effect=self.transport):
            for k, origin in (
                (self.a1, "http://model-a.test"),
                (self.a2, "http://model-b.test"),
            ):
                self.a.generate(k, origin)

    def test_incremental_models_and_judges(self):
        self.a.register_cases([case(), case("q2", prompt="Draft an ending.")])
        self.gen()
        with patch.object(arena.common, "request", side_effect=self.transport) as net:
            first = self.a.evaluate(self.j, "http://judge.test")
            self.assertEqual((first["completed"], net.call_count), (2, 4))
            net.reset_mock()
            self.assertEqual(
                arena.Arena(self.db).evaluate(self.j, "http://judge.test")["cached"], 2
            )
            self.assertEqual(
                self.a.generate(self.a1, "http://model-a.test")["cached"], 2
            )
            self.assertEqual(net.call_count, 0)
            c = self.a.register_model("cobalt", "r3", {"temperature": 0.3})
            provisional = self.a.standings(self.j)
            self.assertTrue(
                all(r["provisional"] and not r["balanced"] for r in provisional)
            )
            self.assertEqual(sum(r["pending"] for r in provisional), 8)
            self.assertEqual(self.a.generate(c, "http://model-c.test")["created"], 2)
            self.assertEqual(
                self.a.evaluate(self.j, "http://judge.test")["completed"], 4
            )
            self.assertEqual(net.call_count, 10)
            self.assertEqual(len(self.a.responses()), 6)
            ranked = self.a.standings(self.j)
            self.assertEqual([r["points"] for r in ranked], [12, 6, 0])
            self.assertTrue(all(r["played"] == 4 and not r["pending"] for r in ranked))
            fresh = self.a.register_judge("referee", "j2")
            net.reset_mock()
            self.assertEqual(
                self.a.evaluate(fresh, "http://new-judge.test")["completed"], 6
            )
            self.assertEqual(net.call_count, 12)
            self.assertTrue(
                all(c.args[2]["model"] == "referee" for c in net.call_args_list)
            )

    def test_invalid_second_vote_resumes_without_first_or_regeneration(self):
        self.gen()
        with patch.object(
            arena.common,
            "request",
            side_effect=[completion("A"), completion("A and B")],
        ):
            res = self.a.evaluate(self.j, "http://judge.test")
            self.assertEqual((res["failed"], res["votes_created"]), (1, 1))
        m = self.a.matches(self.j)[0]
        self.assertEqual((m["outcome"], m["votes"]), ("pending", ["A", None]))
        with patch.object(arena.common, "request", side_effect=TimeoutError):
            self.assertEqual(
                arena.Arena(self.db).evaluate(self.j, "http://judge.test")["failed"], 1
            )
        with patch.object(
            arena.common, "request", return_value=completion("<think>hidden</think>B")
        ) as net:
            self.assertEqual(
                arena.Arena(self.db).evaluate(self.j, "http://judge.test")["completed"],
                1,
            )
            self.assertEqual(net.call_count, 1)
        self.assertEqual(self.a.matches(self.j)[0]["outcome"], "left")
        self.assertEqual([r["points"] for r in self.a.standings(self.j)], [3, 0])

    def test_draw_vs_order_disagreement(self):
        self.gen()
        for i, (votes, expected) in enumerate(
            [
                (("TIE", "TIE"), "draw"),
                (("A", "A"), "disagreement"),
                (("B", "TIE"), "disagreement"),
            ]
        ):
            k = self.a.register_judge("referee", str(i))
            with patch.object(
                arena.common, "request", side_effect=list(map(completion, votes))
            ):
                self.a.evaluate(k, "http://judge.test")
            self.assertEqual(self.a.matches(k)[0]["outcome"], expected)
            self.assertTrue(
                all(
                    (t["points"], t["draws"], t["disagreements"])
                    == (1, 1, int(expected == "disagreement"))
                    for t in self.a.standings(k)
                )
            )

    def test_visible_full_text_private_and_anonymous(self):
        big = "  " + ("A small bead. " * 900) + "\n"
        raw = (
            "<think>Internal-A</think>"
            + big[:5000]
            + "<think>Internal-B</think>"
            + big[5000:]
        )
        with patch.object(
            arena.common,
            "request",
            return_value=completion(raw, reasoning_content="Native-secret"),
        ):
            self.a.generate(self.a1, "http://model-a.test")
        with patch.object(arena.common, "request", return_value=completion("Blue")):
            self.a.generate(self.a2, "http://model-b.test")
        got = self.a.response(self.a1, "q1")
        self.assertEqual(
            got,
            {
                "content": big,
                "finish_reason": "stop",
                "images": [],
                "render_status": "not_applicable",
            },
        )
        self.assertEqual(
            self.a.responses()[
                [r["model_key"] for r in self.a.responses()].index(self.a1)
            ]["answer_digest"],
            hashlib.sha256(big.encode()).hexdigest(),
        )
        with patch.object(
            arena.common, "request", return_value=completion("TIE", reasoning="Ignored")
        ) as net:
            self.a.evaluate(self.j, "http://judge.test")
        views = [
            json.loads(c.args[2]["messages"][1]["content"]) for c in net.call_args_list
        ]
        self.assertEqual(views[0]["A"], views[1]["B"])
        self.assertEqual(views[0]["B"], views[1]["A"])
        self.assertEqual({views[0]["A"], views[0]["B"]}, {"Blue", big})
        self.assertEqual(set(views[0]), {"A", "B", "rubric", "challenge"})
        with closing(sqlite3.connect(self.db)) as local_db:
            private_dump = " ".join(local_db.iterdump())
        for secret in ("Native-secret", "Internal-A", "Internal-B", "Ignored"):
            self.assertNotIn(secret, private_dump)
        self.assertNotIn(big, str(self.a.inventory()))
        self.assertNotIn("rubric", str(self.a.inventory()))
        self.assertEqual(self.db.stat().st_mode & 0o777, 0o600)

    def test_generation_failure_and_model_revision(self):
        for raw in (
            completion("unknown", "content_filter"),
            completion(None),
            completion("<think>unfinished"),
        ):
            with patch.object(arena.common, "request", return_value=raw):
                self.assertEqual(
                    self.a.generate(self.a1, "http://model-a.test")["failed"], 1
                )
                self.assertIsNone(self.a.response(self.a1, "q1"))
        with patch.object(
            arena.common, "request", return_value=completion("completed")
        ) as net:
            self.assertEqual(
                self.a.generate(self.a1, "http://model-a.test")["created"], 1
            )
            self.assertEqual(
                self.a.register_model("atlas", "r1", {"temperature": 0}), self.a1
            )
            for model in (
                self.a.register_model("atlas", "r3"),
                self.a.register_model("atlas", "r1", {"seed": 24}),
            ):
                self.assertNotEqual(model, self.a1)
                self.assertEqual(
                    self.a.generate(model, "http://model-a.test")["created"], 1
                )
            self.assertEqual(net.call_count, 3)

    def test_rubric_and_generation_fingerprints(self):
        self.gen()
        with patch.object(arena.common, "request", side_effect=self.transport) as net:
            self.a.evaluate(self.j, "http://judge.test")
            orig = self.a.matches(self.j)[0]["id"]
            self.a.register_cases([dict(case(rubric=["new rubric"]), rubric_version=4)])
            self.assertEqual(self.a.matches(self.j)[0]["outcome"], "pending")
            for key in (self.a1, self.a2):
                self.assertEqual(self.a.generate(key, "http://model.test")["cached"], 1)
            self.assertEqual(net.call_count, 2)
            self.a.evaluate(self.j, "http://judge.test")
            self.assertEqual(net.call_count, 4)
            self.a.register_cases([case()])
            self.assertEqual(self.a.matches(self.j)[0]["id"], orig)
            self.assertEqual(self.a.evaluate(self.j, "http://judge.test")["cached"], 1)
            self.a.register_cases([case(prompt="New prompt")])
            self.assertIsNone(self.a.response(self.a1, "q1"))
            self.assertEqual(
                self.a.evaluate(self.j, "http://judge.test")["missing_answers"], 1
            )
            self.assertEqual(net.call_count, 4)

    def test_shared_gateway_and_no_self_evaluation(self):
        with patch.object(arena.common, "request", side_effect=self.transport) as net:
            base = "http://gateway.test"
            self.a.generate(self.a1, base)
            self.a.generate(self.a2, base)
            self.assertEqual(self.a.evaluate(self.j, base)["completed"], 1)
            self.assertEqual(net.call_count, 4)
            same = self.a.register_judge("atlas", "r1", {"seed": 43})
            self.assertEqual(
                self.a.evaluate(same, "http://neutral.test")["self_judge"], 1
            )
            self.assertEqual(net.call_count, 4)

    def test_length_is_a_candidate_submission_but_not_a_judge_vote(self):
        with patch.object(
            arena.common,
            "request",
            return_value=completion("Full delivered prefix", "length"),
        ) as net:
            for key in (self.a1, self.a2):
                self.assertEqual(
                    self.a.generate(key, "http://gateway.test")["created"], 1
                )
                got = self.a.response(key, "q1")
                self.assertEqual(
                    (got["content"], got["finish_reason"]),
                    ("Full delivered prefix", "length"),
                )
                self.assertEqual(
                    self.a.generate(key, "http://gateway.test")["cached"], 1
                )
            self.assertEqual(net.call_count, 2)
        with patch.object(
            arena.common, "request", return_value=completion("A", "length")
        ) as net:
            self.assertEqual(
                self.a.evaluate(self.j, "http://gateway.test")["failed"], 1
            )
            shown = json.loads(net.call_args.args[2]["messages"][1]["content"])
            self.assertEqual(shown["completion_status"], {"A": "length", "B": "length"})
            self.assertEqual(self.a.matches(self.j)[0]["outcome"], "pending")

    def test_bare_v1_and_prefixed_urls_for_both_roles(self):
        for i, (base, route) in enumerate(
            (
                ("http://gate.test", "/v1/chat/completions"),
                ("http://gate.test/v1/", "/v1/chat/completions"),
                ("http://gate.test/proxy/v1", "/proxy/v1/chat/completions"),
                ("http://gate.test/proxy/", "/proxy/v1/chat/completions"),
            )
        ):
            self.a.register_cases([case("route" + str(i))])
            with patch.object(
                arena.common, "request", return_value=completion("TIE")
            ) as net:
                self.a.generate(self.a1, base)
                self.a.generate(self.a2, base)
                self.a.evaluate(self.j, base)
                self.assertEqual(net.call_count, 4)
                self.assertEqual(
                    [call.args[0] + call.args[1] for call in net.call_args_list],
                    ["http://gate.test" + route] * 4,
                )

    def test_redirect_completion_body_is_rejected_without_following(self):
        requests = []

        class Endpoint(BaseHTTPRequestHandler):
            def do_POST(self):
                requests.append(self.path)
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
                payload = json.dumps(completion("redirect body")).encode()
                self.send_response(307)
                self.send_header("Location", "/other-origin-or-path")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass

        with ThreadingHTTPServer(("127.0.0.1", 0), Endpoint) as server:
            worker = Thread(
                target=lambda: server.serve_forever(poll_interval=0.01), daemon=True
            )
            worker.start()
            try:
                result = self.a.generate(
                    self.a1, f"http://127.0.0.1:{server.server_port}"
                )
                self.assertEqual(result["failed"], 1)
                self.assertIsNone(self.a.response(self.a1, "q1"))
                self.assertEqual(requests, ["/v1/chat/completions"])
            finally:
                server.shutdown()
                worker.join(1)

    def test_concurrent_read_and_exclusive_resume(self):
        entered, release = Event(), Event()

        def paused(*args, **kw):
            entered.set()
            if not release.wait(5):
                raise RuntimeError("test release missing")
            return completion("stored once")

        with (
            patch.object(arena.common, "request", side_effect=paused) as net,
            ThreadPoolExecutor(2) as pool,
        ):
            first = pool.submit(self.a.generate, self.a1, "http://model-a.test")
            try:
                self.assertTrue(entered.wait(3))
                other = arena.Arena(self.db)
                self.assertEqual(len(other.inventory()["models"]), 2)
                self.assertIsNone(other.response(self.a1, "q1"))
                second = pool.submit(other.generate, self.a1, "http://model-a.test")
                # An unrelated process must observe the same lock during the remote call.
                cmd = "import fcntl,sys; f=open(sys.argv[1],'rb');\ntry: fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)\nexcept BlockingIOError: sys.exit(4)"
                check = subprocess.run(
                    [sys.executable, "-c", cmd, str(self.db) + ".lock"],
                    check=False,
                    timeout=3,
                )
                self.assertEqual(check.returncode, 4)
            finally:
                release.set()
            self.assertEqual(
                (first.result()["created"], second.result()["cached"]), (1, 1)
            )
            self.assertEqual(net.call_count, 1)

    def test_visual_cache_rerender_and_failure_retry(self):
        self.a.register_cases([case(kind="web")])
        rev = ["rOne"]
        unavailable = [False]

        def renderer_key():
            if unavailable[0] == "sentinel":
                return "harness:unavailable"
            if unavailable[0]:
                raise RuntimeError("offline renderer")
            return rev[0]

        def prepare(c, s):
            if s == "broken":
                return {"status": "render_error"}
            return {
                "schema": 1,
                "status": "ok",
                "renderer_key": rev[0],
                "kind": c["kind"],
                "views": [
                    {"name": "desktop", "width": 1, "height": 1, "png_base64": "cG5n"}
                ],
                "diagnostic": "ok",
            }

        renderer = SimpleNamespace(
            renderer_key=renderer_key,
            prepare_artifact=Mock(side_effect=prepare),
            judge_parts=lambda b: [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64," + b["views"][0]["png_base64"]
                    },
                }
            ],
        )
        with patch.dict(sys.modules, arena_render=renderer):
            self.gen()
            self.assertEqual(renderer.prepare_artifact.call_count, 2)
            self.assertEqual(
                self.a.response(self.a1, "q1")["images"], ["data:image/png;base64,cG5n"]
            )
            with patch.object(
                arena.common, "request", return_value=completion("TIE")
            ) as net:
                self.assertEqual(
                    self.a.evaluate(self.j, "http://judge.test")["completed"], 1
                )
                view = net.call_args.args[2]["messages"][1]["content"]
                self.assertEqual(sum(p["type"] == "image_url" for p in view), 2)
                for call in net.call_args_list:
                    parts = call.args[2]["messages"][1]["content"]
                    submissions = [
                        json.loads(p["text"])
                        for p in parts
                        if p["type"] == "text" and "untrusted_source" in p["text"]
                    ]
                    self.assertEqual([d["submission"] for d in submissions], ["A", "B"])
                    self.assertEqual(
                        {d["untrusted_source"] for d in submissions},
                        {r["text"] for r in self.a.responses()},
                    )
                    self.assertTrue(
                        all(
                            set(d) == {"submission", "untrusted_source"}
                            for d in submissions
                        )
                    )
                second = self.a.register_judge("second", "r3")
                previous = self.a.matches(self.j)
                unavailable[0] = "sentinel"
                self.assertEqual(
                    self.a.generate(self.a1, "http://model-a.test")["render_pending"], 0
                )
                self.assertEqual(
                    self.a.evaluate(self.j, "http://judge.test")["cached"], 1
                )
                self.assertEqual(self.a.matches(self.j), previous)
                self.assertEqual(self.a.response(self.a1, "q1")["render_status"], "ok")
                unavailable[0] = True
                self.assertEqual(
                    self.a.evaluate(second, "http://judge.test")["completed"], 1
                )
                self.assertEqual(renderer.prepare_artifact.call_count, 2)
                unavailable[0] = False
                rev[0] = "rTwo"
                self.a.evaluate(self.j, "http://judge.test")
                self.assertEqual(renderer.prepare_artifact.call_count, 4)
                self.assertEqual(net.call_count, 6)
                self.assertEqual(len(self.a.responses()), 2)
                self.assertTrue(
                    all(
                        c.args[2]["model"] in ("referee", "second")
                        for c in net.call_args_list
                    )
                )
                rev[0] = "rThree"
                renderer.prepare_artifact.side_effect = lambda *a: {
                    "status": "render_error"
                }
                self.assertEqual(
                    self.a.evaluate(self.j, "http://judge.test")["render_pending"], 1
                )
                self.assertEqual(self.a.matches(self.j)[0]["outcome"], "pending")
                self.assertEqual(net.call_count, 6)
                renderer.prepare_artifact.side_effect = prepare
                self.assertEqual(
                    self.a.evaluate(self.j, "http://judge.test")["completed"], 1
                )

    def test_invalid_plain_artifacts_still_play(self):
        for count in (1, 2):
            with self.subTest(invalid=count):
                visual = case("bad" + str(count), kind="three")
                self.a.register_cases([visual])

                def prepare(c, source):
                    bad = source == "bad"
                    return {
                        "schema": 1,
                        "renderer_key": "r",
                        "kind": "three",
                        "status": "invalid_artifact" if bad else "ok",
                        "diagnostic": "bad_source" if bad else "ok",
                        "views": []
                        if bad
                        else [
                            {
                                "name": "front",
                                "png_base64": "cG5n",
                                "height": 1,
                                "width": 1,
                            }
                        ],
                    }

                def parts(bundle):
                    self.assertEqual(bundle["status"], "ok")
                    return [
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,cG5n"},
                        }
                    ]

                renderer = SimpleNamespace(
                    renderer_key=lambda: "r",
                    prepare_artifact=Mock(side_effect=prepare),
                    judge_parts=parts,
                    ERRORS={"bad_source"},
                )
                with patch.dict(sys.modules, arena_render=renderer):
                    with patch.object(
                        arena.common,
                        "request",
                        side_effect=[
                            completion("bad"),
                            completion("bad" if count == 2 else "good"),
                        ],
                    ):
                        self.a.generate(self.a1, "http://gate.test")
                        self.a.generate(self.a2, "http://gate.test")
                    self.assertEqual(
                        self.a.response(self.a1, visual["id"])["render_status"],
                        "invalid_artifact",
                    )
                    with patch.object(
                        arena.common, "request", return_value=completion("TIE")
                    ) as net:
                        self.assertEqual(
                            self.a.evaluate(self.j, "http://gate.test")["completed"], 1
                        )
                        for call in net.call_args_list:
                            shown = call.args[2]["messages"][1]["content"]
                            self.assertEqual(
                                sum(p["type"] == "image_url" for p in shown), 2 - count
                            )
                            self.assertEqual(
                                sum("bad_source" in p.get("text", "") for p in shown),
                                count,
                            )
                            self.assertEqual(
                                sum(
                                    "untrusted_source" in p.get("text", "")
                                    for p in shown
                                ),
                                2,
                            )
                        renders = renderer.prepare_artifact.call_count
                        self.a.generate(self.a1, "http://gate.test")
                        self.a.evaluate(self.j, "http://gate.test")
                        self.assertEqual(
                            (net.call_count, renderer.prepare_artifact.call_count),
                            (2, renders),
                        )

    def test_limit_prioritizes_untried_work_before_flaky_retry(self):
        self.a.register_cases([case(), case("q2")])
        with patch.object(
            arena.common,
            "request",
            side_effect=[TimeoutError(), completion("q2"), completion("q1")],
        ):
            first = self.a.generate(self.a1, "http://gate.test", limit=1)
            self.assertEqual(first["failed"], 1)
            self.a.generate(self.a1, "http://gate.test", limit=1)
            self.assertIsNotNone(self.a.response(self.a1, "q2"))
            self.a.generate(self.a1, "http://gate.test", limit=1)
        with patch.object(arena.common, "request", side_effect=self.transport):
            self.a.generate(self.a2, "http://gate.test")
        with patch.object(
            arena.common,
            "request",
            side_effect=[completion("malformed"), completion("TIE"), completion("TIE")],
        ):
            self.assertEqual(
                self.a.evaluate(self.j, "http://gate.test", limit=1)["failed"], 1
            )
            self.assertEqual(
                self.a.evaluate(self.j, "http://gate.test", limit=1)["completed"], 1
            )
        self.assertEqual(
            [m["outcome"] for m in self.a.matches(self.j)], ["pending", "draw"]
        )

    def test_text_parts_and_visible_refusal(self):
        out = completion(
            [
                {"type": "text", "text": " first "},
                {"type": "reasoning", "text": "not visible"},
                {"type": "output_text", "text": "second\n"},
            ]
        )
        self.assertEqual(arena.visible(out), " first second\n")
        self.assertEqual(
            arena.visible(completion(None, refusal="declined")), "declined"
        )
        with self.assertRaises(ValueError):
            arena.visible(completion([{"type": "image_url", "image_url": {}}]))

    def test_cli_self_judge_is_reported_as_incomplete(self):
        self.gen()
        catalog = Path(self.tmp.name) / "cases.jsonl"
        catalog.write_text(json.dumps(case()) + "\n")
        buf = io.StringIO()
        with patch.object(arena.common, "request") as net, redirect_stdout(buf):
            status = arena.main(
                [
                    "run",
                    "--db",
                    str(self.db),
                    "--catalog",
                    str(catalog),
                    "--judge",
                    "atlas",
                    "--judge-revision",
                    "r1",
                    "--judge-url",
                    "http://gateway.test",
                ]
            )
        report = json.loads(buf.getvalue())
        self.assertTrue(
            all(
                r["self_judge_pending"] == 1
                for r in self.a.standings(report["judge_key"])
            )
        )
        self.assertEqual(
            (status, report["evaluation"]["self_judge"], net.call_count), (1, 1, 0)
        )
        self.assertEqual(
            self.a.matches(report["judge_key"])[0]["pending_reason"], "self_judge"
        )

    def test_artifact_reasoning_markers_preserve_stored_bytes_and_digest(self):
        html = '<!doctype html>\n<html lang="en"><body title="<think>literal ✓</think>">A</body></html>\n'
        js = 'const sample = "\u003cthink>literal</think>";\nfunction buildScene(THREE) { return new THREE.Scene(); }\n'
        compact = 'function buildScene(THREE) { const sample="</think>"; return new THREE.Scene(); }'
        templ = "// Template preserves a physical closing-tag line.\nconst notes = `one\n</think>\ntwo`;\n"
        cases = [
            ("web", html),
            ("web", "```html\n" + html + "```\n"),
            ("three", js),
            ("three", compact),
            ("three", templ),
            ("text", "A report\n~~~html\n" + html + "~~~\nEnd.\n"),
        ]
        fixtures = [case("literal" + str(i), kind=k) for i, (k, _) in enumerate(cases)]
        self.a.register_cases(fixtures)
        # Native and prefixed traces can coexist with truly visible code containing the same markers.
        prefix = "<think>private prefix</think>"
        answers = [completion(prefix + t, reasoning="private native") for _, t in cases]
        with (
            patch.object(arena.common, "request", side_effect=answers),
            patch.object(self.a, "_render", return_value=None),
        ):
            self.assertEqual(
                self.a.generate(self.a1, "http://unused.test")["created"], len(cases)
            )
        records = {r["case_id"]: r for r in self.a.responses()}
        for i, (kind, original) in enumerate(cases):
            stored = records[fixtures[i]["id"]]
            self.assertEqual(stored["text"].encode(), original.encode())
            self.assertEqual(
                stored["answer_digest"], hashlib.sha256(original.encode()).hexdigest()
            )
            self.assertEqual(
                arena.visible(
                    completion(original, reasoning_content="native"), kind=kind
                ),
                original,
            )
        self.assertEqual(
            arena.visible(completion("scratch</think>" + html), kind="web"), html
        )
        fenced = "```js\n" + js + "```\n"
        self.assertEqual(
            arena.visible(
                completion(prefix + fenced + "<think>trailing trace</think>"),
                kind="three",
            ),
            fenced,
        )

    def test_one_user_transport_slot_across_archives_and_processes(self):
        self.gen()
        neighbor = arena.Arena(Path(self.tmp.name) / "other.sqlite3")
        neighbor.register_cases([case()])
        writer = neighbor.register_model("new-model", "r9")
        first, release = Event(), Event()
        guard = Lock()
        sent = []
        active = peak = 0

        class Service(BaseHTTPRequestHandler):
            def do_POST(self):
                nonlocal active, peak
                req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                with guard:
                    active += 1
                    peak = max(peak, active)
                    sent.append(req["model"])
                try:
                    if req["model"] == "new-model":
                        first.set()
                        release.wait(15)
                    data = json.dumps(
                        completion("stored" if req["model"] == "new-model" else "TIE")
                    ).encode()
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                finally:
                    with guard:
                        active -= 1

            def log_message(self, *args):
                pass

        script = """
import sys
from contextlib import contextmanager
import arena
db, action, key, url=sys.argv[1:]
store=arena.Arena(db)
real=arena.transport_slot
@contextmanager
def announced():
    print("reached transport",flush=True)
    with real():
        yield
arena.transport_slot=announced
result=getattr(store,action)(key,url)
print(result,flush=True)
"""
        children = []
        with ThreadingHTTPServer(("127.0.0.1", 0), Service) as server:
            thread = Thread(
                target=lambda: server.serve_forever(poll_interval=0.01), daemon=True
            )
            thread.start()

            def start(db, action, key):
                env = dict(os.environ)
                env.pop("BENCH_API_KEY", None)
                # These controls must never partition the per-user request budget.
                local_home = db.parent / action
                local_home.mkdir(exist_ok=True)
                env.update(
                    HOME=str(local_home),
                    TMPDIR=str(local_home),
                    BENCH_CACHE=str(db.parent / action),
                )
                child = subprocess.Popen(
                    [
                        sys.executable,
                        "-c",
                        script,
                        str(db),
                        action,
                        key,
                        f"http://127.0.0.1:{server.server_port}",
                    ],
                    cwd=Path(arena.__file__).parent,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                children.append(child)
                return child

            try:
                start(neighbor.path, "generate", writer)
                self.assertTrue(first.wait(8))
                judge = start(self.db, "evaluate", self.j)
                self.assertTrue(select.select([judge.stdout], [], [], 8)[0])
                self.assertEqual(judge.stdout.readline().strip(), "reached transport")
                with (
                    Path("/tmp") / f"weakbench-arena-{os.geteuid()}" / "transport.lock"
                ).open("rb") as held:
                    with self.assertRaises(BlockingIOError):
                        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.assertIsNone(arena.Arena(neighbor.path).response(writer, "q1"))
                self.assertEqual(
                    arena.Arena(self.db).matches(self.j)[0]["outcome"], "pending"
                )
                release.set()
                for child in children:
                    _, err = child.communicate(timeout=10)
                    self.assertEqual(child.returncode, 0, err)
                self.assertEqual((peak, sent), (1, ["new-model", "referee", "referee"]))
                self.assertEqual(self.a.matches(self.j)[0]["outcome"], "draw")
            finally:
                release.set()
                for child in children:
                    if child.poll() is None:
                        child.kill()
                    child.communicate()
                server.shutdown()
                thread.join(1)

    def test_cli_read_inventory_and_standings(self):
        for command in (["inspect"], ["standings", "--judge-key", self.j]):
            buf = io.StringIO()
            with redirect_stdout(buf):
                self.assertEqual(arena.main([*command, "--db", str(self.db)]), 0)
            self.assertIsInstance(json.loads(buf.getvalue()), dict)


if __name__ == "__main__":
    unittest.main()
