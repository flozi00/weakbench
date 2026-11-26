"""Hermetic checks for the closed quality runner. All test data are public placeholders."""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import urllib.request

import common
import hidden
import quality


class ClosedSuiteTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.tmp = Path(temporary.name)
        self.enter(patch.dict(os.environ, {"BENCH_CACHE": str(self.tmp / "cache"), "HF_TOKEN": "dummy"}))
        self.enter(patch.multiple(common, RUNS=self.tmp / "runs", RESULTS=self.tmp / "results"))
        rows = []
        for family in hidden.FULL_COUNTS:
            for lang in hidden.LANGS:
                for i in range(4):
                    exact = family in hidden.PLAIN_TASKS
                    rows.append({"family": family, "language": lang, "id": f"{family}_{lang}_{i}",
                                 "prompt": f"sentinel-input/{family}/{lang}/{i}",
                                 "mode": "exact" if exact else "judge",
                                 **({"expected": ["sentinel-answer", i]} if exact else
                                    {"rubric": ["sentinel-criterion", "sentinel-other"]})})
        self.payload = "\n".join(json.dumps(r) for r in rows).encode()
        self.enter(patch.multiple(hidden, FULL_COUNTS=dict.fromkeys(hidden.FULL_COUNTS, 4),
                                  DATA_SHA256=hashlib.sha256(self.payload).hexdigest()))
        path = hidden.cache_path()
        path.parent.mkdir(parents=True)
        path.write_bytes(self.payload)

    def enter(self, ctx):
        value = ctx.__enter__()
        self.addCleanup(ctx.__exit__, None, None, None)
        return value

    def test_release_lock_and_visibility(self):
        meta = {"id": hidden.DATA_REPO, "private": False, "sha": hidden.DATA_REVISION}
        with patch("urllib.request.OpenerDirector.open", return_value=io.BytesIO(json.dumps(meta).encode())) as fetch:
            with self.assertRaisesRegex(ValueError, "private dataset"):
                hidden.prepare()
            self.assertEqual(fetch.call_count, 1)
        path = hidden.cache_path()
        path.unlink()
        meta["private"] = True
        with patch("urllib.request.OpenerDirector.open", side_effect=[
                io.BytesIO(json.dumps(meta).encode()), io.BytesIO(self.payload)]):
            hidden.prepare()
        self.assertEqual(path.stat().st_mode & 0o077, 0)
        path.write_bytes(self.payload + b" ")
        with self.assertRaisesRegex(ValueError, "checksum"):
            hidden.release()
        req = urllib.request.Request("https://huggingface.co/file", headers={"Authorization": "Bearer dummy"})
        redirected = hidden.HubRedirect().redirect_request(req, None, 302, "", {}, "https://cdn.example/file")
        self.assertNotIn("Authorization", dict(redirected.header_items()))

    def test_sampling_and_scores(self):
        release = hidden.release()
        for family in hidden.FULL_COUNTS:
            full = hidden.load(release, family, "en", None, 43)
            quick = hidden.load(release, family, "de", 2, 43)
            self.assertEqual([r["id"] for r in full[:2]], [r["id"] for r in quick])
            self.assertEqual([r["messages"][0]["content"].split("/")[-1] for r in full[:2]],
                             [r["messages"][0]["content"].split("/")[-1] for r in quick])
            if family in hidden.PAIRED_VARIANTS:
                first, second = (int(r["messages"][0]["content"].split("/")[-1]) for r in quick)
                self.assertEqual((first % 2, second - first), (0, 1))
        expected = {"expected": {"x": [1], "z": False}}
        for text, valid, correct in [('{"z":false,"x":[1]}', True, True),
                                     ('{"z":0,"x":[true]}', True, False), ('true', True, False),
                                     ('{"z":false,"x":[],"x":[1]}', False, False), ('NaN', False, False),
                                     ('text {"x":[1],"z":false}', False, False)]:
            result = hidden.exact(expected, {"content": text})
            self.assertIs(result["correct"], correct)
            self.assertEqual(result["detail"]["valid_json"], float(valid))
        for bad in ["[]", "[true]", '["true","false"]', "[true,false] extra"]:
            with self.assertRaises(ValueError):
                hidden.verdict(lambda *a, **kw: bad, "task", ["c1", "c2"], "answer")

    def test_cli(self):
        def request(url, route, body=None, **kw):
            if route == "/v1/models":
                return {"data": [{"id": url, "max_model_len": 8192}]}
            if route == "/version":
                return {"version": "mock"}
            self.assertEqual(route, "/v1/chat/completions")
            prompt = body["messages"][-1]["content"]
            if url == "http://judge":
                case = json.loads(prompt)
                controls = {p: grades for p, _, _, grades in hidden.CANARIES}
                content = json.dumps(controls.get(case["task"], [True, False]))
            elif prompt == quality.PREFLIGHT:
                content = "391"
            else:
                self.assertTrue(prompt.startswith("sentinel-input/"))
                self.assertNotIn("sentinel-criterion", json.dumps(body))
                if fail:
                    raise RuntimeError(prompt)
                content = json.dumps(["sentinel-answer", int(prompt.split("/")[-1])])
            return {"choices": [{"message": {"content": content}, "finish_reason": "stop"}]}

        self.enter(patch.object(common, "request", side_effect=request))
        output = io.StringIO()
        self.enter(contextlib.redirect_stdout(output))
        self.enter(contextlib.redirect_stderr(output))
        argv = ["--url", "http://candidate", "--no-drift"]
        fail = False
        run = quality.main([*argv, "--arm", "judged", "--judge-url", "http://judge"])
        summary = json.loads((run / "summary.json").read_text())
        self.assertEqual(len(summary), len(hidden.FULL_COUNTS) * 2)
        for row in summary:
            self.assertEqual(row["n"], 2)
            self.assertEqual(row["accuracy"], float(row["task"] in hidden.PLAIN_TASKS))
            self.assertEqual(row["score_mean"], 1 if row["task"] in hidden.PLAIN_TASKS else 0.5)
            self.assertEqual(row["extra"]["xl_correct_agree"], 1)
        info = json.loads((run / "run_info.json").read_text())
        self.assertEqual(info["data_revision"], hidden.DATA_REVISION)
        self.assertEqual(info["data_sha256"], hidden.DATA_SHA256)
        self.assertEqual(info["expected_items"], 4 * len(hidden.FULL_COUNTS))
        fail = True
        with self.assertRaisesRegex(SystemExit, "unscored items"):
            quality.main([*argv, "--arm", "unjudged", "--limit", "1"])
        partial, = common.RUNS.glob("*-unjudged")
        self.assertEqual({r["task"] for r in json.loads((partial / "summary.json").read_text())}, set(hidden.PLAIN_TASKS))
        fail = False
        quality.main(["--resume", str(partial)])
        self.assertTrue(all(r["accuracy"] == 1 for r in json.loads((partial / "summary.json").read_text())))
        artifacts = [path.read_text() for path in common.RUNS.rglob("*") if path.is_file()]
        artifacts += [output.getvalue(), (common.RESULTS / "quality.jsonl").read_text()]
        self.assertFalse(any("sentinel-" in text for text in artifacts))
        with self.assertRaises(SystemExit):
            quality.main([*argv, "--arm", "public", "--tasks", "factuality"])


if __name__ == "__main__":
    unittest.main()
