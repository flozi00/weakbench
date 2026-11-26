"""Public in-memory rasters and a fake endpoint exercise the private image path offline."""

import base64
from contextlib import redirect_stdout, redirect_stderr
import hashlib
import io
import json
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
import zlib

import check_response
import closed_images
import common
import hidden
import quality


def chunk(kind, data):
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))


def png_from(w, h, samples, *, color=0, depth=8, interlace=0, compressed=None, optional=b""):
    ihdr = struct.pack(">IIBBBBB", w, h, depth, color, 0, 0, interlace)
    data = zlib.compress(samples) if compressed is None else compressed
    return (
        closed_images.SIGNATURE + chunk(b"IHDR", ihdr) + optional + chunk(b"IDAT", data) + chunk(b"IEND", b"")
    )


def media(raw):
    return {"mime_type": "image/png", "data": base64.b64encode(raw).decode()}


class PNGTests(unittest.TestCase):
    def test_png_profiles(self):
        for raw in (
            png_from(3, 1, b"\0\xff\x80\x00"),
            quality.png(2, 3, lambda x, y: (x, y, x + y)),
            png_from(1, 1, b"\0\0", color=3, depth=1, optional=chunk(b"PLTE", b"\1\2\3")),
            png_from(1, 1, b"\4\xfd", interlace=1),
        ):
            closed_images.validate_media(media(raw))

    def test_reject_malformed_or_oversize_rasters(self):
        baseline = png_from(2, 1, b"\0ab")
        checksum = bytearray(baseline)
        checksum[-1] ^= 1
        damaged_zlib = bytearray(zlib.compress(b"\0ab"))
        damaged_zlib[-1] ^= 1
        for raw in (
            bytes(checksum),
            baseline[:-1],
            baseline + b"x",
            b"<svg/>",
            png_from(0, 1, b""),
            png_from(1, closed_images.MAX_DIMENSION + 1, b""),
            png_from(1, 1, b"\5x"),
            png_from(1, 1, b"\0xx"),
            png_from(2, 2, b"\0"),
            png_from(2, 1, b"", compressed=damaged_zlib),
            png_from(2, 1, b"", compressed=zlib.compress(b"\0ab") + zlib.compress(b"surplus")),
            png_from(1, 1, bytes(1 << 20)),
            png_from(2048, 2048, b"", color=6),
            png_from(1, 1, b"\0x", optional=chunk(b"acTL", bytes(8))),
            png_from(1, 1, b"\0x", optional=chunk(b"ABCD", b"")),
            png_from(1, 1, b"\0x", optional=chunk(b"tRNS", b"x")),
        ):
            with self.assertRaises(ValueError):
                closed_images.validate_media(media(raw))
        for wrong in (
            {"url": "https://invalid.invalid/test.png"},
            {"mime_type": "image/png", "data": "https://invalid.invalid/test.png"},
            {"mime_type": "image/jpeg", "data": media(baseline)["data"]},
            {**media(baseline), "path": "image.png"},
            {"mime_type": "image/png", "data": media(baseline)["data"] + "\n"},
            {"mime_type": "image/png", "data": "!"},
            {"mime_type": "image/png", "data": 17},
        ):
            with self.assertRaises(ValueError):
                closed_images.validate_media(wrong)
        with patch.object(closed_images, "MAX_PNG_BYTES", len(baseline) - 1), self.assertRaises(ValueError):
            closed_images.validate_media(media(baseline))


class VisualRunnerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.tmp = Path(temporary.name)

        def add(ctx):
            ctx.__enter__()
            self.addCleanup(ctx.__exit__, None, None, None)

        add(patch.dict(os.environ, {"BENCH_CACHE": str(self.tmp / "cache")}))
        add(patch.multiple(common, RUNS=self.tmp / "runs", RESULTS=self.tmp / "results"))
        add(patch.multiple(hidden, FULL_COUNTS={"snapshot_trace": 4}, AUDIO_COUNTS={}, PLAIN_TASKS=("snapshot_trace",)))
        self.rows = []
        self.fixtures = {}
        for family, count in (hidden.FULL_COUNTS | hidden.VISUAL_COUNTS).items():
            for lang in hidden.LANGS:
                for i in range(count):
                    prompt = f"synthetic-private-visual/{family}/{i}/{lang}"
                    row = {
                        "id": f"{family}_{lang}_{i}",
                        "family": family,
                        "language": lang,
                        "mode": "exact",
                        "prompt": prompt,
                        "expected": {"answer": i},
                    }
                    if family == "panel_grounding":
                        row.update(mode="program", expected={"size": [2, 1], "regions": [[i % 2, 0, i % 2 + 1, 1]]})
                    if family in hidden.VISUAL_COUNTS:
                        row["media"] = media(png_from(2, 1, bytes((0, i, 255 - i))))
                    self.rows.append(row)
                    self.fixtures[prompt] = row
        self.payload = self.encode(self.rows)
        add(patch.object(hidden, "DATA_SHA256", hashlib.sha256(self.payload).hexdigest()))
        cache = hidden.cache_path()
        cache.parent.mkdir(parents=True)
        cache.write_bytes(self.payload)

    @staticmethod
    def reference_answer(family, spec):
        return {"point": spec["regions"][0][:2]} if family == "panel_grounding" else spec

    @staticmethod
    def encode(rows):
        return "\n".join(json.dumps(row) for row in rows).encode()

    def test_release_checks_and_paired_loading(self):
        cells = hidden.release()
        registered = hidden.tasks(cells, None)
        for family in hidden.VISUAL_COUNTS:
            self.assertEqual(
                registered[family]["counts"], {"quick": {"en": 16, "de": 16}, "full": {"en": 32, "de": 32}}
            )
            en = hidden.load(cells, family, "en", None, 91)
            de = hidden.load(cells, family, "de", 16, 91)
            self.assertEqual([r["id"] for r in en[:16]], [r["id"] for r in de])
            self.assertEqual([r["expected"] for r in en[:16]], [r["expected"] for r in de])
            for a, b in zip(en[::2], en[1::2]):
                if family == "panel_grounding":
                    self.assertNotEqual(a["expected"]["regions"], b["expected"]["regions"])
                else:
                    self.assertEqual(a["expected"]["answer"] + 1, b["expected"]["answer"])
            content = en[0]["messages"][0]["content"]
            self.assertEqual([p["type"] for p in content], ["image_url", "text"])
            source = self.fixtures[content[1]["text"]]
            self.assertEqual(
                content[0]["image_url"]["url"], "data:image/png;base64," + source["media"]["data"]
            )
        self.assertFalse(set(hidden.VISUAL_COUNTS) & {n for c in hidden.CATEGORIES for n in c["tasks"]})
        panel = next(row for row in self.rows if row["family"] == "panel_grounding")
        image = next(row for row in self.rows if "media" in row)
        for before, after in (
            (panel, {**panel, "expected": {**panel["expected"], "size": [3, 1]}}),
            (image, {k: v for k, v in image.items() if k != "media"}),
            (image, {**image, "media": {"url": "https://invalid.invalid/"}}),
            (self.rows[0], {**self.rows[0], "media": image["media"]}),
        ):
            changed = self.encode([after if row is before else row for row in self.rows])
            with (
                patch.object(hidden, "DATA_SHA256", hashlib.sha256(changed).hexdigest()),
                self.assertRaisesRegex(ValueError, "invalid private release row"),
            ):
                hidden.decode(changed)
        missing = self.encode(self.rows[:-1])
        with (
            patch.object(hidden, "DATA_SHA256", hashlib.sha256(missing).hexdigest()),
            self.assertRaisesRegex(ValueError, "incomplete"),
        ):
            hidden.decode(missing)

    def test_selection_privacy_and_offline_replay(self):
        seen = []

        def responder(url, route, body=None, **kw):
            if route == "/v1/models":
                return {"data": [{"id": "local", "max_model_len": 8192}]}
            if route == "/version":
                return {"version": "mock"}
            self.assertEqual(route, "/v1/chat/completions")
            content = body["messages"][-1]["content"]
            if content == quality.PREFLIGHT:
                answer = "391"
            elif isinstance(content, list):
                data, text = content
                question = text["text"]
                if question in self.fixtures:
                    row = self.fixtures[question]
                    self.assertEqual(
                        data["image_url"]["url"], "data:image/png;base64," + row["media"]["data"]
                    )
                    solution = self.reference_answer(row["family"], row["expected"])
                    answer = json.dumps(solution)
                    seen.append(row["family"])
                else:
                    seen.append("image")
                    answer = "red"
            else:
                row = self.fixtures[content]
                seen.append(row["family"])
                answer = json.dumps(row["expected"])
            return {"choices": [{"message": {"content": answer}, "finish_reason": "stop"}]}

        main_args = [
            "--url",
            "http://mock",
            "--langs",
            "en",
            "--limit",
            "1",
            "--no-drift",
            "--concurrency",
            "1",
        ]
        captured = io.StringIO()
        trials = (
            ("default", [], {"snapshot_trace"}),
            ("synthetic", ["--image"], {"image", "snapshot_trace"}),
            ("panel", ["--tasks", "panel_grounding", "--image"], {"panel_grounding"}),
            ("voxels", ["--tasks", "voxel_projections", "--image"], {"voxel_projections"}),
            ("charts", ["--tasks", "interval_plot,image", "--image"], {"interval_plot", "image"}),
            (
                "mixed",
                ["--tasks", "snapshot_trace,interval_plot", "--image"],
                {"snapshot_trace", "interval_plot"},
            ),
        )
        with (
            patch.object(common, "request", side_effect=responder),
            redirect_stdout(captured),
            redirect_stderr(captured),
        ):
            for arm, flags, selected in trials:
                seen.clear()
                run = quality.main([*main_args, "--arm", arm, *flags])
                self.assertEqual(set(seen), selected)
                summary = json.loads((run / "summary.json").read_text())
                self.assertEqual({x["task"] for x in summary}, selected)
                self.assertEqual(
                    json.loads((run / "run_info.json").read_text())["expected_items"], len(selected)
                )
                for name in selected & hidden.VISUAL_COUNTS.keys():
                    records = common.read_jsonl(run / f"items-{name}.jsonl")
                    (record,) = records
                    self.assertIs(record["correct"], True)
                    self.assertIsNone(record["reply"])
                    prompts = hidden.load(hidden.release(), name, "en", None, quality.SEED)
                    item = next(it for it in prompts if it["id"] == record["id"])
                    path = self.tmp / "completion.json"
                    path.write_text(
                        json.dumps({"choices": [{"message": {"content": json.dumps(self.reference_answer(name, item["expected"]))}}]})
                    )
                    self.assertEqual(
                        check_response.main(
                            ["--task", name, "--lang", "en", "--id", item["id"], "--response", str(path)]
                        ),
                        0,
                    )
        # A named visual task without its opt-in must report a selection error before contacting a candidate.
        with patch.object(common, "request", side_effect=AssertionError), redirect_stderr(captured):
            for name in ("voxel_projections", "interval_plot", "image"):
                with self.assertRaises(SystemExit) as failure:
                    quality.main([*main_args, "--arm", "absent", "--tasks", name])
                self.assertEqual(failure.exception.code, 2)
        self.assertIn("require --image", captured.getvalue())
        public = captured.getvalue() + "".join(
            p.read_text()
            for p in self.tmp.rglob("*")
            if p.is_file() and (common.RUNS in p.parents or common.RESULTS in p.parents)
        )
        self.assertNotIn("data:image/png", public)
        self.assertNotIn("synthetic-private-visual", public)
        self.assertNotIn(self.rows[-1]["media"]["data"], public)


if __name__ == "__main__":
    unittest.main()
