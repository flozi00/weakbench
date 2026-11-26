"""Exercise persisted submissions, tracked evaluation and versioned catalogues without a GPU."""

from contextlib import redirect_stdout
import gzip
from hashlib import sha256
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import gpu_bench as gpu
import kernel_catalog
import kernel_cuda


def fixture():
    program = {"source": "out = A(i)", "threads": 32, "outputs_per_thread": 1}
    cases = [
        {
            "shape": [1, 1, 1],
            "a": [2],
            "b": [3],
            "expected": [2],
            "atol": 0,
            "rtol": 0,
            "timing": timed,
        }
        for timed in (False, True)
    ]
    return {
        "format": 1,
        "workloads": {"demo": {"metal": program, "cuda": program, "vectors": cases}},
        "tasks": [
            {
                "id": lang,
                "language": lang,
                "family": "demo",
                "prompt": "example " + lang,
                "cuda_prompt": "cuda " + lang,
            }
            for lang in ("de", "en")
        ],
    }


class KernelArchiveTest(unittest.TestCase):
    def test_catalogue(self):
        data = fixture()

        def encoded():
            return gzip.compress(json.dumps(data, allow_nan=False).encode())

        self.assertEqual(kernel_catalog.decode(encoded()), data)
        with self.assertRaises(ValueError):
            kernel_catalog.decode(encoded(), pinned=True)
        data["tasks"][1]["language"] = "de"
        with self.assertRaises(ValueError):
            kernel_catalog.decode(encoded())
        for value in (b"broken", gzip.compress(b'{"format":1,"format":2}')):
            with self.assertRaises(ValueError):
                kernel_catalog.decode(value)

    def test_model_cache_and_report(self):
        with tempfile.TemporaryDirectory() as work:
            folder = Path(work)
            bundle = folder / "catalog.gz"
            bundle.write_bytes(gzip.compress(json.dumps(fixture()).encode()))
            shared = ["--db", str(folder / "archive"), "--catalog", str(bundle)]
            generate = [
                "generate",
                "--target",
                "cuda",
                "--model",
                "sample",
                "--revision",
                "r1",
                "--url",
                "http://localhost:9999",
                "--lang",
                "de",
                *shared,
            ]
            raw = json.dumps(fixture()["workloads"]["demo"]["metal"])
            response = {
                "choices": [
                    {
                        "message": {"role": "assistant", "content": raw},
                        "finish_reason": "stop",
                    }
                ]
            }
            with (
                patch("arena.answer_request", return_value=response) as provider,
                redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(gpu.main(generate), 0)
                self.assertEqual(gpu.main(generate), 0)
                self.assertEqual(provider.call_count, 1)
                path = folder / "saved.json"
                evaluate = [
                    "evaluate",
                    "--track",
                    "cuda-compile",
                    "--output",
                    str(path),
                    *shared,
                ]
                with patch.object(
                    kernel_cuda, "compile_native", return_value={"status": "compiled"}
                ) as compiler:
                    self.assertEqual(
                        gpu.main(evaluate), 1
                    )  # the English task was never generated
                    self.assertEqual(compiler.call_count, 1)
                report = json.loads(path.read_bytes())
                self.assertEqual(
                    report["catalog_sha256"], sha256(bundle.read_bytes()).hexdigest()
                )
                model = report["models"][0]
                self.assertEqual(
                    model["summary"],
                    {
                        "statuses": {"compiled": 1, "missing_submission": 1},
                        "timed_shapes": 0,
                        "faster_shapes": 0,
                        "selected_submissions": 2,
                    },
                )
                self.assertEqual(
                    model["results"][0]["answer_sha256"],
                    sha256(raw.encode()).hexdigest(),
                )
                self.assertNotIn("out = A(i)", path.read_text())
                saved = path.read_bytes()
                with self.assertRaises(FileExistsError):
                    gpu.save_report(path, {"overwrite": True})
                self.assertEqual(saved, path.read_bytes())
                with patch.object(
                    kernel_cuda,
                    "measure",
                    side_effect=kernel_cuda.BackendUnavailable("unavailable"),
                ):
                    path = folder / "no-device.json"
                    self.assertEqual(
                        gpu.main(
                            [
                                "evaluate",
                                "--track",
                                "cuda-hardware",
                                "--lang",
                                "de",
                                "--output",
                                str(path),
                                *shared,
                            ]
                        ),
                        1,
                    )
                    self.assertEqual(
                        json.loads(path.read_bytes())["models"][0]["summary"][
                            "statuses"
                        ],
                        {"backend_unavailable": 1},
                    )
                self.assertEqual(provider.call_count, 1)

    def test_controls_are_independent_of_model_archive(self):
        with patch.object(
            kernel_cuda, "compile_native", return_value={"status": "compiled"}
        ) as compiler:
            controls = gpu.reference_control(fixture(), "cuda-compile", 17)
        self.assertEqual(compiler.call_count, 1)
        self.assertEqual(controls[0]["model"]["key"], "reference_control")
        self.assertEqual(controls[0]["summary"]["statuses"], {"compiled": 1})

    def test_answer_contract_and_scoring(self):
        program = fixture()["workloads"]["demo"]["metal"]
        self.assertEqual(
            gpu.submission("```json\n" + json.dumps(program) + "\n```"), program
        )
        for raw in (
            "{}",
            json.dumps({**program, "threads": True}),
            json.dumps({**program, "source": ""}),
        ):
            self.assertEqual(
                gpu.grade(raw, {}, "metal", 17)["status"], "invalid_submission"
            )
        rows = [
            {
                "measurement": {
                    "status": s,
                    "cases": [{"timing": {"comparison": comparison}}],
                }
            }
            for s, comparison in (
                ("correct", "faster"),
                ("correct", "overlap"),
                ("incorrect", "faster"),
            )
        ]
        self.assertEqual(gpu.totals(rows)["timed_shapes"], 2)
        self.assertEqual(gpu.totals(rows)["faster_shapes"], 1)


if __name__ == "__main__":
    unittest.main()
