import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import calibrate_judge as calibration


class CalibrationTests(unittest.TestCase):
    def test_controls_require_digest_and_judged_case(self):
        fixture = json.dumps({"id": "example", "label": "pass", "response": "test"})
        data = fixture.encode()
        with self.assertRaisesRegex(calibration.CalibrationError, "checksum"):
            calibration.load_controls(data, {})
        with patch.object(calibration, "CONTROLS_SHA256", hashlib.sha256(data).hexdigest()):
            with self.assertRaisesRegex(calibration.CalibrationError, "invalid controls"):
                calibration.load_controls(data, {})

    def test_aggregates_and_private_errors(self):
        marker = "private-test-marker"
        case = {"language": "de", "prompt": marker, "rubric": ["a", "b"]}
        controls = [(case, "pass", marker)] * 5 + [(case, "fail", marker)]
        replies = iter(["[true,true]", "[true,false]", "invalid", "[1]",
                        RuntimeError(marker), "[false,false]"])

        def judge(system, payload, max_tokens):
            reply = next(replies)
            if isinstance(reply, Exception):
                raise reply
            return reply

        results = calibration.calibrate(controls, judge)
        self.assertNotIn(marker, str(results))
        self.assertEqual(results[0]["matched"], 1)  # negative control
        self.assertEqual(results[1], {"language": "de", "label": "pass", "total": 5, "valid": 2,
                                      "matched": 1, "mismatched": 1, "errors": 3, "invalid_json": 1,
                                      "invalid_verdict": 1, "request_errors": 1})

    def test_entry_point_configures_tls_and_scores_controls(self):
        case = {"mode": "judge", "id": "id", "language": "en", "prompt": "private-test-marker", "rubric": ["sample"]}
        cells = {("demo", "en"): [case]}
        data = b'{"id":"id","label":"pass","response":"private-test-marker reply"}\n'
        with tempfile.TemporaryDirectory() as tmp, contextlib.ExitStack() as stack:
            path = Path(tmp)/"controls"
            path.write_bytes(data)
            stack.enter_context(patch.object(calibration, "CONTROLS_SHA256", hashlib.sha256(data).hexdigest()))
            stack.enter_context(patch.object(calibration.hidden, "release", return_value=cells))
            tls = stack.enter_context(patch.object(calibration.common, "verify_tls"))
            stack.enter_context(patch.object(calibration.common, "server_info",
                                             return_value={"model": "example", "max_model_len": 4096}))
            setup = stack.enter_context(patch.object(calibration.tasks_judge, "configure"))
            stack.enter_context(patch.object(calibration.tasks_judge, "judge", return_value="[true]"))
            out = stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            status = calibration.main(["--controls", str(path), "--judge-url", "https://example.invalid"])
            self.assertEqual(status, 0)
            tls.assert_called_once()
            setup.assert_called_once_with("https://example.invalid", "example", 4096, 180, 0)
            self.assertNotIn("private-test-marker", out.getvalue())
            self.assertEqual(json.loads(out.getvalue())["cells"][0]["matched"], 1)


if __name__ == "__main__":
    unittest.main()
