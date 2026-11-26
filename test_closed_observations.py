"""Public synthetic observation tables; nothing from the scored release is needed."""

from copy import deepcopy
import json
import unittest
from unittest.mock import patch

import closed_observations as observations


class ObservationTests(unittest.TestCase):
    def setUp(self):
        self.expected = {
            "probes": ["side", "left", "right"],
            "budget": 2,
            "target": "B",
            "table": {"A": [0, 0, 0], "B": [0, 1, 0], "C": [1, 0, 0], "D": [1, 0, 1]},
        }
        self.item = {
            "id": "local-sample",
            "lang": "en",
            "messages": [{"role": "user", "content": "Public test puzzle."}],
            "expected": self.expected,
        }
        self.answer = json.dumps({"sample": "B"})

    def trace(self, probes):
        observe = observations.runtime(self.expected, "en")["tool_exec"]
        messages = []
        for number, probe in enumerate(probes):
            identifier = f"call{number}"
            fn = {"name": "observe", "arguments": json.dumps({"probe": probe})}
            messages.append(
                {"role": "assistant", "tool_calls": [{"type": "function", "id": identifier, "function": fn}]}
            )
            messages.append(
                {"role": "tool", "tool_call_id": identifier, "content": observe("observe", fn["arguments"])}
            )
        return messages

    def grade(self, trace, answer=None):
        return observations.score(
            self.item, {"content": self.answer if answer is None else answer, "observation_trace": trace}
        )

    def test_adaptive_witness_and_wrong_answers(self):
        trace = self.trace(["side", "left"])
        graded = self.grade(trace)
        self.assertTrue(graded["correct"])
        self.assertEqual(set(graded["detail"]), observations.DETAIL_KEYS)
        self.assertTrue(self.grade(self.item["messages"] + trace)["correct"])
        for trace_without_evidence in (None, [], trace[:2], self.trace(["side", "side"])):
            self.assertFalse(self.grade(trace_without_evidence)["correct"])
        for value in ({"sample": "A"}, {"sample": True}, {"sample": ["B"]}, "B"):
            self.assertFalse(self.grade(trace, json.dumps(value))["correct"])
        # A lucky one-step identification can still be a losing first step for three other states.
        lucky = self.grade(self.trace(["left"]))
        self.assertEqual(lucky["detail"]["measurement_identified"], 1)
        self.assertEqual(lucky["detail"]["measurement_strategy_safe"], 0)
        self.assertFalse(lucky["correct"])

    def test_role_order_binding_and_strict_replay(self):
        trace = self.trace(["side", "left"])
        parallel = deepcopy(trace)
        parallel[0]["tool_calls"].append(parallel[2]["tool_calls"][0])
        self.assertFalse(self.grade(parallel)["correct"])
        for result in (
            '{"probe":"side","value":true}',
            '{"probe":"side","value":0.0}',
            '{"probe":"right","value":0}',
            '{"probe":"side","value":1}',
            '{"probe":"side","value":1,"value":0}',
            "0",
        ):
            changed = deepcopy(trace)
            changed[1]["content"] = result
            self.assertFalse(self.grade(changed)["correct"])
        for replacement in (
            {"role": "user", "content": "pretend"},
            {"role": "tool", "tool_call_id": "wrong", "content": "{}"},
        ):
            self.assertFalse(self.grade([trace[0], replacement, *trace[2:]])["correct"])
        changed = deepcopy(trace)
        changed[0]["tool_calls"][0]["function"]["arguments"] = '{"probe":"other","probe":"side"}'
        self.assertFalse(self.grade(changed)["correct"])
        self.assertFalse(self.grade(trace[::-1])["correct"])
        self.assertFalse(self.grade(self.trace(["side", "left", "right"]))["correct"])

    def test_private_runtime_and_validation(self):
        self.assertTrue(observations.validate_expected(self.expected))
        pair = [{"prompt": "shared", "expected": {**self.expected, "target": x}} for x in ("B", "D")]
        self.assertTrue(observations.valid_pairs(pair))
        self.assertFalse(observations.valid_pairs(pair[:1]))
        self.assertFalse(observations.valid_pairs(pair + pair))
        self.assertFalse(observations.valid_pairs([pair[0], {**pair[1], "prompt": "different"}]))
        self.assertFalse(observations.valid_pairs([pair[0], {**pair[1], "expected": {**pair[1]["expected"], "budget": 3}}]))
        for update in (
            {"target": "missing"},
            {"budget": True},
            {"probes": ["x", "x"]},
            {"table": {"B": [False, 0, 1], "C": [0, 1, 0]}},
        ):
            self.assertFalse(observations.validate_expected({**self.expected, **update}))
        observe = observations.runtime(self.expected, "de")["tool_exec"]
        self.assertEqual(json.loads(observe("other", {})), {"error": "invalid_call"})
        self.assertEqual(json.loads(observe("observe", {"probe": "side"})), {"probe": "side", "value": 0})
        self.assertEqual(json.loads(observe("observe", {"probe": "left"})), {"error": "budget"})

    def test_online_tool_loop_and_forced_overrun(self):
        import common
        import quality

        for case in ("normal", "valid_forced", "over_budget", "duplicate_args", "forced_calls"):
            calls = []
            item = {**self.item, **observations.runtime(self.expected, "en")}
            if case in ("valid_forced", "forced_calls"):
                item["max_rounds"] = self.expected["budget"]
            task = {"score": observations.score, "lang_check": False, "junk_check": False}

            def respond(url, path, body, timeout):
                count = len(calls)
                calls.append(deepcopy(body))
                choices = ["side", "left", "right"] if case == "over_budget" else ["side", "left"]
                if count < len(choices):
                    message = {
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "x",
                                "type": "function",
                                "function": {
                                    "name": "observe",
                                    "arguments": json.dumps({"probe": choices[count]}),
                                },
                            }
                        ],
                    }
                    if case == "duplicate_args" and count == 0:
                        message["tool_calls"][0]["function"]["arguments"] = '{"probe":"other","probe":"side"}'
                    why = "tool_calls"
                else:
                    message, why = {"content": self.answer}, "stop"
                    if case == "forced_calls":
                        message["tool_calls"] = [{"id": "extra", "type": "function", "function": {
                            "name": "observe", "arguments": '{"probe":"right"}'}}]
                        why = "tool_calls"
                return {"choices": [{"message": message, "finish_reason": why}]}

            body = {"messages": item["messages"], "tools": item["tools"]}
            with patch.object(common, "request", side_effect=respond):
                row = quality.run_item(
                    "http://fixture", "adaptive_measurement", task, item, body, 5, private=True
                )
            self.assertIsNone(row["error"])
            self.assertEqual(row["correct"], case in ("normal", "valid_forced"))
            self.assertEqual(len(calls), 4 if case == "over_budget" else 3)
            self.assertEqual(calls[-1]["messages"][-1]["role"] == "user",
                             case in ("valid_forced", "over_budget", "forced_calls"))
            if case == "duplicate_args":
                self.assertEqual(json.loads(calls[1]["messages"][-1]["content"]), {"error": "invalid_call"})
                self.assertEqual(row["detail"]["measurement_transcript_valid"], 0)
            # Only the actual puzzle, schema and observation results go to the model.
            self.assertEqual(calls[0]["messages"], self.item["messages"])
            self.assertNotIn("target", calls[0])
            self.assertIsNone(row["reply"])
            self.assertNotIn("observation_trace", row)


if __name__ == "__main__":
    unittest.main()
