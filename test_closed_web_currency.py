"""Synthetic controls for the time-sensitive search scorer and ordinary quality entry points."""

from copy import deepcopy
import json
import unittest
from unittest.mock import patch

import closed_web_currency as scorer
import closed_web_tools as tools


def fixture(condition="current"):
    claim = {"entity": "Sample Hall", "scope": "evening", "field": "opening time"}
    doc = {
        **claim,
        "id": "notice",
        "authority": "Sample Hall",
        "title": "Sample Hall evening notice",
        "url": "https://example.invalid/notice",
        "issued": "2025-02-01",
        "effective_from": "2025-02-02",
        "effective_until": None,
        "value": "18:00",
        "text": "The evening doors open at 18:00. Tickets are printed.",
    }
    second = dict(
        doc,
        id="revision",
        title="Sample Hall other notice",
        value="19:00",
        text="The evening doors open at 19:00.",
    )
    docs = (
        [doc, second]
        if condition == "unresolved"
        else []
        if condition == "unavailable"
        else [doc]
    )
    evidence = [{"id": d["id"], "quote": d["text"].split(".")[0]} for d in docs]
    return {
        "condition": condition,
        "as_of": "2025-03-10",
        "claim": claim,
        "docs": docs,
        "evidence": evidence,
        "value": "18:00" if condition in scorer.KNOWN else None,
        "given_ids": ["notice"] if condition == "historical_given" else [],
    }


def question(spec):
    prompt = "What time do the evening doors open at Sample Hall?"
    if spec["given_ids"]:
        prompt += " [notice] " + spec["docs"][0]["text"]
    return prompt


def answer(spec):
    return {
        "status": "known" if spec["condition"] in scorer.KNOWN else "unknown",
        "value": spec["value"],
        "citations": deepcopy(spec["evidence"]),
    }


def call_message():
    return {
        "role": "assistant",
        "tool_calls": [
            {
                "type": "function",
                "id": "native-17",
                "function": {
                    "name": "web_search",
                    "arguments": json.dumps({"query": "Sample Hall evening"}),
                },
            }
        ],
    }


def trace(spec):
    return [
        call_message(),
        {
            "role": "tool",
            "tool_call_id": "native-17",
            "content": json.dumps(
                {"results": tools.search(spec, "Sample Hall evening")}
            ),
        },
    ]


def evaluate(spec, candidate=None, messages=None):
    item = {
        "expected": spec,
        "lang": "en",
        "messages": [{"role": "user", "content": question(spec)}],
    }
    return scorer.score(
        item,
        {
            "content": json.dumps(answer(spec) if candidate is None else candidate),
            "observation_trace": messages or [],
        },
    )


class SearchGroundingTests(unittest.TestCase):
    def test_four_conditions_and_eligible_denominators(self):
        for condition in scorer.CONDITIONS:
            spec = fixture(condition)
            self.assertTrue(scorer.validate_expected(spec))
            result = evaluate(
                spec,
                messages=trace(spec) if condition in scorer.REQUIRES_SEARCH else [],
            )
            self.assertTrue(result["correct"], condition)
            self.assertEqual(
                "web_required_search_used" in result["detail"],
                condition in scorer.REQUIRES_SEARCH,
            )
            self.assertEqual(
                "web_without_tool_correct" in result["detail"],
                condition == "unavailable",
            )
        spec = fixture("historical_given")
        result = evaluate(spec, messages=trace(spec))
        self.assertTrue(result["correct"])
        self.assertEqual(result["detail"]["web_given_answered_directly"], 0)

    def test_wrong_visible_and_ungrounded_answers(self):
        spec = fixture()
        good = answer(spec)
        self.assertFalse(evaluate(spec, good)["correct"])
        for bad in [
            dict(good, value="19:00"),
            dict(good, citations=[]),
            dict(good, citations=[{"id": "notice", "quote": "Tickets are printed."}]),
            dict(
                good,
                citations=[{"id": "unseen", "quote": good["citations"][0]["quote"]}],
            ),
            dict(
                good,
                citations=[
                    {
                        "id": "notice",
                        "quote": "The evening doors open at 18:00 and never close.",
                    }
                ],
            ),
        ]:
            self.assertFalse(evaluate(spec, bad, trace(spec))["correct"])
        alt = deepcopy(good)
        alt["citations"][0]["quote"] = spec["docs"][0]["text"]
        self.assertTrue(evaluate(spec, alt, trace(spec))["correct"])
        forged = trace(spec)
        forged[1]["content"] = json.dumps({"results": []})
        self.assertFalse(evaluate(spec, good, forged)["correct"])
        conflict = fixture("unresolved")
        half = answer(conflict)
        half["citations"].pop()
        self.assertFalse(evaluate(conflict, half, trace(conflict))["correct"])
        self.assertFalse(evaluate(fixture("unavailable"), good)["correct"])

    def test_offline_and_live_native_calls_have_same_score(self):
        import quality

        spec = fixture()
        item = {
            "expected": spec,
            "lang": "en",
            "id": "fixture",
            "messages": [{"role": "user", "content": question(spec)}],
            **tools.runtime(spec, "en"),
        }
        answers = [
            {
                "choices": [{"message": call_message(), "finish_reason": "tool_calls"}],
                "usage": {"completion_tokens": 4},
            },
            {
                "choices": [
                    {
                        "message": {"content": json.dumps(answer(spec))},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"completion_tokens": 6},
            },
        ]
        with patch.object(quality.common, "request", side_effect=answers) as requests:
            record = quality.run_item(
                "http://mock",
                "web_currency",
                {"score": scorer.score},
                item,
                {"model": "placeholder", "messages": item["messages"]},
                2,
                private=True,
            )
        self.assertEqual(requests.call_count, 2)
        self.assertTrue(record["correct"])
        self.assertEqual(record["completion_tokens"], 10)
        self.assertEqual(
            record["detail"], evaluate(spec, messages=trace(spec))["detail"]
        )
        self.assertIsNone(record["reply"])
        self.assertNotIn(spec["docs"][0]["id"], json.dumps(record))


if __name__ == "__main__":
    unittest.main()
