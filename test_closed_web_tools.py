"""Hermetic native function-call, relevance and evidence-replay tests on an invented archive."""

from copy import deepcopy
import json
import unittest

import closed_web_tools as tools


def document(i, title, text, issued="2025-04-01"):
    return {
        "id": f"Page-{i}",
        "url": f"https://example.invalid/page/{i}",
        "title": title,
        "authority": "Archiv Nord",
        "issued": issued,
        "effective_from": issued,
        "effective_until": None,
        "entity": "verifier-only entity",
        "scope": "verifier-only scope",
        "field": "verifier-only field",
        "value": "verifier-only value",
        "text": text,
    }


def item(condition="current", lang="en"):
    docs = [
        document(
            1,
            "Leuchtturm Nord Eintrittspreis",
            "Der Preis für die Westführung beträgt fünf Marken.",
        ),
        document(2, "Lighthouse west tour fee", "A west tour costs six tokens."),
        document(3, "Lighthouse opening hours", "Visitors may enter at noon."),
        document(
            4, "Archivregister", "Four new shelves were installed in the archive."
        ),
        document(5, "Pond measurements", "The lake held ten litres of water."),
    ]
    return {
        "expected": {
            "condition": condition,
            "as_of": "2026-01-01",
            "claim": {"entity": "h", "scope": "s", "field": "f"},
            "value": "hidden-gold",
            "evidence": [{"id": "Page-2", "quote": "unpublished key"}],
            "docs": docs,
        },
        "lang": lang,
        "messages": [{"role": "user", "content": "Find the applicable local fee."}],
    }


def native(id, query=None, *, name="web_search", raw=None):
    payload = raw if raw is not None else json.dumps({"query": query})
    return {
        "id": id,
        "type": "function",
        "function": {"name": name, "arguments": payload},
    }


def raw(*calls, text=None, reasoning=None, finish=None):
    msg = {"role": "assistant", "content": text, "reasoning_content": reasoning}
    if calls:
        msg["tool_calls"] = list(calls)
    return {
        "choices": [
            {
                "message": msg,
                "finish_reason": finish or ("tool_calls" if calls else "stop"),
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2},
    }


def run(it, replies):
    bodies = []

    def fake(body):
        bodies.append(deepcopy(body))
        if len(bodies) > len(replies):
            raise AssertionError("extra inference call")
        return replies[len(bodies) - 1]

    reply, usage = tools.run(
        it, {"model": "offline", "messages": it["messages"], "temperature": 0.3}, fake
    )
    return reply, usage, bodies


class WebToolsTest(unittest.TestCase):
    def test_lexical_search_uses_public_fields_and_is_bounded(self):
        spec = item()["expected"]
        self.assertEqual(tools.search(spec, "x"), [])
        self.assertEqual(tools.search(spec, "der die das"), [])
        self.assertEqual(tools.search(spec, "verifier-only"), [])
        self.assertEqual(tools.search(spec, "unpublished key"), [])
        self.assertEqual(
            tools.search(spec, "Wie hoch ist der PREIS für die Westführung?")[0]["id"],
            "Page-1",
        )
        self.assertEqual(
            tools.search(spec, "What fee does the lighthouse west tour charge?")[0][
                "id"
            ],
            "Page-2",
        )
        self.assertEqual([r["id"] for r in tools.search(spec, "PAGE-5")], ["Page-5"])
        self.assertEqual(
            [r["id"] for r in tools.search(spec, spec["docs"][3]["url"])], ["Page-4"]
        )
        results = tools.search(spec, "Archiv Nord")
        self.assertLessEqual(len(results), tools.MAX_RESULTS)
        self.assertEqual(
            tools.search({**spec, "docs": spec["docs"][::-1]}, "Archiv Nord"), results
        )
        self.assertTrue(all(set(r) == set(tools.PUBLIC_FIELDS) for r in results))
        for k in ("evidence", "value", "field", "scope", "entity"):
            self.assertNotIn(k, results[0])

    def test_native_ids_parallel_batch_and_cumulative_usage(self):
        it = item()
        first = native("gw:opaque-9", "Page-1")
        second = native("call_xyz/123", "Page-2")
        last = raw(text="final", reasoning="native reasoning")
        reply, usage, requests = run(
            it, [raw(first, second, reasoning="avoid forwarding reasoning"), last]
        )
        self.assertEqual(usage, {"prompt_tokens": 10, "completion_tokens": 4})
        self.assertEqual(reply["content"], "final")
        self.assertEqual(reply["reasoning"], "native reasoning")
        self.assertEqual((reply["tool_rounds"], reply["tool_forced"]), (1, False))
        self.assertEqual(requests[0]["tool_choice"], "auto")
        self.assertEqual(requests[0]["tools"][0]["function"]["name"], "web_search")
        self.assertTrue(all(r["temperature"] == 0.3 for r in requests))
        self.assertEqual(requests[1]["messages"][1]["tool_calls"], [first, second])
        self.assertNotIn("avoid forwarding", str(requests[1]))
        for call, entry in zip((first, second), requests[1]["messages"][2:]):
            self.assertEqual(
                (entry["role"], entry["tool_call_id"]), ("tool", call["id"])
            )
        obs = tools.observed(it, reply["observation_trace"])
        self.assertEqual(
            (obs["valid"], obs["calls"], obs["searches"], obs["errors"]),
            (True, 2, 2, 0),
        )
        self.assertEqual(
            obs["documents"], {d["id"]: d["text"] for d in it["expected"]["docs"][:2]}
        )
        self.assertNotIn("verifier-only", str(requests))
        self.assertNotIn("hidden-gold", str(requests))

    def test_unavailable_forbids_tools_and_native_calls(self):
        it = item("unavailable", "de")
        setup = tools.runtime(it["expected"], "de")
        self.assertNotIn("tools", setup)
        self.assertNotIn("tool_choice", setup)
        requests = []

        def fake(body):
            requests.append(body)
            return raw(text="keine Daten")

        reply, _ = tools.run(
            it,
            {
                "messages": it["messages"],
                "tools": ["untrusted inherited"],
                "tool_choice": "required",
            },
            fake,
        )
        self.assertNotIn("tools", requests[0])
        self.assertNotIn("tool_choice", requests[0])
        self.assertEqual(
            tools.observed(it, reply["observation_trace"]),
            {"valid": True, "calls": 0, "searches": 0, "errors": 0, "documents": {}},
        )
        reply, _, requests = run(it, [raw(native("not-offered", "Page-1"))])
        self.assertTrue(reply["web_invalid_final"])
        self.assertEqual(len(requests), 1)

    def test_direct_given_answer_and_irrelevant_search(self):
        it = item("historical_given")
        reply, _, _ = run(it, [raw(text="provided fact")])
        self.assertTrue(tools.observed(it, reply["observation_trace"])["valid"])
        self.assertEqual(tools.observed(it, None)["calls"], 0)
        for condition in ("current", "historical_given", "unresolved"):
            it = item(condition)
            reply, _, _ = run(it, [raw(native("r1", "x")), raw(text="no match")])
            obs = tools.observed(it, reply["observation_trace"])
            self.assertEqual((obs["searches"], obs["documents"]), (1, {}))

    def test_errors_are_real_tool_results_and_recovery_is_possible(self):
        it = item()
        calls = [
            native("one", "Page-1", name="unlisted_search"),
            native("two", raw='{"query":"Page-1","query":"Page-2"}'),
            native("three", raw='{"query":["Page-1"]}'),
        ]
        reply, _, requests = run(
            it, [raw(*calls), raw(native("four", "Page-2")), raw(text="final")]
        )
        errors = [
            json.loads(entry["content"])["error"]
            for entry in requests[1]["messages"][2:]
        ]
        self.assertEqual(
            errors, ["unknown_tool", "invalid_arguments", "invalid_arguments"]
        )
        self.assertNotIn("tools", requests[-1])
        self.assertNotIn("tool_choice", requests[-1])
        self.assertEqual(requests[-1]["messages"][-1]["role"], "user")
        self.assertTrue(reply["tool_forced"])
        obs = tools.observed(it, reply["observation_trace"])
        self.assertEqual(
            (obs["calls"], obs["errors"], obs["searches"], list(obs["documents"])),
            (4, 3, 1, ["Page-2"]),
        )

    def test_malformed_identifiers_and_over_budget_cannot_create_evidence(self):
        it = item()
        x = native("x", "Page-2")
        for calls in (
            [dict(x, id="")],
            [{k: v for k, v in x.items() if k != "id"}],
            [dict(x, type="custom")],
            [
                dict(
                    x, function={"name": "web_search", "arguments": {"query": "Page-2"}}
                )
            ],
            [x, x],
            [native(str(i), "Page-1") for i in range(tools.MAX_CALLS + 1)],
        ):
            reply, _, requests = run(it, [raw(*calls)])
            self.assertEqual(len(requests), 1)
            self.assertTrue(reply["web_invalid_final"])
        reply, _, _ = run(it, [raw(x), raw(x)])
        self.assertTrue(reply["web_invalid_final"])
        calls = [native(str(n), "x") for n in range(tools.MAX_CALLS)]
        reply, _, requests = run(it, [raw(*calls), raw(native("fresh", "Page-2"))])
        self.assertEqual(len(requests), 2)
        self.assertNotIn("tools", requests[-1])
        self.assertTrue(reply["web_invalid_final"])

    def test_full_trace_replay_rejects_injection_swaps_and_stale_editions(self):
        it = item()
        reply, _, _ = run(
            it,
            [
                raw(native("distinct:1", "Page-1"), native("distinct:2", "Page-2")),
                raw(text="final"),
            ],
        )
        trace = reply["observation_trace"]
        expected = tools.observed(it, trace)
        self.assertEqual(tools.observed(it, it["messages"] + trace), expected)

        def bad(mut):
            altered = deepcopy(trace)
            mut(altered)
            obs = tools.observed(it, altered)
            self.assertFalse(obs["valid"])
            self.assertFalse(obs["documents"])

        bad(lambda t: t[1].update(tool_call_id="wrong-id"))
        bad(lambda t: t.__setitem__(slice(1, 3), [t[2], t[1]]))
        bad(lambda t: t[1].update(content=t[2]["content"]))
        bad(lambda t: t[1].update(content='{"results":[],"results":[]}'))
        bad(lambda t: t[1].update(content=t[1]["content"].replace("fünf", "neun")))
        bad(
            lambda t: t[0]["tool_calls"][0]["function"].update(
                arguments='{"query":"Page-4"}'
            )
        )
        bad(
            lambda t: t.append(
                {"role": "tool", "tool_call_id": "unseen", "content": "{}"}
            )
        )
        bad(lambda t: t.pop())
        changed = deepcopy(it)
        changed["expected"]["docs"][0]["text"] = "An independent change."
        self.assertFalse(tools.observed(changed, trace)["valid"])

    def test_long_and_invalid_queries_cannot_expose_hidden_docs(self):
        invalid = [
            {"query": ""},
            {"query": " " * 4},
            {"query": "a" * (tools.MAX_QUERY + 1)},
            {"query": True},
            {"query": "Page-1", "offset": 0},
            ["Page-1"],
            {},
        ]
        for bad in invalid:
            it = item()
            reply, _, _ = run(
                it, [raw(native("id", raw=json.dumps(bad))), raw(text="final")]
            )
            obs = tools.observed(it, reply["observation_trace"])
            self.assertEqual(
                (obs["valid"], obs["errors"], obs["documents"]), (True, 1, {})
            )

    def test_saved_and_live_final_turns_use_the_same_protocol(self):
        it = item()
        valid = raw(text="answer", reasoning="private rationale")
        valid["choices"][0]["message"].pop("role")
        self.assertFalse(tools.parse_final(valid)["web_invalid_final"])
        self.assertEqual(tools.parse_final(valid)["reasoning"], "private rationale")
        for field, value in (
            ("role", "tool"),
            ("function_call", {"name": "legacy"}),
            ("refusal", "Declined"),
            ("tool_calls", [None]),
            ("tool_calls", {}),
            ("content", ["text"]),
            ("reasoning_content", []),
            ("content", "<think>private only</think>"),
            ("content", None),
        ):
            broken = deepcopy(valid)
            broken["choices"][0]["message"][field] = value
            with self.subTest(field=field, value=value):
                parsed = tools.parse_final(broken)
                self.assertTrue(parsed["web_invalid_final"])
                reply, _, _ = run(it, [raw(native("seen", "Page-1")), broken])
                self.assertTrue(reply["web_invalid_final"])
                self.assertEqual(reply["content"], parsed["content"])
                self.assertTrue(tools.observed(it, reply["observation_trace"])["valid"])
                self.assertEqual(
                    tools.observed(it, reply["observation_trace"])["calls"], 1
                )
        for reason in ("tool_calls", "function_call", "content_filter"):
            broken = raw(text="text", finish=reason)
            self.assertTrue(tools.parse_final(broken)["web_invalid_final"])
        self.assertTrue(
            tools.parse_final(raw(native("remains", "x"), text="answer"))[
                "web_invalid_final"
            ]
        )
        for malformed in (
            None,
            {"choices": []},
            {"choices": [None]},
            {"choices": [{"message": []}]},
        ):
            with self.assertRaises(ValueError):
                tools.parse_final(malformed)

    def test_transport_failure_propagates_without_a_fictitious_result(self):
        def fail(_):
            raise TimeoutError("synthetic fault")

        with self.assertRaises(TimeoutError):
            tools.run(item(), {"messages": item()["messages"]}, fail)


if __name__ == "__main__":
    unittest.main()
