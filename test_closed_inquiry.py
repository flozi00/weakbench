"""Public arbitrary fixtures, unrelated to the private evidence archive."""

from copy import deepcopy
import json
import unittest
import closed_inquiry as check


def fixture(lang):
    word = "Becher" if lang == "de" else "cups"

    def doc(key, origin, issue, copy, value):
        return {
            "id": key,
            "origin": origin,
            "issue": issue,
            "copy": copy,
            "entries": [
                {"subject": word, "round": "X", "lo": value, "hi": value, "text": f"{word} X {value}."}
            ],
        }

    return {
        "query": "Q",
        "venue": "B",
        "subject": word,
        "round": "X",
        "threshold": 3,
        "rule": "eq",
        "budget": 5,
        "shelves": [
            {"venue": "A", "key": "R1", "documents": [doc("S", "L", 1, False, 3), doc("T", "L", 2, True, 3)]},
            {
                "venue": "B",
                "key": "R2",
                "documents": [
                    doc("P", "M", 1, False, 3),
                    doc("R", "M", 2, False, 4),
                    doc("N", "K", 1, False, 3),
                    doc("V", "M", 3, True, 3),
                ],
            },
        ],
    }


def fixture_candidate(messages, lang):
    stages = sum(m["role"] == "tool" for m in messages)
    if stages < 4:
        name, field, values = (
            ("find_records", "term", ["Q"])
            if stages == 0
            else ("find_records", "term", ["R2"])
            if stages == 1
            else ("read_record", "id", ["N", "R"])
        )
        calls = [
            {"id": v, "type": "function", "function": {"name": name, "arguments": json.dumps({field: v})}}
            for v in values
        ]
        return {"content": "", "tool_calls": calls}
    word = "Becher" if lang == "de" else "cups"
    return {
        "content": json.dumps(
            {
                "verdict": "strittig" if lang == "de" else "conflicting",
                "sources": [
                    {
                        "id": "R",
                        "position": "dagegen" if lang == "de" else "against",
                        "quote": f"{word} X 4.",
                    },
                    {"id": "N", "position": "dafür" if lang == "de" else "for", "quote": f"{word} X 3."},
                ],
            },
            ensure_ascii=False,
        )
    }


def positive(item):
    wire = list(item["messages"])
    runtime = check.runtime(item["expected"], item["lang"])
    execute = runtime["tool_exec"]
    while True:
        ans = fixture_candidate(wire, item["lang"])
        if not ans.get("tool_calls"):
            return dict(ans, observation_trace=wire[len(item["messages"]) :])
        wire.append(dict(ans, role="assistant"))
        for c in ans["tool_calls"]:
            f = c["function"]
            wire.append(
                {"role": "tool", "tool_call_id": c["id"], "content": execute(f["name"], f["arguments"])}
            )

        runtime["tool_batch_finished"]()


class InquiryTests(unittest.TestCase):
    def setUp(self):
        self.item = {
            "expected": fixture("en"),
            "lang": "en",
            "messages": [{"role": "user", "content": "test"}],
        }
        self.good = positive(self.item)

    def test_research_and_languages(self):
        for lang in ("de", "en"):
            item = dict(self.item, expected=fixture(lang), lang=lang)
            self.assertTrue(check.validate_expected(item["expected"]))
            result = check.score(item, positive(item))
            self.assertTrue(result["correct"])
            self.assertEqual(set(result["detail"]), check.DETAIL_KEYS - {"inquiry_uncertainty_recognized"})
        self.assertTrue(
            check.score(
                self.item,
                dict(self.good, observation_trace=self.item["messages"] + self.good["observation_trace"]),
            )["correct"]
        )

    def test_spoofed_or_missing_wire(self):
        trace = self.good["observation_trace"]
        for key in ("role", "tool_call_id", "content"):
            fake = deepcopy(trace)
            fake[-1][key] = "bogus"
            self.assertFalse(check.score(self.item, dict(self.good, observation_trace=fake))["correct"])
        self.assertFalse(check.score(self.item, {"content": self.good["content"]})["correct"])
        self.assertFalse(check.score(self.item, dict(self.good, observation_trace=trace[:4]))["correct"])
        self.assertFalse(check.score(self.item, dict(self.good, tool_calls=[{}]))["correct"])
        dup = deepcopy(trace)
        dup[0]["tool_calls"][0]["function"]["arguments"] = '{"term":"Q","term":"Q"}'
        self.assertFalse(check.score(self.item, dict(self.good, observation_trace=dup))["correct"])

    def test_forged_citations(self):
        for field, value in (
            ("id", "P"),
            ("id", "V"),
            ("id", "absent"),
            ("position", "for"),
            ("quote", "something else"),
        ):
            ans = json.loads(self.good["content"])
            ans["sources"][0][field] = value
            self.assertFalse(check.score(self.item, dict(self.good, content=json.dumps(ans)))["correct"])
        ans = json.loads(self.good["content"])
        ans["sources"].append(deepcopy(ans["sources"][0]))
        self.assertFalse(check.score(self.item, dict(self.good, content=json.dumps(ans)))["correct"])

    def test_attribution_and_conditional_denominators(self):
        def details(id, quote):
            ans = json.loads(self.good["content"])
            ans["sources"][0].update(id=id, quote=quote)
            return check.score(self.item, dict(self.good, content=json.dumps(ans)))["detail"]

        # R records 4; P is its stale predecessor, honestly recording 3.
        wrong_attribution = details("R", "cups X 3.")
        self.assertEqual(
            [
                wrong_attribution[k]
                for k in ("inquiry_ids_authentic", "inquiry_quotes_attributed", "inquiry_quotes_exact")
            ],
            [1, 0, 0],
        )
        obsolete = details("P", "cups X 3.")
        self.assertEqual(
            [
                obsolete[k]
                for k in ("inquiry_ids_authentic", "inquiry_quotes_attributed", "inquiry_sources_current")
            ],
            [1, 1, 0],
        )
        invented = details("ghost", "cups X 3.")
        self.assertEqual(
            [invented[k] for k in ("inquiry_ids_authentic", "inquiry_quotes_attributed")], [0, 0]
        )
        for value, only in (
            (4, "inquiry_conflict_recognized"),
            (None, "inquiry_uncertainty_recognized"),
            (3, None),
        ):
            item = deepcopy(self.item)
            e = item["expected"]["shelves"][1]["documents"][1]["entries"][0]
            e["lo"] = e["hi"] = value
            for malformed in (
                "",
                "{",
                "null",
                '{"verdict":"unresolved","sources":[false]}',
                '{"verdict":"unresolved","sources":[]}',
            ):
                d = check.score(item, {"content": malformed})["detail"]
                self.assertEqual(check.CONDITIONAL_KEYS.intersection(d), {only} if only else set())
                if only:
                    self.assertEqual(d[only], 0)
        self.assertEqual(check.score(self.item, self.good)["detail"]["inquiry_conflict_recognized"], 1)

    def test_no_unlock_in_same_assistant_turn(self):
        t = deepcopy(self.good["observation_trace"])
        t[0]["tool_calls"].extend(t[2]["tool_calls"])
        del t[2]
        self.assertFalse(
            check.score(self.item, dict(self.good, observation_trace=t))["detail"]["inquiry_trace_valid"]
        )
        exe = check.runtime(self.item["expected"], "en")["tool_exec"]
        self.assertEqual(json.loads(exe("read_record", {"id": "N"})), {"error": "unlisted_id"})
        for _ in range(4):
            exe("find_records", {"term": "?"})
        self.assertEqual(json.loads(exe("find_records", {"term": "Q"})), {"error": "budget"})

    def test_batch_visibility_call_budget_and_fresh_runtime(self):
        spec = self.item["expected"]
        a, b = (check.runtime(spec, "en") for _ in range(2))
        read = lambda rt, name, arg: json.loads(rt["tool_exec"](name, arg))
        self.assertIn("collections", read(a, "find_records", {"term": "Q"}))
        self.assertEqual(read(a, "find_records", {"term": "R2"}), {"matches": []})
        a["tool_batch_finished"]()
        self.assertIn("records", read(a, "find_records", {"term": "R2"}))
        self.assertEqual(read(a, "read_record", {"id": "N"}), {"error": "unlisted_id"})
        a["tool_batch_finished"]()
        self.assertIn("lines", read(a, "read_record", {"id": "N"}))
        self.assertEqual(read(b, "read_record", {"id": "N"}), {"error": "unlisted_id"})
        self.assertEqual(read(b, "find_records", {"term": "R2"}), {"matches": []})
        for _ in range(30):
            self.assertEqual(read(a, "read_record", {"id": "N"}), {"error": "budget"})
        self.assertEqual(a["tool_call_budget"], 5)

    def test_runner_retries_and_rejects_an_entire_overbudget_batch(self):
        from unittest.mock import patch
        import common
        import hidden
        import quality

        lang = "en"
        row = {"id": "public-inquiry-fixture", "prompt": "fixture", "expected": fixture(lang)}
        cells = {("source_inquiry", lang): [row]}
        item = hidden.load(cells, "source_inquiry", lang, 1, 0)[0]
        task = hidden.tasks(cells, None)["source_inquiry"]
        body = {"model": "fixture", "messages": item["messages"], "tools": item["tools"]}
        original = item["start_tool_session"]
        counts = []

        def start():
            counts.append(0)
            fresh = original()
            execute = fresh["tool_exec"]

            def counted(*args):
                counts[-1] += 1
                return execute(*args)

            return dict(fresh, tool_exec=counted)

        item["start_tool_session"] = start
        for mode in ("valid", "valid", "overflow_immediately", "overflow_after_reads"):

            def request(url, path, payload, timeout):
                msg = fixture_candidate(payload["messages"], lang)
                if mode == "overflow_immediately" or (
                    mode == "overflow_after_reads" and not msg.get("tool_calls")
                ):
                    k = 129 if mode == "overflow_immediately" else 2
                    msg = {
                        "content": msg.get("content", ""),
                        "tool_calls": [
                            {
                                "type": "function",
                                "id": f"x{i}",
                                "function": {"name": "find_records", "arguments": '{"term":"Q"}'},
                            }
                            for i in range(k)
                        ],
                    }
                return {
                    "choices": [
                        {"message": msg, "finish_reason": "tool_calls" if msg.get("tool_calls") else "stop"}
                    ]
                }

            with patch.object(common, "request", side_effect=request) as transport:
                record = quality.run_item(
                    "http://offline.invalid", "source_inquiry", task, item, body, 2, private=True
                )
            self.assertIsNone(record["error"])
            self.assertIs(record["correct"], mode == "valid")
            self.assertEqual(transport.call_count, 1 if mode == "overflow_immediately" else 4)
        self.assertEqual(counts, [4, 4, 0, 4])

    def test_invalid_metadata(self):
        for kind in ("ambiguous_revision", "duplicate_key", "inverted_interval", "short_budget"):
            s = fixture("en")
            if kind == "ambiguous_revision":
                s["shelves"][1]["documents"][0]["issue"] = 2
            elif kind == "duplicate_key":
                s["shelves"][1]["key"] = "R1"
            elif kind == "inverted_interval":
                s["shelves"][1]["documents"][0]["entries"][0]["hi"] = 0
            else:
                s["budget"] = 3
            self.assertFalse(check.validate_expected(s))


if __name__ == "__main__":
    unittest.main()
