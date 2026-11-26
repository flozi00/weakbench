"""Hermetic checks for the closed quality runner. All test data are public placeholders."""
import contextlib
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import io
import itertools
import json
import os
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import urllib.request

import common
import check_response
import hidden
import quality
import test_closed_auction as auction_fixture
import test_closed_fusion as fusion_fixture
import test_closed_incidence as incidence_fixture
import test_closed_commitments as booking_fixture
import test_closed_perspective as perspective_fixture
import test_closed_proofs as proof_fixture
import test_closed_reference as reference_fixture
import test_closed_relay as relay_fixture
import test_closed_backstage as backstage_fixture
import test_closed_bag as bag_fixture
import test_closed_imagery as imagery_fixture
import test_closed_inquiry as inquiry_fixture
import test_closed_halfplanes as halfplane_fixture
import test_closed_quorum as quorum_fixture


class ClosedSuiteTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.tmp = Path(temporary.name)
        self.enter(patch.dict(os.environ, {"BENCH_CACHE": str(self.tmp / "cache"), "HF_TOKEN": "dummy"}))
        self.enter(patch.dict(hidden.AUDIO_COUNTS, {}, clear=True))
        self.enter(patch.dict(hidden.VISUAL_COUNTS, {}, clear=True))  # the default text portfolio fixture
        self.enter(patch.multiple(common, RUNS=self.tmp / "runs", RESULTS=self.tmp / "results"))
        rows = []
        self.answers = {}
        for family in hidden.FULL_COUNTS:
            for lang in hidden.LANGS:
                for i in range(4):
                    prompt = f"sentinel-input/{family}/{lang}/{i}"
                    answer = ["sentinel-answer", i]
                    if family == "workshop_schedule":
                        fields = {"mode": "program", "expected": {
                            "tasks": {"A": {"duration": 1, "release": 0, "deadline": 2, "resources": ["X"], "after": []}},
                            "closures": {"X": []}, "optimum": 1}}
                        answer = {"A": 0}
                    elif family == "language_evidence":
                        intro = ("Bitte vergleiche die beiden Werte und gib an, welche Wahl sich daraus ergibt."
                                 if lang == "de" else "Compare the two values and explain which choice follows from the result.")
                        prompt = f"sentinel-input/{intro}/{i}"
                        fields = {"mode": "program", "expected": {"values": {"Testa": 7, "Testo": 3}, "prefer": "max"}}
                        answer = ({"begründung": "Für Testa ergibt sich ein Endwert von 7. Für Testo ergibt sich ein Endwert von 3.",
                                   "antwort": "Meine Wahl fällt damit auf Testa. Der absolute Unterschied beträgt hier 4."}
                                  if lang == "de" else
                                  {"justification": "The final value for Testa is 7. The final value for Testo is 3.",
                                   "answer": "My choice is therefore Testa. The two values differ by 4."})
                    elif family == "adaptive_measurement":
                        prompt = f"sentinel-input/{family}/{lang}/{i // 2}"
                        fields = {"mode": "program", "expected": {
                            "budget": 2, "probes": ["side", "left", "right"], "target": ("A", "C")[i % 2],
                            "table": {"A": [0, 0, 0], "B": [0, 1, 0], "C": [1, 0, 0], "D": [1, 0, 1]}}}
                    elif family == "flow_certificate":
                        fields = {"mode": "program", "expected": {
                            "nodes": ["s", "t", "isolated"], "source": "s", "sink": "t",
                            "arcs": {"edge": ["s", "t", 2 + i]}}}
                        answer = {"value": 2 + i, "flow": {"edge": 2 + i}, "source_side": ["isolated", "s"]}
                    elif family == "commitment_repair":
                        spec = booking_fixture.fixture()
                        parties, slots = spec["parties"], spec["slots"]
                        def options(stage):
                            return {k: [slots[n] for n in choices] for k, choices in zip(parties, stage)}
                        first, second, third = map(options, spec["stages"])
                        initial = {"choices": first, "weights": dict(zip(parties, spec["weights"]))}
                        prompt = f"sentinel-input/{family}/{lang}/{json.dumps(initial)}/{i}"
                        updates = [{k: v for k, v in new.items() if old[k] != v}
                                   for old, new in ((first, second), (second, third))]
                        followups = [f"sentinel-input/update/{lang}/{json.dumps(update)}/{i}" for update in updates]
                        fields = {"mode": "program", "expected": spec, "followups": followups}
                        answer = json.loads(booking_fixture.text(spec, booking_fixture.CommitmentTests.start))
                        for turn in followups:
                            self.answers[turn] = booking_fixture.text(spec, booking_fixture.CommitmentTests.short)
                    elif family == "language_relay":
                        spec, questions, answers = relay_fixture.wire_fixture(lang)
                        marked = [f"sentinel-input/{part}/{i}" for part in questions]
                        prompt, *followups = marked
                        fields = {"mode": "program", "expected": spec, "followups": followups}
                        answer = answers[0]
                        for request, response in zip(followups, answers[1:]):
                            self.answers[request] = json.dumps(response, ensure_ascii=False)
                    elif family == "quoted_perspective":
                        prompt = f"sentinel-input/{family}/{perspective_fixture.REQUESTS[lang]}/{i}"
                        fields = {"mode": "program", "expected": perspective_fixture.FACTS}
                        answer = perspective_fixture.render(lang)
                    elif family == "localized_proof":
                        expected, request, answer = proof_fixture.wire_fixture(lang)
                        prompt = f"sentinel-input/{family}/{request}/{i}"
                        fields = {"mode": "program", "expected": expected}
                    elif family == "listener_reference":
                        fields = {"mode": "program", "expected": reference_fixture.fixture(lang)}
                        answer = "Die Karte ist leise und warm." if lang == "de" else "The card is warm and quiet."
                    elif family == "backstage_reveal":
                        fields = {"mode": "program", "expected": backstage_fixture.fixture()}
                        answer = backstage_fixture.witness(lang)
                    elif family in hidden.EXACT_FIELDS:
                        answer = dict.fromkeys(hidden.EXACT_FIELDS[family], i)
                        fields = {"mode": "exact", "expected": answer}
                    elif family == "bag_query":
                        fields = {"mode": "program", "expected": bag_fixture.fixture()}
                        answer = bag_fixture.witness()
                    elif family == "imagery_chain":
                        fields = {"mode": "program", "expected": imagery_fixture.fixture(lang)}
                        answer = imagery_fixture.witness(lang)
                    elif family == "source_inquiry":
                        fields = {"mode": "program", "expected": inquiry_fixture.fixture(lang)}
                    elif family == "halfplane_proof":
                        fields = {"mode": "program", "expected": halfplane_fixture.fixture()}
                        answer = halfplane_fixture.witness()
                    elif family == "comparator_fabric":
                        fields = {"mode": "program", "expected": {"wires": 2, "ticks": [[[0, 1]]],
                                  "budget": 1, "targets": [[1, 1]]}}
                        answer = {"stages": [[[1, 0]]]}
                    elif family == "quoted_quorum":
                        spec = quorum_fixture.fixture(lang)
                        prompt = f"sentinel-input/{family}/{quorum_fixture.sample_prompt(lang, spec)}/{i}"
                        fields = {"mode": "program", "expected": spec}
                        answer = quorum_fixture.answer(lang, spec)
                    elif family == "failure_incidence":
                        fields = {"mode": "program", "expected": incidence_fixture.fixture()}
                        answer = incidence_fixture.accepted()
                    elif family == "sentence_fusion":
                        fields = {"mode": "program", "expected": fusion_fixture.fixture()}
                        answer = ("Ida sichtete Otto nicht, bevor Otto Elin fand." if lang == "de"
                                  else "Ida did not spot Otto before Otto found Elin.")
                    elif family == "package_auction":
                        fields = {"mode": "program", "expected": {"bids": auction_fixture.offers()}}
                        answer = {"accepted": ["Ax", "By"], "payments": {"A": 3, "B": 0, "C": 0}}
                    elif family == "stream_transducer":
                        answer = {"start": "q", "states": {"q": {"p": ["q", "m"]}}}
                        fields = {"mode": "program", "expected": {"inputs": ["p"], "outputs": ["m"],
                                  "reference": answer, "max_states": 2}}
                    elif family in hidden.JUDGE_TASKS:
                        fields = {"mode": "judge", "rubric": ["sentinel-criterion", "sentinel-other"]}
                    else:
                        fields = {"mode": "exact", "expected": answer}
                    rows.append({"family": family, "language": lang, "id": f"{family}_{lang}_{i}", "prompt": prompt, **fields})
                    self.answers[prompt] = answer if family in ("listener_reference", "sentence_fusion", "imagery_chain", "backstage_reveal") else json.dumps(answer, ensure_ascii=False)
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
            if family == "adaptive_measurement":
                self.assertEqual(quick[0]["messages"], quick[1]["messages"])
                self.assertNotEqual(quick[0]["expected"]["target"], quick[1]["expected"]["target"])
                self.assertEqual(quick[0]["tools"], quick[1]["tools"])
            elif family in hidden.PAIRED_VARIANTS:
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

    def test_report_release_checks_prompt_language(self):
        for family in hidden.LANGUAGE_REPORTS:
            rows = [json.loads(line) for line in self.payload.splitlines()]
            row = next(r for r in rows if r["family"] == family and r["language"] == "de")
            row["prompt"] = perspective_fixture.REQUESTS["en"]
            mismatched = "\n".join(json.dumps(r) for r in rows).encode()
            with patch.object(hidden, "DATA_SHA256", hashlib.sha256(mismatched).hexdigest()):
                with self.assertRaisesRegex(ValueError, "invalid private release row"):
                    hidden.decode(mismatched)

    def test_relay_release_checks_all_three_request_languages(self):
        for lang in hidden.LANGS:
            for index, intended in enumerate(hidden.closed_relay.turn_languages(lang)):
                rows = [json.loads(line) for line in self.payload.splitlines()]
                row = next(r for r in rows if r["family"] == "language_relay" and r["language"] == lang)
                other = relay_fixture.REQ["de" if intended == "en" else "en"]
                if index:
                    row["followups"][index - 1] = other
                else:
                    row["prompt"] = other
                data = "\n".join(json.dumps(r) for r in rows).encode()
                with patch.object(hidden, "DATA_SHA256", hashlib.sha256(data).hexdigest()), self.assertRaisesRegex(
                        ValueError, "invalid private release row"):
                    hidden.decode(data)

    def test_named_proof_fields_are_graded_independently(self):
        task = hidden.tasks(hidden.release(), None)["distance_repair"]
        item = {"expected": {"edits": 0, "layouts": [[0, 3]], "gaps": [3]}}
        partial = task["score"](item, {"content": '{"edits":false,"gaps":[3],"layouts":[[0,3]]}'})
        self.assertFalse(partial["correct"])
        self.assertEqual([partial["detail"][k] for k in hidden.EXACT_FIELDS["distance_repair"].values()], [0, 1, 1])
        for content in ("oops", '{"edits":0,"gaps":[3]}'):
            checks = task["score"](item, {"content": content})["detail"]
            self.assertTrue(all(checks[k] == 0 for k in hidden.EXACT_FIELDS["distance_repair"].values()))

    def test_exact_diagnostics_have_explicit_denominators(self):
        def check(expected, answer):
            result = hidden.exact({"expected": expected}, {"content": answer})
            return result["correct"], hidden.public_checks("revision_ledger", result["detail"])

        correct, checks = check({"x": [1], "y": True}, '{"x":[1],"y":1}')
        self.assertFalse(correct)
        self.assertEqual(checks, {"valid_json": 1.0, "json_shape_ok": 1.0, "answer_components_correct": 0.5})
        for answer, valid in [('{"x":[1]}', 1.0), ('{"x":1,"x":2}', 0.0), ('[]', 1.0), ('no JSON', 0.0)]:
            self.assertEqual(check({"x": 1, "y": 2}, answer), (False, {"valid_json": valid, "json_shape_ok": 0.0}))
        self.assertEqual(check([0, 2, 3], '[false,2,4]')[1]["answer_components_correct"], 1/3)
        self.assertNotIn("answer_components_correct", check([1, 2], "[1]")[1])
        self.assertEqual(check([], "[]")[1]["answer_components_correct"], 1.0)

    def test_offline_completion(self):
        item = hidden.load(hidden.release(), "language_evidence", "de", 1, 0)[0]
        response = self.tmp / "completion.json"
        argv = ["--task", "language_evidence", "--lang", "de", "--id", item["id"], "--response", str(response)]
        answer = self.answers[item["messages"][0]["content"]]
        completions = [(None, 0), ("We compare the two values and choose the larger one.", 1)]
        for reasoning, expected_exit in completions:
            response.write_text(json.dumps({"choices": [{"message": {"content": answer, "reasoning": reasoning}}]}))
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(check_response.main(argv), expected_exit)
            report = json.loads(out.getvalue())
            self.assertEqual(report["checks"]["reasoning_present"], float(reasoning is not None))
            self.assertNotIn("Testa", out.getvalue())
        response.write_text("invalid PRIVATE_SENTINEL")
        err = io.StringIO()
        with contextlib.redirect_stderr(err), self.assertRaises(SystemExit) as failure:
            check_response.main(argv)
        self.assertEqual(failure.exception.code, 2)
        self.assertNotIn("PRIVATE_SENTINEL", err.getvalue())

    def test_proof_live_and_saved_completions(self):
        name = "localized_proof"
        task = hidden.tasks(hidden.release(), None)[name]
        args = SimpleNamespace(think="auto", max_tokens=None, temperature=None)
        native = {"en": "I check the information and explain why the rules give this result.",
                  "de": "Ich prüfe die Angaben und erkläre, warum dieses Ergebnis aus den Regeln folgt."}
        for lang in hidden.LANGS:
            item, = task["load"](lang, 1, quality.SEED)
            prompt = item["messages"][0]["content"]
            body = quality.build_body(name, task, item, args, "synthetic")
            self.assertEqual(body["messages"], [{"role": "user", "content": prompt}])
            self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": True, "thinking": True})
            self.assertNotIn("expected", body)
            answer = self.answers[prompt]
            fields = json.loads(answer)
            why = hidden.closed_language.FIELDS[lang][0]
            naked = json.dumps({**fields, why: ""})
            other = "en" if lang == "de" else "de"
            for content, reasoning, correct, verified in (
                    (answer, None, True, 0), (answer, native[lang], True, 1),
                    (answer, native[other], False, 0), (naked, native[lang], False, 0)):
                completion = {"choices": [{"message": {"content": content, "reasoning": reasoning}, "finish_reason": "stop"}]}
                with patch.object(common, "request", return_value=completion) as fetch:
                    rec = quality.run_item("http://synthetic", name, task, item, body, 2, private=True)
                fetch.assert_called_once_with("http://synthetic", "/v1/chat/completions", body, timeout=2)
                self.assertIs(rec["correct"], correct)
                self.assertEqual(rec["detail"]["all_channels_verified"], verified)
                self.assertEqual(rec["detail"]["answer_facts_ok"], 1)
                self.assertIsNone(rec["reply"])
                self.assertIsNone(rec["gold"])
                path = self.tmp / "proof-response.json"
                path.write_text(json.dumps(completion))
                out = io.StringIO()
                with patch.object(common, "request", side_effect=AssertionError), contextlib.redirect_stdout(out):
                    code = check_response.main(["--task", name, "--lang", lang, "--id", item["id"], "--response", str(path)])
                report = json.loads(out.getvalue())
                self.assertEqual(code, int(not correct))
                self.assertEqual(report["checks"], rec["detail"])
                self.assertNotIn("Ada/Ray", json.dumps(rec))
                self.assertNotIn("sentinel-input", out.getvalue())

    def test_offline_observations(self):
        item = hidden.load(hidden.release(), "adaptive_measurement", "de", 1, 0)[0]
        known = item["expected"]["target"]
        completion = {"choices": [{"message": {"content": json.dumps({"sample": known})}}]}
        trace = []
        for probe in ("side", "left" if known == "A" else "right"):
            fn = {"name": "observe", "arguments": json.dumps({"probe": probe})}
            trace.append({"role": "assistant", "tool_calls": [{"id": probe, "type": "function", "function": fn}]})
            trace.append({"role": "tool", "tool_call_id": probe, "content": item["tool_exec"]("observe", fn["arguments"])})
        response = self.tmp / "saved.json"
        argv = ["--task", "adaptive_measurement", "--lang", "de", "--id", item["id"], "--response", str(response)]
        forged = [*trace[:-1], {**trace[-1], "content": '{"probe":"sentinel-private","value":0}'}]
        for document, exit_code in (({"messages": trace, "completion": completion}, 0),
                                    ({"messages": item["messages"] + trace, "completion": completion}, 0),
                                    ({"messages": forged, "completion": completion}, 1),
                                    ({"messages": trace[:-2], "completion": completion}, 1), (completion, 1)):
            response.write_text(json.dumps(document))
            out = io.StringIO()
            with patch.object(common, "request", side_effect=AssertionError), contextlib.redirect_stdout(out):
                self.assertEqual(check_response.main(argv), exit_code)
            report = json.loads(out.getvalue())
            self.assertEqual(report["correct"], not exit_code)
            self.assertEqual(set(report["checks"]), set(hidden.PROGRAM_METRICS["adaptive_measurement"]))
            self.assertNotIn("sentinel-", out.getvalue())

    def test_followup_schema(self):
        rows = [json.loads(line) for line in self.payload.splitlines()]
        row = next(r for r in rows if r["family"] == "commitment_repair")
        for followups in (None, [], ["one"], ["one", "two", "three"], ["good", "  "], ["good", {"content": "bad"}]):
            row["followups"] = followups
            malformed = "\n".join(json.dumps(r) for r in rows).encode()
            with patch.object(hidden, "DATA_SHA256", hashlib.sha256(malformed).hexdigest()):
                with self.assertRaisesRegex(ValueError, "invalid private release row"):
                    hidden.decode(malformed)

    def test_real_multiturn_http_and_saved_completions(self):
        name = "commitment_repair"
        task = hidden.tasks(hidden.release(), None)[name]
        item, = task["load"]("en", 1, 11)
        posts, completions = [], []

        class Server(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                posts.append(body)
                messages = body["messages"]
                # The responder only knows the constraints and previous commitments in the HTTP body.
                initial = json.loads(messages[0]["content"].split("/")[-2])
                choices = initial["choices"].copy()
                for update in messages[2::2]:
                    choices.update(json.loads(update["content"].split("/")[-2]))
                previous = json.loads(messages[-2]["content"]) if len(messages) > 1 else {}
                best = None
                for slots in itertools.product(*choices.values()):
                    if len(set(slots)) != len(slots):
                        continue
                    booking = dict(zip(choices, slots))
                    moved = [k for k in previous if previous[k] != booking[k]]
                    rank = (sum(initial["weights"][k] for k in moved), len(moved))
                    if best is None or rank < best[0]:
                        best = rank, booking
                choice = {"message": {"role": "assistant", "content": json.dumps(best[1])}, "finish_reason": "stop"}
                result = {"choices": [choice], "usage": {"prompt_tokens": 17, "completion_tokens": 9}}
                completions.append(result)
                data = json.dumps(result).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        with HTTPServer(("127.0.0.1", 0), Server) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                args = SimpleNamespace(think="auto", max_tokens=None, temperature=None)
                body = quality.build_body(name, task, item, args, "synthetic")
                rec = quality.run_item(f"http://127.0.0.1:{server.server_port}", name, task, item, body, 10, private=True)
            finally:
                server.shutdown()
                thread.join()
        self.assertEqual(item["messages"], item["turns"][0])
        self.assertEqual([len(p["messages"]) for p in posts], [1, 3, 5])
        visible = []
        for turn, p, completion in zip(item["turns"], posts, completions):
            visible += turn
            self.assertEqual(p["messages"], visible)
            self.assertEqual(set(p), set(body))
            self.assertNotIn("expected", p)
            self.assertNotIn("turns", p)
            visible.append(completion["choices"][0]["message"])
        self.assertTrue(rec["correct"])
        self.assertEqual((rec["prompt_tokens"], rec["completion_tokens"]), (51, 27))
        self.assertNotIn("sentinel-", json.dumps(rec))

        response = self.tmp / "dialogue.json"
        argv = ["--task", name, "--lang", "en", "--id", item["id"], "--response", str(response)]
        def run_offline(document):
            response.write_text(json.dumps(document))
            stdout = io.StringIO()
            with patch.object(common, "request", side_effect=AssertionError), contextlib.redirect_stdout(stdout):
                code = check_response.main(argv)
            return code, json.loads(stdout.getvalue())

        code, result = run_offline({"completions": completions})
        self.assertEqual(code, 0)
        self.assertEqual(result["checks"], rec["detail"])
        self.assertNotIn("sentinel-", json.dumps(result))
        for stage in range(3):
            bad = json.loads(json.dumps(completions))
            message = bad[stage]["choices"][0]["message"]
            message["tool_calls"] = [{"function": {"name": "unrequested", "arguments": "{}"}}]
            with self.subTest(stage=stage):
                code, result = run_offline({"completions": bad})
                self.assertEqual(code, 1)
                self.assertEqual(result["checks"]["valid_json"], 0)
                with patch.object(common, "request", side_effect=bad):
                    live = quality.run_item("http://unused", name, task, item, body, 10, private=True)
                self.assertEqual(result["checks"], live["detail"])
        broken_message = {"choices": [{"message": {"role": "user", "content": "{}"}}]}
        for invalid in (completions[-1], {"completions": completions[-1:]}, {"completions": [*completions, completions[2]]},
                        {"completions": [*completions[:2], broken_message]}, {"completions": [*completions[:2], {"choices": []}]},
                        {"completions": completions, "turns": []}):
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                run_offline(invalid)

    def test_relay_live_history_and_strict_offline_parity(self):
        name = "language_relay"
        task = hidden.tasks(hidden.release(), None)[name]
        args = SimpleNamespace(think="auto", max_tokens=None, temperature=None)
        captured = []
        answers = self.answers

        class Server(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                msgs = body["messages"]
                index = len(msgs) // 2
                request = msgs[-1]["content"]
                answer = json.loads(answers[request])
                lang = "de" if "antwort" in answer else "en"
                if wrong_report and index == 0:
                    answer = {}
                trace = relay_fixture.TRACE[lang]
                if index == 0 and lang == "de":
                    trace = relay_fixture.TRACE["en"]  # correct visible response, switched early reasoning
                msg = {"role": "assistant", "content": json.dumps(answer, ensure_ascii=False)}
                if index == 1:
                    msg["content"] = "<think>" + trace + "</think>" + msg["content"]
                else:
                    msg["reasoning" if index == 2 else "reasoning_content"] = trace
                completion = {"choices": [{"message": msg, "finish_reason": "stop"}],
                              "usage": {"prompt_tokens": 8, "completion_tokens": 5}}
                captured.append((body, completion))
                data = json.dumps(completion).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        with HTTPServer(("127.0.0.1", 0), Server) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                for first, wrong_report in (("en", False), ("de", False), ("en", True)):
                    captured.clear()
                    item, = task["load"](first, 1, 41)
                    body = quality.build_body(name, task, item, args, "local")
                    self.assertNotIn("chat_template_kwargs", body)
                    for mode, enabled in (("on", True), ("off", False)):
                        forced_args = SimpleNamespace(think=mode, max_tokens=None, temperature=None)
                        explicit = quality.build_body(name, task, item, forced_args, "local")
                        self.assertEqual(explicit["chat_template_kwargs"], {"enable_thinking": enabled, "thinking": enabled})
                        self.assertEqual(explicit["messages"], body["messages"])
                        if (mode == "on" and first == "de") or (mode == "off" and wrong_report):
                            body = explicit
                    record = quality.run_item(f"http://127.0.0.1:{server.server_port}", name, task,
                                              item, body, 5, private=True)
                    visible, completions = [], []
                    self.assertEqual(len(captured), 3)
                    for index, (turn, (sent, response)) in enumerate(zip(item["turns"], captured)):
                        visible += turn
                        self.assertEqual(sent, {**body, "messages": visible})
                        for trace in relay_fixture.TRACE.values():
                            self.assertNotIn(trace, json.dumps(sent, ensure_ascii=False))
                        response_text = "{}" if wrong_report and index == 0 else answers[turn[0]["content"]]
                        visible.append({"role": "assistant", "content": response_text})
                        completions.append(response)
                    self.assertIs(record["correct"], first == "en" and not wrong_report)
                    self.assertEqual(record["detail"]["correct_with_all_channels"], float(record["correct"]))
                    self.assertEqual(record["detail"]["turn1_reasoning_language_ok"], float(first == "en"))
                    self.assertEqual(record["detail"]["visible_reports_correct"], float(not wrong_report))
                    self.assertEqual(record["detail"]["turn3_turn_correct"], 1)
                    self.assertEqual((record["prompt_tokens"], record["completion_tokens"]), (24, 15))
                    record_text = json.dumps(record, ensure_ascii=False)
                    self.assertNotIn("sentinel-input", record_text)
                    for trace in relay_fixture.TRACE.values():
                        self.assertNotIn(trace, record_text)

                    path = self.tmp / "relay-response.json"
                    def offline(document):
                        path.write_text(json.dumps(document))
                        out = io.StringIO()
                        with patch.object(common, "request", side_effect=AssertionError), contextlib.redirect_stdout(out):
                            code = check_response.main(["--task", name, "--lang", first, "--id", item["id"],
                                                        "--response", str(path)])
                        return code, json.loads(out.getvalue())["checks"]

                    code, detail = offline({"completions": completions})
                    self.assertEqual(code, int(first == "de" or wrong_report))
                    self.assertEqual(detail, record["detail"])
                    if first == "en" and not wrong_report:
                        # Neither a correct final turn nor an unexamined alias should hide an earlier failure.
                        for change, key, value in (
                            ({"reasoning_content": None}, "turn1_reasoning_present", 0),
                            ({"reasoning_content": "3 + 4 = 7"}, "turn1_reasoning_language_decidable", 0),
                            ({"reasoning": relay_fixture.TRACE["de"]}, "turn1_reasoning_language_ok", 0),
                            ({"content": "{}"}, "turn1_turn_correct", 0),
                            ({"refusal": "declined"}, "turn1_response_format_ok", 0),
                        ):
                            changed = json.loads(json.dumps(completions))
                            changed[0]["choices"][0]["message"].update(change)
                            code, detail = offline({"completions": changed})
                            self.assertEqual(code, int(key != "turn1_reasoning_present"))
                            self.assertEqual(detail[key], value)
                            self.assertEqual(detail["turn3_turn_correct"], 1)
                            with patch.object(common, "request", side_effect=changed):
                                result = quality.run_item("http://mock", name, task, item, body, 2, private=True)
                            self.assertEqual(detail, result["detail"])
                        for invalid in (completions[-1], {"completions": completions[1:]},
                                        {"completions": [*completions[:2], {"choices": []}]},
                                        {"completions": [None, *completions[1:]]},
                                        {"completions": completions, "turn_checks": [1, 1, 1]}):
                            with self.assertRaises(SystemExit) as failure, contextlib.redirect_stderr(io.StringIO()):
                                offline(invalid)
                            self.assertEqual(failure.exception.code, 2)
            finally:
                server.shutdown()
                thread.join()

    def test_cli(self):
        def request(url, route, body=None, **kw):
            if route == "/v1/models":
                return {"data": [{"id": url, "max_model_len": 8192}]}
            if route == "/version":
                return {"version": "mock"}
            self.assertEqual(route, "/v1/chat/completions")
            messages = body["messages"]
            prompt = messages[-1]["content"]
            if url == "http://judge":
                case = json.loads(prompt)
                controls = {p: grades for p, _, _, grades in hidden.CANARIES}
                content = json.dumps(controls.get(case["task"], [True, False]))
            elif prompt == quality.PREFLIGHT:
                content = "391"
            elif "source_inquiry" in messages[0]["content"]:
                if fail:
                    raise RuntimeError(messages[0]["content"])
                msg = inquiry_fixture.fixture_candidate(messages, messages[0]["content"].split("/")[2])
                return {"choices": [{"message": msg, "finish_reason": "tool_calls" if msg.get("tool_calls") else "stop"}]}
            elif "adaptive_measurement" in messages[0]["content"]:
                if fail:
                    raise RuntimeError(messages[0]["content"])
                readings = [json.loads(m["content"])["value"] for m in messages if m["role"] == "tool"]
                if len(readings) < 2:
                    probe = "side" if not readings else ("left", "right")[readings[0]]
                    call = {"id": "measurement", "type": "function", "function": {
                        "name": "observe", "arguments": json.dumps({"probe": probe})}}
                    return {"choices": [{"message": {"tool_calls": [call]}, "finish_reason": "tool_calls"}]}
                content = json.dumps({"sample": ("A", "C")[readings[0]]})
            else:
                self.assertTrue(prompt.startswith("sentinel-input/"))
                self.assertNotIn("sentinel-criterion", json.dumps(body))
                if fail:
                    raise RuntimeError(prompt)
                content = self.answers[prompt]
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
            if row["task"] in hidden.LANGUAGE_REPORTS:
                self.assertEqual(row["extra"]["reasoning_present"], 0)
                self.assertNotIn("all_channels_language_ok", row["extra"])
                self.assertEqual(row["extra"]["answer_facts_ok"], 1)
            if row["task"] == "language_relay":
                self.assertEqual(row["extra"]["visible_reports_correct"], 1)
                self.assertEqual(row["extra"]["reasoning_present_fraction"], 0)
                self.assertEqual(row["extra"]["correct_with_all_channels"], 0)
                self.assertEqual(row["extra"]["all_channels_verified"], 0)
                for turn in range(1, 4):
                    self.assertEqual(row["extra"][f"turn{turn}_reasoning_present"], 0)
                    self.assertNotIn(f"turn{turn}_reasoning_language_ok", row["extra"])
            if row["task"] == "listener_reference":
                self.assertEqual(row["extra"]["both_listeners_unique"], 1)
                self.assertEqual(row["extra"]["reference_minimal"], 1)
            if row["task"] == "package_auction":
                self.assertEqual(row["extra"]["auction_payments_correct"], 1)
            if row["task"] == "stream_transducer":
                self.assertEqual(row["extra"]["transducer_equivalent"], 1)
            if row["task"] == "localized_proof":
                self.assertEqual(row["extra"]["proof_steps_supported"], 1)
                self.assertEqual(row["extra"]["proof_steps_relevant"], 1)
                self.assertEqual(row["extra"]["proof_answer_agree"], 1)
            if row["task"] == "adaptive_measurement":
                self.assertEqual(row["extra"]["measurement_identified"], 1)
                self.assertEqual(row["extra"]["measurement_strategy_safe"], 1)
            if row["task"] == "flow_certificate":
                self.assertEqual(row["extra"]["flow_conservation"], 1)
                self.assertEqual(row["extra"]["flow_cut_equal"], 1)
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
