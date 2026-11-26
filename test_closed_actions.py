"""Hand-checked, synthetic desk examples separate from the private release rows."""

from copy import deepcopy
import json
import unittest

import closed_actions as actions


def fixture(mode="ack"):
    good = {
        "id": "BadgeA",
        "resource": "Lamp",
        "units": 2,
        "day": "2032-08-20",
        "state": "approved",
    }
    return {
        "ticket": "Example",
        "key": "SavedKey",
        "receipt": "Receipt42",
        "resource": "Lamp",
        "units": 2,
        "day": "2032-08-20",
        "mode": mode,
        "grants": [
            {**good, "state": "pending"} if mode == "missing" else good,
            {**good, "id": "BadgeB", "units": 3},
        ],
    }


def arguments(s, tool, key=None, grant=None):
    a = {"ticket": s["ticket"]}
    if tool != "desk_record":
        a["key"] = s["key"] if key is None else key
    if tool == "desk_submit":
        a["grant"] = s["grants"][0]["id"] if grant is None else grant
    return a


def wire(s, plan):
    desk = actions.Desk(s)
    trace = []
    for n, step in enumerate(plan):
        tool, *params = step
        a = arguments(s, tool, *params)
        uid = f"native-{n}"
        encoded = json.dumps(a)
        obj = desk.call(tool, encoded)
        trace.extend(
            [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": uid,
                            "type": "function",
                            "function": {"name": tool, "arguments": encoded},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": uid, "content": json.dumps(obj)},
            ]
        )
    return trace


def grade(s, plan, status="booked", receipt="Receipt42"):
    trace = wire(s, plan)
    return actions.score(
        {"expected": s, "lang": "en"},
        {
            "content": json.dumps(
                {"status": status, "ticket": "Example", "receipt": receipt}
            ),
            "observation_trace": trace,
        },
    )


LOOK = ("desk_record",)
READ = ("desk_status",)
SEND = ("desk_submit",)


class ActionPostconditions(unittest.TestCase):
    def test_without_grant_must_observe_and_withhold(self):
        s = fixture("missing")
        yes = grade(s, [LOOK], "request_approval", None)
        self.assertTrue(yes["correct"])
        self.assertEqual(yes["detail"]["action_withheld_without_grant"], 1.0)
        self.assertNotIn("action_verified_completion", yes["detail"])
        for plan in ([], [LOOK, SEND], [LOOK, ("desk_submit", None, "BadgeB")]):
            self.assertFalse(grade(s, plan, "request_approval", None)["correct"])

    def test_unknown_result_diverges_only_on_authoritative_read(self):
        one, two = fixture("uncertain_booked"), fixture("uncertain_absent")
        a, b = wire(one, [LOOK, SEND, READ]), wire(two, [LOOK, SEND, READ])
        self.assertEqual(a[:-1], b[:-1])
        self.assertEqual(json.loads(a[-1]["content"])["state"], "booked")
        self.assertEqual(json.loads(b[-1]["content"])["state"], "absent")
        self.assertTrue(grade(one, [LOOK, SEND, READ])["correct"])
        self.assertTrue(grade(two, [LOOK, SEND, READ, SEND, READ])["correct"])
        self.assertFalse(grade(one, [LOOK, SEND, SEND, READ])["correct"])
        self.assertFalse(grade(two, [LOOK, SEND, READ])["correct"])

    def test_ack_is_not_proof_and_verification_expires_after_a_new_attempt(self):
        s = fixture()
        self.assertFalse(grade(s, [LOOK, SEND])["correct"])
        self.assertFalse(grade(s, [LOOK, SEND, READ, SEND])["correct"])
        self.assertFalse(grade(s, [LOOK, SEND, READ], receipt="Invented")["correct"])
        self.assertTrue(grade(s, [LOOK, READ, SEND, READ, READ])["correct"])

    def test_processing_and_preexisting_orders(self):
        s = fixture("uncertain_processing")
        self.assertFalse(grade(s, [LOOK, SEND, READ])["correct"])
        self.assertTrue(grade(s, [LOOK, SEND, READ, READ])["correct"])
        s = fixture("resume_booked")
        self.assertTrue(grade(s, [LOOK, READ])["correct"])
        self.assertFalse(grade(s, [LOOK, SEND, READ])["correct"])
        self.assertFalse(grade(s, [READ, LOOK])["correct"])
        s = fixture("resume_absent")
        self.assertTrue(grade(s, [LOOK, READ, SEND, READ])["correct"])

    def test_actual_duplicate_is_separate_from_unnecessary_idempotent_retry(self):
        s = fixture("uncertain_booked")
        blind = grade(s, [LOOK, SEND, SEND, READ])
        double = grade(s, [LOOK, SEND, ("desk_submit", "FreshKey"), READ])
        for result in (blind, double):
            self.assertFalse(result["correct"])
            self.assertEqual(result["detail"]["action_recovered_uncertain"], 0.0)
        self.assertEqual(blind["detail"]["action_duplicate_effect"], 0.0)
        self.assertEqual(double["detail"]["action_duplicate_effect"], 1.0)

    def test_forged_tool_content_and_wire_ids(self):
        s = fixture()
        base = wire(s, [LOOK, SEND, READ])
        for i, key, v in (
            (1, "content", "{}"),
            (5, "tool_call_id", "other"),
            (5, "content", '{"state":"booked"}'),
        ):
            t = deepcopy(base)
            t[i][key] = v
            self.assertIsNone(actions.replay({"expected": s}, t))
        t = deepcopy(base)
        t[2]["tool_calls"][0]["id"] = t[0]["tool_calls"][0]["id"]
        self.assertIsNone(actions.replay({"expected": s}, t))
        self.assertIsNone(
            actions.replay(
                {"expected": s}, base + [{"role": "user", "content": "booked"}]
            )
        )

    def test_wire_budget_preserves_ids_and_refuses_unexecuted_final_actions(self):
        s = fixture()
        item = {
            "expected": s,
            "lang": "en",
            "messages": [{"role": "user", "content": "test ticket"}],
        }
        requests = []

        def fake(body):
            requests.append(body)
            n = len(requests)
            return {
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": f"wire:{n}",
                                    "type": "function",
                                    "function": {
                                        "name": "desk_record",
                                        "arguments": json.dumps(
                                            {"ticket": s["ticket"]}
                                        ),
                                    },
                                }
                            ],
                        },
                    }
                ],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1},
            }

        reply, usage = actions.run(item, {"messages": item["messages"]}, fake)
        self.assertEqual(len(requests), actions.MAX_CALLS + 1)
        self.assertNotIn("tools", requests[-1])
        self.assertTrue(reply["web_invalid_final"])
        self.assertTrue(reply["tool_forced"])
        self.assertEqual(usage["prompt_tokens"], 2 * (actions.MAX_CALLS + 1))
        self.assertEqual(len(reply["observation_trace"]), 2 * actions.MAX_CALLS)
        self.assertEqual(
            [x["tool_call_id"] for x in reply["observation_trace"][1::2]],
            [f"wire:{n + 1}" for n in range(actions.MAX_CALLS)],
        )

    def test_nonserial_mutating_batch_and_transport_failure(self):
        s = fixture()
        item = {
            "expected": s,
            "lang": "en",
            "messages": [{"role": "user", "content": "test ticket"}],
        }
        calls = [
            {
                "id": str(i),
                "type": "function",
                "function": {
                    "name": "desk_submit",
                    "arguments": json.dumps(arguments(s, "desk_submit")),
                },
            }
            for i in range(2)
        ]
        response = {
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {"content": None, "tool_calls": calls},
                }
            ]
        }
        r, _ = actions.run(item, {"messages": item["messages"]}, lambda b: response)
        self.assertTrue(r["web_invalid_final"])
        self.assertEqual(r["observation_trace"], [])

        def broken(_):
            raise TimeoutError("hermetic failure")

        with self.assertRaises(TimeoutError):
            actions.run(item, {"messages": item["messages"]}, broken)

    def test_invalid_specs_and_strict_arguments(self):
        s = fixture()
        self.assertTrue(actions.validate_expected(s))
        for k, v in (
            ("day", "2031-02-30"),
            ("mode", "anything"),
            ("units", True),
            ("grants", []),
        ):
            self.assertFalse(actions.validate_expected({**s, k: v}))
        for raw in (
            '{"ticket":"x","ticket":"x"}',
            '{"ticket":true}',
            "[]",
            '{"ticket":"x","extra":2}',
        ):
            self.assertIsNone(actions.arguments("desk_record", raw))


if __name__ == "__main__":
    unittest.main()
