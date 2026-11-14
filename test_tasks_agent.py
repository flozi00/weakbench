"""tasks_agent (tool_env): a scripted perfect agent reaches the expected state, a sloppy one does not. Plain asserts:

    python3 test_tasks_agent.py
"""
import collections
import json

import tasks_agent as ta

reply = {"content": "done", "reasoning": "", "tool_calls": [], "finish_reason": "stop"}


def solve(it):
    """What a correct agent does, through the same tool runtime the model gets."""
    run, p, w = it["tool_exec"], it["params"], it["world"]
    email = json.loads(run("search_contacts", {"name": p["who"].split()[0]}))[0]["email"]
    if it["kind"] == "underspecified":
        return  # nothing to do; the reply asks (see test_underspecified)
    if it["kind"] == "inbox":
        mail = next(m for m in json.loads(run("read_inbox", {})) if m["subject"] == p["title"])
        ev = json.loads(run("search_events", {"query": p["title"]}))[0]
        run("send_email", {"to": mail["from"], "subject": "Re: " + p["title"], "body": f"{ev['date']} {ev['time']}"})
        return
    if it["kind"] == "meeting":
        busy = {e["time"] for e in json.loads(run("list_events", {"date": p["date"]}))}
        t = next(f"{h:02d}:00" for h in range(9, 17) if f"{h:02d}:00" not in busy)
        run("create_event", {"date": p["date"], "time": t, "duration": 60, "title": p["title"], "attendees": [email]})
        return
    fl = json.loads(run("search_flights", {"origin": p["o"], "destination": p["d"], "date": p["date"]}))
    if it["kind"] == "constrained":
        fl = [f for f in fl if f["price"] <= p["budget"] and f["arrive"] < p["arr"]]
    if it["kind"] == "conditional":
        fl = [f for f in fl if f["price"] <= p["budget"]]
        if not fl:
            run("send_email", {"to": email, "subject": "Reise", "body": "verschoben"})
            return
    best = min(fl, key=lambda f: f["price"])["id"]
    out = run("book_flight", {"flight_id": best})
    if out.startswith("error"):  # the flaky kind: retry once
        out = run("book_flight", {"flight_id": best})
    ref = json.loads(out)["booking_reference"]
    if it["kind"] == "cheapest":
        run("send_email", {"to": email, "subject": f"Booking {ref}", "body": ref})


def test_tool_env():
    for lang in ("en", "de"):
        items = ta.load_tool_env(lang, 200, 0)
        assert len(items) == 200 and min(collections.Counter(i["kind"] for i in items).values()) >= 25
        assert [i["id"] for i in ta.load_tool_env(lang, 80, 0)] == [i["id"] for i in items[:80]]
        for it in items:
            solve(it)
            r = ta.score_tool_env(it, reply if it["kind"] != "underspecified" else
                                  {**reply, "content": "For which date should I book the flight?"})
            assert r["correct"] and r["detail"]["side_effects"] == 0.0, (it["kind"], it["params"], it["state"], r)
    assert collections.Counter(i["want"]["book"] == [] for i in items if i["kind"] == "conditional")[True] >= 10
    fresh = ta.load_tool_env("en", 4, 0)
    assert not ta.score_tool_env(fresh[0], reply)["correct"]  # doing nothing is wrong
    it = fresh[0]
    solve(it)
    it["tool_exec"]("book_flight", {"flight_id": it["world"]["flights"][-1]["id"]})  # one booking too many
    r = ta.score_tool_env(it, reply)
    assert not r["correct"] and r["detail"]["side_effects"] == 1.0


def test_through_quality_tool_loop():
    """The real quality.run_item loop against a mock model that books the cheapest flight and mails the ref."""
    import mock_server
    import quality
    it = ta.load_tool_env("en", 1, 0)[0]
    p = it["params"]

    def model(path, body):
        tool_msgs = [m for m in body["messages"] if m["role"] == "tool"]
        call = lambda n, a: {"content": "", "tool_calls": [{"id": "c", "type": "function",  # noqa: E731
                                                          "function": {"name": n, "arguments": json.dumps(a)}}]}
        if len(tool_msgs) == 0:
            return call("search_flights", {"origin": p["o"], "destination": p["d"], "date": p["date"]})
        if len(tool_msgs) == 1:
            best = min(json.loads(tool_msgs[0]["content"]), key=lambda f: f["price"])
            return call("book_flight", {"flight_id": best["id"]})
        if len(tool_msgs) == 2:
            return call("search_contacts", {"name": p["who"]})
        if len(tool_msgs) == 3:
            ref = json.loads(tool_msgs[1]["content"])["booking_reference"]
            return call("send_email", {"to": json.loads(tool_msgs[2]["content"])[0]["email"], "subject": ref,
                                       "body": f"Ref {ref}"})
        return {"content": "Done."}
    url, stop = mock_server.serve(model)
    try:
        t = quality.TASKS["tool_env"]
        body = {"model": "m", "messages": it["messages"], "max_tokens": 100, "tools": it["tools"]}
        rec = quality.run_item(url, "tool_env", t, it, body, 30)
    finally:
        stop()
    assert rec["error"] is None and rec["correct"], rec


def test_meeting_time_formats():
    it = next(i for i in ta.load_tool_env("en", 8, 0) if i["kind"] == "meeting")
    slot = it["want"]["event"][0]
    email = it["world"]["contacts"][it["world"]["who"]]
    h = int(slot[:2])
    it["tool_exec"]("create_event", {"date": it["params"]["date"], "time": f"{h}:00:00", "duration": "60 min",
                                     "title": "x", "attendees": [email]})
    assert ta.score_tool_env(it, reply)["correct"]


def test_underspecified():
    it = next(i for i in ta.load_tool_env("de", 14, 0) if i["kind"] == "underspecified")
    assert "am " not in it["messages"][1]["content"]
    assert not ta.score_tool_env(it, {**reply, "content": "Erledigt."})["correct"]
    it["tool_exec"]("book_flight", {"flight_id": it["world"]["flights"][0]["id"]})
    assert not ta.score_tool_env(it, {**reply, "content": "An welchem Datum?"})["correct"]


if __name__ == "__main__":
    test_underspecified()
    test_meeting_time_formats()
    test_tool_env()
    test_through_quality_tool_loop()
    print("ok")
