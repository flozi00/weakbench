"""tasks_plan (planning) on the real generator, plain asserts. Stdlib only:

    python3 test_tasks_plan.py
"""
import tasks_plan as P

reply = lambda text: {"content": text, "reasoning": "", "tool_calls": [], "finish_reason": "stop"}  # noqa: E731


def test_domains_are_solvable_and_optimal():
    """Every generated problem has a BFS plan, and that plan validates through the simulator."""
    for lang in ("en", "de"):
        items = P.TASKS["planning"]["load"](lang, 60, 0)
        assert len(items) == 60 and len({i["id"] for i in items}) == 60
        assert {i["domain"] for i in items} == set(P.DOMAINS)  # all three domains in one cell
        for it in items:
            plan = P.bfs(P.DOMAINS[it["domain"]], it["start"], it["goal"])
            assert plan and len(plan) == it["optimal"]
            text = "\n".join(f"{k}. {a[0]}" for k, a in enumerate(plan, 1))
            r = P.TASKS["planning"]["score"](it, reply(text))
            assert r["correct"] and r["detail"]["valid"] == 1.0 and r["detail"]["len_ratio"] == 1.0


def test_planning_is_deterministic_and_paired():
    """Same seed, same problems; the DE and EN cells are the same problems behind a different prompt."""
    en = P.TASKS["planning"]["load"]("en", 60, 0)
    de = P.TASKS["planning"]["load"]("de", 60, 0)
    assert [i["id"] for i in en] == [i["id"] for i in de]
    assert [i["start"] for i in en] == [i["start"] for i in de]
    assert [i["goal"] for i in en] == [i["goal"] for i in de]
    assert [i["optimal"] for i in en] == [i["optimal"] for i in de]
    assert en[0]["messages"][0]["content"] != de[0]["messages"][0]["content"]
    assert "Domäne:" in de[0]["messages"][0]["content"] and "Domain:" not in de[0]["messages"][0]["content"]
    assert "Domain:" in en[0]["messages"][0]["content"]
    assert [i["id"] for i in P.TASKS["planning"]["load"]("en", 60, 0)] == [i["id"] for i in en]


def test_plan_parsing():
    """Numbered, bare, indented and prose-wrapped plans all parse; prose lines do not."""
    assert P.parse_plan("1. pickup(a)\n2. stack(a,b)") == ["pickup(a)", "stack(a,b)"]
    assert P.parse_plan("1) pickup(a)\n2) stack(a, b)") == ["pickup(a)", "stack(a,b)"]
    assert P.parse_plan("  pickup(a)\n  putdown(a)") == ["pickup(a)", "putdown(a)"]
    assert P.parse_plan("Here is the plan:\n1. move(rooma,roomb)\nDone.") == ["move(rooma,roomb)"]
    assert P.parse_plan("I cannot solve this.") == []
    assert P.parse_plan("") == []


def test_planning_scoring():
    """A truncated plan, an illegal action and an empty answer all fail; partial credit is honest."""
    it = P.TASKS["planning"]["load"]("en", 60, 0)[0]
    plan = P.bfs(P.DOMAINS[it["domain"]], it["start"], it["goal"])
    assert plan
    full = "\n".join(f"{k}. {a[0]}" for k, a in enumerate(plan, 1))
    assert P.TASKS["planning"]["score"](it, reply(full))["correct"]
    # one step short: still executable, goal not reached, no partial credit for the atoms already true
    short = "\n".join(f"{k}. {a[0]}" for k, a in enumerate(plan[:-1], 1))
    r = P.TASKS["planning"]["score"](it, reply(short))
    assert not r["correct"] and r["detail"]["valid"] == 1.0 and r["score"] == 0.0
    # an illegal action stops the simulation
    r = P.TASKS["planning"]["score"](it, reply("1. stack(a,b)"))
    assert not r["correct"] and r["detail"]["valid"] == 0.0 and r["detail"]["steps"] == 0
    # nothing at all
    r = P.TASKS["planning"]["score"](it, reply("I cannot solve this."))
    assert not r["correct"] and r["score"] == 0.0 and r["detail"]["steps"] == 0
    # the goal reached and then an illegal action: the goal counts, the plan is still invalid
    r = P.TASKS["planning"]["score"](it, reply(full + "\n" + full))
    assert not r["correct"] and r["detail"]["valid"] == 0.0 and r["score"] == 1.0


def test_planning_goal_is_not_already_true():
    """A problem whose goal holds at the start is not a problem, so the generator never emits one."""
    for it in P.TASKS["planning"]["load"]("en", 60, 0):
        assert not it["goal"] <= it["start"]
        assert it["optimal"] >= 1


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
