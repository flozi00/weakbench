"""Planning tasks of the quality suite (quality.py imports TASKS). DE and EN, stdlib only.

planning   A small STRIPS problem: the model gets the objects, the initial state, the goal and the action
           schemas, and has to write an action sequence that reaches the goal. The plan is checked by
           *simulating* it — every action's preconditions must hold when it is applied, and the goal must
           hold at the end — so the score is exact and needs no judge, no VAL, no PDDL parser and no
           execution of model-written code. Three domains (blocksworld, gripper, logistics) with a
           deterministic generator; the optimal length comes from a breadth-first search over the same
           transition function, so `len_ratio` says how far a valid plan is from optimal.

This is the suite's planning cell. `musr` measures multi-step reasoning with a multiple-choice answer;
this one measures whether the model can actually produce a *valid* sequence, which is the part that
matters for an agent and the part a multiple-choice item cannot see.

Deterministic: the generator is seeded, the simulator is exact, and the only randomness is the server's.
"""
import random
import re

# --- STRIPS core: a state is a frozenset of ground atoms, an action is a ground (name, pre, add, del) -----


def ground(schema, args):
    """One ground action from a schema and its argument tuple."""
    name, params, pre, add, dele = schema
    bind = dict(zip(params, args))
    sub = lambda atoms: frozenset(a.format(**bind) for a in atoms)  # noqa: E731
    return (f"{name}({','.join(args)})", sub(pre), sub(add), sub(dele))


def applicable(state, action):
    return action[1] <= state


def apply(state, action):
    return (state - action[3]) | action[2]


def bfs(domain, start, goal, limit=200000):
    """Shortest plan (list of ground actions) from `start` to a state satisfying `goal`, or None."""
    if goal <= start:
        return []
    seen, frontier = {start}, [start]
    parent: dict = {start: None}
    while frontier and len(seen) < limit:
        nxt = []
        for s in frontier:
            for a in domain.actions(s):
                if not applicable(s, a):
                    continue
                t = apply(s, a)
                if t in seen:
                    continue
                seen.add(t)
                parent[t] = (s, a)
                if goal <= t:
                    plan, cur = [], t
                    while parent[cur]:
                        cur, act = parent[cur]
                        plan.append(act)
                    return plan[::-1]
                nxt.append(t)
        frontier = nxt
    return None


class Domain:
    """A STRIPS domain: object names, action schemas, and the ground actions of one state."""

    def __init__(self, name, objects, schemas, grounder, obj_text):
        self.name, self.objects, self.schemas, self.grounder = name, objects, schemas, grounder
        self.obj_text = obj_text

    def actions(self, state):
        return self.grounder(self, state)

    def render(self, lang):
        """The domain description the model sees: objects, action schemas, in the item's language."""
        head = {"en": "Domain: {name}\nObjects: {objs}\n\nActions (name(args): preconditions -> add / "
                      "delete):", "de": "Domäne: {name}\nObjekte: {objs}\n\nAktionen (name(args): "
                      "Vorbedingungen -> hinzufügen / entfernen):"}[lang]
        lines = [head.format(name=self.name, objs=self.obj_text)]
        for name, params, pre, add, dele in self.schemas:
            lines.append(f"  {name}({', '.join(params)}): {', '.join(sorted(pre)) or '-'} -> "
                         f"{', '.join(sorted(add)) or '-'} / {', '.join(sorted(dele)) or '-'}")
        return "\n".join(lines)


def _blocks_actions(dom, state):
    out = []
    for x in dom.objects:
        out.append(ground(dom.schemas[0], (x,)))  # pickup
        out.append(ground(dom.schemas[1], (x,)))  # putdown
        for y in dom.objects:
            if x != y:
                out.append(ground(dom.schemas[2], (x, y)))  # stack
                out.append(ground(dom.schemas[3], (x, y)))  # unstack
    return out


def _gripper_actions(dom, state):
    rooms, balls = dom.objects[0], dom.objects[1]
    out = []
    for a in rooms:
        for b in rooms:
            if a != b:
                out.append(ground(dom.schemas[0], (a, b)))  # move
    for ball in balls:
        for room in rooms:
            for g in ("left", "right"):
                out.append(ground(dom.schemas[1], (ball, room, g)))  # pick
                out.append(ground(dom.schemas[2], (ball, room, g)))  # drop
    return out


def _logistics_actions(dom, state):
    locs, pkgs = dom.objects[0], dom.objects[1]
    out = []
    for a in locs:
        for b in locs:
            if a != b:
                out.append(ground(dom.schemas[0], (a, b)))  # drive
    for pkg in pkgs:
        for loc in locs:
            out.append(ground(dom.schemas[1], (pkg, loc)))  # load
            out.append(ground(dom.schemas[2], (pkg, loc)))  # unload
    return out


BLOCKS = Domain("blocksworld", ("a", "b", "c"), (
    ("pickup", ("x",), ("ontable({x})", "clear({x})", "handempty"), ("holding({x})",),
     ("ontable({x})", "clear({x})", "handempty")),
    ("putdown", ("x",), ("holding({x})",), ("ontable({x})", "clear({x})", "handempty"), ("holding({x})",)),
    ("stack", ("x", "y"), ("holding({x})", "clear({y})"), ("on({x},{y})", "clear({x})", "handempty"),
     ("holding({x})", "clear({y})")),
    ("unstack", ("x", "y"), ("on({x},{y})", "clear({x})", "handempty"), ("holding({x})", "clear({y})"),
     ("on({x},{y})", "clear({x})", "handempty")),
), _blocks_actions, "a, b, c")

GRIPPER = Domain("gripper", (("rooma", "roomb"), ("ball1", "ball2", "ball3")), (
    ("move", ("from", "to"), ("at(robot,{from})",), ("at(robot,{to})",), ("at(robot,{from})",)),
    ("pick", ("ball", "room", "g"), ("at(robot,{room})", "at({ball},{room})", "free({g})"),
     ("carry({ball},{g})",), ("at({ball},{room})", "free({g})")),
    ("drop", ("ball", "room", "g"), ("at(robot,{room})", "carry({ball},{g})"),
     ("at({ball},{room})", "free({g})"), ("carry({ball},{g})",)),
), _gripper_actions, "rooms: rooma, roomb; balls: ball1, ball2, ball3")

LOGISTICS = Domain("logistics", (("city1", "city2"), ("pkg1", "pkg2")), (
    ("drive", ("from", "to"), ("at(truck,{from})",), ("at(truck,{to})",), ("at(truck,{from})",)),
    ("load", ("pkg", "loc"), ("at(truck,{loc})", "at({pkg},{loc})"), ("in({pkg},truck)",),
     ("at({pkg},{loc})",)),
    ("unload", ("pkg", "loc"), ("at(truck,{loc})", "in({pkg},truck)"), ("at({pkg},{loc})",),
     ("in({pkg},truck)",)),
), _logistics_actions, "locations: city1, city2; packages: pkg1, pkg2")

DOMAINS = {"blocksworld": BLOCKS, "gripper": GRIPPER, "logistics": LOGISTICS}


# --- problem generation: a random reachable state, then a random goal over it -------------------------


def _blocks_problem(rng):
    blocks = list(BLOCKS.objects)
    rng.shuffle(blocks)
    state, clear = set(), set(blocks)
    for i, b in enumerate(blocks):  # a random stack order, then a random cut into towers
        if i + 1 < len(blocks) and rng.random() < 0.5:
            state.add(f"on({b},{blocks[i + 1]})")
            clear.discard(blocks[i + 1])
        else:
            state.add(f"ontable({b})")
    state |= {f"clear({b})" for b in clear} | {"handempty"}
    goal = set()
    if rng.random() < 0.5:  # a tower of two
        x, y = rng.sample(blocks, 2)
        goal = {f"on({x},{y})"}
    else:  # a block back on the table
        goal = {f"ontable({rng.choice(blocks)})"}
    return frozenset(state), frozenset(goal)


def _gripper_problem(rng):
    rooms, balls = GRIPPER.objects
    state = {f"at(robot,{rng.choice(rooms)})", "free(left)", "free(right)"}
    state |= {f"at({b},{rng.choice(rooms)})" for b in balls}
    goal = {f"at({b},roomb)" for b in balls if rng.random() < 0.7} or {f"at({balls[0]},roomb)"}
    return frozenset(state), frozenset(goal)


def _logistics_problem(rng):
    locs, pkgs = LOGISTICS.objects
    state = {f"at(truck,{rng.choice(locs)})"} | {f"at({p},{rng.choice(locs)})" for p in pkgs}
    goal = {f"at({p},{rng.choice(locs)})" for p in pkgs}
    return frozenset(state), frozenset(goal)


GENERATORS = {"blocksworld": _blocks_problem, "gripper": _gripper_problem, "logistics": _logistics_problem}

ASK = {
    "en": ("{domain}\n\nInitial state:\n{init}\n\nGoal:\n{goal}\n\nWrite a plan that reaches the goal from "
           "the initial state. One action per line, numbered, in the form \"1. pickup(a)\". Reply with the "
           "plan only."),
    "de": ("{domain}\n\nAusgangszustand:\n{init}\n\nZiel:\n{goal}\n\nSchreibe einen Plan, der vom "
           "Ausgangszustand zum Ziel führt. Eine Aktion pro Zeile, nummeriert, in der Form \"1. pickup(a)\". "
           "Antworte nur mit dem Plan."),
}
PLAN_LINE = re.compile(r"^\s*(?:\d+\s*[.)]\s*)?([a-z_]+)\(([^)]*)\)", re.I)


def _render(atoms):
    return "\n".join(f"  {a}" for a in sorted(atoms))


def load_planning(lang, n, seed):
    """`n` problems, cycling the three domains so every cell covers all of them."""
    names = list(DOMAINS)
    items, seen = [], set()
    i = 0
    while len(items) < (n or 300):
        name = names[i % len(names)]
        rng = random.Random(f"{seed}:{name}:{i}")
        i += 1
        start, goal = GENERATORS[name](rng)
        dom = DOMAINS[name]
        plan = bfs(dom, start, goal)
        if plan is None or not plan:  # unsolvable or already solved: not a problem
            continue
        key = (name, start, goal)
        if key in seen:
            continue
        seen.add(key)
        items.append({"id": f"{name}-{len(items):03d}", "lang": lang, "domain": name,
                      "messages": [{"role": "user", "content": ASK[lang].format(
                          domain=dom.render(lang), init=_render(start), goal=_render(goal))}],
                      "start": start, "goal": goal, "optimal": len(plan)})
    return items


def parse_plan(text):
    """The action names of a numbered plan, in order. Ignores prose lines."""
    out = []
    for line in text.splitlines():
        m = PLAN_LINE.match(line)
        if m:
            out.append(f"{m.group(1).lower()}({','.join(a.strip() for a in m.group(2).split(','))})")
    return out


def score_planning(item, reply):
    """Simulate the plan: `correct` = the goal holds at the end, `score` = share of goal atoms reached.

    A plan whose action is not applicable at its step stops there — the rest is not credited, because an
    agent that issues an illegal action has failed at that step whatever it wrote afterwards. `valid` is
    the all-or-nothing version, `len_ratio` compares a valid plan's length with the BFS optimum (>1 means
    the model found a longer route, which is legal but wasteful).
    """
    dom = DOMAINS[item["domain"]]
    by_name = {a[0]: a for a in dom.actions(item["start"])}
    state, steps, valid = item["start"], 0, True
    for name in parse_plan(reply["content"]):
        act = by_name.get(name)
        if act is None or not applicable(state, act):
            valid = False
            break
        state, steps = apply(state, act), steps + 1
    reached = len(item["goal"] & state) / len(item["goal"])
    done = item["goal"] <= state
    # partial credit is measured against the part of the goal that is not already true at the start,
    # otherwise a model that writes nothing scores the share of the goal the initial state satisfies
    need = item["goal"] - item["start"]
    part = len(need & state) / len(need) if need else 1.0
    return {"correct": done and valid, "score": round(part, 4), "pred": f"{steps} steps",
            "gold": item["optimal"],
            "detail": {"valid": float(valid), "steps": steps, "optimal": item["optimal"],
                       "len_ratio": round(steps / item["optimal"], 3) if valid and steps else 0.0,
                       "goal_share": round(reached, 4)}}


def task(langs, load, score, max_tokens, quick, full=None, lang_check=True):
    return {"langs": langs, "load": load, "score": score, "think": False, "max_tokens": max_tokens,
            "counts": {"quick": dict.fromkeys(langs, quick), "full": dict.fromkeys(langs, full)},
            "lang_check": lang_check}


TASKS = {
    # a plan is action names, not prose: no language to check
    "planning": task(("en", "de"), load_planning, score_planning, 1024, 60, full=300, lang_check=False),
}
