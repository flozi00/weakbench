"""Independent solvers for the self-built-state / search tasks in tasks_novel: each reads the PROMPT TEXT (not the
generator's internals), solves it another way and must reach the stored gold. Run: python test_search_tasks.py"""
import itertools
import re
from collections import deque

import tasks_novel as tn


def prompts(load, n=96):
    return load("en", n, 0)


def test_hanoi_state():  # iterative algorithm (smallest disk cycles), not recursion
    for it in prompts(tn.load_hanoi_state):
        q = it["messages"][0]["content"]
        d, m = int(re.search(r"(\d) disks", q)[1]), int(re.search(r"After move (\d+)", q)[1])
        peg = re.search(r"on peg (\w)\?", q)[1]
        pegs, order, pos = {"A": list(range(d, 0, -1)), "B": [], "C": []}, ("ABC" if d % 2 == 0 else "ACB"), 0
        for k in range(1, m + 1):
            if k % 2:
                src, dst, pos = order[pos], order[(pos + 1) % 3], (pos + 1) % 3
            else:
                a, b = [p for p in "ABC" if p != order[pos]]
                src, dst = (a, b) if pegs[a] and (not pegs[b] or pegs[a][-1] < pegs[b][-1]) else (b, a)
            pegs[dst].append(pegs[src].pop())
        assert pegs[peg] == it["gold"], q


def test_lights_out():  # brute force over press sets, own neighbour table
    nb = {k: [k] + [j for j in range(9) if abs(j // 3 - k // 3) + abs(j % 3 - k % 3) == 1] for k in range(9)}
    for it in prompts(tn.load_lights_out):
        st = [int(x) for row in re.findall(r"^([01] [01] [01])$", it["messages"][0]["content"], re.M) for x in row.split()]
        best = min(m for m in range(10) for c in itertools.combinations(range(9), m)
                   if not any((st[j] + sum(j in nb[k] for k in c)) % 2 for j in range(9)))
        assert best == it["gold"]


def test_knight_moves():  # frontier expansion instead of BFS with a queue
    sq = lambda s: ("abcdefgh".index(s[0]), int(s[1]) - 1)  # noqa: E731
    for it in prompts(tn.load_knight_moves):
        q = it["messages"][0]["content"]
        a, b = sq(re.search(r"stands on (\w\d)", q)[1]), sq(re.search(r"reach (\w\d)", q)[1])
        bl = {sq(x) for x in re.search(r"squares? ([a-h\d, ]+) (?:are|is) blocked", q)[1].split(", ")}
        front, d = {a}, 0
        while b not in front:
            front = {(x + dx, y + dy) for x, y in front for dx, dy in itertools.permutations((1, 2, -1, -2), 2)
                     if abs(dx) != abs(dy) and 0 <= x + dx < 8 and 0 <= y + dy < 8 and (x + dx, y + dy) not in bl}
            d += 1
        assert d == it["gold"], q


def test_river_crossing():  # states as sets of names incl. the keeper
    for it in prompts(tn.load_river_crossing):
        q = it["messages"][0]["content"]
        animals = list(dict.fromkeys(re.findall(r"the (fox|hen|dog|cat|goat)", q.split("\n")[0])))
        cap = 2 if "up to two" in q else 1
        bad = [tuple(re.findall(r"(?i)the (fox|hen|dog|cat|goat)", ln)) for ln in q.split("\n") if ln.startswith("- ")]
        safe = lambda bank: "K" in bank or not any(x in bank and y in bank for x, y in bad)  # noqa: E731
        full = frozenset(animals + ["K"])
        seen, dq = {full: 0}, deque([full])
        while dq:
            left = dq.popleft()
            if not left:
                break
            side = left if "K" in left else full - left
            for m in range(cap + 1):
                for load in itertools.combinations(sorted(side - {"K"}), m):
                    mv = set(load) | {"K"}
                    nl = left - mv if "K" in left else left | mv
                    if safe(nl) and safe(full - nl) and nl not in seen:
                        seen[nl] = seen[left] + 1
                        dq.append(nl)
        assert seen[frozenset()] == it["gold"], q


def test_dice_roll():  # cube as direction vectors
    vec = {"north": (0, 1, 0), "south": (0, -1, 0), "east": (1, 0, 0), "west": (-1, 0, 0)}
    for it in prompts(tn.load_dice_roll):
        q = it["messages"][0]["content"]
        top, north, east = map(int, re.search(r"top face shows (\d).*north\s+shows (\d).*east shows (\d)", q, re.S).groups())
        f = {(0, 0, 1): top, (0, 0, -1): 7 - top, (0, 1, 0): north, (0, -1, 0): 7 - north, (1, 0, 0): east, (-1, 0, 0): 7 - east}
        for d in re.search(r"directions: ([a-z, ]+)\.", q)[1].split(", "):
            v = vec[d]
            neg = tuple(-x for x in v)
            g = dict(f)
            g[v], g[(0, 0, -1)], g[neg], g[(0, 0, 1)] = f[(0, 0, 1)], f[v], f[(0, 0, -1)], f[neg]
            f = g
        assert f[(0, 0, 1)] == it["gold"], q


def test_sliding_puzzle():
    pos = tn.SP_POS["en"]
    for it in prompts(tn.load_sliding_puzzle):
        q = it["messages"][0]["content"]
        b = [0 if x == "_" else int(x) for row in re.findall(r"^([\d_] [\d_] [\d_])$", q, re.M) for x in row.split()]
        for d in re.search(r"old gap\): ([a-z, ]+)\.", q)[1].split(", "):
            g = b.index(0)
            p = g + {"up": -3, "down": 3, "left": -1, "right": 1}[d]
            assert 0 <= p < 9 and (d in ("up", "down") or p // 3 == g // 3)
            b[g], b[p] = b[p], 0
        assert b[pos.index(re.search(r"at the (.+) at the end", q)[1])] == it["gold"]


def test_cube_net_folder():  # the folder accepts exactly the 11 cube nets among the 35 free hexominoes
    def norm(c):
        out = []
        for rot in range(8):
            cc = c
            for _ in range(rot % 4):
                cc = [(y, -x) for x, y in cc]
            if rot >= 4:
                cc = [(-x, y) for x, y in cc]
            mx, my = min(x for x, _ in cc), min(y for _, y in cc)
            out.append(tuple(sorted((x - mx, y - my) for x, y in cc)))
        return min(out)
    shapes = {((0, 0),)}
    for _ in range(5):
        shapes = {norm(list(sh) + [(x + dx, y + dy)]) for sh in shapes for x, y in sh
                  for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)) if (x + dx, y + dy) not in sh}
    assert len(shapes) == 35 and sum(tn.cn_fold(list(sh)) is not None for sh in shapes) == 11


def test_toy_machine():  # interpreter over the listing text
    for it in prompts(tn.load_toy_machine):
        q = it["messages"][0]["content"]
        lines, reg, pc = dict(re.findall(r"^(\d+): (.+)$", q, re.M)), {"A": 0, "B": 0, "C": 0}, 1
        while True:
            w = lines[str(pc)].split()
            if w[0] == "stop":
                break
            if w[0] == "hop" and reg[w[3]] > 0:
                pc = int(w[1])
                continue
            if w[0] == "put":
                reg[w[1]] = int(w[2])
            elif w[0] == "grow":
                reg[w[1]] += reg[w[2]]
            elif w[0] == "shrink":
                reg[w[1]] -= int(w[2])
            elif w[0] == "twice":
                reg[w[1]] *= 2
            pc += 1
        assert reg[re.search(r"register (\w) when", q)[1]] == it["gold"]



def test_paper_punch():  # move every original square through the folds as a point; holes = squares landing on the punch
    for it in prompts(tn.load_paper_punch):
        q = it["messages"][0]["content"]
        folds = re.findall(r"Fold the (left|right|top|bottom) half", q)
        at = re.search(r"on top of square ([A-D][1-4])", q)[1]
        punch = ("ABCD".index(at[0]) + 0.5, int(at[1]) - 0.5)
        holes = []
        for c in range(4):
            for w in range(4):
                x, y, x0, x1, y0, y1 = c + 0.5, w + 0.5, 0.0, 4.0, 0.0, 4.0
                for f in folds:
                    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
                    if f == "left":
                        x, x0 = (2 * mx - x if x < mx else x), mx
                    elif f == "right":
                        x, x1 = (2 * mx - x if x > mx else x), mx
                    elif f == "top":
                        y, y0 = (2 * my - y if y < my else y), my
                    else:
                        y, y1 = (2 * my - y if y > my else y), my
                if (x, y) == punch:
                    holes.append("ABCD"[c] + str(w + 1))
        assert sorted(holes) == it["gold"], q


def test_view_side():  # rotate the map until the visitor's side is at the bottom, then read it like the map
    pos = {v: k for k, v in tn.VS_POS["en"].items()}
    turns = {"south": 0, "east": 1, "north": 2, "west": 3}  # clockwise quarter turns that bring that side to the bottom
    for it in prompts(tn.load_view_side):
        q = it["messages"][0]["content"]
        grid = [ln.split() for ln in q.split("\n\n")[0].split("\n")[1:]]
        side, where = re.search(r"on the (\w+) side .* Which stall is (.+) from", q, re.S).groups()
        for _ in range(turns[side]):
            grid = [list(row) for row in zip(*grid[::-1])]  # rotate clockwise
        depth, lr = pos[where]
        assert grid[2 - depth][lr] == it["gold"], q


def test_nonogram():  # brute force over all 2^25 grids is too slow: column-wise candidates instead of row-wise
    def runs(cells):
        return [len(x) for x in "".join("#" if c else "." for c in cells).split(".") if x]
    for it in prompts(tn.load_nonogram, 24):
        q = it["messages"][0]["content"]
        rows = [[int(x) for x in v.split()] if v != "0" else [] for v in re.findall(r"^Row \d: ([\d ]+)$", q, re.M)]
        cols = [[int(x) for x in v.split()] if v != "0" else [] for v in re.findall(r"^Column \d: ([\d ]+)$", q, re.M)]
        cand = [[p for p in itertools.product((0, 1), repeat=5) if runs(p) == c] for c in cols]
        sols = [g for g in itertools.product(*cand) if all(runs([col[j] for col in g]) == rows[j] for j in range(5))]
        assert len(sols) == 1, q
        ask = int(re.search(r"What is row (\d)", q)[1]) - 1
        assert "".join("#" if col[ask] else "-" for col in sols[0]) == it["gold"], q


def test_word_ladder():  # layered expansion with set ops over the listed words
    for it in prompts(tn.load_word_ladder):
        q = it["messages"][0]["content"]
        words = set(re.search(r"Word list: ([a-z, ]+)\.", q)[1].split(", "))
        a, b = re.search(r'Go from "(\w+)" to "(\w+)"', q).groups()
        layer, seen, d = {a}, {a}, 0
        while b not in layer:
            layer = {v for v in words - seen for w in layer if sum(x != y for x, y in zip(v, w)) == 1}
            seen |= layer
            d += 1
            assert layer, q
        assert d == it["gold"], q


def test_diff_apply():  # apply the printed hunks by their headers, bottom-up so earlier offsets do not move
    for it in prompts(tn.load_diff_apply):
        q = it["messages"][0]["content"]
        blocks = re.findall(r"```\n(.*?)\n```", q, re.S)
        lines, patch = blocks[0].split("\n"), blocks[1].split("\n")
        hunks, cur = [], None
        for ln in patch:
            m = re.match(r"@@ -(\d+)(?:,(\d+))? \+\d+(?:,\d+)? @@", ln)
            if m:
                cur = [int(m[1]), int(m[2] or 1), []]
                hunks.append(cur)
            else:
                cur[2].append(ln)
        for start, length, body in reversed(hunks):
            old = [ln[1:] for ln in body if ln[0] in " -"]
            new = [ln[1:] for ln in body if ln[0] in " +"]
            assert lines[start - 1:start - 1 + length] == old, q
            lines[start - 1:start - 1 + length] = new
        k = int(re.search(r"What is line (\d+)", q)[1])
        assert lines[k - 1] == it["gold"], q


def test_puzzle8_min():  # iterative deepening DFS instead of BFS
    goal = "12345678_"
    for it in prompts(tn.load_puzzle8_min, 48):
        q = it["messages"][0]["content"]
        st = "".join(re.findall(r"^([\d_]) ([\d_]) ([\d_])$", q, re.M)[0] + re.findall(r"^([\d_]) ([\d_]) ([\d_])$", q, re.M)[1]
                     + re.findall(r"^([\d_]) ([\d_]) ([\d_])$", q, re.M)[2])

        def dfs(s, depth, prev):
            if s == goal:
                return True
            if depth == 0:
                return False
            g = s.index("_")
            for p in (g - 3, g + 3, g - 1, g + 1):
                if 0 <= p < 9 and (abs(p - g) == 3 or p // 3 == g // 3) and p != prev:
                    t = list(s)
                    t[g], t[p] = t[p], "_"
                    if dfs("".join(t), depth - 1, g):
                        return True
            return False
        assert next(d for d in range(15) if dfs(st, d, -1)) == it["gold"], q


def test_key_door_maze():  # two plain BFS legs: S -> K (door shut), then K -> G (door open)
    def bfs(grid, a, b, door_ok):
        front, seen, d = {a}, {a}, 0
        while front and b not in front:
            front = {(y + dy, x + dx) for y, x in front for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1))
                     if 0 <= y + dy < len(grid) and 0 <= x + dx < len(grid[0]) and grid[y + dy][x + dx] != "#"
                     and (door_ok or grid[y + dy][x + dx] != "D")} - seen
            seen |= front
            d += 1
        return d if b in front else None
    for it in prompts(tn.load_key_door_maze):
        q = it["messages"][0]["content"]
        grid = [ln.split() for ln in re.findall(r"^((?:[SGK#.D] )+[SGK#.D])$", q, re.M)]
        find = lambda c: next((y, x) for y, row in enumerate(grid) for x, v in enumerate(row) if v == c)  # noqa: E731
        s_, g, k = find("S"), find("G"), find("K")
        direct = bfs(grid, s_, g, False)
        via = (bfs(grid, s_, k, False) or 10 ** 6) + (bfs(grid, k, g, True) or 10 ** 6)
        assert min(x for x in (direct, via) if x) == it["gold"], q


def test_cable_network():  # Prim's algorithm instead of Kruskal
    for it in prompts(tn.load_cable_network):
        q = it["messages"][0]["content"]
        edges = [(a, b, int(c)) for a, b, c in re.findall(r"^- (\w+) – (\w+): (\d+)$", q, re.M)]
        nodes = {x for a, b, _ in edges for x in (a, b)}
        inside, total = {next(iter(sorted(nodes)))}, 0
        while inside != nodes:
            c, v = min((c, b if a in inside else a) for a, b, c in edges if (a in inside) != (b in inside))
            inside.add(v)
            total += c
        assert total == it["gold"], q


def test_job_assign():  # bitmask dynamic programming over the printed table
    for it in prompts(tn.load_job_assign):
        q = it["messages"][0]["content"]
        cost = [[int(x) for x in row.split(" | ")] for row in re.findall(r"^\| [A-Z][a-z]+ \| ([\d | ]+?) \|$", q, re.M)]
        k = len(cost)
        best = {0: 0}
        for p in range(k):
            nxt = {}
            for m, v in best.items():
                for j in range(k):
                    if not m >> j & 1:
                        nxt[m | 1 << j] = min(nxt.get(m | 1 << j, 10 ** 9), v + cost[p][j])
            best = nxt
        assert best[(1 << k) - 1] == it["gold"], q


def test_seat_count():  # rules re-read from the English text, positions checked as a dict
    for it in prompts(tn.load_seat_count):
        q = it["messages"][0]["content"]
        ppl = re.match(r"(.+?) sit on a bench", q)[1].split(", ")
        rules = re.findall(r"^- (.+)$", q, re.M)
        count = 0
        for o in itertools.permutations(ppl):
            pos = {p: j for j, p in enumerate(o)}
            ok = True
            for ru in rules:
                w = ru.split()
                a = w[0]
                if "not sit directly" in ru:
                    ok &= abs(pos[a] - pos[w[-1].rstrip(".")]) != 1
                elif "directly next" in ru:
                    ok &= abs(pos[a] - pos[w[-1].rstrip(".")]) == 1
                elif "ends" in ru:
                    ok &= pos[a] in (0, len(ppl) - 1)
                else:
                    ok &= pos[a] < pos[w[-1].rstrip(".")]
            count += ok
        assert count == it["gold"], q


def test_piece_fit():  # every 2x2 window: with f free squares it holds C(f, 3) L-pieces (each 3 of 4 squares is one L)
    for it in prompts(tn.load_piece_fit):
        q = it["messages"][0]["content"]
        grid = [ln.split() for ln in re.findall(r"^((?:[#.] )+[#.])$", q, re.M)]
        h, w = len(grid), len(grid[0])
        total = 0
        for y in range(h - 1):
            for x in range(w - 1):
                f = sum(grid[y + a][x + b] == "." for a in (0, 1) for b in (0, 1))
                total += {4: 4, 3: 1}.get(f, 0)
        assert total == it["gold"], q


def test_painted_cube():  # count by the number of painted faces each unit cube touches, via coordinates on the boundary
    names = ["top", "bottom", "front", "back", "left", "right"]
    for it in prompts(tn.load_painted_cube):
        q = it["messages"][0]["content"]
        n = int(re.search(r"cube of (\d)x", q)[1])
        painted = re.search(r"Only these faces are painted: ([a-z, ]+) \(", q)[1].split(", ")
        k = int(re.search(r"exactly (\d) painted", q)[1])
        planes = {"top": (2, n - 1), "bottom": (2, 0), "front": (1, 0), "back": (1, n - 1), "left": (0, 0), "right": (0, n - 1)}
        cnt = sum(sum(c[planes[f][0]] == planes[f][1] for f in painted) == k for c in itertools.product(range(n), repeat=3))
        assert cnt == it["gold"] and set(painted) <= set(names), q


def test_cube_turns():  # face-name cycles (where each face's letter goes), read from the printed step text
    cycles = {"forward": ["top", "front", "bottom", "back"], "backward": ["top", "back", "bottom", "front"],
              "to the left": ["top", "left", "bottom", "right"], "to the right": ["top", "right", "bottom", "left"],
              "quarter to the left": ["front", "left", "back", "right"], "quarter to the right": ["front", "right", "back", "left"]}
    for it in prompts(tn.load_cube_turns):
        q = it["messages"][0]["content"]
        face = dict(re.findall(r"(top|bottom|front|back|left|right): ([A-Z])", q.split("\n")[0]))
        for step in re.findall(r"^\d+\. (.+)$", q, re.M):
            key = next(k for k in ("quarter to the left", "quarter to the right", "counter-clockwise", "clockwise", "forward",
                                   "backward", "to the left", "to the right") if k in step)
            key = {"counter-clockwise": "quarter to the left", "clockwise": "quarter to the right"}.get(key, key)
            c = cycles[key]
            face = {**face, **{c[(j + 1) % 4]: face[c[j]] for j in range(4)}}
        ask = re.search(r"on the (\w+) face at the end", q)[1]
        assert face[ask] == it["gold"], q


def test_toy_machine_inverse():  # substitute 1-9 into the listing and run the text interpreter: exactly the gold fits
    for it in prompts(tn.load_toy_machine_inverse):
        q = it["messages"][0]["content"]
        target = int(re.search(r"holds (-?\d+)\.", q)[1])
        ask = re.search(r"register (\w) holds", q)[1]
        fits = []
        for v in range(1, 51):
            lines = {k: ln.replace("?", str(v)) for k, ln in re.findall(r"^(\d+): (.+)$", q, re.M)}
            reg, pc = {"A": 0, "B": 0, "C": 0}, 1
            while True:
                w = lines[str(pc)].split()
                if w[0] == "stop":
                    break
                if w[0] == "hop" and reg[w[3]] > 0:
                    pc = int(w[1])
                    continue
                if w[0] == "put":
                    reg[w[1]] = int(w[2])
                elif w[0] == "grow":
                    reg[w[1]] += reg[w[2]]
                elif w[0] == "shrink":
                    reg[w[1]] -= int(w[2])
                elif w[0] == "twice":
                    reg[w[1]] *= 2
                pc += 1
            if reg[ask] == target:
                fits.append(v)
        assert fits == [it["gold"]], (q, fits)


def test_box_push():  # iterative deepening over pushes: from each state, every box position reachable by walking
    for it in prompts(tn.load_box_push, 48):
        q = it["messages"][0]["content"]
        grid = [ln.split() for ln in re.findall(r"^((?:[#.PBT] )+[#.PBT])$", q, re.M)]
        h, w = len(grid), len(grid[0])
        find = lambda c: next((y, x) for y, row in enumerate(grid) for x, v in enumerate(row) if v == c)  # noqa: E731
        P, B, T = find("P"), find("B"), find("T")
        free = lambda y, x: 0 <= y < h and 0 <= x < w and grid[y][x] != "#"  # noqa: E731

        def walk(p, box):  # cells the worker reaches without pushing
            seen, todo = {p}, [p]
            while todo:
                y, x = todo.pop()
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    c = (y + dy, x + dx)
                    if free(*c) and c != box and c not in seen:
                        seen.add(c)
                        todo.append(c)
            return seen
        layer, seen, d = {(B, P)}, {(B, P)}, 0
        while not any(b == T for b, _ in layer):
            nxt = set()
            for box, p in layer:
                reach = walk(p, box)
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    stand, to = (box[0] - dy, box[1] - dx), (box[0] + dy, box[1] + dx)
                    if stand in reach and free(*to):
                        st = (to, box)
                        if st not in seen:
                            seen.add(st)
                            nxt.add(st)
            layer, d = nxt, d + 1
            assert layer, q
        assert d == it["gold"], q


def test_bridge_torch():  # the proven greedy recursion (Rote 2002): send the two slowest by the cheaper of two patterns
    def best(t):
        t = sorted(t)
        if len(t) <= 2:
            return t[-1]
        if len(t) == 3:
            return sum(t)
        a, b, y, z = t[0], t[1], t[-2], t[-1]
        return min(a + 2 * b + z, 2 * a + y + z) + best(t[:-2])
    for it in prompts(tn.load_bridge_torch):
        q = it["messages"][0]["content"]
        times = [int(x) for x in re.findall(r"\b[A-Z][a-z]+ (\d+) min", q)]
        assert best(times) == it["gold"], q


def test_seesaw_balance():  # moments from the printed seats; the answer seat is the only free one that zeroes them
    for it in prompts(tn.load_seesaw_balance):
        q = it["messages"][0]["content"]
        kids = [(int(w), float(d) * (-1 if side == "left" else 1))
                for w, d, side in re.findall(r"\((\d+) kg\): ([\d.]+) m (left|right)", q)]
        new_w = int(re.search(r"\((\d+) kg\) sits down", q)[1])
        taken = {p for _, p in kids}
        fits = [x / 2 for x in range(-6, 7) if x and x / 2 not in taken
                and abs(sum(w * p for w, p in kids) + new_w * x / 2) < 1e-9]
        assert fits == [it["x"]], (q, fits)


def test_sun_relative():  # bearings in degrees: sun bearing minus facing bearing -> side
    for it in prompts(tn.load_sun_relative):
        q = it["messages"][0]["content"]
        bearing = {"north": 0, "east": 90, "south": 180, "west": 270}
        south = "southern hemisphere" in q
        sun = 90 if "sunrise" in q else 270 if "sunset" in q else (0 if south else 180)
        face = bearing[re.search(r"facing (\w+)\.", q)[1]]
        for turn in re.findall(r"turns (right|left|around)", q):
            face = (face + {"right": 90, "left": -90, "around": 180}[turn]) % 360
        side = {0: "in front of her", 90: "to her right", 180: "behind her", 270: "to her left"}[(sun - face) % 360]
        assert side == it["gold"], q


def test_strip_fold():  # each square as (x, height): a fold mirrors x and puts the moving half, flipped, on top
    for it in prompts(tn.load_strip_fold):
        q = it["messages"][0]["content"]
        labels = re.search(r"labelled ((?:[A-Z] ){7}[A-Z]) from", q)[1].split()
        sq = {c: [j + 0.5, 0] for j, c in enumerate(labels)}
        lo, hi = 0.0, 8.0
        for side in re.findall(r"Fold the (left|right) half", q):
            mid = (lo + hi) / 2
            moving = [c for c, (x, _) in sq.items() if (x < mid) == (side == "left")]
            top = max(h for c, (x, h) in sq.items() if c not in moving)
            mtop = max(sq[c][1] for c in moving)
            for c in moving:
                sq[c] = [2 * mid - sq[c][0], top + 1 + (mtop - sq[c][1])]
            lo, hi = (mid, hi) if side == "left" else (lo, mid)
        m = re.search(r"bottom square is ([A-Z])", q)
        x = sq[m[1]][0] if m else (lo + hi) / 2
        pile = sorted((c for c in sq if sq[c][0] == x), key=lambda c: -sq[c][1])
        assert "".join(pile) == it["gold"], (q, pile)


def test_unsat_detect():  # rules re-read from the text; brute force decides possible vs impossible
    for it in prompts(tn.load_unsat_detect):
        q = it["messages"][0]["content"]
        ppl = re.match(r"(.+?) sit on a bench", q)[1].split(", ")
        rules = re.findall(r"^- (.+)$", q, re.M)

        def holds(o, ru):
            pos = {p: j for j, p in enumerate(o)}
            w = ru.rstrip(".").split()
            a, b = w[0], w[-1]
            if "middle seat" in ru:
                return pos[a] == 2
            if "ends" in ru:
                return pos[a] in (0, 4)
            if "not sit directly" in ru:
                return abs(pos[a] - pos[b]) != 1
            if "directly next" in ru:
                return abs(pos[a] - pos[b]) == 1
            return pos[a] < pos[b]
        possible = any(all(holds(o, ru) for ru in rules) for o in itertools.permutations(ppl))
        assert possible == it["sat"], q
        if possible:
            assert all(holds(it["gold"].split(", "), ru) for ru in rules)


def test_flashback_order():  # offsets re-read from the English text and chained in telling order
    for it in prompts(tn.load_flashback_order):
        q = it["messages"][0]["content"]
        day = {"A": 0}
        for letter, n_, rel, ref in re.findall(r"\(([B-E])\) (\d+) days? (before|after) event ([A-E])", q):
            day[letter] = day[ref] + int(n_) * (-1 if rel == "before" else 1)
        assert "".join(sorted(day, key=day.get)) == it["gold"], q


def test_list_diff():  # set difference of the printed lists
    for it in prompts(tn.load_list_diff):
        q = it["messages"][0]["content"]
        a = set(re.search(r"List A:\n(.+)", q)[1].split(", "))
        b = set(re.search(r"List B:\n(.+)", q)[1].split(", "))
        assert ", ".join(sorted(a - b)) == it["gold"], q


def test_verify_solution():  # each printed rule checked against the printed order; the broken one is the gold
    for it in prompts(tn.load_verify_solution):
        q = it["messages"][0]["content"]
        order = re.search(r"Proposed order, left to right: (.+)\.\n", q)[1].split(", ")
        pos = {p: j for j, p in enumerate(order)}
        broken = []
        for k, ru in re.findall(r"^(\d)\. (.+)$", q, re.M):
            w = ru.rstrip(".").split()
            a, b = w[0], w[-1]
            ok = (pos[a] == 2 if "middle seat" in ru else pos[a] in (0, 4) if "ends" in ru else
                  abs(pos[a] - pos[b]) != 1 if "not sit directly" in ru else abs(pos[a] - pos[b]) == 1 if "directly next" in ru
                  else pos[a] < pos[b])
            if not ok:
                broken.append(int(k))
        assert broken == ([] if it["broken"] is None else [it["broken"]]), q

if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("ok")
