"""Score a nonadaptive test suite for a finite, pure sequence transformation."""

from functools import lru_cache
from itertools import groupby, product
import re

OPS = (
    "rev",
    "sort",
    "uniq",
    "squeeze",
    "nonzero",
    "head2",
    "drop1",
    "rot",
    "bump",
    "echo",
    "pad0",
    "ident",
)
GUARDS = (
    "any",
    "empty",
    "nonempty",
    "first0",
    "first1",
    "last0",
    "last1",
    "repeat",
    "distinct",
    "pal",
    "sorted",
    "len1",
    "len2",
    "len3",
    "len4",
)
DETAIL_KEYS = (
    "valid_json",
    "probe_schema",
    "probe_reference_outputs",
    "probe_distinguished",
    "probe_equivalent_variants",
    "probe_minimum_count",
    "probe_minimum_total_length",
)


def program(values, valid, maximum):
    return (
        isinstance(values, list)
        and 1 <= len(values) <= maximum
        and all(isinstance(x, str) and x in valid for x in values)
    )


def validate_expected(spec):
    if not isinstance(spec, dict) or set(spec) != {
        "alphabet",
        "maxlen",
        "base",
        "variants",
    }:
        return False
    if (
        type(spec["alphabet"]) is not int
        or spec["alphabet"] not in (2, 3)
        or type(spec["maxlen"]) is not int
        or not 1 <= spec["maxlen"] <= 4
    ):
        return False
    if (
        not program(spec["base"], OPS, 6)
        or not isinstance(spec["variants"], list)
        or not 1 <= len(spec["variants"]) <= 9
    ):
        return False
    names = set()
    for item in spec["variants"]:
        if not isinstance(item, list) or len(item) != 3:
            return False
        name, condition, body = item
        if (
            not isinstance(name, str)
            or not re.fullmatch(r"V[0-9]{2}", name)
            or name in names
        ):
            return False
        if not program(condition, GUARDS, 2) or not program(body, OPS, 6):
            return False
        names.add(name)
    return True


def run(word, ops, alphabet):
    for op in ops:
        if op == "rev":
            word = word[::-1]
        elif op == "sort":
            word = tuple(sorted(word))
        elif op == "uniq":
            word = tuple(dict.fromkeys(word))
        elif op == "squeeze":
            word = tuple(k for k, _ in groupby(word))
        elif op == "nonzero":
            word = tuple(v for v in word if v)
        elif op == "head2":
            word = word[:2]
        elif op == "drop1":
            word = word[1:]
        elif op == "rot":
            word = word[1:] + word[:1]
        elif op == "bump":
            word = tuple((v + 1) % alphabet for v in word)
        elif op == "echo":
            word += word[-1:]
        elif op == "pad0":
            word = word or (0,)
        elif op != "ident":
            raise ValueError("unrecognized operation")
    return word


def active(guard, word):
    if guard == "any":
        return True
    if guard == "empty":
        return not word
    if guard == "nonempty":
        return bool(word)
    if guard.startswith("first"):
        return word[:1] == (int(guard[-1]),)
    if guard.startswith("last"):
        return word[-1:] == (int(guard[-1]),)
    if guard == "repeat":
        return len(set(word)) < len(word)
    if guard == "distinct":
        return len(set(word)) == len(word)
    if guard == "pal":
        return word == word[::-1]
    if guard == "sorted":
        return all(a <= b for a, b in zip(word, word[1:]))
    if guard.startswith("len"):
        return len(word) == int(guard[-1])
    raise ValueError("unrecognized guard")


def fingerprint(spec):
    return (
        spec["alphabet"],
        spec["maxlen"],
        tuple(spec["base"]),
        tuple((name, tuple(g), tuple(ops)) for name, g, ops in spec["variants"]),
    )


@lru_cache(maxsize=96)
def truth_table(alphabet, maxlen, base, variants):
    by_input = {}
    shortest = {}
    goal = 0
    for length in range(maxlen + 1):
        for word in product(range(alphabet), repeat=length):
            answer = run(word, base, alphabet)
            mask = 0
            for i, (_, guards, ops) in enumerate(variants):
                if (
                    all(active(g, word) for g in guards)
                    and run(word, ops, alphabet) != answer
                ):
                    mask |= 1 << i
            by_input[word] = (answer, mask)
            if mask:
                shortest[mask] = min(shortest.get(mask, length), length)
            goal |= mask
    # One representative of each coverage mask suffices, and its shortest word
    # gives the minimum length among test suites with the same cardinality.
    dp = {0: (0, 0)}
    for coverage, length in shortest.items():
        for old, (count, chars) in list(dp.items()):
            new = old | coverage
            price = (count + 1, chars + length)
            if new not in dp or price < dp[new]:
                dp[new] = price
    equivalent = frozenset(
        name for i, (name, *_) in enumerate(variants) if not goal & (1 << i)
    )
    return by_input, goal, dp[goal], equivalent


def symbols(value, n, limit):
    return (
        isinstance(value, list)
        and len(value) <= limit
        and all(type(x) is int and 0 <= x < n for x in value)
    )


def score(item, reply):
    from hidden import strict_json

    checks = dict.fromkeys(DETAIL_KEYS, 0.0)
    try:
        raw = reply.get("content")
        if not isinstance(raw, str) or len(raw) > 16384:
            raise ValueError
        ans = strict_json(raw)
        checks["valid_json"] = 1.0
    except (ValueError, TypeError, RecursionError):
        return {"correct": False, "score": 0.0, "detail": checks}
    spec = item["expected"]
    if not isinstance(ans, dict) or set(ans) != {"tests", "equivalent"}:
        return {"correct": False, "score": 0.0, "detail": checks}
    tests, claimed = ans["tests"], ans["equivalent"]
    n = spec["alphabet"]
    lim = spec["maxlen"]
    names = {v[0] for v in spec["variants"]}
    if (
        not isinstance(claimed, list)
        or any(not isinstance(v, str) or v not in names for v in claimed)
        or len(set(claimed)) != len(claimed)
    ):
        return {"correct": False, "score": 0.0, "detail": checks}
    if not isinstance(tests, list) or len(tests) > sum(n**k for k in range(lim + 1)):
        return {"correct": False, "score": 0.0, "detail": checks}
    table, goal, opt, eq = truth_table(*fingerprint(spec))
    seen = set()
    covered = 0
    good_outputs = total_len = 0
    for entry in tests:
        if (
            not isinstance(entry, dict)
            or set(entry) != {"input", "expect"}
            or not symbols(entry["input"], n, lim)
            or not symbols(entry["expect"], n, lim + len(spec["base"]))
        ):
            return {"correct": False, "score": 0.0, "detail": checks}
        word = tuple(entry["input"])
        if word in seen:
            return {"correct": False, "score": 0.0, "detail": checks}
        seen.add(word)
        total_len += len(word)
        expected, mask = table[word]
        if expected == tuple(entry["expect"]):
            good_outputs += 1
            covered |= mask
    checks["probe_schema"] = 1.0
    checks["probe_reference_outputs"] = (
        good_outputs / len(tests) if tests else float(not goal)
    )
    checks["probe_equivalent_variants"] = float(set(claimed) == eq)
    checks["probe_distinguished"] = (
        covered.bit_count() / goal.bit_count() if goal else 1.0
    )
    checks["probe_minimum_count"] = float(covered == goal and len(tests) == opt[0])
    checks["probe_minimum_total_length"] = float(
        covered == goal and (len(tests), total_len) == opt
    )
    correct = all(v == 1 for v in checks.values())
    return {"correct": correct, "score": float(correct), "detail": checks}
