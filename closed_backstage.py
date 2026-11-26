"""Verify staged narratives by replaying physical moves, observations and discoveries."""

import re

DETAIL_KEYS = (
    "story_language_form",
    "story_paragraphs",
    "story_opening",
    "story_belief",
    "story_evidence",
    "story_resolution",
    "story_alternate_endings",
)
FORMS = {
    "de": {
        "move": (
            "{m} trug das Requisit von Fach {u} zu Fach {v}.",
            "Von Fach {u} brachte {m} das Requisit zu Fach {v}.",
        ),
        "search": (
            "Zuerst suchte {s} vergeblich in Fach {u}.",
            "In Fach {u} griff {s} zunächst ins Leere.",
        ),
        "note": (
            "Die Karte in Fach {u} schickte {s} zu Fach {v}.",
            "{s} folgte dem Zettel von Fach {u} zu Fach {v}.",
        ),
        "ask": (
            "Bei Fach {u} verwies {h} {s} auf Fach {v}.",
            "{s} ging von Fach {u} zu Fach {v}, weil {h} dort das Requisit vermutete.",
        ),
        "find": (
            "{s} fand das Requisit in Fach {u}.",
            "In Fach {u} hielt {s} endlich das Requisit in der Hand.",
        ),
    },
    "en": {
        "move": (
            "{m} carried the prop from locker {u} to locker {v}.",
            "From locker {u}, {m} took the prop to locker {v}.",
        ),
        "search": (
            "First, {s} searched locker {u} in vain.",
            "At locker {u}, {s} came up empty-handed.",
        ),
        "note": (
            "The card in locker {u} sent {s} to locker {v}.",
            "{s} followed the note from locker {u} to locker {v}.",
        ),
        "ask": (
            "At locker {u}, {h} directed {s} to locker {v}.",
            "{s} went from locker {u} to locker {v} because {h} thought the prop was there.",
        ),
        "find": (
            "{s} found the prop in locker {u}.",
            "In locker {u}, {s} finally laid hands on the prop.",
        ),
    },
}


def name(value):
    return (
        isinstance(value, str) and re.fullmatch(r"[A-Z][a-z]{1,15}", value) is not None
    )


def distinct_names(items, maximum):
    return (
        isinstance(items, list)
        and 0 < len(items) <= maximum
        and all(name(x) for x in items)
        and len(items) == len(set(items))
    )


def validate_expected(spec):
    if not isinstance(spec, dict) or set(spec) != {
        "cast",
        "start",
        "final",
        "endings",
        "scenes",
    }:
        return False
    if not distinct_names(spec["cast"], 3) or len(spec["cast"]) != 3:
        return False
    if (
        not name(spec["start"])
        or not name(spec["final"])
        or spec["start"] == spec["final"]
        or type(spec["endings"]) is not int
        or spec["endings"] not in (1, 2)
    ):
        return False
    scenes = spec["scenes"]
    if not isinstance(scenes, list) or not 2 <= len(scenes) <= 6:
        return False
    places = {spec["start"], spec["final"]}
    for scene in scenes:
        if not isinstance(scene, dict) or set(scene) != {
            "moves",
            "seeker",
            "helper",
            "card",
        }:
            return False
        moves = scene["moves"]
        if (
            not isinstance(moves, list)
            or not 1 <= len(moves) <= 18
            or not all(
                isinstance(x, list)
                and len(x) == 2
                and all(map(name, x))
                and x[0] != x[1]
                for x in moves
            )
            or len(moves) != len(set(map(tuple, moves)))
        ):
            return False
        destinations = {y for _, y in moves}
        places.update(x for edge in moves for x in edge)
        for key in ("seeker", "helper", "card"):
            xs = scene[key]
            if (
                not isinstance(xs, list)
                or len(xs) > len(destinations)
                or not all(name(x) and x in destinations for x in xs)
                or len(xs) != len(set(xs))
            ):
                return False
    return len(places) <= 32 and places.isdisjoint(spec["cast"])


def read(text, spec, lang):
    if not isinstance(text, str) or len(text) > 8192 or lang not in FORMS:
        return []
    roles = dict(zip(("m", "s", "h"), spec["cast"]))
    places = {x for sc in spec["scenes"] for edge in sc["moves"] for x in edge}
    patterns = []
    for action, pair in FORMS[lang].items():
        for template in pair:
            pattern = re.escape(template)
            for k, person in roles.items():
                pattern = pattern.replace(re.escape("{" + k + "}"), re.escape(person))
            for k in ("u", "v"):
                pattern = pattern.replace(
                    re.escape("{" + k + "}"), "(?P<" + k + ">[A-Z][a-z]{1,15})"
                )
            patterns.append((action, re.compile(pattern)))

    def parse(line):
        for action, pattern in patterns:
            match = pattern.fullmatch(line)
            if match and all(value in places for value in match.groups()):
                return (action, match["u"], match.groupdict().get("v"))
        return None

    return [
        [parse(line.strip()) for line in block.splitlines()]
        for block in re.split(r"\n[ \t]*\n", text.strip().replace("\r\n", "\n"))
    ]


def open_scenes(lines, spec):
    path = [spec["start"]]
    seeker = helper = path[0]
    cards = {}
    if len(lines) != len(spec["scenes"]):
        return None
    for event, scene in zip(lines, spec["scenes"]):
        if not event:
            return None
        action, origin, dest = event
        if (
            action != "move"
            or origin != path[-1]
            or dest in path
            or [origin, dest] not in scene["moves"]
        ):
            return None
        path.append(dest)
        if dest in scene["seeker"]:
            seeker = dest
        if dest in scene["helper"]:
            helper = dest
        if dest in scene["card"]:
            cards[origin] = dest
    if path[-1] != spec["final"]:
        return None
    return path, seeker, helper, cards


def replay_ending(lines, path, seeker, helper, cards):
    if not lines or not all(lines):
        return False, False, False, None, ()
    belief = lines[0] == ("search", seeker, None) and seeker != path[-1]
    current = seeker
    used_witness = False
    clues = lines[1:-1]
    evidence = bool(clues)
    for kind, origin, dest in clues:
        if (
            origin != current
            or dest not in path
            or path.index(dest) <= path.index(current)
            or not (
                (kind == "note" and cards.get(origin) == dest)
                or (kind == "ask" and not used_witness and dest == helper)
            )
        ):
            evidence = False
            break
        current = dest
        used_witness |= kind == "ask"
    resolved = bool(
        belief
        and evidence
        and current == path[-1]
        and lines[-1] == ("find", current, None)
    )
    return (
        belief,
        evidence,
        resolved,
        clues[0][0] if clues and evidence else None,
        tuple(c[2] for c in clues),
    )


def score(item, reply):
    spec = item["expected"]
    count = spec["endings"]
    keys = DETAIL_KEYS if count == 2 else DETAIL_KEYS[:-1]
    metrics = dict.fromkeys(keys, 0.0)
    paragraphs = read(reply.get("content"), spec, item["lang"])
    if paragraphs:
        metrics["story_language_form"] = float(
            all(block and all(block) for block in paragraphs)
        )
    if len(paragraphs) == 1 + count:
        metrics["story_paragraphs"] = 1.0
        opening = open_scenes(paragraphs[0], spec)
        if opening:
            metrics["story_opening"] = 1.0
            endings = [replay_ending(block, *opening) for block in paragraphs[1:]]
            for i, k in enumerate(
                ("story_belief", "story_evidence", "story_resolution")
            ):
                metrics[k] = sum(end[i] for end in endings) / count
            if count == 2:
                metrics["story_alternate_endings"] = float(
                    {end[3] for end in endings} == {"ask", "note"}
                    and endings[0][4] != endings[1][4]
                )
    correct = all(value == 1 for value in metrics.values())
    return {"correct": correct, "score": float(correct), "detail": metrics}
