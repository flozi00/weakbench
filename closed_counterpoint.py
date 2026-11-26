"""Check and minimize edits to a note-against-note upper line under stated voice-leading rules."""

from functools import lru_cache

DETAIL_KEYS = (
    "valid_json",
    "counterpoint_shape",
    "counterpoint_palette",
    "counterpoint_harmony",
    "counterpoint_melody",
    "counterpoint_perfect_motion",
    "counterpoint_cadence",
    "counterpoint_fewest_edits",
    "counterpoint_least_displacement",
)

CONSONANT = frozenset((3, 4, 7, 8, 9, 12, 15, 16, 19))
STEPS = frozenset((0, 1, 2, 3, 4, 5, 7))


def interval_class(distance):
    return {7: "fifth", 12: "octave", 19: "fifth"}.get(distance)


def pitch(value):
    return type(value) is int and 0 <= value <= 127


def validate_expected(spec):
    if not isinstance(spec, dict) or set(spec) != {"bass", "draft", "allowed"}:
        return False
    bass, draft, allowed = (spec[k] for k in ("bass", "draft", "allowed"))
    if (
        not isinstance(bass, list)
        or not 4 <= len(bass) <= 10
        or not all(map(pitch, bass))
    ):
        return False
    if (
        not isinstance(draft, list)
        or len(draft) != len(bass)
        or not all(map(pitch, draft))
    ):
        return False
    if not isinstance(allowed, list) or len(allowed) != len(bass):
        return False
    if not all(
        isinstance(row, list)
        and 1 <= len(row) <= 6
        and all(map(pitch, row))
        and len(row) == len(set(row))
        for row in allowed
    ):
        return False
    return best(tuple(bass), tuple(draft), tuple(map(tuple, allowed))) is not None


def legal_transition(bass, i, previous2, previous, current):
    if i == 0:
        return True, True, True
    delta = current - previous
    melody = abs(delta) in STEPS
    if previous2 is not None and abs(previous - previous2) >= 5:
        melody = melody and 1 <= abs(delta) <= 2 and delta * (previous - previous2) < 0
    bass_delta = bass[i] - bass[i - 1]
    now = interval_class(current - bass[i])
    last = interval_class(previous - bass[i - 1])
    same_direction = delta * bass_delta > 0
    motion = not (
        same_direction and now is not None and (now == last or abs(delta) > 2)
    )
    cadence = i != len(bass) - 1 or (
        1 <= abs(delta) <= 2 and 1 <= abs(bass_delta) <= 2 and delta * bass_delta < 0
    )
    return melody, motion, cadence


def harmonic(bass, i, current):
    d = current - bass[i]
    return d in ((7, 12) if i == 0 else (12,) if i == len(bass) - 1 else CONSONANT)


@lru_cache(maxsize=128)
def best(bass, draft, allowed):
    # Future legality uses at most the last two upper notes. Among prefixes with
    # that history a more expensive prefix cannot win the additive objective.
    states = {(None, None): (0, 0)}
    for i, options in enumerate(allowed):
        step = {}
        for (a, b), score in states.items():
            for p in options:
                if not harmonic(bass, i, p) or not all(
                    legal_transition(bass, i, a, b, p)
                ):
                    continue
                key = (b, p)
                cost = (score[0] + int(p != draft[i]), score[1] + abs(p - draft[i]))
                if key not in step or cost < step[key]:
                    step[key] = cost
        states = step
    return min(states.values(), default=None)


def score(item, reply):
    from hidden import strict_json

    metrics = dict.fromkeys(DETAIL_KEYS, 0.0)
    try:
        given = strict_json(reply.get("content"))
        metrics["valid_json"] = 1.0
    except (TypeError, ValueError, RecursionError):
        given = None
    spec = item["expected"]
    bass = spec["bass"]
    draft = spec["draft"]
    if isinstance(given, list) and len(given) == len(bass) and all(map(pitch, given)):
        metrics["counterpoint_shape"] = 1.0
        metrics["counterpoint_palette"] = float(
            all(note in row for note, row in zip(given, spec["allowed"]))
        )
        metrics["counterpoint_harmony"] = float(
            all(harmonic(bass, i, p) for i, p in enumerate(given))
        )
        grades = [
            legal_transition(
                bass,
                i,
                given[i - 2] if i >= 2 else None,
                given[i - 1] if i else None,
                p,
            )
            for i, p in enumerate(given)
        ]
        for k, vs in zip(
            (
                "counterpoint_melody",
                "counterpoint_perfect_motion",
                "counterpoint_cadence",
            ),
            zip(*grades),
        ):
            metrics[k] = float(all(vs))
        if all(metrics[k] for k in DETAIL_KEYS[:-2]):
            target = best(tuple(bass), tuple(draft), tuple(map(tuple, spec["allowed"])))
            changed = sum(a != b for a, b in zip(given, draft))
            displacement = sum(abs(a - b) for a, b in zip(given, draft))
            metrics["counterpoint_fewest_edits"] = float(changed == target[0])
            metrics["counterpoint_least_displacement"] = float(
                (changed, displacement) == target
            )
    correct = all(metrics.values())
    return {"correct": correct, "score": float(correct), "detail": metrics}
