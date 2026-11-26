"""Grade a pixel selected on an optional visual control panel."""
DETAIL_KEYS = ("valid_json", "point_form_ok", "actionability_correct", "actionable_point_hit")


def validate_expected(spec):
    if not isinstance(spec, dict) or set(spec) != {"size", "regions"}:
        return False
    size, regions = spec["size"], spec["regions"]
    if not (isinstance(size, list) and len(size) == 2
            and all(type(n) is int and 0 < n <= 2048 for n in size)):
        return False
    if not isinstance(regions, list) or len(regions) > 64:
        return False
    for box in regions:
        if not (isinstance(box, list) and len(box) == 4 and all(type(n) is int for n in box)
                and 0 <= box[0] < box[2] <= size[0] and 0 <= box[1] < box[3] <= size[1]):
            return False
    return True


def score(item, reply):
    from hidden import strict_json

    try:
        value = strict_json(reply.get("content"))
        parsed = True
    except (TypeError, ValueError, RecursionError):
        value, parsed = None, False
    spec = item["expected"]
    regions = spec["regions"]
    detail = {"valid_json": float(parsed), "point_form_ok": 0.0, "actionability_correct": 0.0}
    if regions:
        detail["actionable_point_hit"] = 0.0
    hit = False
    if isinstance(value, dict) and set(value) == {"point"}:
        point = value["point"]
        if point is None:
            detail["point_form_ok"] = 1.0
            hit = not regions
            detail["actionability_correct"] = float(hit)
        elif (isinstance(point, list) and len(point) == 2 and all(type(n) is int for n in point)
              and all(0 <= n < limit for n, limit in zip(point, spec["size"]))):
            detail["point_form_ok"] = 1.0
            detail["actionability_correct"] = float(bool(regions))
            x, y = point
            hit = any(left <= x < right and top <= y < bottom for left, top, right, bottom in regions)
            if regions:
                detail["actionable_point_hit"] = float(hit)
    return {"correct": hit, "score": float(hit), "detail": detail}
