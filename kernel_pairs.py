"""Paired end-to-end stages for two presentations of the same native Metal challenge."""

import re


def stages(measurement):
    status = measurement["status"]
    parsed, compiled = measurement.get("parsed"), measurement.get("compiled")
    if status == "missing_submission" and parsed is None and compiled is None:
        return None, None, None
    if status == "invalid_submission" and parsed is False and compiled is None:
        return False, False, False
    if parsed is not True:
        raise ValueError("inconsistent native submission status")
    if status == "backend_unavailable" and (type(compiled) is bool or compiled is None):
        return True, compiled, None
    if status in ("invalid_program", "compile_error") and compiled is False:
        return True, False, False
    if status in ("correct", "incorrect") and compiled is True:
        checks = [form["validation"]["passed"] for form in measurement.get("cases", [])]
        if (
            not all(type(check) is bool for check in checks)
            or status == "correct"
            and (not checks or not all(checks))
            or status == "incorrect"
            and False not in checks
            and measurement.get("reason") != "device_execution_error"
        ):
            raise ValueError(
                "native hardware result lacks consistent validation evidence"
            )
        return True, True, status == "correct"
    raise ValueError("inconsistent native pipeline status")


def paired_summary(rows):
    """An unobserved stage stays outside the four measured outcome counts."""
    pairs, identities, hashes = {}, {}, {}
    for row in rows:
        cid, fmt = row["case_id"], row["response_format"]
        identity = (row["family"], row["language"])
        if fmt not in ("code", "json") or identity != identities.setdefault(
            cid, identity
        ):
            raise ValueError("unmatched native format pair")
        pair = pairs.setdefault(cid, {})
        if fmt in pair:
            raise ValueError("duplicate native format observation")
        measurement = row["measurement"]
        pair[fmt] = stages(measurement)
        if pair[fmt][0] is True:
            sha = measurement.get("source_sha256")
            if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha):
                raise ValueError("missing native source fingerprint")
            hashes.setdefault(cid, {})[fmt] = sha
    categories = {
        (True, True): "both",
        (True, False): "code_only",
        (False, True): "json_only",
        (False, False): "neither",
    }
    counts = {
        stage: dict.fromkeys((*categories.values(), "unobserved"), 0)
        for stage in ("parsed", "compiled", "correct")
    }
    unknown = (None, None, None)
    for pair in pairs.values():
        outcomes = zip(
            pair.get("code", unknown), pair.get("json", unknown), strict=True
        )
        for table, sides in zip(counts.values(), outcomes, strict=True):
            table[categories.get(sides, "unobserved")] += 1
    agreement = {"identical": 0, "different": 0, "unavailable": len(pairs)}
    for pair in hashes.values():
        if len(pair) == 2:
            agreement["unavailable"] -= 1
            agreement["identical" if pair["code"] == pair["json"] else "different"] += 1
    return {"format_pairs": counts, "source_comparison": agreement}
