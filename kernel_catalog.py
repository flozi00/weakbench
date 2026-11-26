"""Load the private GPU development catalogue as data without loading authoring code."""

from collections import Counter
import gzip
from hashlib import sha256
import io
import math
import os
from pathlib import Path
import re

import hidden

REVISION = "7285a459c3ede68a62b6aaea5b5657faf9ea824a"
REMOTE_PATH = "releases/gpu-kernels-v1/bundle.json.gz"
DIGEST = "97de890fae25ad008d9c0c5c54cf88ab72eed7a3e33c0a2c02ee9ab4df49ff25"
MAX_BYTES = 128 << 20


def cache_path():
    base = Path(os.getenv("BENCH_CACHE", Path.home() / ".cache/weakbench"))
    return base / "gpu" / REVISION / "bundle.json.gz"


def decode(data, *, pinned=False):
    if pinned and sha256(data).hexdigest() != DIGEST:
        raise ValueError("private GPU catalogue checksum mismatch")
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(data)) as stream:
            raw = stream.read(MAX_BYTES + 1)
        if len(raw) > MAX_BYTES:
            raise ValueError
        bundle = hidden.strict_json(raw)
        if (
            set(bundle) != {"format", "tasks", "workloads"}
            or type(bundle["format"]) is not int
            or bundle["format"] != 1
        ):
            raise ValueError
        workloads, tasks = bundle["workloads"], bundle["tasks"]
        if (
            not isinstance(workloads, dict)
            or not 1 <= len(workloads) <= 64
            or not isinstance(tasks, list)
        ):
            raise ValueError
        ids = set()
        coverage = Counter()
        for task in tasks:
            if set(task) != {
                "id",
                "family",
                "language",
                "prompt",
                "cuda_prompt",
            } or not all(isinstance(v, str) and v.strip() for v in task.values()):
                raise ValueError
            uid, family, lang = (task[k] for k in ("id", "family", "language"))
            if (
                not re.fullmatch("[a-zA-Z0-9_-]{1,96}", uid)
                or uid in ids
                or family not in workloads
                or lang not in ("de", "en")
            ):
                raise ValueError
            ids.add(uid)
            coverage[family, lang] += 1
        if coverage != {
            (family, lang): 1 for family in workloads for lang in ("de", "en")
        }:
            raise ValueError
        if pinned and len(workloads) != 4:
            raise ValueError
        for spec in workloads.values():
            if set(spec) != {"metal", "cuda", "vectors"} or not all(
                isinstance(spec[k], dict)
                and set(spec[k]) == {"source", "threads", "outputs_per_thread"}
                for k in ("metal", "cuda")
            ):
                raise ValueError
            cases = spec["vectors"]
            if (
                not isinstance(cases, list)
                or not 2 <= len(cases) <= 128
                or {v["timing"] for v in cases} != {False, True}
            ):
                raise ValueError
            for v in cases:
                if (
                    set(v) != {"shape", "a", "b", "expected", "atol", "rtol", "timing"}
                    or not isinstance(v["shape"], list)
                    or len(v["shape"]) != 3
                    or not all(type(n) is int and n > 0 for n in v["shape"])
                    or not all(
                        isinstance(v[k], list) and 0 < len(v[k]) <= 1 << 20
                        for k in ("a", "b", "expected")
                    )
                    or not all(
                        type(n) in (int, float) and math.isfinite(n)
                        for k in ("a", "b", "expected")
                        for n in v[k]
                    )
                    or not all(
                        type(v[k]) in (int, float) and math.isfinite(v[k]) and v[k] >= 0
                        for k in ("atol", "rtol")
                    )
                    or type(v["timing"]) is not bool
                ):
                    raise ValueError
        return bundle
    except (
        AttributeError,
        EOFError,
        IndexError,
        KeyError,
        OSError,
        OverflowError,
        TypeError,
        ValueError,
        RecursionError,
    ) as exc:
        raise ValueError("invalid GPU challenge catalogue") from exc


def load_catalog(path=None):
    file = Path(path) if path else cache_path()
    if not file.is_file():
        raise ValueError("GPU catalogue missing. Run python gpu_bench.py prepare.")
    return decode(file.read_bytes(), pinned=path is None)


def prepare():
    hidden.prepare_private(
        hidden.DATA_REPO,
        REVISION,
        REMOTE_PATH,
        cache_path(),
        lambda data: decode(data, pinned=True),
    )


MSL_REVISION = "2bf30f76e9a4c857c6501dec834e1732da17f7d5"
MSL_DIGEST = "37acfe141dfe043dad135b3da630da2c60ee410ae956f47478f9b1a6eac13841"
MSL_REMOTE = "releases/gpu-msl-v1/tasks.json"


def msl_cache_path():
    return cache_path().parents[1] / MSL_REVISION / "native-msl.json"


def decode_msl(data, parent, parent_sha, *, pinned=False):
    """Join the supplement only to the validated numerical catalogue it was authored for."""
    if len(data) > 2 << 20 or pinned and sha256(data).hexdigest() != MSL_DIGEST:
        raise ValueError("invalid native Metal catalogue size or checksum")
    try:
        extension = hidden.strict_json(data)
        if (
            set(extension) != {"format", "parent_sha256", "tasks", "baselines"}
            or type(extension["format"]) is not int
            or extension["format"] != 1
            or extension["parent_sha256"] != parent_sha
        ):
            raise ValueError
        refs = extension["baselines"]
        if (
            not isinstance(refs, dict)
            or set(refs) != parent["workloads"].keys()
            or not all(
                isinstance(v, str) and 0 < len(v.strip()) <= 65536
                for v in refs.values()
            )
        ):
            raise ValueError
        tasks = extension["tasks"]
        if not isinstance(tasks, list) or len(tasks) != len(parent["tasks"]):
            raise ValueError
        by_id = {t["id"]: t for t in parent["tasks"]}
        overlay = {}
        for t in tasks:
            if set(t) != {"id", "family", "language", "prompt_code", "prompt_json"}:
                raise ValueError
            original = by_id[t["id"]]
            if (
                t["id"] in overlay
                or any(t[k] != original[k] for k in ("family", "language"))
                or not all(
                    isinstance(t[k], str) and t[k].strip()
                    for k in ("prompt_code", "prompt_json")
                )
                or t["prompt_code"] == t["prompt_json"]
            ):
                raise ValueError
            overlay[t["id"]] = t
        return {
            **parent,
            "tasks": [{**t, **overlay[t["id"]]} for t in parent["tasks"]],
            "workloads": {
                k: {**w, "msl": refs[k]} for k, w in parent["workloads"].items()
            },
        }
    except (ValueError, KeyError, TypeError, RecursionError) as exc:
        raise ValueError("invalid native Metal catalogue or numerical parent") from exc


def load_msl(parent, parent_sha, path=None):
    file = Path(path) if path else msl_cache_path()
    if not file.is_file():
        raise ValueError(
            "Native Metal catalogue missing. Run python gpu_bench.py prepare --msl."
        )
    with file.open("rb") as stream:
        data = stream.read((2 << 20) + 1)
    return decode_msl(data, parent, parent_sha, pinned=path is None), sha256(
        data
    ).hexdigest()


def prepare_msl():
    parent = load_catalog()
    hidden.prepare_private(
        hidden.DATA_REPO,
        MSL_REVISION,
        MSL_REMOTE,
        msl_cache_path(),
        lambda data: decode_msl(data, parent, DIGEST, pinned=True),
    )
