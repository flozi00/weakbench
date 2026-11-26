"""Generate kernel submissions once and evaluate numerical correctness and speed on a named backend."""

import argparse
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import importlib
import json
import os
from pathlib import Path
import re
import sys
import sqlite3
import tempfile

from arena import Arena
import common
import hidden
import kernel_catalog

DEFAULT_DB = Path.home() / ".local/share/weakbench/gpu-kernels.sqlite3"
TRACKS = ("metal", "cuda-compile", "cuda-hardware")


class SubmissionError(ValueError):
    """An operator-visible failure category, independent of private response text."""


def evaluator_digest(track):
    root = Path(__file__).resolve().parent
    files = ["gpu_bench.py", "kernel_catalog.py"]
    if track == "metal":
        files.extend(("kernel_metal.py", "kernel_metal_host.m"))
    else:
        files.extend(
            (
                "kernel_cuda.py",
                "kernel-cuda/worker.py",
                "kernel-cuda/Dockerfile",
                "kernel-cuda/install.py",
                "kernel-cuda/nvrtc-wheels.json",
            )
        )
    checksum = sha256()
    for filename in files:
        checksum.update(
            filename.encode() + b"\0" + (root / filename).read_bytes() + b"\0"
        )
    return checksum.hexdigest()


def submission(text):
    match = re.fullmatch(
        r"\s*```(?:json)?[ \t]*\r?\n(.*?)\r?\n```\s*", text, re.S | re.I
    )
    try:
        value = hidden.strict_json(match[1] if match else text)
    except (ValueError, RecursionError):
        raise SubmissionError("invalid_json") from None
    if (
        not isinstance(value, dict)
        or set(value) != {"source", "threads", "outputs_per_thread"}
        or not isinstance(value["source"], str)
        or not 0 < len(value["source"].strip()) <= 64000
        or type(value["threads"]) is not int
        or value["threads"] not in (32, 64, 128, 256)
        or type(value["outputs_per_thread"]) is not int
        or value["outputs_per_thread"] not in (1, 2, 4, 8)
    ):
        raise SubmissionError("invalid_submission_fields")
    return value


def tasks_for(catalog, target, language=None, case=None):
    cases = []
    for task in catalog["tasks"]:
        if language and language != task["language"] or case and case != task["id"]:
            continue
        cases.append(
            {
                "id": target + "-" + task["id"],
                "kind": "text",
                "language": task["language"],
                "prompt": task["cuda_prompt" if target == "cuda" else "prompt"],
                "rubric": [
                    "Evaluate against the numerical oracle on the selected backend."
                ],
            }
        )
    if not cases:
        raise ValueError("No tasks match this selection.")
    return cases


def grade(raw, workload, track, seed):
    try:
        program = submission(raw)
    except SubmissionError as exc:
        return {"status": "invalid_submission", "reason": str(exc)}
    backend = importlib.import_module(
        "kernel_metal" if track == "metal" else "kernel_cuda"
    )
    try:
        if track == "cuda-compile":
            return backend.compile_native(program)
        result = backend.measure(
            program,
            workload["metal" if track == "metal" else "cuda"],
            workload["vectors"],
            seed=seed,
        )
        if track == "metal" and result["status"] == "invalid_candidate":
            result["reason"] = {
                "invalid kernel syntax": "invalid_kernel_syntax",
                "unsupported statement": "unsupported_statement",
            }.get(result.get("reason"), "source_contract")
        return result
    except backend.BackendUnavailable as exc:
        result = {"status": "backend_unavailable"}
        code = getattr(exc, "reason_code", None)
        if code in ("backend_timeout", "dedicated_device_unselected"):
            result["reason"] = code
        return result


def totals(results):
    states = Counter(r["measurement"]["status"] for r in results)
    forms = [
        part
        for r in results
        if r["measurement"]["status"] == "correct"
        for part in r["measurement"].get("cases", [])
        if part.get("timing")
    ]
    return {
        "selected_submissions": len(results),
        "statuses": dict(sorted(states.items())),
        "timed_shapes": len(forms),
        "faster_shapes": sum(p["timing"].get("comparison") == "faster" for p in forms),
    }


def evaluate(archive, catalog, cases, track, model_key, seed):
    candidates = archive.inventory()["models"]
    if model_key:
        candidates = [p for p in candidates if p["key"] == model_key]
    if not candidates:
        raise ValueError("No matching model profile in the submission archive.")
    target = "metal" if track == "metal" else "cuda"
    by_id = {task["id"]: task for task in catalog["tasks"]}
    results = []
    for model in candidates:
        rows = []
        for case in cases:
            cid = case["id"].removeprefix(target + "-")
            task = by_id[cid]
            answer = archive.response(model["key"], case["id"])
            item = {
                "case_id": cid,
                "family": task["family"],
                "language": task["language"],
            }
            if answer is None:
                measured = {"status": "missing_submission"}
            else:
                item["answer_sha256"] = sha256(answer["content"].encode()).hexdigest()
                item["finish_reason"] = answer["finish_reason"]
                measured = grade(
                    answer["content"], catalog["workloads"][task["family"]], track, seed
                )
            rows.append({**item, "measurement": measured})
        results.append({"model": model, "summary": totals(rows), "results": rows})
    return results


def reference_control(catalog, track, seed):
    target = "metal" if track == "metal" else "cuda"
    rows = []
    for family, workload in catalog["workloads"].items():
        measured = grade(json.dumps(workload[target]), workload, track, seed)
        rows.append(
            {
                "case_id": family,
                "family": family,
                "language": "de/en",
                "measurement": measured,
            }
        )
    return [
        {
            "model": {
                "id": "Reference self-check",
                "revision": "",
                "key": "reference_control",
            },
            "summary": totals(rows),
            "results": rows,
        }
    ]


def save_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as target:
        tmp = Path(target.name)
        try:
            target.write(
                json.dumps(
                    report, indent=2, ensure_ascii=False, allow_nan=False
                ).encode()
            )
            target.flush()
            os.fsync(target.fileno())
            # Preserve historical measurements and refuse an accidental overwrite.
            os.link(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    run = sub.add_parser("generate", help="archive visible responses from a model")
    run.add_argument("--target", choices=("metal", "cuda"), required=True)
    for name in ("model", "revision", "url"):
        run.add_argument("--" + name, required=True)
    run.add_argument("--settings", type=hidden.strict_json, default={})
    test = sub.add_parser("evaluate", help="score archived kernels, without inference")
    test.add_argument("--track", choices=TRACKS, required=True)
    test.add_argument("--model-key")
    test.add_argument(
        "--reference-control",
        action="store_true",
        help="evaluate the fixed reference once per workload, independently of archived models",
    )
    test.add_argument("--seed", type=int, default=17)
    test.add_argument(
        "--output",
        type=Path,
        help="new JSON report; default creates a timestamped file next to the database",
    )
    for p in (run, test):
        p.add_argument("--db", type=Path, default=DEFAULT_DB)
        p.add_argument("--lang", choices=("en", "de"))
        p.add_argument("--case")
        p.add_argument(
            "--catalog", type=Path, help="explicit local gzip catalogue for development"
        )
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            kernel_catalog.prepare()
            print("Private GPU catalogue prepared.")
            return 0
        if args.command == "evaluate" and not 0 <= args.seed <= 0xFFFFFFFF:
            raise ValueError("GPU measurement seed must be a uint32")
        catalog = kernel_catalog.load_catalog(args.catalog)
        target = (
            args.target
            if args.command == "generate"
            else "metal"
            if args.track == "metal"
            else "cuda"
        )
        control = args.command == "evaluate" and args.reference_control
        if control:
            if args.model_key or args.lang or args.case:
                raise ValueError(
                    "reference control uses all workloads without model or language filters"
                )
            results = reference_control(catalog, args.track, args.seed)
        else:
            cases = tasks_for(catalog, target, args.lang, args.case)
            archive = Arena(args.db)
            with archive.runner():
                archive.register_cases(cases)
                if args.command == "generate":
                    key = archive.register_model(
                        args.model, args.revision, args.settings
                    )
                    result = archive.generate(key, args.url)
                    print(json.dumps({"model_key": key, **result}))
                    return int(bool(result["failed"]))
                results = evaluate(
                    archive, catalog, cases, args.track, args.model_key, args.seed
                )
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        path = (
            args.output or args.db.parent / "gpu-reports" / f"{stamp}-{args.track}.json"
        )
        report = {
            "timestamp": stamp,
            "role": "reference_control" if control else "models",
            "catalog_revision": kernel_catalog.REVISION
            if args.catalog is None
            else None,
            "catalog_sha256": kernel_catalog.DIGEST
            if args.catalog is None
            else sha256(args.catalog.read_bytes()).hexdigest(),
            "track": args.track,
            "bench_git_sha": common._git_sha(),
            "evaluator_sha256": evaluator_digest(args.track),
            "seed": args.seed,
            "models": results,
        }
        save_report(path, report)
        print(
            json.dumps(
                {
                    "report": str(path),
                    "models": [
                        {"id": r["model"]["id"], **r["summary"]} for r in results
                    ],
                }
            )
        )
        observed = {status for row in results for status in row["summary"]["statuses"]}
        if control:
            return int(bool(observed - {"correct", "compiled"}))
        return int(bool(observed & {"backend_unavailable", "missing_submission"}))
    except (ValueError, OSError, sqlite3.Error) as exc:
        # Payloads in generated source and provider diagnostics can contain private material.
        print(
            f"GPU benchmark failed ({type(exc).__name__}). Check the catalogue, archive and output path.",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    sys.exit(main())
