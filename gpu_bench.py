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
from kernel_pairs import paired_summary

DEFAULT_DB = Path.home() / ".local/share/weakbench/gpu-kernels.sqlite3"
TRACK_TARGETS = {
    "metal": ("metal",),
    "cuda-compile": ("cuda",),
    "cuda-hardware": ("cuda",),
    "msl-native": ("msl-code", "msl-json"),
}
TRACKS = tuple(TRACK_TARGETS)
PROMPT_FIELDS = {
    "metal": "prompt",
    "cuda": "cuda_prompt",
    "msl-code": "prompt_code",
    "msl-json": "prompt_json",
}


class SubmissionError(ValueError):
    """An operator-visible failure category, independent of private response text."""


def evaluator_digest(track):
    root = Path(__file__).resolve().parent
    files = ["gpu_bench.py", "kernel_catalog.py"]
    if track == "metal":
        files.extend(("kernel_metal.py", "kernel_metal_host.m"))
    elif track == "msl-native":
        files.extend(
            ("kernel_msl.py", "kernel_msl_host.m", "kernel_metal.py", "kernel_pairs.py")
        )
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


def native_source(text, response_format):
    """Remove only a single enclosing fence. Treat bare source syntax as the compiler's decision."""
    if response_format not in ("code", "json"):
        raise ValueError("unknown native submission format")
    body = text.strip()
    fence = re.match(r"^(`{3,}|~{3,})([^\r\n]*)\r?\n", body)
    if fence:
        marker, label = fence.groups()
        if (
            marker.startswith("`")
            and "`" in label
            or response_format == "json"
            and label.strip().lower() not in ("", "json")
        ):
            raise SubmissionError("invalid_fence")
        closing = re.compile(
            rf"(?m)^[ ]{{0,3}}{re.escape(marker[0])}{{{len(marker)},}}[ \t]*\r?$"
        )
        last = closing.search(body, fence.end())
        if last is None or last.end() != len(body):
            raise SubmissionError("invalid_fence")
        body = body[fence.end() : last.start()].removesuffix("\n").removesuffix("\r")
    elif re.search(r"(?m)^[ ]{0,3}(?:`{3,}|~{3,})", body):
        raise SubmissionError("invalid_fence")
    if response_format == "json":
        try:
            value = hidden.strict_json(body)
        except (ValueError, RecursionError):
            raise SubmissionError("invalid_json") from None
        if (
            not isinstance(value, dict)
            or set(value) != {"source"}
            or not isinstance(value["source"], str)
        ):
            raise SubmissionError("invalid_submission_fields")
        body = value["source"]
    elif body.startswith(("{", "[")):
        raise SubmissionError("invalid_source")
    if not body.strip():
        raise SubmissionError("invalid_source")
    try:
        length = len(body.encode())
    except UnicodeEncodeError:
        raise SubmissionError("invalid_source") from None
    if length > 64000 or "\x00" in body:
        raise SubmissionError("invalid_source")
    return body


def tasks_for(catalog, target, language=None, case=None):
    cases = []
    for task in catalog["tasks"]:
        if language and language != task["language"] or case and case != task["id"]:
            continue
        cases.append(
            {
                "id": target + "-" + task["id"],
                "base_id": task["id"],
                "target": target,
                "kind": "text",
                "language": task["language"],
                "prompt": task[PROMPT_FIELDS[target]],
                "rubric": [
                    "Evaluate against the numerical oracle on the selected backend."
                ],
            }
        )
    if not cases:
        raise ValueError("No tasks match this selection.")
    return cases


def grade(raw, workload, track, seed, response_format=None):
    if track == "msl-native":
        try:
            source = native_source(raw, response_format)
        except SubmissionError as exc:
            return {"status": "invalid_submission", "parsed": False, "reason": str(exc)}
        backend = importlib.import_module("kernel_msl")
        try:
            result = backend.measure(
                source, workload["msl"], workload["vectors"], seed=seed
            )
        except backend.BackendUnavailable as exc:
            result = unavailable(exc)
        return {
            **result,
            "parsed": True,
            "source_sha256": sha256(source.encode()).hexdigest(),
        }
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
        return unavailable(exc)


def unavailable(exc):
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
        model_keys = {model_key} if isinstance(model_key, str) else set(model_key)
        candidates = [p for p in candidates if p["key"] in model_keys]
        if {p["key"] for p in candidates} != model_keys:
            raise ValueError("Unknown model profile in the submission archive.")
    if not candidates:
        raise ValueError("No matching model profile in the submission archive.")
    by_id = {task["id"]: task for task in catalog["tasks"]}
    results = []
    for model in candidates:
        rows = []
        for case in cases:
            cid = case["base_id"]
            task = by_id[cid]
            answer = archive.response(model["key"], case["id"])
            item = {
                "case_id": cid,
                "family": task["family"],
                "language": task["language"],
            }
            response_format = None
            if track == "msl-native":
                response_format = case["target"].removeprefix("msl-")
                item["response_format"] = response_format
            if answer is None:
                measured = {"status": "missing_submission"}
            else:
                item["answer_sha256"] = sha256(answer["content"].encode()).hexdigest()
                item["finish_reason"] = answer["finish_reason"]
                measured = grade(
                    answer["content"],
                    catalog["workloads"][task["family"]],
                    track,
                    seed,
                    response_format,
                )
            rows.append({**item, "measurement": measured})
        summary = totals(rows)
        if track == "msl-native":
            summary.update(paired_summary(rows))
        results.append({"model": model, "summary": summary, "results": rows})
    return results


def reference_control(catalog, track, seed):
    target = "metal" if track == "metal" else "cuda"
    rows = []
    for family, workload in catalog["workloads"].items():
        if track == "msl-native":
            measured = grade(workload["msl"], workload, track, seed, "code")
        else:
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
    prep = sub.add_parser("prepare")
    prep.add_argument(
        "--msl", action="store_true", help="also prepare the matched native Metal tasks"
    )
    run = sub.add_parser("generate", help="archive visible responses from a model")
    run.add_argument("--target", choices=PROMPT_FIELDS, required=True)
    for name in ("model", "revision", "url"):
        run.add_argument("--" + name, required=True)
    run.add_argument("--settings", type=hidden.strict_json, default={})
    test = sub.add_parser("evaluate", help="score archived kernels, without inference")
    test.add_argument("--track", choices=TRACKS, required=True)
    test.add_argument(
        "--model-key",
        action="append",
        help="select a model profile; repeat to compare profiles",
    )
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
        p.add_argument(
            "--msl-catalog",
            type=Path,
            help="explicit native Metal supplement for development",
        )
        p.add_argument("--lang", choices=("en", "de"))
        p.add_argument("--case")
        p.add_argument(
            "--catalog", type=Path, help="explicit local gzip catalogue for development"
        )
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            kernel_catalog.prepare()
            if args.msl:
                kernel_catalog.prepare_msl()
            print("Private GPU catalogue prepared.")
            return 0
        if args.command == "evaluate" and not 0 <= args.seed <= 0xFFFFFFFF:
            raise ValueError("GPU measurement seed must be a uint32")
        catalog_file = args.catalog or kernel_catalog.cache_path()
        if not catalog_file.is_file():
            raise ValueError("GPU catalogue missing. Run python gpu_bench.py prepare.")
        data = catalog_file.read_bytes()
        catalog = kernel_catalog.decode(data, pinned=args.catalog is None)
        parent_digest = sha256(data).hexdigest()
        targets = (
            (args.target,) if args.command == "generate" else TRACK_TARGETS[args.track]
        )
        native = targets[0].startswith("msl-")
        supplement = {}
        if native:
            catalog, digest = kernel_catalog.load_msl(
                catalog, parent_digest, args.msl_catalog
            )
            supplement = {
                "msl_catalog_sha256": digest,
                "msl_catalog_revision": kernel_catalog.MSL_REVISION
                if args.msl_catalog is None
                else None,
            }
        elif args.msl_catalog:
            raise ValueError("Native Metal catalogue supplied for a different target")
        control = args.command == "evaluate" and args.reference_control
        if control:
            if args.model_key or args.lang or args.case:
                raise ValueError(
                    "reference control uses all workloads without model or language filters"
                )
            results = reference_control(catalog, args.track, args.seed)
        else:
            cases = [
                case
                for target in targets
                for case in tasks_for(catalog, target, args.lang, args.case)
            ]
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
            "catalog_sha256": parent_digest,
            **supplement,
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
