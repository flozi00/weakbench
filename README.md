# weakbench

WeakBench measures where language models struggle in German and English. The default quality suite uses original cases held in a private Hugging Face dataset. This repository contains the runner, scorers, release pins and public calibration controls.

## Run the private suite

Use Python 3.11 or newer. Prepare the pinned release with an account that can read the dataset. The model process can then run with access to the prepared cache alone. Set `BENCH_CACHE` on both commands to choose its location.

```sh
python3 hidden.py prepare
python3 quality.py --url http://candidate:8000 --arm reference --tier quick
python3 quality.py --url http://candidate:8000 --arm judged --tier full --judge-url http://judge:8000
```

The quick tier takes half the cases from each family and language in a repeatable order. Full uses every case. A run without a judge evaluates three exact-match families. Providing a judge adds two families with criterion-level grades. For a subset, pass `--tasks revision_ledger,evidence_handover --judge-url URL`. The judge must pass English and German calibration controls and return one boolean per criterion. Accuracy requires satisfying every criterion. The graded score also reports the fraction satisfied.

Both model endpoints receive private material. Use endpoints approved for the dataset. The candidate receives each prompt. The judge also sees the rubric and candidate response. Closed-run artifacts retain opaque item IDs, scores, timing, token counts, error types and the release identity. Their contents support paired A/B comparisons. Prefill drift uses separate public technical probes because its output includes token IDs. Use `--no-drift` for a generation-only run, or `--image` for the optional generated vision capability check.

```sh
python3 compare.py --help
python3 speed.py --help
```

See [the task catalogue](TASKS.md) for counts and scoring. `run_info.json` records the private data revision and SHA-256. The manifest records the scorer hash. Keep the revision, scorer, item selection, judge and generation settings fixed when comparing runs.

## Maintain the release

The private repository `flozi00/weakbench-data` holds the release JSONL. Each row has a unique ID, family, language, scoring mode and prompt. Exact rows carry the expected JSON value. Judged rows carry an ordered list of criteria. Author and independently validate replacements outside the code checkout, upload to the private dataset, then update the immutable revision, checksum and cell sizes in `hidden.py`. The preparation command checks repository visibility and file integrity before staging the data.

For a judge evaluation, download `authoring/closed-v1/calibration.jsonl` from the private dataset to a private local directory. Run `python3 calibrate_judge.py --controls PATH --judge-url URL` with the prepared release. It reports agreement with labelled passing and failing answers separately for German and English.

## Historical diagnostics

Historical public datasets and generated probes are available through an explicit diagnostic profile. These commands use the dependencies declared in `pyproject.toml`. Diagnostic runs write richer artifacts, including model responses, for analysis of historical benchmark results.

```sh
uv run quality.py --suite diagnostics --url http://candidate:8000 --arm exploration --tasks traps,de_grammar
uv run pairwise.py --help
```

## Weakness report

`uv run weakness_report.py runs/<run> [runs/<other>]` ranks what a run exposes: the trap-vs-control gaps (largest first), how often each bait was taken (`took_*`, `guessed`, `naive_sum`, ...), difficulty cliffs (the digit count, depth, block count or number of operations where accuracy falls below half its best) and the least consistent cells.

## Shortcut check

`uv run shortcut_check.py` scores every generated task against constant replies ("yes", "A", "die", "cannot be determined", ...). A task where a constant reaches 0.6 or more has a shortcut. 0.5 on a trap/control-twin task is the expected ceiling of always abstaining.

## Tests

```sh
for t in test_*.py; do uv run -q python "$t" || echo "FAIL $t"; done
```

Adding a task: implement the `TASKS` contract in one of the `tasks_*.py` files (described in `quality.py`'s docstring) and add a test next to it. The quick tier must stay a prefix of the full tier.
