# weakbench

WeakBench measures where language models struggle in German and English. The default quality suite uses original cases held in a private Hugging Face dataset. This repository contains the runner, scorers, release pins and public calibration controls.

## Run the private suite

Use Python 3.11 or newer. Prepare the pinned release with an account that can read the dataset. The model process can then run with access to the prepared cache alone. Set `BENCH_CACHE` on both commands to choose its location.

```sh
python3 hidden.py prepare
python3 quality.py --url http://candidate:8000 --arm reference --tier quick
python3 quality.py --url http://candidate:8000 --arm judged --tier full --judge-url http://judge:8000
```

The `closed-v20` text suite has 61 families scored by code. Across German and English, these contribute 2,036 quick or 4,072 full cases. A judge adds four rubric-graded families, bringing the totals to 2,100 quick or 4,200 full. Quick samples half of each family in each language, keeping each counterfactual pair together and matching the sample across languages. The judge must pass English and German calibration controls and return one boolean per criterion. Accuracy requires satisfying every criterion. The graded score also reports the fraction satisfied.

To run the new text tasks on a local model:

```sh
python3 quality.py --url http://localhost:8000 --arm local --langs de,en --no-drift \
  --tasks web_currency,belief_revision,action_postconditions --concurrency 4 --max-parallel 4
```

Three optional private image families add 96 quick or 192 full questions across the two languages. To send the held-out charts, silhouettes and control panels to a local vision model, run:

```sh
python3 quality.py --url http://localhost:8000 --arm local-vision --langs de,en --no-drift \
  --image --tasks voxel_projections,interval_plot,panel_grounding --max-parallel 4 --concurrency 4
```

To evaluate timing differences between recorded taps, run the private audio family against a local audio-capable chat endpoint. The quick tier has 16 recordings per language and the full tier has 32.

```sh
python3 quality.py --url http://localhost:8000 --arm local-audio --langs de,en --no-drift \
  --audio --tasks tap_timing --max-parallel 4 --concurrency 4
```

The scheduling verifier accepts alternative optimal timetables. The language verifier grades a short visible justification and a final answer against the facts and the requested language. It also measures the language of any API-exposed reasoning and reports how often the model supplied that channel.

To check a saved chat completion offline, use the ID from the run's `items-<task>.jsonl` and the same private release. The command reads the full completion from a local file or standard input and prints only the verdict and numeric checks. Exit code 0 means correct, 1 means incorrect and 2 indicates an input or release error.

```sh
python3 check_response.py --task language_evidence --lang de --id "$ITEM_ID" --response completion.json
```

For `adaptive_measurement`, save the assistant tool calls and their tool replies in an array under `messages`, with the final chat completion under `completion`. The offline verifier replays those measurements against the case and checks that they identify the specimen. A correct guess alone earns zero. The ordinary runner keeps the trace in memory during scoring.

For `commitment_repair`, use the same offline command with all three original chat completions in order in an object under `completions`. The verifier computes the cost of each repair relative to the preceding booking in that conversation.

Both model endpoints receive private material. Use endpoints approved for the dataset. The candidate receives each prompt. The judge also sees the rubric and candidate response. Closed-run artifacts retain opaque item IDs, scores, timing, token counts, error types and the release identity. Their contents support paired A/B comparisons. Prefill drift uses separate public technical probes because its output includes token IDs. Use `--no-drift` for a generation-only run, or `--image` for the optional generated vision capability check.

```sh
python3 compare.py --help
python3 speed.py --help
```

`run_info.json` records the private data revision and SHA-256. The manifest records the scorer hash. Compare runs within a release, keeping the scorer, item selection, judge and generation settings fixed. Earlier runners are available for `closed-v8` at `8e0308bf`, `closed-v7` at `07db9356`, `closed-v6` at `fbdce2de`, `closed-v5` at `de074642`, `closed-v4` at `cfb63f1c`, `closed-v3` at `c38602ef` and `closed-v2` at `605df74c`. Family counts, scoring and metric denominators appear in the [task catalogue](TASKS.md). The [coverage analysis](COVERAGE.md) records how the new tasks differ from the earlier diagnostics.

## Maintain the release

The private repository `flozi00/weakbench-data` holds the release JSONL. Each row has a unique ID, family, language, scoring mode and prompt. Exact rows carry the expected JSON value. Program rows carry the verifier's inputs. Judged rows carry an ordered list of criteria. Author and independently validate replacements outside the code checkout, upload to the private dataset, then update the immutable revision, checksum and cell sizes in `hidden.py`. The preparation command checks repository visibility and file integrity before staging the data.

For a judge evaluation, download `authoring/closed-v2/calibration.jsonl` from the pinned private dataset revision to a private local directory. Run `python3 calibrate_judge.py --controls PATH --judge-url URL` with the prepared release. It reports agreement with labelled passing and failing answers separately for German and English.

## GPU kernel development

Generate and archive one kernel per model profile and brief. The private catalogue has four numerical workloads in German and English and a separate source contract for each backend. Each of the 16 briefs requests executable code. The GPU performs the work and a CPU oracle checks each output. A compile-only track records which submissions produce a native NVIDIA SM120 CUBIN.

```sh
python3 gpu_bench.py prepare
python3 gpu_bench.py generate --target cuda --model "$MODEL" --revision "$MODEL_REV" --url "$LOCAL_URL"
docker build -t weakbench-kernel-cuda:1 kernel-cuda
python3 gpu_bench.py evaluate --track cuda-compile
python3 gpu_bench.py generate --target metal --model "$MODEL" --revision "$MODEL_REV" --url "$LOCAL_URL"
python3 gpu_bench.py evaluate --track metal
```

Run `python3 gpu_bench.py evaluate --track metal --reference-control` to verify and time the fixed references on local hardware. The same option works on both CUDA tracks. These reports are labeled as controls and appear separately from model results.

The two targets use different prompts. Generate once per target. A model revision, changed generation settings or a changed prompt creates a new archive entry. The CUDA hardware run reuses the CUDA compilation submission. On a dedicated SM120 card with the NVIDIA Container Toolkit and a passing hardware control, set `WEAKBENCH_CUDA_DEVICE` to its full GPU UUID and run `python3 gpu_bench.py evaluate --track cuda-hardware`. Generation sends one request at a time. Evaluation requires no model endpoint. Use `--lang`, `--case` and `--model-key` to select saved entries for scoring. Each evaluation saves a fresh report under `~/.local/share/weakbench/gpu-reports/`. The private SQLite archive next to that directory contains the generated source and prompts.

The four workloads cover a strided contraction, a two-dimensional edge stencil, a segmented sequence reduction and a numerically stable weighted softmax. Each has eight correctness shapes and two separately chosen timing shapes. The independent oracle checks every output including both timing shapes. Only a submission passing the entire workload earns timing. For each timed shape the runner records seven paired GPU durations for the candidate and its fixed scalar reference. A speedup above one in every pair counts as faster. Compare models on the same device, backend, release and shapes. The suite reports failed submissions and missing infrastructure separately. Speed is relative to the included hand-written reference. [Metal source contract and measurement](KERNEL-METAL.md) · [CUDA C++ compiler and hardware measurement](kernel-cuda/README.md)

## Creative league

The league archives complete visible submissions and runs a round robin for each challenge. Its first private catalogue contains 24 tasks across German and English: four writing briefs, four responsive website briefs and four Three.js briefs in each language. The ordinary quality suite and its release scores are separate.

Prepare the catalogue once, then use one private SQLite database for the season. Each participant has a served model ID, immutable build revision and generation settings. Run the following commands from the WeakBench checkout, supplying the endpoints and actual build revisions of your models.

```sh
python3 arena_catalog.py prepare
db="$HOME/.local/share/weakbench/creative-league.sqlite3"
python3 arena.py run --db "$db" --model "$MODEL_A" --revision "$BUILD_A" --url "$URL_A"
python3 arena.py run --db "$db" --model "$MODEL_B" --revision "$BUILD_B" --url "$URL_B" \
  --judge "$VISION_JUDGE" --judge-revision "$JUDGE_BUILD" --judge-url "$JUDGE_URL"
python3 arena.py inspect --db "$db"
```

Add a third model with another `run` using the same database and judge. On a completed 24-task season this adds 24 candidate responses and 48 matches. Each new match takes two judge votes. Previously completed answers and votes are reused. Registering just a new judge scores the archived answers. Changing only a rubric invalidates the judgments for that challenge and preserves its answers and images. A candidate prompt or effective model setting change creates a separate generation identity. The model revisions are supplied by the operator. Use a judge independent of the candidates and a vision-capable endpoint for image comparisons.

Each vote sees the brief, rubric and anonymous submissions. Writing is compared as text. For a website, the judge receives source and desktop/mobile PNGs. For Three.js it receives source and three camera views. The model supplies a self-contained HTML document or a JavaScript `buildScene(THREE)` function. Chromium renders in a disposable offline container, with a pinned local copy of Three.js. A confirmed artifact error is evidence for the judge. A renderer or judge outage leaves the fixture open. Previously generated images can still be judged when the renderer is offline. [Local renderer setup](arena-render/README.md)

A consistent winner earns three points and the loser earns zero. Two tie votes, or judgments that disagree after swapping the submissions, earn one point per side. Match records distinguish agreed draws from order disagreements. `standings --db "$db" --judge-key KEY` reports pending games and whether opponents have completed comparable sets of challenges. An invalid judge response can be retried. Each accepted vote is saved immediately, including the first vote of a two-vote comparison. Visible responses that reached the output limit remain eligible and retain their finish status.

The database contains private task text, rubrics, visible model answers and render images. Store it outside version control. Generation and judging share one request slot per operating-system user across all local league processes and databases. An interruption after a provider has answered but before the local transaction commits can repeat that uncommitted call. A custom local catalogue may be supplied with `--catalog FILE`, using JSONL or a JSON array of records with `id`, `kind` (`text`, `web` or `three`), `language` (`de` or `en`), `prompt` and a nonempty list of `rubric` criteria.

## Historical diagnostics

Historical public datasets and generated probes are available through an explicit diagnostic profile. These commands use the dependencies declared in `pyproject.toml`. Diagnostic runs write richer artifacts, including model responses, for analysis of historical benchmark results.

The runner retired the public `gsm8k` and `wmt24pp` tasks in v2. The closed suite now measures quantitative reasoning with `quantity_contract` and contextual translation with `context_translation`. To reproduce historical runs of the retired tasks, use runner revision `97d56842` and its data pins.

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
