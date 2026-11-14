# weakbench

A benchmark suite built to **find where an LLM is weak**, not to produce a headline number.

- **Generated, contamination-free tasks.** Most items are produced from a seed (`rng_of(task, i)`), so they cannot have been in the training data. Covered: rule induction, invented operators, state tracking, logic grids, scheduling, code tracing, probability and others (see `TASKS.md`).
- **Structured LLM-as-judge.** The judge works from checklists, verified quotes, per-sentence labels and MQM error spans instead of a free 1-10 score. Before a run it must pass a 4-canary preflight. `pairwise.py` compares A/B in both orders.
- **Language quality (German and English).** Grammar minimal pairs, case and verb morphology, language switching, register, idioms, pragmatics and coreference. Cross-lingual consistency is reported as `xl_*` sub-metrics: the same item in DE and EN should get the same verdict.
- **Traps.** Sycophancy, false premises, injected instructions, context conflicts, distractors and underspecified tool calls. See `TRAPS.md`.

## Quickstart

Any OpenAI-compatible chat server works:

```sh
uv run quality.py --url http://host:8000 --arm mymodel --tier quick
uv run quality.py --url ... --arm mymodel --tasks rule_induction,coref --langs de,en
uv run quality.py --dry-run --tasks all            # list items without a server
```

Judge tasks need a judge server: `--judge-url URL --judge-model ID`.

Comparing two runs, item by item and with statistics:

```sh
uv run compare.py runs/<A> runs/<B>
uv run pairwise.py --help                          # A/B judged in both orders
uv run speed.py --help                             # throughput / latency
```

Results go to `runs/<timestamp>-<arm>/` (jsonl per task plus `summary.json`).

## Datasets

- Public datasets are downloaded at pinned revisions and cached in `~/.cache/weakbench`.
- Datasets we write ourselves go to private Hugging Face dataset repos under `flozi00/`. They are read with the token from `hf auth login`, and the read fails loudly if it can't get them.

## Tests

```sh
for t in test_*.py; do uv run -q python "$t" || echo "FAIL $t"; done
```

Adding a task: implement the `TASKS` contract in one of the `tasks_*.py` files (described in `quality.py`'s docstring) and add a test next to it. The quick tier must stay a prefix of the full tier.
