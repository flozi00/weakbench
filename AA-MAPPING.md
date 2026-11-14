# AA-MAPPING.md — our suite against Artificial Analysis

Artificial Analysis's **Intelligence Index v5.0** is nine evaluations in four categories, each category
weighted 25 %:

| Category | Evaluations |
|---|---|
| Knowledge Work | AA-Briefcase v1.1 (15 %), AutomationBench-AA, GDP.pdf |
| Coding | Terminal-Bench 4.0, FrontierCode 2.0 |
| Science | HLE-Diamond, CritPt v2 (Hard Subset), Terminal-Bench-Science 0.1 |
| General | AA-Omniscience v1.1 (15 %: 10 % accuracy + 5 % non-hallucination rate) |

AA also publishes standalone evaluations that are not in the index (Math-500, MMMU-Pro, AIME 2025,
Global-MMLU-Lite, AA-LCR, GPQA Diamond, IFBench, LiveCodeBench, AA-PersonalityBench, τ²/τ³-Bench,
ITBench-AA, MLCR-AA, GDPval-AA, Harvey LAB-AA, AA-AnalystAgent, APEX-Agents-AA, AA-3D,
EnterpriseOps-Gym-AA).

This file maps every one of them onto this suite. The rule that decides the verdicts is
`RESEARCH.md:123-144`: a task here must be **scorable offline, deterministic, pairable DE/EN, and never
execute model-written code**. That filter is what makes the suite usable as an A/B gate on a dev pool;
it is also why half of AA's index is out of reach.

## The index, eval by eval

| AA evaluation | Category | Our cell | Verdict |
|---|---|---|---|
| AA-Briefcase v1.1 | Knowledge Work | — | **Blocked.** Multi-week agentic projects graded on deliverables by a rubric plus pairwise Elo. Needs a sandbox, a file-producing runtime and a judge panel. |
| AutomationBench-AA | Knowledge Work | — | **Blocked.** Simulated SaaS workflows (Zapier-style) against a live simulator. |
| GDP.pdf | Knowledge Work | `aa_lcr` | **Approximated.** Same shape — 10k-100k-token real documents, a question, judge-graded criteria. GDP.pdf adds PDF page images and per-task rubrics; `aa_lcr` is text-only and opt-in. |
| Terminal-Bench 4.0 | Coding | — | **Blocked.** Agentic terminal work in a container. |
| FrontierCode 2.0 | Coding | — | **Blocked.** Not public (Cognition). |
| HLE-Diamond | Science | — | **Blocked.** `cais/hle-diamond` is gated: a maintainer has to accept the licence on Hugging Face before any token can read it. 1,000 questions, 272 with images, LLM-equality judge. |
| CritPt v2 (Hard Subset) | Science | — | **Excluded by design.** `CritPt-Benchmark/CritPt` is public (Apache-2.0, one parquet) but every answer is a Python function (`answer_code`) that has to be *executed* to compare against the model's code — the same rule that excludes the agentic evals. |
| Terminal-Bench-Science 0.1 | Science | — | **Blocked.** Container. |
| AA-Omniscience v1.1 | General | `aa_omniscience` | **Mirrored.** Same public dataset, same four-way grade (CORRECT / INCORRECT / PARTIAL / NOT_ATTEMPTED), and both index components are reported: `accuracy` is AA's accuracy, the `non_hallucination` sub-metric is AA's non-hallucination rate. Opt-in, needs `--judge-url`. |

**What the index measures that we do not:** Knowledge Work and Coding are 50 % of AA's score and both
are agentic — a model driving a terminal, a browser or a SaaS API and producing a file. We measure 0 %
of that. Everything we do measure is single-turn, offline-scorable capability.

## Standalone AA evaluations

| AA evaluation | Our cell | Verdict |
|---|---|---|
| Math-500 | `math_500` | **Mirrored** (added 2026-10-31). 500 problems, 7 subjects, 5 levels, LaTeX answers. |
| MMMU-Pro | `mmmu_pro` | **Mirrored** (added 2026-10-31). `standard (10 options)` config, 1,730 questions, 30 disciplines, 1-7 images each. Opt-in: 682 MB of images. |
| AIME 2025 | `aime25` | **Mirrored.** 30 problems. At ceiling on the 27B class, so it is a damage probe, not a ranking. |
| Global-MMLU-Lite | `global_mmlu_lite` | **Mirrored.** DE/EN/FR/ES/IT. |
| AA-LCR | `aa_lcr` | **Mirrored.** Opt-in, needs `--judge-url`. |
| GPQA Diamond | `gpqa` | **Mirrored.** 198 questions, DE/EN/FR/ES. |
| IFBench | `ifeval` | **Mirrored.** 214 verifiable instructions, DE/EN. |
| LiveCodeBench | `lcb_output` | **Mirrored** (output prediction, not full generation — no execution). |
| AA-PersonalityBench | — | **Out of scope.** Communication style, not capability; it cannot gate a kernel change. |
| τ²-Bench, τ³-Banking | `bfcl` | **Approximated.** `bfcl` v3 scores tool-call correctness against a schema, DE/EN. τ²/τ³ run a multi-turn conversation against a simulated user and a live database. |
| ITBench-AA | — | **Blocked.** Kubernetes incident snapshots plus a live cluster to act on. |
| MLCR-AA | — | **Blocked.** 25k-64k-token medical documents graded by a three-model judge panel; we have one judge. |
| GDPval-AA | — | **Blocked.** Real occupational deliverables, human-graded. |
| Harvey LAB-AA v1.1 | — | **Blocked.** Legal agentic tasks, private. |
| AA-AnalystAgent | — | **Blocked.** Agentic data analysis with a code runtime. |
| APEX-Agents-AA | — | **Blocked.** Agentic, private (Mercor). |
| AA-3D | — | **Blocked.** Interactive 3D scene generation, needs a renderer. |
| EnterpriseOps-Gym-AA | — | **Blocked.** Simulated enterprise systems (ServiceNow). |

## What the suite covers that AA does not

AA's index has no German, no translation, no refusal behaviour, no long-context needle test, no
multi-language code comprehension, and no numeric drift anchor. Those are the cells that catch a
kernel or quantisation regression on *our* pools:

| Cell | Why it exists |
|---|---|
| `mgsm`, `global_mmlu_lite`, `belebele`, `truthfulqa`, `gpqa`, `mmlu_prox_lite` (DE) | German is a first-class language here, not a translation afterthought. |
| `wmt24pp` | Translation quality, chrF, DE↔EN. |
| `xstest` | Over-refusal and under-refusal, reported apart. |
| `ruler` | Needle-in-a-haystack at the served context length, DE/EN. |
| `cruxeval_x` | Code comprehension in 8 languages (Python, Rust, Java, C++, TypeScript, Go, Ruby, Shell). |
| `drift` | Perplexity on a fixed corpus — the only cell that sees a numerical change before it shows up as a wrong answer. |
| `musr` | Multi-step planning and constraint satisfaction. |
| `ifstruct` | Structured-output conformance. |
| `image` | Proves the vision path works at all (synthetic pictures, keyword-scored). |

## The honest summary

- **Mirrored exactly:** AA-Omniscience v1.1, Math-500, MMMU-Pro, AIME 2025, Global-MMLU-Lite, AA-LCR,
  GPQA Diamond, IFBench, LiveCodeBench.
- **Approximated:** GDP.pdf (by `aa_lcr`), τ²/τ³-Bench (by `bfcl`), AA-PersonalityBench (by `eqbench_cw`'s
  rubric shape — a judge over free text, not the same criteria).
- **Closed with our own task:** the capability areas AA covers through agentic evals are covered where they
  can be scored offline — `research` for search-then-answer, `planning` for multi-step action sequences,
  `multiturn` for conversation state, `factuality` for hallucination rate in DE and EN.
- **Excluded by design:** every evaluation that executes model-written code — AA-Briefcase,
  AutomationBench-AA, Terminal-Bench 4.0, Terminal-Bench-Science 0.1, CritPt v2, ITBench-AA,
  AA-AnalystAgent, APEX-Agents-AA, AA-3D, EnterpriseOps-Gym-AA — plus the private ones (FrontierCode 2.0,
  Harvey LAB-AA, GDPval-AA) and the gated one (HLE-Diamond, blocked on accepting its licence).

The exclusion is one rule, not ten tasks: the suite runs from an operator machine against prod-adjacent
pools, so it never executes code a model wrote. A sandbox runtime (a pod with a shell, a filesystem and no
network) would lift it for Terminal-Bench, CritPt, AA-Briefcase and AutomationBench at once;
`benchmarks/agentic.py` was dropped 2026-10-15 for exactly this reason and is restorable from `653b8cc`.
