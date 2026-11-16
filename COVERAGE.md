# COVERAGE.md — what the suite measures, and what it does not

The A/B gate answers one question: *did this build of the same model lose quality or speed?* That needs tasks that
are **scorable offline, deterministic, pairable DE/EN, and never execute model-written code**. This file maps the
suite against the capability areas a model is judged on, so a gap is a decision and not an oversight.

Read `README.md` for the protocol, the gate rules and the per-task data. This file is the map.
`GAP-CLOSURE.md` is the work log behind the rows marked *closed*; `AA-MAPPING.md` is the same map eval by eval
against Artificial Analysis's Intelligence Index v5.0.

## Covered

| Area | Task(s) | Langs | Scoring | Notes |
|---|---|---|---|---|
| Knowledge / reasoning | `mmlu_prox_lite`, `global_mmlu_lite` | en,de,fr,es,it | MC letter | human-translated in `global_mmlu_lite` |
| Reading comprehension | `belebele` | en,de,fr,es,it | MC letter | FLORES passages |
| Science / graduate reasoning | `gpqa` | en,de,fr,es | MC letter, thinking | 198 items, DE translation |
| Math (multilingual) | `mgsm` | en,de,fr,es | final number | DE number format aware |
| Math (grade school) | `gsm8k` | en | final number | |
| Math (competition) | `aime25` | en | `#### n` | **at ceiling** on the 27B class: a regression probe, not headroom |
| Math (competition, headroom) | `math_500` | en | normalised LaTeX, else numeric | 500 problems, 7 subjects, 5 levels — the band between `gsm8k` and `aime25` |
| Multi-step reasoning | `musr` | en | option text | murder mystery, object placement, team allocation |
| **Planning** | `planning` | en,de | plan simulation | 3 STRIPS domains, optimal length from BFS, `len_ratio` |
| Hallucination / misconception | `truthfulqa` (MC1) | en,de | MC letter | deterministic stand-in for a hallucination rate |
| **Hallucination rate (factuality)** | `factuality` | en,de | CORRECT / ABSTAIN / INCORRECT | SimpleQA-Verified, item-paired DE/EN, 984 each |
| **Hallucination rate, human-written German** | `factuality_de` | de | same protocol + numeric family | scrutineer, 1,928 hand-written items, no MT |
| Hallucination rate (judge) | `aa_omniscience` | en | judge: +1 / −1 / 0 | opt-in, needs `--judge-url` |
| **Calibration of reasoning** | `calib_math` | en,de | 1 − Brier on generated math | knows when its own derivation is shaky |
| **Calibration** | `calibration` | en,de | 1 − Brier, over-/underconfidence | does the model know what it knows |
| **Unanswerable questions** | `math_missing` | en,de | answer vs "cannot be determined" | half the items lack a needed quantity |
| **Self-correction** | `self_correct` | en,de | corrected vs carried value | an injected wrong assistant turn |
| **Sycophancy** | `sycophancy` | en,de | held / caved under a user's wrong pushback | knowledge items, second turn insists on a wrong letter |
| **Prompt-injection robustness** | `injection` | en,de | hijack rate + answer EM | indirect injection in a document, paired with `grounded_qa` |
| **Fairness** | `fairness` | en,de | says equal + pick rate by group | name-swap counterfactuals |
| Refusals (over- and under-) | `xstest` | en | answered-safe / refused-unsafe | one cell, both directions |
| **Refusals, German** | `xstest_de` | de | same classifier | pinned MT artifact, `data/xstest_de.jsonl` |
| **Instruction following, generated** | `if_gen` | en,de | verifiable constraints | fresh prompt per id |
| Instruction following | `ifeval` | en,de | prompt-level strict | BenchMAX DE adaptation |
| **System-prompt adherence under pressure** | `sys_adherence` | en,de | rules checked every turn | generated |
| **Dialogue state with corrections** | `dialog_state` | en,de | count + name set | generated, real chat turns |
| **Multi-turn** | `multiturn` | en,de | strict + override | 3 turns, `forgot` / `stale` sub-metrics |
| Structured output | `ifstruct` | en | JSON/YAML schema validator | |
| **Structured output, German** | `ifstruct_de` | de | same validator | item-paired by `doc_id` |
| **Extraction from business email** | `extract` | en,de | exact fields | generated, distractor amounts and dates |
| **Structured output, German + content** | `german_json` | de | schema + content F1 | 202 real German texts, gold is the annotation |
| Tool calling | `bfcl` v3 | en,de | AST checker | simple/multiple/parallel/irrelevance |
| **Agentic errands (stateful tools)** | `tool_env` | en,de | final state + side effects | simulated world, nothing executed |
| Translation | `wmt24pp` | en↔de | chrF | |
| **Summarization** | `summarize` | en,de | chrF | MLSUM DE + XSum EN |
| **Grounded QA** | `grounded_qa` | en,de | EM + token F1 | GermanQuAD + SQuAD v1.1 |
| **Context vs world knowledge** | `context_conflict` | en,de | exact value | counterfactual reference card, "by the text" vs "in reality" |
| **Citation verification** | `citation_check` | en,de | find the unsupported citation | generated, paraphrased claims |
| **Research with citations** | `research` | en,de | span EM + citation | 5 passages, 1 gold, 4 distractors |
| **Search with a tool loop** | `search_rag` | en | answer + insufficiency | MultiHop-RAG, BM25 `search` tool, 4 rounds, 301 null queries |
| **Creative writing** | `eqbench_cw` | en | rubric judge | opt-in, needs `--judge-url`; EN only (no upstream licence) |
| **Generalization (contamination-free)** | `rule_induction`, `invented_ops`, `mini_language`, `theory_of_mind`, `temporal`, `state_tracking`, `math_variants`, `code_trace`, `spatial`, `logic_grid`, `sci_calc`, `table_qa`, `seq_induction`, `error_spot`, `schedule`, `syllogism`, `graph_path`, `probability`, `event_order`, `mini_sudoku`, `code_bug`, `timezones`, `kinship`, `game_theory`, `rulebook`, `knights_knaves`, `number_bases`, `copycat`, `gears`, `jugs`, `linalg` | en,de | exact | generated from a seed at load time: no item exists in any training set, so benchmaxxing cannot help |
| **Mathematical proofs (judge)** | `proofs` | en,de | VALID / GAP / INVALID vs a reference proof | opt-in |
| **Open-ended assistance** | `open_checklist` | en,de | judge checklist (YES/NO per criterion) | opt-in; false premises, ambiguity, trade-offs, counterfactuals, critique, estimates |
| **Translation quality (MQM judge)** | `wmt_mqm` | en↔de | error annotation with verified quotes | opt-in; complements chrF |
| **Data-to-text faithfulness** | `data2text` | en,de | coverage + invented numbers | deterministic |
| **Summary faithfulness** | `summary_faithful` | en,de | judge per-sentence support | opt-in; hallucination in generation, where chrF only measures overlap |
| **Code-switched input** | `mgsm_mixed` | en,de | final number + reply language | sentences alternate DE/EN inside one problem |
| **Language switching, 5 languages** | `polyglot_switch` | de,en,fr,es,it | langdetect | all 20 directions |
| **Language switching** | `lang_switch` | en,de | target-language share per reply and per sentence | system prompt / earlier turn / source note in the other language |
| **Grammaticality** | `grammar_pairs` | en,de | A/B minimal pair | MultiBLiMP agreement errors, with attractor sub-metric |
| **Proofreading** | `spell_fix` | en,de | injected errors fixed + over-editing | deterministic, real professional text |
| **Plain language / readability control** | `simplify` | en,de | reading ease + key facts kept | deterministic; officialese to plain language |
| **Pronoun reference** | `coref` | en,de | twin sentences | own, not public |
| **Irony & indirect speech** | `pragmatics` | en,de | MC + literal-reading rate | own scenes |
| **Idioms (pragmatics)** | `idioms` | en,de | MC letter + literal-reading rate | own sets, not public |
| **German sentence production** | `de_produce` | de | pattern checks on a free sentence | generated constraints |
| **German verb forms** | `de_verbs` | de | exact | generated |
| **German case & declension** | `de_case` | de | exact article + adjective | generated |
| **German letter conventions** | `de_letter` | de | DIN 5008 form checks | deterministic |
| **Register (du/Sie)** | `register_shift` | de | pronoun forms + facts | deterministic |
| **Language quality (grammar, spelling, idiom, register)** | `lang_quality` | en,de | judge error list with verified quotes + fluency | opt-in, needs `--judge-url`; own prompts, licence-free |
| **Cross-lingual long-context retrieval** | `ruler_xl` | en,de | both needles | text and question in different languages |
| **Long-context reasoning** | `ruler_reason` | en,de | variable chain / aggregation | multi-hop and sum over the whole haystack, not lookup |
| Long-context retrieval | `ruler` | en,de | both needles | 8k–128k |
| Long-context reasoning | `aa_lcr` | en | judge | opt-in, needs ~150k `max_model_len` |
| Code reasoning (no execution) | `cruxeval_o`, `lcb_output` | en | literal equality | Python |
| Code reasoning, 8 languages | `cruxeval_x` | en | normalised literal / atoms | Python, Rust, Java, C++, TypeScript, Go, Ruby, Shell |
| **Code reasoning, German instruction** | `cruxeval_x_de` | de | same scorer | item-paired by id |
| Code generation (executed) | `humaneval_plus`, `mbpp_plus` | en | EvalPlus tests | opt-in, runs model code in a subprocess |
| Vision | `image` | en,de | keyword | `--image` |
| **Chart / spatial vision, generated** | `chart_read` | en,de | exact | opt-in; contamination-free PNGs |
| Multimodal reasoning | `mmmu_pro` | en | MC letter A–J | opt-in, 682 MB of images, 30 disciplines |
| Numerics | `drift` | en,de,code | KL / top-1 / ppl ratio | prefill kernels only |
| Throughput / latency | `speed.py` | en,de | tok/s, TTFT, ITL, e2e | 8 scenarios × c1/8/32, incl. long-form 2048-token decode |
| Speech | `asr.py` | de | WER | German, real audio |

## Measuring general ability, not benchmark tuning (2026-11-14)

The direction: measure general AI, not hype and benchmaxxing. Static public test sets reward whoever
trained closest to them, so the suite now leans on four kinds of evidence that tuning cannot buy:

1. **Generated per seed** (`tasks_novel.py`): every item is built at load time from a seeded generator — new
   string rules, invented operators, an invented language, false-belief stories, calendar arithmetic, long swap
   chains, word-problem templates with a no-op distractor. No item exists anywhere to memorise; difficulty
   knobs (`ok_d1..3`, `ok_n10/20/30`, `ok_noop`) show where a model's reasoning breaks.
2. **Consistency instead of a single score**: `xl_correct_agree` / `xl_pred_agree` (the same item in DE and EN),
   `pairwise.py --xl` (the judge: do the DE and EN free-text replies agree in substance),
   `sycophancy` (the same answer under a user's wrong pushback), `calibration` (confidence that matches
   accuracy), `injection` (the same answer with a hostile document), `math_variants` noop vs base.
3. **Judge with structure, not a vibe score**: checklists answered YES/NO per criterion (`open_checklist`),
   error lists whose quotes must exist in the reply (`lang_quality`), per-sentence support labels
   (`summary_faithful`), pairwise in both orders (`pairwise.py`), four canaries every judge must pass first, and
   `pairwise.py --audit` for the judge's own position consistency and length bias, and `--rejudge` for agreement
   between two judges on the same replies.
4. **Language as a first-class skill**: reply-language control under pull (`lang_switch`, `mgsm_mixed`),
   grammaticality (`grammar_pairs`), proofreading without over-editing (`spell_fix`), plain-language rewriting
   with fact retention (`simplify`), native-editor error rates and fluency (`lang_quality`).

The board's **General index** averages only the categories built this way (generalization, open, truthfulness,
robustness). Smoke run 2026-11-14 on a 4B model (`--limit 5`, a 27B judge, run
`20261114T135901Z-quality-smoke-small`): 64 cells, 320 items, 0 errors, all judge canaries passed; every task scored
between floor and ceiling except `ruler_reason` (0 %) and `mini_language`/de (0 %), and the replies show genuine
failures (a chain not followed, a wrong sum, swapped word order), not parse errors. Ceilings on that model already:
`state_tracking`, `spatial`, `math_variants`, `logic_grid`, `context_conflict`, `fairness`, `idioms` — read
their difficulty knobs (`ok_n20`, `ok_h4`, ...) on stronger models.

Second smoke run on a 4B model (2026-11-14, `--limit 3`, run `20261114T164222Z-quality-smoke2-small`):
46 newer tasks, 261 items, 0 errors, judge canaries passed. Two scorer artefacts found and fixed (`data2text` now
asks for digits — the model wrote "fünfundfünfzig"; `en_produce` accepts "had been ..."); the remaining zeros are genuine
(`sys_adherence` replies grow too long under pressure, `event_order` misplaced an event, `en_produce` ignores the
topic word). At n = 3 these were at 100 % on that model and are ceiling suspects for the first full run: `kinship`,
`game_theory`, `syllogism`, `schedule`, `rulebook`, `knights_knaves`, `table_qa`, `seq_induction`, `error_spot`,
`code_bug`, `linalg`, `extract`, `calib_math` — `quality.py` names every cell at >= 95 % after a run.

**Search vs. given steps (2026-11-16, `--tasks search_state`).** On LFM2.5-24B-A2B (quick tier, 24 items per cell)
the same puzzle splits cleanly by whether the steps are given or must be found. Given steps are near ceiling:
`sliding_puzzle` 0.92-1.00, `toy_machine` 0.88-0.92, `diff_apply` 0.96-1.00, `lru_cache` 0.92-1.00. Searching for the
steps, or folding space mentally, falls to chance: `puzzle8_min` 0.13-0.17 (chance 0.17), `river_crossing` 0.25-0.33,
`cube_net` 0.29-0.33 (bent opposites 0.08-0.17), `hanoi_state` 0.29-0.46, `paper_punch` 1 fold 0.88-1.00, 2-3 folds
0.00-0.13. Most failures run to the 12k-token cap without an answer. These tasks keep headroom that a static set
loses once trained on, and every gold is re-solved from the prompt text by an independent solver in
`test_search_tasks.py`.

**Next candidates** (not built yet): argument/debate quality by pairwise judge, and a German HLE-style hard set
once someone with access unlocks `cais/hle-diamond`.

## Not covered, and why

| Area | Status | What it would take |
|---|---|---|
| **German creative writing** | closed differently | EQ-Bench's prompts carry no licence, so `de_creative` uses our own prompt set instead: 12 German story prompts with hard constraints (perspective, tense, motif, situation, length) and a judge rubric for correct idiomatic German, vivid language and a coherent arc. It has no reference scores; it is for the paired A/B. |
| **Web search / browsing** | out of scope | BrowseComp, GAIA, FRAMES all need a live browser and a sandbox. `research` is the offline analogue (find the evidence among distractors and cite it) and `search_rag` is the tool-loop analogue (formulate a query, search, search again, or say the evidence is not there). Neither touches the live web. |
| **Agentic / engineering** | excluded by design | Terminal-Bench, SWE-bench, SciCode, GDPval, AA-Briefcase need per-task containers **and** execute model-written code, which the filter above forbids — the suite runs from an operator machine against prod-adjacent pools. The runtime is no longer the wall: `brew install container` gives Apple's `container`, which Harbor 0.24 supports, so `benchmarks/agentic.py` is restorable (`git checkout 653b8cc -- benchmarks/agentic.py benchmarks/harbor_amd64.py benchmarks/test_agentic.py`). What blocks it is the rule, and that no dev pool was reachable during the survey. |
| **PDDL planning (PlanBench)** | closed differently | PlanBench needs VAL. `planning` validates plans by simulating them against domains we wrote, so the semantics are ours and no external validator is needed. |
| **HLE** | blocked on a decision | `cais/hle-diamond` is gated: a maintainer has to accept the licence on Hugging Face before a token can read it. 1,000 questions, 272 with images, LLM-equality judge. |

## Adding a task

1. Put it in the module that owns the area (`tasks_code.py`, `tasks_reasoning.py`, `tasks_safety.py`, `tasks_if.py`,
   `tasks_text.py`, `tasks_plan.py`, `tasks_tools.py`, `tasks_judge.py`, `tasks_lang.py`, `tasks_novel.py`, `tasks_open.py`), or add a module and register it in
   `quality.py`'s `TASKS`.
2. Pin the dataset revision. `load(lang, n, seed)` must be deterministic and `quick` must be a prefix of `full`.
3. `score(item, reply)` returns `{"correct", "score", "pred", "gold", "detail"}`. Prefer exact scoring; when the
   scorer is approximate (chrF, normalised literals, a judge), say so in the module docstring and in `README.md` —
   the paired A/B stays exact even when the absolute number does not.
4. Add a `test_tasks_*.py` that loads the real pinned data and asserts counts, the quick prefix, and that the gold
   scores against itself.
5. Add the README table row and the test command.
