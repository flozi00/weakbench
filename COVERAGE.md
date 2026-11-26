# COVERAGE.md — what the suite measures, and what it does not

The A/B gate answers one question: *did this build of the same model lose quality or speed?* That needs tasks that
are **scorable offline, deterministic, pairable DE/EN, and evaluate submitted programs only with bounded, task-specific interpreters**. This file maps the
suite against the capability areas a model is judged on, so a gap is a decision and not an oversight.

Read `README.md` for the protocol, the gate rules and the per-task data. This file is the map.
`GAP-CLOSURE.md` is the work log behind the rows marked *closed*; `AA-MAPPING.md` is the same map eval by eval
against Artificial Analysis's Intelligence Index v5.0.

## GPU kernel development coverage

The text coding families check functional programs on a CPU or by replay. GPU submissions add address layout, numerical stability, behavior at size boundaries and the time for a dispatch on a named device. The catalogue pairs English and German versions of each workload. Each target receives eight briefs. The four numerical workloads cover 40 distinct shapes and 403,989 expected output elements. A second implementation checked the frozen numerical inputs and outputs. Sixteen hand-written incorrect variants passed source compilation on the M4 and failed the output checks. Those controls exercise the checker independently of model performance.

KernelBench uses correctness and a speed threshold to evaluate generated GPU kernels. Here the runtime reports each comparison separately against a fixed reference on the same shape, with all seven measured pairs retained. The Apple path supports an explicit bounded kernel language. The CUDA path accepts native C++ with a fixed entry point and launch contract. Compilation alone supplies a distinct result. Cross-vendor ratios, multi-GPU collectives, end-to-end inference throughput and speed versus tuned vendor libraries are outside this catalogue. [KernelBench](https://github.com/ScalingIntelligence/KernelBench) · [Apple GPU timing](https://developer.apple.com/documentation/metal/mtlcommandbuffer/gpustarttime) · [NVIDIA NVRTC](https://docs.nvidia.com/cuda/nvrtc/index.html)

### Native Metal format comparison

The first three-model Apple pilot recorded 24 answers. Each missed either the JSON submission contract or the bounded Metal language before any model kernel could run. The matched native track tests which stages a model reaches when it can write complete MSL source. Within a pair only the response-format instruction changes. Changing the format can also change the program the model chooses to write. Comparing these new counts with those from the older bounded language changes multiple conditions.

Apple's runtime can compile a source string into an MTLLibrary and return pipeline reflection for a compute function. The harness checks reflected buffer bindings, numerical output, and seven time pairs after correctness. Following KernelBench's distinction between functional correctness and measured performance, it retains separate counts for extractable responses, accepted pipelines and all-shape correctness. [Apple shader libraries](https://developer.apple.com/documentation/metal/mtllibrary?language=objc) · [Apple pipeline reflection](https://developer.apple.com/documentation/metal/MTLComputePipelineReflection?language=objc) · [KernelBench](https://proceedings.mlr.press/v267/ouyang25a.html)

## Numerical anchoring and contingent planning · closed-v24

**Numerical anchoring.** Does a suggestion change how a model uses the same weighted observations? Each dossier appears in two fresh conversations in each language. An independently reconstructed optimum can be a single rational number or an interval. The suggested value may be justified, too low or too high. Old and updated records independently establish whether a real correction is warranted. A check on 72 dossiers found that a classifier using only the topic, suggested number, response range or masked wording identified 24 of 72 anchor positions, the balanced baseline. The evidence itself contains useful cues. Choosing the value with the largest weight for each estimate solved five complete dossiers. This task measures numerical decisions under explicit scoring rules. It cannot establish the model’s motive or how it handles open-ended forecasts. [Tversky and Kahneman](https://doi.org/10.1126/science.185.4157.1124) · [Gneiting](https://doi.org/10.1198/jasa.2011.r10138)

**Contingent planning.** A plan must choose actions that work with the information available when each choice is made. The scorer follows the submitted policy in each possible world and computes the minimum achievable worst-case time. A second implementation reconstructs the problem from the visible text and checks which integer time budgets admit a policy. Of 32 scenarios per language, 24 are feasible and eight are impossible by the deadline. In eight feasible scenarios, even the best fixed inspection order with early stopping is slower than an adaptive policy. These tasks measure a submitted plan under a supplied finite model. Execution by an external agent is outside the score. [Hoffmann and Brafman](https://cdn.aaai.org/ICAPS/2005/ICAPS05-008.pdf)

## Proof feedback, directed search and balanced reporting · closed-v23

**Proof feedback** varies the requested evaluation of a mathematical derivation. The verifier enumerates each permitted setting on a finite rational domain. A model must distinguish a faulty argument with a true conclusion from an argument for a false claim and identify the first failed step and relevant counterexamples. The score concerns the supplied finite domain. During authoring, a lookup on two printed parameters recovered 30 of 32 verdicts when each dossier was held out in turn. In the released design each formula layout and request direction contains all four cases, in both tiers. A lookup based on the parameters and formula shape now matches the majority class on 32 of 64 full-tier dossiers and 16 of 32 quick-tier dossiers. An independent reader checked all 256 localized presentations. Submitting the correct labels without the requested evidence passed none.

**Directed search** varies the user's desired result while preserving the same two-site record archive. The model chooses native search calls. The verifier replays the visible records and bounds the possible contribution of an unread site. A single cited report can legitimately settle the answer. An answer that echoes a request before the observed evidence settles it receives a separate diagnostic. Human studies of search-term selection motivated this experiment. [Leung and Urminsky](https://doi.org/10.1073/pnas.2408175122)

**Selective reporting** adds paired factual alternatives. In both tiers each visible reporting context now occurs equally often with a mandatory and an exempt target. Any rule that gives the same answer for an identical domain and framing classifies exactly half of those target decisions correctly. Correct reports also require the figures and the other material entries. Existing `closed-v22` archives retain the prior sampling and outcomes.

## Short reviews, false premises and selective reporting · closed-v22

In the v21 local Qwen3-4B pilot, the model supplied just two parseable JSON audits in 16 attempts. The short review family measures the same substantive decision with a short German or English verdict. It exhaustively checks a small input domain and possible configurations. Cases cover boundary conditions, filtering, ordering and dictionary access. Correct, faulty and underdetermined programs appear under both pressure to accuse and pressure to approve.

Two further families test behavior that an explicit agreement question can miss. **Presupposition repair** requires deciding from records whether a premise is established, then answering an applicable follow-up or leaving it unanswered. **Selective reporting** checks whether a report omits a material exception or promotes an immaterial item while its totals and other entries remain accurate. A permitted omission and a warranted disclosure are positive controls. The distinction between a false premise and an inapplicable follow-up builds on work on presupposition failure in QA. Research on truthful but misleading statements motivated the distinction between selecting facts and inventing them. These tasks use original bilingual materials and finite rules. [CREPE](https://aclanthology.org/2023.acl-long.583/) · [Rogers et al.](https://www.hks.harvard.edu/publications/artful-paltering-risks-and-rewards-using-truthful-statements-mislead-others)

Each new family contributes 24 pairs per language. The full release adds 288 answers. The quick tier takes half of every stratum and preserves both arms of a pair. Across all six pressure families, runs now record how many completed misleading pairs have both a correct neutral decision and an interpretable pressured response. Older summaries retain the denominators they originally observed. The full task score and eligibility together expose the effect of malformed answers and errors in the underlying work.

## People Pleaser · closed-v21

The earlier three-turn belief-revision task tests whether a model revises its conclusion after new evidence or unsupported pushback. It leaves open whether a model invents a finding in an initially clean review, mirrors a group’s conclusion, or converts a legitimate subjective preference into a false statement about merit. Matched prompts now keep the task and evidence constant and vary the user's request. Each language includes 48 audit pairs, 72 evidence pairs and 32 choice pairs. The quick tier keeps half of every stratum.

| Family | Pressure or control | Observable outcome |
|---|---|---|
| Audit | Requested finding or clearance across authorization rules, billing, data quality and workflow approvals | False accusations, clearance of a demonstrated defect, certainty when the active implementation is unknown, agreement supported by a complete audit |
| Evidence | Claimed authority, majority, conviction, praise, attribution and reassurance | Requested false verdict, unwarranted contradiction, invented source, warranted uncertainty, correct verdict supported by the decisive records |
| Choice | Authorship, prior promise, incentive and deadline | Choosing a prohibited favorite, increasing its objective score, falsely naming it a leader, honoring a permitted preference |

For every family the paired failure rate conditions on a correct neutral decision and a recognizable pressured judgment when the request is wrong or unsupported. For audits it also records switches separately on clean, defective and undetermined cases. Each language and each model has its own denominator. A zero denominator leaves the rate unmeasured. Full task scores still count missing evidence or malformed answers as failures. Each pair uses independent conversations with identical generation settings, including the seed. An observed switch describes behavior on that pair. It does not identify intent or measure the quality of empathy, politeness or open-ended advice.

Sharma and colleagues show that stated user views can influence factual judgments and feedback. SYCON Bench studies pressure over multiple turns. ELEPHANT studies preservation of the user's self-image in open-ended social responses. These studies informed the coverage review. The scored tasks here use authored scenarios, explicit decision rules and independent programmatic checks. [Sharma et al.](https://proceedings.iclr.cc/paper_files/paper/2024/file/0105f7972202c1d4fb817da9f21a9663-Paper-Conference.pdf) · [SYCON Bench](https://aclanthology.org/2025.findings-emnlp.121/) · [ELEPHANT](https://proceedings.iclr.cc/paper_files/paper/2026/hash/d3362f84979d16cee000f09eef61244c-Abstract-Conference.html)

## Private coverage review, closed-v20

The release adds 64 interactive cases about authorization, retry decisions and verifying that an action took effect. The full suite contains 4,200 text cases, 192 image cases and 64 audio cases. Each language has 16 pairs with identical opening requests. The tool responses reveal whether the model may act, must wait for approval, should retry under the original key or can confirm an existing booking.

Three scores use different denominators. Withholding an unapproved action applies to 16 cases. Recovery after an ambiguous submission applies to 24. Verified completion applies to all 48 approved cases. The duplicate-effect rate records whether a second operation actually existed. A needless retry under the original key can produce exactly one booking and still fail the retry protocol.

A second ledger agreed with the service simulator on 5,768 reachable transitions for a seven-action probe set. Two observation-based policies each passed all 64 cases. Reusing a valid history and answer from the paired case failed in all 64 directions. Changing the key after an uncertain result produced two physical bookings in 16 of 24 cases. These checks cover one simulated operator and a fixed set of delivery outcomes. Concurrent operators and real-world service failures require separate evaluation.

## Private coverage review, closed-v19

The release adds 96 time-sensitive search tasks and 64 three-turn evidence-revision conversations. Every case is available in English or German, with aligned language pairs. The full suite contains 4,136 text cases, 192 image cases and 64 audio cases.

The earlier source-inquiry task asks for an investigation from the outset. The new search task varies whether decisive information is already supplied, discoverable, contradictory or inaccessible. A direct answer can be correct in one case and unsupported in its paired case. Sixty-four requests use original fictional notices. Thirty-two use dated primary-source snapshots. The local search index is reproducible across runs. Live search-engine freshness and coverage require separate observations. These design choices draw on multi-turn relevance testing, changing facts, reproducible retrieval and citation attribution. [BFCL](https://gorilla.cs.berkeley.edu/blogs/13_bfcl_v3_multi_turn.html) · [FreshQA](https://arxiv.org/abs/2310.03214) · [CRAG](https://proceedings.neurips.cc/paper_files/paper/2024/hash/1435d2d0fca85a84d83ddcb754f58c29-Abstract-Datasets_and_Benchmarks_Track.html) · [ALCE](https://aclanthology.org/2023.emnlp-main.398/)

In the revision conversations a new message can change a fact, withdraw a source or simply urge a different answer. The model must reconsider the remaining candidates and cite evidence still active at that turn. A verifier checked 192 decision states and 6,912 subsets of displayed evidence. A confirmed final answer can follow an incorrect intermediate decision, so the report includes both endpoint accuracy and all-turn correctness.

## Private coverage review, closed-v18

The four additions contribute 256 requests across German and English. Each family has 32 distinct scenarios, arranged in 16 contrast pairs and translated into both languages. The complete release contains 3,976 text requests, 192 image requests and 64 audio requests.

| Skill | What the earlier suite covered | What the new answers demonstrate |
|---|---|---|
| Interactive writing | Imagery and narrative evidence in a supplied sentence grammar | In `reader_gamebook`, the paragraphs are free prose. The model designs the links, fulfills each reader’s promised destination and preserves earlier motif words along every path. |
| Musical planning | Reading notes and ties from a finished score | In `counterpoint_repair`, the model composes the least disruptive correction to an upper voice over a given bass. A changed bass note can require several different upper notes. |
| Software testing | Writing programs and finding one distinguishing word for two automata | In `mutation_suite`, the model designs one shared suite that detects every distinguishable variation of a program, supplies the correct outputs and identifies equivalent variations. |
| Causal planning | Numerical counterfactuals with specified mechanisms and repairs to a previous schedule | In `causal_recourse`, the model learns how a device works from experiments, then chooses one intervention that must work in every stated environment. |

**Interactive writing and language.** The independent reader found between two and 248 valid link layouts per scenario. A browser made 3,840 actual clicks in 128 reference books, completed 1,152 readings and confirmed all 528 promised routes. An early version of the scorer accepted all 64 explicitly negated destination claims and 512 embedded-word variants. The released scorer rejects them all. Each visible choice now uses one of two affirmative forms. The paragraphs remain open. Deliberately non-narrative word lists, reversed prose and filler still pass in all 64 cases when they satisfy the vocabulary and navigation contract. A language probe checks each paragraph, choice, request and API-exposed reasoning segment. In a positive alternative set, all paragraphs had identifiable target-language evidence in 44 of 64 books. The other 20 books still satisfied the writing contract. These measures cannot establish voice, fluency or literary merit. [inkle’s interactive narrative documentation](https://github.com/inkle/ink/blob/master/Documentation/WritingWithInk.md) · [Guan et al., OpenMEVA](https://aclanthology.org/2021.acl-long.500/)

**Musical planning.** Enumerating 179,400 candidate note sequences across the bilingual requests reproduced every reference optimum. An independent integer solver proved all 64 optimality bounds and identified 82 accepted lines. All 82 failed on their own contrast twin. Minimizing the number of changed notes alone left excessive pitch displacement in 28 requests. A nearest-note strategy failed to complete a valid line in 16 of the 32 distinct situations. The score concerns the voice-leading and cadence rules given in the request. The task uses symbolic pitches and does not assess listening or phrasing. [Herremans and Sörensen on computational counterpoint](https://www.tandfonline.com/doi/full/10.1080/17513472.2012.738554)

**Software testing.** Two independent interpreters and a pseudo-Boolean solver agreed on the minimum test count, minimum combined input length and observationally equivalent variants on all 64 requests. Greedy selection missed the optimum in nine of 32 scenarios, including five where it chose too many inputs. Nineteen scenarios allow substantively different optimal test sets. One needs no tests because every variation agrees with the reference on the specified domain. Eight use an empty input to expose a difference. The verifier rejected 128 adequate suites containing an unnecessary test. Equivalence is relative to the full finite input domain stated in each request. [Harrold, Gupta and Soffa on test-suite reduction](https://www.cs.virginia.edu/~soffa/Soffa_Pubs_all/Journals/Methodology.Harrold.1993.pdf)

**Causal planning.** Complete enumeration of 2,040 legal interventions over the bilingual requests and a second forward simulation confirmed the costs and attainable targets. Nine of 32 scenarios are impossible, three achieve the target without intervention, and 20 require a paid intervention. The price and coverage diagnostics apply to the 23 feasible scenarios in each language. Treating an intervention as an observation falsely legitimates 46 plans across the 32 scenarios. Allowing a different plan per environment changes the optimum or feasibility in 11. Results concern the stated synthetic device and its finite set of environments. [Karimi, Schölkopf and Valera on interventions](https://arxiv.org/abs/2002.06278) · [Dominguez-Olmedo et al. on robust recourse](https://proceedings.mlr.press/v162/dominguez-olmedo22a.html)

## Private coverage review, closed-v17

The four additions contribute 256 new requests across German and English. Each family contains 16 paired situations in each language. The complete release has 3,720 text requests, including the four existing rubric-graded families, plus 192 private image and 64 private audio requests.

**Writing with conflicting beliefs.** In `backstage_reveal`, a writer chooses how a prop moves and how a mistaken character learns its whereabouts. Each character can act only on the evidence available in the story. Half the briefs require two endings with a common opening, different initial kinds of evidence and different intermediate destinations. Exhaustive forward and backward search agreed on all 1,932 narrative solutions across the bilingual collection. The checker accepted 3,864 prose realizations and rejected all 3,864 on the corresponding twin with one change to what a character could see. The task measures narrative choice and causal coherence within a specified sentence grammar. Independent evaluation of open-ended literary style remains a separate question. [Riedl and Young on narrative planning](https://doi.org/10.1613/jair.2989) · [Le et al. on evaluating models of others’ beliefs](https://aclanthology.org/D19-1598/)

**Evidence across time.** `rule_time_slice` asks what a fictional gallery could decide using the records filed at a past date, then asks how the decision about those same days changes under a later review. The two timelines must cite the decisive versions. A new source can replace an old one without changing the decision itself. Three independent implementations produced identical results on the 64 requests and on 551 additional small histories. Of the 32 distinct histories, ignoring filing time fails in 32, ignoring the effective interval fails in 31, and treating a change of source as a change of verdict fails in 16. Twenty-two histories include an unresolved priority tie. The rule hierarchy belongs to the fictional exercise. [Snodgrass, The Temporal Query Language TQuel](https://www2.cs.arizona.edu/~rts/pubs/TODS87.pdf)

**Data queries.** `bag_query` asks for one program that produces the prescribed report even when records repeat, a join has no match or a field is NULL. The domain consists of precisely 1,540 small pairs of tables. A bounded interpreter compares the submitted query's result multisets with independently computed SQLite results on every member of that domain. The four overlapping subsets with duplicates, an empty table, NULL fields or an unmatched left key supply their own diagnostics. Independent checks accepted 128 correct programs, rejected all 128 programs for the opposite contrast and retained 43 syntactically altered but equivalent programs in a mutation study. Removing duplicates would change 31 of the 32 report specifications on at least one database. Equivalence is established within the stated finite domain. [SQLite SELECT semantics](https://www.sqlite.org/lang_select.html) · [SQLite aggregate functions](https://www.sqlite.org/lang_aggfunc.html)

**Mathematical reconstruction.** `distance_repair` asks how many pairwise distance readings must change, which layouts can realize that minimum, and which adjacent gaps every optimum shares. Four of the 32 distinct situations admit different layouts even when every measurement is exact. Twenty-two have multiple optimal layouts. In two cases no adjacent gap is guaranteed. Three implementations agreed on all 784 small multisets in a separate exhaustive check. An independent SAT formulation recovered all 73 optimal layouts across the 32 situations, treating reflections as equivalent. The leaderboard distinguishes the minimum edit count, the complete layout set and the shared gaps. [Skiena, Smith and Lemke on reconstructing sets from distances](https://researchconnect.stonybrook.edu/en/publications/reconstructing-sets-from-interpoint-distances-extended-abstract/) · [Grimm and Baake on homometric sets](https://arxiv.org/abs/0808.0094)

## Private coverage review, closed-v16

The private suite adds 320 requests across five families. Each has 16 contrast pairs in German and the same 16 in English. All five have code-based scoring. The complete profile has 3,464 text, 192 image and 64 audio requests. The text figure includes the four existing rubric-graded families.

| Family | Gap in the earlier closed suite | What earns credit |
|---|---|---|
| `imagery_chain` | Sentence fusion tests meaning preservation. It does not ask for two different imaginative treatments of a brief. | Compose a connected series of comparisons that share a licensed sensory quality, cross image domains, vary syntax and satisfy the grammatical forms of the target language. Half of the requests require a second poem with a different intermediate image. |
| `source_inquiry` | Archive retrieval links records presented in the request. A handover can cite supplied material without searching. | Search the local collection, open the current independent originals, quote the relevant line in each, classify its stance and reach the conclusion warranted by the task's explicit evidence rule. |
| `halfplane_proof` | Propositional cores and network cuts test other kinds of proof. | Supply an exact rational counterexample, an inconsistency certificate, or both a feasible point and a proof that the proposed bound follows. Every submitted certificate is checked algebraically. |
| `transit_exposures` | Earlier signal questions give instantaneous samples. Earlier adaptive measurements supply the observation table. | Integrate the light over each exposure, identify all compatible periodic models, then select every follow-up window that minimizes remaining ambiguity in the worst case. |
| `comparator_fabric` | Program tracing asks for a result and scheduling accepts one timetable. Neither requires a single parallel program to meet an output specification on every input. | Schedule directed compare-exchanges on permitted connections within a shared budget. Each required output must have the requested rank for all values, including ties. |

### Verification and scope

* **Writing.** An independent parser reconstructed the briefs in both languages. Two implementations found 6,852 combinations of image path and sensory properties across the bilingual requests. The scorer accepted 181,768 licensed individual formulations, using valid companion drafts where required. All 32 two-draft tests rejected a second draft that merely rephrased the same image path. This measures variation and coherence within a finite poetic form. Literary taste, voice and open-ended originality require additional evidence. [Eval4NLP study of poetry evaluation](https://aclanthology.org/2025.eval4nlp-1.9/)
* **Research and factuality.** Both sides of each pair receive the same question and the same search index. Changing one line in a retrieved source flips the conclusion in all 16 pairs. Within each language there are eight supported, eight refuted, eight conflicting and eight unresolved cases. A correct guess without a retrieval trace failed all 64 cases. All 448 wrong-line quotations from the right document and all 224 fabricated quotations failed. Separate checks distinguish a fabricated ID, a real quote assigned to the wrong source, and an authentic but superseded source. The conflict and insufficient-evidence checks each have eight eligible cases per language. The comparisons use a stable, authored archive and a specified corroboration rule. The measured skill is gathering and attributing evidence within that archive. [FEVER](https://aclanthology.org/N18-1074/) · [AVeriTeC](https://papers.neurips.cc/paper_files/paper/2023/hash/cd86a30526cd1aff61d6f89f107634e4-Abstract-Datasets_and_Benchmarks.html)
* **Mathematics and logic.** Exact vertex enumeration, strict Fourier–Motzkin elimination and an independent Z3 linear-real solver agreed on 22 entailments, 22 refutations and 20 inconsistent systems. Treating equality as a violation would misclassify all 22 entailed cases. The scorer accepted distinct certificates on 54 cases. The ten others have a unique counterexample point. [Boyd and Vandenberghe, Convex Optimization](https://web.stanford.edu/~boyd/cvxbook/)
* **Observation planning.** Interval intersections, unit-time integration and a circular phase-mask implementation agreed on all 64 cases. In 27 of the 32 underlying observation histories several models fit. A follow-up observation improves the worst-case ambiguity for 25 of those 27. Replacing a whole exposure by its midpoint yields an incorrect answer in 30 of 32 histories. The idealized signal has a rectangular dip, exact photometry and a fixed list of candidate periods. [Kipping, finite integration time](https://academic.oup.com/mnras/article/408/3/1758/1075347) · [Kovács, Zucker and Mazeh, box-shaped transits](https://www.aanda.org/articles/aa/pdf/2002/31/aa2422.pdf)
* **Coding and planning.** A binary threshold commutes with each comparison and each requested order statistic. Enumerating all binary inputs therefore tests the specified ranks for every integer input. An independent solver found legal programs in all 32 underlying situations. Its programs also passed 6,960 permutation checks and 8,640 checks with repeated values. Of 480 rejected changes that reverse one comparator, 20 fail on just one binary input. [Bundala et al., optimal-depth sorting networks](https://arxiv.org/abs/1412.5302)

### Local model pilot

A preregistered Mistral-7B-Instruct-v0.3 Q4_K_M pilot used one local slot, temperature zero, a fixed diagnostic seed namespace and a 768-token cap. Each of the two families contributed four held-out requests per language and one simple control per language. None of the eight poetry responses passed the specified grammar. All eight ended normally. All eight held-out program responses hit the token cap and lacked a complete extractable program. The four controls ended normally and received zero full passes. These single-model observations are separate from the release scores. [Aggregate pilot record, private](https://huggingface.co/datasets/flozi00/weakbench-data/blob/e1b5dd8d3b778f76923ffae4dbfbfb149dac1aa4/reports/closed-v16/mistral-7b-instruct-v03-q4km-poetry-and-programs.json)

A preregistered follow-up crossed output caps of 768 and 3,072 tokens with an optional generic response grammar on the same eight held-out programming requests and two controls. It issued 40 requests to a single local model slot. Thirty-six returned completions and four had request errors. None of the 36 completions passed the program checker. Among the eight held-out requests in each arm, 0/8 and 0/6 completed responses were strict JSON without the grammar at the two caps. With the grammar, the corresponding figures were 3/8 and 4/7. The four baseline aggregates matched the earlier run on correctness and termination. The extraction diagnostic differed on one German item. These results describe this quantized checkpoint under the recorded settings. [Matched cap and format pilot, private](https://huggingface.co/datasets/flozi00/weakbench-data/blob/5f973674b07a521538f2b941497f14cc122ce170/reports/closed-v16/mistral-7b-instruct-v03-q4km-comparator-cap-and-format.json)

## Private coverage review, closed-v15

**Target-language minutes with an exact foreign quotation.** The earlier language tasks test the language of visible statements and any reasoning supplied by the API. Here the model must resolve a museum council vote, count delegations that end in abstentions or cycles, apply the quorum and cite the correct rule from four foreign-language originals. The requested language governs both report sentences. Only an exact source line can be exempted from the request-language probe. Exactly one identified quotation belongs in the response. Three solvers agreed on the ballots. Across the 64 localized cases, the grader accepted 256 permitted formulations and rejected all 320 altered quotations, 192 incorrect source IDs and 384 attempts to introduce foreign prose through purported quotation markup. [Christoff and Grossi on delegated voting](https://arxiv.org/abs/1707.08741)

The report sentences have a finite grammar. Freely worded native reasoning has weaker evidence. On an independent, hand-annotated bilingual corpus, an experimental lexical detector identified 218 of 751 short embedded foreign spans and abstained on the other 533. In an exploratory wider-window check it incorrectly confirmed two mixed windows as monolingual. The detector experiment is separate from scoring. An absent native channel leaves the combined measure unverified. [Denglisch corpus](https://aclanthology.org/2023.sigtyp-1.5/) and [held-out language audit](https://huggingface.co/datasets/flozi00/weakbench-data/blob/8225e380945454865326f7bcef72215b0442ac03/experiments/quoted-language-v2/README.md)

## Private coverage review, closed-v14

**Reliability with competing failures.** Earlier `censored_bounds` questions bound missing binary outcomes. These fleet histories ask for the number of machines at risk and the probability of each of three states on two dates. Machines can enter observation late, leave observation without a recorded failure, or fail in either of two ways. On 29/32 histories, calculating each cause separately as one minus Kaplan–Meier changes at least one answer. Two exact solvers agreed on the questions and 675 additional synthetic histories. R survival reproduced all 64 risk-set sizes and 192 state probabilities to within 1.7 × 10⁻¹⁶. Its multistate call handled 31 original histories directly. For the fully censored history the single-state fit confirmed survival at one, with neither failure observed. [Aalen and Johansen](https://www.jstor.org/stable/4615704) and [R survival interval convention](https://stat.ethz.ch/R-manual/R-devel/library/survival/html/Surv.html)

**Speech prototype held back.** An acoustic template matcher solved all 64 bilingual announcements about revised pickup instructions. On the same synthetic eSpeak signals, an independent Whisper base recognizer recovered the complete spoken number sequence on 0/32 German recordings and 17/32 English recordings. Its normalized word error counts were 783/1,016 in German and 332/904 in English. These recordings are excluded from the scored catalog. [Redacted intelligibility study](https://huggingface.co/datasets/flozi00/weakbench-data/blob/0610f605efcc5f072a8cf79e8d0ea6734a5d4d4f/experiments/spoken-revision-espeak-v1/README.md)

Two subsequent experiments used natural-sounding synthetic voices with the same statements. The normal-speed recording recovered both the ordered numbers and group names on 29/32 German and 28/32 English trials. It missed six of the eight English withdrawal markers. A slower rendition passed those original checks and missed the separately declared short-control threshold in both languages. Neither experimental condition contributes to a rank. [Natural-voice comparison and supplementary controls](https://huggingface.co/datasets/flozi00/weakbench-data/blob/8225e380945454865326f7bcef72215b0442ac03/experiments/spoken-natural-v1-v2/README.md)

## Private coverage review, closed-v13

**Sentence fusion.** Translation and proofreading tasks already test language production. This family asks for one sentence that preserves both source events and their temporal, causal or additive relationship. A parser checks who did what to whom, negation and the direction of the relationship. An independent reader reconstructed 118 accepted German and 158 accepted English formulations directly from the questions. The scorer rejected 2,358 grammatically valid changes of meaning and all 276 canonical answers in the other language. Results measure the controlled grammar explained in each question. [DiscoFuse](https://aclanthology.org/N19-1348/)

On 24 matched trials with the released v13 runner, local Mistral-7B-Instruct-v0.3 Q4_K_M met the sentence contract once in English and zero times in German. In one other English answer the events were preserved and their temporal order was reversed. The remaining 22 answers had undetermined meaning under the two offline checks. Adding two synthetic format examples to each request produced 0/24 accepted answers. All trials ended normally in both arms. [Original pilot](https://huggingface.co/datasets/flozi00/weakbench-data/blob/efd68732e29a91df55a221f4080cd3fac9ff759f/reports/closed-v13/mistral-7b-instruct-v03-q4km-sentence_fusion.json) and [matched demonstration study](https://huggingface.co/datasets/flozi00/weakbench-data/blob/656eaa06965f3c5095be5c5a6c632c3041f95703/reports/closed-v13/mistral-7b-instruct-v03-q4km-sentence_fusion-demonstrations.json)

**Selecting an actionable point.** The existing image families measure chart reading and spatial inference. The new panel images ask for an integer pixel on a control selected by its row, relative position and action. Any exposed active pixel is valid. Of the 32 scenes, 28 offer a usable target. Even with the correct button identified, its geometric center only works on 16 of those 28. An independent PNG reader recovered all 64 answers. Chromium checked all 2,772,480 pixel locations across the 32 layouts and confirmed the accepted regions. The scenes use synthetic button geometry and fixed scale. [SeeClick](https://aclanthology.org/2024.acl-long.505/) and [CSSOM View hit testing](https://www.w3.org/TR/cssom-view-1/)

## Private coverage review, closed-v12

| Earlier measurement | New measurement |
|---|---|
| `auction_price` asks what the winner pays for one good | `package_auction` asks which competing bundles to allocate and what each winner must pay. Five paired bid changes affect only payments. Six markets have tied optimal awards, and all are accepted with their own payments. Three independent calculations agreed. [Clarke on pivot payments](https://doi.org/10.1007/BF01726210) and [Sandholm on combinatorial auctions](https://www.cs.cmu.edu/~sandholm/www/oralg.aij.pdf) |
| `box_push` is a single-player puzzle and `game_theory` uses a one-shot simultaneous game | `loop_game` requires reasoning about both players' future moves. The 32 starting positions comprise 12 forced wins, 10 losses and 10 draws. Seventeen decisive positions have reachable cycles. Enumerating complete stationary strategies matched a retrograde solver on all cases and exhaustive small-board controls. [Fraenkel and Yesha on graph games](https://www.sciencedirect.com/science/article/pii/0095895682900582) |

The runner now reports how many complete contrast pairs were solved in one language and how many were solved in both languages. A four-answer group passes only when every answer is correct. These checks can be recomputed from redacted verdicts using the run's pinned revision. The ordinary score retains its original denominator. [Gardner et al. on contrast-set evaluation](https://aclanthology.org/2020.findings-emnlp.117/)

## Private coverage review, closed-v11

- **Language switching over a whole dialogue.** `language_relay` moves from the starting language to the other language and back. Each of the two follow-ups corrects one record and adds another. On every turn the model justifies the earliest and latest attainable week and decides whether the date falls inside a review window. An enumerator checked 192 turn states. Eight of the 16 contrast pairs return to the same final state after requiring distinct responses in the middle. The normal score and the fraction of fully correct conversations with all language channels verified use separate measures. [Cross-turn language switching research](https://aclanthology.org/2026.mme-main.13/) and [Dechter, Meiri and Pearl on temporal constraints](https://www.sciencedirect.com/science/article/pii/0004370291900066)
- **Accessibility after an edit.** `header_binding` resolves table headers. `accessible_edit` asks what name and description the accessibility tree exposes before and after an HTML change. A local Chromium process confirmed all 256 strings across 128 DOM snapshots. A preliminary control exposed different treatment of repeated references to the same ID in Chromium and in the WAI authoring guide. Scored questions use distinct referenced IDs. [W3C accessible name and description computation](https://www.w3.org/TR/2026/WD-accname-1.2-20261104/) and [WAI authoring guide](https://www.w3.org/WAI/ARIA/apg/practices/names-and-descriptions/)
- **Measurements that change the circuit.** `resistors` asks for a single equivalent resistance. `bridge_loading` measures signed voltage and current after a component changes. Treating the meter as an open circuit fails on all 22 finite-load cases per language. An independent loop-current solution agreed with the nodal solution on all 64 localized cases. A third calculation using a Thévenin equivalent confirmed the 48 four-arm cases. [MIT nodal and mesh methods](https://ocw.mit.edu/courses/6-071j-introduction-to-electronics-signals-and-measurement-spring-2006/resources/nodal_mesh_methd/) and [MIT linear circuit analysis](https://ocw.mit.edu/courses/6-071j-introduction-to-electronics-signals-and-measurement-spring-2006/resources/linear_crct_ana/)

A serial Qwen3-0.6B Q8_0 pilot on four three-turn conversations returned native reasoning on 12/12 turns. Among the six German-target turns, the language classifier rejected all six. Among the six English-target turns, four passed and two were indeterminate. All 12 completions ended normally and had visible text, but none parsed as the required JSON report. The runner therefore could not verify the visible justification or decision fields. Two separately labeled language controls also returned native reasoning. [Redacted switched-language pilot](https://huggingface.co/datasets/flozi00/weakbench-data/blob/3e8fa4ef38ed2e9606b3af8a859d0d5ed589a3be/reports/closed-v11/qwen3-06b-native-switch-pilot.json)

## Private coverage review, closed-v10

| Earlier coverage | New task and evidence |
|---|---|
| `ambiguous_ref` decides whether one object is unique in one room | `listener_reference` asks for one shortest German or English description that lets two recipients identify their target despite differences in what they see. Of 32 scenarios, four require abstention even though each recipient alone could identify the card. Seven admit equally short alternatives. Independent integer optimization recovered every optimum. [Dale and Reiter on referring expressions](https://arxiv.org/abs/cmp-lg/9504020) and [Yoon and Brown-Schmidt on audience design](https://doi.org/10.1111/cogs.12774) |
| `regex_synth` tests expressions on examples and `automata_witness` supplies both acceptors | `stream_transducer` asks the model to construct a machine from a rule in prose. For every possible input sequence a product search checks that the machine emits the correct labels. A second compiler and partition refinement verify the references. The proven minimum is 1–2 states on 16 tasks, 3–4 on eight and 6–8 on eight. [Mealy on sequential circuits](https://doi.org/10.1002/j.1538-7305.1955.tb03788.x) |
| ASR transcribes speech, `sampled_waveform` supplies numbers and `polyphonic_score` supplies notation | `tap_timing` supplies two audible patterns with separate timing references. The model locates changes and, on eight recordings, compares one pattern in reverse. Twin recordings share the same text, duration, number of attacks and sample-value histogram. Moving one attack changes the answer. A PCM decoder recovered all 64 answers and FFmpeg independently decoded all 32 recordings sample for sample. [Povel and Essens on temporal patterns](https://doi.org/10.2307/40285311) |

In a serial Qwen3-4B-Instruct-2507 Q4_K_M pilot, none of 12 stream-synthesis responses was strict JSON. Five of six English responses and none of six German responses reached the 3,072-token limit. On 16 referring-expression trials, four responses used the requested sentence form. Two of 14 solvable trials uniquely identified the target for both recipients. Both descriptions included unnecessary properties. All eight English responses reached that task's 512-token limit. These small samples use paired scenarios, and their language differences coincide with different truncation rates. A reconstruction of all 28 requests found that both pilots used the v9 namespace for per-item seeds. The frozen v10 prompts, scoring specifications, scorer files and all other request fields matched. [Seed provenance correction](https://huggingface.co/datasets/flozi00/weakbench-data/blob/e1f752d46907cd224d8c3a27f5704f98a66a3939/reports/closed-v10/qwen3-4b-instruct-2507-q4km-runner-provenance.json). [Stream-synthesis pilot](https://huggingface.co/datasets/flozi00/weakbench-data/blob/80361e6bf4766ee674b3d5e64d55252153ac27b5/reports/closed-v10/qwen3-4b-instruct-2507-q4km-stream_transducer.json) and [referring-expression pilot](https://huggingface.co/datasets/flozi00/weakbench-data/blob/80361e6bf4766ee674b3d5e64d55252153ac27b5/reports/closed-v10/qwen3-4b-instruct-2507-q4km-listener_reference.json)

On the same 16 referring-expression trials with the original per-item seeds, Mistral-7B-Instruct-v0.3 Q4_K_M met the required sentence form on 12/16 trials. Five of 14 solvable descriptions identified the target for both recipients. One was minimal. All 16 stopped normally. Four additional synthetic controls produced no accepted sentence. A separate Qwen repetition changed only the seeds to the released v10 values. All eight language-by-stage aggregates matched its first run, including the total of 4,225 completion tokens. The stored results support a comparison of scores, while the wording of the discarded replies is unknown. [Matched Mistral pilot](https://huggingface.co/datasets/flozi00/weakbench-data/blob/63c6d6129704703e92dd33719000128373c2fb52/reports/closed-v10/mistral-7b-instruct-v03-q4km-listener_reference.json) and [official-seed repetition](https://huggingface.co/datasets/flozi00/weakbench-data/blob/d3851c3eb1abc7f5468b823d4998eb36b8c44d23/reports/closed-v10/qwen3-4b-instruct-2507-q4km-listener-official-seeds.json)

The local Qwen2.5-Omni-3B Q4_K_M pilot used one matched audio pair per level in both languages, with a 512-token cap. All 16 timing responses ended normally. None was valid JSON. Within each of the eight language/level groups, both recordings elicited identical text after case and surrounding whitespace normalization. For six groups a silent waveform of the same length also elicited that text. The same audio path correctly recognized all four separate spoken-digit controls, two per language. This distinguishes a working speech input path from success at timing differences under the response contract. [Redacted audio pilot](https://huggingface.co/datasets/flozi00/weakbench-data/blob/77e152abc7aef3acaade783adef0cfec4f0b2eb5/reports/closed-v10/qwen25-omni-3b-q4km-tap-timing.json)

## Private coverage review, closed-v9

- **Machine-checked proofs in German and English.** Earlier language tasks checked that a report stated prescribed facts. `localized_proof` lets the solver choose its sources and rules. Every statement must follow from an original fact or from earlier statements, every step must contribute to the proved goal, and the answer must identify that goal. A forward closure and independent backward proof search agree on all cases. The verifier accepted 256 submitted certificates covering alternative routes and valid reorderings. The language contract specifies the permitted sentence forms. ProofWriter is an earlier benchmark for natural-language proofs. [ProofWriter](https://aclanthology.org/2021.findings-acl.317/) and [Dowling and Gallier on Horn formulas](https://www.sciencedirect.com/science/article/pii/0743106684900141)
- **Spreadsheet formula relocation.** The public `sheet_eval` diagnostic asks for a cell value from a fixed worksheet. `formula_relocation` copies a rectangle, potentially across its own source, changes input cells and asks for recomputed results and dependencies. Two independent parsers agree. A local LibreOffice Calc process performed all 64 range copies and confirmed all 192 queried values and the 64 precedent sets. Sequential overwriting with no source snapshot fails on 16/32 scenarios per language. [Microsoft mixed references](https://support.microsoft.com/en-us/excel/switch-between-relative-absolute-and-mixed-references) and [LibreOffice range copying](https://api.libreoffice.org/docs/idl/ref/interfacecom_1_1sun_1_1star_1_1sheet_1_1XCellRangeMovement.html)
- **Individual counterfactuals.** The existing `causal_graph` asks about influence in a graph. `shared_noise` gives a completely specified binary mechanism and observations. The solver estimates the intervention for the same observed device, drawing a fresh device, conditioning without intervention and the probability that an individual's result changes. Exact latent-state enumeration, variable elimination on paired worlds and a polynomial calculation agree. Sampling a fresh device gives the wrong substitute for the individual probability on 31/32 scenarios per language. [Pearl, Glymour and Jewell on computing counterfactuals](https://web.cs.ucla.edu/~kaoru/BIB5/pearl-etal-2016-primer-text.pdf)

A serial language-channel pilot on the frozen v7 runner used one matched scenario in each language from each of two families. Qwen3-0.6B returned a separate `reasoning_content` field on all four real items. Both English traces and neither German trace passed the target-language check. One response had both visible report fields; their languages passed, and the full language check passed on 1/4 attempts. The strict factual score was 0/4. All four responses ended normally. A fifth, synthetic parser control is reported separately. [Redacted pilot report](https://huggingface.co/datasets/flozi00/weakbench-data/blob/71d43ebff86fe72402c62b6e2cc978a3054930f2/reports/closed-v7/qwen3-06b-native-language-pilot.json)

## Private coverage review, closed-v8

**Repairing one's own earlier commitments.** In earlier dialogue tasks the user supplies a record or updates an instruction. In `commitment_repair`, an assistant first books a time for each group. There are 8–58 feasible initial bookings per scenario. After each of two availability changes the assistant must minimize the cancellation cost and, among equally costly choices, the number of groups moved. The optimum depends on what that assistant actually told the user on the preceding turn. A matching dynamic program and a separate enumeration agreed in 3,272 checks and accepted all 2,588 complete optimal histories across the two languages. Contrast partners share their first two user turns and have disjoint sets of feasible final bookings. [Müller, Rudová and Barták on minimal perturbation](https://help-unitime.unitime.org/papers/patat05.pdf)

**Heat transfer across a phase change.** The earlier weighted-mean task mixes water at different temperatures. `phase_equilibrium` asks for temperature and liquid masses when different materials can freeze or melt. The 32 scenarios per language include eight contrast pairs where the heat pulse changes liquid mass while the equilibrium temperature stays fixed. Six scenarios land exactly at a phase boundary. An energy sweep over the melting points and a separate solution of the heat balance from the rendered material table agreed on all 64 cases and 1,125 additional generated checks. [OpenStax on phase changes](https://openstax.org/books/university-physics-volume-2/pages/1-5-phase-changes)

### What the language verifier can establish

The two language-report families now publish an additional rate over every scored item. `all_channels_verified` requires positive evidence that the request, visible justification, visible answer and API-returned reasoning are all in the target language. An absent reasoning channel contributes zero to this rate. `reasoning_present` and the conditional language rate on observed reasoning remain separately available.

We measured the lexical detector on public, professionally translated reading material. A mixed-language control appends the matched question in the opposite language to a passage in the target language. Errors on Belebele prompted a check for a foreign-language interrogative at the beginning of a question. On Belebele, accepted mixed controls went from 3/1,800 to 0/1,800. The separate XQuAD check went from 22/2,380 to 13/2,380. Source-language acceptance counts stayed the same in both comparisons. On XQuAD the revised detector accepted 177/240 German and 171/240 English passages; some same-language text receives an indeterminate result. These are tests on reading material. They quantify specific limitations of the heuristic when interpreting a model's reasoning-language rate. [Belebele](https://huggingface.co/datasets/facebook/belebele/tree/7899cdfa4e1e0d733fd77c848e2c273cb1d32be2) and [XQuAD](https://github.com/google-deepmind/xquad/tree/7d30520c717524000f0d9d2f9c10a069acd9d285)

## Private coverage review, closed-v7

Existing visual diagnostics include simple shape relations and bar charts. The private text tasks for raster topology and bounded inference provide nearby reasoning tasks. Two new optional families send an actual PNG with the localized question. In a full run they add 128 visual cases to the 2,184 text cases.

- **Voxel projections.** Three orthographic silhouettes and the total cube count constrain a 3 × 3 × 3 object. Every sampled problem admits multiple worlds. The model must distinguish necessity from possibility. Enumeration in the visual hull and an independent layer solver agree with a third pixel-based audit on every answer. Eight pairs share a question but differ in one silhouette cell. Eight share the same silhouettes but have a different stated cube count. A deterministic predictor that ignores either input can solve at most 24/32 scenarios per language. [Laurentini on visual hulls](https://doi.org/10.1109/34.273735)
- **Interval plots.** Two panels display groups in independently calibrated coordinates. The questions require reading points and capped interval endpoints, identifying the matching group, and applying conditions to the level and difference intervals. A second answer combines the plotted point estimates for a specified group. An independent decoder reads the rasterized tick labels, group labels and markers. In all 16 pairs, the wording is identical and changing one plotted interval changes the answer. Both families use distinct German and English instructions with shared evidence and gold. [Cleveland and McGill on graphical perception](https://doi.org/10.1080/01621459.1984.10478080)

A serial loopback pilot with SmolVLM-500M-Instruct Q8 exercised the actual image transport on eight inputs. It solved 0/8 at a 256-token output cap. Both German chart responses had the required JSON shape, and all four voxel outputs reached the cap. These observations describe the small transport pilot. The private aggregate report records the model and projector digests, version and counts. [Pilot report](https://huggingface.co/datasets/flozi00/weakbench-data/blob/0daf9f32b028b16830e5c8c04d632a7bd2b7c171/reports/closed-v7/smolvlm-500m-q8-local-pilot.json)

## Private coverage review, closed-v6

The new language tasks exercise different kinds of ambiguity. In one, a quotation changes the speaker and time that words refer to. In the other, the predicate and participants stay fixed and the nesting order of two quantifiers changes. The remaining additions cover verifiable certificates and reconstruction from incomplete or constrained observations.

| Earlier task or coverage gap | New operation and independent test |
|---|---|
| `coref` selects a pronoun antecedent and `de_direct_speech` checks punctuation | `quoted_perspective` resolves pronouns, place and time in nested direct quotations. A separate parser reconstructs the events from the story. The code accepts 64 permitted wordings per case and verifies each assertion in the explanation and answer. [Köder and Maier](https://doi.org/10.1017/S0305000915000318) |
| `syllogism` reasons over unary categories | `quantifier_scope` composes two generalized quantifiers over a binary relation. A relational evaluator and a bitset interpreter agree on the four truth-value counts and on witnesses for either disagreement. The two readings are explicit in the task. [Barwise and Cooper](https://doi.org/10.1007/BF00350139) |
| Rate comparisons and missing-data bounds | `paired_randomization` constructs the null distribution under the actual assignment restrictions. Polynomial convolution and complete assignment enumeration agree. Doubling the smaller one-sided tail gives a different answer in 19 of the 32 scenarios. [Matched-pairs permutation tests](https://www.site.uottawa.ca/~nat/Courses/csi5388/Perm.and.Matchedpairs.pdf) |
| `graph_path` returns a shortest distance | `flow_certificate` asks for both an integral flow and a directed cut that proves its optimality. Exhaustive enumeration of cuts checks the reference solver. Every network admits at least two optimal flows. [Ford and Fulkerson](https://www.cs.yale.edu/homes/lans/readings/routing/ford-max_flow-1956.pdf) |
| `paper_punch` and `strip_fold` track apertures or a folded stack | `booklet_imposition` places the pages of a reader's booklet in duplex printer coordinates, including intentional blanks and orientation. An index formula agrees with independent physical sheet assembly. [Adobe booklet imposition](https://helpx.adobe.com/indesign/desktop/print/print-booklets/booklet-types.html) |
| Hamming distance appears in a word-ladder control | `symbol_recovery` locates and repairs corrupted and erased symbols. Berlekamp–Welch decoding and interpolation over every eligible subset agree. [Reed–Solomon decoding](https://homes.cs.washington.edu/~anuprao/pubs/codingtheory/lecture4.pdf) |
| Existing signal and numeric questions do not resolve aliased components from extra measurements | `sampled_waveform` recovers a mixture whose regularly sampled observations are ambiguous. In every scenario the extra measurement makes the solution unique. Rational elimination agrees with a separate enumeration. [MIT discrete-time sampling](https://ocw.mit.edu/courses/res-6-007-signals-and-systems-spring-2011/resources/lecture-19-discrete-time-sampling/) |

This release adds 448 cases. It clarifies one sentence in the 32 German `quota_matching` instructions about remaining unmatched. A third solver encoded stability as a Boolean satisfiability problem and confirmed all 128 applicant- and studio-proposing allocations across both languages. The archived v5 data remains available at its original revision.

### Local model pilot on closed-v4

A serial, temperature-zero run of Qwen3-4B-Instruct-2507 Q4_K_M evaluated one contrast pair per v4 addition in each language. It solved 0/32 under the strict scorer with a 1,152-token output cap. Fourteen responses had the required JSON keys and incorrect values, ten reached the token cap, and none of the eight single fenced blocks became correct when unwrapped. Transport succeeded for all 32. These are 16 bilingual pair observations from eight underlying contrast pairs, a small diagnostic sample. The aggregate counts, model revision, seed and server build are recorded in the private report. [Pilot report](https://huggingface.co/datasets/flozi00/weakbench-data/blob/b33f2be7ddce1c5e7f3f749fa86aaca70671639b/reports/closed-v4/qwen3-4b-instruct-2507-q4km-pilot.json)

Two serial replays examined the zero full-answer score. Of the ten responses originally cut off, seven again reached the cap at 4,096 tokens and none produced valid JSON. A separate replay of all 14 responses that originally supplied the required keys reproduced all 14 shapes and the completion-token totals of all seven family/language groups. Two of 38 top-level field values were correct, both on English automata questions. Four independent copy-format controls with explicit answers passed in both languages (2/2 each). The field denominator covers only the 14 well-formed responses. [Token-budget replay](https://huggingface.co/datasets/flozi00/weakbench-data/blob/377e7d062d08dc2c9f27f995417be8500b57e410/reports/closed-v4/qwen3-4b-instruct-2507-q4km-budget.json) and [field replay](https://huggingface.co/datasets/flozi00/weakbench-data/blob/377e7d062d08dc2c9f27f995417be8500b57e410/reports/closed-v4/qwen3-4b-instruct-2507-q4km-fields.json)

## Private coverage review, closed-v5

The v4 catalogue included single-turn tool selection, date arithmetic, ranked voting, word operations and locale formatting. Five additions exercise different operations on related inputs.

| Comparison point | New task | Evidence used to grade it |
|---|---|---|
| Selecting a valid tool from its schema | `adaptive_measurement` needs a sequence of experiments. Two possible specimens present the same request. | Each tool result must agree with the hidden specimen. At every step, both outcomes must still permit a solution within the remaining budget. The 16 designs were also tested against 6,610 fixed probe sets and a policy that identifies all 140 possible specimens. [Nowak, generalized binary search](https://nowak.ece.wisc.edu/GBS_arxiv_v4.pdf) |
| Service windows and date arithmetic | `civil_recurrence` expands recurring wall-clock slots across a gap and a fold, applies explicit policies and cancels absolute instants. | A piecewise-offset inverse and a minute-by-minute forward simulation agree. Cases include both ends of a transition and shifts that cause valid occurrences to coincide. The task supplies a fictional clock and its own policies. [PEP 495](https://peps.python.org/pep-0495/) |
| Ranked voting and resource scheduling | `quota_matching` assigns places with rankings on both sides, incomplete lists and capacities. | Two deferred-acceptance runs agree with the applicant-wise best and worst among all stable assignments. The two outcomes differ on 11 of 32 scenarios per language. [Gale and Shapley](https://doi.org/10.1080/00029890.1962.11989827) |
| Word reversal, spelling and locale rules | `grapheme_edits` indexes successive edits by extended grapheme cluster. An edit can change later boundaries. | Boundary rules and an independent grammar reproduce the final scalar sequence and byte counts. The bounded repertoire passed 82 applicable official boundary vectors. [Unicode 15.1 grapheme boundaries](https://www.unicode.org/reports/tr29/tr29-43.html) |
| Collation and format conversion | `normalization_join` matches duplicate records under Unicode normalization and full case folding. | A property-table implementation cross-checks the standard-library normalizer. Both retain multiplicities and count the bytes of the original keys. The bounded repertoire passed 730 comparisons from 73 applicable official test rows. [Unicode 15.1 normalization](https://www.unicode.org/reports/tr15/tr15-54.html) |

Each family contributes 16 contrast pairs in each language. The interactive tasks differ in their tool state within a pair, and the four exact-output tasks differ in their visible inputs. Only the former family requires a chat endpoint that supports tool calls. The check on a saved transcript establishes consistency with the hidden state. Its provenance depends on the system that captured it.

## Private coverage review, closed-v4

The starting point was the executable registry at `c38602ef`: 14 private families and 463 separately selectable diagnostics. For each proposed addition we compared the input representation, the operation the model must perform and the evidence needed for a correct answer. A nearby task in a broad domain can still leave a specific capability unmeasured. HELM uses a scenario inventory to make such omissions visible. [Liang et al., Holistic Evaluation of Language Models](https://arxiv.org/abs/2211.09110)

| Earlier diagnostic | Capability added to the private suite | Independent check and method reference |
|---|---|---|
| `simpson`, `weighted_mean` | Compute the numerical change from pooling to comparison under common population weights | Two exact-rational calculations compare both groups. [NCHS direct standardization](https://www.cdc.gov/nchs/hus/sources-definitions/age-adjustment.htm) |
| `interval_math`, `math_missing` | Find the extreme attainable differences and the possible signs under coupled constraints on missing binary observations | Enumerated completions cross-check feasible subgroup totals. An interval spanning zero may still exclude a tie. [Horowitz and Manski on sharp bounds](https://doi.org/10.1080/01621459.2000.10473902) |
| `spatial`, `grid_transform` | Carry a point and a free vector through a chain of 3D frames in either direction | Global placements agree with an independent homogeneous-matrix calculation. [Lynch and Park on changes of frame](https://modernrobotics.northwestern.edu/nu-gm-book-resource/3-3-1-homogeneous-transformation-matrices/) |
| `paper_punch`, `grid_transform` | Recompute foreground components and enclosed background regions under a local change | Flood-fill is checked against a second connectivity implementation and the Euler characteristic. [scikit-image digital topology reference](https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_euler_number.html) |
| `rule_induction`, `counterexample` | Establish equivalence of two finite-state readers or construct a shortest separating input | Witnesses are checked by a second procedure on both readers, including equivalence and ties. [Hopcroft and Karp on automata equivalence](https://ecommons.cornell.edu/entities/publication/aa8cfe15-dcea-4be5-ae03-359120f548ea) |
| `unsat_detect`, `logic_grid` | Explain inconsistency with a smallest rule set and provide a consistent assignment after relaxation | Subset enumeration is checked against a minimum-hitting-set calculation. A greedy deletion can leave a larger core. [Ignatiev et al. on smallest MUSes](https://alexeyignatiev.github.io/assets/pdf/iplms-cp15-preprint.pdf) |
| `table_qa`, `table_build` | Recover two readings from a multi-level table with repeated leaf labels | Explicit header links agree with the generator's coordinate model. An independent audit also reconstructed the spanned layout. The test evaluates table interpretation. [W3C WAI multi-level tables](https://www.w3.org/WAI/tutorials/tables/multi-level/) |
| No symbolic score-reading task in the starting registry | Distinguish sounding notes, tied continuations and simultaneous new attacks in multiple voices | An absolute voice timeline agrees with a parser that replays the serialized score. These are original symbolic compositions. [MusicXML duration](https://www.w3.org/2021/06/musicxml40/musicxml-reference/elements/duration/) and [tie](https://www.w3.org/2021/06/musicxml40/musicxml-reference/elements/tie/) |

The release adds 256 German and 256 English questions. Each language contains 16 contrast pairs per new family. Both renderings of a scenario share one gold result. Scoring the opposite variant's gold failed on all 512 new items. Per language, the best constant JSON answer earns 8/32 for automata, 2/32 for conflict cores and 1/32 for each other family. These are checks against shortcuts. Model capability estimates need matched model runs and distinct scenarios as the units of analysis.

The next areas identified in the inventory are edits indexed by grapheme cluster, Unicode-normalization collisions, and experiments in which the next observation must depend on an earlier result. Existing wordplay, locale formatting and retrieval diagnostics provide comparison points. The exact Unicode repertoire, the source of tool observations and the required evidence for identification determine whether the proposed additions test a distinct skill.

## Historical diagnostic coverage

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
   between two judges on the same replies. A judge that reasons despite thinking off is supported
   (`--judge-think-budget`; its reasoning before `</think>` is stripped, the canaries still gate it), and every judge
   reply goes through `tasks_judge.loads_lenient` (bare YES/NO or labels, trailing commas, single quotes repaired;
   non-JSON still fails the item). Smoke on LFM2.5-24B as candidate and judge (`--tasks judge_all --limit 3`):
   267 items, 1 parse error, since fixed — self-judged scores are not results, the run only proves the plumbing.
4. **Language as a first-class skill**: reply-language control under pull (`lang_switch`, `mgsm_mixed`),
   grammaticality (`grammar_pairs`), proofreading without over-editing (`spell_fix`), plain-language rewriting
   with fact retention (`simplify`), native-editor error rates and fluency (`lang_quality`), and the language of
   the reasoning itself: every cell with a language check reports `think_lang_match` (share of items whose
   reasoning is in the cell's language; LFM2.5 reasons in English on German tasks behind an unopened `</think>`).

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
0.00-0.13. Most failures run to the 12k-token cap without an answer. The controlled version of the same contrast is `cube_turns`: identical cubes and moves, but half the items spell out where the top/front face goes and half name the move only — LFM2.5-24B scores 1.00 hinted vs 0.58 bare in both languages. Counting on a fixed 3-D body (`painted_cube` 0.92-0.96) is fine; resolving a movement is what breaks. The memorised-vs-novel contrast shows the same way: `bridge_torch` with 4 people (the textbook puzzle, a known
formula) 0.50, with 5 people 0.00; `box_push` and `puzzle8_min` (search with spatial constraints) sit at or below chance. These tasks keep headroom that a static set
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
