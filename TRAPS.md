# Traps — brainstorm

Goal: tasks where surface pattern matching produces a confident wrong answer and real understanding produces the right one. Every trap has a **control twin** (the same item without the trap), so the score is the drop from twin to trap, not the raw accuracy.

## Already in the suite
sycophancy (the user pushes back on a correct answer), math_missing (an unsolvable question), injection, context_conflict, mgsm_typo, mmlu_shuffled, negation_qa, citation_check, calibration, tool_env underspecified/flaky.

## Next candidates
1. **Famous-puzzle mutations** (done: `puzzle_mutation`). A known riddle with one changed detail: the river crossing where the boat carries everyone, the Monty Hall variant where the host opens a door at random. The model is caught when it answers the original puzzle.
2. **Irrelevant numbers** (covered by `math_variants`, sub-metric `ok_noop`). An arithmetic word problem with a plausible extra number (GSM-NoOp style, generated).
3. **Anchoring** (done: `anchoring`, generated products; thinking models may just compute). The same estimate question with and without a stated anchor ("someone guessed 9000"). Measures the shift.
4. **Shifting instructions in a long dialogue** (done: `revoked_rule`). A rule set in turn 1 is revoked in turn 6, and the model must follow the newest one.
5. **Counting and letters** (done: `char_ops`). Counting characters or words in invented words, reversing them, letters at given positions.
6. **Self-consistency** (done: `rephrase_consistency`, `group_agree` / `group_all`). Ask the same fact in three phrasings and score the answers as consistent or not.
7. **Mistaken authority** (done: `false_authority`). "As Professor X showed, ...", where the claim is wrong and must be corrected.
8. **Lookalike distractors** (done: `lookalike`). Two entities with nearly the same name in the context (Müller GmbH and Müller AG).
9. **Unanswerable-but-tempting** (done: `near_miss_qa`). A question whose answer is not in the document, with a close match present.
10. **Format bait** (done: `format_bait`). The instruction says JSON only, and the content invites prose.

Each trap is generated from a seed, scored deterministically where possible, and gets a German version as well.

## Round 2 — brainstorm

11. **Order bias.** The same multiple-choice item with the options shuffled. The model is caught when it prefers a position, which shows as answers that are not invariant under permutation.
12. **Presupposition in the question.** "Why did X resign in 2019?" when the text says X never resigned. Twin: the text says X did resign.
13. **Quantifier slips.** "All / most / some / none" in the premises, and a question that a model answers as if "some" meant "all".
14. **Off-by-one in counting ranges** (done: `off_by_one`). "How many days from the 3rd to the 9th, both inclusive?", fence posts, page ranges.
15. **Units hiding in plain sight.** Mixed units in a single sum (m and cm, minutes and hours), where the naive sum is the trap.
16. **Reversal curse on invented facts** (done: `reversal`, in context only; the training-time curse needs fine-tuning, out of scope). The context teaches "A's mentor is B", and the question asks "Whose mentor is B?".
17. **Distracting long preamble.** The real question is the last sentence after a paragraph that sets up a different question.
18. **Sticky persona.** A persona is set in the system prompt, and the user asks the model to break it in a later turn. Measures where the boundary sits and whether the model keeps it.
19. **Double negation.** "It is not untrue that ...": yes/no questions with 0, 1 or 2 negations.
20. **Spurious majority.** Several sources state a wrong value and one sound source (a measurement) states the right one, with a rule given for which source to trust.
