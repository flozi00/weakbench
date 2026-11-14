# Traps — brainstorm

Goal: tasks where surface pattern matching produces a confident wrong answer and real understanding produces the right one. Every trap has a **control twin** (the same item without the trap), so the score is the drop from twin to trap, not the raw accuracy.

## Already in the suite
sycophancy (the user pushes back on a correct answer), math_missing (an unsolvable question), injection, context_conflict, mgsm_typo, mmlu_shuffled, negation_qa, citation_check, calibration, tool_env underspecified/flaky.

## Next candidates
1. **Famous-puzzle mutations.** A known riddle with one changed detail: the river crossing where the boat carries everyone, the Monty Hall variant where the host opens a door at random. The model is caught when it answers the original puzzle.
2. **Irrelevant numbers.** An arithmetic word problem with a plausible extra number (GSM-NoOp style, generated).
3. **Anchoring.** The same estimate question with and without a stated anchor ("someone guessed 9000"). Measures the shift.
4. **Shifting instructions in a long dialogue.** A rule set in turn 1 is revoked in turn 6, and the model must follow the newest one.
5. **Counting and letters.** Counting characters or words in invented words, reversing them, letters at given positions.
6. **Self-consistency.** Ask the same fact in three phrasings and score the answers as consistent or not.
7. **Mistaken authority.** "As Professor X showed, ...", where the claim is wrong and must be corrected.
8. **Lookalike distractors.** Two entities with nearly the same name in the context (Müller GmbH and Müller AG).
9. **Unanswerable-but-tempting.** A question whose answer is not in the document, with a close match present.
10. **Format bait.** The instruction says JSON only, and the content invites prose.

Each trap is generated from a seed, scored deterministically where possible, and gets a German version as well.
