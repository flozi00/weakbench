"""Every generated task's full tier holds distinct prompts (a repeated prompt counts twice and skews the score). Plain
asserts:

    python3 test_distinct.py
"""
import json

import tasks_judge
import tasks_lang
import tasks_novel
import tasks_open


# repeats found when this test was added; burn these down, never raise them
KNOWN = {'math_variants/en': 22,
         'math_variants/de': 22,
         'false_authority/en': 1,
         'false_authority/de': 1,
         'off_by_one/en': 13,
         'off_by_one/de': 13,
         'mixed_units/en': 2,
         'mixed_units/de': 2,
         'nesting_depth/en': 1,
         'nesting_depth/de': 1,
         'blocks_plan/en': 2,
         'blocks_plan/de': 2,
         'number_compare/en': 5,
         'number_compare/de': 5,
         'stale_answer/en': 9,
         'stale_answer/de': 9,
         'prob_fallacies/en': 94,
         'prob_fallacies/de': 94,
         'conclusion_spot/en': 8,
         'conclusion_spot/de': 8,
         'negation_qa/en': 4,
         'negation_qa/de': 4,
         'copycat/en': 1,
         'copycat/de': 1,
         'knights_knaves/en': 13,
         'knights_knaves/de': 13,
         'if_gen/en': 1,
         'if_gen/de': 1,
         'dialog_state/en': 55,
         'dialog_state/de': 55,
         'grammar_pairs/de': 15,
         'grammar_pairs/en': 14,
         'paraphrase/de': 3,
         'paraphrase/en': 3,
         'locale_fmt/de': 1,
         'locale_fmt/en': 1,
         'de_case/de': 4,
         'de_address/de': 1,
         'mixed_input/de': 7,
         'mixed_input/en': 7,
         'wug/de': 2,
         'wug/en': 17,
         'revise_on_feedback/de': 2,
         'revise_on_feedback/en': 2,
         'sentence_order/de': 1,
         'sentence_order/en': 1,
         'code_comments_lang/de': 35,
         'code_comments_lang/en': 35,
         'i18n_strings/de': 1,
         'i18n_strings/en': 1,
         'reasoning_faithful/de': 2,
         'reasoning_faithful/en': 2,
         'human_persona/de': 36,
         'human_persona/en': 36,
         'memory_dialog/de': 35,
         'memory_dialog/en': 35,
         'socratic_tutor/de': 1,
         'socratic_tutor/en': 1,
         'neutral_summary/de': 1,
         'neutral_summary/en': 1,
         'triage_advice/de': 24,
         'triage_advice/en': 24,
         'tactful_message/de': 2,
         'tactful_message/en': 2,
         'tutor_feedback/de': 3,
         'tutor_feedback/en': 3}


def test_distinct_prompts():
    tasks_judge.configure("http://127.0.0.1:9", "mock", 32768)  # loaders only check that a judge is set
    bad = []
    try:
        for mod in (tasks_novel, tasks_lang, tasks_open):
            for name, t in mod.TASKS.items():
                for lang in t["langs"]:
                    items = t["load"](lang, t["counts"]["full"][lang], 0)
                    keys = [json.dumps(it["messages"], ensure_ascii=False) for it in items]
                    rep = len(keys) - len(set(keys))
                    if rep > KNOWN.get(f"{name}/{lang}", 0):
                        bad.append((name, lang, rep))
    finally:
        tasks_judge.configure(None, None, None)
    assert not bad, bad


if __name__ == "__main__":
    test_distinct_prompts()
    print("ok")
