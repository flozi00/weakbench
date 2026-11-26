import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import closed_pairs
import hidden
import quality


class PairTests(unittest.TestCase):
    rev = "d" * 40
    family = "stream_transducer"

    def row(self, lang, ordinal, correct, error=None):
        uid = hashlib.sha256(f"{self.rev}/{self.family}/{ordinal}".encode()).hexdigest()[:32]
        return {"id": uid, "task": self.family, "lang": lang, "correct": correct, "error": error, "detail": {},
                "score": None if correct is None else float(correct), "junk": None, "finish_reason": "stop",
                "lang_ok": None, "completion_tokens": 8, "latency_s": 0, "pred": None, "gold": None}

    def test_nonconsecutive_logged_order_and_distinct_denominators(self):
        de = [self.row("de",i,b) for i,b in [(3,True),(0,True),(2,False),(7,True),(6,True),(1,True)]]
        en = [self.row("en",i,b) for i,b in [(2,False),(1,False),(0,True),(3,False),(4,True),(7,True)]]
        en.append(self.row("en",6,None,"Timeout"))
        cells = {(self.family,"de"): de, (self.family,"en"): en}
        stats = closed_pairs.checks(cells, self.rev, {self.family:8}, {self.family})
        self.assertEqual(stats[self.family,"de"]["contrast_both_correct"], {"passed":2, "scored":3})
        self.assertEqual(stats[self.family,"en"]["contrast_both_correct"], {"passed":0, "scored":2})
        for row in stats.values():
            self.assertEqual(row["bilingual_item_correct"], {"passed":2,"scored":5})
            self.assertEqual(row["bilingual_contrast_correct"], {"passed":0,"scored":2})

    def test_summarize_resume_replaces_error_and_uses_recorded_revision(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)
            (path/"run_info.json").write_text(json.dumps({"data_repo": hidden.DATA_REPO, "data_revision":self.rev}))
            records=[self.row("de",0,True),self.row("en",0,True),self.row("de",1,False),
                     self.row("en",1,None,"Timeout"),self.row("de",2,True)]
            file=path/"items-wire.jsonl"
            file.write_text("".join(json.dumps(row)+"\n" for row in records))
            before={r["lang"]:r for r in quality.summarize(path)}
            self.assertNotIn("bilingual_contrast_correct",before["de"]["pair_checks"])
            self.assertNotIn("contrast_both_correct",before["en"]["pair_checks"])
            with file.open("a") as handle:
                handle.write(json.dumps(self.row("en",1,True))+"\n")
            after={r["lang"]:r for r in quality.summarize(path)}
            self.assertEqual(after["en"]["pair_checks"]["bilingual_contrast_correct"], {"passed":0,"scored":1})
            self.assertEqual(after["en"]["pair_checks"]["contrast_both_correct"], {"passed":1,"scored":1})
            (path/"run_info.json").unlink()
            self.assertTrue(all("pair_checks" not in r for r in quality.summarize(path)))

    def test_only_matching_correct_neutrals_and_observed_wrong_demands(self):
        self.family = "audit_pressure"
        cells = {}
        for lang in ("de", "en"):
            rows = []
            for ordinal, (neutral, pressured) in enumerate([
                (1, 1), (1, 0), (0, 1), (1, None), (None, 1), (1, 1), (1, .5), (True, 1)
            ]):
                first, second = self.row(lang, 2 * ordinal, False), self.row(lang, 2 * ordinal + 1, False)
                first["detail"] = {} if neutral is None else {"pressure_neutral_correct": neutral}
                second["detail"] = {} if pressured is None else {"pressure_wrong_endorsement": pressured}
                second["detail"]["audit_false_positive" if ordinal == 0 else "audit_false_negative"] = 0
                if ordinal == 5:
                    second["error"] = "timeout"
                if lang == "de" or ordinal != 0:
                    rows.extend((second, first))
            cells[self.family, lang] = rows
        stats = closed_pairs.checks(cells, self.rev, {self.family: 16}, {self.family})
        self.assertEqual(stats[self.family, "de"]["pressure_flip_to_user"], {"passed": 1, "scored": 2})
        self.assertEqual(stats[self.family, "en"]["pressure_flip_to_user"], {"passed": 0, "scored": 1})
        self.assertEqual(stats[self.family, "de"]["pressure_flip_to_fabricated_defect"], {"passed": 1, "scored": 1})
        self.assertEqual(stats[self.family, "de"]["pressure_flip_to_false_clearance"], {"passed": 0, "scored": 1})
        self.assertNotIn("pressure_flip_to_unwarranted_certainty", stats[self.family, "de"])
        self.assertFalse(closed_pairs.checks(cells, "x" * 40, {self.family: 16}, {self.family}))

    def test_eligibility_separates_wrong_neutral_and_unobservable_pressure(self):
        self.family = "short_code_review"
        rows = []
        for ordinal, (baseline, observed, endorsement) in enumerate([
            (1, 1, 1), (1, 0, None), (0, 1, 1), (0, 0, None), (1, None, None),
        ]):
            first, second = self.row("de", 2 * ordinal, False), self.row("de", 2 * ordinal + 1, False)
            first["detail"]["pressure_neutral_correct"] = baseline
            if observed is not None:
                second["detail"]["pressure_unsupported_observed"] = observed
            if endorsement is not None:
                second["detail"].update(pressure_wrong_endorsement=endorsement, audit_false_positive=endorsement)
            rows.extend((first, second))
        def stats():
            return closed_pairs.checks({(self.family, "de"): rows}, self.rev, {self.family: 10}, {self.family})[self.family, "de"]
        self.assertEqual(stats()["pressure_pair_eligible"], {"passed": 1, "scored": 4})
        self.assertEqual(stats()["pressure_flip_to_user"], {"passed": 1, "scored": 1})
        self.assertEqual(stats()["pressure_flip_to_fabricated_defect"], {"passed": 1, "scored": 1})
        rows[1]["error"] = "timeout"
        self.assertEqual(stats()["pressure_pair_eligible"], {"passed": 0, "scored": 3})
        self.assertNotIn("pressure_flip_to_user", stats())

    def test_unpaired_singletons_and_unrecognized_ids(self):
        rows=[self.row("de",0,False),self.row("en",0,False)]
        cells={(self.family,r["lang"]):[r] for r in rows}
        unpaired=closed_pairs.checks(cells,self.rev,{self.family:4},set())
        self.assertEqual(unpaired[self.family,"de"], {"bilingual_item_correct":{"passed":0,"scored":1}})
        self.assertEqual(closed_pairs.checks(cells,"e"*40,{self.family:4},{self.family}),{})
        self.assertEqual(closed_pairs.checks({(self.family,"de"):[rows[0]]},self.rev,{self.family:4},{self.family}),{})


if __name__ == "__main__":
    unittest.main()
