"""Local creative league catalogues and the immutable private cache."""

import io
from hashlib import sha256
import json
import os
import tempfile
import unittest
from unittest.mock import patch

import arena_catalog as catalog


def row(key="demo"):
    return {
        "id": key,
        "kind": "text",
        "language": "en",
        "prompt": "Write a sentence.",
        "rubric": ["The sentence is clear."],
    }


class CatalogueTests(unittest.TestCase):
    def test_custom_forms_have_identical_cases(self):
        cases = [row("one"), row("two")]
        for data in (
            json.dumps(cases),
            json.dumps({"cases": cases}),
            "\n".join(map(json.dumps, cases)),
        ):
            self.assertEqual(catalog.decode(data.encode()), cases)
        self.assertEqual(catalog.decode(json.dumps(row()).encode()), [row()])

    def test_malformed_rows_and_changed_hash_refused(self):
        for rows in (
            [row(), row()],
            [dict(row(), rubric=[])],
            [dict(row(), rubric=[""])],
            [dict(row(), kind="foreign")],
            [dict(row(), language="")],
            [dict(row(), prompt="")],
            [dict(row(), id="../bad")],
        ):
            with self.assertRaises(ValueError):
                catalog.decode(json.dumps(rows).encode())
        with self.assertRaisesRegex(ValueError, "checksum"):
            catalog.decode(json.dumps([row()]).encode(), pinned=True)

    def test_prepare_requires_confirmed_private_revision_and_writes_separate_cache(
        self,
    ):
        cases = [
            dict(row(f"case-{kind}-{lang}-{n}"), kind=kind, language=lang)
            for kind in catalog.KINDS
            for lang in ("de", "en")
            for n in range(4)
        ]
        data = json.dumps(cases).encode()
        meta = {
            "id": catalog.hidden.DATA_REPO,
            "sha": catalog.REVISION,
            "private": False,
        }
        with (
            tempfile.TemporaryDirectory() as d,
            patch.dict(os.environ, {"BENCH_CACHE": d, "HF_TOKEN": "fixture"}),
            patch.object(catalog, "DIGEST", sha256(data).hexdigest()),
        ):
            with patch(
                "urllib.request.OpenerDirector.open",
                return_value=io.BytesIO(json.dumps(meta).encode()),
            ) as get:
                with self.assertRaisesRegex(ValueError, "private dataset"):
                    catalog.prepare()
                self.assertEqual(get.call_count, 1)
            meta["private"] = True
            with patch(
                "urllib.request.OpenerDirector.open",
                side_effect=[io.BytesIO(json.dumps(meta).encode()), io.BytesIO(data)],
            ):
                catalog.prepare()
            self.assertEqual(catalog.load_catalog(), cases)
            self.assertNotEqual(catalog.cache_path(), catalog.hidden.cache_path())
            self.assertEqual(catalog.cache_path().stat().st_mode & 0o077, 0)


if __name__ == "__main__":
    unittest.main()
