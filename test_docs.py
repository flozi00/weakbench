"""Every task quality.py can run has a row in TASKS.md (30 new tasks once went undocumented). Run: python test_docs.py"""
import pathlib

import hidden
import quality


def test_every_task_documented():
    doc = (pathlib.Path(__file__).parent / "TASKS.md").read_text()
    missing = [t for t in (*quality.TASKS, *hidden.FULL_COUNTS, *hidden.VISUAL_COUNTS) if f"`{t}`" not in doc]
    assert not missing, f"tasks without a TASKS.md row: {missing}"


if __name__ == "__main__":
    test_every_task_documented()
    print("ok")
