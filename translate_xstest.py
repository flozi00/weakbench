"""Translate XSTest's 450 prompts to German once, and write the pinned artifact `data/xstest_de.jsonl`.

Run by hand, not by the suite: the suite only ever reads the committed file, so the gate stays offline and
deterministic. The translation is machine-made and unreviewed — that is recorded in the file header and in
README.md, and it is acceptable because both A/B arms see the same prompts, so only the absolute rate is
affected, never the paired delta.

    uv run python translate_xstest.py --url https://llm.example.org

The label (safe/unsafe) is copied from the English row and never translated: the cell is "does the model
refuse the German version of an unsafe prompt", so the label must stay the English one.
"""
import argparse
import json
import pathlib
import re
import sys
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from common import hf_file  # noqa: E402
from tasks_safety import FILE, XSTEST  # noqa: E402

OUT = pathlib.Path(__file__).parent / "data" / "xstest_de.jsonl"
BATCH = 25
ASK = (
    "Translate each English prompt into natural, idiomatic German. Keep the meaning, the register and the "
    "level of detail exactly; do not soften, censor or explain anything. Reply with a JSON array of "
    "objects, one per input, in the same order, each {\"id\": <the id>, \"de\": <the German prompt>}. "
    "No other text.\n\n%s"  # %s, not .format(): the prompt itself contains braces
)


def ask(url, rows, model):
    body = json.dumps({"model": model, "temperature": 0, "max_tokens": 4096,
                       "messages": [{"role": "user", "content": ASK % json.dumps(rows, ensure_ascii=False)}]})
    req = urllib.request.Request(f"{url.rstrip('/')}/v1/chat/completions", body.encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        text = json.load(r)["choices"][0]["message"]["content"]
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        raise ValueError(f"no JSON array in reply: {text[:300]}")
    return {str(o["id"]): o["de"] for o in json.loads(m.group(0))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", default=None, help="default: the server's first /v1/models id")
    a = ap.parse_args()
    model = a.model
    if not model:
        with urllib.request.urlopen(f"{a.url.rstrip('/')}/v1/models", timeout=60) as r:
            model = json.load(r)["data"][0]["id"]
    rows = list(__import__("csv").DictReader(open(hf_file(*XSTEST, FILE), newline="", encoding="utf-8")))
    out, todo = {}, [{"id": r["id"], "en": r["prompt"]} for r in rows]
    for i in range(0, len(todo), BATCH):
        chunk = todo[i:i + BATCH]
        for attempt in range(3):
            try:
                out.update(ask(a.url, chunk, model))
                break
            except Exception as e:  # noqa: BLE001 - a bad batch is retried, then reported
                print(f"batch {i // BATCH} attempt {attempt + 1}: {e}", file=sys.stderr)
        print(f"{len(out)}/{len(todo)}", file=sys.stderr)
    missing = [r["id"] for r in rows if r["id"] not in out or not out[r["id"]].strip()]
    if missing:
        sys.exit(f"incomplete: {len(missing)} ids missing, e.g. {missing[:5]}")
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"_meta": {"source": f"{XSTEST[0]}@{XSTEST[1]}", "file": FILE,
                                      "translator": model, "reviewed": False,
                                      "note": "machine translation, unreviewed; labels copied from the "
                                              "English row and never translated"}}) + "\n")
        for r in rows:
            f.write(json.dumps({"id": r["id"], "label": r["label"], "type": r["type"],
                                "en": r["prompt"], "de": out[r["id"]]}, ensure_ascii=False) + "\n")
    print(f"wrote {OUT} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
