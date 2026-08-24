"""Reconstruct the per-item documents a single-document benchmark handed to judges.

`key.jsonl` records only (item, cond, label, model, id) -- the text lives in the
batch files, wrapped in judge-facing scaffolding. A small-model scorer needs the
documents as data, not as prose, so this rebuilds them from the same sources
`12_make_mixed_single.py` drew from and then VERIFIES the reconstruction against
the batch files byte for byte.

The verification is the point. Rebuilding from sources is a guess about which
row won a (id, model) collision and which text field was used; section 14 of
NARRATIVE.md is a long account of what happens when the thing that built the
sample is trusted rather than checked. If a single character differs, this exits
non-zero rather than emitting a corpus that is subtly not the one that was
judged.

    python3 scripts/15_build_items.py study/single/long2 study/small/items_long2.jsonl

Generated answers for different benchmarks live in different directories and the
same (id, model) pair exists at several prompt versions, so the ai source must be
named rather than guessed -- pass `--ai-src` when the default set is wrong. The
batch-file verification below is what catches a wrong choice: pointing this at
the v7 answers for a v6 benchmark fails on a text mismatch rather than quietly
emitting the wrong corpus.

    python3 scripts/15_build_items.py study/single/short study/small/items_short.jsonl \
        --ai-src study/phase1/v6/answers.jsonl

Also joins the `artifact` and `persona` columns from the generator so a scorer's
margins can later be broken down by injected feature -- the rate-matched Edit:/
TL;DR draws and the first-person/hedge draws -- to test whether the injections
are individually detectable.
"""
import argparse
import json
import os
import re
import sys

_ap = argparse.ArgumentParser()
_ap.add_argument("keydir")
_ap.add_argument("out")
_ap.add_argument("--ai-src", nargs="*", default=None,
                 help="answers.jsonl files to draw the ai class from; defaults to "
                      "the long-form sets")
_a = _ap.parse_args()
KEYDIR, OUT = _a.keydir, _a.out

HUMAN_SRC = ["data/interim/questions_1000.jsonl",
             "data/interim/questions_longform.jsonl"]
AI_SRC = _a.ai_src or ["study/longform/answers.jsonl",
                       "study/longform3/answers.jsonl",
                       "data/interim/ai_answers.jsonl"]

hum = {}
for f in HUMAN_SRC:
    if os.path.exists(f):
        for l in open(f):
            r = json.loads(l)
            hum[r["id"]] = r

# Deduplicate on (id, model), last row wins -- the same rule the item builder
# used, so a restart-appended duplicate resolves to the same text it picked.
ai = {}
for f in AI_SRC:
    if os.path.exists(f):
        for l in open(f):
            r = json.loads(l)
            ai[(r["id"], r["model"])] = r

key = [json.loads(l) for l in open(f"{KEYDIR}/key.jsonl")]

items = []
for k in key:
    if k["label"] == "ai":
        r = ai.get((k["id"], k["model"]))
        if r is None:
            sys.exit(f"{k['item']}: no ai row for {(k['id'], k['model'])}")
        text, question = r["text"], r["question"]
        extra = {"artifact": r.get("artifact"), "persona": r.get("persona"),
                 "prompt_version": r.get("prompt_version"),
                 "target_words": r.get("target_words")}
    else:
        r = hum.get(k["id"])
        if r is None:
            sys.exit(f"{k['item']}: no human row for {k['id']}")
        text, question = r["human_answer"], r["question"]
        extra = {"artifact": None, "persona": None, "prompt_version": None,
                 "score": r.get("score")}
    items.append({**k, "question": question, "text": text,
                  "words": len(text.split()), **extra})

# --- verify against what the judges actually read -------------------------
# Batch files hold "ITEM Innn ... QUESTION: q ... COMMENT:\n<text>\n" blocks.
# `long2` keeps batches in a `batches/` subdirectory with the key outside it;
# the earlier sets wrote both into the benchmark directory together. Accept both
# so the older benchmarks can still be verified.
BATCHDIR = f"{KEYDIR}/batches" if os.path.isdir(f"{KEYDIR}/batches") else KEYDIR

seen = {}
for bf in sorted(f for f in os.listdir(BATCHDIR) if f.startswith("batch_")):
    raw = open(f"{BATCHDIR}/{bf}").read()
    for m in re.finditer(
            r"ITEM (I\d{3})\n=+\nQUESTION: (.*?)\n\nCOMMENT:\n(.*?)\n\n(?==+|\Z)",
            raw, re.S):
        seen[m.group(1)] = (m.group(2), m.group(3))

mismatch = []
for it in items:
    if it["item"] not in seen:
        mismatch.append(f"{it['item']}: not found in any batch file")
        continue
    q, t = seen[it["item"]]
    if t.strip() != it["text"].strip():
        mismatch.append(f"{it['item']}: TEXT differs from the batch file "
                        f"(rebuilt {len(it['text'])} chars, batch {len(t)})")
    elif q.strip() != it["question"].strip():
        mismatch.append(f"{it['item']}: QUESTION differs from the batch file")

extra_in_batches = sorted(set(seen) - {i["item"] for i in items})
if extra_in_batches:
    mismatch.append(f"{len(extra_in_batches)} items in batches but not in key: "
                    f"{', '.join(extra_in_batches[:8])}")

n_ai = sum(1 for i in items if i["label"] == "ai")
print(f"{KEYDIR}: {len(items)} items ({n_ai} ai / {len(items) - n_ai} human)")
print(f"  distinct ai documents    {len({(i['id'], i['model']) for i in items if i['label'] == 'ai'})}")
print(f"  distinct human documents {len({i['id'] for i in items if i['label'] == 'human'})}")
print(f"  verified against batches {len(seen)} blocks parsed")
ws = sorted(i["words"] for i in items)
print(f"  words: median {ws[len(ws) // 2]}  range {ws[0]}-{ws[-1]}")

if mismatch:
    print("\nRECONSTRUCTION FAILED", file=sys.stderr)
    for m in mismatch[:20]:
        print(f"  {m}", file=sys.stderr)
    sys.exit(f"{len(mismatch)} item(s) do not match the judged text -- refusing "
             f"to emit a corpus that is not the one that was benchmarked")

os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
with open(OUT, "w") as f:
    for it in items:
        f.write(json.dumps(it) + "\n")
print(f"\nall {len(items)} items match the batch files exactly -> {OUT}")
