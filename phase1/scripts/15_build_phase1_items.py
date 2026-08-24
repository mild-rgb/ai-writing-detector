"""Build a single-document corpus spanning every generator prompt version.

`study/phase1/{v6,v7,v8_invert,v9}/answers.jsonl` all answer the SAME 100 `dev`
questions, so the human side is identical across conditions and version-to-
version comparisons are paired on the document. The project only ever
benchmarked v6 single-document; this makes the other three measurable on the
same footing.

    python3 scripts/15_build_phase1_items.py study/small/items_phase1.jsonl

Unlike `15_build_items.py` there is no batch file to verify against, because
these conditions were judged 2AFC rather than one document at a time. That means
the usual reconstruction check is unavailable, so this asserts what it can
instead: every ai row joins a human answer on `id`, sources are deduplicated on
(version, id, model), and the human set is emitted exactly once no matter how
many ai rows point at it.
"""
import collections
import glob
import json
import os
import sys

OUT = sys.argv[1] if len(sys.argv) > 1 else "study/small/items_phase1.jsonl"

hum = {}
for f in ("data/interim/questions_1000.jsonl", "data/interim/questions_longform.jsonl"):
    if os.path.exists(f):
        for l in open(f):
            r = json.loads(l)
            hum[r["id"]] = r

pool = {}
for src in sorted(glob.glob("study/phase1/*/answers.jsonl")):
    ver = src.split("/")[-2]
    for l in open(src):
        r = json.loads(l)
        pool[(ver, r["id"], r["model"])] = r          # last row wins
print(f"{len(pool)} ai rows after deduplicating on (version, id, model)")

items, used_q = [], set()
n = 0
for (ver, qid, model), r in sorted(pool.items()):
    if qid not in hum:
        sys.exit(f"{ver}/{qid}: no human answer to pair with")
    items.append({"item": f"P{n:04d}", "cond": ver, "label": "ai", "model": model,
                  "id": qid, "question": r["question"], "text": r["text"],
                  "words": len(r["text"].split()),
                  "artifact": r.get("artifact"), "persona": r.get("persona")})
    used_q.add(qid)
    n += 1

# The human class is emitted ONCE per question, however many ai rows point at
# it. Emitting it per ai row is the exact defect section 14 records: 75 human
# items that were only 39 distinct documents, and every interval too narrow.
for qid in sorted(used_q):
    h = hum[qid]
    items.append({"item": f"P{n:04d}", "cond": "human", "label": "human",
                  "model": "human", "id": qid, "question": h["question"],
                  "text": h["human_answer"], "words": len(h["human_answer"].split()),
                  "artifact": None, "persona": None})
    n += 1

n_ai = sum(i["label"] == "ai" for i in items)
assert len({(i["cond"], i["id"], i["model"]) for i in items if i["label"] == "ai"}) == n_ai
assert len({i["id"] for i in items if i["label"] == "human"}) == len(items) - n_ai

by = collections.Counter((i["cond"], i["model"].split("/")[-1])
                         for i in items if i["label"] == "ai")
print(f"{len(items)} items: {n_ai} ai / {len(items)-n_ai} human "
      f"({len(used_q)} distinct questions, human emitted once each)")
for k, v in sorted(by.items()):
    print(f"  {k[0]:10s} {k[1]:18s} {v}")

os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
with open(OUT, "w") as f:
    for i in items:
        f.write(json.dumps(i) + "\n")
print(f"-> {OUT}")
