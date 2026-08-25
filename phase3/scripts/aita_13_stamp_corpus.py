#!/usr/bin/env python3
"""Assemble the self-describing AITA corpus, the way ELI5's docs.jsonl is.

The provenance gap: label (human/ai), source, and — the one that bit ELI5 — the
train/test SPLIT lived only in side files, joined by q_id, never stamped on the
records. This writes one row per document carrying all of it, so nothing downstream
has to reconstruct a split or guess a label. Schema is ELI5-docs.jsonl-compatible
where the fields overlap (label, generator, split, q_id, question, text, text_norm,
words), so the same reader loads both corpora.

The split is a fresh, question-grouped 80/10/10 train/dev/test over the 2,900 corpus
questions, seeded. These are POOL questions — disjoint from the dev/bench/heldout
used to tune the floor prompt — so they are clean for detector training, unlike the
tuning splits. A q_id's human and AI documents always share a split (no title-level
leakage).

    python3 phase3/scripts/aita_13_stamp_corpus.py
Writes phase3/aita/data/aita_docs.jsonl (gitignored: contains Reddit text).
"""
import json, os, re, random, collections

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
SEED = 20260825
OUT = f"{ROOT}/aita/data/aita_docs.jsonl"

def norm(t): return re.sub(r"\s+", " ", t or "").strip()

ai = [json.loads(l) for l in open(f"{ROOT}/aita/data/corpus_answers.jsonl") if l.strip()]
hum = {r["q_id"]: r for r in map(json.loads, open(f"{ROOT}/aita/data/aita_human.jsonl"))}

# question-grouped 80/10/10 split over the corpus questions
qids = sorted({r["q_id"] for r in ai if r.get("text") and r["q_id"] in hum})
random.Random(SEED).shuffle(qids)
n = len(qids); a, b = int(0.8 * n), int(0.9 * n)
split_of = {q: ("train" if i < a else "dev" if i < b else "test") for i, q in enumerate(qids)}

rows = []
for r in ai:
    q = r["q_id"]
    if q not in split_of:
        continue
    t = r["text"]
    rows.append({
        "doc_id": f"aita-{q}-ai", "q_id": q, "label": "ai",
        "generator": r["model"], "source": "aita_floor", "split": split_of[q],
        "question": r.get("question"), "text": t, "text_norm": norm(t),
        "words": len(t.split()),
        "seed": r.get("seed"), "draws": r.get("draws"), "post": r.get("post"),
    })
    h = hum[q]; ht = h["human_answer"]
    rows.append({
        "doc_id": f"aita-{q}-human", "q_id": q, "label": "human",
        "generator": "human", "source": "scruples", "split": split_of[q],
        "question": h.get("question"), "text": ht, "text_norm": norm(ht),
        "words": len(ht.split()),
        "date": h.get("date"), "votes": h.get("votes"), "verdict": h.get("verdict"),
    })

rows.sort(key=lambda r: (r["q_id"], r["label"]))
with open(OUT, "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")

# ---- verify ----
bysplit = collections.Counter(r["split"] for r in rows)
bylabel = collections.Counter((r["split"], r["label"]) for r in rows)
straddle = sum(1 for q in split_of if len({r["split"] for r in rows if r["q_id"] == q}) > 1)
print(f"wrote {len(rows)} docs -> {OUT}")
print(f"splits: {dict(bysplit)}")
for s in ("train", "dev", "test"):
    print(f"  {s:5} ai {bylabel[(s,'ai')]:4}  human {bylabel[(s,'human')]:4}")
print(f"q_ids straddling a split: {straddle}  (must be 0)")
print(f"every record stamped with label/source/generator/split: "
      f"{all(all(k in r for k in ('label','source','generator','split')) for r in rows)}")
