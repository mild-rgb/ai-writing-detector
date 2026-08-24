"""Split a condition into small per-judge batches.

200 answers in one context is enough for a judge to reverse-engineer the
generator from repetition across the file -- v8_invert's planted phrases were
visible at 56-87% frequency and the judges locked onto them. Small batches keep
each judge closer to how a detector actually meets text: a couple of dozen
documents, no sibling set to compare against.
"""
import json
import os
import random
import sys

ANSWERS, OUTDIR = sys.argv[1], sys.argv[2]
PER_BATCH = int(sys.argv[3]) if len(sys.argv) > 3 else 25
SEED = int(sys.argv[4]) if len(sys.argv) > 4 else 0

hum = {json.loads(l)["id"]: json.loads(l)
       for l in open("data/interim/questions_1000.jsonl")}
ai = [json.loads(l) for l in open(ANSWERS)]
by_model = {}
for r in ai:
    by_model.setdefault(r["model"], []).append(r)

os.makedirs(OUTDIR, exist_ok=True)
made = 0
for mi, (model, rows) in enumerate(sorted(by_model.items())):
    slug = model.split("/")[-1]
    rnd = random.Random(SEED * 100 + mi)
    rows = sorted(rows, key=lambda r: r["id"])
    rnd.shuffle(rows)
    for bi in range(0, len(rows), PER_BATCH):
        chunk = rows[bi:bi + PER_BATCH]
        if len(chunk) < PER_BATCH:
            break
        tag = f"{slug}_b{bi // PER_BATCH}"
        trials, key = [], []
        for r in chunk:
            h = hum[r["id"]]
            flip = rnd.random() < 0.5
            a, b = ((h["human_answer"], r["text"]) if not flip
                    else (r["text"], h["human_answer"]))
            tid = f"T{len(trials):03d}"
            trials.append((tid, h["question"], a.strip(), b.strip()))
            key.append({"trial": tid, "ai_side": "B" if not flip else "A",
                        "model": model, "id": r["id"]})
        with open(f"{OUTDIR}/key_{tag}.jsonl", "w") as f:
            for k in key:
                f.write(json.dumps(k) + "\n")
        lines = ["# Blind evaluation batch", "",
                 "Each trial shows a question from r/explainlikeimfive and two answers.",
                 "Exactly one is a real human Reddit comment from 2011-2019; the other",
                 "was written by a 2026 language model imitating that style.", "",
                 "For every trial, decide which answer (A or B) is the LANGUAGE MODEL.",
                 "", f"There are {len(trials)} trials, each a different question.", ""]
        for tid, q, a, b in trials:
            lines += ["", "=" * 78, f"TRIAL {tid}", "=" * 78, f"QUESTION: {q}", "",
                      "--- ANSWER A ---", a, "", "--- ANSWER B ---", b, ""]
        open(f"{OUTDIR}/blind_{tag}.txt", "w").write("\n".join(lines) + "\n")
        made += 1
print(f"{made} batches of {PER_BATCH} trials -> {OUTDIR}")
