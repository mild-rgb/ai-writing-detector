"""Build a single-document classification set.

Every measurement so far is 2AFC: the judge sees the human and AI answer to the
same question and picks one. That is easier than the real task -- a detector
meets one document, alone, with no reference and no sibling set. This builds
one file per item so a judge sees exactly one answer and must decide in
isolation.
"""
import json
import os
import random
import sys

ANSWERS, OUTDIR, N = sys.argv[1], sys.argv[2], int(sys.argv[3])
MODEL = sys.argv[4] if len(sys.argv) > 4 else None
SEED = 99

hum = {json.loads(l)["id"]: json.loads(l)
       for l in open("data/interim/questions_1000.jsonl")}
ai = [json.loads(l) for l in open(ANSWERS)]
if MODEL:
    ai = [r for r in ai if MODEL in r["model"]]

rnd = random.Random(SEED)
rnd.shuffle(ai)
picks = ai[:N // 2]
items = []
for r in picks:
    items.append({"label": "ai", "model": r["model"], "id": r["id"],
                  "question": r["question"], "text": r["text"]})
    h = hum[r["id"]]
    items.append({"label": "human", "model": "human", "id": r["id"],
                  "question": h["question"], "text": h["human_answer"]})
rnd.shuffle(items)
items = items[:N]

os.makedirs(OUTDIR, exist_ok=True)
with open(f"{OUTDIR}/key.jsonl", "w") as f:
    for i, it in enumerate(items):
        f.write(json.dumps({"item": f"I{i:03d}", **{k: it[k] for k in
                            ("label", "model", "id")}}) + "\n")
for i, it in enumerate(items):
    body = [
        "# Single comment",
        "",
        "Below is one comment posted in reply to a question on the subreddit",
        "r/explainlikeimfive. It was either written by a human redditor between",
        "2011 and 2019, or by a language model in 2026 imitating that style.",
        "",
        "You are seeing this comment on its own. There is no comparison answer.",
        "",
        f"QUESTION: {it['question']}",
        "",
        "COMMENT:",
        it["text"],
        "",
    ]
    open(f"{OUTDIR}/item_{i:03d}.txt", "w").write("\n".join(body) + "\n")
n_ai = sum(1 for it in items if it["label"] == "ai")
print(f"{len(items)} single items -> {OUTDIR}  ({n_ai} ai, {len(items)-n_ai} human)")
