"""Build blind 2AFC judging batches -- one file per model.

A question's human answer is identical across the three models, so putting all
three trials for a question in one file lets a judge spot the repeated text and
win without reading style at all (observed: 100% accuracy on cue "reused text
indicates human"). Each model therefore gets its own file, in which every
question -- and so every human answer -- appears exactly once. The files must be
judged by separate agents that do not share context.
"""
import json
import os
import random
import sys

ANSWERS, OUTDIR = sys.argv[1], sys.argv[2]
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 0

hum = {json.loads(l)["id"]: json.loads(l)
       for l in open("data/interim/questions_1000.jsonl")}
ai = [json.loads(l) for l in open(ANSWERS)]

by_model = {}
for r in ai:
    by_model.setdefault(r["model"], []).append(r)

os.makedirs(OUTDIR, exist_ok=True)
for mi, (model, rows) in enumerate(sorted(by_model.items())):
    slug = model.split("/")[-1]
    rnd = random.Random(SEED * 100 + mi)
    rows = sorted(rows, key=lambda r: r["id"])
    rnd.shuffle(rows)
    seen, trials, key = set(), [], []
    for r in rows:
        if r["id"] in seen:          # one trial per question, always
            continue
        seen.add(r["id"])
        h = hum[r["id"]]
        flip = rnd.random() < 0.5
        a, b = ((h["human_answer"], r["text"]) if not flip
                else (r["text"], h["human_answer"]))
        tid = f"T{len(trials):03d}"
        trials.append({"trial": tid, "question": h["question"],
                       "A": a.strip(), "B": b.strip()})
        key.append({"trial": tid, "ai_side": "B" if not flip else "A",
                    "model": model, "id": r["id"]})

    with open(f"{OUTDIR}/key_{slug}.jsonl", "w") as f:
        for k in key:
            f.write(json.dumps(k) + "\n")

    lines = [
        "# Blind evaluation batch", "",
        "Each trial shows a question from the subreddit r/explainlikeimfive and",
        "two answers. Exactly one answer is a real comment written by a human",
        "redditor between 2011 and 2019. The other was written by a language",
        "model in 2026 instructed to imitate that style.", "",
        "For every trial, decide which answer (A or B) was written by the LANGUAGE MODEL.",
        "", f"There are {len(trials)} trials. Every trial is a different question.", "",
    ]
    for t in trials:
        lines += ["", "=" * 78, f"TRIAL {t['trial']}", "=" * 78,
                  f"QUESTION: {t['question']}", "", "--- ANSWER A ---", t["A"],
                  "", "--- ANSWER B ---", t["B"], ""]
    open(f"{OUTDIR}/blind_{slug}.txt", "w").write("\n".join(lines) + "\n")
    print(f"  {slug}: {len(trials)} trials, {len(seen)} distinct questions")
