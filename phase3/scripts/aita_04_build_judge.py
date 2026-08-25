"""Build single-document judge batches for the AITA sub-project.

A port of build_bench.py. Every invariant it asserts is kept, because each one
fixes a defect that had already produced a published number:

  * ONE model per question, so each question contributes exactly one ai item and
    one human item and no document is ever emitted twice.
  * No batch may contain two documents for the same question -- that would be a
    2AFC pair inside a benchmark whose whole premise is that no reference text
    exists.
  * key.jsonl lives OUTSIDE the directory handed to judges.
  * Every invariant is asserted before anything is written.

Replicate passes reshuffle which items share a batch (seeded by --pass) but
never change the item set, so the passes are independent judge draws over
identical documents. That is what phase3/study/judge_compare/RESULTS.md requires:
re-running the same route on the same items moved detection by 13 points and
balanced accuracy by 2, so the headline is a mean over passes with its SD.

    python3 phase3/scripts/aita_04_build_judge.py PROMPT_DIR --split dev --pass 0
"""
import argparse
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HUMANS = f"{ROOT}/phase3/aita/data/aita_human.jsonl"
PER_BATCH = 5

ap = argparse.ArgumentParser()
ap.add_argument("prompt_dir")
ap.add_argument("--split", default="dev")
ap.add_argument("--pass", dest="pass_n", type=int, default=0)
ap.add_argument("--answers")
ap.add_argument("--out")
a = ap.parse_args()

pdir = a.prompt_dir.rstrip("/")
answers = a.answers or f"{pdir}/answers_{a.split}.jsonl"
rows = [json.loads(l) for l in open(answers) if l.strip()]
pool = {}
for r in rows:
    if r["id"] in pool:
        sys.exit(f"FATAL: {r['id']} has more than one ai document in {answers}. "
                 f"One model per question -- a question contributing two ai "
                 f"items is a different benchmark.")
    pool[r["id"]] = r

humans = {json.loads(l)["id"]: json.loads(l) for l in open(HUMANS)}

items = []
for qid in sorted(pool):
    r = pool[qid]
    items.append({"label": "ai", "model": r["model"], "id": qid,
                  "question": r["question"], "text": r["text"]})
    h = humans[qid]
    items.append({"label": "human", "model": "human", "id": qid,
                  "question": h["question"], "text": h["human_answer"]})

rnd = random.Random(9000 + a.pass_n)
for _ in range(4000):
    rnd.shuffle(items)
    if all(len({it["id"] for it in items[b:b + PER_BATCH]})
           == len(items[b:b + PER_BATCH])
           for b in range(0, len(items), PER_BATCH)):
        break
else:
    sys.exit("FATAL: no batch layout without a within-batch question collision")

outdir = a.out or f"{pdir}/{a.split}/pass{a.pass_n}"
bdir = f"{outdir}/batches"
os.makedirs(bdir, exist_ok=True)
with open(f"{outdir}/key.jsonl", "w") as f:
    for i, it in enumerate(items):
        f.write(json.dumps({"item": f"I{i:03d}", "label": it["label"],
                            "model": it["model"], "id": it["id"]}) + "\n")

n = 0
for b in range(0, len(items), PER_BATCH):
    chunk = items[b:b + PER_BATCH]
    body = [
        "# Post classification", "",
        f"Below are {len(chunk)} separate posts, each submitted to the subreddit",
        "r/AmItheAsshole. Each was either written by a human redditor between 2018",
        "and 2019, or by a language model in 2026 imitating that style.", "",
        "They are unrelated to each other. Judge each on its own; there is no",
        "reference text and no fixed proportion of human to AI.", "",
    ]
    for j, it in enumerate(chunk):
        body += ["", "=" * 76, f"ITEM I{b + j:03d}", "=" * 76,
                 f"TITLE: {it['question']}", "", "POST:", it["text"], ""]
    open(f"{bdir}/batch_{n:02d}.txt", "w").write("\n".join(body) + "\n")
    n += 1

n_ai = sum(1 for it in items if it["label"] == "ai")
assert len({(it["id"], it["model"]) for it in items if it["label"] == "ai"}) == n_ai
assert len({it["id"] for it in items if it["label"] == "human"}) == len(items) - n_ai
assert not os.path.exists(f"{bdir}/key.jsonl")

print(f"{len(items)} items ({n_ai} ai / {len(items) - n_ai} human), {n} batches "
      f"of {PER_BATCH} -> {outdir}")
print("blend: " + ", ".join(
    f"{m.split('/')[-1]} {sum(1 for it in items if it['model'] == m)}"
    for m in sorted({it["model"] for it in items if it["label"] == "ai"})))
