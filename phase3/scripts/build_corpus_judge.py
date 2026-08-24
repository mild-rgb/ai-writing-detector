"""Build a single-document judge set from the corpus answers generated so far.

Descriptive read on the floor generator before the rest of the corpus is built.
Same design as every other single-document pass in this project: one document
alone, no reference text, no fixed class proportion, and no two answers to the
same question inside one batch.

The corpus TEST split is excluded by default so it stays clean for the detector.
Each pass reshuffles which items share a batch but never changes the item set,
so the passes are independent judge draws over identical documents.

    python3 phase3/scripts/build_corpus_judge.py --per-model 10 --pass 0
"""
import argparse
import collections
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
Q = f"{ROOT}/phase3/corpus/questions.jsonl"
A = f"{ROOT}/phase3/corpus/answers.jsonl"
OUT = f"{ROOT}/phase3/corpus/judge"
PER_BATCH = 5

ap = argparse.ArgumentParser()
ap.add_argument("--per-model", type=int, default=10)
ap.add_argument("--pass", dest="pass_n", type=int, default=0)
ap.add_argument("--include-test", action="store_true")
a = ap.parse_args()

questions = {json.loads(l)["id"]: json.loads(l) for l in open(Q) if l.strip()}
answers = {}
for line in open(A):
    if line.strip():
        r = json.loads(line)
        answers[r["id"]] = r

pool = collections.defaultdict(list)
for qid, ans in sorted(answers.items()):
    q = questions.get(qid)
    if q is None:
        continue
    if q["split"] == "test" and not a.include_test:
        continue
    pool[ans["model"]].append(qid)

short = {m: len(v) for m, v in pool.items() if len(v) < a.per_model}
if short:
    sys.exit(f"FATAL: {short} have fewer than {a.per_model} eligible documents. "
             f"A blend that is short in one generator is a different blend.")

rnd = random.Random(4242)
picked = []
for m in sorted(pool):
    ids = sorted(pool[m])
    rnd.shuffle(ids)
    picked += ids[:a.per_model]

items = []
for qid in picked:
    items.append({"label": "ai", "model": answers[qid]["model"], "id": qid,
                  "question": questions[qid]["question"],
                  "text": answers[qid]["text"]})
    items.append({"label": "human", "model": "human", "id": qid,
                  "question": questions[qid]["question"],
                  "text": questions[qid]["human_answer"]})

lay = random.Random(9000 + a.pass_n)
for _ in range(4000):
    lay.shuffle(items)
    if all(len({it["id"] for it in items[b:b + PER_BATCH]}) == PER_BATCH
           for b in range(0, len(items), PER_BATCH)):
        break
else:
    sys.exit("FATAL: no batch layout without a within-batch question collision")

outdir = f"{OUT}/pass{a.pass_n}"
bdir = f"{outdir}/batches"
os.makedirs(bdir, exist_ok=True)
with open(f"{outdir}/key.jsonl", "w") as f:
    for i, it in enumerate(items):
        f.write(json.dumps({"item": f"I{i:03d}", "label": it["label"],
                            "model": it["model"], "id": it["id"]}) + "\n")

n = 0
for b in range(0, len(items), PER_BATCH):
    chunk = items[b:b + PER_BATCH]
    body = ["# Comment classification", "",
            f"Below are {len(chunk)} separate comments, each posted in reply to a",
            "different question on the subreddit r/explainlikeimfive. Each was either",
            "written by a human redditor between 2011 and 2019, or by a language model",
            "in 2026 imitating that style.", "",
            "They are unrelated to each other. Judge each on its own; there is no",
            "reference text and no fixed proportion of human to AI.", ""]
    for j, it in enumerate(chunk):
        body += ["", "=" * 76, f"ITEM I{b + j:03d}", "=" * 76,
                 f"QUESTION: {it['question']}", "", "COMMENT:", it["text"], ""]
    open(f"{bdir}/batch_{n:02d}.txt", "w").write("\n".join(body) + "\n")
    n += 1

ai_docs = {it["id"] for it in items if it["label"] == "ai"}
assert len(ai_docs) == len(picked), "an ai document is duplicated"
print(f"pass{a.pass_n}: {len(items)} items ({len(picked)} ai / {len(picked)} human), "
      f"{n} batches of {PER_BATCH}, test split "
      f"{'INCLUDED' if a.include_test else 'excluded'}")
print("blend: " + ", ".join(f"{m.split('/')[-1]} {a.per_model}" for m in sorted(pool)))
