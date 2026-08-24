"""Fold the reserved bench/heldout questions into the corpus and re-split.

The 130 questions held back for adversarial evaluation are reclaimed: no
adversarial arm was ever generated on them and no judging ever touched them, so
the reservation bought nothing. The corpus's own test split is the holdout now.

Existing question ids and their generated answers are NOT disturbed. New
questions are appended with ids continuing the sequence and the round-robin
continuing from where it stopped, so the model balance holds over the full pool
rather than only over the first 2,772.

Splits are recomputed over everything, allocated PER MODEL so that each split is
model-balanced exactly rather than approximately: for each generator its
questions are shuffled and cut 80/10/10, and the pieces are unioned. A split
that is 80% of the corpus but 95% of one generator's documents would let a
detector learn the split rather than the task.

    python3 phase3/scripts/extend_corpus.py
"""
import collections
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
QUESTIONS = f"{ROOT}/phase3/corpus/questions.jsonl"
LONGFORM = f"{ROOT}/phase1/data/interim/questions_longform.jsonl"
PARTITION = f"{ROOT}/phase3/data/longform_partition.json"
SEED = 20260824
MODELS = [
    "x-ai/grok-4.6", "qwen/qwen3.8-max", "deepseek/deepseek-v4-pro",
    "nvidia/nemotron-3.5-lightning", "google/gemini-3.7-flash",
    "openai/gpt-5.6-luna-pro", "z-ai/glm-5.3",
]

rows = [json.loads(l) for l in open(QUESTIONS) if l.strip()]
have_qids = {r["q_id"] for r in rows}
print(f"corpus currently holds {len(rows)} questions")

part = json.load(open(PARTITION))
lf = {json.loads(l)["id"]: json.loads(l) for l in open(LONGFORM)}
reclaim = [lf[i] for i in part["bench"] + part["heldout"] if i in lf]
reclaim = [r for r in reclaim if r["q_id"] not in have_qids]
print(f"reclaiming {len(reclaim)} bench/heldout questions")
if not reclaim:
    sys.exit("nothing to reclaim -- already folded in")

rnd = random.Random(SEED)
order = list(MODELS)
rnd.shuffle(order)          # same shuffle as sample_corpus.py, same seed
start = len(rows)
for j, src in enumerate(sorted(reclaim, key=lambda r: r["q_id"])):
    i = start + j
    rows.append({"q_id": src["q_id"], "question": src["question"],
                 "selftext": src.get("selftext", ""),
                 "human_answer": src["human_answer"],
                 "human_words": src["human_words"], "score": src["score"],
                 "url": src["url"], "id": f"eli5c-{i:04d}",
                 "model": order[i % len(order)], "prompt": "floor",
                 "seed": f"eli5c-{i:04d}|{order[i % len(order)]}",
                 "split": None, "reclaimed": True})

# --- splits, allocated per model so each split is model-balanced -----------
by_model = collections.defaultdict(list)
for r in rows:
    by_model[r["model"]].append(r["id"])
srnd = random.Random(SEED + 1)
split_of = {}
for m in sorted(by_model):
    ids = sorted(by_model[m])
    srnd.shuffle(ids)
    n = len(ids)
    n_tr = int(round(0.80 * n))
    n_dev = int(round(0.10 * n))
    for k, qid in enumerate(ids):
        split_of[qid] = ("train" if k < n_tr else
                         "dev" if k < n_tr + n_dev else "test")
for r in rows:
    r["split"] = split_of[r["id"]]

with open(QUESTIONS, "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")

print(f"\nwrote {len(rows)} questions -> {QUESTIONS}")
print(f"{'split':<8} {'questions':>10}   per-generator counts")
for s in ("train", "dev", "test"):
    sub = [r for r in rows if r["split"] == s]
    c = collections.Counter(r["model"].split("/")[-1] for r in sub)
    print(f"{s:<8} {len(sub):>10}   " +
          " ".join(f"{k[:8]} {v}" for k, v in sorted(c.items())))
print(f"\nreclaimed questions needing generation: "
      f"{sum(1 for r in rows if r.get('reclaimed'))}")
