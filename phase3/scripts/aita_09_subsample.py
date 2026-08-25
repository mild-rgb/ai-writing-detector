"""Draw the early-read subsample: N questions per model, SEEDED and RANDOM.

Not the first N by id. Reddit base36 post ids sort chronologically -- the
property this corpus uses to date itself -- so an id-ordered head takes a
model's OLDEST questions. Measured on glm's 414, the id-sorted first 150 spans
only 55% of that model's date range.

That would corrupt the one thing the early read measures. AITA topics drift over
time, so a temporally clustered subsample shows narrower scenario spread for
reasons unrelated to the generator, biasing the near-duplicate and
nearest-neighbour figures toward a false alarm -- and because the human control
is drawn on the same titles it moves in lockstep, hiding the artefact rather
than exposing it.

The draw is a subset of the frozen corpus assignment, so these are corpus
documents and the remainder is exactly the rest of the corpus.

    python3 phase3/scripts/aita_09_subsample.py --per-model 150
"""
import argparse
import json
import os
import random
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SEED = 20260826

ap = argparse.ArgumentParser()
ap.add_argument("--per-model", type=int, default=150)
ap.add_argument("--assignment", default=f"{ROOT}/phase3/aita/data/corpus_assignment.json")
ap.add_argument("--humans", default=f"{ROOT}/phase3/aita/data/aita_human.jsonl")
ap.add_argument("--out", default=f"{ROOT}/phase3/aita/data/early_corpus.jsonl")
a = ap.parse_args()

A = json.load(open(a.assignment))
humans = {json.loads(l)["id"]: json.loads(l) for l in open(a.humans)}
by = defaultdict(list)
for q, m in A["assignment"].items():
    by[m].append(q)

rnd = random.Random(SEED)
picked = []
for m in sorted(by):
    qs = sorted(by[m])            # sort first so the shuffle is reproducible
    rnd.shuffle(qs)
    picked += [(q, m) for q in qs[:a.per_model]]
picked.sort()

with open(a.out, "w") as f:
    for q, m in picked:
        r = humans[q]
        f.write(json.dumps({"id": r["id"], "q_id": r["q_id"],
                            "question": r["question"],
                            "human_answer": r["human_answer"],
                            "human_words": r["human_words"], "model": m}) + "\n")

meta = a.out.replace(".jsonl", "_seed.json")
json.dump({"seed": SEED, "per_model": a.per_model, "n": len(picked),
           "assignment": os.path.basename(a.assignment),
           "note": "seeded random draw per model, NOT id-ordered; ids are "
                   "chronological so an id-ordered head is temporally clustered"},
          open(meta, "w"), indent=1)

print(f"{len(picked)} questions ({a.per_model}/model, seed {SEED}) -> {a.out}")
for m, c in sorted(Counter(m for _, m in picked).items()):
    print(f"  {m.split('/')[-1]:<28} {c}")


def n36(s):
    try:
        return int(s, 36)
    except Exception:
        return None


glm = sorted(q for q, m in picked if m == "z-ai/glm-5.3")
allglm = sorted(by["z-ai/glm-5.3"])
sp = (n36(glm[-1]) - n36(glm[0])) / (n36(allglm[-1]) - n36(allglm[0]))
print(f"\ntemporal-coverage check (glm): the draw spans {100*sp:.1f}% of that "
      f"model's id range, against 55.3% for an id-ordered head of the same size")
