"""Partition the long-form corpus into fixed, disjoint evaluation sets.

phase1/scripts/08_build_eval_sets.py partitioned the short 1,000 questions and
nothing ever partitioned the long-form 1,000. Six prompts evaluated on
unpartitioned questions would repeat the v1-v7 mistake: fresh questions per
iteration makes every comparison between-subjects, and phase 1 measured
question-set effects at 2-2.5x binomial -- larger than most of the effects being
compared.

204 of the 1,000 long-form ids are already touched by phase 1 (study/longform,
study/longform3, and the long/long2/long3 benchmark keys). Those are BURNED:
their human answers have been judged and prompts were written while looking at
them. The remaining 796 are partitioned here, once, and the split is written to
disk so it cannot drift between runs.

  DEV      60 questions. Every prompt iteration runs on these. Reused
           deliberately so prompt comparisons are paired.
  BENCH    70 questions. The frozen confirmation set: one ai document and one
           human document each, blend composition fixed across all six prompts.
  HELDOUT  60 questions. Untouched until the end, if at all.
  POOL     the remainder.
"""
import glob
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "phase3/data/longform_partition.json")
SEED = 20260823
N_DEV, N_BENCH, N_HELD = 60, 70, 60

rows = [json.loads(l) for l in
        open(f"{ROOT}/phase1/data/interim/questions_longform.jsonl")]
burned = set()
for f in [f"{ROOT}/phase1/study/longform/questions.jsonl",
          f"{ROOT}/phase1/study/longform3/questions.jsonl"]:
    if os.path.exists(f):
        burned |= {json.loads(l)["id"] for l in open(f)}
for f in glob.glob(f"{ROOT}/phase1/study/single/long*/key.jsonl"):
    burned |= {json.loads(l)["id"] for l in open(f)}

clean = sorted([r for r in rows if r["id"] not in burned], key=lambda r: r["id"])
if len(clean) < N_DEV + N_BENCH + N_HELD:
    sys.exit(f"FATAL: only {len(clean)} clean questions, need "
             f"{N_DEV + N_BENCH + N_HELD}")

rnd = random.Random(SEED)
rnd.shuffle(clean)
dev = sorted(r["id"] for r in clean[:N_DEV])
bench = sorted(r["id"] for r in clean[N_DEV:N_DEV + N_BENCH])
held = sorted(r["id"] for r in clean[N_DEV + N_BENCH:N_DEV + N_BENCH + N_HELD])
pool = sorted(r["id"] for r in clean[N_DEV + N_BENCH + N_HELD:])

assert len({*dev, *bench, *held, *pool}) == len(clean), "partition overlaps"
assert not ({*dev, *bench, *held, *pool} & burned), "burned id leaked in"

part = {"seed": SEED, "burned": sorted(burned & {r["id"] for r in rows}),
        "dev": dev, "bench": bench, "heldout": held, "pool": pool}
if os.path.exists(OUT):
    old = json.load(open(OUT))
    if any(old[k] != part[k] for k in ("dev", "bench", "heldout")):
        sys.exit(f"FATAL: {OUT} exists and differs. The partition is permanent; "
                 f"refusing to overwrite it. Delete it deliberately if you mean "
                 f"to repartition, and know that every prior number was measured "
                 f"on the old split.")
with open(OUT, "w") as fh:
    json.dump(part, fh, indent=1)

words = {r["id"]: r["human_words"] for r in rows}
for name in ("dev", "bench", "heldout", "pool"):
    ids = part[name]
    w = sorted(words[i] for i in ids)
    print(f"{name:<8} {len(ids):>4} questions   human words median {w[len(w)//2]}"
          f"  [{w[0]}, {w[-1]}]")
print(f"burned   {len(part['burned']):>4} questions (phase 1 saw these)")
print(f"-> {OUT}")
