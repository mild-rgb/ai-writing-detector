"""Partition the AITA human corpus, mirroring partition_longform.py.

Same sizes and the same reason: prompt iterations run on DEV every round so
comparisons are paired within question, because phase 1 measured question-set
effects at 2-2.5x binomial -- larger than most of the effects being compared.
Nothing in the AITA set is burned; it has never been judged or read.

  DEV      60   every prompt iteration
  BENCH    70   frozen confirmation, one ai + one human document each
  HELDOUT  60   untouched
  POOL     the remainder
"""
import json
import os
import random

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = f"{ROOT}/phase3/aita/data/aita_human.jsonl"
OUT = f"{ROOT}/phase3/aita/data/aita_partition.json"
SEED = 20260824
N_DEV, N_BENCH, N_HELD = 60, 70, 60

ids = sorted(json.loads(l)["id"] for l in open(SRC))
rnd = random.Random(SEED)
shuf = list(ids)
rnd.shuffle(shuf)
part = {"dev": sorted(shuf[:N_DEV]),
        "bench": sorted(shuf[N_DEV:N_DEV + N_BENCH]),
        "heldout": sorted(shuf[N_DEV + N_BENCH:N_DEV + N_BENCH + N_HELD]),
        "pool": sorted(shuf[N_DEV + N_BENCH + N_HELD:]),
        "seed": SEED, "source": "phase3/aita/data/aita_human.jsonl"}
assert len(set(part["dev"]) | set(part["bench"]) | set(part["heldout"])
           | set(part["pool"])) == len(ids), "partition is not disjoint"
json.dump(part, open(OUT, "w"), indent=1)
print(f"{len(ids)} AITA questions -> dev {len(part['dev'])}, "
      f"bench {len(part['bench'])}, heldout {len(part['heldout'])}, "
      f"pool {len(part['pool'])}  -> {OUT}")
