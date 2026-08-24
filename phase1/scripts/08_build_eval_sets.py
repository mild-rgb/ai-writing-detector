"""Partition the 1000 sampled questions into fixed, disjoint evaluation sets.

The v1-v7 loop drew fresh questions per iteration to avoid tuning to specific
items. That removed a bias risk but made every version-to-version comparison a
between-subjects one, where the judge-instance component is not differenced out.
A fixed DEV set makes comparisons paired: the same questions, and the same human
answers, under every candidate prompt.

  BURNED   questions already used in iterations 1-7. Never reused: judges have
           seen these human answers, and the prompts were written while looking
           at them.
  DEV      fixed set for all prompt-version comparisons. Reused deliberately.
  HELDOUT  touched once, at the end, to confirm the chosen config. Every look
           costs some of its value.
  POOL     remainder, feeding the final 1000-question dataset.
"""
import json
import random
import sys

N_DEV = int(sys.argv[1]) if len(sys.argv) > 1 else 150
N_HELD = int(sys.argv[2]) if len(sys.argv) > 2 else 150
SEED = 20260821
BURNED_THROUGH = 150            # iterations 1-7 consumed questions 0000-0149

rows = [json.loads(l) for l in open("data/interim/questions_1000.jsonl")]
burned = [r for r in rows if int(r["id"].split("-")[1]) < BURNED_THROUGH]
rest = [r for r in rows if int(r["id"].split("-")[1]) >= BURNED_THROUGH]

rnd = random.Random(SEED)
rnd.shuffle(rest)
dev, held, pool = rest[:N_DEV], rest[N_DEV:N_DEV + N_HELD], rest[N_DEV + N_HELD:]

for name, rs in [("burned", burned), ("dev", dev), ("heldout", held), ("pool", pool)]:
    with open(f"data/eval/{name}.jsonl", "w") as f:
        for r in rs:
            f.write(json.dumps(r) + "\n")

ids = [{r["id"] for r in s} for s in (burned, dev, held, pool)]
assert not (ids[0] & ids[1] or ids[1] & ids[2] or ids[2] & ids[3] or ids[1] & ids[3]), "overlap"
print(f"{'set':<10} {'n':>5}   {'median words':>13}")
print("-" * 32)
for name, rs in [("burned", burned), ("dev", dev), ("heldout", held), ("pool", pool)]:
    w = sorted(r["human_words"] for r in rs)
    print(f"{name:<10} {len(rs):>5}   {w[len(w)//2]:>13}")
print(f"\ntotal {sum(len(s) for s in (burned, dev, held, pool))}, all disjoint")
