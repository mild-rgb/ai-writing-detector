"""Freeze the corpus question -> generator assignment, once.

Mirrors phase3/scripts/build_corpus.py's rule: one ai document per question,
generator assigned round-robin AT SAMPLING TIME so the blend is balanced within
every split rather than only overall, and provenance is decided once and
survives into phase-4 attribution.

Differs from ELI5 in one respect, deliberately. ELI5 folded its bench and
heldout questions back into the corpus because they were never actually used.
Here dev (60) and bench (70) HAVE been generated against and judged, so they are
excluded, and heldout (60) is excluded to keep one clean untouched split. The
corpus is drawn from the 6,153-question pool only.

    python3 phase3/scripts/aita_06_corpus_assignment.py --n 2900
"""
import argparse
import json
import os
import random
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = f"{ROOT}/phase3/aita/data/corpus_assignment.json"
SEED = 20260825

MODELS = [
    "qwen/qwen3.8-max", "deepseek/deepseek-v4-pro",
    "nvidia/nemotron-3.5-lightning", "google/gemini-3.7-flash",
    "openai/gpt-5.6-luna-pro", "z-ai/glm-5.3", "x-ai/grok-4.6",
]

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=2900)
ap.add_argument("--out", default=OUT)
a = ap.parse_args()

part = json.load(open(f"{ROOT}/phase3/aita/data/aita_partition.json"))
pool = sorted(part["pool"])
used = set(part["dev"]) | set(part["bench"]) | set(part["heldout"])
assert not (set(pool) & used), "pool overlaps an evaluation split"

rnd = random.Random(SEED)
picked = list(pool)
rnd.shuffle(picked)
picked = sorted(picked[:a.n])
rnd.shuffle(picked)
assign = {q: MODELS[i % len(MODELS)] for i, q in enumerate(picked)}

json.dump({"models": MODELS, "n": len(assign), "seed": SEED,
           "excluded": {"dev": len(part["dev"]), "bench": len(part["bench"]),
                        "heldout": len(part["heldout"])},
           "assignment": assign}, open(a.out, "w"), indent=1)
print(f"{len(assign)} questions assigned from a {len(pool)}-question pool "
      f"(dev/bench/heldout excluded) -> {a.out}")
for m, c in sorted(Counter(assign.values()).items()):
    print(f"  {m.split('/')[-1]:<28} {c}")
