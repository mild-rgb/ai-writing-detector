"""Score a study phase: R replicate judges x N trials per model, pooled,
with Wilson CIs, exact binomial tests and Holm correction across the phase."""
import glob
import json
import math
import os
import sys
from math import comb
from collections import defaultdict

def binom_p(k, n, p=0.5):
    pmf = lambda i: comb(n, i) * p ** i * (1 - p) ** (n - i)
    obs = pmf(k)
    return min(1.0, sum(pmf(i) for i in range(n + 1) if pmf(i) <= obs * (1 + 1e-9)))

def wilson(k, n, z=1.959964):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z / d * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return 100 * (c - h), 100 * (c + h)

DIRS = sys.argv[1:]
cells, reps = [], defaultdict(list)
for d in DIRS:
    cond = os.path.basename(d.rstrip("/"))
    for keyf in sorted(glob.glob(f"{d}/key_*.jsonl")):
        slug = os.path.basename(keyf)[4:-6]
        key = {k["trial"]: k for k in map(json.loads, open(keyf))}
        tot_c = tot_n = 0
        for predf in sorted(glob.glob(f"{d}/pred_{slug}_r*.json")):
            try:
                pred = json.load(open(predf)).get("predictions", [])
            except Exception:
                continue
            c = sum(1 for p in pred if p["trial"] in key
                    and str(p.get("guess", "")).strip().upper()[:1] == key[p["trial"]]["ai_side"])
            n = sum(1 for p in pred if p["trial"] in key)
            if n:
                reps[(cond, slug)].append(c / n * 100)
                tot_c += c; tot_n += n
        if tot_n:
            cells.append((cond, slug, tot_c, tot_n))

if not cells:
    sys.exit("no predictions yet")
print(f"{'condition':<12}{'model':<18}{'pooled':>10}{'acc':>8}   {'95% CI':>16}"
      f"{'|d-50|':>8}{'replicates':>22}")
print("-" * 96)
ps = []
for cond, slug, c, n in cells:
    lo, hi = wilson(c, n); p = binom_p(c, n); ps.append(p)
    r = ",".join(f"{x:.0f}" for x in reps[(cond, slug)])
    print(f"{cond:<12}{slug:<18}{f'{c}/{n}':>10}{c/n*100:>7.1f}%   [{lo:>5.1f},{hi:>5.1f}]"
          f"{abs(c/n*100-50):>7.1f} {r:>21}")
m = len(ps); order = sorted(range(m), key=lambda i: ps[i]); holm = [0]*m; run = 0
for rank, i in enumerate(order):
    run = max(run, min(1.0, (m - rank) * ps[i])); holm[i] = run
print(f"\nHolm-corrected across {m} cells (H0: judge at chance):")
for i, (cond, slug, c, n) in enumerate(cells):
    verdict = "DIFFERS FROM CHANCE" if holm[i] < .05 else "consistent with chance"
    print(f"  {cond:<12}{slug:<18}{c/n*100:>6.1f}%  adj p={holm[i]:.4f}  {verdict}")
