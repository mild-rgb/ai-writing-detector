"""Are two judging routes the same instrument?

Phase 1's published numbers came from Claude Code subagent judges. If API judges
read the same documents differently, no phase-3 number is comparable to a
phase-1 number. This compares two prediction sets item by item -- agreement,
Cohen's kappa, and a McNemar test on the discordant pairs, which is the right
test for two raters on the SAME items (a two-proportion test would throw away
the pairing and overstate the uncertainty).

    python3 phase3/scripts/compare_judges.py DIR arm_a arm_b
"""
import argparse
import glob
import json
import math
import sys


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * (c - h), 100 * (c + h))


def load(d, arm):
    P = {}
    files = sorted(glob.glob(f"{d}/{arm}/pred_*.json"))
    if not files:
        sys.exit(f"FATAL: no predictions in {d}/{arm}")
    for f in files:
        obj = json.load(open(f))
        for p in obj["predictions"]:
            if p["item"] in P:
                sys.exit(f"FATAL: item {p['item']} predicted twice in {arm}")
            P[p["item"]] = p
    return P


def binom_two_sided(k, n, p=0.5):
    def pmf(i):
        return math.comb(n, i) * p ** i * (1 - p) ** (n - i)
    obs = pmf(k)
    return min(1.0, sum(pmf(i) for i in range(n + 1) if pmf(i) <= obs + 1e-12))


ap = argparse.ArgumentParser()
ap.add_argument("dir")
ap.add_argument("arms", nargs=2)
a = ap.parse_args()

key = {json.loads(l)["item"]: json.loads(l) for l in open(f"{a.dir}/key.jsonl")}
A, B = (load(a.dir, arm) for arm in a.arms)
items = sorted(set(A) & set(B))
if len(items) != len(key):
    sys.exit(f"FATAL: {len(items)} items in common, key has {len(key)}")

n = len(items)
agree = sum(1 for i in items if A[i]["verdict"] == B[i]["verdict"])
pa = agree / n
mA = sum(1 for i in items if A[i]["verdict"] == "ai") / n
mB = sum(1 for i in items if B[i]["verdict"] == "ai") / n
pe = mA * mB + (1 - mA) * (1 - mB)
kappa = (pa - pe) / (1 - pe) if pe < 1 else float("nan")

b_only = sum(1 for i in items if A[i]["verdict"] != "ai" and B[i]["verdict"] == "ai")
a_only = sum(1 for i in items if A[i]["verdict"] == "ai" and B[i]["verdict"] != "ai")
disc = a_only + b_only
p_mcnemar = binom_two_sided(min(a_only, b_only), disc) if disc else 1.0

print(f"{n} items judged by both\n")
print(f"per-item agreement   {100*pa:.1f}%  ({agree}/{n})")
print(f"Cohen's kappa        {kappa:.3f}")
print(f"calls 'ai'           {a.arms[0]} {100*mA:.1f}%   {a.arms[1]} {100*mB:.1f}%")
print(f"discordant pairs     {a.arms[0]}-only {a_only}, {a.arms[1]}-only {b_only}"
      f"   McNemar p = {p_mcnemar:.4f}\n")

for arm, P in zip(a.arms, (A, B)):
    det = sum(1 for i in items if key[i]["label"] == "ai" and P[i]["verdict"] == "ai")
    nai = sum(1 for i in items if key[i]["label"] == "ai")
    fp = sum(1 for i in items if key[i]["label"] == "human" and P[i]["verdict"] == "ai")
    nhu = n - nai
    bal = 50 * (det / nai + 1 - fp / nhu)
    dlo, dhi = wilson(det, nai)
    flo, fhi = wilson(fp, nhu)
    print(f"{arm:<10} detection {100*det/nai:5.1f}% [{dlo:4.1f},{dhi:4.1f}]   "
          f"fpr {100*fp/nhu:5.1f}% [{flo:4.1f},{fhi:4.1f}]   balanced {bal:5.1f}%")

for label in ("ai", "human"):
    sub = [i for i in items if key[i]["label"] == label]
    ag = sum(1 for i in sub if A[i]["verdict"] == B[i]["verdict"]) / len(sub)
    print(f"  agreement on {label:<6} items: {100*ag:.1f}%  (n={len(sub)})")
