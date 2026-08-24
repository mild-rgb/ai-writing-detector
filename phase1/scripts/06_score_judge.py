"""Score per-model judge predictions against their keys."""
import glob
import json
import os
import sys

ITERDIR = sys.argv[1]
rows, cues = [], {}
for keyf in sorted(glob.glob(f"{ITERDIR}/key_*.jsonl")):
    slug = os.path.basename(keyf)[len("key_"):-len(".jsonl")]
    predf = f"{ITERDIR}/pred_{slug}.json"
    if not os.path.exists(predf):
        print(f"  (no predictions yet for {slug})")
        continue
    key = [json.loads(l) for l in open(keyf)]
    d = json.load(open(predf))
    pred = d.get("predictions", d if isinstance(d, list) else [])
    cues[slug] = d.get("overall_cues", [])
    c = n = 0
    for k in key:
        p = next((x for x in pred if x.get("trial") == k["trial"]), None)
        if p is None:
            continue
        c += str(p.get("guess", "")).strip().upper()[:1] == k["ai_side"]
        n += 1
    from collections import Counter
    gc = Counter(str(p.get("guess", "")).strip().upper()[:1] for p in pred)
    skew = abs(gc["A"] - gc["B"]) / max(1, gc["A"] + gc["B"])
    rows.append((slug, c, n, gc["A"], gc["B"], skew))

tc = sum(r[1] for r in rows)
tn = sum(r[2] for r in rows)
print(f"{'model':<20} {'correct':>9} {'accuracy':>10} {'guess A/B':>11} {'skew':>7}")
print("-" * 62)
for slug, c, n, ga, gb, skew in rows:
    warn = "  <-- position bias" if skew >= 0.6 else ""
    print(f"{slug:<20} {c:>4}/{n:<4} {c/n*100 if n else 0:>9.1f}% "
          f"{ga:>5}/{gb:<5} {skew:>6.0%}{warn}")
print("-" * 62)
acc = tc / tn * 100 if tn else 0
print(f"{'OVERALL':<20} {tc:>4}/{tn:<4} {acc:>9.1f}%   (chance 50%)")
# Detectability is distance from chance in EITHER direction. A judge scoring
# 20% is not fooled -- it has an inverted but perfectly usable signal, and a
# trained detector would simply learn the rule backwards and score 80%.
print(f"{'DETECTABILITY':<20} {'':>9} {abs(acc - 50):>9.1f} points from chance"
      + ("   <-- INVERTED: AI reads MORE human than the humans" if acc < 50 else ""))
print()
for slug, cl in cues.items():
    print(f"cues [{slug}]:")
    for i, c in enumerate(cl, 1):
        print(f"   {i}. {c}")
    print()
