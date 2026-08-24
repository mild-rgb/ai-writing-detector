"""Turn a continuous margin file into prediction files `14_score_single.py` reads.

The small-model detector produces a real-valued score per item, not a verdict.
Everything downstream in this repo speaks the judge format, so this applies the
frozen threshold and writes `pred_NN.json` files that the existing scorer can
consume unmodified -- which is the point. A separate scorer written for this run
would be a second implementation of the same statistics, and the value of
reusing `14_score_single.py` is that its item-count assertions and duplicate
checks apply to these numbers on exactly the same terms as to Haiku's and
Sonnet's.

    python3 scripts/15_margins_to_pred.py study/small/margins_long2.json \
        study/single/long2 lfm2 --threshold 4.6883

The threshold is a required argument with no default. It is tuned on dev, and a
default would let it travel to a confirmation set silently.
"""
import argparse
import json
import os

ap = argparse.ArgumentParser()
ap.add_argument("margins")
ap.add_argument("keydir", help="benchmark dir holding key.jsonl")
ap.add_argument("detector", help="subdirectory name to write predictions into")
ap.add_argument("--threshold", type=float, required=True)
ap.add_argument("--per-file", type=int, default=5)
ap.add_argument("--axis", default="a4_generic")
a = ap.parse_args()

M = json.load(open(a.margins))
margins = dict(zip(M["item"], M["margin"]))
key = [json.loads(l) for l in open(f"{a.keydir}/key.jsonl")]

missing = [k["item"] for k in key if k["item"] not in margins]
extra = sorted(set(margins) - {k["item"] for k in key})
if missing or extra:
    raise SystemExit(f"margin file does not match the key: {len(missing)} items "
                     f"missing ({missing[:6]}), {len(extra)} unknown ({extra[:6]})")

# Confidence from distance to the threshold, split at the tertiles of |margin -
# t| over this run. It is a monotone restatement of the score, not new
# information -- but it lets the existing accuracy-by-confidence check run, and
# a detector whose confidence carries no signal is worth knowing about (Haiku's
# did not: 59% at "high" against 57% at "medium").
dist = sorted(abs(margins[k["item"]] - a.threshold) for k in key)
q1, q2 = dist[len(dist) // 3], dist[2 * len(dist) // 3]

out = f"{a.keydir}/{a.detector}"
os.makedirs(out, exist_ok=True)
for old in os.listdir(out):
    if old.startswith("pred_") and old.endswith(".json"):
        os.remove(os.path.join(out, old))

n = 0
for c in range(0, len(key), a.per_file):
    preds = []
    for k in key[c:c + a.per_file]:
        m = margins[k["item"]]
        d = abs(m - a.threshold)
        preds.append({
            "item": k["item"],
            "verdict": "ai" if m > a.threshold else "human",
            "confidence": "high" if d > q2 else "medium" if d > q1 else "low",
            "reason": f"{a.axis} margin {m:+.4f} vs threshold {a.threshold:+.4f}",
            "margin": m,
        })
    with open(f"{out}/pred_{n:02d}.json", "w") as f:
        json.dump({"predictions": preds}, f, indent=1)
    n += 1

flagged = sum(margins[k["item"]] > a.threshold for k in key)
print(f"{len(key)} items -> {n} files in {out}/")
print(f"threshold {a.threshold:+.4f}  flagged 'ai' {flagged}/{len(key)}")
print(f"confidence tertiles at |margin-t| = {q1:.3f} / {q2:.3f}")
