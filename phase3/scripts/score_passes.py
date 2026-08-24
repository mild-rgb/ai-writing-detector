"""Score the five judge replicates of one prompt and report the spread.

phase 1 method note 8: report the SD, never the mean alone. Two conditions
averaging 52% -- one with judges at 45/48/55, one at 22/53/91 -- are not the
same result, and every conclusion in its v1-v7 loop was a mean without an SD.

phase3/study/judge_compare/RESULTS.md gives that rule teeth here: re-running the
SAME judging route on the SAME 150 items moved detection by 13 points and the
false-positive rate by 9, while balanced accuracy moved 2. So this reports
balanced accuracy as the headline, its SD across passes as the second number,
and the components underneath where they belong.

    python3 phase3/scripts/score_passes.py PROMPT_DIR
"""
import argparse
import glob
import json
import math
import os
import statistics
import sys


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * (c - h), 100 * (c + h))


ap = argparse.ArgumentParser()
ap.add_argument("prompt_dir")
ap.add_argument("--split", default="bench",
                help="`bench` is the frozen confirmation; `dev` is descriptive "
                     "only -- the prompts were tuned on those questions and no "
                     "version decision may be made from a dev judge number.")
ap.add_argument("--json", dest="json_out")
a = ap.parse_args()

pdir = a.prompt_dir.rstrip("/")
passes = sorted(glob.glob(f"{pdir}/{a.split}/pass*"))
if not passes:
    sys.exit(f"FATAL: no {a.split} passes under {pdir}/{a.split}/")
if a.split != "bench":
    print(f"*** {a.split.upper()} PASS -- DESCRIPTIVE ONLY, NOT A HELD-OUT "
          f"RESULT ***\n")

rows, per_item = [], {}
for pd in passes:
    name = os.path.basename(pd)
    key = {json.loads(l)["item"]: json.loads(l) for l in open(f"{pd}/key.jsonl")}
    files = sorted(glob.glob(f"{pd}/*/pred_*.json"))
    if not files:
        print(f"  {name}: no predictions yet", file=sys.stderr)
        continue
    route = os.path.basename(os.path.dirname(files[0]))
    P = {}
    for f in files:
        try:
            obj = json.load(open(f))
        except json.JSONDecodeError as e:
            sys.exit(f"FATAL {f}: {e}. Not skipping it.")
        for p in obj["predictions"]:
            if p["item"] in P:
                sys.exit(f"FATAL: {p['item']} predicted twice in {name}")
            P[p["item"]] = p["verdict"]
    if len(P) != len(key):
        sys.exit(f"FATAL: {name} has {len(P)} predictions for {len(key)} items. "
                 f"Assert the count before scoring -- three phase-1 figures were "
                 f"computed on fewer items than they claimed.")
    nai = sum(1 for k in key.values() if k["label"] == "ai")
    nhu = len(key) - nai
    det = sum(1 for i, k in key.items() if k["label"] == "ai" and P[i] == "ai")
    fp = sum(1 for i, k in key.items() if k["label"] == "human" and P[i] == "ai")
    bal = 50 * (det / nai + 1 - fp / nhu)
    rows.append({"pass": name, "route": route, "n": len(key),
                 "detection": 100 * det / nai, "fpr": 100 * fp / nhu,
                 "balanced": bal, "det_k": det, "n_ai": nai,
                 "fp_k": fp, "n_hu": nhu})
    for i, k in key.items():
        # keyed by document, not by item id: item ids are reshuffled per pass
        per_item.setdefault((k["id"], k["label"], k["model"]), []).append(P[i])

if not rows:
    sys.exit("FATAL: no scored passes")

print(f"{pdir}\n")
print(f"{'pass':<8} {'route':<10} {'n':>4} {'detection':>10} {'fpr':>8} {'balanced':>9}")
for r in rows:
    print(f"{r['pass']:<8} {r['route']:<10} {r['n']:>4} {r['detection']:>9.1f}% "
          f"{r['fpr']:>7.1f}% {r['balanced']:>8.1f}%")

bal = [r["balanced"] for r in rows]
det = [r["detection"] for r in rows]
fpr = [r["fpr"] for r in rows]
sd = statistics.stdev if len(rows) > 1 else (lambda x: float("nan"))
print(f"\nbalanced accuracy   mean {statistics.mean(bal):.1f}%   SD {sd(bal):.1f}"
      f"   range [{min(bal):.1f}, {max(bal):.1f}]")
print(f"detection           mean {statistics.mean(det):.1f}%   SD {sd(det):.1f}")
print(f"false positives     mean {statistics.mean(fpr):.1f}%   SD {sd(fpr):.1f}")

# Pooled across passes, which is the tightest interval available but treats the
# same document judged five times as five observations. Quoted as pooled, and
# the per-pass SD above is what says whether that is fair.
tot_det = sum(r["det_k"] for r in rows)
tot_nai = sum(r["n_ai"] for r in rows)
tot_fp = sum(r["fp_k"] for r in rows)
tot_nhu = sum(r["n_hu"] for r in rows)
dl, dh = wilson(tot_det, tot_nai)
fl, fh = wilson(tot_fp, tot_nhu)
print(f"\npooled over {len(rows)} passes ({tot_nai} ai judgements, {tot_nhu} human):")
print(f"  detection      {100*tot_det/tot_nai:5.1f}% [{dl:.1f}, {dh:.1f}]")
print(f"  false positive {100*tot_fp/tot_nhu:5.1f}% [{fl:.1f}, {fh:.1f}]")
print(f"  balanced       {50*(tot_det/tot_nai + 1 - tot_fp/tot_nhu):5.1f}%")

# Judge clustering, measured per document rather than per judge: with five
# passes over the same documents, the fraction of documents on which the passes
# unanimously agree is the cleanest available read on whether the verdict is a
# property of the text or of the reader.
unan = sum(1 for v in per_item.values() if len(set(v)) == 1)
print(f"\nunanimous across passes: {100*unan/len(per_item):.1f}% of "
      f"{len(per_item)} documents")

by_model = {}
for (qid, label, model), v in per_item.items():
    if label == "ai":
        by_model.setdefault(model, []).extend(1 if x == "ai" else 0 for x in v)
print("\ndetection by generator (pooled over passes):")
for m in sorted(by_model):
    v = by_model[m]
    lo, hi = wilson(sum(v), len(v))
    print(f"  {m.split('/')[-1]:<28} {100*sum(v)/len(v):5.1f}% [{lo:4.1f}, {hi:4.1f}]  n={len(v)}")

if a.json_out:
    json.dump({"passes": rows, "balanced_mean": statistics.mean(bal),
               "balanced_sd": sd(bal) if len(rows) > 1 else None,
               "unanimous": unan / len(per_item)}, open(a.json_out, "w"), indent=1)
