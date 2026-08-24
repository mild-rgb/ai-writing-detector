"""Score a single-document benchmark pass. Fails loudly on missing data.

Written after three reported figures in this project turned out to be computed
on fewer items than they claimed, none of it visible in the output: two
prediction files were malformed JSON and the scorer's `except: continue` skipped
them, and one detector's rows were double-counted after a restart appended
instead of resuming. This script asserts the expected item count, refuses to
parse-and-skip, and reports the effective document count next to the item count
so a duplicated class cannot masquerade as a larger sample.

    python3 scripts/14_score_single.py study/single/long2 haiku sonnet

Each detector is a subdirectory of predictions: <dir>/<detector>/pred_NN.json,
each holding {"predictions": [{"item", "verdict", "confidence", "reason"}, ...]}.
"""
import collections
import glob
import json
import math
import os
import sys

DIR = sys.argv[1]
DETECTORS = sys.argv[2:] or ["haiku", "sonnet"]


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * (c - h), 100 * (c + h))


key = {json.loads(l)["item"]: json.loads(l) for l in open(f"{DIR}/key.jsonl")}
n_items = len(key)
ai_docs = {(k["id"], k["model"]) for k in key.values() if k["label"] == "ai"}
hu_docs = {k["id"] for k in key.values() if k["label"] == "human"}
n_ai = sum(1 for k in key.values() if k["label"] == "ai")
n_hu = n_items - n_ai

print(f"{DIR}: {n_items} items ({n_ai} ai / {n_hu} human)")
print(f"  distinct documents: {len(ai_docs)} ai / {len(hu_docs)} human"
      f"   distinct questions: {len({k['id'] for k in key.values()})}")
if len(ai_docs) < n_ai or len(hu_docs) < n_hu:
    print(f"  WARNING: the item count overstates the sample -- "
          f"{n_ai - len(ai_docs)} ai and {n_hu - len(hu_docs)} human items are "
          f"repeats of a document already in the set. Intervals below on the "
          f"item count are too narrow; use the per-document rows.")
print()

failures = []
for det in DETECTORS:
    files = sorted(glob.glob(f"{DIR}/{det}/pred_*.json"))
    if not files:
        failures.append(f"{det}: no prediction files in {DIR}/{det}/")
        continue
    P = {}
    for f in files:
        try:
            preds = json.load(open(f))["predictions"]
        except Exception as e:                       # never skip silently
            failures.append(f"{det}: {os.path.basename(f)} did not parse -- {e}")
            continue
        for p in preds:
            if p["item"] in P:
                failures.append(f"{det}: {p['item']} predicted twice")
            P[p["item"]] = p
    missing = sorted(set(key) - set(P))
    extra = sorted(set(P) - set(key))
    if missing:
        failures.append(f"{det}: {len(missing)} items have no prediction "
                        f"({', '.join(missing[:8])}{'...' if len(missing) > 8 else ''})")
    if extra:
        failures.append(f"{det}: {len(extra)} predictions for unknown items "
                        f"({', '.join(extra[:8])})")
    if missing or extra:
        continue

    tp = sum(1 for i, k in key.items() if k["label"] == "ai" and P[i]["verdict"] == "ai")
    fp = sum(1 for i, k in key.items() if k["label"] == "human" and P[i]["verdict"] == "ai")
    rec, fpr = tp / n_ai, fp / n_hu
    prec = tp / (tp + fp) if tp + fp else float("nan")
    print(f"  {det}  (n={len(P)}, all items predicted)")
    print(f"    detection      {tp}/{n_ai} = {rec:6.1%}  "
          f"[{wilson(tp, n_ai)[0]:.1f}, {wilson(tp, n_ai)[1]:.1f}]")
    print(f"    false positive {fp}/{n_hu} = {fpr:6.1%}  "
          f"[{wilson(fp, n_hu)[0]:.1f}, {wilson(fp, n_hu)[1]:.1f}]")
    print(f"    precision      {prec:6.1%}")
    print(f"    balanced       {(rec + 1 - fpr) / 2:6.1%}")

    if len(hu_docs) < n_hu or len(ai_docs) < n_ai:
        # Collapse repeated documents to one verdict each (majority; a tie
        # counts as flagged, which is the conservative direction for an FPR).
        def collapse(label, keyfn):
            d = collections.defaultdict(list)
            for i, k in key.items():
                if k["label"] == label:
                    d[keyfn(k)].append(P[i]["verdict"])
            flagged = sum(1 for vs in d.values()
                          if any(v == "ai" for v in vs)
                          and sum(v == "ai" for v in vs) * 2 >= len(vs))
            return flagged, len(d)
        dtp, dn_ai = collapse("ai", lambda k: (k["id"], k["model"]))
        dfp, dn_hu = collapse("human", lambda k: k["id"])
        print(f"    per document:  detection {dtp}/{dn_ai} = {dtp/dn_ai:.1%} "
              f"[{wilson(dtp, dn_ai)[0]:.1f}, {wilson(dtp, dn_ai)[1]:.1f}]   "
              f"false positive {dfp}/{dn_hu} = {dfp/dn_hu:.1%} "
              f"[{wilson(dfp, dn_hu)[0]:.1f}, {wilson(dfp, dn_hu)[1]:.1f}]")

    g, gt = collections.Counter(), collections.Counter()
    for i, k in key.items():
        if k["label"] != "ai":
            continue
        m = k["model"].split("/")[-1]
        gt[m] += 1
        g[m] += P[i]["verdict"] == "ai"
    print("    by generator:  " + "  ".join(
        f"{m} {g[m]}/{gt[m]}={g[m]/gt[m]:.0%}" for m in sorted(gt)))

    conf = collections.Counter()
    ok = collections.Counter()
    for i, p in P.items():
        c = str(p.get("confidence", "?")).lower()
        conf[c] += 1
        ok[c] += p["verdict"] == key[i]["label"]
    print("    by confidence: " + "  ".join(
        f"{c} {ok[c]}/{conf[c]}={ok[c]/conf[c]:.0%}" for c in sorted(conf, key=lambda x: -conf[x])))

    # Repeated documents are an unintentional test-retest experiment: identical
    # text, different batch, different judge instance.
    groups = collections.defaultdict(list)
    for i, k in key.items():
        groups[(k["id"], k["model"])].append(i)
    pairs = agree = 0
    for items in groups.values():
        vs = [P[i]["verdict"] for i in items]
        for a in range(len(vs)):
            for b in range(a + 1, len(vs)):
                pairs += 1
                agree += vs[a] == vs[b]
    if pairs:
        print(f"    self-consistency on identical text: {agree}/{pairs} = {agree/pairs:.0%}")
    print()

if failures:
    print("FAILURES", file=sys.stderr)
    for f in failures:
        print(f"  {f}", file=sys.stderr)
    sys.exit(f"{len(failures)} problem(s) -- refusing to report a rate on "
             f"an incomplete sample")
