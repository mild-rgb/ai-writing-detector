"""Aggregate the subagent QA reads for one round.

Reports defect rates by tag for the generated documents and, separately, the
rate at which the same reader flagged the unlabelled HUMAN controls. The second
number is the only thing that makes the first interpretable: a defect tag that
fires as often on genuine 2011-2019 redditors as on generated text is telling
you about the reader, not the output, and it does not motivate a prompt change.

It is a credibility control and nothing else. It is not scored as a
false-positive rate and no detection statistic is computed here.

    python3 phase3/scripts/qa_summary.py --round v1
"""
import argparse
import collections
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ap = argparse.ArgumentParser()
ap.add_argument("--round", default="v1")
ap.add_argument("--prompts", nargs="*")
a = ap.parse_args()

dirs = sorted(glob.glob(f"{ROOT}/phase3/adversarial_prompts/p*"))
if a.prompts:
    dirs = [d for d in dirs if os.path.basename(d) in a.prompts]

overall_notes = []
for d in dirs:
    name = os.path.basename(d)
    qadir = f"{d}/qa/{a.round}"
    files = sorted(glob.glob(f"{qadir}/agent/qa_*.json"))
    if not files:
        continue
    key = {json.loads(l)["item"]: json.loads(l)
           for l in open(f"{qadir}/key.jsonl")}
    tags_ai = collections.Counter()
    tags_hu = collections.Counter()
    n_ai = n_hu = flagged_ai = flagged_hu = 0
    sev = collections.Counter()
    for f in files:
        try:
            obj = json.load(open(f))
        except json.JSONDecodeError as e:
            sys.exit(f"FATAL {f}: {e}")
        for doc in obj.get("documents", []):
            k = key.get(doc["item"])
            if k is None:
                sys.exit(f"FATAL {f}: unknown item {doc['item']}")
            defects = [t for t in doc.get("defects", []) if t]
            if k["kind"] == "human":
                n_hu += 1
                flagged_hu += bool(defects)
                tags_hu.update(defects)
            else:
                n_ai += 1
                flagged_ai += bool(defects)
                tags_ai.update(defects)
                if defects:
                    sev[doc.get("severity", "?")] += 1
        if obj.get("overall"):
            overall_notes.append((name, os.path.basename(f), obj["overall"]))
    if not n_ai:
        continue
    print(f"\n=== {name} ({a.round}) ===")
    print(f"generated: {flagged_ai}/{n_ai} documents flagged   "
          f"human controls: {flagged_hu}/{n_hu} flagged")
    print(f"{'tag':<20} {'generated':>10} {'human ctrl':>11}   reading")
    for t, c in tags_ai.most_common():
        ai_rate = 100 * c / n_ai
        hu_rate = 100 * tags_hu[t] / n_hu if n_hu else 0.0
        note = ""
        if n_hu and tags_hu[t] and hu_rate >= ai_rate * 0.6:
            note = "reader effect -- fires on real redditors too"
        elif ai_rate >= 30:
            note = "systematic"
        print(f"{t:<20} {ai_rate:>9.0f}% {hu_rate:>10.0f}%   {note}")
    only_hu = [t for t in tags_hu if t not in tags_ai]
    if only_hu:
        print(f"tags seen only on human controls: {', '.join(only_hu)}")
    print(f"severity of flagged generated docs: {dict(sev)}")

print("\n--- what each reader said was systematic ---")
for name, f, note in overall_notes:
    print(f"\n[{name} {f}] {note.strip()[:400]}")
