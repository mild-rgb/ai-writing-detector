"""Measure each (prompt, model) length ratio and write the calibration table.

The first calibration constants were measured on a pilot of p1_pinned and
applied to all six prompts. That was wrong in a way v1 made obvious: grok lands
at 0.94x its request under p1 and at roughly 0.6x under p4 and p2, because the
length a model produces depends on what else the prompt asked it to do. Word
count then comes back as a separable feature in exactly the prompts whose
mechanism pulls hardest against length (grok's `words` separability: 0.508 under
p1, 0.859 under p2, 0.860 under p4).

So the constant is per (prompt, model), measured from that prompt's own previous
run. This is phase 1's "same rules, per-model constants" taken one step further
because the measurement said it had to be.

    python3 phase3/scripts/calibrate_length.py                     # write it
    python3 phase3/scripts/calibrate_length.py --pattern=answers_dev_v2.jsonl
    python3 phase3/scripts/calibrate_length.py --show    # print it
"""
import collections
import re
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = f"{ROOT}/phase3/data/length_calib.json"
MIN_N = 15

table = {}
PATTERN = next((x.split("=", 1)[1] for x in sys.argv if x.startswith("--pattern=")),
               "answers_dev.jsonl")
for path in sorted(glob.glob(f"{ROOT}/phase3/adversarial_prompts/*/{PATTERN}")):
    prompt = path.split("/")[-2]
    by = collections.defaultdict(list)
    byp = collections.defaultdict(list)
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        # ratio of what came back to what was ASKED, so the new constant
        # composes with the one already in force when the text was generated.
        asked = r.get("asked_words") or r["target_words"]
        by[r["model"]].append((r["words"], asked))
        got = len([x for x in re.split(r"\n\s*\n", r["text"].strip()) if x.strip()])
        byp[r["model"]].append((got, max(1, r.get("asked_paras")
                                         or r.get("target_paras") or 1)))
    # Pooled ratio (sum produced / sum asked), not the median of per-document
    # ratios. Paragraph counts are small integers, so a per-document ratio
    # quantises hard: nemotron's median ratio reads exactly 1.000 while its
    # median paragraph count is 5 against a human 6, because "one short" on a
    # 6-paragraph target rounds to a ratio near 1 in half the documents and the
    # median never sees the tail. The pooled ratio does.
    entry = {}
    for m, v in by.items():
        if len(v) < MIN_N:
            continue
        entry[m] = {"len": round(sum(x[0] for x in v) / sum(x[1] for x in v), 3),
                    "para": round(sum(x[0] for x in byp[m]) /
                                  sum(x[1] for x in byp[m]), 3)}
    if entry:
        table[prompt] = entry

if "--show" in sys.argv:
    old = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for p in sorted(table):
        print(p)
        for m in sorted(table[p]):
            e = table[p][m]
            print(f"    {m.split('/')[-1]:<26} length {e['len']:.3f}   "
                  f"paragraphs {e['para']:.3f}")
    sys.exit()

# The constant is a DIVISOR on the target: asked = target / C, and what comes
# back is produced = r * asked where r is the ratio measured here. Setting
# C = r makes produced = target. Composing with the previous constant would be
# wrong -- r is already measured against the inflated ask, so it carries the
# old constant inside it. (Written down because the first version of this file
# multiplied them and would have asked grok for 2.2x the target under a setting
# that was already landing on it.)
prev = json.load(open(OUT)) if os.path.exists(OUT) else {}
merged = {p: dict(e) for p, e in table.items()}
for p in prev:
    merged.setdefault(p, prev[p])
json.dump(merged, open(OUT, "w"), indent=1)
for p in sorted(merged):
    row = "  ".join(f"{m.split('/')[-1][:8]} {merged[p][m]['len']:.2f}/"
                    f"{merged[p][m]['para']:.2f}" for m in sorted(merged[p]))
    print(f"{p:<16} {row}")
print(f"-> {OUT}")
