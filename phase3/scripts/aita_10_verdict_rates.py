"""Measure how the verdict question is PHRASED, not just whether it exists.

markers.py counts a question mark. That closed to within 2 points of the human
rate once the floor stopped forbidding the verdict question -- and hid a second
problem underneath, because a rate can match while the WORDING is a fingerprint.

A diversity check found the generated class reaching for "tell me if I" (12% ai,
0% human) and avoiding the canonical "am I the asshole" (23% human, 10% ai),
because the draw forbade stock phrasing. Forbidding the genre's own name makes
the text less human, not more.

The draw now permits the canonical phrasing without mandating it. Both strings
are measured here, because the failure this replaces could recur in the mirror:
a model that likes a canonical phrase can overshoot as easily as a prohibition
can suppress. Stopping rule agreed in advance -- if canonical clears 35% against
a human 23%, it becomes a rate-matched draw at 23%, verified on the 1,050-doc
early corpus rather than by paying for another bench.

    python3 phase3/scripts/aita_10_verdict_rates.py AI.jsonl [AI2.jsonl ...]
"""
import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Two readings of "canonical", reported separately because they give very
# different human baselines and the stopping rule must be applied against the
# SAME definition its baseline came from. The narrow one is the phrase spelled
# out; the broad one also counts the AITA acronym, which is how a lot of real
# posters actually write it.
PATTERNS = {
    "canonical spelled out": re.compile(
        r"\bam i (?:the|an?) (?:asshole|ass hole|a\*+hole|ah)\b", re.I),
    "canonical incl. AITA acronym": re.compile(
        r"\bam i (?:the|an?) (?:asshole|ass hole|a\*+hole|ah)\b|\baita\b", re.I),
    "'tell me if I'": re.compile(r"\btell me if i\b", re.I),
    "'was I wrong'": re.compile(r"\b(?:was|am) i (?:in the )?wrong\b", re.I),
    "'who is the a**hole'": re.compile(r"\bwho(?:'s| is) the (?:asshole|a\*+hole)\b", re.I),
    "any question mark": re.compile(r"\?"),
}
# The agreed rule was "fires if canonical clears ~35% against a human 23%" --
# about 1.5x the human rate. It is expressed as a RATIO here rather than as the
# absolute 35%, because the absolute number is only meaningful against the
# baseline it was set from, and the two regexes above give human baselines of
# 23% and 65%. A ratio keeps the intent -- fire on OVERSHOOT past the human
# class -- under either definition.
OVERSHOOT_RATIO = 1.5
KEY = "canonical spelled out"


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * (c - h), 100 * (c + h))


ap = argparse.ArgumentParser()
ap.add_argument("ai", nargs="+")
ap.add_argument("--humans", default=f"{ROOT}/phase3/aita/data/aita_human.jsonl")
ap.add_argument("--partition", default=f"{ROOT}/phase3/aita/data/aita_partition.json")
ap.add_argument("--human-ids", nargs="*", default=["dev", "bench"])
a = ap.parse_args()

part = json.load(open(a.partition))
want = set()
for k in a.human_ids:
    want |= set(part[k])
hum = [json.loads(l) for l in open(a.humans)]
htexts = [r["human_answer"] for r in hum if r["id"] in want]

ai = []
for p in a.ai:
    ai += [json.loads(l) for l in open(p) if l.strip()]
atexts = [r["text"] for r in ai]
by_model = defaultdict(list)
for r in ai:
    by_model[r["model"]].append(r["text"])

print(f"human n={len(htexts)} ({'+'.join(a.human_ids)})   ai n={len(atexts)}\n")
print(f"{'phrasing':<32}{'human':>18}{'ai':>18}{'gap':>8}")
res = {}
for name, rx in PATTERNS.items():
    hk = sum(1 for t in htexts if rx.search(t))
    ak = sum(1 for t in atexts if rx.search(t))
    hr, ar = 100 * hk / len(htexts), 100 * ak / len(atexts)
    hl, hh = wilson(hk, len(htexts))
    al, ah = wilson(ak, len(atexts))
    res[name] = (hr, ar)
    print(f"{name:<32}{hr:>6.1f}% [{hl:4.1f},{hh:5.1f}]"
          f"{ar:>6.1f}% [{al:4.1f},{ah:5.1f}]{ar-hr:>+8.1f}")

print(f"\nper generator, {KEY} (human {res[KEY][0]:.1f}%):")
rx = PATTERNS[KEY]
for m in sorted(by_model):
    v = by_model[m]
    k = sum(1 for t in v if rx.search(t))
    print(f"  {m.split('/')[-1]:<28}{100*k/len(v):>6.1f}%  ({k}/{len(v)})")

hr, ar = res[KEY]
trigger = hr * OVERSHOOT_RATIO
print()
print(f"stopping rule on '{KEY}': fires if ai exceeds {OVERSHOOT_RATIO}x the "
      f"human rate, i.e. above {trigger:.1f}%")
if ar > trigger:
    print(f"*** FALLBACK FIRES: ai {ar:.1f}% against human {hr:.1f}%.")
    print("    Convert to a rate-matched draw at the human rate and verify on the")
    print("    1,050-document early corpus -- NOT on another bench.")
else:
    print(f"fallback does NOT fire: ai {ar:.1f}% against human {hr:.1f}% "
          f"(trigger {trigger:.1f}%). No draw change.")
    if ar < hr / OVERSHOOT_RATIO:
        print(f"    NOTE: ai is well UNDER the human rate -- the direction the "
              f"fix was meant to correct. Worth reporting either way.")
