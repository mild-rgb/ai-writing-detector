"""Did the no-name branch produce stilted relationship-term repetition?

The naming draw's negative branch tells a model to refer to people by their
relationship instead of a name. That removes the over-naming tell and could
plant a new one: "my wife did X. My wife said Y. I told my wife..." where a
person writes the term once and then switches to "she".

Real posters suppress names in about 89% of posts and still read naturally,
because they pronoun away after the first mention. This checks whether the
models do the same, comparing only like with like -- AI documents where the
draw suppressed the name, against human documents that name nobody.

Two numbers per class:
  * relationship-term mentions per 1,000 words
  * the pronoun ratio: third-person pronouns divided by relationship terms.
    A person's ratio is high, because the term is introduced once and carried by
    pronouns. A model repeating "my wife" every sentence drives it down.

This is the p2_mechanical lesson applied to a generation instruction rather than
a transformation: the fix passes the statistic it targets by construction, so
the question worth asking is what it broke somewhere else.

    python3 phase3/scripts/aita_12_relationship_check.py AI.jsonl
"""
import argparse
import json
import os
import re
import statistics as st
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, f"{ROOT}/phase3/scripts")
from aita_08_diversity import names_in  # noqa: E402

REL = re.compile(
    r"\bmy (wife|husband|girlfriend|boyfriend|partner|fiance|fiancee|mom|mum|mother|dad|father|"
    r"sister|brother|son|daughter|friend|best friend|roommate|room-?mate|boss|coworker|co-worker|"
    r"neighbou?r|aunt|uncle|cousin|grandma|grandmother|grandpa|grandfather|in-?law|ex|manager|"
    r"landlord|stepmom|stepdad|stepmother|stepfather|stepsister|stepbrother|niece|nephew)\b", re.I)
PRON = re.compile(r"\b(he|him|his|she|her|hers|they|them|their)\b", re.I)


def stats(text):
    w = max(1, len(text.split()))
    rel = len(REL.findall(text))
    pro = len(PRON.findall(text))
    return rel * 1000 / w, (pro / rel if rel else None), rel


ap = argparse.ArgumentParser()
ap.add_argument("ai")
ap.add_argument("--humans", default=f"{ROOT}/phase3/aita/data/aita_human.jsonl")
ap.add_argument("--pool", default=f"{ROOT}/phase3/aita/data/name_pool.json")
a = ap.parse_args()

pool = {p["name"] for p in json.load(open(a.pool))["pool"]}
ai = [json.loads(l) for l in open(a.ai) if l.strip()]
humans = {json.loads(l)["id"]: json.loads(l) for l in open(a.humans)}


def named(t):
    return bool([n for n in names_in(t) if n in pool])


# like with like: AI documents that name nobody, human documents that name nobody
ai_free = [r for r in ai if not named(r["text"])]
hu_free = [humans[r["id"]] for r in ai if not named(humans[r["id"]]["human_answer"])]

print(f"AI documents naming nobody   : {len(ai_free)} of {len(ai)}")
print(f"human documents naming nobody: {len(hu_free)} of {len(ai)}\n")


def report(label, texts):
    d = [stats(t) for t in texts]
    per = [x[0] for x in d]
    ratios = [x[1] for x in d if x[1] is not None]
    heavy = sum(1 for x in d if x[2] >= 6)
    print(f"{label:<30}{st.mean(per):>9.2f}{st.median(ratios) if ratios else 0:>12.2f}"
          f"{100*heavy/max(1,len(texts)):>11.1f}%")
    return st.mean(per), (st.median(ratios) if ratios else 0)


print(f"{'class':<30}{'rel/1k words':>9}{'pronoun ratio':>12}{'>=6 terms':>11}")
h = report("HUMAN (names nobody)", [r["human_answer"] for r in hu_free])
by = defaultdict(list)
for r in ai_free:
    by[r["model"]].append(r["text"])
worst = None
for m in sorted(by):
    v = report(m.split("/")[-1] + " (no name)", by[m])
    if worst is None or v[0] > worst[1]:
        worst = (m.split("/")[-1], v[0])
allai = report("ALL AI (names nobody)", [r["text"] for r in ai_free])

print(f"\nhuman baseline: {h[0]:.2f} relationship terms per 1,000 words, "
      f"pronoun ratio {h[1]:.2f}")
if allai[0] > h[0] * 1.5:
    print(f"*** AI repeats relationship terms {allai[0]/h[0]:.1f}x the human rate. "
          f"The no-name branch has planted a new tell.")
elif allai[1] and h[1] and allai[1] < h[1] / 1.5:
    print(f"*** AI pronoun ratio {allai[1]:.2f} against human {h[1]:.2f}: the models "
          f"restate the relationship where a person would switch to a pronoun.")
else:
    print("no stilted-repetition tell: AI sits within 1.5x of the human class on "
          "both measures.")
print(f"worst generator by term density: {worst[0]} at {worst[1]:.2f} per 1,000 words")
