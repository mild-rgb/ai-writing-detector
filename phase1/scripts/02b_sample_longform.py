"""Sample a long-form companion set: answers at or above Pangram's 250-word
'highest accuracy' threshold.

The main set (50-250 words) sits entirely inside the band where Pangram reports
elevated false positive and false negative rates. Any detector benchmarked
against it is being asked to work where the reference tool is weakest. This set
is drawn from the same corpus with a 250-word floor so results can be compared
across the length boundary.

Disjoint from the main set by construction: a question's top-scored answer is
either under 250 words or over, never both.
"""
import json
import random
import re
import sys

N = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
MIN_WORDS, MAX_WORDS = 250, 800     # upper cap keeps generation cost sane and
                                    # avoids multi-part essay outliers
MIN_SCORE = 10
SEED = 20260821

IN = "data/raw/eli5_questions.jsonl"
OUT = "data/interim/questions_longform.jsonl"
PREFIX = re.compile(r"^\s*[\[\(]?\s*eli\s*[-:_ ]?\s*5\s*[\]\)]?\s*[:,-]?\s*", re.I)


def clean_title(t):
    t = PREFIX.sub("", t).strip()
    return t[0].upper() + t[1:] if t else t


def norm(t):
    return re.sub(r"[^a-z0-9 ]", "", t.lower()).strip()


short_ids = {json.loads(l)["q_id"]
             for l in open("data/interim/questions_1000.jsonl")}

cands, seen = [], set()
for line in open(IN):
    rec = json.loads(line)
    ans = rec["answers"]
    if not ans["text"]:
        continue
    score, text = max(zip(ans["score"], ans["text"]))
    text = text.strip()
    words = len(text.split())
    if score < MIN_SCORE or not (MIN_WORDS <= words <= MAX_WORDS):
        continue
    if "_URL_" in text or text in ("[deleted]", "[removed]"):
        continue
    q = clean_title(rec["title"])
    if len(q) < 15 or "_URL_" in q:
        continue
    key = norm(q)
    if key in seen or rec["q_id"] in short_ids:
        continue
    seen.add(key)
    cands.append({"q_id": rec["q_id"], "question": q,
                  "selftext": rec["selftext"].strip(),
                  "human_answer": text, "human_words": words,
                  "score": score, "url": rec["url"]})

print(f"candidates ({MIN_WORDS}-{MAX_WORDS} words, score>={MIN_SCORE}): {len(cands)}")
random.Random(SEED).shuffle(cands)
sample = cands[:N]
for i, r in enumerate(sample):
    r["id"] = f"eli5lf-{i:04d}"
with open(OUT, "w") as f:
    for r in sample:
        f.write(json.dumps(r) + "\n")

w = sorted(r["human_words"] for r in sample)
n = len(w)
print(f"wrote {n} -> {OUT}")
print(f"words: min {w[0]}  p25 {w[n//4]}  median {w[n//2]}  p75 {w[3*n//4]}  max {w[-1]}")
print(f"all >= 250 (Pangram 'highest accuracy'): {all(x >= 250 for x in w)}")
