"""Select 1000 clean question / human-answer pairs from the ELI5 dump.

Filters aim at answers that are unambiguously real human prose: well-received
(score >= 10), substantial but not essay-length, and free of the dataset's own
`_URL_n_` scrubbing artifacts, which would be a giveaway feature no AI answer
could ever have.
"""
import json
import random
import re

IN = "data/raw/eli5_questions.jsonl"
OUT = "data/interim/questions_1000.jsonl"

N = 1000
SEED = 20260821
MIN_SCORE = 10
MIN_WORDS, MAX_WORDS = 50, 250

PREFIX = re.compile(r"^\s*[\[\(]?\s*eli\s*[-:_ ]?\s*5\s*[\]\)]?\s*[:,-]?\s*", re.I)


def clean_title(t):
    t = PREFIX.sub("", t).strip()
    return t[0].upper() + t[1:] if t else t


def norm(t):
    return re.sub(r"[^a-z0-9 ]", "", t.lower()).strip()


cands, seen_titles = [], set()
with open(IN) as f:
    for line in f:
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
        if key in seen_titles:
            continue
        seen_titles.add(key)
        cands.append({
            "q_id": rec["q_id"],
            "question": q,
            "selftext": rec["selftext"].strip(),
            "human_answer": text,
            "human_words": words,
            "score": score,
            "url": rec["url"],
        })

print(f"candidates after filtering: {len(cands)}")
random.Random(SEED).shuffle(cands)
sample = cands[:N]
for i, r in enumerate(sample):
    r["id"] = f"eli5-{i:04d}"

with open(OUT, "w") as f:
    for r in sample:
        f.write(json.dumps(r) + "\n")

wc = sorted(r["human_words"] for r in sample)
print(f"wrote {len(sample)} -> {OUT}")
print("human answer words: p25=%d median=%d p75=%d mean=%.0f"
      % (wc[len(wc) // 4], wc[len(wc) // 2], wc[3 * len(wc) // 4], sum(wc) / len(wc)))
