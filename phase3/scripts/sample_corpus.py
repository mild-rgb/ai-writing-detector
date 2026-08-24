"""Sample the phase-3 corpus questions, assign generators, and split.

The strict filter over the 107,280-row r/eli5 dump yields exactly 2,902
long-form candidates (top-scored answer per question, score >= 10, 250-800
words, no `_URL_` placeholder, deduped on a normalised title). The corpus spec
asks for "about 2,900", which is therefore the whole available pool rather than
a draw from it -- worth stating, because nothing here is a random sample of
long-form ELI5 and no sampling variance argument applies to the question set.

Two exclusions, for one reason each:

  * the adversarial `bench` and `heldout` questions are held out of the corpus
    so they stay clean for detector evaluation. `dev`, `burned` and `pool` are
    NOT excluded -- they were used for prompt development, not for any held-out
    measurement, and their human answers are the same public comments either
    way.
  * nothing else. Every candidate the filter admits is kept.

Each question is assigned ONE generator, round-robin over the seven models in a
seeded shuffle, so the ai class is model-balanced by construction and the
per-document provenance survives into phase 4's attribution task without
regenerating anything.

Splits are grouped by question, which here is automatic: a question contributes
exactly one human document and one ai document, so grouping by question and
grouping by document pair are the same partition. That matters -- a human answer
and the ai answer written against it share topic, length target and paragraph
target, and splitting them across train and test would leak.

    python3 phase3/scripts/sample_corpus.py
"""
import json
import os
import random
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DUMP = f"{ROOT}/phase1/data/raw/eli5_questions.jsonl"
LONGFORM = f"{ROOT}/phase1/data/interim/questions_longform.jsonl"
PARTITION = f"{ROOT}/phase3/data/longform_partition.json"
OUT = f"{ROOT}/phase3/corpus/questions.jsonl"
SEED = 20260824
MIN_WORDS, MAX_WORDS, MIN_SCORE = 250, 800, 10
SPLITS = (("train", 0.80), ("dev", 0.10), ("test", 0.10))

MODELS = [
    "x-ai/grok-4.6",
    "qwen/qwen3.8-max",
    "deepseek/deepseek-v4-pro",
    "nvidia/nemotron-3.5-lightning",
    "google/gemini-3.7-flash",
    "openai/gpt-5.6-luna-pro",
    "z-ai/glm-5.3",
]

PREFIX = re.compile(r"^\s*[\[\(]?\s*eli\s*[-:_ ]?\s*5\s*[\]\)]?\s*[:,-]?\s*", re.I)


def clean_title(t):
    t = PREFIX.sub("", t).strip()
    return t[0].upper() + t[1:] if t else t


def norm(t):
    return re.sub(r"[^a-z0-9 ]", "", t.lower()).strip()


part = json.load(open(PARTITION))
lf = {json.loads(l)["id"]: json.loads(l) for l in open(LONGFORM)}
excluded_qids = {lf[i]["q_id"] for i in part["bench"] + part["heldout"] if i in lf}
print(f"excluding {len(excluded_qids)} questions held out for detector evaluation "
      f"({len(part['bench'])} bench + {len(part['heldout'])} heldout)")

cands, seen = [], set()
for line in open(DUMP):
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
    if key in seen:
        continue
    seen.add(key)
    if rec["q_id"] in excluded_qids:
        continue
    cands.append({"q_id": rec["q_id"], "question": q,
                  "selftext": rec["selftext"].strip(),
                  "human_answer": text, "human_words": words,
                  "score": score, "url": rec["url"]})

print(f"candidates after filter and exclusion: {len(cands)}")
if not cands:
    sys.exit("FATAL: no candidates")

rnd = random.Random(SEED)
rnd.shuffle(cands)

n = len(cands)
n_train = int(round(SPLITS[0][1] * n))
n_dev = int(round(SPLITS[1][1] * n))
bounds = {"train": (0, n_train), "dev": (n_train, n_train + n_dev),
          "test": (n_train + n_dev, n)}

# Round-robin over a shuffled model order, assigned AFTER the split boundaries
# are fixed and walked in index order, so every split is model-balanced rather
# than only the corpus as a whole.
order = list(MODELS)
rnd.shuffle(order)
with open(OUT, "w") as f:
    for i, r in enumerate(cands):
        split = next(s for s, (a, b) in bounds.items() if a <= i < b)
        r["id"] = f"eli5c-{i:04d}"
        r["split"] = split
        r["model"] = order[i % len(order)]
        r["prompt"] = "floor"
        r["seed"] = f"{r['id']}|{r['model']}"
        f.write(json.dumps(r) + "\n")

rows = [json.loads(l) for l in open(OUT)]
import collections
print(f"\nwrote {len(rows)} -> {OUT}")
for s in ("train", "dev", "test"):
    sub = [r for r in rows if r["split"] == s]
    c = collections.Counter(r["model"] for r in sub)
    w = sorted(r["human_words"] for r in sub)
    print(f"  {s:<6} {len(sub):>5} questions   words median {w[len(w)//2]}   "
          f"models {min(c.values())}-{max(c.values())} each")
overlap = sum(1 for r in rows if r["q_id"] in {v["q_id"] for v in lf.values()})
print(f"\n{overlap} of {len(rows)} questions also appear in the phase-1 long-form "
      f"set (dev/burned/pool only -- bench and heldout are excluded by "
      f"construction)")
print(f"ai documents to generate: {len(rows)} (one per question)")
print(f"human documents: {len(rows)} (already on disk, no generation)")
