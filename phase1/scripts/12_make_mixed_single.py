"""Build single-document batches, mixed across models and conditions.

One agent per item is the cleanest isolation but costs one agent per item. This
puts a handful of items in each batch and SHUFFLES ACROSS SOURCES, so a judge
never sees several answers from the same generator together and cannot calibrate
a threshold off a coherent within-file pattern. Items are independent
classifications, never pairs -- the judge has no reference text either way.

Four defects in the first version of this script are fixed here; the benchmark
built with it (`study/single/long`, `study/single/short`) has all three.

1. The human sibling was appended INSIDE the per-model loop, once per AI row
   drawn. When two or three models were drawn on the same question, its human
   answer was emitted two or three times as separate, byte-identical items. The
   long-form set reported 75 human items that were only 39 distinct documents,
   and every false-positive interval computed from it was too narrow. Each
   question now contributes exactly ONE ai item and ONE human item.

2. Source rows were used as-is. `study/longform/answers.jsonl` holds 65
   duplicate (id, model) rows from a restart that appended instead of resuming,
   so the same generation could be drawn twice. Sources are deduplicated on
   (id, model), last row wins.

3. `key.jsonl` -- the answer key -- was written into the same directory as the
   batch files handed to judges. Nothing suggests a judge ever read it (Haiku
   scored 53%), but a benchmark should not depend on that. Batches now go in a
   `batches/` subdirectory and the key stays outside it.

4. Batches were sequential chunks of a flat shuffle, so an ai answer and the
   human answer to the SAME question could land in the same 5-item file -- a
   2AFC pair handed to a judge whose whole design premise is that no reference
   text exists. It happened in 6 of 30 long-form batches. Batching now rejects
   any layout that puts two answers to one question in one batch.

Selection is one model per question, so the ai class spans as many distinct
questions as it has documents, and the human class is topic-matched to it
without ever duplicating a document.
"""
import json
import os
import random
import sys

OUTDIR = sys.argv[1]
PER = int(sys.argv[2])
AI_PER_MODEL = int(sys.argv[3])   # ai DOCUMENTS per model; humans = total ai
SOURCES = sys.argv[4:]            # answers.jsonl files
SEED = 4242
MAX_LAYOUT_TRIES = 2000

hum = {}
for f in ("data/interim/questions_1000.jsonl", "data/interim/questions_longform.jsonl"):
    if os.path.exists(f):
        for l in open(f):
            r = json.loads(l)
            hum[r["id"]] = r

rnd = random.Random(SEED)

# --- gather, deduplicating on (id, model) --------------------------------
pool = {}
for src in SOURCES:
    tag = src.split("/")[-2]
    for l in open(src):
        r = json.loads(l)
        pool[(tag, r["id"], r["model"])] = r
by_model = {}
for (tag, qid, model), r in pool.items():
    by_model.setdefault((tag, model), []).append(r)

# --- one model per question ----------------------------------------------
# Models are served in a rotating order so that a question shortage is shared
# evenly rather than starving whichever model is drawn last.
claimed = set()
chosen = []
keys = sorted(by_model)
for k in keys:
    rnd.shuffle(by_model[k])
quota = {k: AI_PER_MODEL for k in keys}
progress = True
while progress and any(quota[k] for k in keys):
    progress = False
    for k in keys:
        if not quota[k]:
            continue
        while by_model[k]:
            r = by_model[k].pop()
            if r["id"] in claimed or r["id"] not in hum:
                continue
            claimed.add(r["id"])
            chosen.append((k[0], r))
            quota[k] -= 1
            progress = True
            break

short = {k: quota[k] for k in keys if quota[k]}
if short:
    print(f"WARNING: could not fill quota for {short} -- not enough distinct "
          f"questions in the source pool", file=sys.stderr)

items = []
for tag, r in chosen:
    h = hum[r["id"]]
    items.append({"cond": tag, "label": "ai", "model": r["model"], "id": r["id"],
                  "question": r["question"], "text": r["text"]})
    items.append({"cond": tag, "label": "human", "model": "human", "id": r["id"],
                  "question": h["question"], "text": h["human_answer"]})

# --- invariants -----------------------------------------------------------
n_ai = sum(1 for i in items if i["label"] == "ai")
n_hu = len(items) - n_ai
assert n_ai == n_hu, (n_ai, n_hu)
assert len({(i["id"], i["model"]) for i in items if i["label"] == "ai"}) == n_ai, \
    "duplicate ai document"
assert len({i["id"] for i in items if i["label"] == "human"}) == n_hu, \
    "duplicate human document"

# --- layout: no two answers to one question in the same batch -------------
for attempt in range(MAX_LAYOUT_TRIES):
    rnd.shuffle(items)
    if all(len({it["id"] for it in items[b:b + PER]}) == len(items[b:b + PER])
           for b in range(0, len(items), PER)):
        break
else:
    sys.exit(f"no collision-free layout after {MAX_LAYOUT_TRIES} shuffles; "
             f"raise the question count or lower PER")

os.makedirs(f"{OUTDIR}/batches", exist_ok=True)
with open(f"{OUTDIR}/key.jsonl", "w") as f:
    for i, it in enumerate(items):
        f.write(json.dumps({"item": f"I{i:03d}",
                            **{k: it[k] for k in ("cond", "label", "model", "id")}}) + "\n")
nb = 0
for b in range(0, len(items), PER):
    chunk = items[b:b + PER]
    if not chunk:
        break
    lines = ["# Comment classification", "",
             f"Below are {len(chunk)} separate comments, each posted in reply to a",
             "different question on the subreddit r/explainlikeimfive. Each was either",
             "written by a human redditor between 2011 and 2019, or by a language model",
             "in 2026 imitating that style.", "",
             "They are unrelated to each other. Judge each on its own; there is no",
             "reference text and no fixed proportion of human to AI.", ""]
    for j, it in enumerate(chunk):
        lines += ["", "=" * 76, f"ITEM I{b + j:03d}", "=" * 76,
                  f"QUESTION: {it['question']}", "", "COMMENT:", it["text"], ""]
    open(f"{OUTDIR}/batches/batch_{nb:02d}.txt", "w").write("\n".join(lines) + "\n")
    nb += 1

print(f"{len(items)} items ({n_ai} ai / {n_hu} human) -> {nb} batches of {PER}")
print(f"distinct documents: {n_ai} ai / {n_hu} human; "
      f"distinct questions: {len({i['id'] for i in items})}")
print(f"collision-free layout found on shuffle {attempt + 1}")
print(f"batches in {OUTDIR}/batches/ ; answer key OUTSIDE it at {OUTDIR}/key.jsonl")
