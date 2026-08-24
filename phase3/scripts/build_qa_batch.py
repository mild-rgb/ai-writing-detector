"""Sample documents for a qualitative QA read by a fresh subagent.

The report card counts things. It passed p2_mechanical on every structural
statistic while the documents themselves were torn -- a markdown header split
into its own paragraph with its title merged into the prose behind it. Nothing
in 35 markers and four distributions can see that. A reader can.

This is NOT a detection pass. The subagent is asked what is *wrong* with a
document, not whether a model wrote it, and its output never becomes a rate. The
end-of-loop detection judging stays separate: Haiku x5 plus Sonnet, once, on the
frozen bench split, for the reasons in PROTOCOL.md.

Human documents are mixed in unlabelled as a **credibility control**. If a QA
pass flags genuine 2011-2019 redditors for the same defect it flags the
generated text for, that flag is a property of the reader rather than of the
output, and it does not motivate a prompt change. The control is read that way
and only that way -- it is never scored as a false-positive rate.

    python3 phase3/scripts/build_qa_batch.py PROMPT_DIR --round v1 --batches 2
"""
import argparse
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CORPUS = f"{ROOT}/phase1/data/interim/questions_longform.jsonl"

ap = argparse.ArgumentParser()
ap.add_argument("prompt_dir")
ap.add_argument("--round", default="v1")
ap.add_argument("--answers")
ap.add_argument("--batches", type=int, default=2)
ap.add_argument("--ai-per-batch", type=int, default=12)
ap.add_argument("--human-per-batch", type=int, default=3)
a = ap.parse_args()

pdir = a.prompt_dir.rstrip("/")
answers = a.answers or (f"{pdir}/answers_dev.jsonl" if a.round == "v1"
                        else f"{pdir}/answers_dev_{a.round}.jsonl")
rows = {}
for i, line in enumerate(open(answers), 1):
    line = line.strip()
    if not line:
        continue
    try:
        r = json.loads(line)
    except json.JSONDecodeError as e:
        sys.exit(f"FATAL {answers}:{i}: {e}")
    rows[(r["id"], r["model"])] = r
humans = {json.loads(l)["id"]: json.loads(l) for l in open(CORPUS)}

# Stratified by generator so one model cannot dominate the sample, and sorted
# so the draw is reproducible.
by_model = {}
for (qid, m), r in sorted(rows.items()):
    by_model.setdefault(m, []).append(r)
rnd = random.Random(hash((os.path.basename(pdir), a.round)) % (2 ** 31))
for m in by_model:
    rnd.shuffle(by_model[m])

need = a.batches * a.ai_per_batch
picked, i = [], 0
models = sorted(by_model)
while len(picked) < need:
    m = models[i % len(models)]
    if by_model[m]:
        picked.append(by_model[m].pop())
    elif all(not by_model[x] for x in models):
        break
    i += 1

hum_ids = sorted({qid for qid, _ in rows})
rnd.shuffle(hum_ids)
hum_pick = hum_ids[:a.batches * a.human_per_batch]

items = [{"kind": "generated", "model": r["model"], "id": r["id"],
          "question": r["question"], "text": r["text"]} for r in picked]
items += [{"kind": "human", "model": "human", "id": q,
           "question": humans[q]["question"], "text": humans[q]["human_answer"]}
          for q in hum_pick]
rnd.shuffle(items)

outdir = f"{pdir}/qa/{a.round}"
bdir = f"{outdir}/batches"
os.makedirs(bdir, exist_ok=True)
with open(f"{outdir}/key.jsonl", "w") as f:
    for i, it in enumerate(items):
        f.write(json.dumps({"item": f"D{i:03d}", "kind": it["kind"],
                            "model": it["model"], "id": it["id"]}) + "\n")

per = -(-len(items) // a.batches)
n = 0
for b in range(0, len(items), per):
    chunk = items[b:b + per]
    body = ["# Document review", "",
            "Below are several comments written in reply to questions on the",
            "subreddit r/explainlikeimfive. Read each one as a reader would.", ""]
    for j, it in enumerate(chunk):
        body += ["", "=" * 76, f"DOCUMENT D{b + j:03d}", "=" * 76,
                 f"QUESTION: {it['question']}", "", "COMMENT:", it["text"], ""]
    open(f"{bdir}/batch_{n:02d}.txt", "w").write("\n".join(body) + "\n")
    n += 1

print(f"{len(items)} documents ({len(picked)} generated / {len(hum_pick)} human "
      f"controls) over {n} batches -> {outdir}")
print("blend: " + ", ".join(
    f"{m.split('/')[-1]} {sum(1 for it in items if it['model'] == m)}"
    for m in sorted({it['model'] for it in items})))
