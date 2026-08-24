"""Build the frozen confirmation benchmark for one prompt.

The blend is fixed once and reused by all six prompts: the same 70 questions,
the same question->model assignment, the same 70 human documents. Prompt
comparisons are therefore paired within question AND within generator, which is
the design phase 1 arrived at only after measuring question-set effects at
2-2.5x binomial -- larger than most of the effects it was trying to compare.

Rules carried over from phase1/scripts/12_make_mixed_single.py, each of which
fixes a defect that had already produced published numbers:

  * ONE model per question, so each question contributes exactly one ai item and
    one human item and no document is ever emitted twice.
  * No batch may contain two answers to the same question -- that would be a
    2AFC pair inside a benchmark whose entire premise is that no reference text
    exists. It happened in 6 of 30 batches of the original long-form pass.
  * key.jsonl lives OUTSIDE the directory handed to judges.
  * Every invariant is asserted before anything is written.

Replicate passes reshuffle which items share a batch (seeded by --pass) but
never change the item set, so the five passes are five independent judge draws
over identical documents.

    python3 phase3/scripts/build_bench.py PROMPT_DIR --pass 0
"""
import argparse
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ASSIGN_DEFAULT = f"{ROOT}/phase3/data/bench_assignment.json"
CORPUS = f"{ROOT}/phase1/data/interim/questions_longform.jsonl"
PER_BATCH = 5
PER_MODEL = 10

MODELS = [
    "qwen/qwen3.8-max", "deepseek/deepseek-v4-pro",
    "nvidia/nemotron-3.5-lightning", "google/gemini-3.7-flash",
    "openai/gpt-5.6-luna-pro", "z-ai/glm-5.3",
]


def build_assignment(models, split):
    """Question -> generator, drawn once per roster and then permanent.

    Every bench question is used, round-robin over the roster, so the blend is
    as balanced as the question count allows and the item set stays at 2 x the
    number of bench questions however many models are in the roster. With seven
    models that is 10 each; with six it is 12/12/12/12/11/11. Unequal by one is
    fine -- what matters is that it is identical across all six prompts.
    """
    part = json.load(open(f"{ROOT}/phase3/data/longform_partition.json"))
    bench = sorted(part[split])
    rnd = random.Random(31337)
    picked = list(bench)
    rnd.shuffle(picked)
    assign = {qid: models[i % len(models)] for i, qid in enumerate(picked)}
    return {"models": models, "assignment": assign}


ap = argparse.ArgumentParser()
ap.add_argument("prompt_dir")
ap.add_argument("--pass", dest="pass_n", type=int, default=0)
ap.add_argument("--split", default="bench",
                help="question partition to build from. `bench` is the frozen "
                     "confirmation set; `dev` builds a descriptive pass over "
                     "the tuning split, which is NOT a held-out result and "
                     "must not be used to choose between prompt versions.")
ap.add_argument("--answers")
ap.add_argument("--models", nargs="*", default=MODELS)
ap.add_argument("--allow-substitute", action="store_true",
                help="reassign a question whose assigned generator is missing. "
                     "Descriptive passes only -- never the frozen confirmation.")
ap.add_argument("--assign", default=None,
                help="path to the frozen assignment; defaults to one named for "
                     "the roster size, so a 6-model blend cannot silently "
                     "overwrite the 7-model one")
a = ap.parse_args()

ASSIGN = a.assign or ASSIGN_DEFAULT.replace(
    "bench_assignment.json", f"{a.split}_assignment_{len(a.models)}.json")
if not os.path.exists(ASSIGN):
    json.dump(build_assignment(a.models, a.split), open(ASSIGN, "w"), indent=1)
    print(f"wrote the permanent blend assignment -> {ASSIGN}")
A = json.load(open(ASSIGN))
if A["models"] != a.models:
    sys.exit(f"FATAL: {ASSIGN} was frozen for {A['models']}, not {a.models}. "
             f"The blend is fixed across prompts; refusing to change it.")
assign = A["assignment"]

pdir = a.prompt_dir.rstrip("/")
answers = a.answers or f"{pdir}/answers_{a.split}.jsonl"
pool = {}
for i, line in enumerate(open(answers), 1):
    line = line.strip()
    if not line:
        continue
    try:
        r = json.loads(line)
    except json.JSONDecodeError as e:
        sys.exit(f"FATAL {answers}:{i}: {e}")
    pool[(r["id"], r["model"])] = r

humans = {json.loads(l)["id"]: json.loads(l) for l in open(CORPUS)}

missing = [(q, m) for q, m in assign.items() if (q, m) not in pool]
if missing and not a.allow_substitute:
    sys.exit(f"FATAL: {len(missing)} of {len(assign)} assigned (question, model) "
             f"pairs are missing from {answers}. First: {missing[:3]}. The blend "
             f"is fixed across prompts; a short arm is not a smaller benchmark, "
             f"it is a different one. Pass --allow-substitute only for a "
             f"descriptive pass, never for the frozen confirmation.")
if missing:
    # Descriptive passes only. Rate-limit holes leave a handful of assigned
    # pairs unfilled; rather than shrink the item set (which changes what is
    # being measured) the question is reassigned to whichever available model
    # is currently furthest below its quota. Every substitution is printed,
    # because a silently rebalanced blend is a different blend.
    counts = {m: sum(1 for q, mm in assign.items()
                     if mm == m and (q, mm) in pool) for m in a.models}
    subs = []
    for q, m in missing:
        cands = [x for x in a.models if (q, x) in pool]
        if not cands:
            del assign[q]
            subs.append((q, m, None))
            continue
        pick = min(cands, key=lambda x: counts[x])
        assign[q] = pick
        counts[pick] += 1
        subs.append((q, m, pick))
    print(f"substituted {len(subs)} assignments (descriptive pass):")
    for q, was, now in subs[:8]:
        print(f"    {q}  {was.split('/')[-1]} -> "
              f"{now.split('/')[-1] if now else 'DROPPED (no model has it)'}")
    if len(subs) > 8:
        print(f"    ... and {len(subs) - 8} more")

items = []
for qid, model in sorted(assign.items()):
    r = pool[(qid, model)]
    items.append({"label": "ai", "model": model, "id": qid,
                  "question": r["question"], "text": r["text"]})
    h = humans[qid]
    items.append({"label": "human", "model": "human", "id": qid,
                  "question": h["question"], "text": h["human_answer"]})

rnd = random.Random(9000 + a.pass_n)
for _ in range(4000):
    rnd.shuffle(items)
    ok = True
    for b in range(0, len(items), PER_BATCH):
        ids = [it["id"] for it in items[b:b + PER_BATCH]]
        if len(set(ids)) != len(ids):
            ok = False
            break
    if ok:
        break
else:
    sys.exit("FATAL: could not find a batch layout with no within-batch "
             "question collision after 4000 tries")

outdir = f"{pdir}/{a.split}/pass{a.pass_n}"
bdir = f"{outdir}/batches"
os.makedirs(bdir, exist_ok=True)
with open(f"{outdir}/key.jsonl", "w") as f:
    for i, it in enumerate(items):
        f.write(json.dumps({"item": f"I{i:03d}", "label": it["label"],
                            "model": it["model"], "id": it["id"]}) + "\n")

n_batches = 0
for b in range(0, len(items), PER_BATCH):
    chunk = items[b:b + PER_BATCH]
    body = [
        "# Comment classification", "",
        f"Below are {len(chunk)} separate comments, each posted in reply to a",
        "different question on the subreddit r/explainlikeimfive. Each was either",
        "written by a human redditor between 2011 and 2019, or by a language model",
        "in 2026 imitating that style.", "",
        "They are unrelated to each other. Judge each on its own; there is no",
        "reference text and no fixed proportion of human to AI.", "",
    ]
    for j, it in enumerate(chunk):
        body += ["", "=" * 76, f"ITEM I{b + j:03d}", "=" * 76,
                 f"QUESTION: {it['question']}", "", "COMMENT:", it["text"], ""]
    open(f"{bdir}/batch_{n_batches:02d}.txt", "w").write("\n".join(body) + "\n")
    n_batches += 1

ai_docs = {(it["id"], it["model"]) for it in items if it["label"] == "ai"}
hu_docs = {it["id"] for it in items if it["label"] == "human"}
n_ai = sum(1 for it in items if it["label"] == "ai")
assert len(ai_docs) == n_ai, "an ai document is duplicated"
assert len(hu_docs) == len(items) - n_ai, "a human document is duplicated"
assert not os.path.exists(f"{bdir}/key.jsonl"), "key.jsonl must stay out of batches/"

print(f"{len(items)} items ({n_ai} ai / {len(items) - n_ai} human), "
      f"{len(ai_docs)} distinct ai documents, {len(hu_docs)} distinct human, "
      f"{n_batches} batches of {PER_BATCH}")
print("blend: " + ", ".join(f"{m.split('/')[-1]} "
                            f"{sum(1 for it in items if it['model'] == m)}"
                            for m in A["models"]))
print(f"-> {outdir}")
