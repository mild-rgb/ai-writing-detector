"""Assemble the labeled detector dataset and report generation quality.

Splits are grouped by q_id: a question's human answer and all three of its AI
answers land in the same split. Splitting at the row level would leak topic
across train/test and inflate accuracy.
"""
import json
import random
import re
from collections import Counter, defaultdict

import pandas as pd

QUESTIONS = "data/interim/questions_1000.jsonl"
AI = "data/interim/ai_answers.jsonl"
OUT_CSV = "data/final/dataset.csv"
OUT_JSONL = "data/final/dataset.jsonl"
REPORT = "data/final/quality_report.md"

SEED = 20260821
SPLITS = [("train", 0.70), ("val", 0.15), ("test", 0.15)]

ARTIFACT_PATS = [("Edit:", r"(?mi)^\s*edit\s*[:\-]"), ("TL;DR", r"(?i)\btl;?dr\b"),
                 ("slang", r"(?i)\b(upvot|downvot|OP\b|karma)\w*")]


def human_artifact(text):
    """Which Reddit artifact a human answer actually contains, if any."""
    for name, pat in ARTIFACT_PATS:
        if re.search(pat, text):
            return name
    return None


MD = re.compile(r"(^\s*[-*+]\s+|^\s*\d+\.\s+|^#{1,6}\s|\*\*|__)", re.M)
REFUSAL = re.compile(r"\b(I can'?t|I cannot|as an AI|language model|I'?m unable)\b", re.I)

questions = {json.loads(l)["id"]: json.loads(l) for l in open(QUESTIONS)}
ai_rows = [json.loads(l) for l in open(AI)]

# Assign every question to a split, then inherit it for all of its answers.
ids = sorted(questions)
random.Random(SEED).shuffle(ids)
split_of, cut = {}, 0
for name, frac in SPLITS:
    n = round(len(ids) * frac)
    for i in ids[cut:cut + n]:
        split_of[i] = name
    cut += n
for i in ids[cut:]:
    split_of[i] = SPLITS[-1][0]

rows = []
for qid, q in questions.items():
    rows.append({
        "row_id": f"{qid}-human", "id": qid, "q_id": q["q_id"],
        "question": q["question"], "text": q["human_answer"],
        "label": 0, "source": "human", "words": q["human_words"],
        "target_words": q["human_words"], "score": q["score"],
        "artifact": human_artifact(q["human_answer"]),
        "artifact_requested": None, "split": split_of[qid],
    })
for r in ai_rows:
    slug = r["model"].split("/")[-1]
    rows.append({
        "row_id": f"{r['id']}-{slug}", "id": r["id"], "q_id": r["q_id"],
        "question": r["question"], "text": r["text"],
        "label": 1, "source": r["model"], "words": r["words"],
        "target_words": r["target_words"], "score": None,
        "artifact": human_artifact(r["text"]),
        "artifact_requested": r.get("artifact"), "split": split_of[r["id"]],
    })

df = pd.DataFrame(rows).sort_values(["id", "source"]).reset_index(drop=True)
df.to_csv(OUT_CSV, index=False)
with open(OUT_JSONL, "w") as f:
    for r in df.to_dict("records"):
        f.write(json.dumps(r) + "\n")

# ---- quality report -------------------------------------------------------
lines = ["# Dataset quality report", ""]
lines.append(f"- rows: **{len(df)}**  (human {int((df.label == 0).sum())}, "
             f"ai {int((df.label == 1).sum())})")
lines.append(f"- questions: **{df.id.nunique()}**")
lines.append("")
lines.append("## Rows per source and split\n")
piv = df.pivot_table(index="source", columns="split", values="row_id",
                     aggfunc="count", fill_value=0)
piv = piv[[s for s, _ in SPLITS if s in piv.columns]]
lines.append(piv.to_markdown())

lines.append("\n## Style-match check\n")
lines.append("| source | median words | mean |len-target| | markdown leak | refusal/meta |")
lines.append("|---|---|---|---|---|")
for src, g in df.groupby("source"):
    dev = (g.words - g.target_words).abs().mean()
    md = MD.search
    md_rate = sum(bool(md(t)) for t in g.text) / len(g)
    ref = sum(bool(REFUSAL.search(t)) for t in g.text) / len(g)
    lines.append(f"| {src} | {g.words.median():.0f} | {dev:.1f} | "
                 f"{md_rate:.1%} | {ref:.1%} |")

missing = Counter()
by_q = defaultdict(set)
for r in ai_rows:
    by_q[r["id"]].add(r["model"])
models = sorted({r["model"] for r in ai_rows})
for qid in questions:
    for m in models:
        if m not in by_q.get(qid, ()):
            missing[m] += 1
lines.append("\n## Reddit-artifact parity\n")
lines.append("Share of rows whose text actually *contains* each artifact. AI rows are")
lines.append("instructed at the human base rate; deviation here is model non-compliance.\n")
art = df.pivot_table(index="source", columns=df.artifact.fillna("none"),
                     values="row_id", aggfunc="count", fill_value=0)
lines.append((art.div(art.sum(axis=1), axis=0) * 100).round(1).to_markdown())

lines.append("\n## Coverage gaps\n")
lines.append("\n".join(f"- {m}: {n} questions missing" for m, n in missing.items())
             or "- none: every question has an answer from every model")

open(REPORT, "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
print(f"\nwrote {OUT_CSV}, {OUT_JSONL}, {REPORT}")
