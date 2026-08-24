"""Build the public, redistribution-safe version of the corpus for Hugging Face.

The human class is verbatim r/explainlikeimfive text (questions and top-scored
answers) from the Facebook ELI5 corpus, which we do not have the right to
redistribute. So this strips every Reddit-authored string and keeps only
identifiers, so a user can reconstruct the human side from the source corpus
themselves (the "hydration" pattern):

  - drop `question` (the Reddit post title) from every row
  - drop `text` from human rows (the Reddit comment)
  - keep the AI `text` (ours), all identifiers (q_id, url, doc_id, question_id),
    all provenance (model, prompt, seed, ...), and non-text metadata (score,
    words, split, label)

Asserts the redaction is complete before writing, because this feeds a public
push and a leaked string cannot be un-published.
"""
import json
import os

SRC = "corpus"
OUT = "corpus/hf_public"
FILES = ["dataset.jsonl", "train.jsonl", "dev.jsonl", "test.jsonl"]

os.makedirs(OUT, exist_ok=True)

for fname in FILES:
    n = n_human = n_ai = 0
    with open(f"{SRC}/{fname}") as f, open(f"{OUT}/{fname}", "w") as out:
        for line in f:
            r = json.loads(line)
            r.pop("question", None)                 # Reddit post title
            if r.get("label") == "human":
                r.pop("text", None)                 # Reddit comment
                n_human += 1
            else:
                assert r.get("text"), f"ai row lost its text: {r.get('doc_id')}"
                n_ai += 1
            # hard guarantees before this row is written to a public file
            assert "question" not in r, "question text survived redaction"
            if r.get("label") == "human":
                assert "text" not in r, "human text survived redaction"
            assert r.get("q_id"), "identifier q_id missing -- reconstruction broken"
            out.write(json.dumps(r) + "\n")
            n += 1
    print(f"{fname:14} {n:5} rows  ({n_human} human redacted, {n_ai} ai kept with text)")

print(f"\nwrote redacted files to {OUT}/")
