#!/usr/bin/env python3
"""Build the public, redistribution-safe AITA corpus for Hugging Face.

Mirrors scripts/16_redact_for_hf.py (the ELI5 redaction), applied to the stamped
aita_docs.jsonl. The human class is verbatim r/AmItheAsshole text, obtained via
AI2's Scruples Anecdotes, which is Reddit-authored and NOT ours to redistribute.
So strip every Reddit-authored string and keep only identifiers, so a user can
reconstruct the human side from Scruples themselves (the hydration pattern):

  - drop `question` (the AITA post title) from every row
  - drop `text` and `text_norm` from HUMAN rows (the Reddit post body)
  - keep the AI `text`/`text_norm` (ours), all identifiers (doc_id, q_id -- the
    base36 Reddit post id, the hydration key), provenance (generator, source,
    split, seed, draws, post-ops), and non-text metadata (words, date, votes,
    verdict, label)

Asserts the redaction is complete before writing, because this feeds a public push
and a leaked string cannot be un-published.

    python3 phase3/scripts/aita_14_redact_for_hf.py
"""
import json, os

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
SRC = f"{ROOT}/aita/data/aita_docs.jsonl"
OUT = f"{ROOT}/aita/hf_public"
os.makedirs(OUT, exist_ok=True)

rows = [json.loads(l) for l in open(SRC)]
bysplit = {"train": [], "dev": [], "test": []}
n_h = n_a = 0
for r in rows:
    r = dict(r)
    r.pop("question", None)                       # Reddit post title
    if r.get("label") == "human":
        r.pop("text", None); r.pop("text_norm", None)   # Reddit post body
        n_h += 1
    else:
        assert r.get("text"), f"ai row lost its text: {r.get('doc_id')}"
        n_a += 1
    # hard guarantees before this row is written to a public file
    assert "question" not in r, "title survived redaction"
    if r.get("label") == "human":
        assert "text" not in r and "text_norm" not in r, "human text survived redaction"
    assert r.get("q_id"), "q_id missing -- hydration key gone"
    bysplit[r["split"]].append(r)

allrows = bysplit["train"] + bysplit["dev"] + bysplit["test"]
with open(f"{OUT}/dataset.jsonl", "w") as f:
    for r in allrows: f.write(json.dumps(r) + "\n")
for s in ("train", "dev", "test"):
    with open(f"{OUT}/{s}.jsonl", "w") as f:
        for r in bysplit[s]: f.write(json.dumps(r) + "\n")

print(f"redacted {len(allrows)} rows ({n_h} human stripped, {n_a} ai kept with text) -> {OUT}/")
print(f"  splits: train {len(bysplit['train'])}  dev {len(bysplit['dev'])}  test {len(bysplit['test'])}")

# independent re-scan of what was written: no Reddit-authored text anywhere it must not be
bad = 0
for l in open(f"{OUT}/dataset.jsonl"):
    r = json.loads(l)
    if "question" in r or (r["label"] == "human" and ("text" in r or "text_norm" in r)):
        bad += 1
print(f"  post-write re-scan, leaked human strings: {bad}  (must be 0)")
print(f"  human rows still carry q_id for hydration: "
      f"{all(json.loads(l).get('q_id') for l in open(f'{OUT}/dataset.jsonl'))}")
