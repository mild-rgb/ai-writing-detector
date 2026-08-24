# OOD eval — Stack Exchange (cross-site "best of")

**Evaluation material only. Never used as training labels.**

4,995 guaranteed-human documents (pre-2022, so pre-LLM-era) — high-score answers
(Score ≥ 15) drawn evenly across 17 prose-oriented Stack Exchange sites, capped
at 320/site to keep register diversity so no single topic dominates. This is an
out-of-distribution probe for the ELI5 human-vs-AI detector: a different domain,
different register, human writing only, to measure the detector's false-positive
behavior off distribution against the bag-of-words floor.

- **Source & license:** the official Stack Exchange data dumps on archive.org.
  All user contributions are CC-BY-SA; attribution is the post author + site.
- **Filters:** `PostTypeId=2` (answers), `Score ≥ 15`, `CreationDate < 2022-01-01`,
  250–800 words after HTML strip, code-heavy posts dropped (>15% code chars).
- **Sites:** academia, boardgames, cooking, english, gardening, history, law,
  literature, movies, outdoors, parenting, philosophy, politics, scifi, skeptics,
  travel, worldbuilding.
- **Format:** `chunks.jsonl` — `{doc_id, source, site, post_id, score, created,
  words, label:"human", text}`. Companion `chunks.jsonl.bow.json` holds the BoW
  floor's per-document p_ai.
- **BoW floor:** flags 0.5% of these human docs as AI at threshold 0.5.
- **Builder:** `scripts/build_ood_stackexchange.py` (seed 20260824, reproducible).
