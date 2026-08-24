# OOD eval — CC-News

**Evaluation material only. Never used as training labels.**

5,005 guaranteed-human news articles from Common Crawl News (dates 2017–2018, so
wholly pre-LLM-era) via the `vblagoje/cc_news` HF dataset. An out-of-distribution
probe for the ELI5 human-vs-AI detector: journalism register, human writing only,
to measure false-positive behavior off distribution against the bag-of-words floor.

- **Source:** `vblagoje/cc_news` (Common Crawl News). Articles across Reuters,
  Guardian, DailyMail, Indian Express, and many other outlets.
- **Filters:** 250–800 words, deduped, reservoir-sampled uniformly from 256,492
  in-band articles (seed 20260824).
- **Format:** `chunks.jsonl` — `{doc_id, source, domain, date, url, words,
  label:"human", text}`. Companion `chunks.jsonl.bow.json` holds the BoW floor's
  per-document p_ai.
- **BoW floor:** flags 2.5% of these human docs as AI at threshold 0.5.
- **Detector result (24 Aug 2026, 240-doc sample):** **1.67% false positives**,
  below the bag-of-words floor's 2.92% on the same documents. News register does
  not push this detector toward "AI". Scored output in
  `../_ood_scored_20260824/` (local only, not in the repo).
- **Builder:** `scripts/build_ood_ccnews.py` (reads cached parquet, reproducible).
