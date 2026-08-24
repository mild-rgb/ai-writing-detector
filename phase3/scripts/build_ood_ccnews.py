#!/usr/bin/env python3
"""Build a CC-News OOD human-eval set.

EVALUATION MATERIAL ONLY. Guaranteed-human news articles (2017-2019, so wholly
pre-LLM-era) from Common Crawl News via the `vblagoje/cc_news` HF dataset.
Never used as training labels.

We download the dataset's parquet shards directly (bypassing `datasets`, which
has a fsspec version clash in this env), scan them with pyarrow, and reservoir-
sample the target count from articles in the model's 250-800 word band.

  python build_ood_ccnews.py --target 1800
"""
import argparse, hashlib, json, os, random, re
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "study", "ood_ccnews")
REPO = "vblagoje/cc_news"
SHARDS = [f"plain_text/train-{i:05d}-of-00005.parquet" for i in range(5)]
MIN_WORDS, MAX_WORDS = 250, 800
SEED = 20260824
WS_RE = re.compile(r"\s+")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=1800)
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "chunks.jsonl"))
    a = ap.parse_args()

    rng = random.Random(SEED)
    reservoir = []            # list of kept rows
    seen = set()
    n_qual = scanned = 0

    for shard in SHARDS:
        path = hf_hub_download(REPO, shard, repo_type="dataset")
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=2000, columns=["text", "domain", "date", "url"]):
            d = batch.to_pydict()
            for text, domain, date, url in zip(d["text"], d["domain"], d["date"], d["url"]):
                scanned += 1
                if not text:
                    continue
                text = WS_RE.sub(" ", text).strip()
                n = len(text.split())
                if not (MIN_WORDS <= n <= MAX_WORDS):
                    continue
                h = hashlib.sha1(text[:200].encode()).hexdigest()
                if h in seen:
                    continue
                seen.add(h)
                n_qual += 1
                rec = {
                    "source": "cc_news",
                    "domain": domain,
                    "date": (str(date) or "")[:10],
                    "url": url,
                    "words": n,
                    "label": "human",
                    "text": text,
                }
                # reservoir sampling: uniform target-size sample over all qualifying rows
                if len(reservoir) < a.target:
                    reservoir.append(rec)
                else:
                    j = rng.randint(0, n_qual - 1)
                    if j < a.target:
                        reservoir[j] = rec
        print(f"  {shard.split('/')[-1]}: scanned {scanned}, qualifying {n_qual}", flush=True)

    rng.shuffle(reservoir)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        for i, rec in enumerate(reservoir):
            rec = {"doc_id": f"ccnews-{i:05d}", **rec}
            f.write(json.dumps(rec) + "\n")
    print(f"\nscanned {scanned} rows, {n_qual} in band -> sampled {len(reservoir)} -> {a.out}")


if __name__ == "__main__":
    main()
