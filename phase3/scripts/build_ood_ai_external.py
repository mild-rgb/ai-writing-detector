#!/usr/bin/env python3
"""Build OOD *AI* eval sets from external detection benchmarks.

EVALUATION MATERIAL ONLY. Never used as training labels.

These are pure machine-generated documents from RAID, MAGE, and WildChat. Unlike
the SE/CC-News human sets (which measure false positives), these measure the
detector's out-of-domain FALSE-NEGATIVE rate. Caveat baked into the provenance:
their generators are 2022-2024 vintage (GPT-4 and earlier), NOT the 2026 frontier
models in the training corpus -- so this is a "can it catch older, unseen models"
probe, and mixes domain shift with generator shift. Each record keeps the
generating `model` so results can be sliced by it.

  python build_ood_ai_external.py --source raid|mage|wildchat --target 2500
"""
import argparse, hashlib, json, os, random, re
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

STUDY = os.path.join(os.path.dirname(__file__), "..", "study")
MIN_WORDS, MAX_WORDS = 250, 800
SEED = 20260824
WS_RE = re.compile(r"\s+")


def clean(t):
    return WS_RE.sub(" ", t or "").strip()


def keep(text, seen):
    """Shared length + dedup gate. Returns (text, n_words) or None."""
    text = clean(text)
    n = len(text.split())
    if not (MIN_WORDS <= n <= MAX_WORDS):
        return None
    h = hashlib.sha1(text[:200].encode()).hexdigest()
    if h in seen:
        return None
    seen.add(h)
    return text, n


def build_raid(target):
    """AI generations only (model != human), no adversarial attack. Lazily pull
    interleaved train shards and stop once the target is met with good coverage."""
    rng = random.Random(SEED)
    order = [0, 5, 2, 7, 4, 9, 1, 6, 3, 8]
    per_domain_cap = max(target // 5, 200)
    seen, by_domain, kept = set(), {}, []
    for sh in order:
        path = hf_hub_download("liamdugan/raid",
                               f"raid/partial-train/{sh:04d}.parquet",
                               repo_type="dataset", revision="refs/convert/parquet")
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=2000,
                columns=["model", "attack", "domain", "decoding", "generation"]):
            d = batch.to_pydict()
            for model, attack, domain, decoding, gen in zip(
                    d["model"], d["attack"], d["domain"], d["decoding"], d["generation"]):
                if model == "human" or attack != "none":
                    continue
                if by_domain.get(domain, 0) >= per_domain_cap:
                    continue
                r = keep(gen, seen)
                if not r:
                    continue
                text, n = r
                by_domain[domain] = by_domain.get(domain, 0) + 1
                kept.append({"source": "raid", "model": model, "domain": domain,
                             "decoding": decoding, "words": n, "label": "ai", "text": text})
        print(f"  raid shard {sh}: total kept {len(kept)}  domains {dict(sorted(by_domain.items()))}", flush=True)
        if len(kept) >= target:
            break
    rng.shuffle(kept)
    return kept[:target]


def build_mage(target):
    """Machine-generated rows only (src contains '_machine')."""
    rng = random.Random(SEED)
    path = hf_hub_download("yaful/MAGE", "default/train/0000.parquet",
                           repo_type="dataset", revision="refs/convert/parquet")
    seen, kept = set(), []
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=2000, columns=["text", "label", "src"]):
        d = batch.to_pydict()
        for text, label, src in zip(d["text"], d["label"], d["src"]):
            src = src or ""
            if "machine" not in src:                       # human rows carry '_human'
                continue
            r = keep(text, seen)
            if not r:
                continue
            t, n = r
            # src looks like '<domain>_machine_<setting>_<model>'
            parts = src.split("_machine")
            domain = parts[0]
            model = parts[1].strip("_") if len(parts) > 1 else "unknown"
            kept.append({"source": "mage", "model": model, "domain": domain,
                         "words": n, "label": "ai", "text": t})
    rng.shuffle(kept)
    print(f"  mage: {len(kept)} machine docs in band", flush=True)
    return kept[:target]


def build_wildchat(target):
    """Assistant turns only (pure AI), English, non-toxic."""
    rng = random.Random(SEED)
    path = hf_hub_download("allenai/WildChat", "data/train-00000-of-00006.parquet",
                           repo_type="dataset")
    seen, kept = set(), []
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=500,
            columns=["model", "language", "toxic", "conversation"]):
        d = batch.to_pydict()
        for model, lang, toxic, conv in zip(d["model"], d["language"], d["toxic"], d["conversation"]):
            if lang != "English" or toxic:
                continue
            for turn in conv:
                if turn.get("role") != "assistant" or turn.get("toxic"):
                    continue
                r = keep(turn.get("content", ""), seen)
                if not r:
                    continue
                t, n = r
                kept.append({"source": "wildchat", "model": model, "domain": "chat",
                             "words": n, "label": "ai", "text": t})
        if len(kept) >= target * 3:                        # enough to sample from one shard
            break
    rng.shuffle(kept)
    print(f"  wildchat: {len(kept)} assistant turns in band (from 1 shard)", flush=True)
    return kept[:target]


BUILDERS = {"raid": build_raid, "mage": build_mage, "wildchat": build_wildchat}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, choices=list(BUILDERS))
    ap.add_argument("--target", type=int, default=2500)
    a = ap.parse_args()

    recs = BUILDERS[a.source](a.target)
    out_dir = os.path.join(STUDY, f"ood_{a.source}")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "chunks.jsonl")
    with open(out, "w") as f:
        for i, rec in enumerate(recs):
            rec = {"doc_id": f"{a.source}-{i:05d}", **rec}
            f.write(json.dumps(rec) + "\n")
    from collections import Counter
    print(f"wrote {len(recs)} AI docs -> {out}")
    print(f"  models: {dict(Counter(r['model'] for r in recs).most_common(12))}")


if __name__ == "__main__":
    main()
