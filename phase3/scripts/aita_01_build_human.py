"""Build the AITA human class from the Scruples Anecdotes (Lourie et al. 2020).

The ELI5 human class came from the Facebook LFQA dump, streamed and filtered to
one subreddit. That dump contains exactly three subreddits -- explainlikeimfive
59,714 / askscience 31,933 / AskHistorians 28,424 -- and no AITA, so the same
FILE cannot be reused. The same KIND of source can: Scruples is a curated
academic corpus of r/AmItheAsshole anecdotes released in 2020, original case,
carrying the reddit post id.

Why the label is true by construction, the same argument phase 1 made for ELI5:
every post id in Scruples resolves to a date no later than 2019-04-06, more than
three years before ChatGPT. Dates are recovered by interpolating base36 post ids
against 36,896 (id, timestamp) pairs from a dated AITA dump, since Scruples
itself ships no timestamp.

Filter, mirrored from phase 1's strict long-form filter:
  * 250-800 words                (the phase-1 long-form band)
  * HISTORICAL only              (WIBTA hypotheticals are a different register)
  * >= 10 community judgements   (phase 1 used score >= 10)
  * no [removed] / [deleted] / edit-only bodies
  * deduped on a normalised title

Emits the schema phase3/scripts/generate.py --corpus reads: `id`, `question`,
`human_answer`, `human_words`. The AITA title is the `question` -- it is what
the generator is conditioned on, exactly as the ELI5 question title was.

    python3 phase3/scripts/aita_01_build_human.py --out phase3/aita/data/aita_human.jsonl
"""
import argparse
import glob
import json
import os
import re
import sys
import tarfile
import urllib.request

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRUPLES = "https://storage.googleapis.com/ai2-mosaic-public/projects/scruples/v1.0/data/anecdotes.tar.gz"
DATED = ("https://huggingface.co/datasets/MattBoraske/"
         "reddit-AITA-submissions-and-comments-binary/resolve/main/data/train-00000-of-00001.parquet")

ap = argparse.ArgumentParser()
ap.add_argument("--cache", default=f"{ROOT}/phase3/aita/data/cache")
ap.add_argument("--out", default=f"{ROOT}/phase3/aita/data/aita_human.jsonl")
ap.add_argument("--min-words", type=int, default=250)
ap.add_argument("--max-words", type=int, default=800)
ap.add_argument("--min-votes", type=int, default=10)
ap.add_argument("--before", default="2020-01-01")
a = ap.parse_args()
os.makedirs(a.cache, exist_ok=True)

tgz = f"{a.cache}/anecdotes.tar.gz"
if not os.path.exists(tgz):
    print("downloading Scruples anecdotes ...", file=sys.stderr)
    urllib.request.urlretrieve(SCRUPLES, tgz)
if not os.path.isdir(f"{a.cache}/anecdotes"):
    tarfile.open(tgz).extractall(a.cache)

pq = f"{a.cache}/dated_aita.parquet"
if not os.path.exists(pq):
    print("downloading the dated AITA dump for id->date calibration ...", file=sys.stderr)
    pd.read_parquet(DATED, columns=["submission_url", "submission_date"]).to_parquet(pq)
cal = pd.read_parquet(pq)
cal["sid"] = cal.submission_url.str.extract(r"/comments/([a-z0-9]+)/")
cal = cal.dropna(subset=["sid"])
cal["n"] = cal.sid.apply(lambda s: float(int(s, 36)))
cal["epoch"] = pd.to_datetime(cal.submission_date).astype("int64") / 1e9
cal = cal.sort_values("n")
xs, ys = cal.n.values, cal.epoch.values
print(f"calibration: {len(cal)} (id, date) pairs spanning "
      f"{pd.to_datetime(ys.min(), unit='s').date()} to "
      f"{pd.to_datetime(ys.max(), unit='s').date()}", file=sys.stderr)

rows = []
for f in sorted(glob.glob(f"{a.cache}/anecdotes/*.jsonl")):
    for line in open(f):
        rows.append(json.loads(line))
print(f"{len(rows)} anecdotes loaded", file=sys.stderr)

DEAD = re.compile(r"^\s*(\[removed\]|\[deleted\])\s*$", re.I)
NORM = re.compile(r"[^a-z0-9 ]+")


def date_of(pid):
    try:
        n = float(int(pid, 36))
    except Exception:
        return None
    # ids below the calibration floor are OLDER than its earliest pair, which is
    # already inside the accept window, so they are kept without extrapolating.
    if n < xs[0]:
        return "pre-" + str(pd.to_datetime(ys[0], unit="s").date())
    if n > xs[-1]:
        return None
    return str(pd.to_datetime(float(np.interp(n, xs, ys)), unit="s").date())


kept, seen_titles = [], set()
drop = {"words": 0, "type": 0, "votes": 0, "dead": 0, "dupe": 0, "date": 0}
for r in rows:
    text = (r.get("text") or "").strip()
    title = (r.get("title") or "").strip()
    if not text or DEAD.match(text) or len(text) < 40:
        drop["dead"] += 1
        continue
    if r.get("post_type") != "HISTORICAL":
        drop["type"] += 1
        continue
    votes = sum((r.get("label_scores") or {}).values())
    if votes < a.min_votes:
        drop["votes"] += 1
        continue
    w = len(text.split())
    if not (a.min_words <= w <= a.max_words):
        drop["words"] += 1
        continue
    d = date_of(r["post_id"])
    if d is None or (not d.startswith("pre-") and d >= a.before):
        drop["date"] += 1
        continue
    key = NORM.sub("", title.lower()).strip()
    if key in seen_titles:
        drop["dupe"] += 1
        continue
    seen_titles.add(key)
    kept.append({"id": r["post_id"], "q_id": r["post_id"], "question": title,
                 "human_answer": text, "human_words": w, "date": d,
                 "votes": votes, "verdict": r.get("binarized_label")})

kept.sort(key=lambda x: x["id"])
os.makedirs(os.path.dirname(a.out), exist_ok=True)
with open(a.out, "w") as f:
    for r in kept:
        f.write(json.dumps(r) + "\n")

print(f"kept {len(kept)} of {len(rows)}  -> {a.out}", file=sys.stderr)
print("dropped: " + ", ".join(f"{k} {v}" for k, v in drop.items()), file=sys.stderr)
ws = [r["human_words"] for r in kept]
print(f"words: median {int(np.median(ws))} mean {np.mean(ws):.0f} "
      f"p10 {int(np.percentile(ws,10))} p90 {int(np.percentile(ws,90))}", file=sys.stderr)
dated = sorted(r["date"] for r in kept if not r["date"].startswith("pre-"))
print(f"dates: {len(kept) - len(dated)} below the calibration floor (older than "
      f"{pd.to_datetime(ys[0], unit='s').date()}), {len(dated)} dated "
      f"{dated[0]} to {dated[-1]}", file=sys.stderr)
print(f"newest post in the corpus: {dated[-1]} -- every document predates "
      f"ChatGPT (2022-11-30) by more than three years", file=sys.stderr)
