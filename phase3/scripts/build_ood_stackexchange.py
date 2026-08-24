#!/usr/bin/env python3
"""Build a cross-site "best of" Stack Exchange OOD human-eval set.

EVALUATION MATERIAL ONLY. These are guaranteed-human documents (pre-2022, so
pre-LLM-era) drawn from many Stack Exchange sites, for out-of-distribution
detector evaluation. Never used as training labels.

Source: the official Stack Exchange data dumps on archive.org, which are
CC-BY-SA licensed (all user contributions are irrevocably so). We pull one 7z
per site, extract only Posts.xml, and keep high-score prose *answers* written
before the cutoff, in the model's 250-800 word band, with code-heavy posts
dropped (a prose detector should not be judged on code listings).

  python build_ood_stackexchange.py --sites english,history,... --pilot
"""
import argparse, html, json, os, random, re, subprocess, sys, urllib.request
from lxml import etree

CACHE = os.path.expanduser("~/.cache/se_dumps")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "study", "ood_stackexchange")
CUTOFF = "2022-01-01"          # CreationDate strictly before this (ISO => lexical compare ok)
MIN_SCORE = 15                 # "best of": community-vetted answers
MIN_WORDS, MAX_WORDS = 250, 800
PER_SITE_CAP = 320             # keep cross-site diversity; no single site dominates
SEED = 20260824

# Prose-oriented sites only. Deliberately excludes code/math-heavy networks
# (Stack Overflow, superuser, math, tex) whose answer bodies are not prose.
SITE_DOMAIN = {
    "english":      "english.stackexchange.com",     # English Language & Usage
    "history":      "history.stackexchange.com",
    "philosophy":   "philosophy.stackexchange.com",
    "literature":   "literature.stackexchange.com",
    "cooking":      "cooking.stackexchange.com",
    "travel":       "travel.stackexchange.com",
    "skeptics":     "skeptics.stackexchange.com",
    "politics":     "politics.stackexchange.com",
    "law":          "law.stackexchange.com",
    "academia":     "academia.stackexchange.com",
    "worldbuilding":"worldbuilding.stackexchange.com",
    "scifi":        "scifi.stackexchange.com",
    "movies":       "movies.stackexchange.com",
    "gardening":    "gardening.stackexchange.com",
    "parenting":    "parenting.stackexchange.com",
    "outdoors":     "outdoors.stackexchange.com",
    "boardgames":   "boardgames.stackexchange.com",
}

CODE_RE = re.compile(r"<(code|pre)[ >].*?</\1>", re.S)
TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")


def download(domain):
    os.makedirs(CACHE, exist_ok=True)
    dst = os.path.join(CACHE, f"{domain}.7z")
    if os.path.exists(dst) and os.path.getsize(dst) > 1000:
        return dst
    url = f"https://archive.org/download/stackexchange/{domain}.7z"
    print(f"  downloading {url}", flush=True)
    tmp = dst + ".part"
    urllib.request.urlretrieve(url, tmp)
    os.rename(tmp, dst)
    return dst


def extract_posts(archive, workdir):
    """Extract only Posts.xml from the 7z into workdir; return its path."""
    os.makedirs(workdir, exist_ok=True)
    subprocess.run(["7z", "e", "-y", f"-o{workdir}", archive, "Posts.xml"],
                   check=True, stdout=subprocess.DEVNULL)
    return os.path.join(workdir, "Posts.xml")


def html_to_text(body):
    """Strip code/pre blocks, then all tags; return (text, code_ratio)."""
    full_len = max(len(body), 1)
    code_len = sum(len(m.group(0)) for m in CODE_RE.finditer(body))
    no_code = CODE_RE.sub(" ", body)
    text = TAG_RE.sub(" ", no_code)
    text = html.unescape(text)
    text = WS_RE.sub(" ", text).strip()
    return text, code_len / full_len


def harvest(site, posts_path):
    """Stream Posts.xml, yield qualifying answer dicts for one site."""
    picks = []
    ctx = etree.iterparse(posts_path, events=("end",), tag="row")
    for _, row in ctx:
        try:
            if row.get("PostTypeId") != "2":            # answers only
                continue
            if int(row.get("Score", "0")) < MIN_SCORE:
                continue
            cdate = row.get("CreationDate", "")
            if not cdate or cdate >= CUTOFF:
                continue
            body = row.get("Body") or ""
            text, code_ratio = html_to_text(body)
            if code_ratio > 0.15:
                continue
            n = len(text.split())
            if not (MIN_WORDS <= n <= MAX_WORDS):
                continue
            picks.append({
                "source": f"stackexchange:{site}",
                "site": SITE_DOMAIN[site],
                "post_id": row.get("Id"),
                "score": int(row.get("Score")),
                "created": cdate[:10],
                "words": n,
                "label": "human",
                "text": text,
            })
        finally:
            row.clear()
            while row.getprevious() is not None:
                del row.getparent()[0]
    return picks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", default="", help="comma slugs; default = all prose sites")
    ap.add_argument("--pilot", action="store_true", help="two small sites only")
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "chunks.jsonl"))
    a = ap.parse_args()

    if a.pilot:
        sites = ["literature", "skeptics"]
    elif a.sites:
        sites = [s.strip() for s in a.sites.split(",") if s.strip()]
    else:
        sites = list(SITE_DOMAIN)
    bad = [s for s in sites if s not in SITE_DOMAIN]
    if bad:
        sys.exit(f"unknown site slug(s): {bad}")

    rng = random.Random(SEED)
    all_picks = []
    for site in sites:
        domain = SITE_DOMAIN[site]
        arc = download(domain)
        work = os.path.join(CACHE, site)
        posts = extract_posts(arc, work)
        picks = harvest(site, posts)
        rng.shuffle(picks)
        capped = picks[:PER_SITE_CAP]
        all_picks.extend(capped)
        print(f"  {site:13} {len(picks):5} qualify -> {len(capped)} kept", flush=True)
        try:
            os.remove(posts)                            # reclaim disk immediately
        except OSError:
            pass

    rng.shuffle(all_picks)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        for i, rec in enumerate(all_picks):
            rec = {"doc_id": f"se-{i:05d}", **rec}
            f.write(json.dumps(rec) + "\n")
    print(f"\nwrote {len(all_picks)} docs from {len(sites)} sites -> {a.out}")


if __name__ == "__main__":
    main()
