"""Stream the ELI5 (Facebook LFQA) dump and keep only r/explainlikeimfive rows.

Source corpus spans 2011-07 to 2019-07, so every record predates ChatGPT by
years. We stream-filter rather than storing the full 656MB mixed-subreddit file.
"""
import json
import sys

import requests

URL = "https://huggingface.co/datasets/Pavithree/eli5/resolve/main/train.json"
OUT = "data/raw/eli5_questions.jsonl"

kept = seen = 0
with requests.get(URL, stream=True, timeout=120) as r:
    r.raise_for_status()
    with open(OUT, "w") as out:
        buf = ""
        for chunk in r.iter_content(chunk_size=1 << 20, decode_unicode=True):
            buf += chunk
            *lines, buf = buf.split("\n")
            for line in lines:
                if not line.strip():
                    continue
                seen += 1
                rec = json.loads(line)
                if rec.get("subreddit") != "explainlikeimfive":
                    continue
                out.write(line + "\n")
                kept += 1
            if seen % 50000 < 1000:
                print(f"  seen={seen} kept={kept}", file=sys.stderr, flush=True)

print(f"done: seen={seen} kept={kept} -> {OUT}", file=sys.stderr)
