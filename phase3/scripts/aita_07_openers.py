"""Generate only the OPENER of each corpus document, to test scenario diversity.

The question this answers: at corpus scale, do the generators invent 2,900
distinct situations, or do they recycle a handful of scenarios dressed
differently? That is not visible at n=70 and it is expensive to discover after
2,900 full documents have been paid for.

Openers are the right probe. The scenario is chosen in the first sentence, and
whatever a model is going to reuse -- the same setup, the same character name,
the same "It was a Tuesday and I was in the break room" scene-setter -- shows up
there. Capping output at a few dozen tokens costs about a tenth of the full run.

The prompt is assembled by importing generate.py's own build_system(), so the
openers are produced under the identical system prompt, the identical per-model
length and paragraph calibration, and the identical per-document feature draws
as the real corpus would use. Nothing is re-implemented here.

This is INDEPENDENT of the two pending fixes: the glm length constant and the
verdict-question phrasing affect how a document is sized and how it ENDS, not
which situation it opens with.

DO NOT USE THIS EXPECTING IT TO BE CHEAP. Measured on this roster: capping
output tokens does not truncate these models. Reasoning runs first and is billed
whether or not the cap lets any content through, so five of the seven return an
EMPTY completion at max_tokens=160 and charge ~160 tokens for it, while grok
ignores the cap and bills 4,183. A 30-word opener costs what a full document
costs. The full table is in phase3/aita/README.md. Kept because the prompt
assembly is correct and reusable, not because the saving exists.

    python3 phase3/scripts/aita_07_openers.py --limit 70    # pilot
    python3 phase3/scripts/aita_07_openers.py               # all 2,900
"""
import argparse
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, f"{ROOT}/phase3/scripts")
import generate as G  # noqa: E402  -- reuse the real prompt assembly

ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

# Enough headroom that a reasoning model still emits visible content. generate.py
# records nemotron spending its ENTIRE budget on reasoning and returning an empty
# completion at max_tokens 4000 and 8000, which `reasoning.exclude` fixes; the
# same override is carried over here. 160 tokens is roughly 3x what 30 words
# needs, and still about a tenth of a full document.
MAX_TOKENS = 160
WORDS = 30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-dir", default=f"{ROOT}/phase3/aita/prompts/aita_floor")
    ap.add_argument("--assignment", default=f"{ROOT}/phase3/aita/data/corpus_assignment.json")
    ap.add_argument("--humans", default=f"{ROOT}/phase3/aita/data/aita_human.jsonl")
    ap.add_argument("--out", default=f"{ROOT}/phase3/aita/data/openers.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    pdir = a.prompt_dir.rstrip("/")
    G.PROMPT_NAME[0] = os.path.basename(pdir)
    template = open(f"{pdir}/prompt.txt").read().strip()
    cfg = json.load(open(f"{pdir}/config.json"))
    assign = json.load(open(a.assignment))["assignment"]
    humans = {json.loads(l)["id"]: json.loads(l) for l in open(a.humans)}

    rows = [(q, m) for q, m in sorted(assign.items())]
    if a.limit:
        rows = rows[:a.limit]

    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            if line.strip():
                r = json.loads(line)
                done.add((r["id"], r["model"]))
    todo = [(q, m) for q, m in rows if (q, m) not in done]
    print(f"{len(rows)} assigned, {len(done)} present, {len(todo)} to generate",
          file=sys.stderr)

    if a.dry_run:
        q, m = todo[0]
        sysmsg, n, fields = G.build_system(template, humans[q], cfg, m)
        print(f"\n--- {q} / {m} ({n} draws, targets {fields}) ---\n{sysmsg}\n--- end ---")
        print("dry run: nothing sent, nothing spent", file=sys.stderr)
        return

    key = G.load_key()
    if not key:
        sys.exit("FATAL: OPENROUTER_API_KEY not found")
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    lock = threading.Lock()
    out = open(a.out, "a")
    usage = {"in": 0, "out": 0}
    fails = []
    n_done = [0]

    def one(item):
        q, m = item
        rec = humans[q]
        sysmsg, _, _ = G.build_system(template, rec, cfg, m)
        payload = {"model": m, "temperature": 1.0, "max_tokens": MAX_TOKENS,
                   "messages": [{"role": "system", "content": sysmsg},
                                {"role": "user", "content": rec["question"]}]}
        payload.update(G.REQUEST_OVERRIDES.get(m, {}))
        last = None
        for attempt in range(a.retries):
            try:
                r = requests.post(ENDPOINT, headers=headers, json=payload, timeout=120)
                if r.status_code == 429:
                    time.sleep(min(45, 4 * 2 ** attempt) + random.random())
                    raise RuntimeError("429")
                if r.status_code == 402:
                    sys.exit("FATAL: 402 Payment Required. Stopping the run.")
                r.raise_for_status()
                d = r.json()
                txt = (d["choices"][0]["message"]["content"] or "").strip()
                if not txt:
                    raise ValueError("empty completion")
                opener = " ".join(txt.split()[:WORDS])
                u = d.get("usage", {})
                with lock:
                    out.write(json.dumps({"id": q, "model": m,
                                          "question": rec["question"],
                                          "opener": opener,
                                          "raw_words": len(txt.split())}) + "\n")
                    out.flush()
                    usage["in"] += u.get("prompt_tokens", 0)
                    usage["out"] += u.get("completion_tokens", 0)
                    n_done[0] += 1
                    if n_done[0] % 200 == 0:
                        print(f"  {n_done[0]}/{len(todo)}", file=sys.stderr, flush=True)
                return
            except Exception as e:
                last = e
                time.sleep((1.5 ** attempt) + random.random())
        with lock:
            fails.append((q, m, str(last)))

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        list(ex.map(one, todo))
    out.close()
    print(f"generated {n_done[0]}, failures {len(fails)} -> {a.out}", file=sys.stderr)
    print(f"tokens in {usage['in']:,} out {usage['out']:,}", file=sys.stderr)
    if fails:
        with open(a.out.replace(".jsonl", "_failures.jsonl"), "w") as f:
            for q, m, e in fails:
                f.write(json.dumps({"id": q, "model": m, "error": e}) + "\n")
        from collections import Counter
        print("failures by model: " +
              ", ".join(f"{m.split('/')[-1]} {c}"
                        for m, c in Counter(m for _, m, _ in fails).items()),
              file=sys.stderr)


if __name__ == "__main__":
    main()
