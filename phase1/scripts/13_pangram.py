"""Run the labeled single-document benchmark through Pangram.

Async task API: POST /task returns a task_id, then poll GET /task/{id} until
stage is STAGE_SUCCESS or STAGE_FAILED. Resumable -- reruns skip items already
written. Rate limit is 5 QPS, so concurrency is capped low.
"""
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

KEYF, AI_SRC, OUT, MODEL = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
BASE = "https://text.external-api.pangram.com"
WORKERS = 4                      # under the 5 QPS limit
MIN_WORDS = 50                   # Pangram's stated minimum for a prediction

key = [l.partition("=")[2].strip() for l in open(".env")
       if l.startswith("PANGRAM_API_KEY")][0]
H = {"x-api-key": key, "Content-Type": "application/json"}

hum = {}
for f in ("data/interim/questions_1000.jsonl", "data/interim/questions_longform.jsonl"):
    if os.path.exists(f):
        for l in open(f):
            r = json.loads(l)
            hum[r["id"]] = r["human_answer"]
ai = {}
for l in open(AI_SRC):
    r = json.loads(l)
    ai[(r["id"], r["model"])] = r["text"]

done = set()
if os.path.exists(OUT):
    for l in open(OUT):
        try:
            done.add(json.loads(l)["item"])
        except json.JSONDecodeError:
            pass

jobs = []
for l in open(KEYF):
    k = json.loads(l)
    if k["item"] in done:
        continue
    text = hum[k["id"]] if k["label"] == "human" else ai.get((k["id"], k["model"]), "")
    if not text:
        continue
    jobs.append((k, text))
print(f"{len(done)} already done, {len(jobs)} to submit", file=sys.stderr)

lock = threading.Lock()
counter = [0]


def run(job):
    k, text = job
    words = len(text.split())
    for attempt in range(4):
        try:
            r = requests.post(f"{BASE}/task", headers=H, timeout=90,
                              json={"text": text, "model": MODEL,
                                    "public_dashboard_link": False})
            if r.status_code == 429:
                time.sleep(3 * (attempt + 1))
                continue
            r.raise_for_status()
            tid = r.json()["task_id"]
            for _ in range(45):
                time.sleep(2)
                g = requests.get(f"{BASE}/task/{tid}", headers=H, timeout=60)
                d = g.json()
                if d.get("stage") in ("STAGE_SUCCESS", "STAGE_FAILED"):
                    break
            row = {"item": k["item"], "label": k["label"], "model": k["model"],
                   "id": k["id"], "words": words, "stage": d.get("stage"),
                   "version": d.get("version"), "pred": d.get("prediction_short"),
                   "fraction_ai": d.get("fraction_ai"),
                   "fraction_ai_assisted": d.get("fraction_ai_assisted"),
                   "fraction_human": d.get("fraction_human")}
            with lock:
                with open(OUT, "a") as f:
                    f.write(json.dumps(row) + "\n")
                counter[0] += 1
                if counter[0] % 25 == 0:
                    print(f"  {counter[0]}/{len(jobs)}", file=sys.stderr, flush=True)
            return
        except Exception as e:
            if attempt == 3:
                print(f"FAIL {k['item']}: {e}", file=sys.stderr, flush=True)
                return
            time.sleep(2 * (attempt + 1))


with ThreadPoolExecutor(max_workers=WORKERS) as ex:
    list(ex.map(run, jobs))
print(f"completed {counter[0]}", file=sys.stderr)
