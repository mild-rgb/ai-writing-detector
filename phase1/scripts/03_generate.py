"""Generate style-matched AI answers for each ELI5 question via OpenRouter.

Every question is answered by all three models. The prompt constrains register
to a 2019 Reddit comment and pins the target length to the *specific* human
answer's word count, so the detector cannot win on length or markdown alone.

Resumable: reruns skip (id, model) pairs already present in the output file.
"""
import json
import os
import re
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

IN = os.environ.get("GEN_IN", "data/interim/questions_1000.jsonl")
OUT = os.environ.get("GEN_OUT", "data/interim/ai_answers.jsonl")
LIMIT = int(os.environ.get("GEN_LIMIT", "0"))   # 0 = all questions in IN
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

MODELS = ["x-ai/grok-4.6", "qwen/qwen3.8-max", "deepseek/deepseek-v4-pro"]
WORKERS = 6
MAX_RETRIES = 8

# One shared prompt cannot calibrate three models: the same instruction moves
# each a different distance, sometimes in opposite directions. v7's edits took
# qwen from 70% to 53.3% judge accuracy while taking grok from 56.7% to 26.7%.
# Each model therefore gets the version that measured closest to chance for it.
PROMPT_VERSION = os.environ.get("PROMPT_VERSION", "mixed")
PER_MODEL = {
    "x-ai/grok-4.6": "v6",              # v6 56.7% vs v7 26.7% (inverted)
    "qwen/qwen3.8-max": "v7",           # v7 53.3% vs v6 70.0%
    "deepseek/deepseek-v4-pro": "v6",   # v6 50.0% (n=30) vs v7 68.3% (n=60)
}
_cache = {}


def ver_of(model):
    return PER_MODEL.get(model, "v7") if PROMPT_VERSION == "mixed" else PROMPT_VERSION


def system_for(model):
    ver = ver_of(model)
    if ver not in _cache:
        _cache[ver] = open(f"prompts/{ver}.txt").read().strip()
    return ver, _cache[ver]

# Reddit-native artifacts appear in 7.4% of the human answers and no AI answer
# would produce them unprompted -- making them a free shortcut feature for any
# detector. We inject them into the AI class at the measured human base rates so
# the detector cannot win on them. Assignment is seeded per (id, model), so a
# rerun reproduces the same dataset exactly.
ARTIFACTS = [
    (0.051, "- End your comment with a line beginning 'Edit:' making a small "
            "correction or thanking people for a reply, the way a redditor "
            "amends a post after the fact."),
    (0.019, "- Open your comment with a 'TL;DR' line of one sentence, then a "
            "blank line, then the full explanation."),
    (0.007, "- Work in a bit of casual Reddit idiom (a reference to OP, "
            "upvotes, or karma) the way a regular commenter would."),
]


# v4 fixed the total absence of personal voice by instructing it unconditionally,
# which overshot: first person hit 80% against a human rate of 23%, hedging 43%
# against 3%. Presence and absence are both tells; only the RATE is neutral. So
# voice is drawn per generation at the measured human base rate, same mechanism
# as the Reddit artifacts.
PERSONA = [
    (0.233, "- Write this one in the first person. You are a person with an "
            "opinion and a life, not a reference work."),
    (0.032, "- Hedge your confidence somewhere: 'I think', 'I'm no expert but', "
            "'IIRC', 'not 100% sure'."),
    (0.009, "- Mention a scrap of your own experience -- your job, something a "
            "relative told you, something you saw once."),
    (0.027, "- You find this a bit stupid, and it shows. Blunt, a little "
            "sarcastic. Mild swearing is fine."),
    (0.041, "- Let one sentence end on an exclamation mark."),
]


# v8_invert mandated its markers and stamped them onto 24-87% of answers -- a
# giveaway feature no human in the sample ever wrote. v9 draws them instead, at
# rates modestly above the human base rate, and each draw asks for the idea in
# the model's own words rather than a fixed phrase.
INVERT_DRAWS = [
    (0.30, "- Somewhere in this comment, admit the limit of what you know. Phrase it "
           "however you would actually phrase it -- do not use a stock disclaimer."),
    (0.30, "- Ground one claim in something you have personally seen or done. Drop it "
           "in mid-thought where it is relevant, not as an opening credential."),
    (0.25, "- Add a line at the end beginning 'Edit:' that fixes a detail or answers "
           "a reply. Keep it short and specific to what you wrote."),
    (0.20, "- You find the premise a bit silly and it shows. Blunt, dry. Mild "
           "swearing if it fits."),
    (0.15, "- Let one sentence start lowercase or drop an apostrophe. Do not "
           "otherwise draw attention to it."),
]


def invert_draws_for(row_id, model):
    out = []
    for i, (rate, instruction) in enumerate(INVERT_DRAWS):
        if random.Random(f"{row_id}|{model}|inv{i}").random() < rate:
            out.append(instruction)
    return out


def persona_for(row_id, model):
    """Independent per-feature draws at the human base rates."""
    lines = []
    for i, (rate, instruction) in enumerate(PERSONA):
        if random.Random(f"{row_id}|{model}|persona{i}").random() < rate:
            lines.append(instruction)
    return lines


def artifact_for(row_id, model):
    """Deterministically assign an artifact instruction at the human base rate."""
    r = random.Random(f"{row_id}|{model}|artifact")
    draw, floor = r.random(), 0.0
    for rate, instruction in ARTIFACTS:
        if floor <= draw < floor + rate:
            return instruction
        floor += rate
    return None

lock = threading.Lock()
done = set()
if os.path.exists(OUT):
    with open(OUT) as f:
        for line in f:
            try:
                r = json.loads(line)
                done.add((r["id"], r["model"]))
            except json.JSONDecodeError:
                pass

def load_key():
    """Env var wins; otherwise read a KEY=value .env at the project root."""
    if os.environ.get("OPENROUTER_API_KEY"):
        return os.environ["OPENROUTER_API_KEY"]
    if os.path.exists(".env"):
        for line in open(".env"):
            k, _, v = line.partition("=")
            if k.strip() == "OPENROUTER_API_KEY":
                return v.strip().strip("'\"")
    return None


key = load_key()
if not key:
    sys.exit("OPENROUTER_API_KEY not found in the environment or ./.env")
headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

rows = [json.loads(l) for l in open(IN)]
if LIMIT:
    rows = rows[:LIMIT]
jobs = [(r, m) for r in rows for m in MODELS if (r["id"], m) not in done]
print(f"{len(done)} already done, {len(jobs)} to generate", file=sys.stderr)

counter = [0]


def gen(job):
    rec, model = job
    q = rec["question"]
    if rec["selftext"]:
        # The source corpus replaced links with _URL_n_ placeholders. Left in,
        # they put a token in the prompt that appears nowhere in natural text
        # and that a model may echo into its answer -- which would contaminate
        # the AI class with a marker the human class was already screened for.
        # 61% of the markers sit inside markdown links -- [anchor](_URL_0_) --
        # so deleting the token alone leaves a broken "[anchor]()". Unwrap those
        # to their anchor text first, then drop any bare markers.
        body_text = re.sub(r"\[([^\]]*)\]\(\s*_URL_\d*_\s*\)", r"\1",
                           rec["selftext"][:800])
        body_text = re.sub(r"_URL_\d*_", "", body_text)
        body_text = re.sub(r"\s+([,.;:!?])", r"\1", body_text)
        body_text = re.sub(r"[ \t]{2,}", " ", body_text).strip()
        if body_text:
            q += "\n\n" + body_text
    ver_now = ver_of(model)
    if ver_now == "v9":
        extra, voice = None, invert_draws_for(rec["id"], model)
    elif ver_now.startswith("v8_invert"):
        extra, voice = None, []
    else:
        extra = artifact_for(rec["id"], model)
        voice = persona_for(rec["id"], model)
    ver, template = system_for(model)
    system = template.format(n=rec["human_words"])
    for line in voice + ([extra] if extra else []):
        system += f"\n{line}"
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": q},
        ],
        # All three endpoints make reasoning mandatory, and reasoning tokens are
        # drawn from max_tokens BEFORE the visible answer. A budget sized only to
        # the answer truncated Gemini to ~15 words in the pilot. Size for
        # reasoning (~1-2k tokens observed) plus the answer.
        "max_tokens": max(2500, int(rec["human_words"] * 4) + 2000),
        "temperature": 1.0,
        "reasoning": {"effort": "low"},
    }
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(ENDPOINT, headers=headers, json=body, timeout=180)
            if resp.status_code == 429:
                # Honour the server's own pacing rather than guessing.
                wait = float(resp.headers.get("Retry-After") or 0) or min(60, 5 * 2 ** attempt)
                time.sleep(wait + random.random())
                raise RuntimeError("http 429")
            if resp.status_code >= 500:
                raise RuntimeError(f"http {resp.status_code}")
            resp.raise_for_status()
            data = resp.json()
            choice = data["choices"][0]
            text = (choice["message"]["content"] or "").strip()
            # Em dashes survived explicit prohibitions in v3 and v5 (10% vs a
            # 0.4% human rate), so stop asking and substitute. This is a
            # punctuation transformation, not a content edit; it is recorded in
            # the README as a documented normalisation.
            text = re.sub(r"\s*—\s*", ", ", text).replace("–", "-")
            text = text.replace("\u2019", "'").replace("\u2018", "'")
            text = text.replace("\u201c", '"').replace("\u201d", '"')
            text = re.sub(r"\s*…\s*", "... ", text)
            text = re.sub(r",\s*,", ",", text).strip()
            if not text:
                raise RuntimeError("empty completion")
            if choice.get("finish_reason") == "length":
                raise RuntimeError("truncated (finish_reason=length)")
            # Guard against silent truncation that does not set finish_reason.
            if len(text.split()) < 0.5 * rec["human_words"]:
                raise RuntimeError(f"too short: {len(text.split())}w vs {rec['human_words']}w")
            out = {
                "id": rec["id"],
                "q_id": rec["q_id"],
                "model": model,
                "question": rec["question"],
                "text": text,
                "target_words": rec["human_words"],
                "words": len(text.split()),
                "prompt_version": ver,
                "artifact": (extra.split("'")[1] if extra and "'" in extra
                             else ("slang" if extra else None)),
                "persona": len(voice),
                "usage": data.get("usage", {}),
            }
            with lock:
                with open(OUT, "a") as f:
                    f.write(json.dumps(out) + "\n")
                counter[0] += 1
                if counter[0] % 50 == 0:
                    print(f"  {counter[0]}/{len(jobs)}", file=sys.stderr, flush=True)
            return
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                print(f"FAIL {rec['id']} {model}: {e}", file=sys.stderr, flush=True)
                return
            time.sleep((2 ** attempt) + random.random())


with ThreadPoolExecutor(max_workers=WORKERS) as ex:
    list(ex.map(gen, jobs))
print(f"generated {counter[0]}", file=sys.stderr)
