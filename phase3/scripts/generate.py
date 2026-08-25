"""Generate long-form answers for one phase-3 prompt across the model blend.

Adapted from phase1/scripts/03_generate.py. Four things are different:

1. **Seven models, one prompt.** Phase 1 concluded that one prompt cannot
   calibrate several models. This runs one prompt across all seven anyway,
   because the deliverable is a blend held constant across six prompts -- but
   every measurement downstream is reported per model as well as pooled, since
   a pooled figure earned by models cancelling on opposite sides of the human
   median is an artifact, not a match (see baseline_surface.py:best_interval).

2. **The draws live in the prompt's own config**, not in this file. Each of the
   six mechanisms rate-matches a different set of features, and the rates come
   from phase3/data/human_marker_rates.json -- measured on the LONG-FORM human
   class, which differs from the short-form rates phase 1 used (first person
   56.9% not 23.3%; `Edit:` 13.8% not 5.1%).

3. **Post-processing is rate-matched, not normalising.** Phase 1 substituted em
   dashes at write time because two prohibitions failed, which drove the rate to
   a perfect 0.0% against a human 2.3%. Here each punctuation feature is drawn
   per (id, model) at its measured human rate: drawn documents get the feature
   ensured, undrawn ones get it removed. A transformation either way, recorded
   as one, but the resulting rate matches instead of vanishing.

4. **Failures are recorded, not just printed.** A run writes failures.jsonl
   beside the answers so a short arm cannot masquerade as a complete one.

    python3 phase3/scripts/generate.py PROMPT_DIR --split dev --limit 5

Env: OPENROUTER_API_KEY, or a KEY=value line in phase1/.env.
"""
import argparse
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
CALIB_PATH = f"{ROOT}/phase3/data/length_calib.json"
CALIB_TABLE = json.load(open(CALIB_PATH)) if os.path.exists(CALIB_PATH) else {}
PROMPT_NAME = [""]
CORPUS = f"{ROOT}/phase1/data/interim/questions_longform.jsonl"
PARTITION = f"{ROOT}/phase3/data/longform_partition.json"

# Per-model request overrides. nemotron-3.5-lightning reasons until it hits
# max_tokens and returns an EMPTY completion with finish_reason=length: 3,479
# reasoning tokens and zero content at max_tokens=4000, and 6,551 at 8,000, so
# no budget fixes it. `reasoning.exclude` turns it off and the model answers
# normally. This is phase 1's "reasoning tokens come out of max_tokens before
# the visible answer" (§ generation) in its worst form -- there the answer was
# truncated to 15 words, here there is no answer at all.
# Per-model length calibration, measured on a 21-question pilot of p1_pinned.
# The prompt asks for {n} words; each model lands at a stable fraction of that,
# and the fraction is a property of the model: grok 0.66, gpt 0.86, qwen 0.86,
# gemini 0.91, deepseek 1.07, glm 1.08, nemotron 1.11. Asking every model for
# the same number therefore produces seven different length distributions, and
# word count -- which phase 1 pinned per question and measured at AUC 0.49 on
# short form -- comes back as a separable feature at 250-799 words (pooled 0.56,
# grok 0.80).
#
# The request is divided by the model's ratio so that what comes back lands on
# the human answer's length. This is phase 1's own recommendation from
# prompts/HISTORY.md -- "same rules, per-model constants" -- applied to the one
# dimension where it was never needed at short length. The constants are a first
# order correction measured on one prompt; whatever residual survives is
# reported by report_card.py rather than assumed away.
LENGTH_CALIB = {
    "x-ai/grok-4.6": 0.66,
    "openai/gpt-5.6-luna-pro": 0.86,
    "qwen/qwen3.8-max": 0.86,
    "google/gemini-3.7-flash": 0.91,
    "deepseek/deepseek-v4-pro": 1.07,
    "z-ai/glm-5.3": 1.08,
    "nvidia/nemotron-3.5-lightning": 1.11,
}
# Rejection band as a backstop, on the ACTUAL human target rather than the
# asked-for number. A completion outside it is resampled at most this many
# times, then accepted with its ratio recorded. Rejection sampling on a feature
# is a documented selection procedure, not a prose edit -- but it does select,
# so the accepted-on-last-try count is printed at the end of a run.
LENGTH_BAND = (0.80, 1.30)
LENGTH_RETRIES = 2
# A 429 and a length resample are not failures of the request, so neither may
# consume the budget reserved for real errors. Running 60 workers against the
# provider produced 62 rate-limit failures out of 420 documents -- a 16% hole in
# the arm that looked exactly like a model refusing to answer. Soft events get
# their own, larger budget and their own backoff.
# Lowered from 20 after a rate-limited model (grok) monopolised workers: 20
# retries with waits up to 45s can hold one worker for a quarter of an hour, and
# the account-wide rate limit means adding concurrency makes it worse rather
# than better. Fail the document quickly, record it, and fill the gaps in a
# dedicated low-concurrency pass instead.
SOFT_LIMIT = 8

REQUEST_OVERRIDES = {
    "nvidia/nemotron-3.5-lightning": {"reasoning": {"exclude": True}},
}

# x-ai/grok-4.6 was removed from the project after v1. Its account-wide rate
# limit was the binding constraint on every run here, and the 429-shaped holes
# it left were concentrated in one generator -- a shortfall that looks exactly
# like a real per-model effect unless the failure log is read. Its v1 cells are
# kept as the record, and they carry the clearest evidence in this phase that
# the paragraph pin works: newline separability 0.955 in phase 1 against 0.509
# under p1_pinned.
MODELS = [
    "qwen/qwen3.8-max",
    "deepseek/deepseek-v4-pro",
    "nvidia/nemotron-3.5-lightning",
    # The roster named `google/gemini-3.7-flash:batch`. That variant is
    # async-only -- the synchronous chat/completions endpoint returns 404 for
    # it -- so the sync model is used instead. Same model, different scheduling
    # tier: $0.38/$1.88 per M against the batch tier's $0.19/$0.94, which is
    # about a dollar across this whole project.
    "google/gemini-3.7-flash",
    "openai/gpt-5.6-luna-pro",
    "z-ai/glm-5.3",
]


def seeded(row_id, model, tag):
    return random.Random(f"{row_id}|{model}|{tag}").random()


def draw_name(row_id, model, pool):
    """Pick a character name, weighted by how often REAL posters used it.

    Weighted rather than uniform on purpose. A uniform draw over a few hundred
    names would give every document a near-unique name and land the corpus
    *more* name-diverse than any real population -- phase 1's v4 overshoot in a
    new place, where inverting a tell is as detectable as leaving it. Weighting
    by the human counts reproduces the human distribution, mild recurrence and
    all: John really is commoner than Wyatt.
    """
    names = [p["name"] for p in pool]
    weights = [p["human_count"] for p in pool]
    return random.Random(f"{row_id}|{model}|name").choices(names, weights)[0]


# --- rate-matched post-processing ----------------------------------------
# Each op takes (text, keep) where keep is the result of the per-document draw.
# keep=True  -> ensure the feature is present at least once.
# keep=False -> ensure it is absent.
# Neither branch is a prohibition: across a corpus the feature lands at the
# drawn rate, which is the human rate.

def op_em_dash(text, keep, ctx):
    # Position is drawn rather than always-first: an insertion that always lands
    # at the document's first comma is a positional signature even when the
    # per-document RATE matches the human class exactly.
    if keep:
        if "\u2014" in text:
            return text
        spots = list(re.finditer(r"(\w), (\w)", text))
        if not spots:
            return text
        m = spots[ctx["rnd"].randrange(len(spots))]
        return text[:m.start()] + m.group(1) + " \u2014 " + m.group(2) + text[m.end():]
    return re.sub(r"\s*\u2014\s*", ", ", text)


def op_en_dash(text, keep, ctx):
    return text if keep else text.replace("–", "-")


def op_curly(text, keep, ctx):
    """All or nothing, never one.

    The first version converted a SINGLE apostrophe when drawn, which matched
    the human rate perfectly on the marker regex ("does this document contain a
    curly quote") and manufactured a fingerprint no counter could see. A QA
    reader found it immediately: "exactly one curly apostrophe against straight
    apostrophes everywhere else, suggesting one sentence was pasted or edited".
    A real person's editor produces curly punctuation throughout a document or
    not at all -- the feature is a property of their client, not of a sentence.
    """
    if keep:
        text = re.sub(r"(?<=\w)'(?=\w)", "\u2019", text)
        text = re.sub(r"(?<!\w)'", "\u2018", text)
        text = re.sub(r"'", "\u2019", text)
        parts = text.split('"')
        if len(parts) > 1:
            out = parts[0]
            for i, seg in enumerate(parts[1:]):
                out += ("\u201c" if i % 2 == 0 else "\u201d") + seg
            text = out
        return text
    for a, b in (("\u2019", "'"), ("\u2018", "'"), ("\u201c", '"'), ("\u201d", '"')):
        text = text.replace(a, b)
    return text


def op_ellipsis(text, keep, ctx):
    # Same all-or-nothing rule as op_curly: a document that spells one ellipsis
    # as a character and the next as three dots is displaying an edit, not a
    # habit.
    if keep:
        return re.sub(r"\.\.\.", "\u2026", text)
    return re.sub(r"\s*\u2026\s*", "... ", text)


def op_double_space(text, keep, ctx):
    if keep:
        return re.sub(r"(?<=[.!?]) (?=[A-Z])", "  ", text)
    return re.sub(r"(?<=[.!?])  +(?=[A-Z])", " ", text)


def op_lowercase_para(text, keep, ctx):
    """A paragraph that starts lowercase. Human long-form rate 9.2%.

    The paragraph is drawn, not always the second one. A feature that always
    lands in the same position is a fingerprint at the right rate.
    """
    paras = text.split("\n\n")
    if keep:
        cands = [i for i, p in enumerate(paras) if i and p[:1].isupper()]
        if cands:
            i = cands[ctx["rnd"].randrange(len(cands))]
            paras[i] = paras[i][0].lower() + paras[i][1:]
        return "\n\n".join(paras)
    return "\n\n".join(p[0].upper() + p[1:] if p[:1].islower() else p for p in paras)


STRUCTURED = re.compile(r"^\s*(?:[-*•]\s+|\d+[.)]\s+|#{1,6}\s+|>\s+)")


def _sentences(block):
    return [s.strip() for s in
            re.findall(r"[^.!?]*[.!?]+[\])'\"]*|[^.!?]+$", block) if s.strip()]


def op_repara(text, keep, ctx):
    """Re-wrap to this question's own human paragraph count, mechanically.

    The deliberate contrast with asking the model for a paragraph count in the
    prompt. Phase 1 has both outcomes on record -- "write in the first person"
    moved a rate 0% -> 80%, and two explicit em-dash prohibitions moved nothing
    at all -- so whether structure is promptable is an open question with a
    known-good fallback. This is the fallback, and it is a transformation, not a
    prompting success. Sentence order and wording are untouched; only where the
    blank lines fall changes.

    Markdown structure is preserved as-is. The first version of this flattened
    every newline and re-split on sentence boundaries, which turned

        ### 1. Ancient and Classical Contexts

    into a paragraph reading "### 1." followed by its title merged into the next
    sentence -- in range on every counter this project computes and visibly
    broken to a reader. Structured lines are now hard block boundaries: they are
    never merged into prose and never split apart.
    """
    if not keep:
        return text
    target = max(1, ctx["fields"]["paras"])
    blocks, prose = [], []
    for line in text.split("\n"):
        if STRUCTURED.match(line):
            if prose:
                blocks.append(("prose", _sentences(" ".join(prose))))
                prose = []
            if blocks and blocks[-1][0] == "struct":
                blocks[-1][1].append(line.strip())
            else:
                blocks.append(("struct", [line.strip()]))
        elif line.strip():
            prose.append(line.strip())
    if prose:
        blocks.append(("prose", _sentences(" ".join(prose))))

    n_struct = sum(1 for kind, _ in blocks if kind == "struct")
    want = max(1, target - n_struct)
    rnd = ctx["rnd"]

    # Paragraphs are allocated to each prose block in proportion to its own
    # sentence count and split WITHIN that block. An earlier version pooled
    # every prose sentence in the document and re-cut the pool, which preserved
    # sentence order but moved prose across a header -- the closing paragraph
    # came out above the section title it followed. Order relative to structure
    # has to be preserved, not just order among sentences.
    prose_idx = [i for i, (kind, _) in enumerate(blocks) if kind == "prose"]
    total_sents = sum(len(blocks[i][1]) for i in prose_idx) or 1
    alloc = {}
    left = want
    for n, i in enumerate(prose_idx):
        if n == len(prose_idx) - 1:
            alloc[i] = max(1, left)
        else:
            share = max(1, round(want * len(blocks[i][1]) / total_sents))
            share = min(share, max(1, left - (len(prose_idx) - n - 1)))
            alloc[i] = share
            left -= share

    out = []
    for i, (kind, body) in enumerate(blocks):
        if kind == "struct":
            out.append("\n".join(body))
            continue
        k = min(alloc.get(i, 1), len(body))
        if k <= 1 or len(body) <= 1:
            out.append(" ".join(body))
            continue
        cuts = sorted(rnd.sample(range(1, len(body)), k - 1))
        prev = 0
        for c in cuts + [len(body)]:
            out.append(" ".join(body[prev:c]))
            prev = c
    return "\n\n".join(p for p in out if p.strip())


OPS = {"em_dash": op_em_dash, "en_dash": op_en_dash, "curly": op_curly,
       "ellipsis": op_ellipsis, "double_space": op_double_space,
       "lowercase_para": op_lowercase_para, "repara": op_repara}


def postprocess(text, row_id, model, post, fields):
    applied = {}
    # repara restructures the whole document, so it runs before the ops that
    # place a feature at a particular position.
    order = sorted(post, key=lambda k: (k != "repara", k))
    for name in order:
        rate = post[name]
        if name not in OPS:
            sys.exit(f"FATAL: unknown post op '{name}'. Have: {sorted(OPS)}")
        keep = seeded(row_id, model, f"post:{name}") < rate
        ctx = {"fields": fields,
               "rnd": random.Random(f"{row_id}|{model}|op:{name}")}
        text = OPS[name](text, keep, ctx)
        applied[name] = keep
    return re.sub(r"[ \t]+\n", "\n", text).strip(), applied


# --- prompt assembly ------------------------------------------------------

def build_system(template, rec, cfg, model):  # noqa: C901
    """Fill the template, then append whatever draws came up for this document.

    Placeholders available to a template: {n} target words (pinned to this
    question's own human answer, as phase 1 did), {paras} target paragraph
    count, {wpp} target words per paragraph. The last two exist because
    paragraph convention is the largest known defect in the phase 1 corpus
    (AUC 0.78-0.86, better than every detector below Sonnet) and nothing in v6
    said a word about it.
    """
    text = human_of(rec)
    paras = max(1, len([p for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]))
    cell = CALIB_TABLE.get(PROMPT_NAME[0], {}).get(model) or {}
    calib = cell.get("len") or LENGTH_CALIB.get(model, 1.0)
    asked = rec["human_words"] / max(0.35, min(2.0, calib))
    # Paragraph count gets the same treatment as length: nemotron lands at 0.93
    # of the paragraph count it is asked for where the other six average 1.05,
    # and that alone kept its newline separability at 0.66 under a prompt where
    # every other generator sat at 0.51.
    pcal = max(0.5, min(1.8, cell.get("para") or 1.0))
    fields = {"n": int(round(asked)), "paras": max(1, int(round(paras / pcal))),
              "wpp": max(1, round(asked / max(1, round(paras / pcal)))),
              "newlines": text.count("\n"),
              # the human answer's own paragraph count, before calibration, so
              # the emitted row can record target and ask separately
              "human_paras": paras}
    pool = cfg.get("name_pool")
    if pool:
        fields["name"] = draw_name(rec["id"], model, pool)
    try:
        system = template.format(**fields)
    except KeyError as e:
        sys.exit(f"FATAL: prompt uses unknown placeholder {e}. "
                 f"Available: {sorted(fields)}")
    lines = []
    for i, d in enumerate(cfg.get("draws", [])):
        hit = seeded(rec["id"], model, f"draw{i}") < d["rate"]
        # A two-sided draw instructs BOTH branches. Features the models produce
        # spontaneously cannot be rate-matched by a one-sided draw: asking for
        # markdown emphasis in 32.3% of documents produced it in 42.5% under one
        # prompt and 84.4% under another, because the draw only ever adds to
        # whatever the model was going to do anyway. Instructing the negative
        # branch pins the observed rate to the drawn rate regardless of the
        # model's own tendency, which is the same keep/remove logic the
        # punctuation post-ops use.
        line = d["text"] if hit else d.get("else_text")
        if line:
            # Draw text may carry {name}; ELI5's does not, so this is a no-op
            # there. Formatted only when a placeholder is present, so a literal
            # brace in any other prompt cannot raise.
            if "{" in line:
                line = line.format(**fields)
            lines.append("- " + line)
    # An exclusive draw, phase 1's artifact_for mechanism: at most one option
    # fires, drawn from a cumulative distribution rather than as independent
    # Bernoullis. Use it where the options are mutually exclusive descriptions
    # of the same thing -- a document has one shape, not three.
    for name, options in sorted(cfg.get("choice", {}).items()):
        draw, floor = seeded(rec["id"], model, f"choice:{name}"), 0.0
        for o in options:
            if floor <= draw < floor + o["rate"]:
                lines.append("- " + o["text"])
                break
            floor += o["rate"]
    if lines:
        system += "\n" + "\n".join(lines)
    return system, len(lines), fields


def human_of(rec):
    return rec["human_answer"]


def load_key():
    if os.environ.get("OPENROUTER_API_KEY"):
        return os.environ["OPENROUTER_API_KEY"]
    for path in (f"{ROOT}/phase1/.env", f"{ROOT}/.env"):
        if os.path.exists(path):
            for line in open(path):
                k, _, v = line.partition("=")
                if k.strip() == "OPENROUTER_API_KEY":
                    return v.strip().strip("'\"")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt_dir")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--corpus", default=None,
                    help="a questions file whose rows each carry their OWN "
                         "`model`. One ai document per question, generator "
                         "fixed by the corpus assignment rather than by the "
                         "roster, so provenance is decided once at sampling "
                         "time and survives into attribution.")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--out")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--retries", type=int, default=6)
    ap.add_argument("--dry-run", action="store_true",
                    help="assemble every prompt and print one, spend nothing")
    a = ap.parse_args()

    pdir = a.prompt_dir.rstrip("/")
    PROMPT_NAME[0] = os.path.basename(pdir)
    template = open(f"{pdir}/prompt.txt").read().strip()
    cfg_path = f"{pdir}/config.json"
    cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
    if a.corpus:
        rows = [json.loads(l) for l in open(a.corpus) if l.strip()]
        rows.sort(key=lambda r: r["id"])
        missing = [r["id"] for r in rows if not r.get("model")]
        if missing:
            sys.exit(f"FATAL: {len(missing)} corpus rows carry no assigned "
                     f"model (first {missing[:3]}). Provenance is assigned at "
                     f"sampling time, not here.")
    else:
        part = json.load(open(PARTITION))
        if a.split not in part:
            sys.exit(f"FATAL: no split '{a.split}' in {PARTITION}")
        ids = set(part[a.split])
        rows = [json.loads(l) for l in open(CORPUS) if json.loads(l)["id"] in ids]
        rows.sort(key=lambda r: r["id"])
    if a.limit:
        rows = rows[:a.limit]

    out_path = a.out or f"{pdir}/answers_{a.split}.jsonl"
    fail_path = out_path.replace(".jsonl", "_failures.jsonl")
    done = set()
    if os.path.exists(out_path):
        for i, line in enumerate(open(out_path), 1):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError as e:
                sys.exit(f"FATAL {out_path}:{i}: {e}. Refusing to resume past a "
                         f"line I cannot parse -- a silent skip here is a silent "
                         f"sample-size cut.")
            done.add((r["id"], r["model"]))
    # A restart that overlaps a still-running process appends instead of
    # resuming, and both writers then emit the same (id, model). Phase 1 shipped
    # 65 such pairs and three figures computed on them. Rewrite the file deduped
    # at startup, last row wins, and say so loudly rather than leaving a
    # duplicated arm to be discovered by whoever scores it.
    if os.path.exists(out_path):
        uniq, order = {}, []
        for line in open(out_path):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            k = (r["id"], r["model"])
            if k not in uniq:
                order.append(k)
            uniq[k] = line
        if len(uniq) != len(order) or sum(1 for _ in open(out_path) if _.strip()) != len(uniq):
            n_before = sum(1 for _ in open(out_path) if _.strip())
            with open(out_path, "w") as f:
                for k in order:
                    f.write(uniq[k] + "\n")
            print(f"DEDUPED {out_path}: {n_before} rows -> {len(uniq)} unique "
                  f"(id, model) pairs. A duplicated arm is not a larger sample.",
                  file=sys.stderr)
            done = set(uniq)

    if a.corpus:
        jobs = [(r, r["model"]) for r in rows if (r["id"], r["model"]) not in done]
    else:
        jobs = [(r, m) for r in rows for m in a.models
                if (r["id"], m) not in done]

    print(f"prompt   {pdir}", file=sys.stderr)
    if a.corpus:
        print(f"corpus   {a.corpus}: {len(rows)} questions, one ai document "
              f"each, generator fixed per question", file=sys.stderr)
    else:
        print(f"split    {a.split}: {len(rows)} questions x {len(a.models)} "
              f"models", file=sys.stderr)
    print(f"resume   {len(done)} present, {len(jobs)} to generate", file=sys.stderr)
    print(f"draws    {len(cfg.get('draws', []))}   post-ops "
          f"{sorted(cfg.get('post', {}))}", file=sys.stderr)

    if a.dry_run:
        rec, model = jobs[0]
        system, nd, fields = build_system(template, rec, cfg, model)
        print(f"\n--- assembled system prompt for {rec['id']} / {model} "
              f"({nd} draws hit, targets {fields}) ---\n{system}\n--- end ---")
        hits = {}
        for i, d in enumerate(cfg.get("draws", [])):
            k = sum(1 for r in rows for m in a.models
                    if seeded(r["id"], m, f"draw{i}") < d["rate"])
            hits[d["text"][:46]] = f"{100*k/(len(rows)*len(a.models)):.1f}% (target {100*d['rate']:.1f}%)"
        for k, v in hits.items():
            print(f"  draw  {k:<48} {v}")
        for name, options in sorted(cfg.get("choice", {}).items()):
            tally = {}
            for r in rows:
                for m in a.models:
                    draw, floor = seeded(r["id"], m, f"choice:{name}"), 0.0
                    for o in options:
                        if floor <= draw < floor + o["rate"]:
                            tally[o["text"][:44]] = tally.get(o["text"][:44], 0) + 1
                            break
                        floor += o["rate"]
            tot = len(rows) * len(a.models)
            for o in options:
                k = tally.get(o["text"][:44], 0)
                print(f"  choice {name}: {o['text'][:40]:<42} "
                      f"{100*k/tot:.1f}% (target {100*o['rate']:.1f}%)")
        for name, rate in sorted(cfg.get("post", {}).items()):
            k = sum(1 for r in rows for m in a.models
                    if seeded(r["id"], m, f"post:{name}") < rate)
            print(f"  post  {name:<48} {100*k/(len(rows)*len(a.models)):.1f}% "
                  f"(target {100*rate:.1f}%)")
        print("\ndry run: nothing sent, nothing spent", file=sys.stderr)
        return

    key = load_key()
    if not key:
        sys.exit("FATAL: OPENROUTER_API_KEY not found")
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    lock = threading.Lock()
    counter = [0]
    out_of_band = [0]
    usage_total = {"prompt_tokens": 0, "completion_tokens": 0}

    def gen(job):
        rec, model = job
        q = rec["question"]
        if rec.get("selftext"):
            body_text = re.sub(r"\[([^\]]*)\]\(\s*_URL_\d*_\s*\)", r"\1",
                               rec["selftext"][:800])
            body_text = re.sub(r"_URL_\d*_", "", body_text)
            body_text = re.sub(r"\s+([,.;:!?])", r"\1", body_text)
            body_text = re.sub(r"[ \t]{2,}", " ", body_text).strip()
            if body_text:
                q += "\n\n" + body_text
        system, n_draws, fields = build_system(template, rec, cfg, model)
        length_tries = [0]
        payload = {
            "model": model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": q}],
            # Reasoning tokens come out of max_tokens BEFORE the visible answer
            # on every endpoint phase 1 used; a budget sized to the answer alone
            # silently produced 15-word completions against 62-word targets.
            "max_tokens": max(3000, int(rec["human_words"] * 4) + 2500),
            "temperature": 1.0,
            "reasoning": {"effort": "low"},
        }
        payload.update(REQUEST_OVERRIDES.get(model, {}))
        last = None
        attempt = 0
        soft = 0
        while attempt < a.retries and soft < SOFT_LIMIT:
            try:
                resp = requests.post(ENDPOINT, headers=headers, json=payload,
                                     timeout=600 if ":batch" in model else 240)
                if resp.status_code == 429:
                    wait = (float(resp.headers.get("Retry-After") or 0)
                            or min(20, 3 * 2 ** min(soft, 3)))
                    time.sleep(wait + random.random() * 2)
                    soft += 1
                    last = RuntimeError("http 429")
                    continue
                if resp.status_code >= 500:
                    raise RuntimeError(f"http {resp.status_code}: {resp.text[:160]}")
                if resp.status_code == 402:
                    sys.exit(f"FATAL: 402 Payment Required from OpenRouter. "
                             f"Stopping the whole run rather than logging 400 "
                             f"failures -- phase 1 lost a long-form pass to "
                             f"exactly this and reported it as 'crawling'.")
                resp.raise_for_status()
                data = resp.json()
                if "choices" not in data:
                    raise RuntimeError(f"no choices in response: {str(data)[:200]}")
                choice = data["choices"][0]
                text = (choice["message"].get("content") or "").strip()
                if not text:
                    raise RuntimeError("empty completion")
                if choice.get("finish_reason") == "length":
                    raise RuntimeError("truncated (finish_reason=length)")
                text, applied = postprocess(text, rec["id"], model,
                                            cfg.get("post", {}), fields)
                ratio = len(text.split()) / rec["human_words"]
                if not (LENGTH_BAND[0] <= ratio <= LENGTH_BAND[1]):
                    if length_tries[0] < LENGTH_RETRIES:
                        length_tries[0] += 1
                        soft += 1
                        last = RuntimeError(f"length {ratio:.2f}x outside band")
                        continue
                    with lock:
                        out_of_band[0] += 1
                if len(text.split()) < 0.4 * rec["human_words"]:
                    raise RuntimeError(f"too short: {len(text.split())}w vs "
                                       f"{rec['human_words']}w target")
                # NO `split` here, deliberately. It was written once and went
                # stale the moment the corpus was re-split: 73 rows ended up
                # saying "train" for questions that had become test or dev, and
                # anything trusting the answer's copy would have pulled 73
                # held-out documents into training. The question file is the
                # single authority on which split a document belongs to.
                # Everything below is a fact about the generation EVENT, which
                # cannot go stale, or an identifier, which cannot change.
                out = {"id": rec["id"], "q_id": rec["q_id"], "model": model,
                       "seed": rec.get("seed"),
                       "question": rec["question"], "text": text,
                       "target_words": rec["human_words"],
                       "words": len(text.split()),
                       "asked_words": fields["n"],
                       "length_ratio": round(ratio, 3),
                       "length_tries": length_tries[0],
                       "target_paras": fields["human_paras"],
                       "asked_paras": fields["paras"],
                       "target_wpp": fields["wpp"],
                       "prompt": os.path.basename(pdir), "draws": n_draws,
                       "post": applied, "usage": data.get("usage", {})}
                with lock:
                    with open(out_path, "a") as f:
                        f.write(json.dumps(out) + "\n")
                    counter[0] += 1
                    u = data.get("usage", {})
                    usage_total["prompt_tokens"] += u.get("prompt_tokens", 0)
                    usage_total["completion_tokens"] += u.get("completion_tokens", 0)
                    if counter[0] % 25 == 0:
                        print(f"  {counter[0]}/{len(jobs)}", file=sys.stderr, flush=True)
                return
            except Exception as e:
                last = e
                attempt += 1
                if attempt < a.retries:
                    time.sleep((2 ** attempt) + random.random())
        with lock:
            with open(fail_path, "a") as f:
                f.write(json.dumps({"id": rec["id"], "model": model,
                                    "error": str(last)}) + "\n")
        print(f"FAIL {rec['id']} {model}: {last}", file=sys.stderr, flush=True)

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        list(ex.map(gen, jobs))

    n_fail = sum(1 for _ in open(fail_path)) if os.path.exists(fail_path) else 0
    print(f"generated {counter[0]}, failures {n_fail} -> {out_path}", file=sys.stderr)
    print(f"tokens in {usage_total['prompt_tokens']:,} "
          f"out {usage_total['completion_tokens']:,}", file=sys.stderr)
    if out_of_band[0]:
        print(f"{out_of_band[0]} documents accepted outside the length band "
              f"{LENGTH_BAND} after {LENGTH_RETRIES} resamples", file=sys.stderr)
    if n_fail:
        print(f"NOTE: {n_fail} failures recorded in {fail_path}. The arm is "
              f"short by that many; do not score until they are resolved or "
              f"the shortfall is stated.", file=sys.stderr)


if __name__ == "__main__":
    main()
