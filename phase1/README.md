# Can you tell AI writing from human writing?

This repository set out to build a labeled corpus for AI-text detection. It
became, mostly, a measurement of how badly that task goes and how easily a
detection result can be an artifact instead.

The single most useful sentence in it is this: **on every benchmark here,
counting the line breaks in a document — one fitted integer, no model —
detects AI better than Claude Haiku 4.5 does.** Nobody noticed for months.

---

## Status

**The production dataset was never built.** `data/final/` is empty and
`data/interim/ai_answers.jsonl` holds 2 rows. Generation was run for the
detectability study — 100 `dev` questions across four prompt versions, and two
long-form benchmark sets — but never for the 650-question production pool.

What exists is the corpus machinery, the eval-set partition, and a study. If you
came here for a dataset you will need to run stage 3 yourself (§ Pipeline). If
you came here for findings about detection, they are below.

## What is here

| | |
|---|---|
| `README.md` | this file — what the repo is and what is true |
| `NARRATIVE.md` | how it was built and what went wrong, including retracted claims |
| `study/DESIGN.md` | the detectability protocol: power, splits, stopping rules |
| `study/small/RESULTS.md` | a sub-1.7B local detector, benchmarked against the above |
| `prompts/HISTORY.md` | the nine generator prompt versions and why each changed |

`NARRATIVE.md` is the one to read. It is a lab notebook that records its own
errors, and roughly a third of it is corrections to the other two thirds.

---

## The design

**Human side.** 1,000 question/answer pairs from r/explainlikeimfive, via the
Facebook ELI5 (LFQA) corpus, which spans 2011-07 to 2019-07. Every human answer
predates ChatGPT by more than three years, so the human label is true by
construction rather than by assumption. 107,280 r/eli5 rows were streamed out of
a 656MB mixed-subreddit dump; 17,368 passed filtering (top-scored answer, score
>= 10, 50-250 words, deduped by question); 1,000 were sampled. A second
companion set of 1,000 long-form questions (250-799 words, median 345) is
disjoint by construction.

**AI side.** Three flagship models answer the same questions, giving paired data
rather than three disjoint thirds:

| Model | OpenRouter ID | $/1M in | $/1M out |
|---|---|---|---|
| Grok 4.6 | `x-ai/grok-4.6` | 2.00 | 6.00 |
| Qwen3.8 Max | `qwen/qwen3.8-max` | 2.00 | 6.00 |
| DeepSeek V4 Pro | `deepseek/deepseek-v4-pro` | 1.60 | 3.20 |

Reasoning is mandatory on all three endpoints and reasoning tokens are drawn
from `max_tokens` before the visible answer, so a budget sized to the answer
alone truncates output silently — an early pilot produced 15-word answers
against 62-word targets with no error raised. `max_tokens` is sized for
reasoning plus answer and the generator rejects any completion under half its
target length. That guard still fires: it caught a 14-word answer against a
284-word target during the last generation run in this repo.

**Style matching.** Generation is pinned to 2019-era Reddit register — plain
text, no markdown, no greeting or sign-off, no restating the question — and the
target length is pinned per question to *that question's* human answer word
count. An unconstrained LLM writes 300 formatted words where the median ELI5
answer is 75 unformatted ones, and a detector trained on that gap learns to
count bullet points instead of reading prose.

**Rate matching, the one technique that kept working.** Shortcut features get
injected into the AI class at their measured human base rates, seeded from
`(id, model)` so runs reproduce exactly. Not prohibited — *matched*. Every
prohibition drives a feature to exactly 0.0% where humans sit at 2-7%, and **a
perfect zero is a fingerprint**; absence is as detectable as excess.

The program has since been verified end to end: designed 5.1% / 1.9% / 0.7%
against observed 4.3% / 2.0% / 0.7%, four rates across two draw mechanisms, all
within sampling error — and the injections do not move a detector's score.

**Splits are grouped by question**, and the question set is partitioned once and
permanently: `burned` (150, retired — judges saw those answers), `dev` (100,
reused deliberately so comparisons are paired), `heldout` (100, never touched),
`pool` (650, feeds the production dataset that was never generated).

---

## What the study found

### 1. Detection on a single short document is bad

Every item judged alone — no reference text, no sibling set — because that is
the task a detector actually faces. Generation is v6, the ordinary prompt.

| | detector | detection | false positives | balanced |
|---|---|---|---|---|
| **short** — `short`, median 101w | Haiku 4.5 | 26.7% | 20.0% | 53.3% |
| | Sonnet | 56.0% | 8.0% | 74.0% |
| | Pangram 3.3.2 | 77.3% | 0.0% | 88.7% |
| **long** — `long2`, median 342w | Haiku 4.5 | 24.0% | 9.3% | 57.3% |
| | Sonnet | 78.7% | 1.3% | 88.7% |
| **long** — `long`, median 312w † | Pangram 3.3.2 | 97.6% | 0.0% | 98.8% |

† Pangram's long-form figure is from the earlier `long` set (41 AI documents
after deduplication, not 75) and was **not** re-run on the rebuilt `long2`. It
sits in a separate row because it is not measured on the same items as the two
above it. The `short` rows are all the same 150 items. See `NARRATIVE.md` §12
and §14 for how `long` was rebuilt and what did and did not move.

Haiku catches one AI answer in four while falsely accusing one human in five.
Pangram's published ~98% at ~0.05% replicates — **at the length it recommends**;
on short form the same model drops to 77.3%. Their 250-word threshold is not
marketing.

**Length is decisive, but only for detectors capable of using it.** Sonnet gains
22.7 points from short to long (p=0.003); Haiku gains nothing (−2.7, p=0.71).

### 2. `wc -l` beats Haiku 4.5 everywhere

Count the line breaks. That is the whole classifier.

| set | newline AUC | best-threshold balanced accuracy |
|---|---|---|
| `long` | 0.862 | **79.2%** |
| `long3` | 0.803 | 72.7% |
| `long2` | 0.802 | 74.0% |
| `short` | 0.760 | 74.9% |

All on distinct documents. The threshold is fitted in place — one integer, so
close to free, and the AUC column needs no threshold at all — but it is fitted,
and no detector figure in this project was ever quoted against any baseline at
all.

**Like for like on short form**, every figure on the same 75 AI and 58 distinct
human documents:

| | detection | false positives | balanced |
|---|---|---|---|
| `wc -l` | 77.3% | 25.3% | **74.9%** |
| Sonnet | 56.0% | 8.6% | 73.7% |
| Haiku 4.5 | 26.7% | 25.9% | **50.4%** |

Two things fall out of matching the denominators. **Haiku 4.5 on a short single
document is at chance to one decimal place** once its false-positive arm is
de-duplicated — the 53.3% reported elsewhere in this repo is computed on 75
human items that are only 58 documents. And `wc -l` edges out Sonnet here too,
not merely Haiku — narrowly, on a fitted threshold against Sonnet's zero-shot
verdicts, so it is not a headline, but "beats Haiku" was leaving something out.

(The detector table in §1 keeps its item-level figures, because those are what
`NARRATIVE.md` §12 and §14 report. It is only this comparison that needs matched
denominators, and only `short` and `long` are affected — `long2` and `long3` have
no duplication.)

It is not a planted artifact. `prompts/v6.txt` says nothing about paragraphs. It
is an omission from the rate-matching program: word count, markdown, `Edit:`/
`TL;DR` and persona were all matched, and paragraph structure — the most
separable surface feature of the lot — was never considered.

**The spread across generators is the real finding.** Median long-form structure:
human 11 newlines / 6 paragraphs / 59 words per paragraph; qwen 8 / 5 / 65;
deepseek 6 / 4 / 99; **grok 1 / 1 / 275**. Grok's median long-form answer is a
single unbroken block, 60% of the time on long form and 100% on short. Qwen
paragraphs like a redditor. Same prompt, three models, a ~0.30 AUC spread against
a per-draw sampling SD of 0.040.

So this is not a shortcut-feature problem. It is **one prompt cannot calibrate
several models**, resurfacing in a dimension nobody thought to match. The fix is
the rate-matching principle applied one step further: draw a target paragraph
count per `(id, model)` from the specific human answer, the way word count
already is. Whether that works is an open experiment — some instructions take
(first person moved 0% → 80% → 23.3%) and some do not (em dashes survived
explicit prohibition twice and needed mechanical substitution).

**The published detector numbers survive it.** Split each class at its own median
newline count and Sonnet's and Haiku's conditional accuracy is flat in every
cell. Neither is riding the artifact. This is a benchmark validity problem, not a
retraction.

### 3. A 1.17B model on one GPU reaches AUC 0.771

`LiquidAI/LFM2.5-1.2B-Instruct`, single forward pass, **zero generated tokens**,
18,099 prefill tokens/sec on one Colab L4. Scored 0.771 [0.689, 0.846] on a
confirmation set built and frozen before any tuning. Full method and caveats in
`study/small/RESULTS.md` — including that `wc -l` scores 0.803 on the same items.

The transferable result is about prompting, not the score. **Seven
classification prompts all came out inverted**, including one listing only human
cues and one explicitly warning against the trap. **Ten descriptive yes/no axes
all came out correct.** At 1.2B the prompt controls which feature the model
attends to, but not which label it assigns that feature to — so ask for a
description and do the mapping in code.

### 4. Nine prompt versions bought nothing after the sixth

Single-document AUC against the fixed `dev` human set:

| version | pooled AUC | `Edit:` rate |
|---|---|---|
| v6 | 0.718 | 4% |
| v7 | 0.703 | 4% |
| v9 | 0.622 | 26% |
| v8_invert | 0.461 | 100% |
| *(human)* | | *11%* |

v7 buys −0.015 over v6 with the injected artifacts held literally constant. The
adversarial loop was at the floor by v6, and the nine versions after it were
optimising a 2AFC number that a real detector never sees.

`v8_invert` drives detection below chance, but by planting `Edit:` in **100%** of
documents against a human 11% — a giveaway feature as blatant as the ones filtered
out in week one, and there is no marker-free subset at any n. `v9` gets a third
of that evasion at a 26% marker rate, by register rather than by stamp.

---

## Pipeline

```
scripts/01_fetch_eli5.py       stream 656MB dump -> r/eli5 rows only
scripts/02_sample_pairs.py     filter + sample -> questions_1000.jsonl
scripts/02b_sample_longform.py the 250-799 word companion set
scripts/03_generate.py         OpenRouter fan-out -> ai_answers.jsonl
scripts/04_build_dataset.py    merge + split + QA -> data/final/
scripts/08_build_eval_sets.py  burned / dev / heldout / pool partition
```

Stage 3 needs `OPENROUTER_API_KEY` (env or a `.env` at the root) and is
resumable — rerunning skips `(id, model)` pairs already written. It honours
`GEN_IN`, `GEN_OUT`, `GEN_LIMIT` and `PROMPT_VERSION`.

**Benchmarking** (`12`/`14`) builds single-document batches and scores
predictions. `14_score_single.py` asserts the expected item count, refuses to
parse-and-skip, and prints distinct-document counts next to item counts —
because three reported figures in this project turned out to be computed on
fewer items than they claimed.

**The local detector** (`15_*`) rebuilds a benchmark's documents verified
byte-for-byte against the judged batch files, scores them with a small model,
and emits predictions in the same format `14_score_single.py` reads.

---

## Known limits

- **The production dataset does not exist.** See Status.
- Class balance would be 1:3 human:AI. The attribution task is balanced 1:1:1.
- One human answer per question, the top-scored one. High-scoring ELI5 answers
  skew toward a confident-explainer voice, so the human class is narrower than
  "human writing" — and long-form answers come disproportionately from people
  writing inside their own profession. **Length does not make a human look more
  human; it makes them look more professional, and professional reads as
  synthetic.** That is where the false positives come from.
- Three models from 2026. Nothing here should be assumed to generalise to models
  it never saw, or to unconstrained model output — every AI document in this repo
  was written under a register constraint.
- `study/single/short` and `study/single/long` carry a duplicated-human-arm
  defect: 58 and 39 distinct human documents behind 75 items each. Their AI arms
  are clean; their false-positive intervals are not quotable. `long2` and `long3`
  are the clean sets.
- **Every per-generator figure in this repo is a 25-to-100 document draw**, with a
  sampling SD around 0.04-0.06. Between-model differences of ~0.30 are solid;
  within-model differences across sets are not, and were twice mistaken for
  instability. `study/longform/answers_prededup.jsonl` holds 65 `(id, model)`
  pairs generated twice, which is a free within-model variance estimate; it has
  been used on the paragraph feature and no other.

## The method notes

`NARRATIVE.md` ends with two dozen of them. The four that would have saved the
most time:

1. **Measure the task you actually care about, first.** Nine prompt versions
   optimised a 2AFC score. The realistic task — one document, no reference — was
   measured last and showed the work had been unnecessary since v6.
2. **A near-perfect score in either direction means a shortcut, not skill.** 1%
   and 100% on the same file, from judges reading the same cue with opposite
   priors, is a separability result masquerading as a detection result.
3. **Quote the trivial baseline.** Every detector number in this repo went months
   without one, and the baseline was `wc -l`.
4. **Get a second reader with a different objective.** Not a more careful one.
   The paragraph artifact survived two audits by the person who wrote the
   sections it invalidates, and fell in about an hour to somebody who had come
   for something else. Care was not the variable.
