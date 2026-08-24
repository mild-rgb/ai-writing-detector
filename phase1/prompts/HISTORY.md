# Prompt iteration log

Adversarial loop: generate a batch of answers, have a judge model try to tell
them from the real human comments, and rewrite the prompt to close whatever the
judge is using. Each batch uses **fresh questions** so the prompt is not tuned
to specific items.

Judge accuracy is 2AFC — the judge sees one human and one AI answer to the same
question and picks the AI. **Chance is 50%.**

The score to watch is **distance from 50%, in either direction**. Below chance
is not success -- it means the AI answers read *more* human than the humans, and
a trained detector would simply learn the rule backwards and score just as well.
v3 at 80% and v4's grok judge at 20% are equally detectable. Saturation means
landing *on* chance, not below it.

| version | change | judge | trials | judge accuracy |
|---|---|---|---|---|
| v1 | baseline: forbids all markdown | - | - | not judged |
| v2 | removed the "plain text only" line | - | - | not judged |
| v3 | added ASCII punctuation + anti-AI-register rules | haiku | 30 | **80.0%** (grok 70, qwen 80, deepseek 90) |
| v4 | first person + narrow scope + no scaffolding/flourish/false-balance + leave rough edges | haiku | 30 | **36.7%** (grok 20, deepseek 40, qwen 50) — 13.3 pts, INVERTED |
| v5 | v4 rules, but personal voice moved from an unconditional instruction to per-generation draws at human base rates | haiku | 30 | **33.3%** (grok 20, deepseek 30, qwen 50) - 16.7 pts, INVERTED |
| v6 | dropped the "leave a stray lowercase start" line; banned extended metaphor and parenthetical glosses; em dash normalised mechanically | haiku | 90 | **58.9%** (deepseek 50.0, grok 56.7, qwen 70.0) - 8.9 pts |
| v7 | anti-institutional register; "answer only the easy half"; uneven sentence length; go easy on "just"/"basically" | haiku | 90 | **40.0%** so far (deepseek 53.3, grok 26.7, qwen pending) - INVERTED |

## The central finding: one prompt cannot calibrate three models

The same instruction moves each model a different distance and sometimes in
opposite directions. "Go easy on 'just'" (human rate 26.9%):

| model | v6 | v7 |
|---|---|---|
| grok-4.6 | 66.7% | **16.7%** (overshot past human) |
| deepseek-v4-pro | 46.7% | 43.3% (barely moved) |
| qwen3.8-max | 46.7% | **50.0%** (went up) |

Judge accuracy followed:

| judge | v6 | v7 |
|---|---|---|
| grok-4.6 | 56.7% (near ideal) | **26.7%** (badly inverted) |
| deepseek-v4-pro | 50.0% (at chance) | 53.3% (at chance) |
| qwen3.8-max | 70.0% (detectable) | pending |

v7's changes were written to fix **qwen** and broke **grok**. A shared prompt has
a floor set by whichever model is currently most miscalibrated, and closing that
model's gap opens another's. Both v7 judge readings had a perfect 15/15 A/B
split, so neither is position-bias contamination.

**Implication:** either fork per-model prompt variants (same rules, per-model
constants, tuned against each model's own judge score), or accept the floor and
document it. For a detector dataset the second is defensible -- the goal is
realistically hard AI text, not text tuned until one small judge gives up.

## Batch size raised at iteration 4

n=10 per judge gives a standard error of ~16 points, and v4 (13.3 pts off
chance) vs v5 (16.7) is well inside that. The iter03 deepseek judge also guessed
"B" on 9 of 10 trials -- an 80% position skew that moves the score by 30 points
on its own, independent of the text. Batches are now **30 questions**, so each
judge sees 30 distinct-question trials (SE ~9 pts) and the aggregate is n=90
(SE ~5 pts). The scorer reports per-judge A/B skew and flags anything over 60%.

## v5 signal parity (what the base-rate draws achieved)

Total absolute gap across measured signals: v4 160.8 -> v5 51.0.

| signal | human | v4 | v5 |
|---|---|---|---|
| first person "I" | 23.3% | 80.0% | **23.3%** |
| hedge | 3.2% | 43.3% | 6.7% |
| personal experience | 0.9% | 30.0% | 3.3% |
| snark | 2.7% | 13.3% | 3.3% |
| exclamation | 4.1% | 0.0% | 3.3% |
| lowercase sentence start | 6.6% | 6.7% | **33.3%** (fixed in v6) |
| em dash | 0.4% | 10.0% | **10.0%** (normalised in v6) |

## Why v5 exists: overshoot

v4 instructed personal voice unconditionally and inverted the tell rather than
removing it.

| signal | human | v3 | v4 |
|---|---|---|---|
| first person "I" | 23.3% | 0.0% | **80.0%** |
| hedge | 3.2% | 0.0% | **43.3%** |
| personal experience | 0.9% | 0.0% | **23.3%** |
| snark | 2.7% | 0.0% | 13.3% |
| lowercase sentence start | 6.6% | 0.0% | 6.7% |

Absence and excess are both detectable; only the *rate* is neutral. v5 draws
each voice feature per generation at the measured human rate (first person
22.7%, hedge 3.1%, experience 0.8%, snark 2.6%, exclamation 3.7% as drawn),
seeded on (id, model) so runs reproduce.

## Escalation ladder

Judge is escalated when the current one can no longer beat chance (~60% at
n=30): **haiku -> sonnet -> opus**.

## What v3 leaked (measured over 1000 human answers vs v3 output)

| signal | human | AI v3 |
|---|---|---|
| first person "I" | 23.3% | **0.0%** |
| double space after period | 23.8% | **0.0%** |
| lowercase sentence start | 6.6% | 0.0% |
| exclamation mark | 4.1% | 0.0% |
| hedge ("I think", "I'm no expert") | 3.1% | 0.0% |
| snark / mild profanity | 2.7% | 0.0% |

All three judges independently converged on the same cues: personal voice and
hedging read human; encyclopedic completeness, enumerated scaffolding, stacked
jargon, false-balance clauses and punchline endings read AI.

## Harness fixes

**iter01, first attempt — discarded (100% judge accuracy, invalid).** All three
models' trials for a question sat in one file, so the *human* answer repeated
verbatim across three trials while the AI answers differed. The judge's own
top-ranked cue was "reused identical text appearing in multiple trials indicates
genuine human comments" — it won on bookkeeping, not prose. Batches are now
split one file per model, each with every question appearing exactly once, and
judged by separate agents that do not share context. Kept at
`judge/iter01_INVALID/` as a record.


---

# Final Haiku results (pooled, iterations 1-7)

Each (model, prompt) measured on 30 fresh questions per iteration. **Pooled
figures are the ones to trust**; single 30-trial batches swing far too much to
select on.

| model | prompt | batches | pooled | n | dist | verdict |
|---|---|---|---|---|---|---|
| qwen3.8-max | v7 | 53.3, 50.0, 50.0 | **51.1%** | 90 | 1.1 | **SATURATED** |
| deepseek-v4-pro | v6 | 50.0, 63.3 | 56.7% | 60 | 6.7 | marginal |
| deepseek-v4-pro | v7 | 53.3, 83.3 | 68.3% | 60 | 18.3 | detectable |
| grok-4.6 | v6 | 56.7, 36.7, 20.0 | 37.8% | 90 | 12.2 | inverted, unstable |
| grok-4.6 | v7 | 26.7 | 26.7% | 30 | 23.3 | inverted |

## The measurement is dominated by question-set effects

Repeated measurements of the *same* model on the *same* prompt, differing only
in which 30 questions were used:

| model / prompt | batch accuracies | observed SD | binomial SD | overdispersion |
|---|---|---|---|---|
| qwen3.8-max / v7 | 53.3, 50.0, 50.0 | 1.9 | 9.1 | **0.2x** |
| deepseek-v4-pro / v6 | 50.0, 63.3 | 9.4 | 9.0 | 1.0x |
| grok-4.6 / v6 | 56.7, 36.7, 20.0 | 18.4 | 8.9 | **2.1x** |
| deepseek-v4-pro / v7 | 53.3, 83.3 | 21.2 | 8.5 | **2.5x** |

Three of four cells vary 2-2.5x more than binomial sampling allows, so the
between-batch differences are **question-set effects, not sampling noise**.
Comparing prompt version A on questions 31-60 against version B on questions
61-90 confounds the prompt change with the question change, and the confound is
larger than most of the effects being measured.

**Fix for any further iteration: paired design.** Hold one fixed comparison set
of questions and run every candidate prompt on it, so version differences are
measured within-question. Keep a separate untouched set for final validation
only. Fresh questions per iteration were chosen to avoid overfitting to specific
items; that was the right worry but the wrong remedy, and it cost more in
variance than it saved in bias.

**Qwen/v7 is the one unambiguous success**: three independent batches at 53.3,
50.0, 50.0, with *less* spread than binomial, pooling to 51.1% over 90 trials.
That is a model held stably at chance, not a lucky reading.
