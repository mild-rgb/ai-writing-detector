# phase 3 / exp01 — findings

Results as they land. Arm (a) only: the basic human-vs-AI detector, all seven
generators seen in training. LOGO and `raw` were held by the user pending a
working model.

Everything below is recomputed from the stored per-document probabilities;
`aggregate.py` re-derived all 10 records exactly, and `contract.py` passes
10/10.

---

## The run

```
binary|norm   ModernBERT-large   A100-SXM4-80GB, driver 580.82.07
torch 2.11.0+cu128 (CUDA 12.8, cuDNN 91900)   transformers 5.15.1   py 3.13.15
5 seeds x {cls, mean} pooling = 10 runs, 2 epochs, bf16, sdpa, grad ckpt ON
checkpoint selected on dev LOSS      docs 3de9c320200087cb   splits 3bbc53497cfdbbc5
one environment (aaada103037d78a7) for all 10 runs      ~7.6 min/run
```

## Headline: the test set is saturated

**AUC 1.000000 ± 0.000000**, both poolings, all five seeds, and **every one of the
seven generators individually**. Not rounded — the separation margin is positive,
meaning every AI document scores above every human document.

| generator | AUC (cls) | recall (cls) | AUC (mean) | recall (mean) |
|---|---|---|---|---|
| deepseek-v4-pro | 1.0000 ±0.0000 | 1.000 ±0.000 | 1.0000 ±0.0000 | 1.000 ±0.000 |
| gemini-3.7-flash | 1.0000 ±0.0000 | 1.000 ±0.000 | 1.0000 ±0.0000 | 1.000 ±0.000 |
| nemotron-3.5-lightning | 1.0000 ±0.0000 | 1.000 ±0.000 | 1.0000 ±0.0000 | 1.000 ±0.000 |
| gpt-5.6-luna-pro | 1.0000 ±0.0000 | 1.000 ±0.000 | 1.0000 ±0.0000 | 1.000 ±0.000 |
| qwen3.8-max | 1.0000 ±0.0000 | **0.990 ±0.013** | 1.0000 ±0.0000 | 1.000 ±0.000 |
| grok-4.6 | 1.0000 ±0.0000 | 1.000 ±0.000 | 1.0000 ±0.0000 | 1.000 ±0.000 |
| glm-5.3 | 1.0000 ±0.0000 | 1.000 ±0.000 | 1.0000 ±0.0000 | 1.000 ±0.000 |

**The per-generator spread is 0.0000.** The 73-point judge spread (6.7% GPT Luna
→ 80% Nemotron) does not survive contact with a fine-tuned detector. Those are
measurements of two different things: what a single-pass LLM judge notices, and
what a supervised classifier can separate. The per-generator reporting rule was
still right to impose — it is how we know the spread is gone rather than assuming it.

False positive rate on human documents: **0.000, every run, both poolings.**

## Pooling: `mean` wins, and only a finer metric could say so

AUC cannot separate them; both are exactly 1.0. On anything finer, `mean` is
clearly better:

| | cls | mean |
|---|---|---|
| test balanced accuracy | 0.999310 ±0.000944 | **1.000000 ±0.000000** |
| misclassified at 0.5 (of 580) | 0.4 avg — `[0,1,1,0,0]` | **0 in every seed** |
| separation margin | 0.505 ±0.497 | **0.976 ±0.033** |
| dev loss (the selector) | **0.000263** | 0.000821 |
| train seconds | 451 | 459 |

Separation margin = `min(p_ai on AI) − max(p_ai on human)`. `mean` leaves a
0.976 gap with almost no seed variance; `cls` averages 0.505 with an SD of
0.497, i.e. some seeds sit near zero margin. Same ranking, very different
robustness — visible only because per-document probabilities are stored.

Worth noting against ourselves: **dev loss preferred `cls`, test behaviour
preferred `mean`.** The selector and the outcome disagree. With dev also
saturated this is not a reason to distrust dev-loss selection over dev-AUC
selection — it is a reason to say the dev set can no longer discriminate either.

**Use `mean`.** Our documents are long, and mean pooling reads all of them.

## Floors: the model is not reading shortcuts, but it barely needs to be a model

| control | AUC | balanced acc |
|---|---|---|
| newline count alone | **0.500** | — (by construction under `norm`) |
| ~40 mechanical markers | 0.687 | 0.631 |
| bag-of-words, words + punctuation | 0.9997 | 0.9897 |
| bag-of-words, **punctuation deleted** | 0.9993 | 0.9845 |
| ModernBERT-large | **1.0000** | 1.0000 |

The newline baseline is 0.500 exactly in all ten records, so nothing here is
whitespace. The marker floor at 0.687 says it is not the Reddit/LLM regex set.

But a logistic regression on binary word-presence — no ordering, no syntax, no
embeddings, no GPU, seconds to fit — reaches **AUC 0.9993 with every punctuation
mark deleted**. The transformer's contribution over bag-of-words is the last
0.0007 of AUC and 580/580 instead of ~574/580.

That is the honest framing: **the classes are nearly separable in plain word
space**, and the expensive model is confirming that, not transcending it.

### What the separation is not

- **Not one leaked token.** Best single feature on test reaches balanced accuracy
  0.814 — `(`/`)`, present in 76.6% of human documents against 13.8% of AI ones.
- **Not punctuation.** Delete it all; bag-of-words still gets 0.9993.
- **Not the generator's blacklist.** The floor prompt forbids *delve, crucial,
  vital, robust, leverage, tapestry…*, so "avoided the banned list ⇒ AI" is the
  obvious worry. It is false: banned terms appear in 8.6% of human and 1.7% of AI
  documents, and that rule alone scores balanced accuracy **0.534**.

## The one document the detector ever gets wrong

Across all 10 runs, exactly one test document is ever misclassified:
`eli5c-0731-a`, a qwen answer about pear-shaped atomic nuclei, wrong in 2 of 10
runs (both `cls`).

It earns it. It opens by correcting the asker's premise — *"So this is about the
nucleus of an atom, not a cell"* — which is the same move its human sibling
makes, and it asks and answers its own rhetorical question mid-answer. Its human
sibling, at exactly the same 347 words, contains *"without delving into nuclear
structure"* — a word the AI was forbidden to use.

## What this does and does not license

It licenses: a working human-vs-AI detector on this corpus, at ceiling, stable
across seeds, with no per-generator weak spot and no false positives.

It does not license any claim about detecting AI text in general. Every AI
document here came from one prompt, one length calibration and one
post-processing pass; every human document came from one subreddit. A ceiling
result on a corpus that bag-of-words nearly solves is a statement about the
corpus at least as much as about the detector.

The two open questions are now the only interesting ones, and the second matters
more:

1. **Unseen generator** (LOGO) — held. A single 1-epoch pilot on held-out gemini
   reached AUC 0.99998, so this will likely also be at ceiling.
2. **Out of domain** — the planned wild-AI-text slice. Nothing measured here
   constrains it, and with in-domain saturated it is the only remaining test that
   can fail.

---

## Out-of-domain probe — speculative fiction, 364 chunks

`story.syn.gl`, one continuous finished narrative, chunked to training document
size (median 366 words, all 250–800). Scored with the chosen mean-pooling
checkpoint `ffb2472159d98e29` (seed 1, picked on dev loss).

Pipeline validity first, because otherwise the number measures preprocessing:
same `norm_ws` as `prep.py`, tokenizer loaded from the model directory,
`max_length` 2048, **0 of 364 truncated**, pooling asserted `mean` against the
shipped training record. Input sha256 `92432f67a63a879c`, verified identical on
both machines. `label: "unknown"` throughout, used for nothing.

**Provenance, confirmed by the user after the probe was run: `story.syn.gl` is
AI-generated.** The scoring was done blind — the input carried `label: "unknown"`
and no label was used for anything — and the numbers below are signed only now.
Every AI-called document is therefore a true positive and every human-called one
is a miss.

### Signed result: 83.8% recall, incoherently held

### It is confident, and it contradicts itself

```
p_ai   mean 0.8297   median 1.0000   sd 0.3431   min 0.000000   max 1.000000
called AI at 0.5:  305/364 = 83.8%   <- RECALL on out-of-domain AI text
missed:            59/364 = 16.2%    <- roughly one document in six
of which decisively human (p<=0.01): 28/364 = 7.7%

  p >= 0.99   decisively AI        267   73.4%  ############################################
  0.90-0.99                         17    4.7%  ##
  0.50-0.90                         21    5.8%  ###
  0.10-0.50                         16    4.4%  ##
  0.01-0.10                         15    4.1%  ##
  p <= 0.01   decisively human      28    7.7%  ####

  saturated at the extremes:  295/364 = 81.0%
  ambiguous band 0.1-0.9:      37/364 = 10.2%
```

This is **one story, by one author, start to finish.** It cannot be 73%
decisively machine and 8% decisively human. The detector is not hedging
off-distribution — it is answering at full confidence and giving incompatible
answers about the same text.

The two extremes are not stylistically distinguishable. Most human-like
(`story-0110`, p_ai = 0.000000) and most AI-like (`story-0361`, p_ai = 1.000000)
are both dialogue between the same recurring characters in the same voice.
Dialogue density does not explain it either: chunks with ≥8 quotation marks
average p_ai 0.816, chunks with none average 0.919 — both high.

### Adjacent chunks of the same story disagree

The chunks are in narrative order, so neighbours are consecutive passages of one
text:

```
                       flips its AI/human call between neighbours
  transformer                82/363  = 22.6%   (37 of those swing by >0.98)
  bag-of-words              186/363  = 51.2%
```

37 times the model moves from near-certain to near-certain-opposite between one
passage and the next.

### The floors disagree with the transformer, and that is the finding

| | transformer | bag-of-words |
|---|---|---|
| mean p_ai | 0.8297 | 0.5569 |
| called AI at 0.5 | 83.8% | 56.6% |
| **saturated at extremes** | **81.0%** | **9.1%** |
| adjacent-neighbour flips | 22.6% | 51.2% |

They agree on the call for only **63.5%** of chunks; Pearson r between their
scores is **0.366**.

In domain these two models were nearly the same function — AUC 0.9993 vs
1.0000, agreeing on essentially every document. Off distribution they agree
barely more than a coin flip would, and only the transformer is confident.

**In-domain agreement was not evidence of a shared transferable signal.** The
transformer's extra 0.0007 of in-domain AUC turns into extreme, unstable
confidence on text it was not trained for. The trivial model, being trivial, at
least stays near 0.5 and looks as uncertain as it is.

### Is the floor uncertain, or just balanced between the extremes?

Bag-of-words calls 56.6% AI, near chance. That could mean two very different
things — honest uncertainty sitting near 0.5, or the same saturated confidence
as the transformer merely split evenly between the poles. The histogram settles
it, and the answer is the first.

| band | bag-of-words | transformer |
|---|---|---|
| p ≥ 0.99  decisively AI | 28  (7.7%) | **267  (73.4%)** |
| 0.90–0.99 | 58  (15.9%) | 17  (4.7%) |
| 0.50–0.90 | 120  (33.0%) | 21  (5.8%) |
| 0.10–0.50 | 111  (30.5%) | 16  (4.4%) |
| 0.01–0.10 | 42  (11.5%) | 15  (4.1%) |
| p ≤ 0.01  decisively human | 5  (1.4%) | 28  (7.7%) |
| **ambiguous band 0.1–0.9** | **231  (63.5%)** | 37  (10.2%) |
| **saturated at extremes** | 33  (9.1%) | **295  (81.0%)** |

```
bag-of-words   mean 0.5569  median 0.5917  sd 0.3382  min 0.0019  max 0.9992
transformer    mean 0.8297  median 1.0000  sd 0.3431  min 0.0000  max 1.0000
```

**Bag-of-words is genuinely uncertain.** Nearly two thirds of its scores sit in
the ambiguous band, its median is 0.59, and its most extreme score in either
direction never reaches 0.999 or drops below 0.0018. It is a weak detector
behaving like one.

The transformer's median is exactly 1.0000, and **164 of 364 chunks are tied at
p_ai = 1.000000** — floating-point saturation, not merely high confidence.

They do not even agree on which passages are most machine-like. BoW's most-AI
chunk is `story-0246`, the transformer's is `story-0361`; BoW's most-human is
`story-0166`, the transformer's is `story-0110`. Neither pick matches. (They do
partially overlap at one end: the transformer also scores BoW's most-AI chunk at
1.000000, and gives BoW's most-human chunk 0.0105.)

### CORRECTION — the human-fiction control overturns the conclusion below

The section that follows was written before the false-positive side was
measured, and its conclusion was wrong. It is kept as written, with this
correction above it, because the point of a findings log is that predictions and
readings cannot be edited to match what came later.

**What was measured second:** 364 chunks of `r/WritingPrompts` fiction (Fan et
al. 2018, pre-LLM, human by construction), same size band, same normalisation,
same checkpoint.

```
                    recall (syn AI)   FPR (human)   balanced acc   OOD AUC
  transformer            0.8379         0.1126         0.8626       0.9447
  bag-of-words           0.5659         0.0769         0.7445       0.8901
```

**83.8% was not an off-distribution lean toward "AI".** The same model calls
genuinely human fiction AI only 11.3% of the time, and puts 83.0% of it below
p ≤ 0.01. It is separating, not drifting.

And the transformer is **clearly better than the floor off-domain** — AUC 0.9447
vs 0.8901, balanced accuracy 0.8626 vs 0.7445. The claim below that its
in-domain edge over bag-of-words was worthless off-domain is **false**. In domain
the two were the same function; out of domain the transformer holds a real and
substantial lead. The in-domain tie understated it.

**What survives:** the calibration failure. The model is confidently wrong on a
minority of both classes — 6.3% of human fiction sits at p ≥ 0.99, 7.7% of known
AI at p ≤ 0.01 — and 89.3% of its human-fiction scores are pinned to the
extremes. Adjacent passages of one text still flip (18.2% here, 22.6% on syn).
It transfers with real signal and unreliable confidence, which is a different and
much less damning fault than not transferring at all.

**Revised conclusion:** in domain, balanced accuracy 1.000; out of domain, 0.863.
A genuine detector that degrades substantially and stops being trustworthy at the
document level, not a corpus artifact.

| band | human fiction: transformer | bag-of-words |
|---|---|---|
| p ≥ 0.99  decisively AI | 23 (6.3%) | 1 (0.3%) |
| 0.90–0.99 | 9 (2.5%) | 4 (1.1%) |
| 0.50–0.90 | 9 (2.5%) | 23 (6.3%) |
| 0.10–0.50 | 6 (1.6%) | 60 (16.5%) |
| 0.01–0.10 | 15 (4.1%) | 106 (29.1%) |
| p ≤ 0.01  decisively human | **302 (83.0%)** | 170 (46.7%) |
| saturated at extremes | 89.3% | 47.0% |
| neighbour flips | 18.2% | 15.4% |

### The signed conclusion (SUPERSEDED — see the correction above)

On out-of-domain AI fiction the ELI5-trained detector **recalls 83.8%, but holds
that answer incoherently** — 81% of its scores pinned to the extremes, 22.6% of
adjacent passages of one AI-authored text getting opposite calls, 37 of those
swinging by more than 0.98, and 7.7% of known-AI text called decisively human.

The bag-of-words floor it was indistinguishable from in domain (0.9993 vs
1.0000) recalls 56.6% off domain, near chance — and is honest about it.

**This is not a general detector.** It is a very good ELI5-corpus detector whose
confidence does not survive the domain boundary, and whose in-domain agreement
with a trivial model was not evidence of any transferable signal.
