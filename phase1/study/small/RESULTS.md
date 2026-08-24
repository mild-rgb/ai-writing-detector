# A sub-1.7B detector

Can a model small enough to run on one consumer GPU do the single-document task
this project settled on — one comment, no reference text, no sibling set?

**Model:** `LiquidAI/LFM2.5-1.2B-Instruct`, 1.17B parameters (measured), 16
layers of which 10 are gated convolutions and 6 are grouped-query attention.
Picked as the most capable thing under 1.7B that is also the fastest: the hybrid
backbone keeps a KV cache on 6 of 16 layers, and the vendor reports 2.3-2.8x
prefill and 1.7-2.2x decode against Qwen3-1.7B, which it also beats on IFEval
(86.23) and MMLU-Pro (44.35). **Those benchmark figures are vendor-reported and
were not verified here.** The two numbers measured in this repo are the
parameter count and the throughput.

**Hardware:** one Colab L4 (23GB, sm89), bf16, PyTorch SDPA. No vLLM: the
scoring pass generates zero tokens, so continuous batching and paged attention —
vLLM's actual advantages — have nothing to do.

---

## 1. The headline

`long3` is a confirmation set of 150 items over 75 questions, generated with v6
and **built before any prompt was tuned**, then scored exactly once. It is
disjoint from `long2` by question id (asserted, not assumed).

| | AUC | 95% CI | detection | FPR | balanced |
|---|---|---|---|---|---|
| **long3** (confirmation, frozen threshold) | **0.771** | [0.689, 0.846] | 66.7% [55.4, 76.3] | 24.0% [15.8, 34.8] | 71.3% |
| long2 (dev, threshold tuned here) | 0.725 | [0.646, 0.806] | 54.7% | 17.3% | 68.7% |
| short (never tuned on, distinct docs) | 0.744 | [0.660, 0.820] | — | — | — |

The pre-registered line was **AUC 0.65**, chosen because it is roughly the
smallest effect n=150 can resolve (power 0.89; at 0.60 it would have been 0.56,
i.e. a null there would have meant "not measurable here", not "not there"). The
confirmation set clears it, and clears it *upward* from dev rather than
regressing, which is the direction that does not happen when a result is fitted.

Scored with `scripts/14_score_single.py` unmodified, so these numbers are
computed identically to Haiku's and Sonnet's.

## 2. The number that has to be quoted next to it

**`wc -l` scores AUC 0.802 on long2 and 0.803 on long3.**

Fewer newlines implies AI. Counting line breaks — no model, no prompt, no GPU —
beats the 1.2B detector on both long-form sets. Found by the session maintaining
`NARRATIVE.md` while auditing these results, and independently replicated here.

A detector result is only meaningful against the trivial baseline on the same
data. Reporting 0.771 without 0.803 would be the shape of claim this repo has
recorded going wrong five separate times.

Two things stop this from being a straightforward loss:

* **The detector is not using it.** Correlation of margin with newline count,
  within the ai class: −0.04 on long2, +0.04 on long3, +0.15 on short. Removing
  the worst-offending generator costs the baseline 0.075-0.096 AUC and costs
  this detector 0.005-0.014.
* **They measure different quantities.** 0.80 is a formatting-convention gap;
  0.77 is prose. Which matters depends on whether the question is "was a model
  involved" or "can a small model read machine prose". For the first, use the
  baseline — it is free. Only the second is what this was built to answer.

The artifact is not planted: `prompts/v6.txt` says nothing about paragraphs. It
is an omission from the section-3 rate-matching program, which matched word
count, markdown rate, artifact rate and persona rate, and never matched
paragraph structure.

## 3. The result worth keeping: ask for a description, not a label

Seventeen prompts. The split is clean and has no exceptions in either direction.

**Seven classification prompts, all inverted** (AUC below chance):

| prompt | AUC |
|---|---|
| s2_polish — machine-cue checklist | 0.292 |
| s5_humancues — human-cue checklist only | 0.330 |
| s1_naive — no cues at all | 0.382 |
| s4_era — register framing | 0.384 |
| s3_twosided — both checklists, decided by comparison | 0.401 |
| s6_expert — explicitly warns "these humans are professionals" | 0.430 |
| s7_specificity — first-hand knowledge framing | 0.447 |

The two written specifically to *fix* the inversion (s6, s7) are the two weakest.
Pushing harder toward the correct mapping moved the result toward chance, not
toward correctness.

**Ten descriptive yes/no axes, all correct** (label mapping done in code):

| axis | question | AUC |
|---|---|---|
| a4_generic | would this be just as good an answer to a slightly different question? | **0.723** |
| a1_polished | is it polished, comprehensive, tidily closed? | 0.712 |
| a10_textbook | could this be printed in a textbook unedited? | 0.690 |
| a8_firsthand | does the writer say anything they could only know from doing this? | 0.682 |
| a5_concludes | does it wrap up rather than stop? | 0.696 |
| a7_evenpace | is it evenly paced? | 0.635 |
| a3_effort | typed once and posted without rereading? | 0.569 |
| a9_opinion | any opinion, complaint or joke? | 0.540 |
| a2_person | a specific identifiable person behind it? | 0.534 |
| a6_necessary | is every sentence doing work? | 0.500 |

`a1_polished` reads polish at 0.712 in the right direction. `s2_polish` asks
about the same feature as a classification and gets 0.292 in the wrong one.

**The generalisation: at 1.2B the prompt controls which feature the model
attends to, but not which label it assigns that feature to.** Ask a descriptive
question and do the mapping outside the model.

The audit path shows why. Asked to name its strongest cue, the model names real
ones correctly — "comprehensive, structured, textbook-like", "parallel and
contrastive phrasing" — and finds *more* of them in the human documents (+5.5
mean) than the machine ones (+3.0), concluding "machine" on all eight sampled,
humans included. It reads the text right and maps it wrong.

**Tested and rejected as the cause:** that the inverted prompts were mechanically
reading paragraph whitespace. Within-class correlation of margin with newline
count is ≤0.20 for every inverted prompt (s2_polish: −0.09 ai, +0.04 human).

## 4. A pre-registered failure

A logit lens on the frozen axis — every layer's residual read through the final
norm and the tied unembedding:

| layer | kind | AUC |
|---|---|---|
| 0 | embeddings | 0.500 |
| 3 | attention | 0.318 |
| 4 | conv | 0.638 |
| **6** | **attention** | **0.747** |
| 10-11 | conv / attention | 0.391 / 0.389 |
| 13 | attention | 0.746 |
| 16 | output | 0.725 |

Layer 0 is exactly 0.500, which is the sanity check: the position read is the
last prompt token, identical across all items. The decision is essentially made
by layer 6 of 16; the attention blocks carry it; there is a transient sign
inversion at layers 10-11 that layer 13 undoes; the final layers slightly
*degrade* it.

That suggested an early exit: 6 of 16 blocks, **2.63x throughput and +0.022 AUC**
on dev. It was frozen as variant B with the caveat written into the config —
selected on dev, off a non-monotone curve (exit@5 0.568, exit@6 0.747, exit@7
0.663), and therefore likely to be a spike.

**It was.** And the shortcut audit says exactly why:

| set | r(margin, newlines) within ai — **A** full 16 | **B** exit@6 | AUC A | AUC B | AUC drop-grok A | B |
|---|---|---|---|---|---|---|
| long2 | −0.04 | **−0.67** | 0.725 | 0.747 | 0.711 | 0.655 |
| long3 | +0.04 | **−0.65** | 0.771 | 0.791 | 0.766 | 0.696 |
| short | +0.15 | −0.25 | 0.749 | 0.698 | 0.737 | 0.660 |

Variant B is substantially a whitespace detector. It looks better than A on
long-form, where the artifact is strong; it loses to A on short form, where it is
weaker; and its drop-grok collapse tracks the `wc -l` baseline's almost exactly.

That last point is the most direct evidence and deserves its own line. **What
each detector loses when the worst-offending generator is removed:**

| set | A (full 16) | B (exit@6) | `wc -l` |
|---|---|---|---|
| long2 | **−0.014** | −0.092 | −0.074 |
| long3 | **−0.005** | −0.095 | −0.096 |
| short | **−0.012** | −0.037 | −0.041 |

B and the baseline die at the same rate, on every set. A does not move. A
shortcut that collapses in lockstep with the baseline it is imitating is about as
clean as this kind of evidence gets.

**The ten layers the early exit skips are what move the decision off the surface
feature.** The truncated model is faster and reads whitespace; the full model is
slower and reads prose. Variant A is the detector. Variant B is a cautionary
tale that would have been reported as a free 2.6x win by anyone who did not run
the audit.

### The pre-registration and the shortcut audit disagreed, and the audit won

Worth separating, because these are two instruments answering different
questions and this is the first claim in the project with both pointed at it.

The pre-registration called variant B unreliable — correctly — but for the wrong
reason: *a spike selected on dev off a non-monotone curve, therefore probably
noise*. It was not noise. It replicated at r = −0.67 and −0.65 on two independent
sets and behaved identically on a third. It was systematic, reproducible, and
reading the wrong feature.

Only the surface-feature correlation could tell the difference. A pre-registered
prediction establishes whether a result **survives**; a shortcut audit
establishes whether it is **the thing you think it is**. Variant B would have
survived on long3 — 0.791, beating variant A — and still been wrong. The two
checks are not redundant and neither substitutes for the other.

Put at its sharpest: this is the first result in the project that **passed every
check the method had and was caught only by a check the method did not have.**
Held-out confirmation, a pre-registered caveat, a frozen threshold and a rising
number from dev to confirmation — all satisfied, all passed, and the thing was
counting line breaks.

## 5. The operating point does not transfer across length

AUC is length-portable. The threshold is not. Margins fall about 2 nats as
documents shorten (long2 ai +4.66 / human +3.10; short ai +2.52 / human +0.99).

| | detection | FPR | balanced |
|---|---|---|---|
| short, long2-tuned threshold | 9.3% | 0.0% | 54.7% |
| short, re-tuned on short | 58.7% | 19.0% | 69.9% |

The transferred threshold calls almost everything human. **Quote AUC, or quote
an operating point with the length it was tuned at attached.**

## 6. Every generator prompt, one document at a time

The four `phase1` conditions answer the same 100 `dev` questions, so the human
side is one fixed set and the versions are paired. The artifact seed is
`(id, model)` and excludes prompt version, so v6 and v7 carry *identical*
injected artifacts — making that comparison a clean read on the prose changes
alone. As far as anyone here can tell, v7, v9 and v8_invert had never been
measured one document at a time.

| version | pooled AUC | deepseek | qwen | grok |
|---|---|---|---|---|
| v6 | 0.718 | 0.755 | 0.707 | 0.693 |
| v7 | 0.703 | 0.726 | 0.670 | 0.712 |
| v9 | 0.622 | 0.677 | 0.571 | 0.616 |
| v8_invert | **0.461** | 0.593 | 0.425 | 0.363 |

The pooled column is ~300 documents against 100 humans. **The three per-generator
columns are ~100 documents each and carry a sampling SD of about ±0.035; no
pairwise difference between generators within a row is resolved** (see §8). Read
the rows, not the columns.

**v7 buys nothing over v6** (−0.015), reproducing section 11's conclusion on a
different instrument: the adversarial loop was already at the floor by v6.

**v8_invert drives the detector below chance — but not by planting a tell it
reads.** Marker rates: `Edit:` 100% of v8_invert documents against 11% of
humans, "I might be wrong" 76%, relative/coworker 45%. There is no marker-free
subset at any n, so the leave-them-out test is unrunnable. What can be measured:
within v8_invert, documents carrying a given marker score within −0.36 to +0.30
of those without. The detector is not responding to the strings. The whole
distribution collapsed — v8_invert's mean genericness is +1.02 against the human
+1.15, i.e. *below* it.

v8_invert does not fool this detector. It deletes the property being measured.
That is the reciprocal of section 9 rather than a restatement: those judges
failed because the separation was labelled backwards; this fails because on its
axis the separation is gone. Neither is a detection result.

**v9 gets about a third of v8_invert's evasion (0.718 → 0.622) at a 26% marker
rate against 100%.** Section 10 predicted v9 would land closer to chance than to
1% and that the planted phrases were doing most of v8_invert's work. On the
realistic task that prediction resolves in its favour.

## 7. Throughput

The scoring pass generates **zero tokens**. The verdict is read from the logits
at the final prompt position, so the decode path is never entered.

| | prefill tok/s | items/s | 150 docs | decode tokens |
|---|---|---|---|---|
| naive (32k batches, full lm_head) | 11,089 | 18.6 | 8.07s | 0 |
| **tuned (2k batches, sliced head)** | **18,099** | **30.4** | **4.94s** | **0** |
| variant B, exit@6 *(rejected — see §4)* | 46,029 | 78.4 | 1.91s | 0 |

18,099 prefill tok/s is **69.6% of the L4's dense bf16 peak**. 1.62x over the
naive configuration, from two changes:

1. **Slice the hidden states before `lm_head`.** The stock forward projects every
   position through a 1024x65536 head, computing `B*L*65536` logits to use
   `B*65536` of them. Worth ~9% and it removes the memory ceiling that OOMs above
   a 32k-token batch.
2. **Smaller batches, not larger.** Counter-intuitive but decisive: with
   length-sorted batching a 2k-token budget pads 0.6% where a 32k budget pads
   25.3%, and the padding saved beats the larger kernels.

`vLLM` was considered and not used. Its advantages are continuous batching and
paged KV for the decode path, and this workload has no decode path.

### Two implementation notes

**Padding side is a correctness question on a conv-hybrid model, and the obvious
answer is wrong in an interesting way.** Left padding perturbs LFM2 because a
causal convolution slides a backward-looking window that pad tokens enter. But
right padding drifted too — and an item receiving *zero* padding drifted as well,
while eight identical rows in one batch agreed to 0.000000. It is bf16 GEMM
reduction order changing with batch shape, not padding. Quantified: AUC moves
0.719-0.725 across every batch configuration tried, so it cannot reach the
headline, but it can move a tuned threshold. `tok_budget` is therefore part of
the frozen config.

**The repo script reproduces the notebook bit-exactly** — max |Δmargin| 0.000000
over 150 items, zero verdict disagreements.

## 8. What is clean, and what is not

Audited, and clean:

* Not a length detector. Word count *alone* scores AUC 0.473 on long2 — there is
  no signal there to steal. Within-class margin/word correlations ≤0.30.
* Not a whitespace detector (§2), unlike variant B.
* Margins unimodal on both sides, deciles smooth and overlapping — no
  planted-feature bimodality.
* The rate-matched injections do not move it: persona 0 vs 1 gives +4.67 vs
  +4.61; artifact `None` +4.72 vs `Edit:` +2.63 (n=3, so nothing). As far as this
  can see the section-3 injections are individually undetectable, which is a
  modest positive result for how the corpus was built.

Not clean, and stated as limits:

* **The trivial baseline still wins on long form** (§2).
* **Corpus-specific.** The human class is top-voted long-form ELI5 — people
  writing inside their own profession — and the machine class is style-matched to
  plain 2019 Reddit register with markdown banned. Both halves are unusual. None
  of this should be assumed to transfer to unconstrained model output.
* **No per-generator claim in this document is resolved, and an earlier draft
  made two that were not.** Every per-generator cell on long2, long3 and short is
  25 documents, and a bootstrap puts the sampling SD at **±0.05 to ±0.06**. Even
  the best-powered cell available — phase1 v6, 100 documents per generator
  against 100 humans — leaves all three pairwise differences unresolved:

  | comparison (phase1 v6) | difference | 95% CI | |
  |---|---|---|---|
  | deepseek − qwen | +0.048 | [−0.028, +0.127] | not resolved |
  | deepseek − grok | +0.062 | [−0.020, +0.138] | not resolved |
  | qwen − grok | +0.013 | [−0.065, +0.095] | not resolved |

  What can be said: qwen is the lowest of the three in all four sets
  (0.707 / 0.691 / 0.650 / 0.676), and it is also the generator whose paragraphing
  is closest to human. Four-for-four in the same direction is suggestive, the sets
  are not independent, and no single comparison clears its own noise. That is a
  hypothesis worth a powered test, not a finding.

  An earlier draft of this file asserted "qwen is consistently the hardest
  generator" and "the grok result is unstable across sets" (0.754 on long2 against
  0.780 on long3 — a difference of 0.026 against an SD of 0.05-0.06, i.e. nothing
  at all). Both were variances read off a handful of 25-document draws. They are
  withdrawn here rather than quietly deleted, because this is the fourth instance
  of that error the surrounding project has recorded and the failure mode is
  evidently not one anyone learns once.

  **Which of those two noise sources it is has since been measured, on the
  whitespace feature at least.** `study/longform/answers_prededup.jsonl` holds 245
  rows over 180 unique `(id, model)` keys — **65 pairs generated twice**, same
  question, same model, same prompt, two independent samples, all 65 differing in
  text (deepseek 24, grok 21, qwen 20; verified here, and every pair is still
  represented in the deduplicated working copy, so nothing was lost). The session
  maintaining `NARRATIVE.md` scored both runs: regenerating moves a per-generator
  newline AUC by **0.010-0.014**; drawing a different 25 documents moves it by
  **0.040**. Each model's median paragraph count is identical across runs. So on
  that feature the wobble is the *sample*, not the generation, and a model's
  paragraph disposition is a stable property of the model.

  **That has not been done for the margins in this document, or for any axis
  other than paragraphs.** It is the obvious missing variance component under
  every per-generator figure in both documents, it costs seconds of GPU against
  those 65 pairs, and until someone runs it the per-generator columns here should
  be read as ±0.05 **of unknown composition** rather than assumed to be sampling.
* One prompt version's generation is incomplete: `long3` is 224/225 rows
  (`eli5lf-0099` qwen returned 14 words against a 284-word target — the section-2
  truncation mode, caught by the length guard). `phase1/v9` is 288 rows over 99
  questions, not 300/100.

## 9. Files

```
scripts/15_build_items.py         rebuild a benchmark's documents, verified
                                  byte-for-byte against the judged batch files
scripts/15_build_phase1_items.py  cross-prompt-version corpus, human class once
scripts/15_run_small.py           the scorer; zero generated tokens
scripts/15_margins_to_pred.py     margins + threshold -> 14_score_single.py format
study/small/frozen_config.json    frozen before long3 existed; see PROVENANCE_NOTE
study/small/peritem_*.jsonl       per-item margins, surface features, injections
study/small/lfm2_detector.ipynb   the Colab notebook, as run, outputs included
```

**The notebook is a record, not a reproduction script.** It is exported as it
ran, which means two things. Cells 12 and 13 contain real `OutOfMemoryError`
tracebacks — they are left in because the OOM is what exposed the `lm_head`
waste described in §7, and deleting it would remove the evidence for the fix.
And several steps were run transiently rather than as cells (the question-order
diagnostic, the freeze, the margin dumps), so the notebook does not execute
cleanly top to bottom: cell 6 references an `m_base` defined in one of those
transient runs. For a clean re-run use `scripts/15_run_small.py`, which
reproduces the notebook's margins bit-exactly (max |Δ| 0.000000 over 150 items).

`frozen_config.json` carries a `PROVENANCE_NOTE`: the freeze timestamp was
overwritten by a later download from the Colab VM and restored from the session
record, so it should not be taken on trust from the file alone. Corroboration:
long3 held 118/225 generations at freeze time and did not exist as a benchmark;
the full dev result was sent to and independently replicated by the session
maintaining `NARRATIVE.md` before long3 finished generating; and
`margins_long2.json` (00:51) predates `study/single/long3/key.jsonl` (01:14).
