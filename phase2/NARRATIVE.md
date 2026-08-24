# Narrative — phase 2

Continues `phase1/NARRATIVE.md`. Phase 1 is closed and sealed; this file records
what happened after, starting with an audit of what phase 1 actually left behind.

Same convention as phase 1: where a claim here is unsupported by the data, it
says so. Numbers that were recomputed from the artifacts in this repo are marked
as such; numbers taken from published literature are marked separately, because
one kind can be checked here and the other cannot.

---

## 1. Reorganisation

Everything that was at repo root moved into `phase1/`, including `.env` and
`.gitignore`. `phase2/` was created empty. Two consequences worth remembering:

* Anything that loaded `.env` by a path relative to the old root needs updating.
* `.gitignore` now sits one level down. Not a git repo at time of writing, so
  nothing broke, but a future `git init` at the real root will want it back.

## 2. Audit: what phase 1 left behind

### The corpus is complete on the human side and stalled on the AI side

| stage | contents |
|---|---|
| `data/raw/` | 107,280 ELI5 posts, 278 MB, filtered from 216,147 seen |
| `data/interim/` | `questions_1000` (50–250 words, median 101) and `questions_longform` (250–799, median 344), disjoint |
| `data/eval/` | pool 650 / dev 100 / heldout 100 / burned 150 |
| `data/final/` | empty |

The four eval splits are **exactly disjoint and sum to 1,000** — verified by set
intersection, all six pairwise overlaps are zero. `questions_longform` has no
split file; it is staged but never partitioned.

`data/interim/ai_answers.jsonl` holds **2 rows**. `generate.log` and
`generate3.log` both open with "0 already done, 3000 to generate" and then hit a
wall of HTTP 429s across grok and qwen. `generate2.log` claims "70 already done"
but that output is not in the file, so a prior run's results went somewhere else
or were lost. At the ~$0.002/answer visible in the surviving rows, the full 3,000
is roughly $6.

This is a dead end from phase 1, not a phase 2 task, but it is the reason
`data/final/` is empty and it should not be mistaken for work in progress.

### The eval results reproduce exactly

Scoring `study/small/long2_eval/` and `long3_eval/` predictions against their
keys, independently of `RESULTS.md`:

| | AUC | balanced | detection | FPR |
|---|---|---|---|---|
| long2 (dev) | 0.725 | 68.7% | 41/75 = 55% | 17% |
| long3 (confirmation) | 0.771 | 71.3% | 50/75 = 67% | 24% |

Digit-for-digit agreement with the published table. The confirmation set moved
*up* from dev under a frozen threshold, which is the direction that does not
happen when a result has been fitted.

Per-generator recall is uneven and not stable across sets — qwen evades at
exactly 48% on both, while deepseek swings 56% → 88%:

```
long2   human 83%   grok 60%   deepseek 56%   qwen 48%
long3   human 76%   grok 64%   deepseek 88%   qwen 48%
```

### Generation census by prompt version

**2,134 AI answers**, all texts unique (verified by md5 over stripped text, zero
collisions). 388,016 words, mean 182 per answer, $11.99 in recorded
`usage.cost`. Near-perfectly balanced by generator: deepseek 713, qwen 711,
grok 710.

| prompt | n | share |
|---|---|---|
| v6 | 974 | 45.6% |
| v7 | 480 | 22.5% |
| v8_invert | 300 | 14.1% |
| v9 | 288 | 13.5% |
| v4 / v5 | 30 / 30 | 1.4% each |
| v3 + unversioned | 32 | 1.5% |

The raw file total is 2,409 rows. Two files are staging artifacts, not data, and
are excluded above:

| excluded | rows | why |
|---|---|---|
| `study/longform/answers_prededup.jsonl` | 245 | 180 of its texts also appear in the deduped `answers.jsonl` |
| `judge/iter01_INVALID/answers.jsonl` | 30 | byte-identical to `iter01/`, marked invalid |

An earlier count in this session gave 2,407 by summing every `answers*.jsonl`
without excluding those two. That figure was inflated by ~275 and is superseded.

**Untidy provenance worth knowing:** `answers_prededup.jsonl` is not a clean
pre-dedup superset. It has 65 texts that appear nowhere else, and 90 fewer texts
than the file it supposedly precedes — it looks like a mid-generation snapshot
rather than a true pre-dedup copy. Nothing depends on it, but that run cannot be
cleanly reconstructed from these two files alone.

`items_phase1.jsonl` carries no `prompt_version` column; its 1,188 AI rows were
resolved by matching text back against the four versioned `study/phase1/*/`
answer files. All 1,188 matched, no unmatched rows.

**The adversarial inversion prompt is quarantined and stayed quarantined.**
`v8_invert` exists only in `study/phase1/v8_invert/` and appears only in
`items_phase1.jsonl`. The four headline eval sets — `long`, `long2`, `long3`,
`short` — are **100% v6 on the AI side**, checked row by row. Nothing measured in
phase 1's headline was contaminated by the inverting prompt.

### A provenance gap, found and closed

Phase 1 used seventeen detector prompts in two families. The seven `s*`
classification prompts were versioned as files and archived in `prompts.tar`.
The ten `a*` descriptive axes were **not** — they existed only as inline Python
dicts in `lfm2_detector.ipynb` cells 8 and 9, plus the single winner stored in
`frozen_config.json`'s `template` field. Losing the notebook would have taken
nine of the ten axis texts with it, including all nine losers, which are the
comparison that makes the winner legible.

All ten were extracted to `phase1/study/small/prompts/a*.txt` with an index at
`prompts/AXES.md`. Each file holds the complete prompt — the shared `HEAD` block
plus that axis's question.

**The extraction is verified rather than assumed:** sha256[:16] of
`a4_generic.txt` is `aedc7fb4af42fd16`, matching `frozen_config.json`'s
`prompt_sha` exactly. The file is byte-identical to the text that produced the
0.771 confirmation result. The other nine hashes are recorded in `AXES.md`.

This is the only modification made to `phase1/` beyond its move, and it adds
files without altering any existing one.

## 3. The paragraph artifact, re-examined

Phase 1 §15 found that `wc -l` scores AUC 0.802 on long2 and 0.803 on long3,
beating the 1.2B detector on both. Phase 2 opened by testing a natural reading of
that finding: *v6 makes grok never use line breaks, and that is the artifact.*

Three parts, and the data supports one of them.

### "Never" is too strong

Grok's zero-newline rate on the long-form eval sets is 44% (long2) and 72%
(long3). Heavy, but not absolute.

### It is not v6's doing

Grok's zero-newline rate by prompt version, across every generation run in the
repo:

| prompt | v3 | v4 | v5 | v6 | v7 | v8_invert | v9 |
|---|---|---|---|---|---|---|---|
| grok zero-newline | 100% | 70% | 90% | 69% | 88% | **0%** | 69% |

The behaviour is prompt-invariant across the whole v1–v9 lineage. The single
prompt that suppresses it is `v8_invert`, which forces structure by planting
`Edit:` blocks — the prompt that was rejected for exactly that kind of planting.

So v6 did not cause the artifact. It failed to correct it, which is what phase 1
§15 already said: the rate-matching program matched word count, markdown rate,
artifact rate and persona rate, and never matched paragraph structure.

### It is not only grok

Newline-baseline AUC computed per generator against the same 75 humans
(recomputed here):

| | long2 | long3 | short |
|---|---|---|---|
| grok alone | 0.950 | **0.996** | 0.859 |
| deepseek alone | 0.795 | 0.801 | 0.817 |
| qwen alone | 0.659 | 0.614 | 0.658 |
| all three | 0.802 | 0.803 | 0.778 |
| **all minus grok** | 0.727 | 0.707 | 0.737 |

Grok alone is near-perfectly separable on long3 at 0.996 — effectively a lookup
table. But **deepseek alone scores 0.80, matching the full baseline on its own.**
Removing grok does not clean the benchmark; it leaves a 0.707 baseline and merely
lowers the bar the detector has to clear. Phase 1 quoted the leave-one-out cost
(0.075–0.096 AUC) and these figures land inside that range; the part not
previously recorded is that deepseek stands at 0.80 unaided.

### A length confound in the artifact itself

Zero-newline rates differ sharply between the short-form and long-form corpora
for the same model:

* short-form (`study/phase1`, 50–250 words): deepseek 88%, grok 88%, qwen 48%
* long-form eval sets (250–799 words): deepseek 0%, grok 72%, qwen 0%

Long documents acquire paragraphs from every model except grok. On the long-form
sets the ordering is monotonic in median newline count — grok 0, deepseek 6,
qwen 8, human 12 — so all three generators sit below the humans and all three are
separable, but only grok is separable by *absence* of line breaks.

Any rate-matching fix therefore needs per-model, per-length constants. A single
global paragraph instruction will not do it.

## 4. The stratified-AUC control, and what it found

The open worry going into phase 2 was that anything trained on this corpus
becomes a whitespace detector, since newline count (0.803) is a stronger feature
than the detector's own output (0.771).

To test whether the *existing* detector already had that problem, both eval sets
were binned by newline count and AUC recomputed **within bins only**, so the
confounder carries no information about the label.

| | overall | newline held constant |
|---|---|---|
| long2 variant A (full 16) | 0.725 | **0.719** |
| long3 variant A (full 16) | 0.771 | **0.784** |
| long2 variant B (exit@6) | 0.747 | **0.617** |
| long3 variant B (exit@6) | 0.791 | **0.630** |

**Variant A loses essentially nothing.** It goes up on long3. Whatever it reads,
it is not line breaks.

**Variant B loses 0.13–0.16 and lands near chance**, collapsing to 0.548 in the
largest band. Phase 1 §4 rejected variant B on a within-class correlation
argument; this confirms the rejection by an independent route. The stratified
test is the stronger of the two, because a within-class correlation can be zero
while the feature still drives the decision through the *other* class.

This is the main new result of the session: **the frozen zero-shot readout is
already artifact-immune, and it is immune because it was never fitted to the
corpus.** It cannot overfit to a feature it was never shown.

### What stratified AUC costs

AUC's unit of data is the pair, not the document. With 75 AI and 75 human items
there are 5,625 pairs; stratifying into five newline bands keeps 1,104 of them
(20%). Only 18 documents drop out entirely — the 0-newline band, which is 100% AI
and has no humans to compare against — but every cross-band comparison is deleted.

The retention is mostly structural, not a property of this data. For **k**
equally-sized bins with balanced classes, the retained fraction is exactly
**1/k**:

```
per bin (N/2k)²  ×  k bins  =  N²/4k        divided by total N²/4   =  1/k
```

Five bins keeps a fifth. If the bins here were class-balanced the retention would
be about 25%; the gap down to 20% is the confounding cost, most of it that one
all-AI band contributing literally zero comparisons.

So the loss is ~80% from choosing five bins and a few points more from the
artifact. An earlier framing in this session attributed the loss primarily to
confounding; the arithmetic says binning granularity dominates.

**Consequence:** 0.784 is a control, not a headline. On 1,104 pairs the interval
is wide enough that 0.784 and 0.771 are not meaningfully different — which is
itself the finding. Two or three bins would retain 33–50% and still remove most
of the free wins, and is probably the better operating point than five.

The measure of how much the artifact is worth: in **77% of all 5,625 pairs the AI
document has strictly fewer newlines than the human one.** Those are winnable by
`wc -l` with no language understanding at all, and that 77% plus ties is exactly
where the 0.803 comes from.

## 5. Literature: LLMs as classifiers, and which layers are disposable

Prompted by variant B's failure. **These are citations, not measurements — none
of the numbers in this section were verified in this repo**, and the ones drawn
from search summaries rather than from reading the paper are flagged.

### The consensus is: drop the deep layers, read from the middle

* [Gromov et al., *The Unreasonable Ineffectiveness of the Deeper Layers*](https://arxiv.org/abs/2403.17887)
  — prune a contiguous block of **deep** layers chosen by angular distance
  between block input and output, heal with QLoRA. Up to half the layers come out
  with minimal degradation. Note the shape: it removes a deep-but-not-final block
  and keeps the final layers. Structurally different from truncating at layer k.
* [ShortGPT](https://arxiv.org/html/2403.03853v3) — Block Influence,
  `BI = 1 − cos(layer_input, layer_output)` over a calibration set; prune lowest
  BI one-shot, ~25% of LLaMA-2.
* [Does Representation Matter?](https://arxiv.org/abs/2412.09563) (NeurIPS 2024
  workshop) — intermediate layers beat final layers on representation quality,
  scored by prompt entropy, curvature, augmentation-invariance.
* [A BERTology View of LLM Orchestrations](https://arxiv.org/html/2601.13288v2) —
  probe accuracy peaks at layer 16/32 for Llama-8B and 18/~36 for Qwen-4B, i.e.
  50–60% depth. Also finds all-layer concatenation beats any single layer.
* [A Primer in BERTology](https://arxiv.org/pdf/2002.12327) — the stable
  hierarchy: surface and lexical features at the bottom, syntax in the middle,
  semantics at the top.

**This predicts variant B's failure.** Exiting at layer 6 of 16 keeps the bottom
37.5% — the surface-feature band — and discards everything above. A detector
built there reads surface form almost by construction, and that is what the
stratified test found. The pre-registered caveat worried the AUC spike would not
replicate; the AUC *did* replicate (0.747 → 0.791). What broke was what the layer
was reading, which is a different failure than the one that was registered.

### Classification tolerates pruning better than generation

Reported retention at a 25% prune rate is ~92% on classification against ~68% on
generation, because generation compounds error token by token and a scoring pass
emits zero tokens. *(From search summaries of
[the pruning-hierarchy work](https://arxiv.org/html/2603.24652), not verified by
reading the paper. Check before quoting.)*

### The readout is a bigger lever than the layer — with a catch

[LLM2Vec](https://arxiv.org/abs/2404.05961) (bidirectional attention + MNTP +
contrastive) and [Label Supervised LLaMA Finetuning](https://arxiv.org/pdf/2310.01208)
(drop the causal mask, linear head on the last token) both report large gains
over zero-shot readouts, and the BERTology-view probes hit 95.15% on IMDB against
DeBERTa's 95.34% using frozen features and a head of 0.003M–35M parameters.

**But a trained probe is the wrong move on this corpus,** and this was a wrong
recommendation made and retracted within the session. Fitting anything to data
where newline count outscores the detector hands the optimiser the artifact.
Section 4's result exists *because* the current readout is unfitted.

### Caveat specific to LFM2.5

Effectively all of this literature is on pure transformer stacks. LFM2.5 is a
hybrid — 10 gated convolutions and 6 GQA attention blocks — and phase 1 §4 found
the attention blocks carry the decision. Block Influence assumes layers are
comparable residual updates, which conv and attention blocks are not, and the 6
attention blocks are unlikely to be evenly spaced. Dump the actual block-type
ordering before applying any of it: the real question may be "which convs sit
between the attention blocks I need", which is not the question these papers
answer.

## 6. Open decisions for phase 2

Not yet acted on. Recorded so the reasoning is not lost.

1. **Report stratified AUC as standard.** Free on existing runs, and it makes the
   failure mode visible the moment it appears instead of three sets later. Use
   k=2 or 3, not 5.
2. **Normalise whitespace at input.** Collapse newline runs before scoring. Then
   the feature is not in the tensor, no fitting can find it, and the control runs
   on all 5,625 pairs with no bin choice to defend.
3. **Add paragraph-structure matching upstream**, per model and per length band
   (§3). This is the honest fix and the only one that produces a clean corpus
   rather than a corrected measurement.
4. **Leave-one-generator-out evaluation.** Highest-value item. The corpus is
   3 models × 1 prompt, so a fitted classifier can win by identifying *which
   model wrote this* rather than *is this machine prose*. Question-level splits
   do not catch this — long2 and long3 are disjoint by question id but share all
   three generators. Train on two models, test on the third.
5. **If a readout is trained at all**: read at ~layer 10 of 16 rather than 6,
   select depth by Block Influence on *unlabeled* calibration text so no eval
   labels are spent, train on newline-normalised text, and hold the successor to
   long3 completely untouched until one shot.

### The standing risk

Newlines are the artifact that was found, and it was found by one reader looking
once. Nothing suggests it is the only one. Strip it and a fitted probe finds the
next-strongest surface feature — sentence-length variance, comma rate, hapax
ratio, whatever v6 also forgot to match. A corpus generated by three models under
one prompt has generator fingerprints throughout, and discriminative training is
very good at finding fingerprints.

That is the argument for keeping the readout unfitted, and for item 4 above being
the split that actually matters.

## 7. exp01 — ModernBERT, complete

Addresses items 2 and 4 of section 6 together: whitespace normalisation and
generator generalisation, in one design. Spec and data in
`phase2/exp01_modernbert/`, pre-registered in its `DESIGN.md` before any model
trained. Executed by a peer session handling Colab; this session designed it.

**Three tasks as designed, 20 cells, 60 runs. Task C was cut mid-sweep by the
user; 24 runs executed plus the frozen control, zero failures.** v6 only, so generator is not confounded with
prompt. 294 questions carrying all three generators and a human answer.
Question-level split 206/88 — never document-level, or a question's human answer
and its three AI answers land on both sides.

* **A_4way** (2 cells) — human/grok/deepseek/qwen. **Natively balanced at
  294 per class**: every core question contributes exactly one document to each.
  The binary task's 2:1 AI:human skew was an artifact of collapsing three AI
  classes into one. Headline is the 4x4 confusion matrix and the six pairwise
  AUCs, not accuracy.
* **B_unseen_assign** (6 cells) — train 3-way on human + two generators, test
  only the held-out generator. It must be assigned to a seen class, and which
  one it picks measures which known model it resembles. This is
  leave-one-generator-out reinterpreted for a multiclass label set: you cannot
  ask a classifier to name a class it has never seen, but you can ask who it
  mistakes it for. The share landing on `HUMAN` is the false-negative rate for a
  genuinely novel generator.
* **C_binary** (12 cells) — the original registration, kept for continuity.

**Baselines fixed before training.** `wc -l` on the three test sets: grok 0.887,
deepseek 0.749, qwen 0.585 raw; exactly 0.500 for all three under `norm`, by
construction. That 0.500 doubles as the pipeline sanity check.

### A determinism bug, found by the peer and fixed

The first `prep.py` seeded its per-cell subsample with
`random.Random(hash((SEED, g, cond)) & 0xffffffff)`. Python randomises `hash()`
of tuples containing strings per interpreter, so **re-running produced different
splits** — three processes, three values, confirmed. The claim that the build was
deterministic, made in this file and to the peer, was false when made.

Fixed to a string-derived seed, `random.Random(f"{SEED}|{g}|{cond}")`. The peer
recommended freezing the existing `splits.json` instead, on the grounds that
regenerating would change the cells. With **zero cells run there were no results
to protect**, so freezing an unreproducible file bought nothing and cost
reproducibility — the same failure mode as the clobbered `frozen_at` timestamp
and the notebook-only axis prompts, both already recorded here. Regenerated
instead. Only the 6 `seen` cells changed; all 6 `unseen` cells were byte-identical,
because their sample pool is exactly the sample size.

Hashes: `docs.jsonl` `d8722c165e142b81`, `splits.json` `dfc459701c2fbbb5`.

### On the record before any numbers existed

Full predictions are in `DESIGN.md`; outcomes and all working in `FINDINGS.md`,
which is kept separate so the pre-registration cannot drift toward what happened.

## 8. exp01 — what it found

Detail and every caveat in `exp01_modernbert/FINDINGS.md`. The headlines:

### The whitespace worry did not materialise

With newlines provably absent — 0 of 1,176 normalised documents contain one, all
six newline baselines pinned at exactly 0.500 — the fitted model still separates
every generator pair at **≥0.983**. Normalisation also *stabilised* Task A: seed
range 0.0426 → 0.0028, a **15x collapse**, corroborated independently by stopping
epochs going [10, 5, 8] → [10, 10, 9].

Read as: whitespace was a **noisy** feature. Predictive in aggregate but
inconsistent per document, so a fitted model partly chased it and paid in
variance. Remove it and the model is forced onto prose, which is **no worse and
far more stable**.

**"Stronger" would overstate it and is not supported.** Accuracy went 0.922 →
0.941 on the mean, but the seed ranges overlap — raw's best (0.9460) exceeds
norm's best (0.9432) — and a mean difference of 0.019 between two overlapping
three-point samples is exactly what this geometry cannot adjudicate. The claim is
**15x variance collapse with accuracy statistically unchanged**. The noisy-feature
reading rests on the variance result alone and does not need the accuracy delta. That is the fitted-model counterpart to phase 1's finding that the
frozen readout was artifact-immune.

**Not generalisable, and this was caught:** across all eight comparable raw/norm
quantities the tally is three more stable, two less, three unchanged.
"Normalisation stabilises" is **not a finding**; "normalisation stabilised Task
A" is.

### Detection works; attribution fails in a structured way

Across all 18 Task B runs, a generator absent from training is called *human*
between **0.000 and 0.205** of the time against a chance rate of **0.333** —
never at chance, in any fold, either condition. But it does not spread across the
remaining classes: it **collapses onto its nearest neighbour**, with confidence
0.663 to 1.000.

The magnitude is seed-unstable and must not be quoted as a point estimate. One
run in nine gives 0.205 where its siblings give 0.011.

### The strongest result: a similarity structure that predicts out-of-sample

Task A's **prior-free** pairwise AUCs imply a nearest-neighbour ordering —
deepseek→qwen, qwen→deepseek, grok→deepseek. Task B, with the generator deleted
from training and a different label set, produces exactly that in **all six
cells**, both conditions.

Pairwise AUC is a two-class ranking statistic, invariant to class priors and
decision-region size. A competing "deepseek is just a hub with a large decision
region" hypothesis was raised and killed by direct simulation: biasing deepseek's
logit moved argmax share 91→97 while pairwise AUC did not move to four decimals.

deepseek and qwen are the closest pair in the corpus, on **four structurally
independent measurements**. That is a finding about the models, not the detector.

### The frozen control

The unfitted LFM2.5 readout scores **0.720 pooled raw / 0.678 norm**, 0.66–0.75
per generator, with zero training and no exposure to this corpus. Reported as a
standalone reference: it is a binary AI-vs-human AUC, while the fitted numbers
are 4-way accuracy and 3-way assignment shares — different quantities on
different label sets. **The like-for-like comparison was Task C, and Task C was
cut.**

Its most interesting property: holding newline count constant **raises** its AUC
in every fold (+0.047 to +0.097). Whitespace is mildly working *against* it.
Phase 1 saw the same sign on long3 at +0.013 — same direction, larger magnitude,
different documents.

### Predictions, settled

| # | outcome |
|---|---|
| 1 | HUMAN easiest class — **confirmed**, both conditions |
| 2 | **right for the wrong reason** — grok stayed most separable with newlines gone, so the formatting rationale was false |
| 3 | **FAILED decisively** — predicted a pair below 0.65; the closest is 0.983 |
| 5 | **satisfied on its letter**, by a mechanism it did not anticipate |
| 6 | **unresolved** — mean and median disagree on direction |
| 4, 7-11 | **UNRUN** by scope |

Prediction 3's refutation is worth more than a pass would have been. A
confirmation would have said two models are hard to tell apart. The refutation
says all three are individually identifiable from prose with the artifact
provably absent.

## 9. What the process cost and bought

**At least nine substantive claims were withdrawn or corrected during the sweep**,
every one because enough was recorded for a second party to re-derive from rather
than reading conclusions. None reached a writeup. An earlier draft of this section
said six; that was a grouping, and the defensible count is higher:

| withdrawn claim | who |
|---|---|
| gradient checkpointing is "mathematically identical" | executor |
| prediction 2's F1 ranking (grok "second-worst" among generators) | executor |
| "normalisation helped, +0.019" | executor |
| "the CIs do not overlap" | executor |
| pooled n=264 as independent trials | **both** — designer first |
| "no stable subset of human-looking documents" (a power error) | executor |
| "grok is uniquely unstable" and its distance mechanism | executor |
| "normalisation stabilises" as a general claim | executor, relayed by designer |
| skew bimodality, "nothing in between" | executor |
| "max_epochs is a floor" (read off a partial val curve) | designer |
| deepseek mass on held-out qwen exceeds held-out grok | designer, pre-registered |
| the "similarity gap" repair of that test | designer |
| "removing the artifact made the model stronger" | designer, in this file |

Both sessions produced them — eight from the executor, five from the designer.
The asymmetry is only in how each surfaced: the executor's when the designer
re-derived from the mirror, the designer's when the executor read the write-up.
**Neither mechanism catches the other's errors**, which is the real argument for
two sessions rather than one careful one. Had this section read as one session
auditing another it would have implied the executing session was the unreliable
one, and the record does not support that.

### A withdrawn claim does not stay withdrawn on its own

The most instructive entry in that table is the last. "Removing the artifact made
the model stronger" had already been withdrawn — by the designer, that morning,
after catching the executor's version of it — and was then written into the
narrative by the same session that withdrew it, in the most quotable sentence of
§8.

So a claim does not stay retracted by having been retracted. It needs to be
**recorded as withdrawn somewhere the writer will encounter it**, or the original
framing reasserts itself when the prose gets written — prose reaches for the
strong version, and the strong version is the one that was retracted.

`FINDINGS.md` marks withdrawn claims as withdrawn rather than deleting them, and
that convention is what made this catchable. It is worth keeping for that reason,
not merely for bookkeeping.

Three of the executor's were **selection errors** — a favourable comparator chosen rather than
computed over everything available. The diagnosis that generalises, from the
executing session: *the comparator that comes to mind is the one that made the
contrast salient enough to report*. Selection happens before conscious
comparison, so "check your comparator" is unreliable advice to give oneself. Four
scripts now exist that enumerate by default, and **all four were written because a
hand-picked version of the same check failed first**.

Two were **power errors** — interpreting a statistic without first computing what
it could resolve. A zero triple-overlap was read as evidence against a stable
hard core when its expectation under independence was 0.008.

**Pre-registration earned its keep twice, in different ways.** The case-3
reporting rule, fixed before Task B produced a number, forced an outcome nobody
had predicted to be reported as *satisfied on its letter, by an unanticipated
mechanism*. And the skew threshold refused to claim on the grok fold when seeds
disagreed — a result that at 0.65 would have read as a clean three-for-three.

But the case machinery was valuable as a **discipline**, not as a
**measurement**: the verdict flips on a boundary chosen semi-arbitrarily, and the
robust quantity is the continuous skew, true under every threshold. Conflating
those two contributions is how pre-registration gets oversold.

## 10. Reproducibility

Full assessment in `FINDINGS.md`. Three layers:

* **Data and splits — exact.** `prep.py` regenerates both files byte-identically,
  verified across randomised `PYTHONHASHSEED` and again after the sweep.
* **Analysis — exact, without a GPU.** Per-document `probs` in every record made
  every number recomputable by a second party. That is what caught nearly all of
  the withdrawn claims in §9.
* **Training — not reproducible, and demonstrated.** Same cell, same seed, same
  hyperparameters, changing only gradient checkpointing: accuracy 0.9347 → 0.9034
  and epochs 10 (cap) → 5 (early stop). Floating-point non-associativity in
  recomputed activations perturbs gradients and trajectories never re-converge.

**The environment was never captured** — no library, CUDA or GPU versions in any
of the 24 records. Biggest gap and the cheapest to have closed.

The concrete form of it is a better argument than any general statement. The
executing session hit **two `transformers` v5 API removals during setup** —
`overwrite_output_dir` and `warmup_ratio`, both surfacing as tracebacks that
named the library — and reconstructed `warmup_ratio` as an explicit step count so
the pre-registered recipe survived intact. Then recorded neither the transformers
version, nor CUDA, nor the GPU model. **The evidence for why environment capture
matters was in the traceback that made the fix necessary.**

**Conclusions are reproducible; numbers are not.** Ordinal and directional
findings rest on agreement across many runs. Magnitude claims rest on three seeds.

## 11. exp02, well-specified

1. **More generators.** Three gives three pairwise distances, two differing by
   0.003 — the design cannot test a similarity *metric* at all, only its ordering.
   Four or five would give 6–10 distances. A corpus change, not a training change,
   and the same lever leave-one-generator-out wanted anyway.
2. **More seeds, 5-10**, for three distinct reasons that should not be merged:
   seed variance is a lower bound at n=3; overlap statistics buy a *different
   question* once counts rise; and the headline h estimate is simply not usable at
   n=3.
3. **Independent training corpora** to separate "intrinsically human-like
   document" from "document all models fail on for a shared reason". No number of
   restarts within one setup separates these.
4. **Capture the environment** in every record; pin versions.
5. **Keep per-document `probs`** — a hard rule, not a suggestion.

## 12. State

* `phase1/` sealed. Two changes only: its move, and ten `a*.txt` files plus
  `AXES.md` under `study/small/prompts/` (§2). No data, result or config file was
  altered.
* `phase2/` holds this file and `exp01_modernbert/` — spec, data, splits, code,
  and `results/` with 24 A+B records, the frozen control, and three archived
  files (superseded run 1, the out-of-scope Task C fragment, and the old-path
  comparison).
* **exp01 complete.** 24 of 24 runs plus the frozen control, zero failures.
  Nothing running.
* Task C **not run**: predictions 7-11 and the 0.65 falsification line are
  neither passed nor failed.
