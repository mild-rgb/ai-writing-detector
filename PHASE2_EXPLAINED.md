# Phase 2, explained

*A plain-language guide to what phase 2 did, what it found, and what it had to
retract. Assumes you have read `PHASE1_EXPLAINED.md`, or at least know that
phase 1 ended by discovering that counting line breaks beat a real detector.*

Companion to `phase2/NARRATIVE.md` and `phase2/exp01_modernbert/`.

---

## Where phase 1 left things

Phase 1 ended with an uncomfortable pair of numbers:

- A small language model, used as a detector without any training, scored
  **AUC 0.771** on held-out long-form documents.
- Counting the line breaks in those same documents scored **AUC 0.803**.

The formatting artifact was *stronger than the detector*. That raises an obvious
worry: if you take this corpus and actually **train** something on it, what stops
the trained model from simply becoming a whitespace counter?

Phase 2 is the answer to that question, plus an audit of everything phase 1 left
behind.

---

## Part 1: The audit

Phase 1 was closed and sealed. Before building anything new, phase 2 checked what
was actually in the box.

### The corpus

| stage | what is there |
|---|---|
| raw | 107,280 ELI5 posts, 278 MB |
| interim | 1,000 short-form questions (median 101 words) and 1,000 long-form (median 344), disjoint |
| eval splits | pool 650 / dev 100 / heldout 100 / burned 150 |
| final | **empty** |

The four evaluation splits were verified to be exactly disjoint and to sum to
1,000 — all six pairwise overlaps are zero. Good.

`data/final/` being empty is not work in progress. It is a dead end from phase 1:
the generation logs show runs that opened with "3,000 to generate" and then hit a
wall of HTTP 429 rate-limit errors. At the observed cost of about $0.002 per
answer, the full production dataset would cost roughly $6 to produce. It was
simply never run.

### The generation census

**2,134 AI answers exist**, all texts verified unique (no duplicates by
checksum). 388,016 words, $11.99 in recorded API cost, near-perfectly balanced
across the three generators (713 / 711 / 710).

By prompt version: v6 accounts for 45.6%, v7 for 22.5%, the inversion prompt
v8_invert for 14.1%, v9 for 13.5%.

**The important check:** the inversion prompt — the one that planted `Edit:` into
100% of documents and was rejected as unusable — was verified to be quarantined.
The four headline evaluation sets are **100% v6 on the AI side**, checked row by
row. Nothing in phase 1's published numbers was contaminated by it.

### A provenance gap, found and closed

Phase 1's small-detector work used seventeen prompts in two families. Seven were
saved as files. **The other ten existed only as inline Python dictionaries inside
a Colab notebook.** Losing that notebook would have destroyed nine of the ten,
including all nine losing variants — which are the comparison that makes the
winner meaningful.

All ten were extracted to files. The extraction was *verified* rather than
assumed: the checksum of the winning prompt matches the hash recorded in the
frozen config exactly, so the extracted file is byte-identical to the text that
produced the published 0.771 result.

This is the only change made to phase 1 beyond moving it into a subdirectory, and
it adds files without altering any existing one.

---

## Part 2: Re-examining the paragraph artifact

Phase 2 opened by testing the natural reading of phase 1's finding: *the v6
prompt makes Grok never use line breaks, and that is the artifact.*

Three parts to that claim. The data supports one.

### "Never" is too strong

Grok's zero-newline rate on the long-form sets is 44% and 72%. Heavy, but not
absolute.

### It is not v6's fault

Grok's zero-newline rate, across every prompt version ever used:

| prompt | v3 | v4 | v5 | v6 | v7 | v8_invert | v9 |
|---|---|---|---|---|---|---|---|
| grok, no line breaks | 100% | 70% | 90% | 69% | 88% | **0%** | 69% |

The behaviour is essentially **prompt-invariant** across the entire nine-version
lineage. The only prompt that suppresses it is `v8_invert` — and it does so by
forcing structure through planted `Edit:` blocks, which is exactly the technique
that got that prompt rejected.

So v6 did not *cause* the artifact. It failed to *correct* it. That is what phase
1 already said: the rate-matching programme matched word count, markdown,
Reddit artifacts and persona, and never matched paragraph structure.

### It is not only Grok

The line-break baseline, computed per generator against the same human documents:

| | long2 | long3 | short |
|---|---|---|---|
| grok alone | 0.950 | **0.996** | 0.859 |
| deepseek alone | 0.795 | 0.801 | 0.817 |
| qwen alone | 0.659 | 0.614 | 0.658 |
| all three | 0.802 | 0.803 | 0.778 |
| **all except grok** | 0.727 | 0.707 | 0.737 |

Grok alone is near-perfectly separable — effectively a lookup table. But
**DeepSeek alone scores 0.80, matching the full baseline on its own.** Dropping
Grok does not clean the benchmark. It leaves a 0.71 baseline and merely lowers
the bar the detector has to clear.

### A length confound inside the artifact

The same model behaves differently at different lengths:

- **Short documents** (50–250 words): deepseek 88% zero-newline, grok 88%,
  qwen 48%
- **Long documents** (250–799 words): deepseek 0%, grok 72%, qwen 0%

Long documents acquire paragraphs from every model *except* Grok. So any
paragraph-matching fix needs per-model, per-length constants. A single global
instruction will not do it.

---

## Part 3: The control that mattered

The worry was that anything trained on this corpus becomes a whitespace detector.
But phase 1's detector was never *trained* on the corpus — it was a frozen,
off-the-shelf model being read out. Did it have the problem anyway?

### Stratified AUC

The test is simple in concept. Sort the documents into bins by line-break count,
then measure the detector's performance **only within each bin**. Inside a bin,
every document has roughly the same number of line breaks, so line-break count
carries no information about the label. If the detector's score holds up, it was
not using the artifact.

| | normal AUC | line-breaks held constant |
|---|---|---|
| long2, full model | 0.725 | **0.719** |
| long3, full model | 0.771 | **0.784** |
| long2, truncated variant | 0.747 | **0.617** |
| long3, truncated variant | 0.791 | **0.630** |

**The full model loses essentially nothing.** It goes *up* on long3. Whatever it
reads, it is not line breaks.

**The truncated variant collapses**, losing 0.13–0.16 and landing near chance in
the largest bin. (This variant was an experiment that read the model's internal
state at layer 6 of 16 rather than running all the way through — a
cheaper-and-faster idea that phase 1 had already rejected on separate grounds.)

**The main new result of the session:** the frozen, unfitted readout is already
artifact-immune, *and it is immune precisely because it was never fitted*. It
cannot overfit to a feature it was never shown.

### What the control costs

Stratifying is not free. AUC's unit of data is the *pair* — every AI document
compared against every human one. With 75 of each there are 5,625 pairs.
Splitting into five bins keeps only 1,104 of them, about 20%.

Most of that loss is structural, not a property of this data. For k equally-sized
bins with balanced classes, the fraction retained is exactly **1/k**. Five bins
keeps a fifth, no matter what you are studying. So the practical advice is to use
two or three bins, not five — you retain 33–50% and still remove most of the free
wins.

An earlier framing in the session blamed the loss primarily on the confounding.
The arithmetic says binning granularity dominates. Corrected.

**And the measure of how much the artifact is worth:** in **77% of all 5,625
pairs, the AI document has strictly fewer line breaks than the human one.** Those
pairs are winnable with no language understanding at all. That 77% is where the
0.803 comes from.

---

## Part 4: The main experiment — exp01

The audit established what the *unfitted* readout does. The real question was
what happens when you fit a model to this corpus.

Two shortcuts are available to a fitted classifier, and neither is prose:

1. **Formatting** — line breaks, worth up to AUC 0.887 on one split.
2. **Generator fingerprints** — recognising *Grok, DeepSeek, or Qwen
   specifically*, rather than machine prose in general. A classifier that has
   learned "this reads like Grok" will fail completely on a model it has never
   seen — which is the deployment condition.

exp01 removes both and reports what is left.

### The design, pre-registered

The full specification — including predictions, thresholds, and how to report
each possible outcome — was written into `DESIGN.md` **before any model was
trained**, and that file has not changed since. Results go into a separate
`FINDINGS.md`. The separation exists so the predictions cannot quietly drift
toward what happened.

**The data:** 294 questions that carry a human answer and an answer from all
three generators, using v6 only (mixing prompt versions would confound generator
with prompt). That is 1,176 documents.

**The split:** by *question*, never by document. 206 questions for training, 88
for testing. If you split by document, a question's human answer could land in
training while its AI answers land in testing, and the model can cheat by
recognising the topic. Verified: zero question leakage.

**The two text conditions:**
- `raw` — the text as generated.
- `norm` — all runs of whitespace collapsed to a single space. **Line breaks are
  physically gone.**

**The sanity check this buys:** under `norm`, the line-break baseline is exactly
0.500 — pure chance — for all six class pairs, by construction. Verified: 0 of
1,176 normalised documents contain a newline. So any separation measured in the
`norm` condition is *not* whitespace, because there is none left to use.

**The model:** ModernBERT-large, 395 million parameters. Three random seeds per
configuration.

### The three tasks

**Task A — four-way identification.** Classify each document as human, Grok,
DeepSeek, or Qwen. Naturally balanced at 294 per class, because every core
question contributes exactly one document to each. (The binary task's awkward
2:1 AI:human skew turned out to be an artifact of collapsing three AI classes
into one.)

**Task B — unseen-generator assignment.** This is the clever one. Train on human
plus *two* generators, then test only on documents from the **third generator,
which the model has never seen**. It has no correct answer available — the true
class does not exist in its label set. So the question becomes: *who does it
mistake this for?*

The share that lands on `HUMAN` is the false-negative rate for a genuinely novel
generator. That is the deployment-realistic number this whole project has been
circling.

**Task C — the original binary human-vs-AI task.** **Cut by user decision**
partway through the sweep, on cost: it was 16.1 hours of a 27.7-hour sweep. This
is recorded as a scope cut, never as a result — Task C was the only task
producing a number directly comparable to phase 1's 0.771, so five of the eleven
pre-registered predictions are **neither passed nor failed**.

### A bug caught before it mattered

The first version of the data-preparation script seeded its random sampling with
`hash()` applied to a tuple containing strings. Python randomises string hashing
per interpreter process, so **re-running produced different splits** — confirmed
across three processes, three different values.

The claim that the build was deterministic, which had been made in writing, was
false when made.

The fix was one line. The interesting part is the disagreement about what to do
next: freezing the existing (unreproducible) split file was proposed, to avoid
changing the experiment. But **zero cells had run**, so there were no results to
protect. Freezing an unreproducible file would have bought nothing and cost
reproducibility permanently. It was regenerated instead.

---

## Part 5: What exp01 found

### The whitespace worry did not materialise

With line breaks provably absent, the fitted model still separates **every pair
of generators at AUC ≥ 0.983**.

The prediction had been that at least one generator pair would be nearly
indistinguishable once formatting was removed — below 0.65, with DeepSeek/Qwen
the best guess. DeepSeek/Qwen **is** the closest pair, in every seed, exactly as
predicted. But it sits at **0.983**, not below 0.65.

> **Prediction 3 failed decisively — and the refutation is worth more than a pass
> would have been.** A confirmation would have said two models are hard to tell
> apart. The refutation says all three are individually identifiable from prose
> alone, with the formatting artifact provably absent.

### Removing the artifact made the model *steadier*, not stronger

| | accuracy across 3 seeds | spread |
|---|---|---|
| `raw` | 0.9034, 0.9176, 0.9460 | **0.0426** |
| `norm` | 0.9403, 0.9403, 0.9432 | **0.0028** |

The spread collapsed by a factor of **15**, corroborated independently by the
training runs themselves (the epoch at which training stopped went from
[10, 5, 8] to [10, 10, 9] — much more consistent).

**The accuracy gain must not be claimed.** The mean rose from 0.922 to 0.941, but
the ranges overlap: `raw`'s best seed (0.9460) beats `norm`'s best (0.9432). A
0.019 difference between two overlapping three-point samples is exactly what this
design cannot adjudicate.

> **The claim is "15× variance collapse with accuracy statistically unchanged."**

The reading this supports: whitespace was a **noisy** feature. Genuinely
predictive in aggregate but inconsistent document by document, so a fitted model
partly chased it and paid for it in instability. Remove it, and the model is
forced onto prose — which is no worse and far more stable.

And a caveat that was itself caught and enforced: across all eight comparable
raw/norm quantities, the tally is three more stable, two less, three unchanged.
**"Normalisation stabilises" is not a finding. "Normalisation stabilised Task A"
is.**

### Detection works. Attribution fails in a structured way.

Task B, across all 18 runs: a generator that was absent from training gets called
`HUMAN` between **0.000 and 0.205** of the time, against a chance rate of 0.333.
Never at chance, in any fold, in either condition.

So an unseen machine still reads as a machine. Good news for detection.

But the documents do not spread evenly across the remaining classes. They
**collapse onto one specific class** — the held-out generator's nearest
neighbour — with concentration between 0.663 and 1.000.

**The magnitude of the human-confusion rate is seed-unstable and must not be
quoted as a point estimate.** One run in nine gives 0.205 where its two siblings
give 0.011 — a factor of nineteen inside a single configuration.

| claim | survives? |
|---|---|
| "a novel generator is not mistaken for human at chance rates" | **yes** — all 18 runs below 0.333 |
| "the rate is about 1–5%" | **no** — one run in nine lands at 20.5% |

The honest statement is: *consistently below chance, magnitude seed-unstable,
ranging 0.000–0.205*. A practitioner reading "1.1%" as the expected
false-negative rate would be badly misled.

### The strongest result: a similarity structure that predicts out of sample

Task A's pairwise separability scores imply a nearest-neighbour ordering among
the three models: DeepSeek's nearest is Qwen, Qwen's nearest is DeepSeek, Grok's
nearest is DeepSeek.

Task B — with the generator **deleted from training entirely** and a **different
label set** — reproduces exactly that ordering in **all six cells, both
conditions**.

DeepSeek and Qwen turn out to be mutual nearest neighbours on **four
structurally independent measurements**: Task A's pairwise AUC (lowest of six, in
both conditions), Task A's confusion matrix, Task B with DeepSeek held out, and
Task B with Qwen held out. Different label sets, different training regimes, both
directions of leave-one-out.

**That is a finding about the models, not about the detector.**

A competing explanation was raised and killed: perhaps DeepSeek is just a "hub"
with a large decision region that swallows everything. Direct simulation settled
it — artificially biasing DeepSeek's output moved its share of predictions from
91% to 97% while the pairwise separability score did not move to four decimal
places. Pairwise AUC is a two-class ranking statistic, invariant to how big a
class's decision region is.

### The frozen control

The unfitted small-model readout from phase 1 was run on the identical documents:
**AUC 0.720 raw / 0.678 norm**, with zero training and no exposure to this
corpus.

It is reported as a standalone reference, *not* as a head-to-head comparison —
it produces a binary AI-vs-human AUC while the fitted numbers are four-way
accuracy and three-way assignment shares. Those are different quantities on
different label sets. **The like-for-like comparison was Task C, and Task C was
cut.**

Its most interesting property: holding line-break count constant **raises** its
score in every fold (+0.047 to +0.097). Whitespace is mildly working *against*
it. Phase 1 saw the same sign on different documents at +0.013.

---

## Part 6: What the process cost and bought

This is the part of phase 2 that generalises furthest beyond AI detection.

**At least nine substantive claims were withdrawn or corrected during the
sweep.** None of them reached a writeup. Every one was caught because enough raw
material was recorded for a second party to re-derive from, rather than reading
someone's conclusion.

The work was split across two sessions — one designing, one executing on GPU.
Eight of the withdrawn claims came from the executor, five from the designer.

**The asymmetry is only in how each surfaced.** The executor's errors were caught
when the designer re-derived numbers from the stored per-document outputs. The
designer's were caught when the executor read the write-up.

> **Neither mechanism catches the other's errors.** That is the real argument for
> two sessions rather than one careful one.

Three of the errors were **selection errors** — a favourable comparator chosen
rather than computed over everything available. The diagnosis that generalises:
*the comparator that comes to mind is the one that made the contrast salient
enough to report*. Selection happens before conscious comparison, so "check your
comparator" is unreliable advice to give yourself. Four scripts now exist that
enumerate every comparison by default, and **all four were written because a
hand-picked version of the same check failed first.**

Two were **power errors** — interpreting a statistic without first computing what
it could possibly resolve. The worst: an observed zero triple-overlap was read as
evidence *against* a stable set of hard documents, when the expected value under
pure independence was 0.008. Observing zero is exactly what independence
predicts. Absence of evidence read as evidence of absence, in a regime with no
power at all.

### A withdrawn claim does not stay withdrawn on its own

The single most instructive entry in the list: **"removing the artifact made the
model stronger"** had already been withdrawn — by the designer, that morning,
after catching the executor's version of the same claim. It was then written back
into the narrative by the same session that withdrew it, in the most quotable
sentence of the section.

> **A claim does not stay retracted by having been retracted.** It has to be
> recorded as withdrawn somewhere the writer will actually encounter it, or the
> original framing reasserts itself when the prose gets written. Prose reaches
> for the strong version, and the strong version is the one that was retracted.

The convention of marking withdrawn claims as withdrawn *in place*, rather than
deleting them, is what made this catchable. That is the reason to keep it — not
bookkeeping.

### What pre-registration actually bought

It earned its keep twice, in two different ways.

Once, a reporting rule fixed before any Task B number existed forced an
unanticipated outcome to be reported as *"satisfied on its letter, by a mechanism
it did not anticipate"* rather than as a clean confirmation.

Once, a stability threshold refused to let a claim be made on the Grok fold
because the three seeds disagreed — a result that would otherwise have read as a
clean three-for-three.

But there is a limit, and it was named rather than glossed:

> **The case machinery was valuable as a discipline, not as a measurement.** The
> verdict flips on a threshold chosen semi-arbitrarily. The robust quantity is
> the underlying continuous number, true under every threshold. Conflating those
> two contributions is how pre-registration gets oversold.

---

## Part 7: Reproducibility, in three layers

The honest answer is different at each layer, and lumping them together would be
misleading.

**Layer 1 — data and splits: EXACT.** The preparation script regenerates both
data files byte-identically, verified across randomised Python hash seeds and
again after the sweep finished. Anyone with the script and phase 1 gets exactly
these inputs.

**Layer 2 — analysis: EXACT, and without a GPU.** Every training record stores
the per-document output probabilities, the full confusion matrix, all pairwise
scores, and the complete hyperparameter block. **Every number in the findings
file was recomputed from those records by a second party without retraining
anything** — and that is how nearly all of the withdrawn claims were caught.

**Layer 3 — training: NOT REPRODUCIBLE, and demonstrated.** Not "probably won't
match" — measured. Re-running the identical cell with the identical seed and
identical hyperparameters, changing only whether gradient checkpointing was
enabled:

| | old path | new path |
|---|---|---|
| accuracy | 0.9347 | 0.9034 |
| training epochs | 10 (hit the cap) | **5 (early stop)** |

Recomputed activations differ in the last bits through floating-point
non-associativity. That perturbs gradients. In this regime the trajectories
separate at epoch 1 and never re-converge. A different GPU, CUDA version or
library version does the same thing.

**The environment was never captured.** No library, CUDA, or GPU versions appear
in any of the 24 records. Biggest gap, and the cheapest to have closed — a dozen
lines writing `pip freeze` and `nvidia-smi` per run.

The concrete form of this is a better argument than any general statement. The
executing session hit **two `transformers` v5 API removals during setup**, both
surfacing as tracebacks that named the library by version-specific behaviour, and
reconstructed one of the removed options by hand so the pre-registered recipe
would survive intact. Then recorded neither the transformers version, nor CUDA,
nor the GPU model.

> **The evidence for why environment capture matters was sitting in the traceback
> that made the fix necessary.**

### Which findings should survive a re-run

| finding | evidence | survives? |
|---|---|---|
| DeepSeek/Qwen are the closest pair | 4 independent measurements | **very likely** |
| nearest neighbour predicts the attractor | 6/6 cells, out of sample | **very likely** |
| unseen generators are called human below chance | 18/18 runs | **very likely** |
| all pairs separate ≥0.983 with newlines gone | all seeds | **likely** |
| the 15× variance collapse | 3 seeds per condition | **uncertain** |
| any specific number | — | **no** |

**Conclusions are reproducible; numbers are not.** Ordinal and directional
findings rest on agreement across many runs. Magnitude claims rest on three seeds.

---

## Where phase 2 ended

- **Phase 1 is sealed.** Two changes only: it moved into a subdirectory, and ten
  recovered prompt files were added. No data, result, or config file was altered.
- **exp01 is complete.** 24 of 24 runs plus the frozen control, zero failures.
- **Task C was not run.** Five of the eleven pre-registered predictions and the
  falsification line are neither passed nor failed. Recorded so the absence is
  never read as a result.

### The standing risk

Line breaks are the artifact that was found, and it was found by one reader
looking once. Nothing suggests it is the only one.

Strip it, and a fitted classifier will find the next-strongest surface feature —
sentence-length variance, comma rate, vocabulary richness, whatever the
generation prompt also forgot to match. A corpus generated by three models under
one prompt has generator fingerprints running all the way through it, and
discriminative training is extremely good at finding fingerprints.

That is the argument for keeping the readout unfitted, and for leave-one-
generator-out being the split that actually matters.

### What phase 3 needs

1. **More generators.** Three models give three pairwise distances, two of which
   differ by 0.003. The design cannot test a similarity *metric* at all — only
   its ordering. Four or five models would give 6–10 distances. This is a corpus
   change, not a training change.
2. **More seeds, 5–10.** For three distinct reasons that should not be merged:
   three seeds is a *lower bound* on true seed variance; overlap statistics buy a
   *different question* once counts rise; and the headline
   mistaken-for-human rate is simply not usable at n=3.
3. **Independent training corpora.** To separate "this document is intrinsically
   human-like" from "this is a document all models fail on for a shared reason".
   No number of restarts within one setup can separate those.
4. **Capture the environment in every record.** Pin versions.
5. **Keep the per-document probabilities.** A hard rule, not a suggestion — it is
   what made layer 2 exact and caught most of the withdrawn claims.
6. **Rate-match paragraph structure upstream**, per model and per length band.
   This is the honest fix, and the only one that produces a clean corpus rather
   than a corrected measurement.

---

## The lessons, condensed

1. **An unfitted model cannot overfit to an artifact it was never shown.** That
   is a real advantage of zero-shot readouts on a corpus you do not fully trust,
   and it was the main new result of the audit.
2. **Remove the artifact from the input rather than correcting for it in the
   analysis.** Normalising whitespace made the sanity check exact (a baseline of
   precisely 0.500) and cost nothing. Stratified analysis is a control; deleting
   the feature is a fix.
3. **Report the variance collapse, not the mean shift**, when the means overlap
   and the spreads do not.
4. **Compute what a statistic can resolve before interpreting its value.** The
   worst error of the session was a reasoning error, not an arithmetic one.
5. **A withdrawn claim needs to be recorded as withdrawn somewhere the writer
   will encounter it**, or it comes back.
6. **Two sessions with different roles catch different errors.** Neither
   mechanism catches the other's.
7. **Pre-registration is a discipline, not a measurement.** It stops
   clean-looking claims. It does not make an arbitrary threshold correct.
8. **Keep the raw per-document outputs.** Every claim caught in this phase was
   caught by re-deriving from them, and none would have been catchable from
   summary metrics alone.

---

## Glossary of terms new in this phase

**Stratified AUC** — measuring performance separately within bins of a
confounding variable, then combining. Inside a bin the confounder carries no
information about the label, so anything left is real signal.

**Normalisation (`norm`)** — here, collapsing all runs of whitespace to a single
space, which physically removes line breaks from the text before the model sees
it.

**Leave-one-generator-out** — training on some models' output and testing on a
model the classifier has never seen. The realistic deployment condition, because
new models keep appearing.

**Pairwise AUC** — separability restricted to two classes at a time. Insensitive
to how common each class is or how large its decision region is, which makes it
much more stable than overall accuracy.

**Seed** — the random number that determines weight initialisation and data
shuffling order. Same data, same settings, different seed, different result. With
a large model and a small dataset, that difference can be substantial.

**Pre-registration** — writing down predictions, thresholds, and reporting rules
before seeing results, in a file that does not change afterwards. It stops the
hypothesis from quietly becoming whatever the data showed.

**Gradient checkpointing** — a memory-saving technique that recomputes
intermediate values during training instead of storing them. Mathematically
equivalent, **not** bitwise identical — and in this project that difference alone
moved accuracy by 0.031 and changed when training stopped.
