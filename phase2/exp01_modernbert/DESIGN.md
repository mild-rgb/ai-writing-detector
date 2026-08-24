# exp01 — ModernBERT under leave-one-generator-out and whitespace normalisation

**Pre-registered before any model is trained.** Written so it can fail.

## Question

Phase 1 established two things: a frozen zero-shot LFM2.5 readout scores AUC
0.771 on long3, and `wc -l` scores 0.803 on the same data. Phase 1 §4 of
`NARRATIVE.md` then showed the LFM2.5 readout survives whitespace stratification
(0.771 -> 0.784) — it is not reading line breaks.

The open question is what happens when you *fit* a model to this corpus. Two
shortcuts are available to a fitted classifier and neither is prose:

1. **Formatting** — newline count, worth up to 0.887 on one fold (below).
2. **Generator fingerprints** — recognising grok/deepseek/qwen rather than
   machine prose in general.

This experiment removes both and reports what is left.

## Design — three tasks

**Revision, before anything ran.** The binary task trains at 2:1 AI:human, which
needed reweighting. The 4-way label set does not: every one of the 294 core
questions carries exactly one human answer and one from each generator, so the
classes are **294 / 294 / 294 / 294 — natively balanced, no reweighting**. The
imbalance was an artifact of collapsing three AI classes into one.

Shared: v6 only, question-level split seeded 20260822, **206 train / 88 test**,
`raw` vs `norm` (`re.sub(r'\s+',' ',t).strip()`), 3 seeds per cell.
**20 cells, 60 runs.**

### Task A — 4-way (2 cells)

Classes: `HUMAN`, `grok-4.6`, `deepseek-v4-pro`, `qwen3.8-max`.
Train 824 (206 x 4), test 352 (88 x 4), balanced throughout.

**The 4x4 confusion matrix is the headline**, not accuracy. Also report, for
each of the six class pairs, the binary AUC restricted to those two classes —
that is the pairwise separability table, and it is what says whether two models
are distinguishable at all.

### Task B — unseen-generator assignment (6 cells)

Train 3-way on `HUMAN` + two generators; test **only** the held-out generator's
88 documents. It must be assigned to one of the three seen classes, and which
one it picks is a direct measurement of which known model it resembles.

This is leave-one-generator-out reinterpreted for a multiclass label set. You
cannot ask a classifier to name a class it has never seen — but you can ask who
it mistakes it for, and that is the more informative question.

Report the **assignment distribution over the three seen classes**. The share
landing on `HUMAN` is the false-negative rate for a genuinely novel generator,
which is the deployment-realistic number this whole project is circling.

### Task C — binary (12 cells) — NOT RUN

**Cut by user decision on 2026-08-22, after Task A run 2, before any Task C run
started.** Measured cost was 16.1 h of a 27.7 h sweep; A and B were kept at full
3-seed scope. The cut is scope only — A and B run exactly as specced.

**Consequence, recorded so the absence is never read as a result:** Task C was
the only task producing a binary AI-vs-human AUC directly comparable to phase
1's 0.771 and to `wc -l`'s 0.803. **Predictions 7-11 and the 0.65 falsification
line are therefore UNRUN — neither passed nor failed.** Predictions 1-6 remain
live and evaluable. Task B's `HUMAN` share covers adjacent ground (a
false-negative rate for a novel generator) but is a different quantity and is
not a substitute.

The original specification follows, unchanged, for the record.

3 held-out generators x {seen, unseen} x {raw, norm}, as originally
pre-registered. Kept unchanged for continuity: test set byte-identical across
conditions within a fold, training AI count equalised at 412, class weights for
the 2:1 imbalance. Now secondary — Task B measures the fingerprinting question
better, because it names the confusion instead of only scoring it.



v6 only. Mixing prompt versions would confound generator with prompt.

* 294 questions that have **all three generators and a human answer**
* 1,176 documents: 882 AI (294 x 3) + 294 human
* Question-level split, seeded 20260822: **206 train / 88 test**
* Test set per fold: 88 AI (held-out generator) + 88 human = **balanced**
* Train AI count **equalised at 412** across conditions, so `seen` vs `unseen`
  is not confounded by training volume
* `docs.jsonl` sha256:`d8722c165e142b81`

**Length is matched by construction** — `target_words == human_words` for all
882 AI rows; medians 258 (AI) vs 270 (human). Length is not a usable shortcut
here, so it needs no separate control.

Splitting is by **question**, never by document: a question's human answer and
all three AI answers always land on the same side. Verified — zero question
leakage across all 12 cells.

## The artifact, in this data

Median newline count: grok 0, deepseek 2, qwen 6, **human 8**. All three
generators sit below the humans; only grok is separable by absence.

## Baselines, computed before training

`wc -l` on each test set:

| held-out generator | raw | norm |
|---|---|---|
| grok-4.6 | **0.887** | 0.500 |
| deepseek-v4-pro | 0.749 | 0.500 |
| qwen3.8-max | 0.585 | 0.500 |

Normalisation drives the baseline to exactly chance by construction. That is the
sanity check: any `norm` cell scoring well is not using whitespace, because
there is none left to use.

### Pairwise newline baselines for Task A

The multiclass analogue of `wc -l`, on the 352-document Task A test set.
Computed by the executing session and added here **before any model was trained
and before any test document was read** — it is a baseline, not a post-hoc
comparison chosen after seeing results.

| pair | raw | norm |
|---|---|---|
| HUMAN / grok | 0.887 | 0.500 |
| qwen / grok | 0.807 | 0.500 |
| HUMAN / deepseek | 0.749 | 0.500 |
| **deepseek / qwen** | **0.675** | 0.500 |
| deepseek / grok | 0.664 | 0.500 |
| HUMAN / qwen | 0.585 | 0.500 |

This sharpens prediction 3 in both directions. Under `raw`, formatting alone
already separates deepseek/qwen at 0.675 — so a model scoring near that is
plausibly adding nothing over whitespace, and the comparison must be made
against 0.675 rather than against 0.5. Under `norm` the floor is exactly 0.500
for all six pairs by construction, so **any** pairwise separability there is
genuinely prose.

Note also that grok is the outlier by formatting on every pair it appears in
(0.887, 0.807, 0.664), which is the surface fact behind the guess that
deepseek/qwen is the pair most likely to collapse.

Also report, on the identical test sets:

* **LFM2.5 frozen `a4_generic` margin** — zero-shot, unfitted, the phase 1
  instrument. This is the comparison that matters: an unfitted readout cannot
  overfit to artifacts, so it is the control for everything below.
* Newline count (above).

## Pre-registered predictions

Stated before any model is trained, so they can be wrong.

**Task A — 4-way**
1. `HUMAN` is the easiest class in every condition; the model/model distinctions
   are harder than the human/machine one.
2. **grok separates most cleanly from the other two generators under `raw`** and
   loses the most under `norm` — its median newline count is 0, against
   deepseek 2 and qwen 6, so it is the class with the strongest formatting tell.
3. **At least one generator pair is close to indistinguishable under `norm`** —
   pairwise AUC below 0.65. Best guess: deepseek/qwen, since grok is the outlier
   on every surface measure phase 1 recorded.
4. 4-way accuracy under `norm` lands well below the binary number. Four-way
   chance is 25%.

**Task B — unseen-generator assignment**
5. The held-out generator is assigned to the *other generators* far more than to
   `HUMAN` — i.e. an unseen machine still reads as machine. If instead a large
   share lands on `HUMAN`, the fingerprint hypothesis is confirmed in the
   strongest form and the binary numbers throughout phase 1 are inflated.
6. Held-out **grok goes to `HUMAN` most often** of the three, because the two
   generators it would be compared against are the two it least resembles.

**Task C — binary** (unchanged from the original registration)
7. `raw|seen` highest, ~0.85+.  8. `raw|unseen` drops most on grok.
9. `norm|seen` < `raw|seen`.  10. **`norm|unseen` lands below the LFM2.5
zero-shot 0.771.**  11. qwen is the hardest `unseen` fold.

**Falsification line**, unchanged: `norm|unseen` mean AUC < 0.65 means phase 1's
0.80 newline baseline was essentially all formatting and fingerprint, and this
corpus does not support a general machine-prose detector at this scale. A real
possible outcome, worth knowing.

**A note on prediction 3.** Two generators being indistinguishable is not a
failure of the experiment — it is a finding about the models, and arguably the
most interesting outcome available here. Report it as a result, never as a bug.

## Stated limitation: what 88 v 88 can adjudicate

Recorded **before any `norm` result existed**, so it cannot be read as post-hoc
excuse-making. Raised by the executing session; standard errors verified here
independently (Hanley-McNeil, n1 = n2 = 88).

| true AUC | SE | 95% CI |
|---|---|---|
| 0.650 | 0.041 | [0.569, 0.731] |
| 0.771 | 0.035 | [0.702, 0.840] |
| 0.800 | 0.034 | [0.734, 0.866] |
| 0.887 | 0.026 | [0.837, 0.937] |

**The test set is the binding constraint, not the training set.** A single fold
reading exactly 0.65 is statistically indistinguishable from anything in
0.57–0.73. Both pre-registered thresholds sit inside that band: the 0.65
falsification line, and the 0.65 near-indistinguishability criterion in
prediction 3.

**The 3-fold mean helps less than it looks.** If the folds were independent the
SE would shrink by sqrt(3) to about ±0.047. They are not independent — all three
Task C folds share the **identical 88 human documents**, and only the AI arm
changes. That shared arm induces positive correlation, so the true interval on
the 3-fold mean is *wider* than ±0.047. The actual correlation is computable
from per-document scores once runs land, and should be computed rather than
assumed.

**Consequence for reporting, fixed now.** The thresholds stand exactly as
pre-registered — they are not moving. But if `norm|unseen` lands near 0.65 the
honest report is *"below the line, but the CI spans it"*, not *"falsified"*. At
0.55 or 0.85 the geometry is entirely adequate to speak plainly. **This
experiment can cleanly detect a large effect; it cannot cleanly adjudicate a
0.05 difference.**

**Seed count is a lower bound on seed variance.** 618 training documents with a
395M model sits in the known-unstable regime for encoder fine-tuning — roughly
3–4x smaller than GLUE RTE, the canonical case where BERT-large fine-tuning is
seed-unstable and where the literature remedy (Dodge et al. 2020, Mosbach et al.
2021) is 5–10 random restarts. We have 3. Three uniform seeds beats a mixed
3-and-7, so this is not changing mid-sweep — but the reported 3-seed spread
should be read as a **lower bound** on true seed variance, and more restarts is
an exp02 item.

**The frozen control may carry more of the argument than "baseline" suggests.**
The LFM2.5 readout scores 0.771 with zero training, and phase 1 found it
artifact-immune precisely because it was never fitted. If the fine-tuned 395M
model does not clearly clear 0.771 under `norm`, one honest reading is not "the
corpus has no signal" but **"this corpus does not support fine-tuning at this
scale, while an unfitted readout still works."** Those are different conclusions
with different implications for phase 3, and the design already contains the
comparison needed to separate them.

## Methodological finding: the design rests on its robust measurements

Recorded after the Task A `raw` seeds but **before any `norm` result existed**.
It is an observation about metric stability, not about the outcome.

Re-running `4way|raw seed=0` under a changed code path (gradient checkpointing,
adopted after a CUDA OOM) moved the metrics unequally:

| quantity | movement | threshold attached? |
|---|---|---|
| accuracy | **0.031** | none |
| macro-F1 | 0.033 | none |
| 5 of 6 pairwise AUCs | ≤ **0.0015** | prediction 3 |
| deepseek/qwen pairwise AUC | 0.0127 | prediction 3 |

**The instability is concentrated in the quantity that has no threshold attached
to it.** Nothing in the pre-registration depends on accuracy. Prediction 3 rests
on the six pairwise AUCs, five of which are near-immovable, and the sixth moves
0.0127 against a ±0.081 confidence interval that this geometry allows anyway.

That is partly luck and should be named as such — the obvious alternative design,
reporting 4-way accuracy as the headline, would have rested its claim on the
fragile measurement. It is also a reason to report **pairwise separability as
Task A's primary metric**, with accuracy secondary, and it generalises past this
experiment.

**Containment, not comparison.** The old-path run is interior to the new path's
seed range on both quantities:

```
accuracy        0.9034 <= 0.9347 <= 0.9460
deepseek/qwen   0.9615 <= 0.9742 <= 0.9817
```

The honest phrasing is **"within seed noise"**, never "identical" — recomputed
activations differ in the last bits through floating-point non-associativity,
which perturbs gradients, and in this regime trajectories separate at epoch 1 and
do not re-converge. The same seed early-stopped at epoch 5 on one path and hit
the 10-epoch cap on the other.

## Model

**ModernBERT-large (395M).** 8,192 context covers every document (max 1,885
words). Class weights for Task C's 2:1 imbalance; Tasks A and B need none — they
are balanced by construction.

An earlier version of this spec ran base (149M) first and escalated to large only
if base cleared 0.65 on `C norm|unseen`. **That gate is removed** — an L4 takes
large comfortably, so there is no reason to spend runs establishing permission to
use it. Any base cells already completed are kept and labelled as base; large is
primary.

Capacity note, on the record before results: 395M fitted on 618–824 training
documents is a lot of model for a small corpus, and the risk is overfitting, not
memory. Early stopping is patience 3 on val, val carved from 15% of train
*questions*. **The stopping epoch is reported alongside every metric** — runs
consistently halting at epoch 1–2 is itself a finding and must not be smoothed
over by adjusting the recipe.

## Reporting

* **Task A:** 4x4 confusion matrix, per-class F1, six pairwise AUCs, 3-seed spread.
* **Task B:** assignment distribution per held-out generator, and the `HUMAN` share.
* **Task C:** per-cell AUC, balanced accuracy, 3-seed spread; headline is the
  `norm|unseen` 3-fold mean **with** its spread, next to `wc -l` and LFM2.5
  zero-shot on the same 176 documents.

Spread matters as much as mean: tight means general, wide means we are detecting
individual models and averaging over luck.

## Files

* `prep.py` — deterministic build. Seeds are string-derived, so it reproduces
  across interpreters; an earlier version used `hash()` on a tuple and did not.
* `docs.jsonl` — 1,176 documents, sha256:`d8722c165e142b81`, `text` + `text_norm`
* `splits.json` — 20 cells, sha256:`dfc459701c2fbbb5`, explicit id lists
