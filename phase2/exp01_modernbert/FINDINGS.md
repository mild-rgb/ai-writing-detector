# exp01 — findings log

Outcomes as they land. **`DESIGN.md` is the pre-registration and does not change
once results exist**; this file is where results go. Kept separate so the
predictions cannot be edited to match what happened.

Scope: Task C was cut by the user after Task A run 2. Predictions 7-11 and the
0.65 falsification line are **UNRUN — neither passed nor failed**.

---

## Status board

| # | prediction | status |
|---|---|---|
| 1 | HUMAN is the easiest class | **CONFIRMED** — raw and norm |
| 2 | grok separates most cleanly under `raw`; loses most under `norm` | **RIGHT FOR THE WRONG REASON** — see below |
| 3 | ≥1 generator pair near-indistinguishable under `norm` (<0.65) | **FAILED, decisively** |
| 4 | 4-way accuracy under `norm` well below binary | **UNEVALUABLE** — references cut Task C |
| 5 | held-out generator assigned to other generators >> HUMAN | **satisfied on its letter, mechanism not anticipated** |
| 6 | held-out grok goes to HUMAN most often | **UNRESOLVED** — mean and median disagree |
| 7-11 | Task C binary | **UNRUN** |

---

## Task A, `4way|raw`, 3 seeds — complete

Chance is 0.250. Artifact fully available in this cell.

```
accuracy  0.922 ±0.021      macro-F1  0.922 ±0.022
stopping epochs [10, 5, 8]  -> 1 cap, 2 early-stop
```

| class | F1 | recall | precision | times predicted |
|---|---|---|---|---|
| HUMAN | 0.965 | 0.977 | 0.952 | 90.3 |
| grok-4.6 | 0.946 | 0.913 | **0.984** | 81.7 |
| qwen3.8-max | 0.897 | 0.856 | 0.946 | 79.7 |
| deepseek-v4-pro | 0.882 | 0.943 | **0.827** | 100.3 |

| pair | model | newline-only (raw) |
|---|---|---|
| qwen / grok | 0.9997 | 0.807 |
| HUMAN / qwen | 0.9997 | 0.585 |
| HUMAN / deepseek | 0.9995 | 0.749 |
| HUMAN / grok | 0.9993 | 0.887 |
| deepseek / grok | 0.9958 | 0.664 |
| **deepseek / qwen** | **0.9783 ±0.0150** | 0.675 |

**Prediction 1 — CONFIRMED.** Only 2 of 264 human documents were assigned to any
generator across all three seeds.

**Prediction 2, first half — CONFIRMED.** grok is the best-separated generator
by F1 (0.946 vs 0.897, 0.882) and every grok-involving pair is more separable
than the one pair without it. Initially misread as failed by fixating on the
grok→deepseek 6.7/88 cell; the marginals show that is a **deepseek precision**
effect — deepseek is predicted 100.3 times against a true 88 and carries
precision 0.827, absorbing false positives from both other generators — while
grok has the highest precision of any class including HUMAN.

**Prediction 3 — direction confirmed, magnitude not.** deepseek/qwen is the
lowest pair in all three seeds independently, but 0.978 is far from 0.65.

### Caveat that must travel with the deepseek/qwen number

±0.0150 is a **3-seed half-range, not a confidence interval**. Hanley-McNeil at
88 v 88 gives **±0.081**, five times wider. Seed spread and sampling error are
different quantities. Every other pair is saturated at ≥0.996 where the interval
is much tighter, so the qualitative conclusion is safe — but deepseek/qwen is the
one number carrying real sampling uncertainty, and it is also the number
prediction 3 turns on. Both facts must appear together or neither is honest.

---

## Task A, `4way|norm`, 3 seeds — complete. **A COMPLETE.**

Sanity check passed: **0 of 1,176 documents contain a newline in `text_norm`.**
The artifact is provably absent, and all six newline baselines are 0.500 by
construction. Anything measured here is not whitespace.

```
          accuracy (3 seeds)              spread    epochs
  raw     0.9034  0.9176  0.9460          0.0426    [10, 5, 8]
  norm    0.9403  0.9403  0.9432          0.0028    [10, 10, 9]
```

| pair | newline raw | model raw | newline norm | model norm | delta |
|---|---|---|---|---|---|
| HUMAN / grok | 0.887 | 0.9993 | 0.500 | 0.9999 | +0.0006 |
| HUMAN / deepseek | 0.749 | 0.9995 | 0.500 | 0.9998 | +0.0003 |
| HUMAN / qwen | 0.585 | 0.9997 | 0.500 | 0.9996 | −0.0001 |
| qwen / grok | 0.807 | 0.9997 | 0.500 | 0.9997 | +0.0000 |
| deepseek / grok | 0.664 | 0.9958 | 0.500 | 0.9966 | +0.0008 |
| **deepseek / qwen** | 0.675 | 0.9783 | 0.500 | **0.9828** | +0.0045 |

**The headline: removing the artifact did not degrade anything.** Every pair
separates at ≥0.983 with whitespace provably unavailable, and every per-class F1
improves (HUMAN +0.014, deepseek +0.021, qwen +0.025, grok +0.018). This is the
first measurement in the experiment that whitespace cannot explain.

### The robust claim is the variance collapse, not the accuracy gain

The mean rose 0.922 → 0.941, but **the accuracy ranges overlap**: raw's best seed
(0.9460) exceeds norm's best (0.9432). On the mean, `norm` wins; on best-case, it
does not. That difference is not resolvable at 3 seeds.

What is unambiguous is stability. Range width **0.0426 → 0.0028, a 15x
collapse**, with no overlap in dispersion and the effect visible in training too
— stopping epochs went [10, 5, 8] to [10, 10, 9].

**Report the 15x, not the +0.019.**

### Prediction 3 — FAILED, decisively

Predicted at least one generator pair below 0.65 pairwise AUC under `norm`, best
guess deepseek/qwen. deepseek/qwen **is** the lowest pair, in every seed, as
predicted — but at **0.983 ±0.003**, not below 0.65. Wrong by a margin far
outside the ±0.081 the geometry allows. No generator pair is anywhere near
indistinguishable. The three models are individually identifiable from prose
alone.

### Prediction 2 — the middle case fired

Per the rule pre-committed above, before this cell existed. F1 delta raw→norm:
qwen +0.025, deepseek +0.021, **grok +0.018 — the smallest gain, but a gain**.
grok loses nothing, and remains the most separable generator under `norm`
(F1 0.963; every grok-involving pair ≥0.9966).

**Prediction right, rationale WRONG.** The stated reason was grok's zero median
newline count. With newlines gone grok is still the most separable generator, so
it is distinguishable by something that is not formatting, and the `raw` result
was right by coincidence. **Not a confirmation.**

### Prediction 4 — unevaluable, listed apart from 7-11

It compares 4-way `norm` accuracy against "the binary number", which was Task C.
C is cut, so there is no in-experiment binary figure. Unrun by scope, like 7-11,
but noted separately because it is a Task A prediction that happens to reference C.

### Interpretation — flagged as interpretation, not measurement

The premise was that a fitted model would lean on whitespace and collapse when it
was removed. The opposite happened. One reading: whitespace was a **noisy**
feature — genuinely predictive in aggregate (0.585–0.887 standalone) but
inconsistent enough per document that a fitted model partly chased it and paid in
variance. Remove it and the model is forced onto prose, which is both stronger
and far more stable. That is the fitted-model counterpart to phase 1's finding
that the frozen readout was artifact-immune.

**Caveat that must travel with it:** this is Task A — generator *identification*
with every class seen in training. It says nothing about the unseen case. A model
that identifies known generators from prose at 0.98+ could still assign a novel
generator to HUMAN at a high rate; those are compatible. **Task B is where the
fingerprinting question is actually tested**, and B's HUMAN share is the
deployment-realistic number.

±0.003 is a 3-seed half-range, not a CI. Hanley-McNeil at 88 v 88 remains ±0.081
and still applies to deepseek/qwen's 0.983.

---

## Task B, `assign|deepseek-v4-pro`, raw complete — **CASE 3 CLAIMED**

Held-out deepseek, 88 documents, assigned across `HUMAN` / qwen / grok. Verified
here by re-deriving every assignment from stored probabilities rather than
reading a reported field.

| seed | HUMAN | qwen | grok | h | skew | case |
|---|---|---|---|---|---|---|
| raw 0 | 0.000 | 0.886 | 0.114 | 0.000 | 0.886 | 3 |
| raw 1 | 0.011 | 0.875 | 0.114 | 0.011 | 0.885 | 3 |
| raw 2 | 0.023 | 0.852 | 0.125 | 0.023 | 0.872 | 3 |
| norm 0 | 0.023 | 0.943 | 0.034 | 0.023 | **0.965** | 3 |

**Case 3 fired in all three raw seeds -> CLAIMED**, under the rule pre-committed
before any B number existed. Classified by the executing session's
`classify_b.py`, written before the numbers and applying the thresholds
unmodified.

### The deployment-realistic number

**h = 3/264 pooled = 0.011, Wilson 95% CI [0.004, 0.033], against a chance rate
of 0.333.** A generator absent from training is mistaken for a human roughly
**1 document in 88**.

Caveat on interpretation: this is a 3-way forced choice with HUMAN as one of
three classes, so h is a false-negative rate only if the two generator classes
are collapsed to "machine". That collapse is legitimate, but the number is not
identical to a binary detector's FNR measured directly. Task C would have given
that, and Task C is cut.

### Prediction 5 — satisfied on its letter, NOT confirmed

Per the pre-committed case-3 rule. Prediction 5 said the held-out generator would
go to the other generators rather than HUMAN. True. It did not anticipate that
the documents would go almost entirely to **one** of them. Reported the same way
as prediction 2's middle case.

### The attractor is consistent — and it cross-validates Task A

Same attractor, **qwen, in all three raw seeds and in the first norm seed**.
Condition-consistency pending two more norm seeds.

This is the "genuine model-similarity finding" branch of the case-3 rule, and it
has corroboration from a **structurally independent design**:

* **Task A** (all classes seen, 4-way): deepseek/qwen was the lowest pairwise
  AUC — 0.978 raw, 0.983 norm — the single hardest pair, with directional
  confusion qwen->deepseek 10/88.
* **Task B** (deepseek removed from training entirely, 3-way): deepseek's
  documents land on qwen ~87% of the time.

Two label sets, two training regimes, same conclusion: **deepseek and qwen are
the closest pair in this corpus.** That is a finding about the models, not about
the detector.

### Open: is it symmetric?

This fold answers "where does deepseek go" — to qwen. The `assign|qwen3.8-max`
fold asks the converse and is six runs out. Task A's directional confusion
(qwen->deepseek 10/88 against deepseek->qwen 4/88) hints it may **not** mirror
cleanly. Asymmetric attraction would itself be a result, and is covered by no
pre-registered prediction — noted here before that fold runs.

---

## Task B, `assign|deepseek-v4-pro|norm` — CASE 3 CLAIMED AGAIN

| seed | HUMAN | qwen | grok | h | skew |
|---|---|---|---|---|---|
| 0 | 0.023 | 0.943 | 0.034 | 0.023 | 0.965 |
| 1 | 0.091 | 0.693 | 0.216 | 0.091 | 0.762 |
| 2 | 0.045 | 0.943 | 0.011 | 0.045 | 0.988 |

**Attractor qwen in 3/3 raw seeds AND 3/3 norm seeds.** The case-3 consistency
follow-up is fully satisfied across seeds *and* conditions, so by the
pre-committed rule this is a **genuine model-similarity finding**. It is not a
formatting artefact: under `norm` the newline feature is provably absent and the
attraction is undiminished.

### The raw→norm rise in h — direction only

h was higher under `norm` in every seed — raw [0.000, 0.011, 0.023], norm
[0.023, 0.045, 0.091] — but **the ranges touch** (raw max 0.023 = norm min 0.023)
and the whole effect is **11 documents**. Not resolvable at this n.

Wilson intervals and a Fisher p were computed and then **withdrawn**: pooling
three seeds over the same 88 documents is not 264 independent trials, and any
interval built that way is too narrow. Seed-as-unit is the defensible treatment.
This error was made by both sessions and caught on re-derivation.

### Open question this experiment CANNOT answer

Whether a stable subset of machine documents reliably evades detection.

Per-seed HUMAN counts are too low (0–8 of 88) for overlap statistics to have
power. Expected triple-overlap **under independence** is **0.008**, so observing
zero is uninformative — it is what independence predicts, not evidence against a
hard core. The pairwise-overlap statistic, which does carry some signal, gives
**2 observed against 0.61 expected, Monte Carlo P = 0.118** — if anything weakly
suggestive of clustering, i.e. the *opposite* of "no stable subset".

An earlier draft claimed "the false negatives are seed-idiosyncratic, not a
stable hard core". That claim is **withdrawn**. It is a strong deployment-facing
sentence and the data is nowhere near it.

**exp02, now well-specified:** the discriminating statistic is pairwise overlap
rate against its independence expectation, and it gains power as counts rise. Ten
seeds at the norm fold's rate would give expected pair-overlap ≈7–8 rather than
0.61 — enough to detect a moderate hard core. Note this is a case where more
seeds buys a **different question**, not a tighter interval on the existing one.

### Symmetry — preliminary, 1 seed, NOT claimed

`assign|qwen3.8-max|raw` seed 0: h = 0.011, skew 0.989, **deepseek 0.977**. If it
holds, attraction is symmetric — deepseek→qwen 0.871, qwen→deepseek 0.977 — with
qwen's pull toward deepseek the stronger of the two, matching Task A's
directional confusion (qwen→deepseek 10/88 vs deepseek→qwen 4/88). Two more seeds
before any claim.


---

## Task B, `assign|qwen3.8-max|raw` — SYMMETRY CONFIRMED

```
  held out deepseek -> qwen      held out qwen -> deepseek
   s0  0.886                      s0  0.977
   s1  0.875                      s1  0.989
   s2  0.852                      s2  0.784
```

Case 3 in all three seeds, attractor deepseek 3/3. **The attraction runs both
ways: deepseek and qwen are mutual nearest neighbours.**

That relationship has now appeared in **four structurally independent
measurements** — Task A pairwise AUC (lowest of six, both conditions), Task A
confusion (directional), Task B deepseek→qwen, Task B qwen→deepseek. Different
label sets, different training regimes, both directions of leave-one-out. This is
as much corroboration as the design can produce.

### The headline needed qualifying: h is seed-unstable

**qwen seed 2 gives h = 0.205 — 18 of 88 — against 0.011 and 0.011 for its two
siblings.** A factor of ~19 within one cell.

The **case classification is unaffected**: even in seed 2, 98.6% of the
generator-directed mass goes to deepseek, so the nearest-neighbour finding is
untouched. What moves is the deployment-relevant number.

Across all **9 completed B runs**: h spans **0.000 to 0.205**, median 0.023, one
run above 0.10, all nine below the 0.333 chance rate.

| claim | survives? |
|---|---|
| "a novel generator is not mistaken for human at chance rates" | **yes** — all nine below 0.333, and seed 2's interval still excludes chance |
| "h ≈ 0.01–0.05" | **no** — one run in nine lands at 0.205 |

The honest statement is **"h is consistently below chance but its magnitude is
seed-unstable, ranging 0.000–0.205 across nine runs."** A practitioner reading
0.011 as the expected false-negative rate would be badly misled.

This is the Dodge/Mosbach instability landing in the number the project cares
most about — which makes 5–10 seeds a question of **usability of the estimate**,
not merely tightness of an interval. Third distinct justification for more seeds
on the exp02 list, and the strongest of the three.

### Power check now automated

The independence-expectation check runs on every completed cell in
`classify_b.py` rather than depending on anyone remembering it. `qwen|raw`:
counts [1, 1, 18], observed pair-overlap 2, expected 0.41 → flagged
**UNINFORMATIVE**. Closest any fold has come to having power; seed 2's 18
documents nearly reach it.

---

## Task B, `assign|qwen3.8-max|norm` — symmetry survives normalisation

Case 3 in all three seeds, attractor deepseek 3/3 (0.932, 0.943, 0.966).

**Both directions, both conditions, twelve runs: deepseek and qwen are mutual
nearest neighbours, undiminished by whitespace removal.** Finding closed.

### First actual evidence on the stable-hard-core question

Overlap statistics across all four completed cells, 50k Monte Carlo each,
verified independently by both sessions:

| cell | counts | obs 3-way | P(≥) | obs 2-way | P(≥) |
|---|---|---|---|---|---|
| deepseek raw | [0, 1, 2] | 0 | 1.000 | 0 | 1.000 |
| deepseek norm | [2, 8, 4] | 0 | 1.000 | 2 | 0.112 |
| qwen raw | [1, 1, 18] | 0 | 1.000 | 2 | 0.040 |
| **qwen norm** | [5, 5, 2] | **1** | **0.006** | **3** | **0.006** |

`qwen|norm` clusters on both statistics: one document called HUMAN by all three
seeds against an expectation of 0.006, and three by ≥2 seeds.

**Multiplicity:** 8 tests examined (4 cells × 2 statistics). Bonferroni over 4
gives 0.023; over 8 gives 0.047. Both survive at 0.05, so the result is slightly
more robust than first stated — and the two statistics within a cell are nested
(obs3 ≥ 1 implies obs2 ≥ 1), making the 8-test correction conservative.

**Effect size, stated plainly: this is ONE document out of 88.** The claim
available is "there may exist a small number of qwen outputs that consistently
read as human under normalisation" — not a characterised hard core.

**This inverts the withdrawn claim rather than resurrecting it.** An earlier draft
asserted "no stable subset"; that was withdrawn as unreachable. The evidence now
leans weakly toward *presence*, consistent with the direction `deepseek|norm`'s
pair statistic hinted at. Wrong to claim absence, and the data tilts the other way.

### The binding caveat: seeds are not exchangeable

These p-values assume the three seeds are exchangeable draws. They are not — they
share training data and differ only in initialisation and shuffling. So a
document all three call human might be **intrinsically human-like**, or might be
one all three models fail on **for a shared reason**. This design cannot separate
those, and no amount of extra seeds within this setup would. Separating them needs
independent training corpora, not more restarts.

### Automated power check was too narrow, and nearly buried this

The rule flagged a cell UNINFORMATIVE when expected *pair* overlap < 1 — and
printed UNINFORMATIVE on `qwen|norm` while that cell held a triple-overlap at
expectation 0.006. Fixed to evaluate both statistics and declare uninformative
only when **neither** has power.

Method note: automating the check was right; implementing it on one statistic was
not. Second instance today of *checked the thing I thought to check, not the thing
that mattered*.

---

## Task B, `assign|grok-4.6|raw` — **UNSTABLE, no case claimed**

```
  seed 0  HUMAN 0.023  deepseek 0.909  qwen 0.068   skew 0.930  -> case 3
  seed 1  HUMAN 0.000  deepseek 0.807  qwen 0.193   skew 0.807  -> case 3
  seed 2  HUMAN 0.057  deepseek 0.625  qwen 0.318   skew 0.663  -> case 1
```

Seeds disagree → **UNSTABLE under the pre-committed rule, no case claimed.** The
rule refused a claim that would otherwise have read as a clean three-for-three.

### Report the skew, not the case label

The verdict flips on the threshold: at 0.60 or 0.65 all three seeds are case 3;
at 0.70 and above it is unstable. Pre-commitment means the boundary was not
chosen after seeing data — that worked — but **it does not make 0.70 correct**.

**The robust quantity is the continuous skew, 0.663–0.930, true under every
threshold.** The case label is bookkeeping.

General lesson: the case machinery earned its keep as a **discipline** — it
stopped a clean-looking claim — not as a **measurement**. Conflating those is how
pre-registration gets oversold.

### What survives

1. **deepseek wins in all three grok seeds.** The nearest-neighbour prediction
   from Task A's prior-free pairwise AUCs is correct in **all six completed
   cells** — ordinal, out-of-sample, across two structurally different designs.
2. Confidence of that win varies 0.663–0.930; no case claimed.
3. **Prediction 6 UNRESOLVED.** h by fold: deepseek [0.000, 0.011, 0.023], qwen
   [0.011, 0.011, 0.205], grok [0.023, 0.000, 0.057]. Mean says qwen highest,
   median says grok — disagreeing entirely because of qwen seed 2. No direction
   reported.
4. Overlap check: counts [2, 0, 5], both statistics uninformative.

### Instability is bimodal and unexplained

```
  qwen|norm       range 0.012      deepseek|norm   range 0.226
  deepseek|raw    range 0.014      grok|raw        range 0.268
  qwen|raw        range 0.014
```

Four of five cells are ≤0.014 or ≥0.226 with **nothing in between**. A
distance-from-nearest-neighbour explanation was proposed and **withdrawn**:
`deepseek|raw` and `deepseek|norm` are the same generator at the same
nearest-neighbour distance (0.9828, the closest pair in the corpus) and differ by
16x in stability, varying only in whitespace normalisation. With n=5 cells the
bimodality itself should not be leaned on.

---

## Unresolved tension: normalisation does not consistently stabilise

Every comparable raw/norm pair, enumerated mechanically rather than selected:

| quantity | raw range | norm range | effect |
|---|---|---|---|
| Task A accuracy | 0.0426 | 0.0028 | norm **more** stable 15x |
| Task A macro-F1 | 0.0443 | 0.0035 | norm **more** stable 13x |
| Task B h, qwen | 0.1932 | 0.0341 | norm **more** stable 6x |
| Task B skew, qwen | 0.0143 | 0.0120 | no clear change |
| Task B h, deepseek | 0.0227 | 0.0682 | norm **less** stable 3x |
| Task B skew, deepseek | 0.0143 | 0.2256 | norm **less** stable 16x |

**Three more stable, two less, one unchanged.** "Normalisation stabilises" is the
Task A result generalised past its evidence and is **not a finding**. The claim
is **"normalisation stabilised Task A"** — 15x on accuracy, 13x on macro-F1,
corroborated by stopping epochs [10,5,8] → [10,10,9]. B-fold effects are reported
per fold; the tension is unresolved.

Caveat within the caveat: the qwen-h row reads as "more stable under norm" only
because the raw condition carried the 0.205 outlier. Even that direction rests on
a single seed.

---

## Pre-committed: how to report a right-for-wrong-reasons outcome

Written **before `4way|norm` existed**, so the reporting rule cannot be chosen
after seeing which way it went.

Prediction 2's stated rationale is that grok's median newline count of 0 (against
deepseek 2, qwen 6) is the strongest formatting tell. `norm` removes newlines
entirely. Three outcomes, and each is reported differently:

* **grok's separability drops most under `norm`** — prediction and rationale both
  confirmed.
* **grok stays the most separable generator under `norm`** — the prediction was
  right and **the rationale was wrong**. grok would then be distinguishable by
  something other than formatting, and the `raw` result was right by coincidence.
  This must be reported as a *failed rationale with a passing prediction*, not as
  a confirmation.
* **grok stops being most separable but not by the most** — partial; report the
  ordering change and the magnitude separately.

The second case is the most interesting of the three and the easiest to
mis-report as a win. It is named here so it cannot be.

---

## Pre-committed: how to report Task B, including the unanticipated case

Written **before any Task B number existed** — first B cell was ~80 minutes out
when this was fixed. Task A's result (all three generators individually
identifiable from prose at ≥0.983 with the artifact provably absent) makes a
third outcome likely that neither prediction 5 nor 6 anticipated, so it is given
a reporting rule here rather than a story afterwards.

### Two statistics, defined now

Task B's test set is 88 documents of a single held-out class, assigned across
three seen classes. There is no AUC and no confusion matrix — the assignment
distribution *is* the result.

* **h** = share assigned to `HUMAN`
* **skew** = larger generator share ÷ (sum of both generator shares).
  0.50 = even split between the two seen generators, 1.00 = all on one.

### Precision, so thresholds respect the geometry

At n = 88 the binomial 95% CI half-width is ±0.098 at p = 1/3 and ±0.104 at
p = 0.50. **The smallest gap from chance this can resolve is ~0.098.** Any claim
resting on a difference smaller than that is not supported, and thresholds below
are set outside it deliberately.

### The four cases

| case | criterion | meaning |
|---|---|---|
| **1. Machine, spread** | h < 0.33 **and** skew < 0.70 | An unseen machine still reads as machine. **Prediction 5 holds.** Good for detection. |
| **2. Human** | h ≥ 0.50 | Fingerprint hypothesis in its strongest form. The 4-way result is attribution of *known* models, not machine-prose detection, and phase 1's binary numbers are inflated. |
| **3. Confident wrong generator** | h < 0.33 **and** skew ≥ 0.70 | **Unanticipated.** Good for detection, bad for attribution. Neither prediction 5 nor 6 covers it: 5 is satisfied on the letter (not HUMAN) while the mechanism is different from what it assumed. |
| **4. Ambiguous** | anything else | Reported as ambiguous. Not assigned to whichever neighbouring case is more convenient. |

**A case is only claimed if it holds in all three seeds.** If seeds disagree on
which case fired, the result is reported as unstable and the per-seed
distributions are given, with no case claimed.

### Reporting rule for case 3 specifically

If it fires, the follow-up question is whether the wrong generator is
**consistent** — the same one across all three seeds, and the same one in `raw`
and `norm`. If it is, that is a **model-similarity finding**, and worth stating
as such: it says the held-out generator's prose resembles that specific model,
which is information about the models rather than about the detector. If the
identity of the wrong generator varies by seed or by condition, it is
concentration without meaning, and must be reported as such rather than dressed
up as similarity.

Case 3 does **not** count as a confirmation of prediction 5. Prediction 5 is
reported as "satisfied on its letter, but by a mechanism it did not anticipate",
in the same manner as prediction 2's middle case.

### Prediction 6

"Held-out grok goes to HUMAN most often of the three." Evaluated by comparing h
across the three folds. Given the ±0.098 resolution, grok's h must exceed both
others by more than that gap in all three seeds to count as confirmed; a smaller
ordering difference is reported as *ordering consistent with the prediction but
not resolvable*.

### Reporting format

Per-seed distributions, never only the mean — n = 88 across three seeds is noisy
and the spread on h is the whole story.

## Reproducibility assessment

Three layers, very different answers.

### Layer 1 — data and splits: EXACT

`prep.py` regenerates `docs.jsonl` (`d8722c165e142b81`) and `splits.json`
(`dfc459701c2fbbb5`) byte-identically — re-verified after the sweep completed, and
previously verified across randomised `PYTHONHASHSEED` from a separate directory.
Anyone with `prep.py` and `phase1/` gets exactly this experiment's inputs.

Not true of the first version: it seeded from `hash()` on a tuple of strings and
produced different `seen` cells per interpreter. Caught before any run, fixed to a
string-derived seed, regenerated.

### Layer 2 — analysis: EXACT, and without a GPU

Every record stores per-document `probs`, the full confusion matrix, all pairwise
AUCs, `epochs_run`, and the complete `hp` block (verified identical across all 24
records). The frozen control stores per-document margins for both text fields.

**Every number in this file was recomputed from those records by a second party
without retraining anything**, and six errors were caught that way. The whole
analysis is redoable from `results/` alone.

### Layer 3 — training: NOT REPRODUCIBLE, and demonstrated

Not "probably won't match" — measured. Re-running `4way|raw seed=0` under
gradient checkpointing, identical data, seed and hyperparameters:

| | old path | new path |
|---|---|---|
| accuracy | 0.9347 | 0.9034 |
| epochs | 10 (cap) | **5 (early-stop)** |

Recomputed activations differ in the last bits through floating-point
non-associativity; that perturbs gradients; trajectories separate at epoch 1 and
never re-converge. A different GPU, CUDA or library version does the same.

**Seed variance exceeds most effects of interest:**

```
  4way|raw accuracy   0.9034 - 0.9460     range 0.043
  Task B h, 18 runs   0.000  - 0.205
```

So even a perfect environment reproduction would not match run-for-run.

**Environment was not captured.** `transformers`, `torch`, Python, CUDA and GPU
versions are absent from every record; known only from session correspondence
(Colab L4 23GB, bf16, transformers 5.15.1). **Biggest gap, and the cheapest to
have closed** — a dozen lines writing `pip freeze` and `nvidia-smi` per record.

### Which findings should survive a re-run?

| finding | evidence | survives? |
|---|---|---|
| deepseek/qwen closest pair | 4 independent measurements | **very likely** |
| nearest-neighbour predicts attractor | 6/6 cells, out-of-sample | **very likely** |
| h below chance | 18/18 runs vs 0.333 | **very likely** |
| pairs separate ≥0.983 under norm | all seeds | **likely** |
| Task A 15x variance collapse | n=3 per condition | **uncertain** |
| any specific number | — | **no** |

Ordinal and directional findings rest on agreement across many runs. Magnitude
claims rest on three seeds and should not be expected to replicate.

### exp02 requirements

1. Capture environment in every record (`pip freeze`, CUDA, GPU, driver).
2. Pin library versions, not "whatever Colab ships".
3. Keep per-document `probs` — hard rule; it is what made layer 2 exact.
4. Record the code path (gradient checkpointing on/off) per run.
5. More seeds — 5-10. Three cannot support a variance claim.

## Method notes accumulating for exp02

1. **Report pairwise separability, not accuracy, as Task A's headline.** A code
   path change moved accuracy 0.031 while 5 of 6 pairwise AUCs moved ≤0.0015.
   The instability sat in the quantity with no threshold attached to it.
2. **Keep full `probs` and `confusion` in every record**, not just derived
   metrics. Two reporting errors this session were caught by re-deriving from the
   mirror rather than reading a summary; neither would have been catchable from
   derived numbers alone.
3. **3 seeds is a lower bound on seed variance.** 618 training documents with a
   395M model is the Dodge/Mosbach unstable regime; the literature remedy is
   5-10 restarts.
4. **Compute what a statistic can resolve before interpreting its value.** The
   worst error of the session was a reasoning error, not an arithmetic one: a
   zero triple-overlap was read as evidence against a hard core when its
   expectation under independence was 0.008. Absence of evidence read as evidence
   of absence, in a regime with no power. Check the expectation first.
5. Gradient checkpointing is **not** bitwise-identical — same seed early-stopped
   at epoch 5 on one path and hit the 10-epoch cap on the other.
