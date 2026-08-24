# Narrative

How this repo got to its current state, including the parts that went wrong.
`README.md` describes what the dataset *is*; this describes how it was built and
what the building taught us. Where a claim here is unsupported by the data, it
says so.

---

## 1. The plan

Build a labeled corpus for two tasks: is this answer human or AI, and if AI,
which model wrote it. 1,000 question/answer pairs from r/explainlikeimfive, AI
answers generated against the same questions by three frontier models, every row
labeled and attributed.

## 2. Clean human labels by construction

The human half comes from the Facebook ELI5 (LFQA) corpus, which spans
2011-07 to 2019-07. Every human answer predates ChatGPT by more than three
years, so the "human" label is true by construction rather than by assumption --
no risk of an LLM-assisted comment being labeled human.

107,280 r/explainlikeimfive rows were streamed out of the 656MB mixed-subreddit
dump; 17,368 passed quality filtering (top-scored answer, score >= 10, 50-250
words, deduped by question); 1,000 were sampled.

## 3. The first real lesson: shortcut features

A detector can score well without reading prose if the two classes differ in
some incidental way. Three such features turned up, each found by looking at the
data rather than by reasoning about it:

**`_URL_n_` placeholders.** The source corpus replaced hyperlinks with tokens
like `_URL_0_`. Present in human text, impossible in AI text. Filtered from
answers at sampling; later also stripped from question bodies, where 61% turned
out to sit inside markdown link syntax -- so the naive strip left broken
`[anchor]()` fragments and had to be rewritten to unwrap the link instead.

**Reddit-native markers.** 7.4% of human answers carry `Edit:`, `TL;DR`, or
karma-speak. No AI answer produces these unprompted.

**Formatting.** 19.1% of human answers use markdown, and the first prompt
*forbade* it -- manufacturing a perfect 0% against a 19% human rate.

The fix that worked, and kept working, was not prohibition but **rate matching**:
inject each feature into the AI class at its measured human base rate, assigned
by a seed derived from `(question id, model)` so runs reproduce exactly. This is
the single most reliable technique found in the whole project.

**The general principle, learned the hard way: a perfect zero is a fingerprint.**
Every prohibition drives a feature to exactly 0.0% where humans sit at 2-7%.
Absence is as detectable as excess. Only a matched *rate* is neutral.

## 4. The adversarial loop (v1 - v7)

Generate a batch, have Haiku judges try to tell the AI answers from the real
ones, rewrite the prompt against whatever the judges said they used, repeat.
Seven prompt versions, seven iterations, ~570 blind trials.

Metric: 2AFC -- judge sees one human and one AI answer to the same question and
picks the AI. Chance 50%. **Distance from 50 in either direction is the score.**
Below chance is not success; it means the AI reads *more* human than the humans,
and a detector would learn the rule backwards and score just as well.

Things the loop genuinely established:

- **Prompt instructions have large, measurable, attributable effects on text
  features.** "Write in the first person" moved first-person usage 0% -> 80%
  (p < 1e-5). Moving that same instruction into base-rate draws moved it
  80% -> 23.3% against a human rate of 23.3% (p = 1e-5).
- **One prompt cannot calibrate several models.** v7's line "go easy on 'just'"
  moved grok -50.0 points (p = 0.0001) and qwen +3.3 (p = 0.80) and deepseek
  -3.3 (p = 0.80). Same words, opposite outcomes. Per-model prompts are
  structural here, not an optimization.
- **Some instructions simply fail.** Em dashes survived explicit prohibition in
  two separate versions at ~10% against a 0.4% human rate, and were eventually
  handled by mechanical substitution at write time. This is documented as a
  transformation, not claimed as a prompting success.

## 5. Where the loop broke: it was never powered

Each condition was measured with 30 trials and a single judge. That design has
SE 9.1 and a **minimum detectable difference of 26.8 points**. Every
version-to-version comparison in the loop was smaller than its own noise floor.

Worse, no version ever changed one line at a time -- v4 changed six, v7 changed
four -- so even a real effect could not have been attributed to a specific
instruction. That is a design failure independent of statistics.

The symptom was visible for several iterations before it was diagnosed: the same
model on the same prompt scored 56.7%, then 36.7%, then 20.0% on different
question batches. Three wrong explanations were offered for this in sequence --
a content drift, then question-set effects, then a 2-2.5x overdispersion
component -- before the obvious experiment was run.

**The experiment that settled it took three Haiku calls.** Four judges on a
byte-identical file scored 20.0, 40.0, 40.0, 36.7. Observed SD 9.6; binomial
expectation at n=30 is 9.1. They match. There is no exotic variance component
and no meaningful inter-judge effect (residual SD 2.9). It was ordinary sampling
noise the whole time, exactly what the textbook formula predicts.

The two failed explanations both came from estimating a variance on two or three
observations, where the 95% CI for an SD spans roughly 0.5x to 6x the truth. A
variance estimated from three points is not evidence of anything.

## 6. What actually survives from v1 - v7

Exact binomial tests with Wilson intervals, Holm-corrected across the 6 pooled
cells:

| model | prompt | score | accuracy | 95% CI | verdict |
|---|---|---|---|---|---|
| grok-4.6 | v6 | 69/180 | 38.3% | [31.5, 45.6] | **inverted** (adj p = 0.013) |
| deepseek-v4-pro | v7 | 41/60 | 68.3% | [55.8, 78.7] | **detectable** (adj p = 0.031) |
| qwen3.8-max | v7 | 46/90 | 51.1% | [41.0, 61.2] | **at chance** (p = 0.92) |
| grok-4.6 | v7 | 8/30 | 26.7% | [14.2, 44.4] | not significant (adj p = 0.065) |
| qwen3.8-max | v6 | 21/30 | 70.0% | [52.1, 83.3] | not significant |
| deepseek-v4-pro | v6 | 34/60 | 56.7% | [44.1, 68.4] | not significant |

Three sentences of defensible result: **qwen/v7 sits at chance across 90 trials.
grok/v6 is inverted across 180. deepseek is detectable on v7 and unresolved on
v6.** Everything else reported during the loop was underpowered, and several
claims made confidently at the time -- "v7 inverted", "grok is saturated" -- do
not survive correction.

## 7. The rebuilt study

`study/DESIGN.md` has the full protocol. The corrections:

| | v1-v7 loop | study |
|---|---|---|
| trials per condition | 30 | 100 |
| judge replicates | 1 | 3 |
| SE | 9.1 | 3.3 |
| min detectable difference | 26.8 pts | 9.3 pts |
| question sets | fresh per iteration | fixed `dev`, paired |
| attribution | 1-6 lines at once | per-model, ablated |
| multiplicity | none | Holm per phase |

**Replicates buy more than trials.** R=3 x n=100 beats R=1 x n=300 for the same
trial count, because a single judge's idiosyncratic read cannot be averaged out
at R=1. At iteration 4 the batch size was tripled and R left at 1, which
improved the term that was already smaller.

Questions are partitioned once and permanently: `burned` (150, retired -- judges
saw those human answers and prompts were written while looking at them), `dev`
(100, reused deliberately so comparisons are paired), `heldout` (100, one look
at the end), `pool` (650, feeds the production dataset).

## 8. The rebuilt study, phase 1: what a powered measurement showed

33 judges, 100 trials each, 3,300 judged trials across three conditions on the
fixed `dev` set. Reporting **mean and SD across judges** rather than a pooled
point estimate turned out to matter more than any single number.

| condition | model | R | mean | SD | reading |
|---|---|---|---|---|---|
| v8_invert | grok-4.6 | 4 | 1.5% | 1.3 | reliably sign-flipped |
| v8_invert | deepseek-v4-pro | 4 | 2.8% | 2.4 | reliably sign-flipped |
| **v6** | **qwen3.8-max** | 3 | **49.3%** | **5.1** | **genuinely ambiguous** |
| v7 | qwen3.8-max | 4 | 55.2% | 14.8 | |
| v6 | deepseek-v4-pro | 3 | 59.7% | 15.3 | |
| v7 | grok-4.6 | 4 | 37.8% | 15.9 | |
| v7 | deepseek-v4-pro | 4 | 53.8% | 19.3 | |
| v6 | grok-4.6 | 3 | 55.3% | 34.6 | judges disagree wildly |
| v8_invert | qwen3.8-max | 3 | 41.0% | 52.2 | judges span 1% to 100% |

**Judge agreement is a second axis, and the mean alone hides it.** grok/v6 and
qwen/v6 both average 49-55%, but one has judges at 22/53/91 and the other at
45/48/55. Only the second is a property of the text. Every conclusion in the
v1-v7 loop was a mean without an SD.

**Precision comes from judge count, not trial count.** Nine judges on identical
v6 text gave SD 19.6 against a binomial expectation of 5.0. The design promised
SE 3.3 from R=3 x n=100; the real figure was about 13. That is the third
consecutive time the variance model was wrong, each time by assuming the wrong
unit of independence.

## 9. The inversion objective, and why 1.2% was not what it looked like

The target was changed from "reach 50%" to "reach 0%" -- make judges reliably
identify the *real human comment* as the machine one.

There is a genuine asymmetry to exploit. The human class is ELI5 answers
filtered to score >= 10, and upvotes on an explainer subreddit go to people who
write like reference works. One real human answer to "What are dog tags for?"
reads:

> 'Dog tags' or more technically known as identification tags serve many
> purposes. There are two types of ID tags. The standard ones which identify the
> service member by NAME, RANK, SOCIAL SECURITY NUMBER, BLOOD TYPE...

while the model wrote blunt unstructured prose. **By the judges' own criteria the
humans in this dataset already look machine-made.**

`prompts/v8_invert.txt` was built from the 21 harvested judge cue lists: every
marker they call human made mandatory, every marker they call AI banned. It hit
**1.5% and 2.8%** on grok and deepseek, with judges agreeing to within 1-2
points. One judge scored **0/100**.

**It was cheating, and the cheat is the same defect this project spent its first
week filtering out of the human class.**

| feature | human (n=100) | v8_invert (n=300) |
|---|---|---|
| `Edit:` line | 11.0% | **87.0%** |
| "I might be wrong" | **0.0%** | 56.3% |
| uncle/cousin/coworker | **0.0%** | 37.7% |
| "IIRC" | **0.0%** | 24.3% |
| "someone correct me" | **0.0%** | 23.7% |

Four of those strings appear in *zero* human answers and a quarter to a half of
the AI ones. That is a planted giveaway feature, structurally identical to the
`_URL_n_` placeholders and the manufactured markdown zero -- a perfect presence
instead of a perfect absence.

It explains both extremes exactly. One qwen judge scored **100/100** by reading
the same anecdotes as evidence of *machine* authorship. Eleven judges applied
the prior "specific personal anecdote means human" and scored ~1%; one applied
the opposite prior and was perfect. **The classes were maximally separated; only
the label assignment varied.**

So the 1.2% measures eleven judges sharing one wrong assumption about a dataset
with a stamp on it -- not text that is hard to detect. A trained classifier
learns the sign from labels and would score ~99% on this text, then fail
completely on ordinary AI output. `v8_invert` is the worst possible production
dataset and a valid answer only to "how low can the number be driven".

The finding that survives is **qwen/v6: 49.3%, SD 5.1** -- no feature gap of this
kind, judges clustered at the binomial floor, no usable rule in either
direction. That is genuinely ambiguous text.

## 10. v9: inversion by register instead of by stamp

`prompts/v9.txt` keeps the inversion objective and removes the mechanism that
made the last one hollow:

- the worn stock phrases are **banned by name** ('I might be wrong', 'someone
  correct me', 'IIRC', "I'm no expert", ...)
- the markers are **drawn, not mandated** -- hedge 30%, personal experience 30%,
  `Edit:` 25%, snark 20%, rough edges 15% -- still above human rates, since the
  goal is inversion, but far from 87%
- each draw asks for the idea **in the model's own words**, so no fixed string
  recurs
- the work is done by register: narrow scope, no teaching structure, uneven
  sentences, plain words

**Judging drops to 25 trials per judge.** 200 answers in one context is enough
for a judge to reverse-engineer the generator from repetition -- v8_invert's
phrases were visible at 56-87% frequency within a single file. 25 trials is
closer to how a detector actually meets text: a couple of dozen documents, no
sibling set to compare against.

Prediction on record before the result: **v9 should land closer to chance than
to 1%.** If it does, v8_invert's number was the planted phrases almost in their
entirety.

## 11. The measurement that mattered, and came last

Every number up to this point is **2AFC**: the judge sees the real human answer
and the AI answer to the same question, side by side, and picks one. A detector
never gets that. It meets one document, alone, with no reference and no sibling
set to compare against.

So: 150 items (75 AI, 75 human), one verdict each, mixed across all three
generator models, five items per judge so no judge could reverse-engineer a
generator from repetition. The condition was **v6** -- the ordinary prompt. No
inversion, no planted markers, artifacts matched to human base rates.

| metric | value | 95% CI |
|---|---|---|
| AI detection (recall) | **26.7%** (20/75) | [18.0, 37.6] |
| false positive rate | **20.0%** (15/75) | [12.5, 30.4] |
| precision on "ai" calls | **57.1%** | |
| **balanced accuracy** | **53.3%** | chance 50% |

Per generator: deepseek 40.0% (10/25), qwen 20.0% (5/25), **grok 20.0%** (5/25).
(An earlier version of this line read 41.7/22.7/17.4 on denominators of 24/22/23 --
it was computed before every prediction file had landed, and summed to 69 items
against a headline of 75.)

**Balanced accuracy 53.3%.** Haiku catches one AI answer in four while falsely
accusing one human in five, and when it says "ai" it is right 57% of the time.
On a single short ELI5 comment it cannot do this task to any useful standard.

**This reframes the whole project.** Nine prompt versions were spent pushing a
2AFC number around. On the realistic task the plain v6 prompt was already at the
floor -- the adversarial work bought nothing that mattered. The 2AFC numbers
were flattering because a known-human reference is a large assist.

A caution about my own reporting: a 16-item pilot of this same test gave 1/8
detection, and it was written up as "detection rate zero". At n=75 it is 26.7%.
The CI on 1/8 was [0.3%, 53%] and the point estimate got quoted. Fourth time in
this project a small sample produced a confident wrong reading.

Two things about that pilot (`study/single/v6_qwen/`) that were not said at the
time. Its false-positive arm was equally thin -- **1/8** there too, never quoted
-- and all eight of its AI items were **qwen only**, so it generalised from one
generator, and not the hard one. But its *reasons* were already right: seven of
its seven missed-AI items cite the same cue ("natural conversational tone",
"casual hedging", "organic phrasing") and its single false positive cites
"absence of hedging, overly clinical". That is exactly the failure mode the
n=150 run confirmed. A sample too small to support a rate can still be large
enough to expose a mechanism -- if you read the reasons instead of the tally.

## 12. The benchmark: three detectors, two lengths

Six cells, 835 single-document judgments. Every item judged alone -- no
reference text, no sibling set -- because that is the task a detector actually
faces. Generation is **v6**, the ordinary prompt: no inversion, no planted
markers, artifacts matched to human base rates.

| | detector | detection | false positives | balanced |
|---|---|---|---|---|
| **short** (median 101w) | Haiku 4.5 | 26.7% [18, 38] | 20.0% [13, 30] | 53.3% |
| | Sonnet | 56.0% [45, 67] | 8.0% [4, 16] | 74.0% |
| | Pangram 3.3.2 | 77.3% [67, 85] | **0.0%** [0, 5] | 88.7% |
| **long** (median 390w) | Haiku 4.5 | 30.7% [21, 42] | 10.7% [6, 20] | 60.0% |
| | Sonnet | 84.0% [74, 91] | 2.7% [1, 9] | 90.7% |
| | Pangram 3.3.2 | **97.6%** [87, 100] | **0.0%** [0, 8] | **98.8%** |

Detector-tier differences on short form, Holm-free pairwise: Sonnet beats Haiku
by 29.3 pts (p=0.00026), Pangram beats Sonnet by 21.3 (p=0.0056), Pangram beats
Haiku by 50.7 (p=5e-10).

**Two denominators are not what they look like.** Pangram's long-form run hit a
402 mid-batch, was restarted, and the restart *appended*: `long_v3.jsonl` holds
146 rows over **85 unique items** (41 AI, 44 human), 61 of them duplicated. The
duplicates agree on every verdict -- the detector is deterministic -- so the
point estimate barely moves, but the figure first written here (97.2% on "n=72")
was counting 31 rows twice. Deduplicated it is 40/41 with a CI of [87, 100], not
[90, 99]. Separately, Pangram's short-form detection counts two items returned
as `Mixed` (`fraction_ai = 0.0`, `fraction_ai_assisted = 1.0`) as catches;
strict `pred == "AI"` gives 74.7%, not 77.3%. The lenient reading is the right
one for "was a model involved", but it was never stated.

### Length is decisive -- for capable detectors only

| detector | long - short detection | p |
|---|---|---|
| Haiku 4.5 | +4.0 pts | 0.59 |
| Sonnet | **+28.0 pts** | **0.00018** |
| Pangram 3.3.2 | **+20.2 pts** | **0.0040** |

**This corrects two earlier conclusions in this file.** Section 11 recorded
"length is not a lever", from a human-only test (16.0% FPR long vs 20.0% short,
p=0.77) and a mixed-pool test that missed significance at n=150 per condition.
Both were true as far as they went and both were about the *false-positive*
side, where length genuinely does little. Detection is where length lives, and
it is worth 20-27 points to a detector capable of using the extra text. Haiku
cannot use it, which is why testing on the weakest detector hid the effect.

The general lesson: an effect measured on one class, or on one detector tier,
does not generalise to the task. Vary the thing you intend to conclude about.

### Pangram's published figures replicate -- at the length it recommends

Pangram 3.3.2 (released 13 May 2026; Pangram 4 arrived 29 July) claims ~98%
detection at ~0.05% false positives. On long form we measured **97.6% at 0.0%**
(40/41 after deduplication -- see above; the long-form cell is 85 items, not
150). On short form the same model drops to 77.3%. Their 250-word recommendation
is not marketing; it is where the model works.

The **false-positive** claim replicates at both lengths: zero flagged out of
**119 unique genuine redditors** (75 short, 44 long -- not the 149 first
reported, which double-counted the appended rows). Of the 19 short-form AI items
Pangram did not call `AI`, all 19 scored `fraction_ai = 0.0` -- a blind spot
rather than uncertainty, with no `fraction_ai` threshold that recovers them.
Two of the 19 were caught on the separate `fraction_ai_assisted` channel, which
is the only reason the headline figure is 77.3% rather than 74.7%.

### Grok 4.6 short-form is the residual weak point

| detector | short | long |
|---|---|---|
| Haiku | 20% | 44% |
| Sonnet | 32% | 76% |
| Pangram | 64% | **100%** |

Grok was the hardest generator for every detector at short length, and at 390
words it is caught 100% of the time at zero false positives -- though Pangram's
long-form grok cell is **15/15** after deduplication, not 25, so "100%" carries
a lower bound of 80%. An earlier claim in
this project that Grok's difficulty "does not yield to a better reader" was
drawn from short-form data and over-generalised: it yields completely to more
text.

## 13. Length, and why a second corpus exists

Pangram -- the reference detector -- requires 50 words minimum and recommends
250+ for its stated accuracy. The main set is 50-250 words, median 101: entirely
inside the band where Pangram reports elevated error rates, capped exactly where
it starts working best. 99.9% of it falls below the recommended threshold.

`data/interim/questions_longform.jsonl` is a companion set of 1,000 questions
whose human answers run 250-799 words (median 345), drawn from the same corpus
and disjoint by construction.

**Tested, and length does not help.** 50 long-form human answers (median 390
words) judged one at a time, every item a genuine redditor:

| | short-form | long-form |
|---|---|---|
| median words | 101 | 390 |
| false positive rate | 20.0% [12.5, 30.4] | **16.0% [8.3, 28.5]** |

(The long-form figure was first reported as 17.5% [8.7, 32.0]. That was 7/40:
one prediction file, `longform_human/pred_04.json`, was malformed JSON and the
scorer skipped it silently, dropping ten of the fifty items -- and the recovered
file contains an eighth false positive. Repaired, it is 8/50.)

No significant difference. Four times the text, same rate of falsely accusing
real people. The flagged answers were if anything *shorter* on average than the
kept ones (361 vs 432 words), and the length bands show no monotone trend.

The reasons say why: humans were rejected for "suspiciously perfect hierarchical
structure", "too systematically comprehensive", "systematic organization". Long
ELI5 answers come disproportionately from people answering inside their own
profession -- an anesthetist on benzodiazepine induction, a nurse, a logistics
worker -- and expert prose is organized. **Length does not make a human look more
human; it makes them look more professional, and professional reads as
synthetic.**

**The full comparison, both conditions at n=150 single-document items:**

| | short (101w) | long (390w) | p |
|---|---|---|---|
| AI detection | 26.7% | 30.7% | 0.588 |
| false positive rate | 20.0% | 10.7% | 0.113 |
| balanced accuracy | 53.3% | 60.0% | 0.244 |
| precision | 57.1% | 74.2% | |

**No difference reaches significance.** All three metrics point the same way and
the FPR gap is the largest, but at n=150 per condition none of it clears p<.05.
"Length halves the false-positive rate" is a hypothesis worth testing at larger
n, not a result.

Two errors of mine are worth recording here. First, the mixed result was
announced as "length does help -- a real result" from an n=90 partial, before
any significance test. At full n it does not clear the bar. Second -- and this
is a correction to the correction -- the human-only long-form test was used to
conclude "length is not a lever", and the fix written here was that "in the
mixed setting **the same human text** was flagged half as often", therefore a
false-positive rate is a property of the pool rather than of the class.

**That claim does not hold, on two counts.** The text is not the same: only
**3 of the 50** `longform_human` ids appear as human items in the mixed
long-form pass, and all three were called `human` in both conditions. The two
rates are measured on near-disjoint samples. And the gap is not significant --
human-only 8/50 (16.0%) against mixed-pool 8/75 (10.7%) gives **Fisher
p = 0.42** (p = 0.39 on the figures as originally reported).

The principle is probably still true and is worth testing properly -- run the
same documents in both pool compositions. But it was written down as a finding
on the strength of two samples that shared 6% of their items and a difference
the data cannot resolve. Sixth instance in this project, and the only one where
the overclaim is a method note rather than a result.

What is solid at both lengths: this is a bad classifier. 27-31% recall, 11-20%
false accusations, balanced accuracy 53-60%.

## 14. The long-form benchmark, rebuilt

An audit of `scripts/12_make_mixed_single.py` found the long-form pass was not
measuring what it claimed. Four defects, in descending order of consequence:

1. **The human arm was 39 documents, not 75.** The builder appended the human
   sibling once per ai row drawn, inside the per-model loop. The long-form
   generation pool held only 60 questions (the rest lost to the 402 in note 12),
   so drawing 25 per model forced most questions to be picked by two or three
   models -- and each pick re-emitted the same human answer as another item. All
   36 excess items were byte-identical copies. The ai arm was clean.
2. **6 of 30 batches contained an ai answer and the human answer to the same
   question** -- a 2AFC pair inside a file whose entire premise is that no
   reference text exists. Sonnet noticed: one reason cites "nearly duplicating
   I052's legal citations". Tested for an effect and there was none (leaked
   batches scored 60% vs 60% for Haiku, 93% vs 90% for Sonnet), so it biased
   nothing measurably -- but the design promised isolation it did not deliver.
3. **The source pool held 65 duplicate `(id, model)` rows**, from the same
   restart-appends-instead-of-resuming failure that double-counted the Pangram
   rows. All 65 pairs differ in text, so the resume check failed rather than the
   writer.
4. **`key.jsonl` sat in the directory handed to the judges.** No evidence any
   judge read it. It should still not have been possible.

The builder now selects **one model per question**, so each question contributes
exactly one ai item and one human item; deduplicates sources on `(id, model)`;
rejects any batch layout placing two answers to one question together; writes
batches to a `batches/` subdirectory with the key outside it; and asserts all of
this before writing. 90 more long-form questions were generated to give the
clean draw room (270 rows, 90 questions, no duplicates).

`study/single/long2` is the rebuilt pass: **150 items, 75 distinct ai documents
across 75 distinct questions, 75 distinct human documents, 25 per generator,
zero within-batch collisions.** 30 batches of 5, one fresh judge per batch,
Haiku 4.5 and Sonnet. Pangram was not re-run.

| detector | detection | false positives | precision | balanced |
|---|---|---|---|---|
| Haiku 4.5 | 24.0% [15.8, 34.8] | 9.3% [4.6, 18.0] | 72.0% | 57.3% |
| **Sonnet** | **78.7% [68.1, 86.4]** | **1.3% [0.2, 7.2]** | 98.3% | **88.7%** |

**Nothing moved.** Against the defective pass, every difference is within noise:

| | defective | clean | diff | p |
|---|---|---|---|---|
| Haiku detection | 30.7% | 24.0% | -6.7 | 0.36 |
| Haiku FPR | 10.7% | 9.3% | -1.3 | 0.79 |
| Sonnet detection | 84.0% | 78.7% | -5.3 | 0.40 |
| Sonnet FPR | 2.7% | 1.3% | -1.3 | 0.56 |

All four move slightly *down*, which is the direction the duplication predicted
-- a false-positive rate computed on 39 documents inflated to 75 items borrows
precision it has not earned -- but no single comparison resolves it, and the two
question sets overlap on only 34 of 75, so this is not a paired test.

And every conclusion the benchmark rests on survives the rebuild:

- **Detector tier.** Sonnet beats Haiku by 54.7 points (p = 2e-11).
- **Length.** Recomputed against the short-form pass: Sonnet **+22.7 pts**
  (p = 0.003), Haiku **-2.7 pts** (p = 0.71). Length is worth ~23 points to a
  detector able to use it and nothing to one that cannot.
- **Coverage is nested, not overlapping.** Of 75 ai documents: 41 caught by
  Sonnet alone, **0 by Haiku alone**, 18 by both, 16 by neither.
- **The failure mode is the same one.** 39 of Haiku's 57 missed-ai items cite a
  surface voice marker (conversational, casual, anecdote) as the overriding
  evidence. 43 of Sonnet's 59 catches cite structure or polish. Same cue
  vocabulary, opposite thresholds.
- **Haiku's confidence still carries no signal**: 59% accurate at "high"
  against 57% at "medium". Sonnet: 94% against 84%.

**What this buys.** Not a new number -- the numbers were right. It buys the
right to quote the false-positive intervals, which were computed on half the
sample they claimed, and it removes a reference-text leak from a benchmark whose
selling point is that it has none. A result that survives having its foundation
replaced is worth more than the same result resting on one that was never
checked.

## 15. What a second reader found

A second session audited this corpus while building an unrelated thing -- a
sub-1.7B single-document detector, written up in `study/small/RESULTS.md`. Its
detector results are its own and are cited, not absorbed. What follows is the
part that lands on the shared corpus, and it is mostly bad news about this file.
Every figure below was recomputed here from the raw texts or from the emitted
per-item files.

### The artifact nobody looked for: paragraph convention

Count the line breaks. That is the whole classifier.

| set | newlines | paragraphs | mean words/para | word count |
|---|---|---|---|---|
| `long2` | **0.802** | 0.772 | 0.778 | 0.527 |
| `long3` | **0.803** | -- | -- | -- |
| `long` * | **0.862** | 0.848 | 0.839 | 0.578 |
| `short` * | **0.760** | 0.758 | 0.801 | 0.524 |

AUC, oriented so higher means more separable. Word count is at chance, because
`03_generate.py` pins it to the specific human answer. Nothing pins paragraphs.

\* `long` and `short` carry the section 14 duplicated-human-arm defect, so these
two rows are computed on **distinct human documents** (39 and 58) rather than on
the 75 items each claims. On items they read 0.856 and 0.778. Correcting for the
duplication moves `long` up and `short` down, and changes no conclusion -- but
quoting the item-level figure in the section that documents the defect would
have been the fourth instance in this file of exactly that.

The best single threshold on `long2` -- `newlines <= 8` means ai -- gives
detection 84.0%, false positives 36.0%, **balanced accuracy 74.0%**. That beats
Haiku 4.5 (57.3%) and the 1.2B detector (68.7%) on the same 150 items, and loses
only to Sonnet. On `short`, distinct documents, it is 74.9% against Haiku's
53.3%. **On a threshold fitted in place, `wc -l` is a better AI detector than
Haiku 4.5 on every benchmark in this repository** -- the comparison is not quite
like for like, since Haiku's verdicts are zero-shot and the baseline gets one
integer fitted on the same set, but one integer is close to free and no result in
section 12 or 14 was ever quoted against any baseline at all.

**It is not uniform across generators, and the spread is the finding.** Newline
AUC against each set's own humans:

| set | grok-4.6 | deepseek-v4-pro | qwen3.8-max | pooled | pooled minus grok |
|---|---|---|---|---|---|
| `long2` | 0.950 | 0.795 | 0.659 | 0.802 | 0.727 |
| `long3` | 0.996 | 0.801 | 0.614 | 0.803 | 0.707 |
| `long` * | 0.961 | 0.911 | 0.715 | 0.862 | 0.813 |
| `short` * | 0.848 | 0.800 | 0.631 | 0.760 | 0.716 |

Median structure on `long2`: human 11 newlines / 6 paragraphs / 59 words per
paragraph; qwen 8 / 5 / 65; deepseek 6 / 4 / 99; **grok 1 / 1 / 275**. Grok's
median long-form answer is a single unbroken block. Its one-block rate is 60%
against a human 1%; on short form it is 100% against a human 25%.

Qwen paragraphs like a redditor and has never left 0.61-0.72 in four sets. Grok
has never left 0.85-1.00. Dropping the worst offender never removes the artifact;
it removes about half of it.

**Every cell in that table is a 25-document draw, and it has an SD of 0.040.**
An earlier version of this section read deepseek's 0.795 on `long2` against 0.911
on `long` as a model unstable with itself. It is not. The two draws share only 8
documents; on the full 90-document long-form pool deepseek's newline AUC is
**0.828**, and 4,000 resamples of 25 documents against 39 humans give mean 0.828,
SD 0.040, 95% range [0.747, 0.902]. Both set-level figures straddle the pool
value, and two independent draws differ by 0.116 or more about **4%** of the
time. Swapping the human comparison set accounts for only 0.031 of the gap. The
words-per-paragraph figures, 99 against 118, are two medians of 25 and say
nothing either.

That was section 5's error committed inside the section that documents it: a
variance read off two observations. It was caught by a second reader running the
resampling test rather than by the person who wrote the claim.

**What survives is the between-model spread, and it survives easily.** Qwen at
~0.61-0.72 against grok at ~0.85-1.00 is a gap of about 0.30 against a per-draw
SD of 0.040 -- seven standard deviations, in the same direction on all four sets.
"One prompt cannot calibrate several models" is untouched; it was only the
within-model claim that dissolved. A reader comparing `long2`'s grok 0.950 with
`long`'s 0.961 should treat those two numbers as indistinguishable.

**And generation noise is small, which nobody had ever measured.** Every
per-generator figure in this file assumes that regenerating the same question
with the same model and prompt would give roughly the same answer. The failed
resume documented in section 14 left 65 `(id, model)` pairs generated twice --
independent samples, different text -- so it can be checked. Same 65 questions,
two runs, newline AUC against their matched humans:

| generator | n | run 1 | run 2 | delta | median newlines |
|---|---|---|---|---|---|
| deepseek-v4-pro | 24 | 0.853 | 0.867 | 0.014 | 6 / 6 |
| grok-4.6 | 21 | 0.967 | 0.955 | 0.012 | 1 / 1 |
| qwen3.8-max | 20 | 0.720 | 0.710 | 0.010 | 10 / 8 |

Regeneration moves a per-generator figure by 0.010-0.014, against a sampling SD
of 0.040 for a 25-document draw. The pairs are at
`study/longform/answers_prededup.jsonl`. Each model's median paragraph habit is
identical across runs. So paragraph disposition is a stable property of the
model, not of the draw -- the between-model spread is measuring the models, and
the within-model wobble is the sample. That is the check that should have
accompanied the claim in the first place, and the data for it existed only by
accident.

So this is not section 3's family. It is **section 4's**: one prompt cannot
calibrate several models, resurfacing in a dimension nobody thought to match.
`prompts/v6.txt` says nothing at all about paragraphs, so this is not a
prohibition driving a feature to zero -- it is an omission from the rate-matching
program. Word count, markdown rate, `Edit:`/`TL;DR`/karma rate and persona rate
were all matched. Paragraph structure, the most separable surface feature of the
lot, was never considered.

**The reported detector numbers survive.** Split each class at its own median
newline count and the detectors' conditional accuracy is flat:

| | few breaks | many breaks |
|---|---|---|
| Sonnet, ai items | 38/48 = 79% | 21/27 = 78% |
| Sonnet, human items | 41/41 = 100% | 33/34 = 97% |
| Haiku, ai items | 12/48 = 25% | 6/27 = 22% |
| Haiku, human items | 37/41 = 90% | 31/34 = 91% |

Neither is riding it. Sonnet's 88.7% is not a whitespace detector in disguise.
This is a benchmark validity problem, not a retraction -- but every detector
figure in this file should have carried the trivial baseline beside it, and none
of them did.

The fix is section 3's own principle applied one dimension further: draw a target
paragraph count and mean paragraph length per `(id, model)` from the specific
human answer, the way `{n}` already pins words. Whether that works is an open
experiment, not a patch -- section 4 has both outcomes on record, first person
moving 0% -> 80% -> 23.3% on instruction and em dashes surviving explicit
prohibition twice. State the prediction before running it.

### The rate-matching program, verified at last

Section 3 calls rate matching "the single most reliable technique found in the
whole project" and never once showed it firing at its specified rate. It does:

| draw | designed | observed (v6, n=300) |
|---|---|---|
| `Edit:` line | 5.1% | 13/300 = 4.3% |
| `TL;DR` opener | 1.9% | 6/300 = 2.0% |
| casual reddit idiom | 0.7% | 2/300 = 0.7% |
| v9 `Edit:` (invert draw) | 25% | 24% strict / 26% loose |

Four rates across two independent draw mechanisms, all within sampling error.
Combined with the second reader's finding that the injections do not move its
detector's margin at all -- artifact `none` +4.76 against `Edit:` +2.67 (n=3),
persona 0 +4.72 against persona 1 +4.65 -- the section 3 program comes out of
this clean. The injected artifacts are individually undetectable, and they are
present at the rate the code claims.

Two incidental corrections. Section 9's `Edit:` figure of 87.0% counts
line-initial occurrences; counting `edit:` anywhere gives **100.0%**, and there
are **zero** marker-free `v8_invert` documents across eight marker families, so
the obvious control -- rescore the subset carrying no planted marker -- cannot be
run at any n. And v6 and v7 share identical artifact draws, because the seed is
`(id, model)` and does not include the prompt version; that makes any v6-vs-v7
comparison better than paired, with the injections literally held constant.

### Section 10's open prediction, closed

Nobody had ever measured the prompt versions one document at a time. On the
fixed 100-question `dev` human set, single-document AUC:

| version | pooled | deepseek | qwen | grok | `Edit:` rate |
|---|---|---|---|---|---|
| v6 | 0.718 | 0.755 | 0.707 | 0.693 | 4% |
| v7 | 0.703 | 0.726 | 0.670 | 0.712 | 4% |
| v9 | **0.622** | 0.677 | 0.571 | 0.616 | 26% |
| v8_invert | **0.461** | 0.593 | 0.425 | 0.363 | 100% |
| *(human)* | | | | | *11%* |

**Read the rows, not the columns.** The pooled column is ~300 documents per
version and resolves the differences between versions. The per-generator columns
are ~100 each; on a paired bootstrap none of the three pairwise generator
differences within v6 clears its own interval (deepseek - qwen +0.048
[-0.028, +0.127], deepseek - grok +0.062 [-0.020, +0.138], qwen - grok +0.013
[-0.065, +0.095]). Qwen is lowest in all four sets, which is suggestive, but the
sets are not independent and no single comparison is powered. Treat the columns
as a hypothesis.

**v7 buys nothing over v6** -- minus 0.015, with the injections held constant, so
it is a clean read on the prose changes alone. That reproduces section 11 from a
different instrument: the adversarial loop was at the floor by v6, and here it is
visible on the realistic task rather than inferred from a 2AFC number.

**v9 buys a real minus 0.096 at a 26% marker rate against v8_invert's 100%.**
Section 10 put a prediction on record that v9 would land closer to chance than to
1% and that the planted phrases were doing most of v8_invert's work. On the
realistic task that resolves in its favour: the register work carries genuine
evasion without the planted-feature saturation that made `v8_invert` unusable as
data. The prediction had sat unresolved partly because the v9 set was never
finished -- it is 288 rows over **99** questions, not 300 over 100, one question
having failed all three generators and never been regenerated.

And `v8_invert` drives a detector below chance by a mechanism that is the
reciprocal of section 9 rather than a restatement of it. It does not plant a tell
this detector picks up -- markers move its margin by at most 0.36 either way.
The whole distribution collapses instead, to a mean genericness *below* the human
class. Section 9's judges failed because the separation was labelled backwards;
this detector fails because on its axis the separation is genuinely gone. Neither
is a detection result.

### Two method results

**An AUC is length-portable. An operating point is not.** Margins fall about two
nats as documents shorten. A threshold tuned on long form, applied to short form,
gave 9.3% detection at 0.0% false positives -- it calls nearly everything human.
Re-tuned in place: 58.7% at 19.0%. Quote an AUC, or quote an operating point with
the length it was tuned at attached. This file has four times reported a
collapsed rate as a property of a detector.

**A pre-registration and a shortcut audit answer different questions.** The
second reader pre-registered an early-exit variant as a dev-selected spike likely
to fail, and it did fail on short form. But the surface-feature audit found
something the pre-registration could not: within the ai class, its margin
correlates with newline count at **-0.67** on `long2` and **-0.65** on `long3`,
against **-0.04** and **+0.04** for the full model. Drop one generator and it
collapses in lockstep with the whitespace baseline:

| set | full model | early exit | `wc -l` |
|---|---|---|---|
| `long2` | -0.014 | **-0.092** | -0.075 |
| `long3` | -0.005 | **-0.095** | -0.096 |
| `short` | -0.012 | -0.038 | -0.041 |

The truncated model reads whitespace; the ten layers it skips are what move the
decision onto prose. And the variant **would have survived** every check this
file had: held-out confirmation at 0.791, beating the full model outright,
pre-registered caveat satisfied, frozen threshold, a number that rose from dev to
confirmation. It is the first result in this project that passed every check the
method had and was caught only by a check the method did not have.

A pre-registration establishes whether a result survives. A shortcut audit
establishes whether it is the thing you think it is. They are not redundant, and
this project had never pointed both at the same claim.

### The detector, cited not absorbed

For context only: LFM2.5-1.2B-Instruct, one forward pass, zero generated tokens,
scored **AUC 0.771 [0.689, 0.846]** on the held-out `long3` -- up from 0.725 on
dev, which is the direction that does not happen when a result has been fitted.
`wc -l` scores 0.803 on the same items. The second reader reports both, framed as
different quantities: 0.80 is a formatting-convention gap, 0.77 is prose. Full
method, caveats and per-item data in `study/small/RESULTS.md`; the model's
capability benchmarks there are vendor-reported and unverified, only the
parameter count and throughput were measured in this repo, and the freeze
timestamp is corroborated by the session record rather than self-certifying.

## 16. Method notes worth carrying elsewhere

1. **Look at the data before reasoning about it.** Every shortcut feature was
   found by printing examples. The markdown-link bug in the `_URL_` strip was
   found by rendering two prompts and reading them.
2. **Run the cheap experiment first.** Three Haiku calls settled a variance
   question that had produced two wrong theories and several paragraphs of
   explanation.
3. **A variance estimated from three points is not evidence.**
4. **Compute the minimum detectable difference before running, not after.**
5. **Change one thing at a time, or accept that nothing is attributable.**
6. **Absence is a fingerprint.** Match rates; do not prohibit.
7. **Below chance is not success.** Score distance from chance, in either
   direction, and say so in the tooling so it cannot be misread later.
8. **Report the SD, never the mean alone.** Two conditions averaging 52% -- one
   with judges at 45/48/55, one at 22/53/91 -- are not the same result. The
   spread is the finding.
9. **Check your own output for the defect you filtered out of theirs.** The
   inversion prompt planted a giveaway feature as blatant as the `_URL_`
   placeholders removed in week one. Run the artifact scan on both classes.
10. **A near-perfect score in either direction means a shortcut, not skill.**
   1% and 100% on the same file, from judges reading the same cue with opposite
   priors, is a separability result masquerading as a detection result.
11. **Measure the task you actually care about, first.** Nine prompt versions
   optimised a 2AFC score. The realistic task -- one document, no reference --
   was measured last and showed the work had been unnecessary since v6.
12. **Read the failure log before reporting slowness.** Long-form generation sat
   at 1/180 and was reported as "crawling"; it had died on a 402 Payment
   Required after the first call.
13. **Run the significance test before announcing the finding.** "Length does
   help -- a real result" was said from an n=90 point estimate; at n=150 the
   comparison gives p=0.11. Fifth instance in this project of reporting a
   difference the data could not support, and it came one message after writing
   a method note against exactly that.
14. **A false-positive rate may not be a property of one class alone.** Stated
   here as established, on a human-only-vs-mixed-pool comparison that turned out
   to use near-disjoint samples (3 shared items of 50) and a difference at
   p = 0.42. Still the right thing to test -- same documents, both pool
   compositions -- but it is a hypothesis. Test the operating condition, not one
   class of it.
15. **Vary the thing you intend to conclude about.** "Length is not a lever" was
   measured on the human class and on the weakest detector -- the two places the
   effect does not appear. On detection, with a capable detector, length is
   worth 20-27 points. The conclusion was not underpowered; it was measured in
   the wrong place.
16. **A silent parse failure is a silent sample-size cut.** Three reported
   figures in this file were computed on fewer items than they claimed, none of
   it visible in the output: two malformed prediction files (an unescaped `"`
   inside a reason string, a doubled `}`) that the scorer skipped without a
   warning, and a Pangram batch whose post-402 restart *appended* its rows, so
   61 items were counted twice. Assert the expected item count before scoring,
   and fail loudly on a file that does not parse.
17. **Audit the thing that built your sample, not just the thing that scored
   it.** Three months of numbers in this file were shaped by one line in the
   item builder that emitted the human sibling inside the per-model loop. The
   scorer was correct, the judges were correct, the statistics were correct, and
   the human arm was still half the size it reported. Print distinct-document
   counts next to item counts, always.
18. **Read the reasons, not just the tally.** The n=16 pilot's rate was
   worthless and its stated cues were already the correct diagnosis. A sample
   too small for a rate can still be large enough for a mechanism.
19. **Quote the trivial baseline beside every detector number.** `wc -l` scores
   AUC 0.76-0.86 on every set in this repo and, on a threshold fitted in place,
   balanced accuracy 74.0% against Haiku's 57.3%; no figure in section 12 or 14
   was ever reported against any baseline at all. A detector result without the dumbest possible baseline on the same items
   is not a result, it is a number.
20. **Match every dimension you can measure, not the ones you thought of.**
   Word count, markdown, reddit artifacts and persona were all rate-matched.
   Paragraph structure was not, and it turned out to be the most separable
   surface feature in the corpus. The omission was invisible for the same reason
   every shortcut in section 3 was: nobody printed the distribution.
21. **An AUC is length-portable; an operating point is not.** A threshold tuned
   on long form gave 9.3% detection at 0.0% false positives on short form -- a
   collapsed operating point, not a blind detector. Quote the AUC, or quote the
   threshold with the length it was tuned at attached.
22. **A pre-registration and a shortcut audit are not redundant.** One
   establishes whether a result survives; the other establishes whether it is
   the thing you think it is. A variant that passed held-out confirmation, a
   pre-registered caveat, a frozen threshold and a dev-to-confirmation rise was
   still counting line breaks, and only the surface-feature correlation caught
   it.
23. **Resample before calling something unstable.** A generator whose newline
   AUC read 0.795 on one set and 0.911 on another looked unstable with itself.
   Both are 25-document draws from a pool whose value is 0.828 with a per-draw
   SD of 0.040; a gap that size occurs about 4% of the time by chance. This is
   note 3 again -- a variance estimated from two observations -- and it was
   written into the section that documents note 3, four sections after the
   lesson was recorded. Knowing the rule does not fire the check; running the
   resample does.
24. **Get a second reader with a different objective.** Sections 12 to 14 were
   audited twice by the person who wrote them and the paragraph artifact
   survived both passes; the second reader had the per-item margins in front of
   it for an hour and did not find it either. It surfaced only when the two
   compared notes across different goals. Attention was not the missing
   ingredient and neither was care -- what neither reader had alone was a reason
   to look at that column.
25. **Compare against a purpose-built tool before concluding a task is hard.**
   Nine prompt versions were spent making text that defeats an LLM judge. A
   commercial classifier reads the same text at 97.6% with zero false positives.
   The difficulty was a property of the evaluator, not the text.

## 17. State

Built and verified: corpus fetch, sampling (short and long form), generation
with per-model prompts and seeded draws, dataset assembly with question-grouped
splits, judge harness, scorer with position-bias diagnostics, power calculator,
eval partitioner.

**Complete.**

- Corpus: 1,000 short-form (50-250w) and 1,000 long-form (250-799w) questions
  with pre-2022 human answers; four disjoint eval partitions.
- Prompts: v1-v9, every version kept, with the measurement behind each change.
- Adversarial loop: 7 iterations, 21 Haiku judges, ~570 2AFC trials (mostly
  underpowered -- see sections 5 and 6).
- Study phase 1: v6, v7, v8_invert on `dev`, 900 generations, 33 judges,
  3,300 judged trials.
- **Benchmark: 300 labeled single-document items x 3 detectors x 2 lengths,
  835 judgments.** Coverage is complete for Haiku and Sonnet (150 items each per
  length); Pangram covers all 150 short-form items but only 85 of the 150
  long-form ones, the rest lost to a 402 mid-batch.
- **Second reader's sets:** `study/single/long3` (150 items, held-out
  confirmation, built with the fixed builder and disjoint from `long2` by id),
  and single-document margins for v6/v7/v9/v8_invert on `dev` --
  `study/small/peritem_*.jsonl`, written up in `study/small/RESULTS.md`.
- **Rebuilt long-form pass (`study/single/long2`): 150 items over 75 distinct
  ai and 75 distinct human documents, Haiku + Sonnet, 300 judgments.** This is
  the artifact worth keeping -- it is the only pass with no duplicated document,
  no reference-text leak, and the answer key outside the judge directory. The
  original `study/single/long` is retained for the before/after comparison in
  section 14; prefer `long2` for any new work.

**The one free experiment left.** The failed resume in section 14 produced 65
`(id, model)` pairs generated twice from the same prompt -- same question, same
model, same prompt, independently sampled, all 65 differing in text. They are
preserved at `study/longform/answers_prededup.jsonl` (245 rows, 180 unique
pairs); `study/longform/answers.jsonl` is the deduplicated working copy and
nothing was lost. Section 15 uses them to show generation noise is small on the
paragraph feature -- 0.010-0.014 against a sampling SD of 0.040. They remain an
unused within-model variance estimate for every other feature this project
measures, at a cost of seconds each. Nobody has separated "what this model does"
from "what this sample did" on any other axis, and every per-generator figure in
this file and in `study/small/RESULTS.md` assumes the separation is small
without having checked it anywhere but here.

**The largest known defect.** Paragraph convention separates the classes at
AUC 0.78-0.86 in every set built so far (section 15). It does not invalidate the
detector figures -- their conditional accuracy is flat across paragraph density
-- but it means `wc -l` outperforms every detector below Sonnet on this corpus,
and a v10 that rate-matches paragraph count and mean paragraph length per
`(id, model)` is the single highest-value change left.

**Cheap things still open.** Re-running `scripts/13_pangram.py` on `long2` would
give the reference detector a clean long-form denominator (its current 40/41 is
on 41 unique items). `scripts/09_score_phase.py:33` still swallows a JSON parse
failure with a bare `except: continue` -- the idiom that hid two of the three
sample-size bugs in this file. `scripts/14_score_single.py` is the corrected
pattern.

**Not done.** The production 3,000-row dataset was never generated -- the
project turned into a detector evaluation instead, which was the more useful
outcome. `scripts/03_generate.py` and `scripts/04_build_dataset.py` are built
and tested; running them produces it.

**Where the headroom is.** Detection at 390 words is close to solved (Pangram
97.6%, Sonnet 78.7% on the clean rebuild). Short-form is not: the best
available recall is 77.3%, and Grok 4.6 short-form sits at 64% against a
commercial detector and 32% against Sonnet. A detector aimed at Reddit comments, chat messages and forum
replies -- the length most real text actually is -- has genuine room, and the
labeled adversarial benchmark to measure against is in this repo.
