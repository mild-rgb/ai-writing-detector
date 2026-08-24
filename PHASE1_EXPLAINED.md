# Phase 1, explained

*A plain-language guide to what phase 1 of this project did, what it found, and
what it got wrong. No statistics background assumed.*

This document is a companion to `phase1/NARRATIVE.md` and `phase1/README.md`.
Those are lab notebooks — dense, precise, written for someone who already has the
context. This one is written for someone who does not.

---

## The question

Can you tell, from a single piece of writing, whether a human or an AI wrote it?

That question sounds simple and turns out not to be. Phase 1 set out to build a
dataset that would let people train and test detectors. It ended up mostly
measuring how hard the task is, and — more usefully — how easy it is to *think*
you have measured it when you have measured something else entirely.

---

## Part 1: Building the raw material

### The human side

You cannot label writing as "human" just because a person posted it. Since late
2022, any text on the internet might have been drafted, edited, or polished by a
language model. If even 5% of your "human" examples are secretly AI-assisted,
your detector is being trained on lies.

The way around this is to go back in time. The project used the **Facebook ELI5
corpus** — an archive of questions and answers from Reddit's
r/explainlikeimfive, covering July 2011 to July 2019. Every answer in it was
written more than three years before ChatGPT existed. The "human" label is
therefore true *by construction*: it does not depend on anyone's judgment, only
on the calendar.

The pipeline:

- A 656 MB compressed dump of mixed Reddit data was streamed through, keeping
  only r/explainlikeimfive posts — **107,280** of them.
- Those were filtered down to **17,368** usable question/answer pairs: take only
  the top-scored answer to each question, require a score of at least 10 (so the
  community endorsed it as a real explanation), require 50–250 words, and remove
  duplicate questions.
- **1,000** were sampled as the working set.

A second, separate set of **1,000 long-form questions** was built the same way,
with human answers of 250–799 words (median ~345). This exists for a specific
reason explained later.

### The AI side

Three frontier models were asked the *same* questions:

| Model | Why it matters |
|---|---|
| Grok 4.6 | |
| Qwen3.8 Max | Each question gets an answer from all three, |
| DeepSeek V4 Pro | so comparisons are apples-to-apples |

Because all three answer every question, the dataset is **paired**: for any given
question you have one real human answer and three machine answers about the same
topic at the same length. That pairing removes a huge amount of noise — if one
question happens to be easier to write about, all four documents get that
advantage equally.

### Making the AI text a fair opponent

Left alone, a language model writes 300 well-formatted words with headings and
bullet points. The median ELI5 answer is 75 words of unformatted prose. A
detector trained on that contrast learns to count bullet points, not to read.

So generation was constrained: 2019-era Reddit register, plain text, no greeting
or sign-off, no restating the question — and, critically, **the target word count
for each AI answer is set to the word count of that specific question's human
answer**. Length is neutralised as a clue.

---

## Part 2: The most important idea in the whole project

Early on, three "shortcut features" turned up. A shortcut feature is anything
that separates the two classes without having anything to do with writing
quality:

1. **`_URL_0_` placeholders.** The source corpus replaced hyperlinks with tokens
   like `_URL_0_`. Present in human text. Impossible in AI text. A detector could
   score perfectly on this alone.
2. **Reddit-native markers.** 7.4% of human answers contain `Edit:`, `TL;DR`, or
   karma-speak. No AI produces these unprompted.
3. **Markdown formatting.** 19.1% of human answers use markdown — and the first
   generation prompt *forbade* markdown, manufacturing a perfect 0% for the AI
   class against a 19% human rate.

These three do not all get the same treatment, and the difference is the point.

### First, separate corpus artifacts from human behaviour

`_URL_0_` is not something a human did. It is something the *dataset builders*
did — they scrubbed hyperlinks out of the archive and left a token behind. Real
Redditors posted links; nobody ever typed `_URL_0_`.

That distinction decides the fix. A corpus artifact has to be **removed**,
because there is no honest rate to match — injecting `_URL_0_` into AI answers
would be planting a dataset scrubbing token into text that never contained a
link. A human *behaviour*, by contrast, must be matched rather than banned, for
the reason in the next section.

So the placeholder was handled in two places, differently, because it appears in
two roles:

**In the human answers — excluded, not repaired.** Any candidate answer
containing the token is dropped outright before sampling
(`scripts/02_sample_pairs.py:44`, and the same in the long-form sampler). The
result is verifiable: **0 of the 1,000 human answers** contain it, and 0 question
titles do.

Worth naming the cost, which the repo does not: this discards every human answer
that cited a source. On top of the "top-scored answer, score ≥ 10" filter, the
human class is narrowed once more — and in a direction that plausibly correlates
with how people write.

**In the question bodies — stripped at generation time.** Bodies were *not*
filtered at sampling, so 29 of the 1,000 still carry the token (38 tokens in
all). They are cleaned when the prompt is assembled instead
(`scripts/03_generate.py:172`). The reason is contamination in the other
direction: leaving the token in the prompt puts a string in front of the model
that appears nowhere in natural text, and the model may echo it into its answer —
handing the AI class the very marker the human class was just screened for.

**And the obvious way to strip it was wrong.** Deleting the token leaves wreckage,
because most of these tokens are not standing alone:

```
before   [this study](_URL_0_) shows...
naive    [this study]() shows...      <- still a fingerprint, now a stranger one
fixed    this study shows...
```

**61% of the tokens sit inside markdown link syntax** (recomputed from the
corpus: 23 of 38, 60.5%). So the strip had to be rewritten to *unwrap* the link
to its anchor text first, then remove any bare tokens, then repair the whitespace
the deletion leaves behind. Checked across every generated file in the repo: no
AI answer anywhere contains the token. Nothing leaked.

That bug was not found by re-reading the regex. It was found by rendering two
prompts and looking at them — which is method note 1 in the source narrative.

### Second, for the features that *are* human behaviour

The obvious fix is to ban them everywhere. That fix is wrong, and understanding
why is the single most transferable lesson here.

> **A perfect zero is a fingerprint.**
>
> If humans use a feature 7% of the time and you drive the AI class to exactly
> 0.0%, you have not removed the clue. You have inverted it. Absence is exactly
> as detectable as excess.

The fix that works, for anything a human actually chose to do, is **rate
matching**: measure how often humans do the thing, then inject it into the AI
class at that same measured rate. Not prohibited —
*matched*. Which documents get the injection is decided by a random seed derived
from `(question id, model)`, so the whole thing reproduces exactly on a re-run.

This was verified end to end much later in the project, and it holds:

| feature | designed rate | observed rate |
|---|---|---|
| `Edit:` line | 5.1% | 4.3% (13/300) |
| `TL;DR` opener | 1.9% | 2.0% (6/300) |
| casual Reddit idiom | 0.7% | 0.7% (2/300) |

All within sampling error, and independent testing showed the injections do not
move a detector's score at all. The technique works. Remember that it *only* works
for features somebody thought to measure — this comes back to bite the project in
Part 6.

---

## Part 3: The adversarial loop, and how it failed

The plan was to make the AI text progressively harder to spot. Generate a batch,
have a small model (Claude Haiku 4.5) act as a judge and try to pick out the AI
answers, then rewrite the generation prompt to defeat whatever the judge said it
was using. Repeat.

Seven prompt versions, seven rounds, about 570 judged trials.

### The scoring rule

The metric was **2AFC** (two-alternative forced choice): the judge sees one
human answer and one AI answer to the same question, side by side, and picks
which is the AI. Chance is 50%.

The score is **distance from 50 in either direction**. This matters more than it
sounds. If the judge scores 20%, that is not a success — it means the AI answers
read *more human than the humans do*, and a trained detector would simply learn
the rule backwards and score 80%. Both directions are equally detectable.

### What the loop genuinely established

- **Prompt instructions have large, measurable effects.** Adding "write in the
  first person" moved first-person usage from 0% to 80%. Moving that same
  instruction into a base-rate draw brought it to 23.3% — against a measured
  human rate of 23.3%.
- **One prompt cannot calibrate several models.** Version 7's instruction "go
  easy on the word *just*" moved Grok by −50 points, Qwen by +3.3, and DeepSeek
  by −3.3. Same words, opposite outcomes. Per-model prompts are structurally
  necessary, not an optimisation.
- **Some instructions simply fail.** Em dashes survived explicit prohibition in
  two separate prompt versions and had to be removed mechanically after
  generation.

### Where it broke

Every condition was measured with 30 trials and a single judge. With those
numbers, the standard error is 9.1 percentage points and the **smallest
difference the design can reliably detect is 26.8 points**. Every single
version-to-version comparison in the loop was smaller than its own noise floor.

Worse, no version ever changed one thing at a time — v4 changed six lines, v7
changed four — so even a real effect could not have been traced to a specific
instruction.

The symptom was visible for several rounds before anyone diagnosed it: the same
model on the same prompt scored 56.7%, then 36.7%, then 20.0% on different
batches of questions. Three increasingly elaborate explanations were proposed
(content drift, question-set effects, an "overdispersion component") before
somebody ran the obvious experiment.

**That experiment cost three API calls.** Four judges were run on a
byte-identical file. They scored 20.0, 40.0, 40.0, and 36.7. The observed spread
was 9.6; the textbook prediction from pure coin-flip randomness at n=30 is 9.1.
They match. There was no exotic effect. It was ordinary sampling noise the entire
time.

The two failed explanations both came from estimating variability off two or
three observations — where the plausible range for the true value spans roughly
half to six times whatever you computed.

> **A variance estimated from three points is not evidence of anything.**

### What survives from the loop

After proper statistical treatment (exact binomial tests, Wilson confidence
intervals, Holm correction for testing many things at once), three sentences
survive out of seven rounds:

- Qwen on prompt v6 sits at chance across 90 trials (51.1%).
- Grok on v6 is inverted across 180 trials (38.3%).
- DeepSeek is detectable on v7 and unresolved on v6.

Claims made confidently during the loop — "v7 inverted", "Grok is saturated" —
do not survive correction.

---

## Part 4: The rebuilt study

The study was redesigned with the arithmetic done first:

| | v1–v7 loop | rebuilt study |
|---|---|---|
| trials per condition | 30 | 100 |
| independent judges per condition | 1 | 3 |
| standard error | 9.1 pts | 3.3 pts |
| smallest detectable difference | 26.8 pts | 9.3 pts |
| question sets | fresh each round | fixed, so comparisons are paired |
| changes per version | 1–6 at once | one at a time |

One counterintuitive point: **replicates buy more than trials**. Three judges on
100 questions beats one judge on 300 questions, for the same total effort,
because a single judge's personal quirks cannot be averaged away when there is
only one judge.

The 1,000 questions were partitioned once and permanently:

- **burned** (150) — retired. Judges saw those human answers and prompts were
  written while looking at them, so they are contaminated forever.
- **dev** (100) — reused deliberately, so every comparison is paired.
- **heldout** (100) — looked at once, at the end.
- **pool** (650) — reserved to feed the production dataset.

### What a properly powered measurement showed

33 judges, 3,300 judged trials, three conditions. The headline result was not a
number — it was a *second* number that nobody had been reporting.

Two conditions both averaged around 50%. In one, the three judges scored 45, 48,
and 55. In the other, they scored 22, 53, and 91.

Those are not the same result. The first is a property of the text: everyone
finds it ambiguous. The second is judges disagreeing violently about text that
may be perfectly separable — they just cannot agree which way.

> **Report the spread, never the average alone.** The disagreement between judges
> is a second axis of the result, and it is often the more informative one.

The cleanest finding of this phase: **Qwen on v6 scored 49.3% with a spread of
5.1** — judges clustered tightly at chance, no usable rule in either direction.
That is genuinely ambiguous text.

---

## Part 5: The inversion trap

At one point the goal changed from "reach 50%" (undetectable) to "reach 0%"
(judges reliably pick the *human* answer as the machine one).

There is a genuine asymmetry to exploit. The human class here is ELI5 answers
that scored 10 or more, and upvotes on an explainer subreddit go to people who
write like reference works. One real human answer to "What are dog tags for?"
opens:

> 'Dog tags' or more technically known as identification tags serve many
> purposes. There are two types of ID tags. The standard ones which identify the
> service member by NAME, RANK, SOCIAL SECURITY NUMBER, BLOOD TYPE...

By the judges' own stated criteria, the humans in this dataset already look
machine-made.

A prompt called `v8_invert` was built by harvesting 21 judges' explanations of
what they thought looked human and what looked machine, then making every
"human" marker mandatory and banning every "machine" one. It scored **1.5% and
2.8%**. One judge got 0 out of 100.

**It was cheating, and it was cheating in exactly the way the project had spent
its first week filtering out.**

| feature | human rate | v8_invert rate |
|---|---|---|
| `Edit:` line | 11.0% | **87%** (100% counting all positions) |
| "I might be wrong" | **0.0%** | 56.3% |
| mentions uncle/cousin/coworker | **0.0%** | 37.7% |
| "IIRC" | **0.0%** | 24.3% |

Four of those strings appear in *zero* human answers and a quarter to a half of
the AI ones. That is a planted giveaway — structurally identical to the `_URL_`
placeholders, only a perfect *presence* instead of a perfect absence.

The proof is in the outlier. One judge scored **100 out of 100** by reading the
same personal anecdotes as evidence of *machine* authorship. Eleven judges
applied the prior "specific personal anecdote means human" and scored ~1%; one
applied the opposite prior and was perfect.

The classes were maximally separated. Only the *label assignment* varied.

> **A near-perfect score in either direction means a shortcut, not skill.** A
> trained classifier learns the sign from the labels; it would score ~99% on this
> text and then fail completely on ordinary AI output.

A successor prompt, `v9`, kept the inversion goal but removed the mechanism: the
stock phrases were banned by name, the markers were *drawn* at elevated-but-not-
absurd rates (25–30% rather than 87%), and each draw asked for the idea in the
model's own words so no fixed string recurs. It bought about a third of the
evasion at a quarter of the marker rate — real work done by register rather than
by stamp.

---

## Part 6: The measurement that mattered, and came last

Everything above is 2AFC: the judge sees the human answer and the AI answer side
by side.

**A real detector never gets that.** It meets one document, alone, with nothing
to compare it against.

So the project finally ran the realistic test: 150 documents, judged one at a
time, five per judge so no judge could reverse-engineer the generator from
repetition. Generated with plain old v6 — no inversion, no planted markers.

| metric | value | plain English |
|---|---|---|
| detection (recall) | **26.7%** | catches 1 AI answer in 4 |
| false positive rate | **20.0%** | falsely accuses 1 real person in 5 |
| precision | 57.1% | when it says "AI", it's right 57% of the time |
| balanced accuracy | **53.3%** | chance is 50% |

Haiku 4.5 cannot do this task on a short Reddit comment to any useful standard.

**This reframed the entire project.** Nine prompt versions had been spent pushing
a 2AFC number around. On the task that actually matters, the plain v6 prompt was
already at the floor — the adversarial work had bought nothing since version 6.
The 2AFC numbers were flattering because having a known-human reference beside
you is an enormous assist.

> **Measure the task you actually care about, first.**

### The full benchmark

Three detectors, two document lengths, 835 single-document judgments:

| length | detector | detects AI | false accusations | balanced accuracy |
|---|---|---|---|---|
| **short** (~101 words) | Haiku 4.5 | 26.7% | 20.0% | 53.3% |
| | Sonnet | 56.0% | 8.0% | 74.0% |
| | Pangram 3.3.2 | 77.3% | **0.0%** | 88.7% |
| **long** (~340 words) | Haiku 4.5 | 24.0% | 9.3% | 57.3% |
| | Sonnet | **78.7%** | 1.3% | **88.7%** |
| | Pangram 3.3.2 † | 97.6% | 0.0% | 98.8% |

† Pangram's long-form figure comes from an earlier, defective version of the
long-form set and was not re-run on the clean rebuild.

Three things fall out of this table:

**Detector tier is enormous.** Sonnet beats Haiku by 54.7 points on long form.
On short form the gap is 29.3 points. These are not the same instrument.

**Length is decisive — but only for a detector capable of using it.** Sonnet
gains 22.7 points going from short to long documents. Haiku gains nothing
(−2.7). Testing on the weakest detector hid the effect entirely, and the project
wrote down "length is not a lever" for a while as a result.

**Pangram's published claims replicate — at the length it recommends.** The
commercial detector advertises ~98% detection at ~0.05% false positives. At
390 words we measured 97.6% at 0.0%. At 101 words the same model drops to 77.3%.
Their 250-word minimum is not marketing; it is where the model works. Zero of
119 genuine redditors were flagged at either length.

### Why long documents do not save the humans

Long ELI5 answers come disproportionately from people writing inside their own
profession — an anesthetist on sedation, a nurse, a logistics worker. Expert
prose is organised. Detectors rejected humans for "suspiciously perfect
hierarchical structure", "too systematically comprehensive", "systematic
organization".

> **Length does not make a human look more human; it makes them look more
> professional, and professional reads as synthetic.**

---

## Part 7: The embarrassment — `wc -l`

Late in phase 1, a second person auditing the corpus for an unrelated reason
counted the line breaks in each document.

That is the whole classifier. Count newlines. Fewer newlines means AI.

| set | AUC of "count the line breaks" |
|---|---|
| `long` | 0.862 |
| `long3` | 0.803 |
| `long2` | 0.802 |
| `short` | 0.760 |

*(AUC — "area under the curve" — is the probability that a randomly chosen AI
document scores higher than a randomly chosen human one. 0.5 is a coin flip, 1.0
is perfect.)*

On a single fitted threshold — `newlines <= 8` means AI — this achieves 74.0%
balanced accuracy on the long-form set. **Claude Haiku 4.5 gets 57.3% on the same
150 items.**

Once you match the denominators properly on short form, `wc -l` gets 74.9% and
Haiku gets 50.4% — chance, to one decimal place.

The structure behind it, in median values on the long-form set:

| | newlines | paragraphs | words per paragraph |
|---|---|---|---|
| human | 11 | 6 | 59 |
| qwen | 8 | 5 | 65 |
| deepseek | 6 | 4 | 99 |
| **grok** | **1** | **1** | **275** |

Grok's median long-form answer is a single unbroken block of text. On short form
it is one block 100% of the time, against a human rate of 25%.

### Why this happened

This is not a new kind of failure. It is Part 2's failure in a dimension nobody
thought to measure. Word count was rate-matched. Markdown was rate-matched.
`Edit:`/`TL;DR` were rate-matched. Persona was rate-matched. **Paragraph
structure was never considered**, and it turned out to be the most separable
surface feature in the entire corpus.

The generation prompt says nothing at all about paragraphs. So this is not a
prohibition driving a feature to zero — it is a plain omission.

### What it does and does not invalidate

It does *not* retract the detector numbers. Split each class at its own median
newline count and the detectors' accuracy is flat in every cell — Sonnet gets
79% on AI documents with few line breaks and 78% on those with many. Sonnet is
not a whitespace detector in disguise.

What it invalidates is the *reporting*. Every detector figure in the project
should have carried the trivial baseline beside it, and none of them did.

> **Quote the dumbest possible baseline next to every detector number.** A
> detector result without one is not a result; it is a number.

### How it was found

Two audits by the person who wrote the sections it undermines missed it. A second
reader with the per-item data in front of them for an hour also missed it. It
surfaced only when the two compared notes across *different goals*.

> **Get a second reader with a different objective — not a more careful one.**
> Care was not the missing variable. What neither reader had alone was a reason
> to look at that column.

---

## Part 8: The small detector

A separate strand asked whether a model small enough to run on one consumer GPU
could do the single-document task.

`LiquidAI/LFM2.5-1.2B-Instruct` — 1.17 billion parameters, one forward pass,
**zero generated tokens** (it reads the document and the score is read off
directly, nothing is written). It scored **AUC 0.771** on a confirmation set that
was built and frozen before any tuning, up from 0.725 on the development set.
Going *up* from dev to confirmation is the direction that does not happen when a
result has been quietly fitted.

`wc -l` scores 0.803 on the same items. Both numbers are reported together,
because they measure different things: 0.80 is a formatting-convention gap, 0.77
is prose.

The transferable result here is about prompting, not the score:

- **Seven classification prompts all came out inverted** — including one that
  listed only human cues, and one that explicitly warned against the trap.
- **Ten descriptive yes/no prompts all came out correct.**

At 1.2B parameters, the prompt controls *which feature the model attends to*, but
not *which label it assigns that feature to*. The practical conclusion: ask a
small model for a description, and do the mapping to labels yourself in code.

---

## Part 9: The bug in the sample builder

An audit found the long-form benchmark was not measuring what it claimed. The
item builder appended the human sibling document once per AI row drawn, inside a
loop over models — so when a question was picked by two or three models, its
human answer was emitted two or three times as separate "items".

**The human arm was 39 distinct documents presented as 75 items.** A false
positive rate computed that way borrows precision it has not earned.

Three smaller defects came with it: six of thirty judging batches accidentally
contained an AI answer and the human answer to the *same question* (in a test
whose entire premise is that no reference text exists); the source pool held 65
duplicate rows; and the answer key was sitting in the directory handed to the
judges.

The benchmark was rebuilt from scratch with a corrected builder that asserts all
of this before writing anything. The result:

**Nothing moved.** Every difference between the defective and clean runs is
within noise (all four p-values between 0.36 and 0.79). Every conclusion the
benchmark rests on survived.

That is worth more than the numbers being right the first time. A result that
survives having its foundation replaced is a stronger result.

> **Audit the thing that built your sample, not just the thing that scored it.**
> The scorer was correct, the judges were correct, the statistics were correct,
> and the human arm was still half the size it reported.

---

## Where phase 1 ended

**Built and working:** corpus fetch, sampling (short and long form), generation
with per-model prompts and reproducible seeded draws, dataset assembly with
question-grouped splits, judge harness, scorer, power calculator, eval
partitioner.

**The production dataset was never generated.** `data/final/` is empty. The
project turned into a detector evaluation instead, which was the more useful
outcome. The scripts to produce it exist and are tested.

**What exists:** 1,000 short-form and 1,000 long-form questions with verified
pre-2022 human answers; nine prompt versions with the measurement behind each
change; 2,134 generated AI answers; and a labeled single-document benchmark
covering three detectors at two lengths.

**The largest known defect:** paragraph convention separates the classes at
AUC 0.78–0.86 in every set built. The fix — rate-match paragraph count and mean
paragraph length per `(question, model)`, the way word count already is — was
identified but not implemented.

**Where the headroom is:** detection at ~390 words is close to solved (Pangram
97.6%, Sonnet 78.7%). Short-form is not. The best available recall on a
100-word comment is 77.3%, and Grok short-form sits at 64% against a commercial
detector and 32% against Sonnet. A detector aimed at Reddit comments, chat
messages and forum replies — the length most real text actually is — has genuine
room.

---

## The lessons, condensed

The full list runs to 25 entries at the end of `phase1/NARRATIVE.md`. These are
the ones that would have saved the most time:

1. **Measure the task you actually care about, first.** Nine prompt versions
   optimised a score no real detector ever sees.
2. **A perfect zero is a fingerprint.** Match rates; do not prohibit.
3. **A near-perfect score in either direction means a shortcut, not skill.**
4. **Quote the trivial baseline.** Every detector number here went months
   without one, and the baseline was `wc -l`.
5. **Report the spread, never the mean alone.** Judges at 45/48/55 and judges at
   22/53/91 average the same and mean opposite things.
6. **Compute the smallest detectable difference before running, not after.**
7. **A variance estimated from three points is not evidence.**
8. **Run the cheap experiment first.** Three API calls settled a question that
   had produced two wrong theories.
9. **A silent parse failure is a silent sample-size cut.** Three reported figures
   were computed on fewer items than they claimed, invisibly.
10. **Get a second reader with a different objective.**

---

## Glossary

**2AFC** — two-alternative forced choice. The judge sees one human and one AI
answer to the same question and picks the AI. Chance is 50%. Easier than the real
task, because the reference answer is a huge assist.

**AUC** — area under the ROC curve. The probability that a randomly chosen AI
document scores higher than a randomly chosen human one. 0.5 = coin flip,
1.0 = perfect. Does not depend on where you set the decision threshold.

**Balanced accuracy** — the average of "fraction of AI caught" and "fraction of
humans correctly left alone". Chance is 50% regardless of how the classes are
balanced.

**Detection rate / recall** — of the AI documents, what fraction were flagged.

**False positive rate** — of the human documents, what fraction were wrongly
flagged as AI. In deployment this is the number that matters most, because it
counts real people falsely accused.

**Precision** — of the documents flagged as AI, what fraction really were.

**Rate matching** — injecting a feature into the AI class at the same frequency
it occurs in the human class, rather than banning it.

**Standard error (SE)** — how much a measured percentage would bounce around if
you re-ran the same experiment. At 30 coin-flip trials it is about 9 points,
which is why 30-trial comparisons in this project were meaningless.

**Minimum detectable difference** — the smallest true effect a study design can
reliably find. Compute it before running the study.

**Holm correction** — an adjustment applied when you run many statistical tests
at once, because running twenty tests at the 5% level will produce roughly one
false positive by chance.

**Wilson interval** — a way of putting error bars on a percentage that behaves
correctly near 0% and 100%, where the naive formula does not.
