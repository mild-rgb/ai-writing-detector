# The AITA sub-project

A second labelled human-vs-AI subreddit corpus, to sit beside the ELI5 corpus so
a frozen bag-of-words model can be cross-applied between two domains and the
detector debiased off lexical shortcuts with Product-of-Experts. r/AmItheAsshole
was chosen deliberately for register distance: judgement narrative, not
explanatory Q&A, so the lexical shortcut genuinely differs.

## What was ported, and what was not

The generator ported here is **`phase3/corpus/floor`** — the prompt that
actually produced the shipped ELI5 AI class. It is v6's register constraints
with the em-dash prohibition and the first-person suppression converted to
rate-matched draws, plus the LLM-lexicon blacklist, and **no structural
instruction of any kind**.

`phase3/adversarial_prompts/p1..p6` were deliberately NOT ported. That
sub-project was halted after v2 (`NARRATIVE.md` §9) because closing the surface
shortcuts made the text *easier* for a judge to catch, not harder. On the ELI5
dev split Haiku's balanced accuracy runs 63.9% (p4) to 82.2% (p2) against the
floor's 57.9%. Porting p1_pinned would have ported the worst of them.

### A bug those runs carry, which this tree does not

`generate.py` reads `prompt.txt` verbatim; it never expands `__FLOOR__`.
`phase3/corpus/floor/prompt.txt` is fully resolved on disk, so the corpus run
was correct. Every `adversarial_prompts/pN/prompt.txt` still begins with a
literal `__FLOOR__` line, and `make_configs.py` writes the resolved text to
`prompt.resolved.txt`, which nothing reads.

Measured: p1_pinned's documents contain the floor's own banned words in **10.7%**
of 420 documents, against **1.6%** of the corpus floor's 2,900. Those six runs
were generated with no floor constraints at all — no register rules, no
blacklist, no anti-flourish rules — only their own shape block plus the draws.

This confounds the conclusion that retired the campaign: p1 differed from the
floor in far more than paragraph structure, so 79.2% against 57.3% is not a
clean measurement of the paragraph pin. Flagged, not acted on.

Here `template.txt` holds the `__FLOOR__` marker and `aita_03_make_configs.py`
writes the **resolved** text to `prompt.txt`, with an assert that no marker
survives.

## The human class

`phase1/scripts/01_fetch_eli5.py` streams the Facebook LFQA dump and keeps one
subreddit. That file cannot be reused: streamed and counted, it holds
explainlikeimfive 59,714 / askscience 31,933 / AskHistorians 28,424 and no AITA.

The same *kind* of source exists. **Scruples Anecdotes** (Lourie, Le Bras & Choi
2020, arXiv:2008.09094) is 32,766 r/AmItheAsshole posts in original case,
carrying the reddit post id.

**Dates.** Scruples ships no timestamp. Dates are recovered by interpolating
base36 post ids against 36,896 `(id, timestamp)` pairs from a dated AITA dump.
Every post resolves to **2019-04-06 or earlier**, three years and seven months
before ChatGPT — the same true-by-construction label guarantee the ELI5 class
has.

**Why not the larger dumps.** The two biggest AITA sets on HuggingFace
(`MattBoraske/reddit-AITA-submissions-and-comments-*`,
`OsamaBsher/AITA-Reddit-Dataset`) are **fully lowercased** in the body field —
measured uppercase fraction 0.000 against normal prose's 0.03. A human class
that is entirely lowercase against a normally-cased AI class is a perfect
shortcut, and exactly the kind of surface tell this project exists to remove.
Scruples measures 0.0296.

**Filter**, mirroring phase 1's strict long-form filter: 250-800 words,
HISTORICAL only (WIBTA hypotheticals are a different register), at least 10
community judgements, no `[removed]`/`[deleted]`, deduped on a normalised title.
Yield **6,343 posts**, median 391 words — 2.2x the ELI5 pool of 2,902.

## The rates were re-measured, not reused

AITA writes nothing like ELI5. Reusing the ELI5 config would have mis-set almost
every draw:

| marker | ELI5 | AITA |
|---|---|---|
| first person I/my | 56.9% | **99.2%** |
| personal experience | 1.5% | **43.1%** |
| question mark | 36.1% | **83.1%** |
| mild profanity | 13.9% | 28.5% |
| own hedge | 10.8% | 26.9% |
| curly quote | 3.9% | **31.5%** |
| TL;DR | 7.7% | 13.1% |
| markdown emphasis | 32.3% | **9.2%** |
| `Edit:` line | 13.8% | **1.5%** |
| numbered list | 7.7% | 1.5% |

The curly-quote rate is a real signature of 2019 AITA rather than noise: a
mobile-heavy sub, and mobile keyboards produce smart quotes throughout a
document. It is handled by the existing all-or-nothing `op_curly` post-op, which
converts every quote or none — a document with one curly apostrophe among
straight ones displays an edit, not a habit.

The `Edit:` rate is the one that contradicts intuition. The marker regex is
case-insensitive, so `EDIT:` and `UPDATE:` would both have matched; 1.5% is what
the 2019 corpus actually contains. Worth revisiting if Scruples captured post
bodies before edits accumulated.

### One mismatch that must NOT be prompted away

At n=59 the AI class trips the `personal experience` marker in 55.9% of
documents against a human 43.1%. That marker is a regex over
`my (job|wife|husband|dad|mom|brother|sister|...)`, and in AITA those nouns are
the subject matter -- a model cannot be instructed away from mentioning the wife
when the title is about the wife.

It is left in deliberately. The generator's job is to strip *spurious mechanical*
shortcuts -- markdown habits, stock vocabulary, whitespace -- so that a detector
is forced onto behaviour rather than bookkeeping. "The AI over-names the
relationships in the scenario" is behaviour, and it is exactly the kind of signal
the detector is supposed to be able to learn. Prompting it away would be teaching
to the test. It is recorded on the marker list and left alone.

Instruction *wording* was rewritten too, not just the rates. "Ground one claim in
something you have seen or done yourself" is a strange thing to ask of a post
that is nothing but firsthand experience, and the 83.1% question rate is mostly
the post asking for its verdict, not an ELI5 aside to the reader.

## Judging

Same instrument as phase 1 and phase 3, so the numbers are directly comparable:
single documents, five to a batch, no reference text, `anthropic/claude-haiku-4.5`
over the API. The instruction differs from the ELI5 one **only in the domain
nouns** (diffed in the commit message). `aita_04_build_judge.py` keeps every
invariant `build_bench.py` asserts, including that no batch may contain two
documents for the same question — that would be a 2AFC pair inside a benchmark
whose premise is that no reference text exists.

Three passes over identical documents, reshuffled per pass. Balanced accuracy is
the headline with the SD beside it; detection and false-positive rates are
reported underneath, never alone. `judge_compare/RESULTS.md` is why: the same
route re-run on the same items moved detection 13 points and balanced accuracy 2.

**The reference point is the floor's 57.9% (SD 2.6), not 50%.** No ELI5 prompt
ever reached chance, and `NARRATIVE.md` §7 notes that 140 documents over five
passes pins "at chance" only to within about eight points.

## Result

The ported floor transfers to AITA without retraining. Dev split, 59 ai + 59
human = 118 items, 24 batches of 5, three independent `claude-haiku-4.5` passes
over the API — the same instrument and the same shape as
`phase3/data/judge_dev_p*.json`, so these numbers sit beside the ELI5 ones
without adjustment.

| pass | detection | fpr | balanced |
|---|---|---|---|
| pass0 | 23.7% | 15.3% | 54.2% |
| pass1 | 13.6% | 16.9% | 48.3% |
| pass2 | 13.6% | 18.6% | 47.5% |

**balanced accuracy 50.0%, SD 3.7, range [47.5, 54.2]**
detection 16.9% (SD 5.9), false positives 16.9% (SD 1.7)
unanimous across passes: 70.3% of 118 documents

Against every ELI5 prompt on the identical instrument:

| prompt | balanced | SD |
|---|---|---|
| **AITA floor (this port)** | **50.0%** | 3.7 |
| ELI5 corpus floor (shipped) | 57.9% | 2.6 |
| p4_voice | 63.9% | 1.7 |
| p5_disfluency | 72.5% | 4.4 |
| p3_shape | 74.2% | 3.0 |
| p6_scope | 74.2% | 2.2 |
| p1_pinned | 79.2% | 1.4 |
| p2_mechanical | 82.2% | 1.7 |

Detection and false positives are equal at 16.9%, which is what chance looks
like from the inside: the judge calls "ai" on about a sixth of what it reads
whatever the label actually is.

**Read per generator, never pooled.** The 50.0% hides the same spread ELI5 had:

    nemotron-3.5-lightning   54.2% [35.1, 72.1]
    gpt-5.6-luna-pro         29.2% [14.9, 49.2]
    gemini-3.7-flash         14.8% [ 5.9, 32.5]
    glm-5.3                   8.3% [ 2.3, 25.8]
    grok-4.6                  8.3% [ 2.3, 25.8]
    deepseek-v4-pro           3.7% [ 0.7, 18.3]
    qwen3.8-max               3.7% [ 0.7, 18.3]

A 50-point spread across generators, the same shape as the ELI5 corpus's 7-80%.
A detector's accuracy on this corpus will mostly be a statement about which
generators its evaluation slice over-weights.

## Bench result — the frozen confirmation

70 ai + 70 human = 140 items, 28 batches of 5, three independent
`claude-haiku-4.5` passes. The `bench` split is disjoint from `dev`, was never
generated against before this run, and no prompt decision was ever made on it.
This is the number that describes the current generator.

| pass | detection | fpr | balanced |
|---|---|---|---|
| pass0 | 21.4% | 11.4% | 55.0% |
| pass1 | 18.6% | 17.1% | 50.7% |
| pass2 | 18.6% | 20.0% | 49.3% |

**balanced accuracy 51.7%, SD 3.0, range [49.3, 55.0]**
detection 19.5% (SD 1.6), false positives 16.2% (SD 4.4)
unanimous across passes: 73.6% of 140 documents

Against the ELI5 corpus floor's 57.9% (SD 2.6), this is floor-parity or better
on a frozen split. It is NOT a claim to beat ELI5 by 6 points: 140 documents
over three passes resolves "at chance" to roughly ±8 (`NARRATIVE.md` §7), so
51.7% and 57.9% are not separated by this measurement.

### Per generator, and this is the headline

    nemotron-3.5-lightning   63.3% [45.5, 78.1]
    gemini-3.7-flash         36.7% [21.9, 54.5]
    deepseek-v4-pro          13.3% [ 5.3, 29.7]
    gpt-5.6-luna-pro         10.0% [ 3.5, 25.6]
    qwen3.8-max               6.7% [ 1.8, 21.3]
    grok-4.6                  6.7% [ 1.8, 21.3]
    glm-5.3                   0.0% [ 0.0, 11.4]

A 63-point spread, wider than dev's 50 and the same shape as the ELI5 corpus's
7-80%. The pooled 51.7% describes none of these seven generators. Any detector
accuracy quoted on this corpus is mostly a statement about which generators its
evaluation slice over-weights.

`glm-5.3` at 0.0% is the one to watch rather than celebrate. Haiku caught none of
its 30 documents across three passes, and glm is simultaneously the generator
furthest off its length target (1.138 against the paired human document). Both
facts point the same way and neither is resolved at n=30; it wants a look before
the corpus is built, not after.

### That dev number belongs to a superseded prompt

The 50.0% above was measured on the prompt as first ported. Two fixes were
applied afterwards, so the dev figure is a record of the port transferring, not
a description of the current generator. `dev/pass0..2` under
`prompts/aita_floor/` are kept as that record. The current prompt's number is
the bench one.

**Fix 1 — the gpt-5.6-luna-pro length constant.** Measured over the paired human
document, six of seven generators landed within ELI5's published 0.94-1.01 band;
`gpt-5.6-luna-pro` sat at 1.142, with a true AITA production ratio of 1.158
against the ELI5 constant of 1.014. Its constant alone was corrected to 1.158.
The other six were left as measured rather than refitted, because refitting seven
constants on 60 documents fits noise.

The write is to a NEW `aita_floor` key in `length_calib.json`, which is why the
prompt directory was renamed: it was called `floor`, and `floor` is already the
key holding ELI5's own constants. Writing AITA calibration under that name would
have silently overwritten the shipped ELI5 corpus's calibration.

The new entry copies all seven of ELI5's floor constants and changes only gpt. A
partial entry would have been worse than none: `generate.py` falls back to its
hardcoded `LENGTH_CALIB` for any model missing from the cell, and those defaults
differ sharply from the floor-measured ones — grok 0.66 against 0.975, a 48%
length inflation on a generator that was already correct.

**Fix 2 — the verdict question.** Human posts carry a question mark 83.1% of the
time; the generated class managed 50.8%, a 32-point gap and the only marker
clearly out at n=59. The draw was already two-sided at the human rate, so the
draw was not the problem. The floor's own "do not end on a flourish... stop when
you have said what happened" was fighting it, and the models obeyed the
prohibition over the instruction.

The fix is a reconciliation in `_floor.txt`, not a stronger draw: asking the sub
whether you were in the wrong is now explicitly exempted from the anti-flourish
rule, because it is the genre's defining convention rather than a summarising
line. Strengthening the draw on its own would have bolted a verdict question onto
a post the floor had just told the model to end abruptly — a worse tell than the
one being closed.

### Caveats, stated rather than buried

* **This is a dev pass, descriptive only.** No prompt version was ever selected
  on it — the ported floor was measured once, as-is — so it is less contaminated
  than a tuned dev number, but it is still not the frozen `bench` confirmation.
* **n=118 over three passes pins "at chance" to roughly ±8 points** (`NARRATIVE.md`
  §7). 50.0% and the floor's 57.9% are not separated by this measurement.
* **The false-positive rate is the thing that moved**, not detection. 16.9%
  against ELI5's 5-10%: Haiku calls genuine 2019 redditors AI about one time in
  six on AITA. Some of the balanced accuracy at chance is bought by the human
  class being harder, not only by the AI class being better.
* One nemotron document failed with an empty completion (its documented failure
  mode), so the arm is 59 rather than 60.

## Do not try to save money by capping output tokens

Measured on this roster, August 2026. An opener-only dry-run was proposed —
generate the first 30 words of each corpus document to test scenario diversity
at roughly a tenth of the cost — and it does not work. Reasoning runs before the
visible answer and is billed whether or not the cap allows any content through.

At `max_tokens=160`, with `reasoning.exclude` set:

| model | visible words | completion tokens billed |
|---|---|---|
| qwen3.8-max | 0 | 162 |
| deepseek-v4-pro | 0 | 160 |
| glm-5.3 | 0 | 160 |
| gpt-5.6-luna-pro | 0 | 526 |
| gemini-3.7-flash | 3 | 156 |
| grok-4.6 | 141 | 4,183 (ignored the cap) |
| nvidia/nemotron-3.5-lightning | 140 | 160 |

Five of seven return nothing usable and charge for it. `reasoning.exclude` only
hides reasoning from the response; it does not stop the model doing it. Raising
the cap does not restore the economics either: gemini at 400 tokens yields 12-14
visible words for a full 396 billed, and gpt at 400 produced a complete 251-word
post billing 1,733.

The failure is also visible in the text rather than only the counts. gemini's
truncated output came back as fragments like `/language model mention. (Yes` and
`neutrality). *` — the model reasoning ABOUT the prompt's own constraints, cut
mid-thought and surfacing as content. That is `generate.py`'s documented
nemotron failure (whole budget on reasoning, empty completion) in a milder form,
and unlike nemotron's it is not fixed by the override.

Reasoning can sometimes be disabled outright, but not usefully here: qwen and
gemini reject it with "Reasoning is mandatory for this endpoint and cannot be
disabled", and the project's standing instruction is that reasoning stays on.

**The consequence for planning: length is not a cost lever on this roster, count
is.** A 30-word opener costs what a full document costs. Any sampling-based
early read must reduce the NUMBER of documents, never their length.

## Final bench — the fixed prompt

Same frozen 70-question split, 140 items, three Haiku passes, after the glm
length constant and the verdict-phrasing replacement.

| pass | detection | fpr | balanced |
|---|---|---|---|
| pass0 | 18.6% | 12.9% | 52.9% |
| pass1 | 22.9% | 21.4% | 50.7% |
| pass2 | 14.3% | 20.0% | 47.1% |

**balanced accuracy 50.2%, SD 2.9, range [47.1, 52.9]**
detection 18.6% (SD 4.3), false positives 18.1% (SD 4.6)
unanimous across passes: 72.9% of 140 documents

Parity held through the fixes: 51.7% before, 50.2% after, against the ELI5
floor's 57.9%. The two are inside each other's spread, which is the point — the
fixes were rate corrections, not detectability work, and they did not move the
headline.

Detection by generator, pooled:

    nemotron-3.5-lightning   46.7% [30.2, 63.9]
    gemini-3.7-flash         30.0% [16.7, 47.9]
    deepseek-v4-pro          26.7% [14.2, 44.4]
    gpt-5.6-luna-pro         16.7% [ 7.3, 33.6]
    glm-5.3                   6.7% [ 1.8, 21.3]
    grok-4.6                  3.3% [ 0.6, 16.7]
    qwen3.8-max               0.0% [ 0.0, 11.4]

The spread narrowed from 63 points to 47, and the identity of the extremes moved
— glm was 0.0% on the previous bench and is 6.7% here, while qwen went 6.7% to
0.0%. At n=30 per generator those are the same number. Report per generator, and
do not read the ordering as stable.

## The length band, as reported rather than corrected

The tripwire fired and was honoured. After correcting glm to 1.24:

| model | calib | bench 1 | bench 2 |
|---|---|---|---|
| grok-4.6 | 0.975 | 0.936 | 0.906 |
| deepseek-v4-pro | 1.180 | 0.947 | 0.967 |
| gpt-5.6-luna-pro | 1.158 | 0.958 | 0.978 |
| qwen3.8-max | 1.001 | 0.990 | 1.005 |
| gemini-3.7-flash | 0.992 | 1.014 | 1.006 |
| glm-5.3 | 1.240 | 1.138 | 1.018 |
| nemotron-3.5-lightning | 0.850 | 0.920 | 1.079 |

Pooled 0.994, band **0.906 to 1.079**, against ELI5's published 0.94-1.01.

glm came down as intended. Two others moved out — and nemotron moved from 0.920
to 1.079 on an **unchanged** constant, a swing as large as the drift the two
corrections were made to remove. That is the tripwire's own justification
arriving on schedule: at n=10 per generator these constants are being fitted to
sampling noise as much as to any stable property. Band reported, no third
constant.

## The verdict fix worked pooled, and cancels per model

After the replacement, the canonical phrase spelled out sits at 21.4% against a
human 27.7% — intervals overlapping, and the planted "tell me if I" is back to
0.0% against a human 0.0%. Pooled, the fix did what it was meant to.

Per generator it did not:

    qwen3.8-max              70.0%
    gemini-3.7-flash         60.0%
    glm-5.3                  20.0%
    deepseek-v4-pro           0.0%
    nemotron-3.5-lightning    0.0%
    gpt-5.6-luna-pro          0.0%
    grok-4.6                  0.0%
    per-model SD 30.8 points; human 27.7%

No generator is near the human rate. Four sit at exactly 0.0% and three well
above, and the pooled 21.4% is the average of two behaviours rather than a
description of any of them. This is `PROTOCOL.md` criterion 2 — "a blend can
cancel", a pooled figure earned by models on opposite sides of the human value
is an artifact, not a match — and it brushes criterion 5, which forbids a marker
at exactly 0.0% where the human rate exceeds 5%.

It is an improvement on what it replaced: before the fix the pooled rate was
5.7% with five generators at zero. The permit-not-mandate wording moved two
models a long way and left four untouched, which is phase 1's own finding that
one prompt cannot calibrate several models, showing up in a single marker.

Not acted on. The agreed stopping rule fires on pooled overshoot, and it did not
fire; the rule also allows one correction with no second bench. Recorded here as
a known property of the corpus rather than fixed, because the honest options are
a per-model draw rate (which the project has never done) or accepting it and
reporting per generator (which is what the project does everywhere else).

## Early-read diversity check (1,050 documents, 150 per generator)

A seeded random draw from the frozen corpus assignment, generated with the fixed
prompt. These are the corpus's first third, not a throwaway. Every AI figure is
paired with a human figure computed the same way, on the SAME questions, at the
SAME n, because both retrieval and nearest-neighbour similarity move with set
size.

### 1 and 2 — scenarios are diverse. Pass.

| generator | n | top-1 ai/human | top-5 ai/human | nn cos ai/human | near-dups |
|---|---|---|---|---|---|
| deepseek-v4-pro | 149 | 28.9 / 32.9 | 52.3 / 54.4 | 0.198 / 0.178 | 0 / 0 |
| gemini-3.7-flash | 150 | 20.0 / 29.3 | 34.0 / 44.7 | 0.225 / 0.166 | 0 / 0 |
| nemotron-3.5-lightning | 150 | 12.7 / 24.0 | 26.0 / 42.7 | 0.188 / 0.178 | 1 / 0 |
| gpt-5.6-luna-pro | 150 | 47.3 / 26.7 | 68.0 / 50.0 | 0.217 / 0.164 | 0 / 0 |
| qwen3.8-max | 150 | 31.3 / 23.3 | 50.0 / 43.3 | 0.214 / 0.173 | 0 / 0 |
| grok-4.6 | 150 | 24.7 / 32.7 | 44.7 / 50.0 | 0.206 / 0.160 | 0 / 0 |
| glm-5.3 | 150 | 30.7 / 28.7 | 44.7 / 44.0 | 0.213 / 0.172 | 0 / 0 |

Random top-1 baseline is 0.67%. Every generator retrieves its own title at 19x
to 71x that, so the scenarios really are being taken from the 6,343 real titles
rather than invented by the model. One near-duplicate pair in 1,049 documents.

Two generators sit outside the human range in opposite directions:
`gpt-5.6-luna-pro` retrieves at 47.3% against a human 26.7%, so it stays closer
to its title than a person does, and `nemotron-3.5-lightning` at 12.7% against
24.0% drifts further from it. Neither is recycling; they are hugging and
wandering. AI nearest-neighbour cosine is consistently 0.03-0.06 above human,
which is a real but small uniformity, and no model shows an attractor.

### 3 — character names are a genuine tell. Fail.

Measured per document, because that is what a detector sees: is this one word
present or absent?

| generator | most-used name | share of its own documents | names per document |
|---|---|---|---|
| gemini-3.7-flash | Sarah | **19.3%** | 1.19 |
| glm-5.3 | Dana | 13.3% | 0.55 |
| qwen3.8-max | Dana | 9.3% | 0.64 |
| grok-4.6 | Jake | 8.0% | 0.63 |
| nemotron-3.5-lightning | Maya | 6.0% | 0.28 |
| deepseek-v4-pro | Jake | 4.0% | 0.47 |
| gpt-5.6-luna-pro | Italian | 2.7% | 0.16 |
| **human, same titles** | German | **0.5%** | 0.23 |

Real posters essentially never reuse a name: 201 distinct names in 253 mentions,
and the most repeated one appears in one document in two hundred. gemini uses
"Sarah" in nearly one document in five — **41x the human rate** — and its top
five names account for 56.5% of every name it writes.

**This one should be fixed, and it is not the same case as `personal
experience`.** That marker is content-bound: a model cannot avoid saying "my
wife" when the title is about his wife, and the over-naming of relationships is
behaviour a detector deserves to learn. A character's name is free. Any name
works equally well, so a recurring proper noun is a shallow lexical shortcut of
exactly the family the LLM-lexicon blacklist exists to remove -- the "delve"
problem wearing a different hat. A detector that learns "Sarah implies machine"
has learned nothing about writing.

Note the mirror case: `gpt-5.6-luna-pro` writes 0.16 names per document against
a human 0.23 and its "top name" is not a name at all. It under-names rather than
over-naming. Absence and excess are both detectable.

### The scope error that understated the whole problem

The first report of this said "gemini uses Sarah in 19.3% of its documents". That
number came from the 30-word openers, measured with the raw capitalisation
heuristic. Both choices were wrong in the same direction.

Re-measured on FULL documents with the curated name list, the picture changed
shape rather than degree. The share of documents that name a person at all:
gemini 84.0%, grok 56.0%, glm 51.3%, deepseek 40.9%, qwen 40.7%, nemotron 30.7%,
gpt 12.7% — against a human 10.7%.

**That changed what the fix had to be.** On the opener measurement the problem
looked like name CHOICE — the same "Sarah" recurring — which a one-sided
"if you name someone, call them X" would have addressed. On the full-document
measurement the dominant problem is the naming RATE, and a one-sided draw would
have fixed which name appeared while leaving gemini naming a character in five
documents out of six. That is why the draw is two-sided.

It also resolved the objection that a two-sided draw would hurt `gpt`, which
under-names: gpt measured 12.7% against a human 10.7%, so a draw at the human
rate leaves it where it is by construction, and no special case is needed.

Scenario choice does happen in the opening sentences, so 30 words was the right
window for the diversity measures. It is the wrong window for anything that
accumulates through a document, and naming is one of those.

### The bug that nearly hid this

The first run put "Ive" at the top of the human name list with 19 uses. That is
"I've": `op_curly` rewrites apostrophes as U+2019 in a third of documents, and
stripping that as punctuation leaves a capitalised word that looks like a name.
It inflated the human unique-name count and made the human class look more
name-diverse than it is, which would have made the AI concentration look
comparatively milder. Curly apostrophes are now normalised before extraction and
contractions are dropped.

## Fixing the name tell: two routes, and why the cheap one is not the simple one

Not yet decided. Recorded so the analysis is not re-derived.

**Route A — regenerate with a per-document name draw.** A name is supplied by a
draw seeded on `(id, model)` like every other feature in `config.json`, so the
rate is matched and the name varies per document rather than by instruction.
Permit, do not mandate: an instruction to "use varied names" would push
`gpt-5.6-luna-pro` further from human, since it already UNDER-names at 0.16 per
document against a human 0.23. Costs roughly $5 to redo the 1,050. No surgery.

**Route B — post-hoc name replacement across the whole corpus.** Cheaper: no API
spend. It must be applied UNIFORMLY to all 2,900 documents, never to a subset,
because a partial fix would put some generators on one prompt version and the
rest on another, and phase-4 attribution could then win on the prompt-version
artifact instead of model style — the shortcut failure this project exists to
remove, reintroduced by the fix for it.

Four requirements, each measured on the 1,049 documents rather than assumed:

1. **The extractor in `aita_08_diversity.py` is not a person-name detector.** It
   yields 582 distinct tokens, of which at least 25 types and 156 occurrences are
   not people: Italian 22, Biscuit 17 (a pet), Ohio 16, Chicago 15, Denver 13,
   Portugal 8, Netflix 8, Tinder 8, Honda 7, Nazi 6. A remap driven by it would
   rewrite "Italian" to "Jake". Any remap must use a curated gendered person-name
   list and touch only tokens on it.
2. **Gender cannot be read from the document.** Of documents containing a name,
   154 of 268 contain both `she/her` and `he/him`, so neighbouring pronouns do
   not identify whose name it is. Remap within a gender class; map unisex to
   unisex. Note that `Dana`, the top name for both glm and qwen, is unisex.
3. **21 of 1,688 name mentions also appear in the document's own title.** Those
   must be preserved or the post stops agreeing with its title.
4. **Match the human shape, do not flatten.** Humans are Zipfian: 201 distinct
   names in 253 mentions, top name in 0.5% of documents, top-5 share 6.7%. A
   remap giving every document a unique name overshoots into more diverse than
   any real population — v4's overshoot in a new place. And `gpt` is left alone.

**Route B requires a QA read; route A does not.** This is `PROTOCOL.md`'s central
warning in a new place. `p2_mechanical` passed every structural statistic while
its documents were torn, and no counter could see it. `op_curly` matched its
marker rate perfectly while manufacturing a fingerprint. A name remap is the same
category of mechanical transformation: it passes the name-concentration statistic
**by construction**, and can still leave "Sarah said she'd..." beside a replaced
"Marcus", a possessive that no longer agrees, or a nickname the mapping missed.
Nothing countable sees any of it; a reader sees it at once.

So route B is gated on a fresh subagent QA read over a stratified sample — asked
what is WRONG with each document, never who wrote it, with human documents mixed
in unlabelled as the credibility control. Clean read, remap stands. Torn
referents, and the remap is worse than the tell it removed.

The conclusion that matters for choosing: **the cheap route is the complicated
one.** Route A costs about $5 and is finished when it finishes. Route B costs
nothing, needs a curated name list, gender handling, title special-casing, a
distribution target, and a reviewer pass to confirm it did not break the prose.

## The naming fix worked, and broke something else

The two-sided naming draw did what it was built to do. Measured on the
regenerated 1,049:

| generator | names someone | was | top name's doc share | was |
|---|---|---|---|---|
| gemini-3.7-flash | 9.3% | 84.0% | 0.7% | 38.0% |
| glm-5.3 | 12.0% | 51.3% | 1.3% | 19.3% |
| grok-4.6 | 8.7% | 56.0% | 1.3% | 12.7% |
| deepseek-v4-pro | 14.0% | 40.9% | 1.3% | 8.1% |
| qwen3.8-max | 6.7% | 40.7% | 0.7% | 14.7% |
| nemotron-3.5-lightning | 19.5% | 30.7% | 3.4% | 14.0% |
| gpt-5.6-luna-pro | 6.7% | 12.7% | 1.3% | 4.0% |
| **human, same titles** | **10.7%** | | **0.6%** | |

Pooled, the AI names someone in 11.0% of documents against a human 10.7%, and
the commonest name is "Sarah" in 0.57% of documents **in both classes**.
Scenario diversity was untouched, as expected. Near-duplicate pairs went from 1
to 6 in 1,049 — still 0.6%, but no longer zero, and worth watching per generator
rather than dismissing.

### And it planted a repetition tell

|  | n | rel terms /1k words | pronoun ratio | ≥6 terms | vs human |
|---|---|---|---|---|---|
| human | 937 | 6.68 | 8.00 | 15.0% | 1.00x |
| AI before the fix | 575 | 8.02 | 11.00 | 17.0% | 1.20x |
| AI after the fix | 934 | 11.88 | 6.54 | 32.4% | **1.78x** |

Told not to name anyone, the models restate "my wife" where a person switches to
"she". The pronoun ratio halved. Documents carrying six or more relationship
terms doubled. Worst is `gpt-5.6-luna-pro` at 15.68 per 1,000 words — the model
that suppresses names hardest pays for it hardest, which is itself evidence this
is a response to the instruction rather than a style.

**This is a trade, not a win.** "my wife my wife my wife" is as countable as
"Sarah", and cheaper for a detector to learn because it needs no name list.

### Why this one is fixable and `personal experience` is not

The same content-boundness test, applied one level down.

The relationship **mention** is content-bound: an AITA post is about the wife, so
"my wife" has to appear, and suppressing it would be teaching to the test. That
is why `personal experience` stays.

The **repetition** is not content-bound. Saying "my wife" six times rather than
once and then "she" conveys identical content — it is a free choice between two
ways of referring, which puts it in the same class as the banned stock
vocabulary.

The decisive number is that the models get this right when left alone: 1.20x
human before the fix, 1.78x after. The instruction broke it. This is the
induced-artifact class, like the "tell me if I" phrasing that a prohibition
planted, not the natural-behaviour class. Fixing it is repairing our own damage,
not stripping real signal.

### Stopping rule, locked before spending

**One more regeneration.** Verify all three numbers on the result — naming rate,
relationship-term density, pronoun ratio — against the human class. If they
converge, ship and finish the corpus. If the wording induces a further artifact,
or fails to converge, **stop tuning and report the residual** rather than iterate
a fourth time. Human behaviour gets described once; it does not get chased.
Same discipline as the length tripwire, and stated in advance for the same
reason.

The fix describes rather than mandates. An explicit "always use he or she" would
push the pronoun ratio past human in the other direction, which is v4's overshoot
yet again.

# THE FINISHED CORPUS — 2,900 documents

Complete and balanced: deepseek 415, qwen 415, gemini 414, glm 414, gpt 414,
grok 414, nemotron 414. One prompt across all 2,900, no version split. Generated
from the frozen assignment in `corpus_assignment.json`, drawn from the 6,153
pool with dev, bench and heldout excluded.

## The naming fix held at scale

| generator | names someone | top name | its doc share |
|---|---|---|---|
| nemotron-3.5-lightning | 19.6% | Maya | 2.9% |
| deepseek-v4-pro | 15.4% | Tom | 1.0% |
| glm-5.3 | 11.1% | Dana | 1.0% |
| gemini-3.7-flash | 9.9% | Jane | 0.7% |
| grok-4.6 | 8.5% | Jane | 0.7% |
| qwen3.8-max | 8.2% | Karen | 0.7% |
| gpt-5.6-luna-pro | 7.5% | Alex | 0.7% |
| **human, same titles** | **11.5%** | John | 0.6% |

Pooled: AI names someone in 11.4% of documents against a human 11.5%, with 142
distinct names against the human class's 222 and a commonest name appearing in
0.45% of documents against 0.59%. gemini went from 84.0% to 9.9%, and its "Sarah
in 38% of documents" is gone.

`nemotron-3.5-lightning` is the residual at 19.6%, 1.7x human, down from 30.7%.
It is the generator that most often does not do what the prompt asks.

## The length band tightened, which is the tripwire's real vindication

| generator | ratio to paired human |
|---|---|
| deepseek-v4-pro | 0.940 |
| grok-4.6 | 0.950 |
| gpt-5.6-luna-pro | 0.973 |
| glm-5.3 | 0.996 |
| gemini-3.7-flash | 1.007 |
| qwen3.8-max | 1.009 |
| nemotron-3.5-lightning | 1.029 |

Pooled 0.986, band **0.940 to 1.029** against ELI5's published 0.94-1.01. At
n=10 per generator the same measurement gave 0.906-1.079 and two constants
looked like they needed fixing. At n=414 the band is inside ELI5's to within
0.02. The width was sampling noise, exactly as the tripwire assumed, and
declining to fit a third constant was right.

# ACCEPTED RESIDUALS

Known properties of this corpus, reported rather than fixed. A detector trained
here can exploit any of them, and any evaluation should say whether it did.

## 1. Relationship-term repetition — the largest one

| class | terms /1k words | pronoun ratio | ≥6 terms |
|---|---|---|---|
| human (names nobody) | 6.63 | 8.00 | 15.4% |
| gpt-5.6-luna-pro | 15.29 | 6.10 | 39.9% |
| gemini-3.7-flash | 12.92 | 6.00 | 36.7% |
| glm-5.3 | 12.13 | 5.75 | 35.6% |
| deepseek-v4-pro | 11.50 | 6.67 | 32.2% |
| qwen3.8-max | 11.31 | 7.25 | 26.8% |
| grok-4.6 | 9.97 | 6.78 | 26.1% |
| nemotron-3.5-lightning | 8.51 | 8.50 | 20.7% |
| **all AI** | **11.72** | **6.60** | **31.3%** |

**This was induced by the naming fix and is knowingly left in.** Told not to name
anyone, the models restate "my wife" where a person switches to "she". Before the
fix the AI class sat at 1.20x human on this measure; after, 1.77x, with the
pronoun ratio down from 11.00 to 6.60 against a human 8.00.

A wording change was drafted — say who someone is once, then use pronouns — and
the project owner chose the accept-and-report branch of the stopping rule rather
than a third regeneration. The tell is real, cheap to learn, and documented here
so nobody discovers it as a surprise. Note `gpt` is worst on this and best on
naming: it suppresses names hardest and pays for it hardest.

## 2. Verdict phrasing is bimodal

Canonical "am I the asshole" is at 21.4% pooled against a human 27.7%, but that
average describes no generator: qwen 70%, gemini 60%, glm 20%, and deepseek,
nemotron, gpt and grok at exactly 0%. Per-model SD 30.8 points. `PROTOCOL.md`
criterion 2 — a blend can cancel.

## 3. Near-duplicate openers: 28 pairs, against zero in the human class

Roughly 1% of documents are involved, and nearest-neighbour cosine runs
0.22-0.27 against a human 0.19-0.20. Per generator: qwen 7, gemini 6, gpt 6,
glm 6, grok 2, deepseek 1, nemotron 0. Every matched human control is 0.

Read carefully, these are **not** recycled scenarios. Inspecting the closest
pairs, they are genuinely similar real titles producing similar openers — two
different posts about a loud upstairs neighbour, two about a step-parent's
financial obligations, two about a Sunday dinner with the in-laws. The finding is
not that the models invent the same story twice. It is that **where two real
situations resemble each other, people still wrote them differently and the
models converged.** That is a uniformity signal, and it is the same one the
raised nearest-neighbour cosine reports.

## 4. Scenario diversity is otherwise sound

Title-match retrieval runs 13.5% to 39.9% against a 0.24% random baseline —
56x to 166x — so scenarios are genuinely inherited from the 6,343 real titles.
Two generators sit outside the human range in opposite directions: `gpt` hugs
its title (39.9% against a human 20.5%), `nemotron` drifts from it (13.5%
against 22.7%).

## Agreed but NOT YET APPLIED

Nothing in this section is in the tree. The bench result above was measured
before any of it. Recorded here so the reasoning survives.

**1. glm-5.3 length constant 1.115 -> 1.24.** Implied production ratio 1.201 on
dev (n=8) and 1.269 on bench (n=10) — high in both, so real drift rather than
noise. The pooled 1.24 is used rather than the bench 1.269, which would fit the
most recent sample alone. Expected result: about 1.02 on bench-like data and
0.97 on dev-like, inside ELI5's 0.94-1.01 band against both.

**2. The verdict draw's "do not use a stock phrasing" clause is replaced, not
deleted.** A diversity check found the clause pushing the generated class to
"tell me if I" (12% ai, 0% human) and away from the canonical "am I the
asshole" (23% human, 10% ai). The prohibition makes the text *less* human on
that axis and plants a lexical tell.

It is replaced rather than removed because the clause is a scar: `make_configs.py`
records that v8_invert planted "I might be wrong" in 56.3% of documents against
a human 0.0%, and asking for the idea in the model's own words is what prevents
a fixed string recurring. The rule is sound in general and wrong only here,
where the stock phrase is the genre's own name. A bare deletion risks the mirror
failure — models like a canonical phrase, and it could overshoot to 60-80%
against a human 23%, trading a 12-point tell for a larger one. Phase 1:
absence and excess are equally detectable, only the rate is neutral.

The replacement permits without mandating. Both strings are measured on the next
bench, not just the question-mark rate, and if canonical clears about 35% it
becomes a rate-matched draw at 23% instead of a free choice.

**4. The 1,050-document early read is subsampled by a SEEDED DRAW, not by id
order.** Reddit base36 post ids sort chronologically — the property this corpus
relies on to date itself — so taking the first 150 of a model's questions in id
order takes its oldest posts. Measured on glm's 414: the id-sorted first 150
spans only 55% of that model's date range, clustered at the early end.

That would corrupt the one thing the early read measures. AITA topics drift over
time, so a temporally clustered subsample shows narrower scenario spread for
reasons that have nothing to do with the generator — biasing the near-duplicate
and nearest-neighbour figures toward a FALSE ALARM. And because the human
control is drawn on the same titles, it moves in lockstep, which hides the
artefact instead of exposing it. The draw is seeded, the seed is recorded, and
the remaining ~264 per model are exactly the rest of the corpus.

**5. The verdict-phrasing fallback gets ONE correction, with no second bench.**
If the bench shows canonical "am I the asshole" clearing ~35%, the rate-matched
draw at 23% is applied and verified on the 1,050-document generation rather than
by paying for another bench. Applying a prompt change and then re-benching it
could trigger the same rule again, which is an unbounded loop at bench prices.
n=1,050 estimates a rate far better than n=70 does, those documents are corpus
documents either way, and if the rate is still off there it is accepted and
reported — the same discipline as the length tripwire.

**3. The length tripwire, locked in advance.** gpt and glm are the only two
constants that get fitted, because each drifted consistently across two
independent samples. The next bench's per-generator length table is the
**reported** band, not another selection round. If a third generator sits
outside it there, that is accepted and reported as a wider band rather than
corrected — a third constant fitted on n~10 would be fitting sampling noise, and
a wider band reported honestly beats whack-a-mole. Stated before the run so it
cannot be decided by whatever the numbers happen to look like afterwards.

## Reproduce

    python3 phase3/scripts/aita_01_build_human.py
    python3 phase3/scripts/aita_02_partition.py
    python3 phase3/scripts/markers.py --human-ids dev bench \
        --human-corpus phase3/aita/data/aita_human.jsonl \
        --partition phase3/aita/data/aita_partition.json \
        --json phase3/aita/data/aita_human_marker_rates.json
    python3 phase3/scripts/aita_03_make_configs.py
    python3 phase3/scripts/generate.py phase3/aita/prompts/floor \
        --corpus phase3/aita/data/dev_corpus.jsonl \
        --out phase3/aita/prompts/floor/answers_dev.jsonl
    bash phase3/scripts/aita_05_judge.sh

## Not committed

`phase3/aita/data/` holds reddit post text, which is not ours to redistribute.
It is regenerated by the scripts above from public sources.
