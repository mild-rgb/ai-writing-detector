# Narrative — phase 3

This follows on from `phase1/NARRATIVE.md` and `phase2/NARRATIVE.md`. Those two
told the story of building a labelled corpus and then discovering how easy it is
to fool yourself when measuring AI-text detection. Phase 3 is where the project
stops studying the problem and starts trying to build a detector that is actually
good.

Same house rule as before: if something here is a plan rather than a result, it
says so. Almost everything in this file is still a plan, because phase 3 has only
just begun. Where a number was actually computed, it is marked as such.

Today's date is 23 August 2026.

---

## 1. What phase 3 is for

The goal is a real human-vs-AI detector. Just two labels: was this written by a
person, or by a model. Telling the different AI models apart (attribution) is
interesting, but it is being pushed to phase 4 so it does not distract from the
main job.

The bar we are aiming at is Pangram, the commercial detector this project
already benchmarked. Pangram reaches about 98% detection at almost no false
positives, but only on longer text (250 words and up). On short Reddit-length
comments it drops to the high 70s, and everything else we tested drops much
further. So "match Pangram" means two things at once: work on long documents,
and get the accuracy up into its range.

The model we plan to fine-tune is ModernBERT, the same family phase 2 used in its
one training experiment.

## 2. How much human data we actually have

Before planning anything, we counted what is really in the raw dump. The dump has
107,280 ELI5 questions. Using the project's existing quality filters (a score of
at least 10, the top answer to each question, no link-scrubbing artifacts,
deduplicated by question), the number of clean human answers by length is:

| Length | Clean human answers | If we allow every good answer, not just the top one |
|---|---|---|
| 50 to 250 words (short) | 17,368 | 37,872 |
| 250 to 800 words (long) | 2,902 | 6,661 |
| 250 words and up | 3,073 | 7,073 |

The important takeaway: the short band is rich, but Pangram's territory is the
long band, and that is exactly where the human pool is thinner. There are about
2,900 long human answers under the strict filter, or about 6,600 if we relax
"top answer only" to "any answer that scored at least 10."

Because we want a balanced set (half human, half AI), the human side is the
limiting factor. The AI side is cheap to generate; humans are fixed at whatever
the dump gives us. So the biggest balanced long-form set we can build is roughly
5,800 documents on the strict filter, or about 13,300 on the relaxed one. Either
size is comfortably large enough to escape the "too little data" problems that
made phase 2's single training run unstable.

## 3. The ModernBERT plan

We looked up the current official guidance for fine-tuning ModernBERT rather than
relying on memory, because most of these models and recipes are newer than what
was known when this project started.

The short version of the recipe: a learning rate around 8e-5 for the smaller
model and lower for the large one, batch size 32, only 2 to 3 epochs, a very
small weight decay, and standard AdamW. A few things matter more than they look:

* Use bf16 precision, not fp16. The model was trained in bf16 and fp16 tends to
  blow up.
* Start with the plain attention setting (sdpa), not Flash Attention 2. There is
  a known bug where Flash Attention 2 produces broken outputs during
  fine-tuning.
* Turn off the model's auto-compile feature, which breaks on many setups.
* ModernBERT is sensitive to the learning rate. If training diverges, lower the
  rate before blaming anything else.

One setting is specific to our plan. Because we are working on long documents,
the default way ModernBERT reads a whole document (through its first token) can
miss information on long inputs. There is an option to average across the whole
document instead, and we should test both.

The biggest lesson we are carrying in from phase 2 is about whitespace. This
corpus has a strong shortcut: the AI models, especially Grok, tend to write in
big unbroken blocks while humans break their text into paragraphs. Counting line
breaks alone beats every detector in the repo below the strongest one. If we
train ModernBERT on the raw text, it will happily rediscover that shortcut and we
will have built an expensive line-break counter. Phase 2 already showed the fix
works: train on text with the whitespace stripped out. That did not hurt
accuracy and it made training far more stable. So we plan to normalise whitespace
before training, and to also train a raw version so we can see how much of any
result was just whitespace.

## 4. Expanding the generators

Phase 1 and 2 used three models: Grok 4.6, Qwen3.8 Max, and DeepSeek V4 Pro. A
detector trained on only three models risks learning "this is what those three
models sound like" instead of "this is what a machine sounds like." So we widened
the set.

We pulled the live model list from OpenRouter and picked recent, capable,
low-cost models from different companies, on the principle that variety across
model families matters more than any single model. The set is seven models, one
per company:

| Model | Company |
|---|---|
| Grok 4.6 | xAI |
| Qwen3.8 Max | Alibaba (Qwen) |
| DeepSeek V4 Pro | DeepSeek |
| Nemotron 3.5 Lightning | NVIDIA |
| Gemini 3.7 Flash | Google |
| GPT-5.6 Luna Pro | OpenAI |
| GLM 5.3 | Z.ai |

Two of the new models needed handling, both caught in the pilot. The Gemini
"batch" tier turned out to be async-only: it returns a 404 on the normal
synchronous API and would need a polling layer with hours of delay to use. Since
this stage is an iterative loop, that delay is not worth the roughly one dollar it
saves, so we locked in the plain synchronous `google/gemini-3.7-flash` instead.
And Nemotron hit the hidden-reasoning-token trap in its worst form: it reasoned
until it ran out of budget and returned an empty answer every time, no matter how
large the budget. Turning reasoning off for that one model fixed it. Both problems
were caught only because the harness logs failures and warns on a shortfall rather
than failing quietly.

Grok has a split role, and it is worth explaining why. It is kept in the detector
training corpus but taken out of the adversarial prompt-development loop.

It stays in the corpus because it is the most valuable class there. It was the
hardest generator for every detector in phase 1 (short-form: Pangram caught 64
percent, Sonnet 32, Haiku 20, against 97 to 100 percent on long form), and phase
2's clustering shows it is the most distinctive of the models, sitting apart from
the DeepSeek-Qwen pair and not covered by any other model. Dropping it would make
the corpus easier in exactly the way we do not want, so a strong detector needs
it in the data.

It comes out of the prompt loop because its provider rate limits are account-wide
and aggressive, and its retries kept tying up the generation workers. In a
one-shot corpus build that is tolerable, because you can let it run slowly. In the
prompt loop, which regenerates many times over, it costs hours per iteration. So
the loop runs on the six non-Grok models, and Grok rejoins for the corpus. Its
first-round loop result is also kept as a record: Grok was the worst offender on
the paragraph shortcut in phase 1 at 0.955, and the new paragraph pin brought it
to 0.509, the single clearest piece of evidence that the pin works.

## 5. The adversarial prompt sub-project

Alongside the detector, we started a smaller job: writing prompts that make the
AI text harder to catch. This is not about hiding the AI. It is about removing the
cheap giveaways so that a detector is forced to read the real writing instead of
winning on tricks.

The intent, stated plainly by the project owner: strip only the blatantly obvious
tells. That means the whitespace and paragraph habits, the tired stock phrases
(the "delve" family), em dashes, and any planted markers. Remove those, matched
to the rate humans actually use them, and see what is left. The point is to make
the eventual detector earn its accuracy on deep patterns rather than on surface
bookkeeping. Nobody expects the strong detectors like Sonnet or Pangram to be
fooled by this, and that is fine. If they still catch it, good. What should stop
working is the dumb stuff.

The deliverable is six separate prompts, each using a different method to remove
the surface tells. Each prompt is tested against a blend of models, not just one,
so the results are not tied to a single model's quirks. (The first round used all
seven; with Grok removed, later rounds use six.) The same blend is used for all
the prompts so they can be compared fairly.

An early wrong turn we avoided: the obvious layout would be one prompt per model,
but that would make it impossible to tell whether a difference came from the
prompt or from the model. The whole reason for a fixed shared blend is to keep the
prompts comparable to each other.

## 6. Working with a second session

Phase 2 found that the single most useful thing it did was have a second person
look at the work with a different goal in mind. Phase 3 is leaning into that from
the start: a second Claude session is doing the adversarial prompt work while this
session coordinates and keeps the narrative.

That second session has already earned its keep before generating a single
document, by catching two design mistakes.

The first: if we judge success only by the blended score across all seven models,
the number can lie. Two models can cancel each other out. If one model always
writes in one block and another breaks its text constantly, the blended score can
look like random chance even though each model on its own is still easy to spot.
The second session proved this with a quick made-up example: the blended score
came out at 0.555, which looks like success, but a slightly smarter rule still
scored 86% on the exact same text. The fix is simple and does not change what we
report: match the feature for each model individually, and still report the blend
as the headline. A blend that is genuinely low because every model was matched is
real. A blend that is low because of cancellation is a mirage.

The second: with the old goal ("beat Haiku") gone, the loop had no stopping
condition. The new stopping rule is based on the surface statistics themselves,
which is a better idea than it first sounds, because those statistics are free to
measure and far more precise than any judge score. So the loop is steered by the
feature measurements, backed by a few subagent passes that read a sample of the
documents for problems the counters cannot see, like mangled formatting or an
obvious tell that is not in the marker list. Those reading passes are a sanity
check, not a detection score. The detection judges, which are noisy, still come
only once, at the very end, to confirm. That keeps the noise that made every
phase-1 comparison impossible to call out of the loop, while still having a real
reader inside it.

The whole episode taught one lesson worth stating on its own: a regex proxy is not
a substitute for a language-model judge. The loop was steered by surface
statistics because they are cheap and free of judge noise, and they turned out to
measure the wrong thing. Closing every surface marker drove the judge's detection
rate up rather than down, so optimizing the proxy optimized against the goal. The
regex checks are worth keeping as a fast floor, but they only ever catch the tells
you already know to name; whether the text actually reads as machine is a question
only a judge can answer. Using a proxy in place of the judge, rather than
alongside it, is never a good idea. A tighter loop that judged every document each
round, even with a noisy judge, would have caught the divergence on the first
iteration instead of after two full rounds of machinery.

The second session also built the tool that measures all this
(`phase3/scripts/baseline_surface.py`). As a check, it re-measured phase 1's v6
data and reproduced the published separability numbers (0.854 for Grok against a
published 0.848, and so on), which tells us the tool is measuring the same thing
the earlier work did.

## 7. How the final check is measured

Because judges only confirm rather than decide, the judged set is deliberately
small: 140 documents per prompt (70 AI, evenly split across the models, and 70
human), each read one at a time in small batches so no judge sees enough to guess
the pattern. This matches the size and shape of `long2`, the one benchmark in the
repo that is actually trustworthy, so the numbers are directly comparable to
phase 1.

Haiku reads the set five times over, Sonnet three times but only for a prompt that
already cleared the surface bar, and Sonnet stops immediately if it shows no
progress. The honest limit, stated up front: even with five passes, 140 documents
can only pin down "at chance" to within about eight points. That is why the
surface statistics, measured on the much larger generation set, are the real
criterion and the judges are only a confirmation.

A question came up about why the judged set is so small. The answer is that
growing it buys a slightly tighter confirmation number, not a better decision, and
it costs a lot more judging. If we want a tighter judge estimate later, the lever
is more documents, not more re-reads of the same ones.

Before trusting the judges at all, we tested how reliable they are, and the answer
sharpened the whole plan. We judged phase 1's clean long2 benchmark three ways:
phase 1's original subagent numbers, a fresh subagent run, and the API. The
finding was that the route you use barely matters, but the run does. On balanced
accuracy all three landed within two points of each other, so for the number we
care about the two routes are the same instrument. But no two runs agreed on more
than about 75 to 80 percent of individual documents, and that held even for one
subagent run against another. Re-running the same route moved a headline detection
rate by as much as 13 points. Oddly, the API run came out closer to phase 1's
original numbers than a fresh subagent run did.

Two things follow. First, this is why we read the set five times rather than once:
a single pass could have reported a detection rate anywhere in a wide band.
Second, balanced accuracy is the only stable number here, so every phase-3 result
will be a mean across five passes with the spread beside it, and any comparison
back to phase 1 will be made on balanced accuracy alone, not on detection or
false-positive rate on their own.

## 8. The whole cycle in one picture

Sections 5 to 7 describe the parts. Here is how they fit together. The left-hand
loop runs on cheap, low-noise checks, and the detection judge appears only once,
after the loop is finished.

```
  MEASURED ONCE, THEN FROZEN
  ┌──────────────────────────────────────────────────┐
  │ human_marker_rates.json  35 marker rates +         │
  │                          paragraph/length targets  │
  │ _floor.txt               shared base prompt        │
  │ mechanism.md  (x6)       one distinct idea each     │
  └──────────────────────────────────────────────────┘
                        │
                        ▼
          ┌─────────────────────────────┐
          │ ASSEMBLE PROMPT             │  floor + shape line + draw config
          │ (this prompt, this version) │  draws seeded per (question, model)
          └─────────────────────────────┘
                        │
        revise ─────────┤◄─────────────────────────────┐
        (v2, v3)        ▼                               │
   ┌─────────────────────────────────────────┐         │
   │ GENERATE on dev (60 Qs x 6 models)       │         │
   │  · seeded marker draws (two-sided in v2) │         │
   │  · post-ops: punctuation rate-match      │         │
   │  · per-(prompt,model) length calibration │         │
   │    + reject band [0.8, 1.3], resample    │         │
   └─────────────────────────────────────────┘         │
                        │  ~360-420 docs                │
          ┌─────────────┴─────────────┐                 │
          ▼                           ▼                 │
 ┌───────────────────────┐  ┌───────────────────────┐  │
 │ REPORT CARD           │  │ SUBAGENT QA PASSES    │  │
 │ ("regex judge")       │  │ (fresh agent / batch, │  │
 │ · separability per    │  │  sample per round)    │  │
 │   model + pooled:     │  │ flags what counts     │  │
 │   newlines/paras/     │  │ can't: mangled text,  │  │
 │   wpp/words           │  │ incoherence, obvious  │  │
 │ · 35 marker gaps      │  │ tells off the list.   │  │
 │  NO detection judge   │  │ QA read, NOT a        │  │
 │                       │  │ detection score.      │  │
 └───────────────────────┘  └───────────────────────┘  │
                        │                               │
                        ▼                               │
                  ╱───────────╲                         │
                 ╱  PASS?      ╲   no ──────────────────┘
                 ╲  every cell ╱    (also: stop a prompt after
                  ╲ in band?  ╱      2 rounds with no change)
                   ╲────┬────╱
                        │ yes
                        ▼
             ┌────────────────────┐
             │ FREEZE this prompt │
             └────────────────────┘

  ── when all 6 prompts are frozen, leave the loop ──
                        │
                        ▼
   ┌─────────────────────────────────────────┐
   │ BENCH GENERATION (70 held-out Qs)        │   bench was never
   │ build 140-item blended benchmark:        │   tuned against
   │ 70 ai (12/12/12/12/11/11) + 70 human     │
   └─────────────────────────────────────────┘
                        │
                        ▼
   ┌─────────────────────────────────────────┐
   │ DETECTION JUDGE  (once, at the end)      │
   │  Haiku x5  = 1 subagent + 4 API passes   │
   │  Sonnet x3 one shot, stop if flat        │
   │  headline = balanced accuracy, mean ± SD │
   └─────────────────────────────────────────┘
                        │
                        ▼
                  FINAL NUMBERS
                        │
                        ▼
        (finished prompts then feed the
         detector corpus — a separate stage,
         where Grok rejoins the roster)
```

Two things the picture makes explicit. The generate-score-decide loop has no
detection judge in it: the stopping decision comes from the report card and the
subagent QA passes, and those passes read for problems the counters cannot see but
hand back flags rather than a score. And the prompt cycle and the detector corpus
are separate; Grok rejoins only downstream, when the finished prompts are used to
build the corpus.

## 9. Decisions made, and what is still open

Settled:

* The generator set is the seven models above, but Grok has a split role: it is
  kept in the detector corpus (it is the hardest and most distinctive class) and
  taken out of the adversarial prompt-development loop (its rate limits cost hours
  in an iterative loop). See section 4.
* The adversarial goal is to strip obvious surface tells, matched to human rates,
  tested against a fixed blend, on long documents.
* Success is measured on surface statistics per model with the blend reported, not
  on the blended number alone.
* Reusing the retired `burned` question set is allowed, since this work is judged
  on feature statistics and cannot be contaminated by a judge having seen a
  document before. (Note: `burned` is short-form, so it mostly matters if we add a
  short-form arm later; the long-form pool already has 796 clean questions, which
  is plenty.)
* The long-form questions, which were never split before, are now partitioned:
  60 for tuning, 70 for the judged benchmark, 60 held out, and 606 in the pool,
  leaving out the 204 questions already used in earlier work. The question set and
  the question-to-model pairing are frozen across all six prompts.
* The judging route was decided by test (see section 7). The two routes are the
  same instrument on balanced accuracy, so we are using a split: one confirmation
  pass per prompt as subagents, the other four passes over the API. This keeps the
  cost sane. Doing all five passes as subagents would have been about 22 million
  subagent tokens across the six prompts; the split brings it down to roughly 4.4
  million subagent tokens plus about $3 of API.

* The adversarial optimization loop was halted after v2. The reason is itself a
  result: closing the trivial surface shortcuts did not make the text harder to
  detect, it made it easier. The prompt that closed all four surface features (p1)
  was caught by Haiku at 79 percent, against 57 for phase 1's ordinary prompt,
  because forcing the models into human paragraph structure pushes them into an
  even, topic-sentence-per-paragraph essay register that a judge reads as more
  machine-made, not less. The whitespace shortcut is better removed on the
  detector side by normalization (phase 2 proved this). So the corpus does not
  need optimized prompts.
* The corpus generator is the floor: v6 with its two over-prohibitions (the
  em-dash ban and the first-person suppression) converted to rate-matched draws,
  plus the stock-word blacklist. Grok rejoins the roster for the corpus, and
  whitespace is normalized on the detector side at training time, not in
  generation.

* The corpus build spec is set: sample about 2,900 long-form questions under the
  strict filter (top answer per question), one AI answer per question assigned
  round-robin across the seven models so the AI class is model-balanced, with each
  answer tagged by its generator. Phase 3 trains on it as a plain human-vs-AI set,
  and the recorded provenance lets phase 4 reuse the same documents for model
  attribution without regenerating. The full strict pool is 2,902 questions; the
  70 bench and 60 heldout questions once reserved for the (now-halted) adversarial
  work were folded back in, since they were never actually used, so the corpus uses
  the whole pool rather than holding those aside. Question-grouped train/dev/test
  splits. Estimated AI generation cost about $15.
* Before generating the full corpus, the floor was judged on the documents already
  produced, single-document Haiku, three passes. It lands where v6 did: balanced
  accuracy 57.9 percent against v6's 57.3. That confirms the floor is a sound
  generator and that the paragraph campaign added only cost. But detection varies
  enormously by model, from 6.7 percent (GPT Luna) to 80 percent (Nemotron), a
  73-point spread, so the corpus is reported per generator, never as a pooled rate.
  Nemotron stays in at equal share as a legitimate easy class; the spread is
  treated as the honest headline rather than hidden behind a pooled 26.7. The judge
  reasons reproduce phase 1: the same cue vocabulary (structure, polish,
  conversational tone) drives catches, misses, and false alarms alike, and
  "technical detail" is cited as a top cue for both catching and missing AI, which
  is how little signal the text carries.

Still open:

* Whether to spend on Pangram as an outside reference on the finished corpus. Core
  generation and judging spend is approved; Pangram is a separate yes-or-no.

## 10. State

* The `phase3/` directory exists. It holds `scripts/baseline_surface.py`, which
  measures the surface features and their separability (checked against phase 1's
  numbers), and `study/judge_compare/RESULTS.md`, the write-up of the
  subagent-vs-API test.
* The generator roster is chosen and verified live on OpenRouter.
* The long-form questions are partitioned, and the human marker rates for the
  long-form set have been measured on 130 documents. They differ enough from phase
  1's short-form rates to matter: first person shows up 56.9 percent of the time
  rather than 23.3, an `Edit:` line 13.8 percent rather than 5.1, and the typical
  answer has about 6 paragraphs and 11 line breaks. All six prompts are drafted,
  each with a written explanation of its method, and their settings are generated
  from that measurement file rather than typed by hand. A dry run confirms every
  feature draw fires at the rate it is supposed to.
* The pilot generation (3 questions across all 7 models) is done: 21 of 21
  documents, about 9 cents, roughly $0.004 per document. It caught and fixed the
  two model problems above. One early signal to watch: every model wrote short of
  its word target, so word count may turn out to be a fourth trivial shortcut, the
  one phase 1 believed it had closed.
* The adversarial prompt loop ran through v1 and v2 and was then halted (see
  section 9). Total spend for the whole loop was $15.77, well inside budget, and
  everything is preserved: six prompt directories, v1 and v2 answer sets, the v1
  dev judge pass, the QA reads, the instrument comparison, and eleven scripts.
  Nothing is running.
* The finding from the loop is banked: closing the trivial surface shortcuts
  pushes detection onto prose, and forcing models into human paragraph structure
  makes them read as more machine, not less.
* The corpus is built and complete, at phase3/corpus/: 5,800 documents over 2,900
  questions, paired 1:1 human/AI, floor generator, all seven models. Per-generator
  AI counts are 413-415, and the splits are train 4,638 / dev 582 / test 580,
  balanced within each split and grouped so no question straddles. Generation cost
  $11.50, about $12.20 including the discarded pilot and the Haiku validation. Two
  questions were dropped, both qwen refusing to write long, each excluded with its
  human sibling so the pairing holds by construction rather than by subsampling.
* Length was recalibrated per generator during the build: every model now lands
  within 6 percent of its human sibling's length, where grok had been at 1.48x
  under borrowed constants. Word count is no longer a usable per-model shortcut,
  which closes the fourth surface tell the floor read had warned about.
* A data trap was caught and fixed during the build. 73 documents carried a stale
  split label from before the reclaim re-split, which would have leaked held-out
  documents into training if anything had trusted it. Fixed at source: the
  generator no longer writes split, and the question file is the single authority.
  The rule it taught is worth keeping: a fact about how a document was made cannot
  go stale, but an organizational fact like which split it belongs to can, so it
  must live in exactly one place.
* Every AI row carries its generator, prompt version, and seed, so phase 4 can run
  attribution on these exact documents without regenerating. The 130 reclaimed
  questions are flagged, and the 60 truly zero-dependency heldout questions are one
  filter from that for a clean secondary evaluation. Whitespace is left
  un-normalized by design; normalization is a training-side step.
* The detector is trained (exp01_detector, arm binary|norm, ModernBERT-large, 5
  seeds, both mean and cls pooling, on an A100). It hits the ceiling in-domain: AUC
  1.000000 with zero variance across every seed, both poolings, and all seven
  generators, zero false positives, and no weak-spot model (the 73-point judge
  spread does not survive a fitted classifier). Mean pooling was chosen for its
  larger, more stable separation margin.
* The result is double-edged, and the honest headline is the floor beneath it. A
  bag-of-words logistic regression, no GPU and seconds to fit, scores AUC 0.9993 on
  the same test set even with all punctuation removed, so the classes are nearly
  separable in plain word space and the transformer's whole edge over trivial is
  the last 0.0007. The perfect score was interrogated three ways and is not an
  artifact (not a leaked token, not punctuation, not the banned-word list); it is a
  real but easy in-domain separation. bow_baseline.py is kept as the binding floor.
  A ceiling number on a corpus that bag-of-words nearly solves is a statement about
  the corpus being easy in-domain as much as about the detector.
* In-domain is therefore saturated with no headroom. The only remaining test that
  can fail, and the one that would show whether the detector generalizes the way a
  commercial detector does, is out-of-domain wild text. The leave-one-generator-out
  sweep is judged low value (a one-epoch pilot on a held-out model already hit
  0.99998, so it would also be at ceiling) and is left un-run.
* The dataset is published, public, at huggingface.co/datasets/mild-rgb/eli5-human-vs-ai,
  with the human text redacted to identifiers so no Reddit text is redistributed.
* The out-of-domain probe was run in two halves. First, 364 chunks of external,
  confirmed AI-generated speculative fiction (story.syn.gl), sized to the training
  median, through the mean-pooling detector: 83.8 percent called AI (a recall
  figure, since the text is AI), 7.7 percent called decisively human, and
  full-confidence verdict flips between adjacent passages of one story (23 percent).
* An earlier version of this section concluded, from that AI-only arm plus the
  bag-of-words comparison, that the detector "does not generalize" and was "not a
  general AI detector." That conclusion is withdrawn. It was drawn before the
  false-positive side was measured, which is exactly the mistake this project keeps
  recording: concluding from one arm before the control is in. (The user caught it.)
* The control settled it. 364 chunks of genuinely human fiction (pre-LLM
  r/WritingPrompts, human by construction), same size and normalization, same
  detector. The false-positive rate is only 11.3 percent, and it puts 83 percent of
  the human fiction at p <= 0.01, so the detector is genuinely separating human from
  AI off-distribution, not drifting toward "AI." Signed out-of-domain numbers:
  transformer balanced accuracy 0.863, AUC 0.945; bag-of-words 0.745 and 0.890. Off
  distribution the transformer is not a near-tie with bag-of-words, it is clearly
  better, so the in-domain tie understated it rather than flattering it. It does
  generalize.
* What survives is a calibration failure, not a generalization failure. The
  detector drops from a perfect 1.000 in-domain to 0.863 out-of-domain, and it is
  untrustworthy at the single-document level: 6.3 percent of human fiction is
  flagged at p >= 0.99 (confident false accusations), the scores pin to the extremes
  (89 percent), and adjacent passages of one text flip the call (18 percent). It is
  a real detector carrying genuine transferable signal with unreliable confidence,
  not a setup-specific artifact.
* Then a wider probe changed what "it generalizes" is allowed to mean. On 24
  August 2026 seven out-of-domain sets were scored in one pass, five of them new:
  guaranteed-human Stack Exchange answers and CC-News articles, and AI text from
  older models — WildChat (GPT-4 and GPT-3.5), MAGE (LLaMA, OPT and the davinci
  family), and RAID (llama-chat and MPT). The two fiction sets from the earlier
  probe were rescored alongside them. On the human side the detector is quiet,
  which is the good news: zero false alarms out of 240 Stack Exchange answers and
  1.67 percent on news. Human fiction is the one high reading at 8.75 percent, but
  the bag-of-words floor sits at 6.67 percent on the same documents, so that looks
  like something about the fiction register rather than a fault in the model. On
  the older-model side it is nearly blind. It
  misses 94.6 percent of the WildChat AI text, 96.3 percent of MAGE, and 90.4
  percent of RAID, and on all three it ties the bag-of-words floor (96.3, 93.8 and
  91.7 percent missed). The one set it clearly wins on, missing 16.2 percent
  against the floor's 43.4, is the AI fiction — the only one of the four written by
  2026 models.
* So the generalization claim needs a boundary drawn through it. The detector
  transfers across subject matter and it does not transfer across model vintage. It
  learned what the 2026 frontier sounds like, and text from 2022-era models does
  not look like AI to it at all. "It generalizes" is true of the fiction result and
  false of everything older, and the two were never the same claim.
* Two honesty notes travel with those numbers. Each older-model set was scored on a
  240-document sample rather than the whole set, so the figures are clearly bad but
  not precisely measured, and a full-size re-run is still owed. And each of them
  changes two things at once, a new domain and an older generator, so they answer
  "can it catch a model it has never seen" rather than "how good is it on today's
  models." The scored output, the per-model breakdowns and the Wilson intervals are
  in phase3/study/_ood_scored_20260824/, which is kept locally and is not in the
  repo.

The one-line summary of phase 3: the corpus exists; the detector is perfect
in-domain and, on new subject matter written by the same generation of models, a
genuine detector (balanced accuracy 0.863, beating the bag-of-words floor at 0.745)
that degrades and becomes untrustworthy at the single-document level — but on AI
text from older models it misses around nine documents in ten and does no better
than counting words. It generalizes across domain, not across model vintage. The
earlier "does not generalize" reading came from concluding on the AI-only arm
before the human control was measured, and is withdrawn; the narrower claim that
replaces it is the vintage boundary, which was measured with both arms in place.

## 11. A second corpus, and what it is for

**The corpus is complete.** 2,900 AI documents paired with 2,900 human posts,
balanced across the seven generators — 415 each from deepseek and qwen, 414 from
the other five — all produced by a single prompt with no version split.
Generation cost $23.30. What follows is why it was built and what building it
taught us, including the two things we got wrong on the way.

The reason for a second corpus comes out of two numbers already on record. The
wider out-of-domain probe showed the detector transfers across subject matter but
not across model vintage. And in domain, a bag-of-words model reaches 0.9993 on the
same test set the transformer scores 1.000 on. Read together, those say the
detector may be leaning on plain word choice more than we would like — a lexical
shortcut that happens to work, rather than a read on how the text was made.

The plan for that is Product-of-Experts: train the transformer through a frozen
bag-of-words model, so it is pushed to learn what the bag-of-words gets wrong, and
then drop the bag-of-words at inference. This only works if the bag-of-words is
imperfect. Ours is not — at 0.9993 in domain it is right about almost everything,
so training against it would leave the transformer nothing to learn. The way out is
to use a bag-of-words fitted on a *different* subreddit. Cross-applied, it is
imperfect in a useful way, and that is the whole reason a second corpus is being
built.

That fixes one of the two problems and not the other. Product-of-Experts can push
the detector off a lexical shortcut. It cannot teach it what a 2022 model sounds
like, because nothing in the training data sounds like one. Vintage-blindness needs
older-model AI text in training, which is a separate piece of work and is deferred.
The current seven 2026 generators are what this corpus uses.

**Why r/AmItheAsshole.** It was chosen for distance from ELI5 rather than
similarity to it. ELI5 is explanatory question-and-answer; AITA is first-person
narrative that ends by asking strangers for a verdict. If the two domains were
close, a bag-of-words from one would behave on the other much as it does at home,
and there would be nothing to cross-apply.

**The human side has to be human by construction, and it is.** The source is AI2's
Scruples Anecdotes, 32,766 AITA posts kept in their original casing with their
reddit post ids. Scruples ships no dates, so they were recovered by interpolating
the base36 post ids against a dated dump: every post lands on 6 April 2019 or
earlier, years before ChatGPT. That is the same guarantee the ELI5 human class has,
arrived at the same way — nobody has to be trusted about provenance.

Two much larger AITA sets on Hugging Face were rejected, and the reason is worth
recording. Both have fully lowercased post bodies. A human class that is entirely
lowercase sitting against a normally-capitalised AI class would hand any detector a
free shortcut, and removing exactly that kind of shortcut is what this project is
for. The strict filter — mirroring phase 1's — leaves 6,343 posts with a median of
391 words. The text used is the post body, and the model is given only the title.

**The machinery was ported, the numbers were not.** The generator is the same floor
prompt that produced the shipped ELI5 AI class, with the same rate-matched draws
and the same per-model length calibration. One correction to the record while we
are here: the prompt that shipped is the corpus floor, not any of the retired p1 to
p6 adversarial prompts. What could not be carried over is the measurements. AITA
writes nothing like ELI5 — first person appears in 99.2 percent of posts against
ELI5's 56.9 — so every rate was measured again on the AITA human class. Reusing the
ELI5 config would have set almost every draw wrong.

**It transfers, and it transfers without retraining.** On the same judging
instrument used throughout the project, the ported floor scores 50.0 percent
balanced accuracy on dev and 51.7 percent on the frozen bench split, against the
ELI5 floor's 57.9. The honest reading is floor parity, not victory. At 140
documents over three passes, "at chance" is only pinned to within about eight
points, so 51.7 and 57.9 are not separated by this measurement, and claiming AITA
beats ELI5 by six points would be reading noise.

One caveat belongs with that number rather than after it. The false-positive rate
went up: the judge now calls roughly one genuine 2019 redditor in six an AI, where
on ELI5 it was five to ten percent. So part of the result is that AITA humans are
harder to distinguish, not purely that the AI is better. Both classes moved, and
only one of those movements is an achievement.

**Three fixes, one of which is a good example of the failure mode.** Human posts
end with a question 83.1 percent of the time; the first generated batch managed
50.8, a 32-point gap and the clearest miss on the marker list. The cause was not
the draw, which was already set at the human rate — it was the floor's own
instruction not to end on a flourish, which the models were obeying over the
instruction to ask. The fix reconciles the two rather than shouting louder: asking
the sub whether you were in the wrong is now explicitly exempt from the
anti-flourish rule, because it is the genre's defining move and not a summarising
line. That closed the gap to under two points. Separately, one generator's length
constant was corrected and a second correction is agreed but not yet applied, under
a tripwire fixed in advance: if a third generator drifts, that is reported as a
wider band rather than corrected, because fitting a third constant on ten documents
is fitting noise. And a stock-phrasing prohibition turned out to be pushing the
models away from "am I the asshole" — the genre's own name — so it is being
replaced by something that permits the canonical phrase without mandating it.

**The diversity question, and a lesson about cost.** The obvious worry about
generating 2,900 posts from seven models is that they will all write the same
story. Most of that is answered by how generation works: the model sees only the
title, so scenario diversity is inherited from 6,343 genuinely distinct human
titles, with each model inventing the specifics. That held up when it was
measured: every generator finds its own title again far more often than chance,
between 56 and 166 times the random rate, so the scenarios really are coming from
the real titles rather than from the models. But the check also found something we
had not thought to look for, and it took two rounds of generation to deal with.

Checking that cheaply turned out to be impossible, and the reason is worth keeping.
The plan was to generate only the first thirty words of each document and compare
openers, at roughly a tenth of the cost. It does not work on this roster: the models
reason before they answer, and reasoning is billed whether or not the output cap
lets any of it through. Five of seven returned nothing usable and charged full
price for it. **Length is not a cost lever here; count is.** Any cheap early read
has to use fewer documents, never shorter ones. So the diversity check will run on
the corpus's real first third instead — about 1,050 documents, drawn as a seeded
random third of the frozen question assignment rather than the first third by id,
because reddit ids sort by date and taking them in order would sample one narrow
slice of time and make the scenarios look artificially alike.

**The models invent names, and people do not.** The first third came back with a
tell nobody had predicted. Asked to write a post about an argument with a
neighbour, the models gave the neighbour a name. Real posters almost never do —
they write "my neighbour" and leave it there. Gemini named someone in 84 percent
of its posts, against about 11 percent for the humans writing on the same titles,
and used "Sarah" in more than a third of them. A real poster's most repeated
name shows up once in two hundred posts.

This mattered more than it might look. A detector that learns "Sarah means
machine" has learned nothing about writing at all. It is the same shallow word-level
shortcut the banned-vocabulary list exists to remove, wearing a different hat. So
it was worth a regeneration, and the fix was a two-sided draw: some documents are
told to name a person, most are told not to, at the rate real posters do it. It
worked. Gemini went from naming someone in 84 percent of posts to 9.9 percent.
Across all seven models the rate is now 11.4 percent against a human 11.5, and no
single name dominates in either class.

**And the fix broke something else.** Told not to name anyone, the models say "my
wife" again and again where a person would say it once and switch to "she". The
density of relationship words went up by more than three quarters against the human
class, and the models that suppress names hardest pay for it hardest. That is a
worse tell than the one we closed, in the sense that it is cheaper to detect: it
needs no list of names, just counting.

The stopping rule had been written down before any of this, and it said one
regeneration, then accept and report. The project owner took that branch. A third
round was not run. The reasoning is that the debiasing step this corpus exists to
support should absorb a shortcut of exactly this kind, and that chasing human
behaviour through round after round of prompt edits is how you end up describing
your own edits instead of people. So the repetition tell is in the corpus, and it
is written down here rather than left for someone to find.

**The length tripwire earned its keep.** Before the corpus was built, the
per-generator length band looked wrong: measured on ten documents each, it ran from
0.906 to 1.079 of the paired human length, and two more generators looked like they
needed their constants refitted. The rule fixed in advance said report a wider band
rather than fit a third constant, because fitting on ten documents fits noise. At
414 documents per generator the band is 0.940 to 1.029, inside the ELI5 corpus's
own published range to within 0.02. The width really was noise. This is the
clearest evidence the project has produced that per-model constants are worth
leaving alone.

**One number needs reading carefully.** The finished corpus contains 28
near-duplicate opener pairs across 2,900 documents, against zero in the human
class. That sounds like the models recycling stories, and it is not. Every close
pair was inspected. They are cases where two genuinely similar real titles — two
posts about a loud upstairs neighbour, two about a Sunday dinner with the in-laws
— produced similar openings. The humans given those same two titles wrote them
differently; the models converged. So the finding is not that a model invents the
same story twice. It is that where two real situations resemble each other, people
still write them apart and machines do not. That is a uniformity signal, and it is
mild — about one percent of documents are involved.

**What is knowingly left in.** Two things, both reported rather than fixed. The
relationship-word repetition described above is the larger. The second is that the
canonical phrase "am I the asshole" is bimodal: pooled across models it sits at
21.4 percent against a human 27.7, which looks like a good match and describes no
generator. Two models use it most of the time, four never use it at all. A pooled
average earned by models sitting on opposite sides of the human value is an
artefact, not a match, and the project's own protocol says so.

**Two measurement bugs worth recording, and they point the same way.** The first
pass at the naming analysis put "Ive" at the top of the human name list. That is
"I've": a third of human posts get curly apostrophes applied, and stripping the
apostrophe as punctuation leaves a capitalised word that looks like a name. It made
the human class look more name-diverse than it is, which in turn made the machine
concentration look milder by comparison. Apostrophes are normalised before names
are extracted now.

The second was a question of where we looked. Naming was first measured on the
first thirty words of each document, the same window used for the scenario-diversity
checks, and that window was wrong for this. Scenario choice happens in the opening
sentences, so thirty words is the right place to look for it. Naming accumulates
through a whole document, so thirty words is the wrong place. Measured on openers,
gemini's problem looked like it was picking the same name too often. Measured on
full documents, the problem was that it was naming anyone at all, in 84 percent of
posts against a human 11. That changed what the fix had to be: a rule about which
name to use would have left gemini naming a character in five posts out of six,
which is why the draw is two-sided and sets the rate rather than the name.

Both errors made the AI look better than it was — one by inflating the human
baseline, the other by measuring in the wrong window. An error that flatters your
own result is the kind you have to go looking for, because nothing about it feels
wrong at the time.

## 12. Which model wrote it, and what that says about the vintage gap

Everything so far asked one question: is this human or machine. Having two corpora
with seven named generators lets us ask a better-posed one. Given a document, which
of the seven 2026 models wrote it — or was it a person? That is an eight-way
question with a chance rate of 12.5 percent, and it turns out to be answerable.

All the numbers here come from bag-of-words, the trivial floor this project keeps
returning to: word and punctuation presence, logistic regression, no GPU. They are
reproduced by `phase3/scripts/attribution.py`, which runs on a laptop in about a
minute. The transformer version is deferred and would likely do better, so read
these as a floor rather than a ceiling.

**Attribution works.** Trained on both domains, the model names the author of a
held-out document 90.3 percent of the time — 87.6 percent on ELI5, 93.1 percent on
AITA, against a chance rate of 12.5. Seven models and a human class, roughly 828
documents per model. Whatever these systems do to a sentence, they do consistently
enough that counting words identifies them.

**The cross-domain collapse was our own doing.** Trained on ELI5 alone and pointed
at AITA, five of the seven fingerprints fall apart: gpt is recognised 10.4 percent
of the time, glm 15.7, nemotron 20.8, deepseek 38.1, qwen 41.7. Two survive the
move intact — grok at 97.3 percent and gemini at 91.3 — so a model's fingerprint
can be domain-invariant, but most are not when only one domain is on offer. Add
AITA to the training data and the same five jump to between 77 and 100 percent. The
limit was in the training set, not in the models. This is the clearest argument yet
for having built a second corpus.

**Human is the cleanest cut in the whole picture.** The human class is recognised
97.6 percent of the time on held-out text from both domains, and only 0.52 percent
of machine documents leak into it. So human-versus-machine is a coarse, sturdy,
domain-invariant boundary, while which-model-exactly is fine-grained and carries
nearly all the error. deepseek is the least distinctive of the seven — at 77.1
percent it is the one the others most often collapse into.

**The headline: the gap is between eras, not between known and unknown models.**
This is the result that changes how the earlier vintage finding should be read. Take
one of the seven models, remove it from training entirely, and ask whether its text
still gets caught. It does, 95.4 percent of the time on average. The held-out model
is misattributed to one of its 2026 contemporaries rather than mistaken for a
person. Evasion runs from 0.1 percent for grok to 17.8 percent for gpt.

Set that against the same bag-of-words approach on older models, from the
out-of-domain probe: it missed 96.3 percent of the GPT-4 and GPT-3.5 text, 93.8
percent of the LLaMA and OPT text, and 91.7 percent of the llama-chat and MPT text.
Same method, same floor. A model it has never seen from 2026 is caught nineteen
times in twenty; a model from 2022 is missed nineteen times in twenty.

So there is a shared signature across the 2026 frontier, and it generalises to
models that were never in the training data. What does not generalise is the jump
backwards across an era. That sharpens the earlier finding rather than replacing
it: asking "is this AI" transfers to unseen models of the same generation, and
fails across generations, which is a data problem with a data solution — older
machine text in training. Asking "which model is this" is genuinely per-model, and
is where the difficulty actually lives.

One thing to keep in view: this is bag-of-words only.

The provenance gap this section originally had to apologise for is now closed.
When §12 was first written, AITA carried no stamped split — its partition lived in
a side file — so the numbers rested on a seeded split invented for the occasion.
Both corpora now stamp label, source and split onto every record, question-grouped,
over pool questions kept clear of the ones used to tune the prompt. The figures
above are measured on that stamped split. It is worth saying why this mattered
enough to fix rather than caveat: the ELI5 corpus already taught the lesson once,
when 73 documents carried a stale split label that would have moved held-out text
into training. A fact about how a document was made cannot go stale. Which split it
belongs to can, so it has to live in exactly one place — on the record.
