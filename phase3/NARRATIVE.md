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

The one-line summary of phase 3: the corpus exists; the detector is perfect
in-domain and, out of domain, a genuine detector (balanced accuracy 0.863, beating
the bag-of-words floor at 0.745) that degrades and becomes untrustworthy at the
single-document level. The earlier "does not generalize" reading came from
concluding on the AI-only arm before the human control was measured, and is
withdrawn.
