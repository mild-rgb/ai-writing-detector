# Phase-3 corpus

5,800 documents over 2,900 r/explainlikeimfive questions, paired 1:1 human/ai.

    train  4,638   2,319 human / 2,319 ai
    dev      582     291 human /   291 ai
    test     580     290 human /   290 ai

`dataset.jsonl` holds everything; `train.jsonl` / `dev.jsonl` / `test.jsonl`
are the same rows split out.

## Report per generator, never a pooled rate

    generator                        docs   train   dev   test
    google/gemini-3.7-flash           415     332    42     41
    openai/gpt-5.6-luna-pro           415     332    42     41
    z-ai/glm-5.3                      415     332    42     41
    deepseek/deepseek-v4-pro          414     331    41     42
    nvidia/nemotron-3.5-lightning     414     331    41     42
    x-ai/grok-4.6                     414     331    41     42
    qwen/qwen3.8-max                  413     330    42     41

A Haiku 4.5 read of this generator on 140 items x 3 passes, before the corpus
was built, found **7% to 80% detection across the seven models**: nemotron
80.0%, deepseek 43.3%, then gemini 16.7%, glm 13.3%, grok 13.3%, qwen 13.3%,
gpt 6.7%. The pooled figure was 26.7% and it describes none of them.

**A detector's accuracy on this corpus is therefore mostly a statement about
which generators its evaluation slice over-weights.** Always print the
per-generator breakdown beside any aggregate. Balanced accuracy pooled over the
blend was 57.9% (SD 2.6), against 57.3% for the phase-1 v6 prompt on the same
detector -- the generator judges the same as the original.

## How it was made

**Human class.** Top-scored answer per question from the Facebook ELI5 (LFQA)
corpus, 2011-07 to 2019-07, score >= 10, 250-800 words, no `_URL_` placeholder,
deduped on a normalised title. Every human answer predates ChatGPT by more than
three years, so the label is true by construction. The strict filter yields
exactly 2,902 candidates from the 107,280-row dump -- this corpus is the whole
available pool, not a sample of it, and no sampling-variance argument applies to
the question set.

**AI class.** One document per question, generator assigned round-robin at
sampling time so the blend is balanced *within each split*, not just overall.
The prompt is `floor/prompt.txt`: v6's register constraints with the em-dash
prohibition and the first-person suppression converted to rate-matched draws,
plus the LLM-lexicon blacklist. No structural instruction of any kind.

**Why no paragraph instruction.** Six prompts were built to make generators
match human paragraph structure. The best of them took `wc -l` from 74.0%
balanced accuracy to 54.3% -- and Haiku's balanced accuracy *rose* from 57.3% to
79.2% on the same text. Matching the human paragraph count pushes models into an
essayistic register (evenly shaped paragraphs, each opening with a topic
sentence) that a language-model judge finds easier to catch, not harder. The
campaign was retired; whitespace is handled on the detector side instead. The
full record is in `../adversarial_prompts/`.

**Rate-matched draws.** Reddit-native and voice features are injected at their
measured long-form human rates -- markdown emphasis 32.3%, first person 56.9%,
`Edit:` 13.8%, a question to the reader 36.1%, and so on, from
`../data/human_marker_rates.json` (n=130). Features the models produce
spontaneously use two-sided draws that instruct both branches, because a
one-sided draw only ever *adds* to whatever the model was going to do: asking
for markdown in 32.3% of documents produced it in 44% under one prompt and 84%
under another.

**Length.** Pinned per question to that question's own human answer, with a
per-model calibration constant. Produced-over-target by generator: deepseek
0.94, gemini 0.97, nemotron 0.98, gpt 0.99, grok 0.99, qwen 1.00, glm 1.01.
Word count is not a usable per-model shortcut here. It nearly was: an earlier
pass borrowed another prompt's constants and grok came out at 1.48.

## Fields

Every row: `doc_id`, `question_id`, `q_id`, `split`, `label`, `model`, `text`,
`words`, `question`. AI rows additionally carry `prompt`, `seed`,
`target_words`, `draws`, `post` -- so **phase 4 model attribution runs on these
exact documents without regenerating anything**.

`split` lives in `questions.jsonl` and in the assembled rows, and nowhere else.
It was briefly also written onto the raw answer rows, went stale when the corpus
was re-split, and 73 rows ended up claiming a split their question no longer
had. The general rule that came out of it: **a fact about how a document was
made cannot go stale; a fact about how the corpus is organised can, and must
live in exactly one file.**

## Caveats

- **Whitespace is not normalised.** That is a training-side step, left out
  deliberately so it can change without regenerating 2,900 documents.
- **Two questions were dropped**, `eli5c-2418` and `eli5c-2719`, both qwen
  returning about a tenth of the requested length. Each was dropped *with its
  human sibling* so the 1:1 pairing holds by construction. Both were in train.
- **130 questions carry `reclaimed: true`**, having been briefly reserved for
  adversarial evaluation. 70 of those (`bench`) contributed to the human marker
  rate measurement, so their human answers have an indirect, document-agnostic
  dependency on prompt development. The 60 `heldout` questions were never read
  by anything and are the zero-dependency subset if a secondary evaluation
  number is wanted.
- **One human answer per question, the top-scored one.** High-scoring long-form
  ELI5 skews toward people writing inside their own profession, so the human
  class is narrower than "human writing" and reads more organised than typical
  Reddit prose. Phase 1 found that is where false positives come from.
- **Seven models from 2026**, all writing under a register constraint. Nothing
  here should be assumed to generalise to unconstrained model output.
