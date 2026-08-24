---
license: cc-by-4.0
task_categories:
- text-classification
language:
- en
tags:
- ai-generated-text-detection
- llm-detection
- machine-generated-text
- eli5
pretty_name: ELI5 Human vs AI (long-form)
size_categories:
- 1K<n<10K
configs:
- config_name: default
  data_files:
  - split: train
    path: train.jsonl
  - split: validation
    path: dev.jsonl
  - split: test
    path: test.jsonl
---

# ELI5 Human vs AI (long-form)

**This dataset is for training and evaluating AI-writing detectors.** It was built
as a clean way to compare *known* AI text against *known* human text: every human
answer predates ChatGPT by more than three years, so it is genuinely human by
construction, and every AI answer was written by a named 2026 model, so its origin
is certain too. Most detection datasets have to guess at their labels; this one
does not.

ELI5 answers were chosen because they are factual, explanatory web knowledge, the
kind of writing people actually use LLMs for online. That keeps the AI side
in-distribution for real-world LLM usage rather than being a contrived writing
task, so a detector trained here is aimed at the text it will actually meet.

It is 5,800 documents over 2,900 long-form r/explainlikeimfive questions, paired
one human answer to one AI answer per question.

> **The human text is not included in this repository.** The human half is
> verbatim Reddit content, which we cannot redistribute. What is published is the
> AI-generated documents (which are ours) plus the identifiers you need to
> reconstruct the human documents from the original source. See
> [Reconstructing the human side](#reconstructing-the-human-side) below. So this
> is, precisely, a set of AI generations with matched human placeholders, not a
> ready-to-load balanced corpus.

```
split        rows   human   ai
train       4,638   2,319   2,319
validation    582     291     291
test          580     290     290
```

Human rows carry identifiers and metadata but no text. AI rows carry the full
generated text and its provenance.

## Report per generator, never a pooled rate

The AI documents come from seven different 2026 models, and they are not equally
easy to detect. A Claude Haiku 4.5 read of this generator (single-document, 140
items over 3 passes) found detection ranging from **7% to 80% across the seven
models**:

```
generator                        docs   Haiku detection
nvidia/nemotron-3.5-lightning     414          80.0%
deepseek/deepseek-v4-pro          414          43.3%
google/gemini-3.7-flash           415          16.7%
z-ai/glm-5.3                      415          13.3%
x-ai/grok-4.6                     414          13.3%
qwen/qwen3.8-max                  413          13.3%
openai/gpt-5.6-luna-pro           415           6.7%
```

The pooled figure was 26.7%, and it describes none of them. **A detector's
accuracy on this corpus is therefore largely a statement about which generators
its evaluation slice over-weights.** Always print the per-generator breakdown
beside any aggregate number. The models are balanced within every split (413-415
documents each, and within each of train/validation/test), so a per-generator
evaluation is well defined on any split.

## Reconstructing the human side

Each row carries the identifiers, so you can fetch the human documents yourself:

- `q_id` — the r/explainlikeimfive submission id (also linked in `url`).
- `score`, `words` — the upvote score and word count of the human answer, so you
  can confirm you matched the right one.

The human document for a question is the **top-scored answer** to submission
`q_id` in the Facebook ELI5 (LFQA) corpus, restricted to score ≥ 10, 250–800
words, with the corpus's `_URL_` link placeholders removed, deduped on a
normalised question title. The question text is the submission title for `q_id`.
Reconstruction requires access to that source corpus, which has its own access
terms, so this is not guaranteed to be frictionless — it is the honest cost of
not redistributing Reddit text.

## Fields

Every row: `doc_id`, `question_id`, `q_id`, `url`, `split`, `label`
(`human`/`ai`), `model`, `words`.

- Human rows also carry `score`. They do **not** carry `text` (redacted) or the
  question title (redacted).
- AI rows also carry `text` (the generated document), `prompt`, `seed`,
  `target_words`, `draws`, and `post`. That provenance means model **attribution**
  can be studied on these exact documents without regenerating anything.

## How it was made

**Human class.** Top-scored answer per question from the Facebook ELI5 (LFQA)
corpus, 2011-07 to 2019-07, score ≥ 10, 250–800 words, deduped. Every human
answer predates ChatGPT by more than three years, so the label is true by
construction rather than by assumption. The strict filter yields exactly 2,902
candidates from a 107,280-row dump — this is the whole available pool, not a
sample of it. (Two questions were later dropped for a generation failure, each
with its human partner, giving 2,900.)

**AI class.** One document per question, generator assigned round-robin so the
blend is balanced within each split. The prompt is a plain 2019-Reddit register
constraint: no greeting or sign-off, no restating the question, not comprehensive,
no scaffolding, no flourish, uneven rhythm, length pinned to that question's human
answer. Surface markers (markdown, first person, `Edit:` lines, and so on) are
**rate-matched** to their measured human rates rather than prohibited, because a
prohibited feature sits at exactly 0% where humans sit at a few percent, and a
perfect zero is itself a fingerprint.

**Why there is no paragraph instruction.** An earlier effort tried to make the
models match human paragraph structure. It worked mechanically — a line-break
count went from a 74% detector to a 54% one — but a language-model judge caught
the text *more* easily, not less, because matching the human paragraph count
pushes models into an evenly-shaped, topic-sentence-per-paragraph essay register.
So paragraph structure is deliberately left alone here, and whitespace is meant to
be handled on the detector side (e.g. normalising it away) instead.

**Length.** Pinned per question to that question's own human answer, with a
per-model calibration constant, so every generator lands within 6% of its human
partner's length. Word count is not a usable per-model shortcut in this corpus.

## Caveats

- **Whitespace is not normalised.** Left raw deliberately, so a detector can
  choose to normalise it (recommended) or not.
- **One human answer per question, the top-scored one.** High-scoring long-form
  ELI5 skews toward people writing inside their own profession, so the human class
  is narrower than "human writing" and reads more organised than typical Reddit
  prose. That is a known source of false positives.
- **Seven models from 2026, all writing under a register constraint.** Nothing
  here should be assumed to generalise to unconstrained model output, or to models
  not in the set.
- **130 questions were briefly reserved for other experiments** and carry a note
  in the project record; 70 of them contributed to the human-rate measurement that
  set the generation constants, an indirect and document-agnostic dependency.

## License and provenance

The `cc-by-4.0` license applies to the **AI-generated documents and the metadata
in this repository**, which are the project's own output. The human text is not
distributed here; anything you reconstruct from the source is subject to the terms
of the Facebook ELI5 (LFQA) corpus and of Reddit, not to this license. This
dataset is a research artifact for studying AI-text detection; it is not a
statement that any specific real person's writing is or is not machine-made.
