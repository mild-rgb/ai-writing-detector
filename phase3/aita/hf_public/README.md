---
license: apache-2.0
language:
- en
tags:
- ai-generated-text-detection
- model-attribution
- text-classification
pretty_name: AITA Human-vs-AI (2026 generators)
---

# AITA Human-vs-AI corpus (2026 generators)

A second human-vs-AI corpus, a companion to
[`mild-rgb/eli5-human-vs-ai`](https://huggingface.co/datasets/mild-rgb/eli5-human-vs-ai),
in a deliberately different register: first-person judgment narratives from
r/AmItheAsshole, versus same-title posts written by seven 2026 models. 2,900
questions, one human post and one AI post each; ~414 documents per generator.

## The human side is redacted — reconstruct it from Scruples

The human posts are verbatim r/AmItheAsshole text, obtained via **AI2's Scruples
Anecdotes** (Lourie, Le Bras & Choi, 2020), which is Reddit-authored and not ours
to redistribute. Every human row here has its `text` **removed** and keeps only
identifiers and non-text metadata. To rebuild the human side, hydrate from Scruples
by `q_id` (the base36 Reddit post id):

- Scruples Anecdotes: https://github.com/allenai/scruples
- each human row's `q_id` is the Reddit submission id; join the post body back on it

The **AI side is included in full** (it is ours). So out of the box this dataset is
the AI class plus the scaffolding to reattach the human class yourself.

**Human-by-construction.** Every post dates to 2019-04-06 or earlier (recovered by
interpolating base36 ids against a dated dump), years before ChatGPT — the same
true-by-construction human guarantee the ELI5 corpus uses.

### One thing the redaction does not catch

The generation prompt tells the models not to restate the post title, and mostly
they do not. But **145 of the 2,900 AI documents — 5.0% — restate their source
title word for word**, so those Reddit-authored titles do appear verbatim in the
AI half of this dataset, even though the `question` field is stripped from every
row.

Saying so plainly is better than leaving it to be discovered. It exposes nothing
new: every row already carries `q_id`, which resolves to the original submission,
title and all. But this card tells you the human text is not ours to redistribute,
and a title is human text, so you should know that 145 of them are in here.

It is a property of the register rather than an oversight. AITA titles are
first-person sentences — "AITA for not including my coworker?" — that a model
writing that post naturally repeats. The companion ELI5 dataset, whose questions
are third-person queries that do not fit inside an answer, has 1 such case in
2,900.

## Fields

`doc_id`, `q_id` (Reddit id / hydration key), `label` (`human`|`ai`),
`generator` (the model, or `human`), `source` (`aita_floor` | `scruples`),
`split` (`train`|`dev`|`test`, question-grouped, stamped on the record),
`words`. AI rows add `text`, `text_norm`, `seed`, `draws`, `post` (post-op flags);
human rows add `date`, `votes`, `verdict` (all non-text).

Split: question-grouped 80/10/10 (4,640 / 580 / 580), a q_id's human and AI always
share a split. These are pool questions, disjoint from the dev/bench/heldout used to
tune the generation prompt, so they are clean for detector training.

## How the AI was generated

A single "floor" prompt (register constraints + an LLM-lexicon blacklist +
rate-matched draws calibrated to measured human marker rates), run across the seven
2026 generators, with per-model length calibration. The prompt was ported from the
ELI5 corpus and transferred without retraining: a 0-shot Haiku judge panel sits at
floor-parity (~51-52% balanced accuracy), the same "not reliably distinguishable"
bar as ELI5. **This is not a claim to beat ELI5** — at this sample size the two are
statistically indistinguishable, and the false-positive rate is partly the AITA
human register being harder to recognise, not the AI being more human-like.

## Honest limitations (recorded, not hidden)

- **A relationship-term repetition tell.** Pinning the naming rate to human levels
  induced over-repetition of "my wife / my sister" where a person switches to "she"
  (~1.8x human density). Accepted and documented rather than iterated away.
- **The verdict-question rate varies sharply by model** (0-70%), a genuine
  behavioural difference left as a corpus property.
- Full method, both measurement bugs found along the way, and the diversity checks
  are in the project's phase-3 write-up.

## Two uses

1. **Human-vs-AI detection** off the ELI5 distribution — a second domain to test
   whether a detector generalises across register, and to debias against a
   cross-domain bag-of-words shortcut.
2. **Model attribution.** Because all seven models saw the same prompt, the
   differences between them are pure model signal. A bag-of-words model identifies
   which of the seven wrote a held-out document ~90% of the time, and — trained on
   both this corpus and ELI5 — it is domain-robust. The signal is a shared
   2026-frontier signature plus a per-model fingerprint on top.

## Note on the era

Like the ELI5 detector, anything trained on this catches 2026-era models (including
ones held out entirely, ~95% of the time) but is near-blind to pre-2026 models.
Suitable for research and aggregate measurement, not for judging any one person's
document.
