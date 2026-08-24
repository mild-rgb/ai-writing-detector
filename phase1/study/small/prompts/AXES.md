# Descriptive axes (a1 - a10)

Extracted from `lfm2_detector.ipynb` cells 8 and 9, where they existed only as
inline Python dicts (`AXES` and `MORE`). Written out here so they are versioned
alongside the `s*` classification prompts and survive the notebook.

Each file is the **complete** prompt: the shared `HEAD` block followed by that
axis's question. `{question}` and `{text}` are the format slots.

## Why these exist

The seven `s*` prompts ask the model to classify directly. All seven scored
below chance (0.292 - 0.447), and the two written specifically to fix the
inversion -- `s6_expert`, `s7_specificity` -- were the weakest. From cell 8:

> At 1.2B the prompt controls how sharply the model reads a feature, but not
> which label it maps that feature to. So stop asking it to classify. Ask it the
> descriptive question it can answer, and do the label mapping outside the model.

The axes ask a yes/no question about a **property**. Margin is
`logP(yes) - logP(no)`; the label mapping happens in code. All ten came out in
the correct direction.

## The ten, as scored on long2 (from RESULTS.md section 3)

| axis | question, in short | AUC | sha256[:16] |
|---|---|---|---|
| a4_generic | just as good an answer to a slightly different question? | **0.723** | `aedc7fb4af42fd16` |
| a1_polished | polished, comprehensive, tidily closed? | 0.712 | `64198534f38b2cc9` |
| a5_concludes | wraps up rather than stops? | 0.696 | `b3a211d94c2947b4` |
| a10_textbook | printable in a textbook unedited? | 0.690 | `c2015b01db127497` |
| a8_firsthand | anything only knowable from doing it? | 0.682 | `831c59eac9dd6b79` |
| a7_evenpace | evenly paced? | 0.635 | `9f5bbccba220917e` |
| a3_effort | typed once and posted without rereading? | 0.569 | `d3119c45c2e8f408` |
| a9_opinion | any opinion, complaint or joke? | 0.540 | `dc092f33d4348955` |
| a2_person | a specific identifiable person behind it? | 0.534 | `e928a612c6483a92` |
| a6_necessary | is every sentence doing work? | 0.500 | `738761d47360ff74` |

`a4_generic` is the frozen axis. Its hash matches `frozen_config.json`'s
`prompt_sha` exactly, which confirms these files are byte-identical to what was
actually scored -- the extraction is verified, not assumed.

## The comparison worth keeping

`s2_polish` asks about polish as a classification and scores 0.292. `a1_polished`
asks about the same feature descriptively and scores 0.712. Same feature,
inverted result; the only difference is who does the labelling.

The nine losing axes are kept because that spread -- 0.500 to 0.723 -- is what
makes the winner legible. An axis reported alone is not a finding.
