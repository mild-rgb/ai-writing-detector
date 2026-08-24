---
license: apache-2.0
base_model: answerdotai/ModernBERT-large
datasets:
- mild-rgb/eli5-human-vs-ai
language:
- en
pipeline_tag: text-classification
tags:
- ai-generated-text-detection
- text-classification
- modernbert
---

# ELI5 Human-vs-AI detector (ModernBERT-large)

> Distilling Pangram would have been boring, and I wouldn't have learned anything.

A ModernBERT-large classifier fine-tuned to label a single document as
human-written or AI-written, trained on the
[mild-rgb/eli5-human-vs-ai](https://huggingface.co/datasets/mild-rgb/eli5-human-vs-ai)
corpus (long-form r/explainlikeimfive answers versus same-question answers from
seven 2026 models).

## Read this before you use it

**This is a real detector, but it detects a particular thing: how 2026 models
write.** It carries that into subject matter it has never seen, it is nearly blind
to text from older models, and it is badly calibrated, so no single verdict from it
should be trusted. It is published as a research artifact.

- **In-domain it is perfect** — balanced accuracy 1.000 on held-out data — but
  "in-domain" means ELI5-style Q&A, the seven specific 2026 generators it was
  trained on, under one generation prompt. That number is not a general claim and
  does not transfer. A bag-of-words logistic regression scores 0.9993 on the same
  easy in-domain data, so the in-domain number alone says little.
- **It moves to a new domain, but not to an older generation of models.** Put it
  on AI-written fiction, a domain it never saw, and it works: 83.8% of the AI text
  caught against an 11.3% false-positive rate on human fiction, balanced accuracy
  0.863 and AUC 0.945, well clear of a bag-of-words baseline at 0.745 and 0.890.
  But that fiction was written by 2026 models, the same generation the detector
  trained on. Give it text from older models and it goes nearly blind:

  | AI text from | this model misses | bag-of-words misses |
  |---|---|---|
  | WildChat — GPT-4, GPT-3.5 | 94.6% | 96.3% |
  | MAGE — LLaMA, OPT, davinci, and others | 96.3% | 93.8% |
  | RAID — llama-chat, MPT | 90.4% | 91.7% |
  | AI fiction — 2026 models | **16.2%** | 43.4% |

  On all three older-model sets it misses nearly everything and ties a trivial
  word-counting baseline. The fiction row is the only one written by 2026 models,
  and it is the only row where the detector clearly wins. So what it learned is
  what today's frontier models sound like. That knowledge crosses into new
  subject matter; it does not reach backwards to how machines wrote in 2022.
- **It rarely mistakes human writing for AI, outside fiction.** It flagged none of
  240 Stack Exchange answers and 1.67% of 240 news articles. Human fiction is the
  exception, at roughly 9%, but the bag-of-words baseline flags nearly as much of
  it, which points at something about the fiction register rather than a fault
  peculiar to this model.
- **Two caveats on the older-model numbers.** Each of those three sets was scored
  on a 240-document sample (the fiction row is the full 364), so read them as
  clearly bad rather than precisely measured. And they change two things at once —
  a new domain *and* an older generator — so they answer "can it catch a model it
  has never seen," not "how well does it do on current models."
- **Its confidence is untrustworthy on any single document.** It pins most scores
  to 0 or 1 and makes confident mistakes: about 6% of human fiction is flagged at
  p >= 0.99, and adjacent passages of one continuous text flip between "certainly
  human" and "certainly machine." The aggregate accuracy is real; an individual
  high-confidence verdict is not reliable.

**Do not use this model for consequential decisions about individuals** —
academic-integrity checks, moderation, hiring, or accusing anyone of using AI.
On human fiction it flags about **11% of genuinely human writing as AI** at the
default 0.5 threshold, roughly one document in nine, and about 6% at near-certainty
(p >= 0.99). A false accusation is a real harm. It is suitable for aggregate
research and measurement, not for judging one person's document.

## Intended use

Research into AI-text detection and its failure modes: reproducing the in-domain
ceiling, studying the bag-of-words floor, and measuring what happens off
distribution. That is what it is good for. It is not a general-purpose detector,
and in particular it should not be pointed at text that may have been written by
an older model, where it barely detects anything.

## How to use

The model was trained on **whitespace-normalized** text (newline runs collapsed)
with **mean pooling** (baked into the config). Normalize inputs the same way or
results will drift.

```python
import re, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

name = "mild-rgb/eli5-human-vs-ai-detector"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForSequenceClassification.from_pretrained(name).eval()

def norm(t):                      # match training preprocessing
    return re.sub(r"\s+", " ", t).strip()

text = norm("your document here")
enc = tok(text, truncation=True, max_length=2048, return_tensors="pt")
with torch.no_grad():
    p_ai = model(**enc).logits.softmax(-1)[0, 1].item()
print(f"P(AI) = {p_ai:.3f}")      # label 1 = AI, 0 = human; threshold 0.5
```

Best on documents of roughly 250-800 words, the training length.

## Training

- **Base model:** `answerdotai/ModernBERT-large`.
- **Data:** the `mild-rgb/eli5-human-vs-ai` corpus, question-grouped
  train/validation/test splits, all seven generators seen during training.
- **Recipe:** bf16, `attn_implementation="sdpa"`, `reference_compile=False`,
  learning rate 3e-5, weight decay 8e-6, 2 epochs, mean pooling, gradient
  checkpointing on. Checkpoint selected on dev loss.
- **This checkpoint** is one seed of a five-seed run; all five reached AUC 1.000
  in-domain with zero variance, so the seed choice is immaterial to the in-domain
  result.
- Full method, the bag-of-words floor, and the out-of-distribution probe are
  documented in the project's phase-3 write-up.

## Limitations, in one line

A genuine detector of 2026-model writing. It carries that skill into new subject
matter — a perfect 1.000 balanced accuracy in-domain, 0.863 on fiction it never
saw — but not to text from older models, where it misses roughly nine documents in
ten and ties a trivial baseline. It is also poorly calibrated: trust it in
aggregate, never on one verdict.
