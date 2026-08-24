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

A ModernBERT-large classifier fine-tuned to label a single document as
human-written or AI-written, trained on the
[mild-rgb/eli5-human-vs-ai](https://huggingface.co/datasets/mild-rgb/eli5-human-vs-ai)
corpus (long-form r/explainlikeimfive answers versus same-question answers from
seven 2026 models).

## Read this before you use it

**This is a real detector that carries transferable signal, but it is poorly
calibrated, so do not trust any single verdict.** It is published as a research
artifact.

- **In-domain it is perfect** — balanced accuracy 1.000 on held-out data — but
  "in-domain" means ELI5-style Q&A, the seven specific 2026 generators it was
  trained on, under one generation prompt. That number is not a general claim and
  does not transfer. A bag-of-words logistic regression scores 0.9993 on the same
  easy in-domain data, so the in-domain number alone says little.
- **Out of domain it genuinely generalizes.** On a completely different domain
  (fiction) it reaches balanced accuracy 0.863 and AUC 0.945: 83.8% recall on AI
  fiction against an 11.3% false-positive rate on human fiction. Off distribution
  it clearly beats the bag-of-words baseline (0.745 / 0.890), so it learned real
  transferable signal, not just the training setup.
- **But its confidence is untrustworthy per document.** It pins most scores to 0
  or 1 and makes confident mistakes: about 6% of human fiction is flagged at
  p >= 0.99, and adjacent passages of one continuous text flip between "certainly
  human" and "certainly machine." The aggregate accuracy is real; an individual
  high-confidence verdict is not reliable.

**Do not use this model for consequential decisions about individuals** —
academic-integrity checks, moderation, hiring, or accusing anyone of using AI.
Off-domain it flags about **11% of genuinely human writing as AI** at the default
0.5 threshold, roughly one human document in nine, and about 6% at near-certainty
(p >= 0.99). A false accusation is a real harm. It is suitable for aggregate
research and measurement, not for judging one person's document.

## Intended use

Research into AI-text detection and its failure modes: reproducing the in-domain
ceiling, studying the bag-of-words floor, and measuring out-of-distribution
collapse. That is what it is good for.

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

A genuine AI-text detector that degrades from a perfect 1.000 in-domain to 0.863
balanced accuracy out-of-domain, beats a bag-of-words baseline off distribution,
and is poorly calibrated at the single-document level. Trust it in aggregate, not
on any one verdict.
