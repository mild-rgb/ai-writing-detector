# ai-writing-detector

Research code for detecting AI-written text, built around a long-form
r/explainlikeimfive human-vs-AI corpus and a ModernBERT-large detector.

**This repository is code only.** Data and model weights are intentionally
excluded (see `.gitignore`) and are **not** redistributable here:

- The human half of the corpus is verbatim Reddit text (ELI5 questions and
  answers) that we do not have the right to redistribute. A redacted, hydration-
  style public version and the trained detector live on Hugging Face instead:
  - Dataset: [`mild-rgb/eli5-human-vs-ai`](https://huggingface.co/datasets/mild-rgb/eli5-human-vs-ai)
  - Model: [`mild-rgb/eli5-human-vs-ai-detector`](https://huggingface.co/mild-rgb/eli5-human-vs-ai-detector)
- Out-of-distribution eval material (news, Stack Exchange, and benchmark AI text)
  is rebuilt from its original sources by the scripts in `phase3/scripts/`, not
  stored here.

## Layout

- `phase1/` — corpus construction, generation, and blind-judge study.
- `phase2/` — first ModernBERT detector experiment.
- `phase3/` — the current corpus, detector training harness
  (`exp01_detector/code/`), bag-of-words floor, and OOD evaluation builders.

Each phase has its own README and write-up (`*_EXPLAINED.md`, `NARRATIVE.md`,
`STATUS.md`, `FINDINGS.md`) with the method and results.

## Reproducing

Generation and judging need `OPENROUTER_API_KEY` (env or a `.env` at the root —
never committed). Training runs on Colab; see
`phase3/exp01_detector/requirements-colab.txt` and `code/setup_colab.sh`.
