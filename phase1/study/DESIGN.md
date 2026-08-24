# Detectability study — design

## What the v1-v7 loop could not answer

n=30 with a single judge has SE 9.1 and a minimum detectable difference of
**26.8 points**. Every prompt comparison in that loop was below its own noise
floor, and no version ever varied a single line, so nothing was attributable.

## This study

**Metric.** 2AFC: judge sees one human and one AI answer to the same question,
picks the AI. Chance 50%. Score is |accuracy - 50|; below chance is as bad as
above (a detector learns an inverted rule just as easily).

**Power.** n=100 questions x R=3 independent judges per condition.
SE 3.3 pts, minimum detectable difference **9.3 pts** (alpha .05, power .80).
Replicates matter more than trials: R=3 x n=100 beats R=1 x n=300, because a
single judge's idiosyncratic read cannot be averaged out at R=1.

**Question sets.** Fixed and disjoint.
- `dev` (n=100) every condition runs on these same questions, so comparisons
  are paired and the question-difficulty component (measured ICC 0.127)
  differences out.
- `heldout` (n=100) touched once, at the end.
- `burned` (n=150) permanently retired: judges saw those human answers during
  v1-v7 and the prompts were written while looking at them.

**Per-model prompts.** The same instruction moves different models by different
amounts and sometimes in opposite directions (v7's "go easy on 'just'": grok
-50.0 pts p=0.0001, qwen +3.3 p=0.80, deepseek -3.3 p=0.80). Each model is
tuned and scored separately.

**Multiplicity.** Holm-Bonferroni across all tests in a phase. Point estimates
are reported with Wilson 95% CIs, never bare.

## Phases

1. Baseline v6 and v7 per model on dev, R=3.  6 conditions, 18 judges.
2. Per-model v8 designed from phase-1 measurements + judge cues.
3. v8 per model on dev, R=3.  3 conditions, 9 judges.
4. Best config vs **sonnet** judges, R=3, on dev; then heldout confirmation.

## Stopping rules

- A model is **saturated** against a judge tier when its pooled accuracy is
  within 9.3 pts of 50 and the Holm-corrected test does not reject chance.
- Escalate to the next judge tier only when all three models are saturated.
- Stop tuning a model after 2 consecutive versions with no significant change;
  record the floor rather than continuing to fit noise.
