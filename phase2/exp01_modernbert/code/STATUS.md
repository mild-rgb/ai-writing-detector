# STATUS — exp01 execution state

**Last updated: 2026-08-22 ~13:00 BST. Supersedes all earlier versions of this
file.** An earlier version of this file said the experiment was PAUSED with 0 of
12 cells, described the prep.py hash bug as unresolved, and told the reader NOT
to regenerate splits.json. **All three of those statements are now wrong.**
They are recorded here only so anyone who remembers them knows they are stale.

## SCOPE CUT, 2026-08-22: TASK C IS NOT RUN

The user cut Task C entirely ("just do A and B"). **24 runs, not 60.** Tasks A
and B run exactly as specced — 3 seeds, unchanged recipe, unchanged splits. This
is a scope cut, not a design change.

**Consequence, recorded so absence is never read as a result:** Task C was the
only task producing a binary AI-vs-human AUC comparable to phase 1's 0.771 and to
`wc -l`'s 0.803. Therefore:

* **Predictions 7-11 and the 0.65 falsification line are UNRUN.**
  Not passed. Not failed. Never executed.
* Predictions 1-6 (Tasks A and B) remain live and evaluable.

Task B's `HUMAN` share covers adjacent ground — it is a false-negative rate for a
novel generator — but it is **not** C's binary AUC and must not be presented as a
substitute for it.

The LFM2.5 frozen control IS still run, on the A and B test documents. It
produces an unfitted binary AI-vs-human AUC on the same test questions, which is
the reference point C would otherwise have been compared against. That is the
*unfitted* control, a different quantity from C's fitted number — it partially
covers the gap but does not close it.

## Current state: RUNNING (A and B only)

The user lifted the pause and gave an explicit go-ahead. The sweep is executing
on a Colab L4.

* **Design: 20 cells, 3 seeds, 60 runs** (A_4way 2, B_unseen_assign 6,
  C_binary 12). Not the earlier 12-cell / 36-run design.
* **Model: ModernBERT-large (395M).** The earlier "base first, escalate only if
  base clears 0.65" gate was removed by the user. Base was never run — there are
  no base cells and no base-vs-large comparison.
* Run order: A, then B, then C, then the frozen LFM2.5 control.

## RESOLVED: the prep.py determinism bug

**Fixed.** The seed is now string-derived, `random.Random(f"{SEED}|{g}|{cond}")`,
not `hash()` on a tuple. Verified independently: prep.py was relocated to a
scratch directory and re-run twice under randomised `PYTHONHASHSEED`, and both
runs reproduced `docs.jsonl` **and** `splits.json` byte-identically (`cmp` clean).

**splits.json WAS deliberately regenerated** and is now
sha256:`dfc459701c2fbbb5` (20 cells). `docs.jsonl` is unchanged at
`d8722c165e142b81`. Only the 6 `seen` cells changed; all `unseen` cells were
byte-identical, as predicted. The earlier instruction "do not regenerate
splits.json" applied only while the file was unreproducible. **It no longer
applies. Regenerating is now safe and is the point of the fix.**

## Where the results live — READ THIS

`results.jsonl` is written **on the Colab VM at `/content/exp01/results.jsonl`**,
not on this machine. It is written *after each run completes*, so it does not
exist until run 1 finishes. Absence of a local `results.jsonl` is therefore NOT
evidence that the sweep failed. Partials are mirrored to
`phase2/exp01_modernbert/results/` on this machine as they land.

## Verification done before the sweep (all passed)

1. Hashes match locally AND on the VM after upload.
2. All 20 cells: no question leakage, no doc in both train and test, train
   classes a subset of declared classes. A is 824/352 balanced 206/88 per class.
   B is 618 train / 88 test with the held-out generator provably absent from
   train and present as the *only* class in test. C is 618/176 with 412/206 and
   88/88.
3. `wc -l` baselines reproduce exactly: 0.887 / 0.749 / 0.585 raw, 0.500 norm.
4. Six pairwise newline AUCs computed for Task A **before any training**:
   raw HUMAN/deepseek 0.749, HUMAN/qwen 0.585, HUMAN/grok 0.887,
   deepseek/qwen 0.675, deepseek/grok 0.664, qwen/grok 0.807; norm all 0.500.
5. `code/smoke_test.py` passes on all three task paths.

## Bugs caught before any test document was read

* **`compute_metrics` was never passed to the Trainer.** `metric_for_best_model`
  was not found, so early stopping was silently disabled. All 60 runs would have
  trained the full 10 epochs with no validation-based model selection and
  produced a normal-looking results file. Fixed. This is why
  `code/smoke_test.py` exists and asserts `val_sel_best is not None`.
* **transformers v5 removals** (Colab ships 5.15.1): `overwrite_output_dir`
  (dropped, inert) and `warmup_ratio`. The latter is part of the pre-registered
  recipe, so it was reconstructed exactly as
  `warmup_steps = 0.1 * steps_per_epoch * max_epochs`, not dropped.

## Recipe (fixed in advance, not tuned against test)

lr 5e-5, batch 8, max 10 epochs, warmup ratio 0.1, weight decay 0.01, early stop
patience 3 on val (macro-F1 for A/B, AUC for C), val = 15% of train **questions**,
balanced accuracy at a fixed 0.5 threshold. Class weights for C only. Test is
read exactly once per run.
