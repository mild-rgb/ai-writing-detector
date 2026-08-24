# exp01 status

**Nothing has been trained. No GPU time spent.** The harness is built and
pre-flighted; the sweep is held until the corpus is complete.

## Ready

- `prep.py`, `code/train_cell.py`, `code/report.py`, `code/metrics.py`,
  `code/env_capture.py`, `code/contract.py`, `code/smoke_test.py`,
  `code/aggregate.py`, `code/setup_colab.sh`, `code/run_sweep.sh`,
  `requirements-colab.txt`, `README.md`.
- `python code/smoke_test.py --offline` — **all checks pass.** Covers prep
  determinism (against a frozen copy of the corpus, since the live one is still
  growing), split-leakage integrity on every cell, run-identity sensitivity,
  scoring maths, environment capture, and the contract checker's own ability to
  reject a record missing each clause.
- `train_cell.py --dry-run` exercised with a stubbed `transformers`; the
  argument, cell-selection and resume paths work.
- `aggregate.py` and `contract.py` exercised end to end on synthetic records
  built from the real split files. Every synthetic record re-derived exactly
  from its stored `probs`, which is the property the whole audit rests on.

## Added since the first pass

- **Warm-startable full checkpoints.** Model + optimizer + scheduler + RNG every
  `ckpt_steps` (default 100), into a directory named by the run's `run_key_hash`
  — so a warm start can only ever resume a run of byte-identical identity.
  `save_total_limit` caps VM disk, the best checkpoint is protected, and a run's
  checkpoints are deleted once its record is written unless `--keep-checkpoints`.
  Cadence is in `hp` and therefore in the run key, because it sets how often dev
  is scored and so which step is selected best. Every record states whether the
  run was warm-started.
- **Off-VM checkpoint mirroring**, opt-in via `MIRROR_CKPT=1` on a 600s cycle
  (results mirror stays at 120s). Opt-in because a ModernBERT-large full
  checkpoint is several GB.
- **Two more contract clauses** — full-state checkpointing and resume provenance
  — plus offline tests that a warm start picks the highest step, ignores a
  half-written checkpoint, and cannot cross run identities.
- **External-eval hook reserved.** `--final-model-dir` persists finished models
  keyed by run hash with a `run_record_meta.json` tracing back to the exact run.
  Nothing built beyond that; the planned Wattpad/Pangram slice is evaluation
  only and its labels are never training labels.

## Colab runtime — up

L4 23GB, driver 580.82.07, torch 2.11.0+cu128 (CUDA 12.8, cuDNN 91900),
python 3.13.15, **transformers==5.15.1 pin resolved**, bf16 available, 188 GB
free disk. `results/pip_lock.txt` written, 702 packages.

Phase 2 also ran on an L4 with transformers 5.15.1, so the two are comparable on
the axes phase 2 actually recorded. Its torch and CUDA are unknown — which is
the gap this harness exists to close.

## Corpus — complete, prepped, verified

`prep.py --require-complete` run against c4's finished corpus.

```
docs.jsonl   5,800 documents   sha256:3de9c320200087cb
splits.json  16 cells          sha256:3bbc53497cfdbbc5
corpus       dataset.jsonl:426cf6264061b3e8
2,900 questions   train 4,638 / dev 582 / test 580
```

The corpus layout changed with completion — `dataset.jsonl` plus
`train/dev/test.jsonl`, replacing `questions.jsonl` + `answers.jsonl` — so
`prep.py` was rewritten to read it. It now checks the two remaining sources of
split truth against each other (the `split` field and the split-file membership)
rather than picking a winner among three.

Verified independently rather than taken on report:
- 2,900 questions, every one exactly one human + one AI document, none straddling
  a split.
- Split files and the `split` field agree exactly on all 5,800 documents.
- Per-generator length is 0.959–1.015x the human median — the ≤6% recalibration
  claim holds, so word count is not a per-model shortcut.
- Normalisation removes every newline in the corpus.

## Timing run — done, contract-checked, ALL CHECKS PASSED

One real run, ModernBERT-large, `logo|gemini-3.7-flash|norm`, 1 epoch, seed 0,
on the finished corpus. L4, driver 580.82.07, torch 2.11.0+cu128 (CUDA 12.8,
cuDNN 91900), transformers 5.15.1, 702 packages in the record.

```
497 steps, 612.6 s  ->  1.233 s/step all-in (incl. eval + checkpoint at cadence 100)
tokens: median 414, max 965, TRUNCATED 0 of 830   (max_length 2048 is generous)
record: 98.5 KB, 830 test + 498 val per-document probabilities
newline_baseline_auc = 0.500 exactly, as `norm` requires by construction
```

Not a result — 1 epoch, 1 seed, and the smoke cell — but recorded because it is
the first real signal: **AUC 0.99998, balanced accuracy 0.988, recall 0.976,
FPR 0.000 on a generator held out whole.**

### Sweep arithmetic (derived from the measured rate, not estimated)

| | runs | GPU-hours on L4 |
|---|---|---|
| a — `binary\|norm`, 5 seeds x cls+mean | 10 | 6.2 |
| b — LOGO norm, 7 generators x 5 seeds | 35 | 18.7 |
| c — `binary\|raw`, 5 seeds | 5 | 3.1 |
| d — LOGO raw, 7 x 5 | 35 | 18.7 |
| **a+b+c (proposed)** | **50** | **27.9** |
| full grid | 85 | 46.6 |

Per run: binary 37.1 min (1,740 steps), LOGO 32.0 min (1,491 steps), at 3 epochs.

Levers, if 28 h is too much: 2 epochs instead of 3 → **18.6 h**; checkpoint
cadence 100 → 250 → 25.1 h; both → 16.8 h. Enabling early stopping would cut
more but makes runtime data-dependent and changes the recipe.

## A100 measured, same recipe, same corpus

Identical run re-executed on an A100-SXM4-80GB — same cell, same seed, same
batch size, same everything; only the hardware changed, and the env capture
records which is which.

```
L4        1.233 s/step   612.6 s   ALL CHECKS PASSED
A100      0.381 s/step   189.1 s   ALL CHECKS PASSED
                         3.24x faster per step, 3.16x on whole-plan wall-clock
```

| | L4 3ep | A100 3ep | L4 2ep | A100 2ep |
|---|---|---|---|---|
| a `binary\|norm` 5 seeds x cls+mean | 6.2 | 2.0 | 4.2 | 1.3 |
| b LOGO norm 7 x 5 | 18.7 | 5.9 | 12.7 | 4.1 |
| c `binary\|raw` 5 seeds | 3.1 | 1.0 | 2.1 | 0.7 |
| **a+b+c** | **27.9** | **8.8** | **19.0** | **6.1** |
| full grid | 46.6 | 14.8 | 31.7 | 10.1 |

Compute-unit rates are not exposed to the runtime and are not visible in the
Colab page as rendered, so they are NOT reported here as fact. The break-even is
arithmetic the user can finish once they read the two rates: **A100 costs fewer
units iff its units/hour is less than 3.16x the L4's.** At a 5x rate ratio it is
1.58x the units for 3.16x the speed.

### The same seed on different hardware moved a reported number

Worth recording because it is phase 2's lesson reproduced inside this harness,
on our own corpus, with everything else pinned:

```
            dev AUC hits 1.000 at   best ckpt   test AUC    test balanced acc
  L4              step 300            step 300   0.999998        0.9880
  A100            step 200            step 200   0.999994        0.9639
```

Same seeds, same `docs_sha256`, same recipe. The dev metric saturates at 1.000,
so once it ties, *which* checkpoint gets called best is decided by which step
first touched the ceiling — and that is downstream of floating-point differences
between the two GPUs. Threshold-free AUC barely moves; balanced accuracy at the
fixed 0.5 threshold moves **2.4 points**.

This is not a defect to fix by chasing determinism. It is the argument for the
standard already adopted: report mean ± SD across seeds, treat AUC as primary
because it is threshold-free, and keep the dev-chosen threshold as the secondary
reading — all of which the harness already records.

It does raise one design question for the sweep, flagged for decision rather than
decided here: **selecting the best checkpoint on a saturating metric is close to
arbitrary.** Selecting on dev cross-entropy instead (which does not saturate:
0.088 → 0.029 → …) would make the choice meaningful and would favour the better
calibrated checkpoint. Both are recorded either way.

## Fixed before it could do damage: early stopping semantics

Switching the checkpoint cadence from epochs to steps silently changed what
`patience=3` meant. Under phase 2's epoch cadence it meant "patience >=
max_epochs", i.e. best-epoch selection and no early stop. Under a step cadence
the same 3 means three *evaluations* — 300 steps — which is early stopping by
accident: a recipe change hiding inside a storage change, and it would have
fired constantly given dev AUC saturates by step 300.

Patience is now named in its unit (`early_stop_patience_evals`), defaults to
never firing, and the record states which behaviour was in force. There is a
contract clause and an offline test for it. No runs were affected — this was
caught before the sweep.

## One expected contract failure, noted so it does not confuse anyone

`results/smoke_results.jsonl` on the VM now fails exactly one clause —
`early-stopping semantics` — because that clause was added *after* the record was
written. The checker is behaving correctly: it refuses a record that predates a
clause rather than assuming the best. That record is a timing measurement, not a
result, and is superseded by the sweep.

## RUNNING: arm (a) only

Scope narrowed by the user: get a working human-vs-AI detector in hand before
deciding anything about generalization. **LOGO and `raw` are held entirely** —
they are absent from `--cells` and `--text`, so the launcher cannot reach them.

```
binary|norm   ModernBERT-large   A100-SXM4-80GB
5 seeds x {cls, mean} pooling = 10 runs, 2 epochs
checkpoint selected on dev LOSS   gradient checkpointing ON
```

Two epochs rather than three, and dev-loss selection rather than dev-AUC, both
on the evidence measured above: dev AUC saturates, so selecting on it is close to
arbitrary and moved balanced accuracy 2.4 points between two GPUs.

### A perfect score, interrogated before it was believed

The first runs returned AUC 1.000000 with FPR 0.000 across all seven generators.
That is a claim, not a result, so it was checked before being reported. Three
cheap CPU-only tests, now permanent in `code/bow_baseline.py`:

```
                                          AUC      balanced acc
  newline count alone (norm)             0.500        —      by construction
  ~40 mechanical markers                 0.687       0.631
  bag-of-words, words + punctuation      0.9997      0.9897
  bag-of-words, PUNCTUATION DELETED      0.9993      0.9845
  ModernBERT-large (seeds 0-2)           1.0000      0.999
```

- **Not one leaked token.** Best single feature on test reaches balanced accuracy
  0.814 — `(` and `)`, in 76.6% of human documents and 13.8% of AI ones.
- **Not punctuation.** Delete every punctuation mark and bag-of-words still gets
  AUC 0.9993.
- **Not the generator's blacklist.** The obvious worry — that the model learns
  "avoided *delve/crucial/robust* ⇒ AI" — is false. Banned terms appear in 8.6%
  of human and 1.7% of AI documents, and the rule "no banned term ⇒ AI" scores
  balanced accuracy **0.534**.

So the separation is real and distributed across ordinary word choice. The
consequence is that **the marker floor was far too easy a bar**: the binding
control is bag-of-words at 0.999, and the transformer must be reported above
that, never against 0.5.

### Checkpoint selection is now a recorded, keyed choice

`select_metric` lives in `hp`, so it is inside the run key — a run selected on
loss and one selected on AUC are different experiments and cannot be confused by
the resume check. The record names the selecting metric (`val_best_metric_name`)
separately from what the selected checkpoint actually scores (`val`), because
conflating those is how a saturating metric hides behind a number that looks like
a result. There is a contract clause and an offline test for it.

## Launch command, dry-run verified on the VM

```bash
bash code/run_sweep.sh /content/exp01 \
  --model answerdotai/ModernBERT-large --cells 'binary|norm' --text norm \
  --init-seeds 0,1,2,3,4 --pooling cls \
  --final-model-dir /content/drive/MyDrive/exp01_phase3/models
```

## Found while building — the corpus

**73 rows of `answers.jsonl` disagree with `questions.jsonl` on `split`**, all in
the same direction: the answer says `train`, the question says `test` or `dev`.
The question-level split was evidently revised after those answers were
generated. `questions.jsonl` is authoritative and matches the intended
2321/291/290.

`prep.py` takes the question's split, hard-fails above a documented ceiling, and
records the count in `dataset_card.json`. Reading the answer's split instead
would have moved 73 held-out documents into training. Not a corpus defect —
stale generation-time metadata — but it is a live trap for anything that reads
`answers.jsonl` alone.

Also noted: `answers_failures.jsonl` holds empty-completion failures,
Nemotron-only so far. Generator counts therefore end up slightly uneven; the
counts are recorded per generator per split in `dataset_card.json`.

## OOD eval — done, 24 Aug 2026 (sampled)

The published checkpoint was scored on seven out-of-domain sets on a Colab T4 via
`score_external.py`, which asserts preprocessing parity against
`run_record_meta.json`. Output in `phase3/study/_ood_scored_20260824/`, which is
kept locally and gitignored rather than committed.

The headline: **it generalises across domain, not across model vintage.** On AI
text from older models (GPT-4 and earlier) it misses 90–96% and ties the
bag-of-words floor; on the one set written by 2026 models it misses 16% against
the floor's 43%. False positives on human prose are low — 0 of 240 Stack Exchange
answers, 1.67% of news — with human fiction the exception at 8.75%, where the
floor is nearly as high.

Two things this run does not settle, both open:

- Six of the seven sets were scored on a 240-document sample, a time-budget
  decision. **A full-size re-run is outstanding.**
- The older-model sets shift domain and generator together, so they cannot
  separate the two. A same-domain, older-generator set would.

Full write-up and the per-generator breakdown are in `FINDINGS.md`.
