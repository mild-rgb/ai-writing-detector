#!/usr/bin/env bash
# Launch the sweep so it SURVIVES losing the notebook kernel or the browser tab.
#
#   bash run_sweep.sh /content/exp01 [extra train_cell.py args...]
#   MIRROR_CKPT=1 bash run_sweep.sh /content/exp01 ...     # also mirror checkpoints
#
# Colab drops kernels. Three things make that survivable:
#   1. setsid + nohup, so the sweep is not a child of the kernel
#   2. results.jsonl gains one line per FINISHED run, and train_cell.py skips any
#      run whose run_key_hash is already there -- a restart resumes on IDENTITY,
#      not on position in a list
#   3. full checkpoints (model + optimizer + scheduler + RNG) every N steps into
#      a directory named by that same run_key_hash -- so a run killed MID-way
#      warm-starts where it stopped instead of redoing the epoch, and can only
#      ever resume a run with byte-identical identity
#
# Mirrors run off the VM because Colab scratch does not survive a VM recycle.
#   results + env + dataset      every 120s   (small: this is the actual artifact)
#   newest checkpoint per run    every 600s   (large: opt-in via MIRROR_CKPT=1)
set -euo pipefail
EXP="${1:-/content/exp01}"; shift || true
MIRROR="${MIRROR:-/content/drive/MyDrive/exp01_phase3}"
MIRROR_CKPT="${MIRROR_CKPT:-0}"
WORK="${WORK:-/content/work}"
cd "$EXP"

if [ -f results/sweep.pid ] && kill -0 "$(cat results/sweep.pid)" 2>/dev/null; then
  echo "A sweep is already running (pid $(cat results/sweep.pid)). Tail: results/sweep.log"
  exit 0
fi

mkdir -p results "$WORK"
setsid nohup python code/train_cell.py \
  --docs "$EXP/docs.jsonl" --splits "$EXP/splits.json" --card "$EXP/dataset_card.json" \
  --out "$EXP/results/results.jsonl" --envdir "$EXP/env" --workdir "$WORK" \
  "$@" >> results/sweep.log 2>&1 &
echo $! > results/sweep.pid
echo "sweep pid $(cat results/sweep.pid) -- log: $EXP/results/sweep.log"

if [ ! -d "$(dirname "$MIRROR")" ]; then
  echo "WARNING: $MIRROR unreachable -- results live only on VM scratch and will be"
  echo "         LOST when the VM recycles. Mount Drive, or scp results.jsonl off."
  exit 0
fi

mkdir -p "$MIRROR"
setsid nohup bash -c "while true; do
    cp -f '$EXP/results/results.jsonl' '$MIRROR/' 2>/dev/null || true
    cp -f '$EXP/results/results_failures.jsonl' '$MIRROR/' 2>/dev/null || true
    cp -f '$EXP/results/sweep.log' '$EXP/results/pip_lock.txt' '$MIRROR/' 2>/dev/null || true
    cp -rf '$EXP/env' '$MIRROR/' 2>/dev/null || true
    cp -f '$EXP/docs.jsonl' '$EXP/splits.json' '$EXP/dataset_card.json' '$MIRROR/' 2>/dev/null || true
  sleep 120; done" >/dev/null 2>&1 &
echo $! > results/mirror.pid
echo "mirroring results to $MIRROR every 120s (pid $(cat results/mirror.pid))"

if [ "$MIRROR_CKPT" = "1" ]; then
  # Only the newest checkpoint of each in-flight run. A ModernBERT-large full
  # checkpoint is several GB (weights + optimizer state), so this is deliberately
  # opt-in and deliberately slower than the results mirror.
  setsid nohup bash -c "while true; do
      for d in '$WORK'/*/; do
        [ -d \"\$d\" ] || continue
        newest=\$(ls -d \"\$d\"checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
        [ -n \"\$newest\" ] || continue
        dest='$MIRROR/checkpoints/'\$(basename \"\$d\")
        mkdir -p \"\$dest\"
        rsync -a --delete \"\$newest\" \"\$dest\"/ 2>/dev/null || true
      done
    sleep 600; done" >/dev/null 2>&1 &
  echo $! > results/ckpt_mirror.pid
  echo "mirroring newest checkpoints to $MIRROR/checkpoints every 600s (pid $(cat results/ckpt_mirror.pid))"
  echo "  NOTE: full checkpoints are several GB each. Watch Drive quota."
else
  echo "checkpoint mirroring OFF (MIRROR_CKPT=1 to enable). Checkpoints protect"
  echo "  against a kernel death, not a VM recycle, unless mirrored."
fi
