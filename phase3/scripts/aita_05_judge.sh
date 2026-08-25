#!/bin/bash
# Three independent Haiku passes over the AITA dev set, then score.
#
# Three passes, not one, for the reason in phase3/study/judge_compare/RESULTS.md:
# re-running the SAME route on the SAME items moved detection by 13 points and
# the false-positive rate by 9, while balanced accuracy moved 2. Balanced
# accuracy is the headline, quoted as a mean with the SD across passes.
#
# Three matches phase3/data/judge_dev_p*.json exactly, so these numbers sit
# beside the six ELI5 prompts and the floor without adjustment.
set -euo pipefail
PDIR=${1:-phase3/aita/prompts/aita_floor}
SPLIT=${2:-dev}
PASSES=${3:-3}

for p in $(seq 0 $((PASSES - 1))); do
  python3 phase3/scripts/aita_04_build_judge.py "$PDIR" --split "$SPLIT" --pass "$p"
done

for p in $(seq 0 $((PASSES - 1))); do
  echo "--- pass $p ---"
  python3 phase3/scripts/judge_api.py \
    "$PDIR/$SPLIT/pass$p/batches" "$PDIR/$SPLIT/pass$p/haiku" \
    --model anthropic/claude-haiku-4.5 \
    --instruction-file phase3/aita/judge/instruction.txt
done

python3 phase3/scripts/score_passes.py "$PDIR" --split "$SPLIT" \
  --json phase3/aita/data/judge_${SPLIT}_floor.json
