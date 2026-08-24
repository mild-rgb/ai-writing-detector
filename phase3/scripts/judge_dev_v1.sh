#!/bin/bash
# Descriptive judge pass over the v1 dev outputs. NOT a held-out result: dev is
# the split the prompts were tuned on, and no v2 decision is made from these
# numbers. Three independent API passes per prompt; balanced accuracy with the
# spread across passes is the reported statistic.
cd /home/me/Documents/ai_writing_detector
for p in p1_pinned p3_shape p2_mechanical p4_voice p5_disfluency p6_scope; do
  d="phase3/adversarial_prompts/$p"
  echo "===== judge $p ====="
  for n in 0 1 2; do
    python3 phase3/scripts/build_bench.py "$d" --split dev --pass $n \
        --allow-substitute 2>&1 
    python3 phase3/scripts/judge_api.py "$d/dev/pass$n/batches" "$d/dev/pass$n/haiku" \
        --model anthropic/claude-haiku-4.5 --workers 6 2>&1 
  done
done
echo "===== DEV JUDGING DONE ====="
