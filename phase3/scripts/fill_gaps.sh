#!/bin/bash
# Fill whatever the main passes lost to rate limiting, at low concurrency so the
# account-wide limit is not the binding constraint. Resumable: only missing
# (question, model) pairs are generated.
cd /home/me/Documents/ai_writing_detector
for round in 1 2 3; do
  for p in p1_pinned p2_mechanical p3_shape p4_voice p5_disfluency p6_scope; do
    d="phase3/adversarial_prompts/$p"
    [ -f "$d/answers_dev.jsonl" ] || continue
    n=$(grep -c . "$d/answers_dev.jsonl")
    if [ "$n" -lt 415 ]; then
      echo "===== fill $p round $round (have $n) ====="
      python3 phase3/scripts/generate.py "$d" --split dev --workers 6 2>&1 | tail -3
    fi
  done
done
echo "===== FILL DONE ====="
