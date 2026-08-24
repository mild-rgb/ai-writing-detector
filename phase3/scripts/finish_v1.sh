#!/bin/bash
# Finish v1 on the six-model roster: p6 from scratch, and top up the cells the
# rate-limit storm left short in the other five. grok is not regenerated -- it
# is out of the project and its v1 cells stand at whatever n they reached.
cd /home/me/Documents/ai_writing_detector
for p in p6_scope p3_shape p2_mechanical p4_voice p5_disfluency p1_pinned; do
  echo "===== finish $p ====="
  python3 phase3/scripts/generate.py "phase3/adversarial_prompts/$p" --split dev --workers 18 2>&1 | tail -3
done
echo "===== V1 COMPLETE ====="
