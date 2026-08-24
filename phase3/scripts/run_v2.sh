#!/bin/bash
# v2 dev generation, six-model roster, all six prompts.
# Run AFTER calibrate_length.py has been refreshed from the completed v1 data:
# the per-(prompt, model) length and paragraph constants are what v2 changes
# most, and they are only valid if measured on the run they are correcting.
cd /home/me/Documents/ai_writing_detector
for p in p1_pinned p3_shape p6_scope p2_mechanical p4_voice p5_disfluency; do
  echo "===== v2 $p ====="
  python3 phase3/scripts/generate.py "phase3/adversarial_prompts/$p" \
      --split dev --workers 28 \
      --out "phase3/adversarial_prompts/$p/answers_dev_v2.jsonl" 2>&1 | tail -3
done
echo "===== V2 COMPLETE ====="
