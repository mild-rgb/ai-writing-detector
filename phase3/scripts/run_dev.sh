#!/bin/bash
# Generate the dev arm for every prompt, one after another.
cd /home/me/Documents/ai_writing_detector
for p in p1_pinned p2_mechanical p3_shape p4_voice p5_disfluency p6_scope; do
  echo "===== $p ====="
  python3 phase3/scripts/generate.py "phase3/adversarial_prompts/$p" \
      --split dev --workers 20 2>&1
done
echo "===== ALL DONE ====="
