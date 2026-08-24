#!/bin/bash
cd /home/me/Documents/ai_writing_detector
for p in p4_voice p5_disfluency p6_scope; do
  echo "===== $p ====="
  python3 phase3/scripts/generate.py "phase3/adversarial_prompts/$p" --split dev --workers 24 2>&1
done
echo "===== GROUP B DONE ====="
