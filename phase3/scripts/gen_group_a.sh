#!/bin/bash
cd /home/me/Documents/ai_writing_detector
for p in p1_pinned p2_mechanical p3_shape; do
  echo "===== $p ====="
  python3 phase3/scripts/generate.py "phase3/adversarial_prompts/$p" --split dev --workers 24 2>&1
done
echo "===== GROUP A DONE ====="
