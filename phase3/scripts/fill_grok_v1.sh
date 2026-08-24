#!/bin/bash
# Bounded top-up of grok's v1 cells. grok leaves the roster in v2, so this is
# not worth fighting the account-wide rate limit for: one low-concurrency pass
# per prompt, whatever lands is labelled with its actual n.
cd /home/me/Documents/ai_writing_detector
for p in p2_mechanical p4_voice p5_disfluency p3_shape; do
  echo "===== grok fill $p ====="
  timeout 900 python3 phase3/scripts/generate.py "phase3/adversarial_prompts/$p" \
      --split dev --models x-ai/grok-4.6 --workers 4 2>&1 | tail -2
done
echo "===== GROK FILL DONE ====="
