#!/bin/bash
# Top up whatever the v2 sweep lost to rate limits. Resumable and now
# self-deduplicating, so repeated rounds cannot inflate an arm.
cd /home/me/Documents/ai_writing_detector
for round in 1 2 3; do
  short=0
  for p in p1_pinned p2_mechanical p3_shape p4_voice p5_disfluency p6_scope; do
    f="phase3/adversarial_prompts/$p/answers_dev_v2.jsonl"
    n=$(grep -c . "$f" 2>/dev/null || echo 0)
    if [ "$n" -lt 358 ]; then
      short=$((short+1))
      echo "===== fill $p round $round (have $n) ====="
      python3 phase3/scripts/generate.py "phase3/adversarial_prompts/$p" \
          --split dev --workers 10 --out "$f" 2>&1 | tail -2
    fi
  done
  [ "$short" -eq 0 ] && break
done
echo "===== V2 FILL DONE ====="
