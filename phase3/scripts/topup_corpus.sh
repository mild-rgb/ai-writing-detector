#!/bin/bash
# Top up the corpus after the main sweep. Threshold is the full 2,902 (the
# original runner was written against 2,772, before the bench/heldout reclaim).
# Resumable and self-deduplicating; low concurrency so a rate-limited generator
# is not the binding constraint on the tail.
cd /home/me/Documents/ai_writing_detector
TARGET=2902
for round in 1 2 3 4; do
  n=$(grep -c . phase3/corpus/answers.jsonl 2>/dev/null || echo 0)
  echo "===== topup round $round, have $n / $TARGET ====="
  [ "$n" -ge $((TARGET - 5)) ] && break
  w=10; [ "$round" -gt 2 ] && w=5
  python3 phase3/scripts/generate.py phase3/corpus/floor \
      --corpus phase3/corpus/questions.jsonl --workers $w \
      --out phase3/corpus/answers.jsonl 2>&1 | tail -3
done
echo "===== CORPUS TOPUP DONE ====="
