#!/bin/bash
# Generate the ai half of the phase-3 corpus. One document per question, the
# generator fixed by the corpus assignment. Resumable and self-deduplicating.
cd /home/me/Documents/ai_writing_detector
for round in 1 2 3 4; do
  n=$(grep -c . phase3/corpus/answers.jsonl 2>/dev/null || echo 0)
  echo "===== round $round, have $n / 2772 ====="
  [ "$n" -ge 2765 ] && break
  w=20; [ "$round" -gt 2 ] && w=8
  python3 phase3/scripts/generate.py phase3/corpus/floor \
      --corpus phase3/corpus/questions.jsonl --workers $w \
      --out phase3/corpus/answers.jsonl 2>&1 | tail -4
done
echo "===== CORPUS GENERATION DONE ====="
