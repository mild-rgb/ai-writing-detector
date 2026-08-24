#!/bin/bash
# Re-run judging for any pass directory holding fewer than 24 prediction files.
# judge_api.py skips batches it has already written, so this fills gaps only.
# Low concurrency: the shortfalls appeared while generation was running against
# the same account, and a judge hole looks exactly like a judge verdict.
cd /home/me/Documents/ai_writing_detector
for round in 1 2 3; do
  short=0
  for d in phase3/adversarial_prompts/*/dev/pass*/haiku; do
    n=$(ls "$d"/*.json 2>/dev/null | wc -l)
    if [ "$n" -lt 24 ]; then
      short=$((short+1))
      b="${d%/haiku}/batches"
      echo "repair $d ($n/24)"
      python3 phase3/scripts/judge_api.py "$b" "$d" \
          --model anthropic/claude-haiku-4.5 --workers 3 2>&1 | tail -2
    fi
  done
  [ "$short" -eq 0 ] && break
done
echo "===== JUDGE REPAIR DONE ====="
