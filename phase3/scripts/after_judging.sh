#!/bin/bash
# Wait for the dev judging sweep to end, repair any short pass directory, then
# restart v2 generation at higher concurrency. The two were deliberately run
# together to overlap wall-clock, but contention on one account is what left
# five pass directories 3-4 batches short, so they get separated for the repair.
cd /home/me/Documents/ai_writing_detector
while pgrep -f "judge_dev_v1[.]sh" > /dev/null; do sleep 20; done
echo "===== dev judging sweep ended ====="
bash phase3/scripts/repair_judging.sh
echo "===== repair finished, raising v2 concurrency ====="
pkill -f "scripts/generate[.]py" 2>/dev/null
sleep 3
sed -i 's/--workers 12/--workers 28/' phase3/scripts/run_v2.sh
pkill -f "run_v2[.]sh" 2>/dev/null
sleep 2
setsid nohup bash phase3/scripts/run_v2.sh >> phase3/data/run_v2.log 2>&1 < /dev/null &
echo "===== v2 restarted at 28 workers ====="
