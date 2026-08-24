#!/bin/bash
# Wait for the main sweep, then top up to the full 2,902 and assemble.
cd /home/me/Documents/ai_writing_detector
while pgrep -f "run_corpus[.]sh" > /dev/null; do sleep 30; done
echo "===== main sweep ended ====="
bash phase3/scripts/topup_corpus.sh
echo "===== assembling ====="
python3 phase3/scripts/build_corpus.py
echo "===== CORPUS COMPLETE ====="
