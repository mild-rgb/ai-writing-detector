#!/usr/bin/env bash
# Colab setup for the phase 3 detector sweep. Idempotent; safe to re-run after a
# kernel drop.
#
#   bash setup_colab.sh /content/exp01
#
# Leaves behind /content/exp01/results/pip_lock.txt -- the resolved environment,
# frozen at setup time, separate from the copy each record carries.
set -euo pipefail
EXP="${1:-/content/exp01}"
cd "$EXP"

echo "== GPU =="
nvidia-smi || { echo "NO GPU. Runtime > Change runtime type > GPU (A100 or L4)."; exit 1; }

echo "== installing pinned libraries =="
pip install -q -r requirements-colab.txt || {
  echo "PINNED INSTALL FAILED -- installing latest transformers instead."
  echo "The version is recorded in every run record; say so in FINDINGS.md."
  pip install -q -U transformers accelerate
}

mkdir -p results env /content/work
pip freeze > results/pip_lock.txt
echo "== resolved environment =="
python code/env_capture.py | head -1
echo
echo "torch:        $(python -c 'import torch;print(torch.__version__, torch.version.cuda)')"
echo "transformers: $(python -c 'import transformers;print(transformers.__version__)')"
echo "bf16:         $(python -c 'import torch;print(torch.cuda.is_bf16_supported())')"
echo
echo "pip_lock.txt written ($(wc -l < results/pip_lock.txt) packages)"
echo "Next: bash code/run_sweep.sh $EXP  (or run smoke_test.py first)"
