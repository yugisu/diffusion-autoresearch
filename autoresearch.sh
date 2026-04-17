#!/usr/bin/env bash
set -euo pipefail

# Fast pre-check
uv run python -m py_compile train.py

rm -f run.log
start_ts=$(date +%s)

set +e
timeout 1800 uv run train.py > run.log 2>&1
run_rc=$?
set -e

end_ts=$(date +%s)
run_seconds=$((end_ts - start_ts))

if [[ $run_rc -ne 0 ]]; then
  echo "METRIC R@1=0"
  echo "METRIC R@5=0"
  echo "METRIC R@10=0"
  echo "METRIC run_seconds=${run_seconds}"
  echo "Run failed with exit code ${run_rc}. Last 50 lines:" >&2
  tail -n 50 run.log >&2 || true
  exit $run_rc
fi

python - <<'PY'
import re
from pathlib import Path

log = Path("run.log").read_text(errors="ignore")

vals = re.findall(r"\[VAL flight 03\]\s+R@1=([0-9.]+)\s+R@5=([0-9.]+)\s+R@10=([0-9.]+)", log)
if vals:
    r1, r5, r10 = vals[-1]
else:
    r1 = r5 = r10 = "0"

print(f"METRIC R@1={r1}")
print(f"METRIC R@5={r5}")
print(f"METRIC R@10={r10}")
PY

echo "METRIC run_seconds=${run_seconds}"

grep "R@1\|R@5\|R@10\|Best checkpoint\|Best val/R@1" run.log | tail -n 40 || true
