#!/usr/bin/env bash
set -euo pipefail

# Fast pre-check
uv run python -m py_compile train.py

rm -f run.log
start_ts=$(date +%s)

set +e
timeout 3600 uv run train.py --batch-size 256 --max-steps 460 > run.log 2>&1
run_rc=$?
set -e

end_ts=$(date +%s)
run_seconds=$((end_ts - start_ts))

metrics=$(python - <<'PY'
import re
from pathlib import Path

log = Path("run.log").read_text(errors="ignore")
vals = re.findall(r"\[VAL flight 03\]\s+R@1=([0-9.]+)\s+R@5=([0-9.]+)\s+R@10=([0-9.]+)", log)
if vals:
    r1, r5, r10 = vals[-1]  # latest metrics (for timed-out runs)
    print(f"R1={r1}")
    print(f"R5={r5}")
    print(f"R10={r10}")
else:
    print("R1=0")
    print("R5=0")
    print("R10=0")

train_stats = re.findall(r"\[TRAIN PROGRESS\]\s+samples_seen=([0-9]+)\s+avg_samples_per_sec=([0-9.]+)", log)
if train_stats:
    samples_seen, samples_per_sec = train_stats[-1]
    print(f"SAMPLES_SEEN={samples_seen}")
    print(f"SAMPLES_PER_SEC={samples_per_sec}")
else:
    print("SAMPLES_SEEN=0")
    print("SAMPLES_PER_SEC=0")
PY
)

eval "$metrics"
echo "METRIC R@1=${R1}"
echo "METRIC R@5=${R5}"
echo "METRIC R@10=${R10}"
echo "METRIC run_seconds=${run_seconds}"
echo "METRIC train_samples_per_sec=${SAMPLES_PER_SEC}"
echo "METRIC train_samples_seen=${SAMPLES_SEEN}"

if [[ $run_rc -ne 0 ]]; then
  # timeout with valid metrics is treated as a successful datapoint
  if [[ $run_rc -eq 124 && "${R1}" != "0" ]]; then
    echo "Timed out at 3600s; using latest validation metrics from log." >&2
  else
    echo "Run failed with exit code ${run_rc}. Last 50 lines:" >&2
    tail -n 50 run.log >&2 || true
    exit $run_rc
  fi
fi

grep "R@1\|R@5\|R@10\|TRAIN PROGRESS\|Best checkpoint\|Best val/R@1" run.log | tail -n 80 || true
