#!/bin/bash
set -euo pipefail

for seed in 42 122 234; do
    uv run train-supervised.py --wandb-run-name "E1-s${seed}-base-model-supervised-routine" --seed $seed --backbone-init base
    uv run train-st2.py        --wandb-run-name "E2-s${seed}-ssl-model-st2-routine"         --seed $seed --backbone-init ssl
    uv run train-supervised.py --wandb-run-name "E3-s${seed}-ssl-model-supervised-routine"  --seed $seed --backbone-init ssl
    uv run train-st2.py        --wandb-run-name "E4-s${seed}-base-model-st2-routine"        --seed $seed --backbone-init base
done
