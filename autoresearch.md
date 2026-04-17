# Autoresearch: Supervised DINOv3 VisLoc R@1 Optimization

## Objective
Optimize `train.py` to maximize retrieval Recall@1 (R@1) for UAV-to-satellite geo-localization on VisLoc validation flight `03`, using supervised fine-tuning of `facebook/dinov3-vitb16-pretrain-lvd1689m` under the existing fixed evaluation in `prepare.py`.

## Metrics
- **Primary**: `R@1` (unitless, higher is better)
- **Secondary**: `R@5`, `R@10`, `run_seconds`

## How to Run
`./autoresearch.sh` — runs one training experiment and emits structured `METRIC name=value` lines.

## Files in Scope
- `train.py` — supervised training pipeline (model, losses, data transforms, optimization, schedules, callbacks); **main file to optimize**.
- `autoresearch.md` — living experiment context and decisions log.
- `autoresearch.sh` — benchmark harness for reproducible run + metric extraction.
- `autoresearch.ideas.md` — backlog for promising but not-yet-tried ideas.

## Off Limits
- `prepare.py` (fixed datasets/evaluation harness)
- Dependency changes / package installs
- Any modification to fixed metric implementation (`evaluate_r1`, `build_ground_truth` in `prepare.py`)

## Constraints
- Keep experiments comparable; preserve evaluation path and validation split (flight `03`).
- Prefer simple changes unless complexity clearly improves primary metric.
- Treat crashes/OOMs as learnings; log and move on.

## Workload Notes
- Training command: `uv run train.py`
- Full run writes `run.log`
- Typical run budget: up to ~30 min wall clock for safety timeout
- Expected key log lines include:
  - `[VAL flight 03] R@1=... R@5=... R@10=...`
  - `Best val/R@1: ...`

## What's Been Tried
- **Baseline (current defaults, batch 256, cosine schedule)**: R@1 **0.4349** (best so far), R@5 0.6549, R@10 0.7604.
- **Removed light geometric augmentations** (`RandomHorizontalFlip`/`RandomRotation` removed): regressed to R@1 **0.4284** → discard. Augmentations appear helpful, keep them.
- **User-directed dynamics test** (batch 128, LR 2e-5, 2-epoch warmup, no cosine, augmentations kept): strong regression (final R@1 **0.3841**, best ckpt ~0.4076) with longer runtime; discard.
- Added harness-level parsing for training diagnostics (`train_samples_per_sec`, `train_samples_seen`) via `[TRAIN PROGRESS]` lines for future step-budget analysis.
