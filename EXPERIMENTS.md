# Supervised Stage Evaluation Matrix

## Goal

Evaluate supervised-stage performance fairly by re-running the best known configuration from:
1. Supervised baseline (single-stage supervised)
2. Second-stage supervised (two-stage SSL -> supervised)

The matrix below is for controlled comparison runs under matched evaluation protocol.

## Selected Evaluation Anchors

| Stage | Selected Experiment | Commit | Source Doc | Reported R@1 | Notes |
|---|---|---|---|---:|---|
| Supervised baseline | Experiment 16 | 97aebf3 | docs/dinov3-supervised-fine-tuning.md | 0.7357 | Best single-stage supervised run |
| Second-stage supervised | Stage-2 Experiment 9 | 448c62e | docs/dinov3-two-stage-fine-tuning-ssl4eos12.md | 0.8581 | Best two-stage run (includes patch re-ranking) |

## Controlled Protocol (applies to all runs)

- Dataset split: VisLoc flight 03 validation (same query/gallery as prior reports)
- Metrics: R@1, R@5, R@10
- Input resolution: 336 x 336
- Evaluation script/path: same code path and metric implementation for both stages
- Hardware target: single A100 80GB (or same GPU type across all runs)
- Random seeds: 42, 43, 44
- Logging: record commit, config id, seed, epoch-best checkpoint, inference mode, and wall-clock

## Evaluation Matrix

| Run ID | Stage | Anchor Commit | Seed | Inference Mode | Primary Output |
|---|---|---|---:|---|---|
| EVAL-S1-01 | Supervised baseline | 97aebf3 | 42 | Baseline eval (as documented) | R@1/R@5/R@10 |
| EVAL-S1-02 | Supervised baseline | 97aebf3 | 43 | Baseline eval (as documented) | R@1/R@5/R@10 |
| EVAL-S1-03 | Supervised baseline | 97aebf3 | 44 | Baseline eval (as documented) | R@1/R@5/R@10 |
| EVAL-S2-01 | Second-stage supervised | 448c62e | 42 | CLS + patch re-ranking (K=50, alpha=0.5) | R@1/R@5/R@10 |
| EVAL-S2-02 | Second-stage supervised | 448c62e | 43 | CLS + patch re-ranking (K=50, alpha=0.5) | R@1/R@5/R@10 |
| EVAL-S2-03 | Second-stage supervised | 448c62e | 44 | CLS + patch re-ranking (K=50, alpha=0.5) | R@1/R@5/R@10 |

## Aggregation Plan

For each stage:
- Report mean and standard deviation across the 3 seeds for R@1, R@5, R@10

For stage comparison:
- Report seed-paired deltas: Delta = Stage-2 - Supervised baseline
- Report mean Delta R@1 and 95% confidence interval

## Result Table Template (fill after runs)

| Stage | Seed | R@1 | R@5 | R@10 | Best Epoch | Elapsed (min) | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| Supervised baseline | 42 |  |  |  |  |  |  |
| Supervised baseline | 43 |  |  |  |  |  |  |
| Supervised baseline | 44 |  |  |  |  |  |  |
| Second-stage supervised | 42 |  |  |  |  |  |  |
| Second-stage supervised | 43 |  |  |  |  |  |  |
| Second-stage supervised | 44 |  |  |  |  |  |  |

## Fairness Note

This matrix compares the best known supervised-stage recipes as end-to-end systems. Because the second-stage anchor includes patch re-ranking at inference, this is a practical performance comparison, not an isolated training-loss-only comparison.

If you also want a strict training-only comparison, add an additional stage-2 row using the best pre-reranking checkpoint (Stage-2 Experiment 8, commit 388d67f).
