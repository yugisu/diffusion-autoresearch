# autoresearch

This is an experiment to have the LLM do its own research.

## Setup

Branch: **`autoresearch/st2-dinov3-ssl4eo`**

In-scope files:

- `prepare.py` — fixed constants, dataset loading, and fixed evaluation. Do not modify.
- `train.py` — SSL pre-training script (Stage-1 complete). Do not modify.
- `st2.py` — Stage-2 supervised SFT script. **The file you modify.**

Data exists at `VISLOC_ROOT` (default from `prepare.py`). No setup needed.

## Experimentation

Each experiment runs on a single GPU. Each Stage-2 experiment has a **budget of ~45 minutes** (wall clock).

Run command:

```bash
uv run st2.py --wandb-run-name st2-expN > run.log 2>&1
```

Task details:

- Model: `facebook/dinov3-vitb16-pretrain-lvd1689m`
- Framework: PyTorch Lightning
- Logging: Weights & Biases (`autoresearch-ssl-dinov3-ssl4eos12-st2`, auth via env/WANDB_API_KEY)
- SSL checkpoint: `checkpoints/dinov3-ssl4eos12-best-r@1=0.615-mvicreg-569ef72.ckpt` (Stage-1 best, fixed)
- Train flights: `01, 02, 04, 05, 06, 08, 09, 10, 11`
- Validation flight: `03` (768 UAV queries, 2860 satellite chunks)
- Primary metric: `R@1` on flight 03, evaluated with fixed `evaluate_r1` in `prepare.py`
- Goal: **R@1 >= 0.90**

Satellite scale priors (from `prepare.py` — do not redefine):

```python
SAT_SCALES = {"01": 0.25, "02": 0.25, "03": 0.25, "04": 0.25, "05": 0.40, "06": 0.60, "08": 0.35, "09": 0.25, "10": 0.50, "11": 0.25}
```

### Baselines

| System | R@1 | R@5 | R@10 |
|--------|-----|-----|------|
| Zero-shot DINOv3 (no training) | 0.3398 | 0.5872 | 0.6732 |
| SSL-only Stage-1 (this branch — SSL4EO-S12, no UAV labels) | 0.6150 | — | — |
| Supervised only — Exp16 (from pretrained DINOv3) | 0.7357 | 0.8685 | 0.9167 |
| Reference two-stage best (ref branch, SSL R@1=0.530 → st2 best) | 0.7786 | 0.8945 | 0.9414 |

Our SSL checkpoint (R@1=0.615) is **+0.085 pp stronger** than the reference branch's SSL checkpoint (R@1=0.530). The two-stage st2-exp1 baseline should start at or above 0.7786.

### Evaluation protocol

- Load SSL checkpoint, merge LoRA into weights, unfreeze all backbone params.
- Backbone CLS token → 2-layer MLP projection head → 512-d L2-normalised embedding.
- Satellite TTA: average embeddings over 4 rotations (0°/90°/180°/270°), re-normalise.
- Evaluate with fixed `evaluate_r1` from `prepare.py`.

**What you CAN do:**

- Modify `st2.py` only. Loss, sampler, augmentations, optimizer, scheduler, head architecture, gradient clipping, warmup, LR values, max_epochs — all fair game.

**What you CANNOT do:**

- Modify `prepare.py` or `train.py`.
- Change the SSL checkpoint path (always load `dinov3-ssl4eos12-best-r@1=0.615-mvicreg-569ef72.ckpt`).
- Install new packages.

**Simplicity criterion**: all else equal, simpler is better. Keep complexity proportional to gains.

## Output format

At run completion, extract key metrics from log:

```bash
grep "VAL flight 03\|Best checkpoint\|Best val/R@1" run.log | tail -20
```

If grep is empty, run crashed. Read stack trace:

```bash
tail -n 50 run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 6 columns:

```tsv
commit	R@1	R@5	R@10	status	description
```

1. git commit hash (short, 7 chars)
2. R@1 achieved (e.g. `0.412760`) — use `0.000000` for crashes
3. R@5 achieved — use `0.000000` for crashes
4. R@10 achieved — use `0.000000` for crashes
5. status: `keep`, `discard`, or `crash`
6. short text description of what this experiment tried

Example:

```tsv
commit	R@1	R@5	R@10	status	description
a1b2c3d	0.312500	0.654948	0.781250	keep	baseline SSL→supervised reference Exp9 config
b2c3d4e	0.338542	0.671875	0.796875	keep	UAV RandomRotation aug + UAV TTA at inference
```

Do not commit `results.tsv`.

## Known lessons from reference two-stage branch (13 experiments)

The reference branch ran the full Stage-2 experiment cycle starting from a weaker SSL checkpoint (R@1=0.530). Key validated findings (do not repeat these experiments):

### What the st2.py baseline already encodes (Exp9 config)

- **GPS exclusion zone (60m pos / 60–150m ignored)**: single largest gain (+0.78 pp). Tiles in the 60–150m ring around a UAV query are ambiguous (share ground features but are wrong answers). Excluding them from the InfoNCE denominator sharpens the loss from ~1.1 → ~0.825 nats. The 60m/150m threshold is validated; 40m/100m was consistently worse.
- **k_flights=3 batch sampler**: harder geographic in-batch negatives. Chunks from 3 different flights per batch give better discriminative signal than k=2.
- **4-tier LLRD** (5e-6 / 1e-5 / 1.5e-5 / 2e-5, head at 2e-5): smoother LR gradient through SSL-adapted blocks 8–11.
- **CosineWarmRestarts T_0=10**: cycle-1 peaks at ~epoch 5, cycle-2 restart at epoch 10 is typically destructive (model finds a sharp basin; restart overshoots it). EarlyStopping patience=6 catches the peak.
- **gradient_clip_val=1.0**: stabilises training without dampening convergence.
- **2-layer MLP head** (no LayerNorm): converges in 5 epochs. 3-layer+LN head was too slow for the 20-epoch budget.

### What did NOT help (do not retry)

| Idea | Why not |
|------|---------|
| Warmup + plain cosine, no restarts | Stalled epoch 4; restarts needed to escape early plateau |
| Stronger UAV aug during SFT (SSL-strength) | Too aggressive; peaked epoch 2 |
| batch=96, k=3 (32/flight) | Bigger batch reduced per-flight hard negative density |
| 3-layer projection head + LayerNorm | Too slow to converge in 20 epochs |
| patience=10 through restart dip | Cycle-2 peaked 0.7604 — the restart itself is the problem |
| ReduceLROnPlateau | Over-decayed LR by epoch 8, collapsed to 0.69 |
| Tighter GPS 40m/100m threshold | 40–60m positives are informative; removing them weakens signal |
| UAV TTA + satellite queue | Queue adds noisy early-epoch denominator terms; UAV TTA inconsistent without rotation aug |

### Training dynamics (reference Exp9)

```
Epoch  1: R@1 ≈ 0.700   (rapid convergence from SSL init)
Epoch  5: R@1 ≈ 0.779   ← cycle-1 peak (T_0 boundary)
Epoch  6: R@1 ≈ 0.737   (LR bottoms at cycle end)
Epoch 10: R@1 ≈ 0.742   (restart — LR spike, overshoots basin)
Epoch 11: R@1 ≈ 0.716   (EarlyStopping fires; cycle-2 never recovers)
```

## Experiment plan

All experiments build on the validated Exp9 config already encoded in `st2.py`. Execute in order.

**st2-exp1 — Baseline: Exp9 config from reference branch, our SSL checkpoint**

Run the exact reference Exp9 configuration with the stronger SSL4EO-S12 checkpoint (R@1=0.615 vs reference R@1=0.530). This establishes our two-stage baseline. Expected to meet or exceed 0.7786 due to better SSL initialisation.

**st2-exp2 — RandomRotation on UAV training aug + UAV TTA at inference**

Reference Exp13 found UAV TTA hurt (-0.028 pp) because UAV was trained with horizontal flip only — rotated views produced inconsistent embeddings that degraded the TTA average. The fix is to add `RandomRotation(degrees=180)` to the UAV training augmentation to make the backbone rotation-invariant across both modalities, then enable 4-rotation TTA at inference for UAV queries. This directly targets the drone-heading orientation gap (drones fly arbitrary headings; satellite is always north-up). Run on top of Exp1 config if Exp1 confirms the baseline holds; otherwise run on Exp1 best commit.

**st2-exp3 — T_0=6 cosine schedule to exploit cycle-1 peak**

Reference dynamics show cycle-1 peaks at ~epoch 5, and every restart at epoch 10 is destructive. Shorten T_0 to 6 epochs so the cycle trough (and thus one restart) occurs at epoch 6, then set EarlyStopping patience=3 to catch the peak before the restart fires — or simply stop after cycle-1 (max_epochs=6). This removes the destructive restart entirely while keeping the CosineWarmRestarts shape that proved necessary to escape early plateaus. Baseline: best config from Exp1/2.

**st2-exp4 — Momentum-encoder satellite queue (MoCo-style hard negatives)**

Reference Exp13 used a satellite embedding queue from the *main encoder*, which caused noisy early-epoch denominator terms. The correct approach is a *momentum encoder* (EMA of main encoder, β=0.995) with a queue of recent satellite embeddings — standard MoCo-v2. The momentum encoder's embeddings are stable across training steps, removing the noise. Queue size=1024 (16× larger than reference queue=128), GPS-far filter (>150m from each UAV anchor). Only queue satellite (K) embeddings; UAV (Q) stays from main encoder. Run on top of best Exp1/2 config.

**st2-exp5 — Free experimentation**

Based on Exp1–4 findings, explore promising directions. Options:
- Listwise ranking loss (AP@1 or SmoothAP) — directly optimises the ranking metric instead of InfoNCE; targets the "correct tile is in top-10 but not rank-1" gap (R@10=0.94, R@1=~0.78)
- Combine Exp2 (UAV rotation invariance) + Exp3 (short T_0) + Exp4 (momentum queue) if each independently improved
- Longer training (25 epochs, single cosine cycle over full budget) if EarlyStopping consistently fires at epoch 5–6
- Patch re-ranking: use spatial patch token similarity to re-rank top-10 CLS candidates at inference (zero extra training cost)

### Training notes

- **Epoch budget**: 20 epochs unless a specific experiment justifies more or less.
- **Checkpoints**: saved to `checkpoints/<wandb-run-name>/` (e.g. `checkpoints/st2-exp1/`). Best by `val/R@1`.
- **Early stopping**: patience=6 on `val/R@1` (unless experiment changes this).
- **Precision**: `16-mixed` on A100.

## Prior experiments findings

### SSL4EO-S12 Stage-1 (train.py — this branch)

- **Multi-positive VICReg on SSL4EO-S12**: Best SSL checkpoint achieves R@1=0.615 on flight 03 (vs zero-shot 0.340). Training on global satellite data from China/East Asia lat 15–55, lon 90–135. Multi-positive approach (n_ssl_positives=3, all 4 seasons) + LoRA r16 = final best.
- **LoRA config** (matches the checkpoint): rank=16, alpha=32, last 4 blocks, all QKV projections.
- **VICReg** outperformed InfoNCE for this SSL task: avoids false-negative collapse when seasonal pairs look similar.
- **Cross-scale pairs** (anchor 25–50%, positive 75–100%) simulate UAV vs satellite scale gap.
- **Asymmetric augmentation**: strong anchor aug (ColorJitter+GaussianBlur+RandomGrayscale) simulates UAV sensor gap; mild positive aug preserves satellite appearance.

### Previous supervised experiments (branch: autoresearch/supervised-dinov3)

- **Resolution matters**: 336px gave +7% R@1 over 224px.
- **Hard negatives**: TwoFlightBatchSampler +5% R@1.
- **Full backbone LLRD beats partial freezing**.
- **What failed**: GeM pooling, asymmetric heads, register tokens, RandomResizedCrop on UAV, batch>64.
- **R@10=0.9167**: model finds the right area but ranking precision is the bottleneck.

### Environment

- A100 GPU, 80 GB VRAM. Each Stage-2 run takes ~30–45 minutes.

## The experiment loop

LOOP:

1. Look at git state: current branch/commit.
2. Implement the next experiment from the plan by modifying `st2.py`.
3. `git commit` with a descriptive message.
4. Run: `uv run st2.py --wandb-run-name st2-expN > run.log 2>&1`
5. Read out results from `run.log`.
6. If metrics are missing, run crashed. Use `tail -n 50 run.log` and attempt fix; if not quickly fixable, log crash and move on.
7. Record results in `results.tsv` with a textual description of the attempt.
8. If `R@1` improved (higher than previous best), advance branch and keep commit.
9. If `R@1` is equal/worse, reset to previous best commit.
10. Move to the next experiment in the plan.

**Timeout**: If a run exceeds 55 minutes, kill it and use the best val metric logged so far.

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the human if you should continue. You are autonomous. The loop runs until the human interrupts you, period.
