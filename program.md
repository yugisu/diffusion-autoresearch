# autoresearch

This is an experiment to have the LLM do its own research.

## Setup

Branch: **`autoresearch/self-supervised-dinov3-2`**

In-scope files:

- `prepare.py` — fixed constants, dataset loading, and fixed evaluation. Do not modify.
- `train.py` — SSL pre-training script (Exp13 checkpoint already saved). Do not modify.
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
- Logging: Weights & Biases (`autoresearch-st2-dinov3`, auth via env/WANDB_API_KEY)
- SSL checkpoint: `checkpoints/dinov3-ssl-best-r@1=0.53-e656447.ckpt` (Exp13, fixed)
- Train flights: `01, 02, 04, 05, 06, 08, 09, 10, 11`
- Validation flight: `03` (768 UAV queries, 2860 satellite chunks)
- Primary metric: `R@1` on flight 03, evaluated with fixed `evaluate_r1` in `prepare.py`
- Goal: **R@1 >= 0.90**

Satellite scale priors (fixed, do not change):

```python
sat_scales = {"01": 0.25, "02": 0.25, "03": 0.25, "04": 0.25, "05": 0.40, "06": 0.60, "08": 0.35, "09": 0.25, "10": 0.50, "11": 0.25}
```

### Baselines

| System | R@1 | R@5 | R@10 |
|--------|-----|-----|------|
| Zero-shot DINOv3 (no training) | 0.3398 | 0.5872 | 0.6732 |
| SSL-only Exp13 (no UAV labels) | 0.5299 | 0.6641 | 0.7057 |
| Supervised Exp16 (from pretrained DINOv3) | 0.7357 | 0.8685 | 0.9167 |
| **st2-exp1 (SSL → supervised SFT)** | **0.7539** | **0.8750** | **0.9232** |

st2-exp1 is the current best and the baseline all further st2 experiments must beat.

### Evaluation protocol

- Load SSL Exp13 checkpoint, merge LoRA into weights, unfreeze all backbone params.
- Backbone CLS token → 2-layer MLP projection head → 512-d L2-normalised embedding.
- Satellite TTA: average embeddings over 4 rotations (0°/90°/180°/270°), re-normalise.
- Evaluate with fixed `evaluate_r1` from `prepare.py`.

**What you CAN do:**

- Modify `st2.py` only. Loss, sampler, augmentations, optimizer, scheduler, head architecture, gradient clipping, warmup, LR values, max_epochs — all fair game.

**What you CANNOT do:**

- Modify `prepare.py` or `train.py`.
- Change the SSL checkpoint path (always load `dinov3-ssl-best-r@1=0.53-e656447.ckpt`).
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
a1b2c3d	0.312500	0.654948	0.781250	keep	baseline supervised contrastive max_epochs=10
b2c3d4e	0.338542	0.671875	0.796875	keep	harder sat augmentation + lower temperature
c3d4e5f	0.329427	0.667969	0.792969	discard	too strong color jitter
d4e5f6g	0.000000	0.000000	0.000000	crash	OOM at batch_size=64
```

Do not commit `results.tsv`.

## Known issues from st2-exp1 & experiment plan

### Loss dynamics analysis (st2-exp1)

Step-level loss statistics per epoch:

| Epoch | mean | std | p2p | Note |
|-------|------|-----|-----|------|
| 1 | 2.098 | **0.403** | 1.720 | no warmup — worst oscillation |
| 6–9 | ~1.32 | ~0.195 | ~0.80 | stable mid-run |
| 10 | 1.258 | 0.207 | 0.911 | restart |
| 15 | 1.066 | 0.185 | 0.829 | restart → 4th cycle regressed |

Three root causes:
1. **No warmup**: Starting at full LR with a freshly initialised projection head causes epoch-1 std=0.403 (2× stable-run average). Large early gradients propagate into the SSL-adapted backbone before the head has converged.
2. **Structural batch variance**: `TwoFlightBatchSampler` + multi-positive InfoNCE with variable positive counts per batch creates irreducible step-level variance (~std=0.18–0.22).
3. **CosineWarmRestarts T_mult=1**: The 4th restart at epoch 15 fired full LR into a model that had just peaked at R@1=0.7539 (epoch 14). The entire final 6-epoch cycle never recovered (0.71–0.74).

### Experiment plan (execute in order)

**st2-exp2 — Stabilise: gradient clipping + warmup + plain cosine decay**

Three targeted fixes in one experiment:
- `gradient_clip_val=1.0` in the Trainer (direct fix for step std)
- 2-epoch linear warmup for all param groups (fixes epoch-1 spike)
- Replace `CosineAnnealingWarmRestarts` with `SequentialLR(linear_warmup, CosineAnnealingLR)` over 20 epochs — removes the destructive 4th restart

Expected: smoother loss curve, best epoch pushed into epochs 16–18 instead of epoch 14, R@1 > 0.7539.

**st2-exp3 — Lower head LR: 5e-5 → 2e-5**

The projection head starts from random init and currently gets lr=5e-5 — 10× the early backbone and 2.5× the late backbone. This asymmetry is the dominant source of early instability. Reduce head LR to match the late-backbone rate (2e-5). The head is shallow and will still adapt quickly. Run on top of the best scheduler config from exp2.

**st2-exp4 — Deeper LLRD: add mid-to-late transition layer**

Current LLRD has 3 backbone tiers (5e-6 / 1e-5 / 2e-5). Add a 4th tier: blocks 6–7 at 5e-6 (currently grouped with blocks 4–5 at 1e-5), blocks 8–9 at 1e-5, blocks 10–11 at 2e-5. This gives a smoother LR gradient through the SSL-adapted layers (8–11 were LoRA-trained and are most sensitive to over-shooting).

**st2-exp5 — Free experimentation**

Based on which fixes helped most in exp2–4, explore:
- Asymmetric augmentation on UAV query (stronger ColorJitter + GaussianBlur + RandomGrayscale to simulate sensor gap — matches SSL Exp13 anchor augmentation)
- Longer training (30 epochs with warmup + cosine)
- Larger projection head (768 → 768 → 512 with LayerNorm)
- Combine best components from exp2–4

### Training notes

- **Epoch budget**: 20 epochs unless a specific experiment justifies more.
- **Checkpoints**: saved to `checkpoints/st2-dinov3/`. Best by `val/R@1`.
- **Early stopping**: patience=6 on `val/R@1`.
- **Precision**: `16-mixed` on A100.

## Prior experiments findings

### SSL branch (train.py)

- **Cross-scale pairs** (anchor=25–50%, positive=75–100%) are the core SSL innovation: directly simulates UAV (zoomed-in) vs satellite (full-scale) domain gap.
- **Asymmetric augmentation** on anchor (stronger ColorJitter + GaussianBlur + RandomGrayscale) simulates UAV sensor/temporal gap and is what enables 5 consecutive epochs of improvement (Exp13). Mild augmentation on positive preserves satellite appearance.
- **LoRA (rank=16, alpha=32, last 4 blocks)** + fixed temperature=0.07 is the stable training config. Learnable temperature collapsed in early experiments.
- **Degradation pattern**: Without strong augmentation, peak is always at epoch 0–1 (warmup boundary), then monotonic degradation as SSL erodes UAV-compatible DINO features.

### Stage-2 SFT (st2.py / st2-exp1)

- **SSL → supervised SFT beats supervised-only by +1.82 pp R@1** (0.7539 vs 0.7357). The SSL backbone adaptation to satellite scale invariance gives a meaningful head-start.
- **Best epoch was 14 of 20** — the 4th cosine restart regressed the model. Removing restarts or using T_mult=2 should push best epoch later and higher.
- **R@10=0.9232** at best: the correct chunk is in the top 10 almost universally. The remaining gap to R@1=0.90 is a ranking precision problem, not a retrieval coverage problem.
- **LoRA merge is clean**: strict load_state_dict → merge → unfreeze → supervised LLRD works correctly. No interference.

### Supervised branch (train-supervised.py, Exp16)

- **TwoFlightBatchSampler** (32+32 per batch from 2 different flights) is the most impactful hard-negative strategy (+5 pp R@1).
- **Satellite TTA** (4 rotations) adds ~0.5 pp R@1 at zero training cost. Always on.
- **GeM pooling, asymmetric heads, register tokens, RandomResizedCrop on UAV, larger batch sizes** all failed.

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
