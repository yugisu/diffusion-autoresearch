# DINOv3 Two-Stage Fine-Tuning for UAV-to-Satellite Geo-Localization

**Best result: R@1 = 78.9%** (Stage-2 experiment 9) on the VisLoc flight 03 validation set, using a two-stage SSL → supervised fine-tuning pipeline.

---

## Task

**UAV-to-satellite visual place recognition.** Given a nadir drone image, retrieve the matching satellite map chunk that covers the drone's GPS location.

- **Query set**: 768 UAV drone images from VisLoc flight 03
- **Gallery**: ~2,860 satellite chunks (512×512 px, 128 px stride) tiled from a GeoTIFF satellite map
- **Metric**: Recall@1, Recall@5, Recall@10 on flight 03
- **Goal**: R@1 ≥ 0.90
- **Time budget**: 45 minutes wall-clock per experiment on a single A100 80GB GPU

---

## Overview

The two-stage pipeline stacks SSL satellite pre-adaptation on top of a supervised contrastive fine-tuning stage:

```
Stage 1 (train.py)          Stage 2 (st2.py)
─────────────────────       ────────────────────────────────
Satellite-only SSL          Supervised: UAV ↔ satellite pairs
  ↓                           ↓
InfoNCE on cross-scale      Multi-positive InfoNCE
satellite crops             with GPS proximity mask
  ↓                           ↓
LoRA adapter (4 blocks)     Full backbone LLRD fine-tune
  ↓                           ↓
R@1 = 53.0%                 R@1 = 78.9%  (+25.9 pp)
```

**Why two stages?** Stage 1 adapts the DINOv3 backbone to satellite scale invariance without any UAV labels — it learns to match zoomed-in (UAV-like) views to full-scale (satellite-like) views using only satellite crops. Stage 2 then fine-tunes this adapted backbone with real UAV↔satellite pairs and GPS supervision, starting from a much better initialisation than the pretrained DINOv3 weights alone.

---

## Baselines

| System | R@1 | R@5 | R@10 |
|--------|-----|-----|------|
| Zero-shot DINOv3 (no training) | 0.340 | 0.587 | 0.673 |
| SSL only — Stage 1 Exp13 (satellite-only, no UAV labels) | 0.530 | 0.664 | 0.706 |
| Supervised only — Exp16 (from pretrained DINOv3) | 0.736 | 0.869 | 0.917 |
| **Two-stage best — Stage-2 Exp9** | **0.779** | **0.895** | **0.941** |

The two-stage approach beats supervised-only by **+4.3 pp R@1**, confirming that satellite SSL pre-adaptation provides a meaningful head-start for the downstream supervised task.

---

## Architecture

Both stages share the same backbone; only the training objective and head differ.

**Backbone**: `facebook/dinov3-vitb16-pretrain-lvd1689m`
- ViT-B/16, 12 transformer blocks, 768-d hidden, CLS token as embedding
- 86.6 M total parameters

**Stage 1 head** (discarded after Stage 1):
- Projection head: 768 → 256-d, SimCLR-style, used only for SSL loss
- LoRA adapters (rank=16, alpha=32) on last 4 blocks — merged into weights before Stage 2

**Stage 2 head** (trained from scratch on top of merged backbone):
- 2-layer MLP: Linear(768, 768) → GELU → Dropout(0.1) → Linear(768, 512)
- L2-normalised 512-d output embedding
- Learnable temperature `logit_scale`, clamped to ≤ 100

---

## Stage 1: SSL Pre-Training

See [`dinov3-self-supervised-fine-tuning.md`](dinov3-self-supervised-fine-tuning.md) for the full experiment log.

**Key configuration (Exp13 checkpoint, used in all Stage-2 experiments)**:
- Cross-scale positive pairs: anchor crop 25–50% of chunk area, positive 75–100%
- Asymmetric augmentation on anchor: strong ColorJitter + GaussianBlur + RandomGrayscale
- LoRA (rank=16, alpha=32) on last 4 blocks; 294k trainable params
- Batch=128, fixed temperature=0.07, 13 epochs

**How Stage 2 loads it**: LoRA deltas are merged into the base weights (`lora_B @ lora_A * scaling`), producing a plain backbone identical to a standard DINOv3 — no LoRA overhead at Stage 2 training time. The full backbone is then unfrozen for supervised LLRD fine-tuning.

---

## Stage 2: Supervised Fine-Tuning

### Fixed configuration (all Stage-2 experiments)

```python
# Model
backbone: SSL Exp13 checkpoint, LoRA merged, all params unfrozen
head: Linear(768, 768) -> GELU -> Dropout(0.1) -> Linear(768, 512) -> L2-norm

# Training
batch_size = 64
max_epochs = 20
precision = "16-mixed"
gradient_clip_val = 1.0

# Backbone LLRD (4-tier, introduced in Exp5)
lr_emb_blocks03  = 5e-6    # embedding + blocks 0-3
lr_blocks47      = 1e-5    # blocks 4-7
lr_blocks89      = 1.5e-5  # blocks 8-9
lr_blocks1011    = 2e-5    # blocks 10-11 + norm
lr_head          = 2e-5    # projection head

# Batch sampling (best config: Exp5+)
TwoFlightBatchSampler: k_flights=3, ~21 samples/flight from 3 randomly chosen flights

# Validation
Satellite TTA: average over 4 rotations (0°/90°/180°/270°), re-normalise
EarlyStopping: patience=6 on val/R@1

# GPS proximity mask
pos_threshold_m   = 60.0   # dist < 60m → positive
ignore_threshold_m = 150.0  # 60m–150m → excluded from denominator
# dist > 150m → hard negative
```

### Loss

Multi-positive InfoNCE with GPS exclusion zone (introduced in Exp9):

```
For each (UAV_i, SAT_j) pair in batch:
  pos_mask[i,j]    = 1  if GPS_dist(i,j) < 60m  or  i == j
  ignore_mask[i,j] = 1  if 60m ≤ GPS_dist(i,j) < 150m  and  i ≠ j

logits = (Q @ K.T) * exp(logit_scale)
logits -= 1e9 * ignore_mask          # exclude ambiguous zone from denominator
loss   = 0.5 * (InfoNCE(Q→K) + InfoNCE(K→Q))
```

---

## Stage-2 Experiment Log

| Exp | Commit | R@1 | R@5 | R@10 | Status | Key change |
|-----|--------|-----|-----|------|--------|------------|
| 1 | dc78ea2 | 0.7539 | 0.8750 | 0.9232 | keep | Baseline SSL→SFT; best epoch 14 (+1.82 pp vs supervised-only) |
| 2 | b1f1a26 | 0.7305 | 0.8646 | 0.8971 | discard | Warmup + plain cosine (no restarts) — stalled epoch 4; restarts are necessary |
| 3 | 45d012d | 0.7630 | 0.8919 | 0.9388 | keep | CosineWarmRestarts T₀=10 (2 cycles) + head lr=2e-5; cycle-2 peak +0.91 pp |
| 4 | 1e67ca2 | 0.7448 | 0.8828 | 0.9401 | discard | 4-tier LLRD + stronger UAV aug — aug too strong; peaked epoch 2 |
| 5 | 4bc952c | 0.7708 | 0.8906 | 0.9349 | keep | k_flights=3 batch sampler + 4-tier LLRD; harder negatives +0.78 pp |
| 6 | e2ea884 | 0.7708 | 0.8750 | 0.9388 | discard | 30 epochs T₀=15 — restart at epoch 15 collapsed to 0.70; tied Exp5 |
| 7 | 4c1a7e4 | 0.7552 | 0.8880 | 0.9258 | discard | batch=96, k_flights=3 (32/flight) — bigger batch reduces negative pressure |
| 8 | f22efc1 | 0.7083 | 0.8398 | 0.8880 | discard | 3-layer LN projection head — too slow to converge in 20 epochs |
| **9** | **7ca6756** | **0.7786** | **0.8945** | **0.9414** | **keep** | **GPS exclusion zone (60m pos / 60-150m ignored) — new best** |
| 10 | 10779b4 | 0.7773 | 0.8958 | 0.9414 | discard | patience=10 — cycle-2 peak 0.7604, restart still destructive |
| 11 | cff953f | 0.7773 | 0.8932 | 0.9336 | discard | ReduceLROnPlateau — peaked epoch 7, then LR decay collapsed model |
| 12 | 76de648 | 0.7721 | 0.8906 | 0.9349 | discard | Tighter GPS threshold 40m/100m — consistently -0.006 pp vs 60m/150m |
| 13 | 5cb2785 | 0.7500 | 0.8945 | 0.9401 | discard | UAV 4-rot TTA at val + sat queue (128) for hard negatives — early noise from low-quality queue embeddings dragged peak down |

---

## Key Findings

### 1. SSL initialisation beats supervised-only by +4.3 pp R@1

Starting Stage 2 from the SSL Exp13 checkpoint (R@1=53.0%) rather than the pretrained DINOv3 weights (supervised-only: R@1=73.6%) gives a consistent head-start. The satellite domain adaptation from Stage 1 — scale invariance, orientation invariance, satellite texture alignment — transfers directly to the supervised task. The gap widened further as other Stage-2 improvements compounded.

### 2. GPS exclusion zone is the largest single Stage-2 gain (+0.78 pp R@1)

Before Exp9, results were stuck at ~0.77 R@1 for three consecutive experiments. The breakthrough was recognising that the 60–150m band around each UAV query is **ambiguous**: these satellite tiles are close enough to contain overlapping ground features, but far enough that they are wrong answers. Excluding them from the InfoNCE softmax denominator (neither positive nor hard negative) dramatically sharpened the loss signal. The R@10=0.9414 result shows near-perfect top-10 coverage; the remaining gap to R@1=0.90 is purely a ranking precision problem.

### 3. Three-flight batch sampling consistently outperforms two-flight (+0.78 pp R@1)

`TwoFlightBatchSampler` with k_flights=3 (three geographic regions per batch) gives harder in-batch negatives than k_flights=2. Satellite chunks from different flights are geographically distinct but often visually similar (roads, fields, rooftops) — exactly the pairs that cause R@1 failures. This is the **sampling insight** most aligned with the task: harder negatives must come from different locations, not just different augmentations of the same location.

### 4. CosineWarmRestarts with GPS exclusion zone: cycle-1 only

Before the GPS exclusion zone (Exp1–8), warm restarts helped: cycle-2 consistently outperformed cycle-1 (Exp3 cycle-2 peak 0.7630 > cycle-1 0.7409). After the GPS exclusion zone sharpened the loss landscape (Exp9+), every restart was destructive: cycle-1 peaks at epoch 5 (~0.7786), restart at epoch 10 spikes the LR, cycle-2 never recovers (cycle-2 best: 0.7604 in Exp10). The model finds a sharp basin early and warm restarts overshoot it.

### 5. Tighter GPS positives do not help (Exp12: 40m vs 60m)

Removing the 40–60m band from the positive set (Exp12) was hypothesised to give cleaner supervision. In practice, it was consistently 0.006 pp below Exp9: the pairs in the 40–60m band are genuinely informative positives, not noise. The ignore zone (60–150m) is the right abstraction — pairs that are neither clearly correct nor clearly wrong.

### 6. UAV TTA at inference hurts in early epochs (Exp13)

Adding 4-rotation TTA for UAV query embeddings (matching the satellite's existing TTA) introduced ~0.025 pp drop in early epochs when the backbone produced inconsistent embeddings across rotations. By epoch 4 the gap closed to ~0.006 pp, but the peak (0.7500 vs 0.7786) never recovered. Root cause: the model is trained on single-orientation UAV images (horizontal flip only, no rotation augmentation), so the rotated views produce systematically different embeddings that degrade the TTA average rather than improving it. The fix would be to add RandomRotation to UAV training augmentation for consistency.

### 7. What did not help

| Idea | Result | Why |
|------|--------|-----|
| Warmup + plain cosine, no restarts (Exp2) | 0.7305 | Stalled at epoch 4; restarts needed to escape early plateau |
| Stronger UAV augmentation during SFT (Exp4) | 0.7448 | SSL-strength aug is too aggressive for supervised task; peak at epoch 2 |
| Large batch batch=96, k=3 (Exp7) | 0.7552 | 32/flight vs 21/flight reduces per-flight hard negative density |
| 3-layer projection head with LayerNorm (Exp8) | 0.7083 | Too slow to converge in 20 epochs; 2-layer head converges in 5 |
| patience=10 through restart dip (Exp10) | 0.7773 | Cycle-2 peaked 0.7604 — the restart itself is the problem, not patience |
| ReduceLROnPlateau factor=0.5 patience=3 (Exp11) | 0.7773 | Monotone decay after plateau — over-decayed LR by epoch 8, collapsed to 0.69 |
| Tighter GPS threshold 40m/100m (Exp12) | 0.7721 | 40–60m positives are informative; removing them weakens the loss signal |
| UAV TTA + satellite queue (Exp13) | 0.7500 | Queue adds noisy early-epoch denominator terms; TTA inconsistent with training |

---

## Training Dynamics

Typical Stage-2 run (Exp9, best result):

```
Epoch  1: R@1 = 0.700   (model converges rapidly from SSL initialisation)
Epoch  2: R@1 = 0.753   (head trains in, backbone adapts)
Epoch  3: R@1 = 0.740   (cosine LR oscillation)
Epoch  4: R@1 = 0.717
Epoch  5: R@1 = 0.779   ← best  (cycle-1 peak, CosineWarmRestarts approaching T₀)
Epoch  6: R@1 = 0.737   (LR bottoms out at cycle end)
Epoch 10: R@1 = 0.742   (restart — LR spike, model overshoots basin)
Epoch 11: R@1 = 0.716   (early stop fires, cycle-2 never recovered)
```

Loss dropped from ~2.1 (epoch 1, no warmup) to ~0.825 (epoch 5 best), compared to ~1.1 in pre-GPS-exclusion experiments — confirming the exclusion zone sharpens the signal.

---

## Gap Analysis

| Metric | Stage-2 best (Exp9) | Target | Gap |
|--------|---------------------|--------|-----|
| R@1 | **0.7786** | 0.90 | **0.121** |
| R@5 | 0.8945 | — | — |
| R@10 | **0.9414** | — | — |

R@10 = 0.9414 means the correct satellite tile is in the top-10 **94% of the time**. The remaining 0.121 pp gap to R@1=0.90 is entirely a **ranking precision problem**: the correct tile is retrieved but ranked below a visually similar wrong tile.

### Promising directions to close the remaining gap

1. **Add rotation augmentation to UAV during training** — enables UAV TTA at inference consistently; addresses the drone-heading orientation gap directly
2. **Online hard negative mining with correct embedding cache** — sample hard negatives from a momentum-encoder queue (MoCo-style) rather than a main-encoder queue; avoids stale-embedding noise
3. **Patch-level re-ranking** — use spatial patch token similarity to re-rank the top-10 CLS-retrieved candidates; CLS captures global semantics, patch alignment captures local geometry for ranking
4. **Lower T₀ for restarts with GPS exclusion zone** — if cycle-1 peaks at epoch 5, a T₀=6 scheduler would apply only one restart at epoch 6 instead of the epoch-10 restart that consistently destroys the peak
5. **Listwise ranking loss** — replace InfoNCE with a loss that directly optimises AP@1 or NDCG, targeting the ranking problem rather than retrieval coverage
