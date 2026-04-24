# DINOv3 Two-Stage Fine-Tuning — SSL4EO-S12 Checkpoint

**Best result: R@1 = 85.8%** (Stage-2 experiment 9: SmoothAP + patch re-ranking) on the VisLoc flight 03 validation set, using a two-stage SSL → supervised fine-tuning pipeline with a stronger SSL checkpoint than the reference branch.

---

## Task

**UAV-to-satellite visual place recognition.** Given a nadir drone image, retrieve the matching satellite map chunk that covers the drone's GPS location.

- **Query set**: 768 UAV drone images from VisLoc flight 03
- **Gallery**: ~2,860 satellite chunks (512×512 px, 128 px stride) tiled from a GeoTIFF satellite map
- **Metric**: Recall@1, Recall@5, Recall@10 on flight 03
- **Goal**: R@1 ≥ 0.90
- **Time budget**: 45 minutes wall-clock per Stage-2 experiment on a single A100 80GB GPU

---

## Overview

This branch uses a stronger Stage-1 SSL checkpoint trained on SSL4EO-S12 global satellite data (R@1=0.615) vs the reference branch's checkpoint (R@1=0.530). Stage 2 explores improving beyond the validated Exp9 baseline configuration from the reference branch.

```
Stage 1 (train.py)              Stage 2 (st2.py)
────────────────────────        ────────────────────────────────────
Multi-positive VICReg           Supervised: UAV ↔ satellite pairs
on SSL4EO-S12 global data         ↓
  ↓                             SmoothAP loss
Cross-scale pairs               with GPS exclusion zone
  ↓                               ↓
LoRA r=16 (4 blocks)            Full backbone LLRD fine-tune
  ↓                               ↓
R@1 = 61.5%                     R@1 = 82.6% (CLS only)
                                R@1 = 85.8% (+ patch re-ranking)
```

**Why this SSL checkpoint is better:** Stage-1 used multi-positive VICReg with all 4 seasonal variants as positives (n_ssl_positives=3) and LoRA r=16, trained on 15–55°N, 90–135°E (China/East Asia). VICReg avoids false-negative collapse when seasonal pairs look visually similar, giving +8.5 pp R@1 over the reference branch's SSL checkpoint.

---

## Baselines

| System | R@1 | R@5 | R@10 |
|--------|-----|-----|------|
| Zero-shot DINOv3 (no training) | 0.340 | 0.587 | 0.673 |
| SSL only — Stage-1 SSL4EO-S12 best (this branch) | 0.615 | — | — |
| SSL only — Stage-1 reference branch | 0.530 | — | — |
| Supervised only — Exp16 (from pretrained DINOv3) | 0.736 | 0.869 | 0.917 |
| Reference two-stage best (SSL R@1=0.530 → st2) | 0.779 | 0.895 | 0.941 |
| **This branch: SmoothAP + patch re-ranking** | **0.858** | **0.947** | **0.970** |

The stronger SSL checkpoint (+8.5 pp vs reference) translates directly to a higher Stage-2 starting point: exp1 baseline (0.793) beats the reference branch's st2-exp1 (0.754).

---

## Architecture

**Backbone**: `facebook/dinov3-vitb16-pretrain-lvd1689m`
- ViT-B/16, 12 transformer blocks, 768-d hidden, CLS token as embedding
- 86.6 M total parameters; all images at **336×336 px** (441 patch tokens per image)

**Stage 1 checkpoint** (fixed for all Stage-2 experiments):
- `checkpoints/dinov3-ssl4eos12-best-r@1=0.615-mvicreg-569ef72.ckpt`
- LoRA r=16, alpha=32, last 4 blocks — merged into weights before Stage 2

**Stage 2 head** (trained from scratch):
- 2-layer MLP: Linear(768, 768) → GELU → Dropout(0.1) → Linear(768, 512) → L2-norm
- Learnable `logit_scale` (used in InfoNCE experiments; vestigial in SmoothAP)

---

## Stage 2: Validated Baseline Configuration

Inherited from reference branch Exp9; all of the following are fixed across all experiments:

```python
image_size        = 336        # px
batch_size        = 64
max_epochs        = 25         # extended from 20 after exp6 showed single-cycle benefits
precision         = "16-mixed"
gradient_clip_val = 1.0
weight_decay      = 1e-4

# UAV augmentation (training only)
RandomHorizontalFlip(p=0.5)
RandomRotation(degrees=180)          # added in exp2; enables UAV TTA
RandomPerspective(distortion_scale=0.2, p=0.5)
ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1)
GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))

# Satellite augmentation (training only)
RandomHorizontalFlip(p=0.5)
RandomVerticalFlip(p=0.5)
RandomRotation(degrees=180)
ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1)

# Batch sampling
TwoFlightBatchSampler: k_flights=3, batch_size=64

# Backbone LLRD (4-tier)
lr_emb_blocks03 = 5e-6
lr_blocks47     = 1e-5
lr_blocks89     = 1.5e-5
lr_blocks1011   = 2e-5
lr_head         = 3e-5    # increased from 2e-5 in exp7

# Scheduler — CosineAnnealingWarmRestarts, step-level
T_0     = 20 * steps_per_epoch  # single 20-epoch cosine descent (no restart)
eta_min = 2e-5 * 0.05

# Validation TTA
UAV:       4-rotation average (0°/90°/180°/270°), re-normalise
Satellite: 4-rotation average (0°/90°/180°/270°), re-normalise

# GPS proximity mask
pos_threshold_m    = 60.0
ignore_threshold_m = 150.0

EarlyStopping: patience=6, monitor=val/R@1
```

---

## Loss Functions

### Multi-positive InfoNCE (exp1–7)

```
logits = (Q @ K.T) * exp(logit_scale)
logits -= 1e9 * ignore_mask             # mask 60–150m ring
loss = 0.5 * (InfoNCE(Q→K) + InfoNCE(K→Q))
```

### SmoothAP (exp8–9)

Directly optimises Average Precision via sigmoid-based soft rank:

```
sims = Q @ K.T                          # cosine similarity, float32
sims -= 1e4 * ignore_mask              # crush ignore zone before rank estimation
diff[i, p, j] = sims[i, j] - sims[i, p]
smooth_rank(i, p) = 1 + Σ_j σ(diff[i,p,j] / τ)   # τ = 0.01
AP(i) = mean_p( 1 / smooth_rank(i,p) )             # over positives p
loss = −0.5 * (mean(AP_Q→K) + mean(AP_K→Q))
```

SmoothAP converges more slowly (peaks at epoch 12 vs epoch 7–8 for InfoNCE) but achieves a significantly higher ceiling and consistently better R@10.

### Patch Re-ranking at Inference (exp9, zero training cost)

Applied on top of any trained checkpoint. After CLS-based retrieval:

```
For each UAV query i:
  top_K = argsort(-CLS_sims[i])[:50]
  For each candidate j in top_K:
    uav_patches = backbone_patch_tokens(uav_i)       # [441, 768], L2-normed, 0° rotation
    sat_patches = backbone_patch_tokens(sat_j)       # [441, 768], L2-normed, 0° rotation
    sim_mat = uav_patches @ sat_patches.T            # [441, 441]
    chamfer = mean_p( max_j sim_mat[p, j] )         # mean-of-max per UAV patch
    final_score[i, j] = 0.5 * CLS_sim + 0.5 * chamfer
Re-rank top-50 by final_score; evaluate R@1 on re-ranked predictions.
```

DINOv3 patch tokens encode local spatial texture that CLS averages away. Chamfer similarity between patch sequences resolves rank-1 ambiguity for geographically close but visually distinct tiles.

---

## Experiment Log

| Exp | Commit | R@1 | R@5 | R@10 | Status | Key change |
|-----|--------|-----|-----|------|--------|------------|
| 1 | 3ebb29a | 0.7930 | 0.9076 | 0.9453 | keep | Baseline: reference Exp9 config + SSL4EO-S12 ckpt (R@1=0.615) |
| 2 | 360001a | 0.8047 | 0.9167 | 0.9596 | keep | UAV RandomRotation(180) aug + UAV 4-rotation TTA at inference (+0.012) |
| 3 | 134f04e | 0.7891 | 0.9049 | 0.9531 | discard | T₀=6 + patience=3: too short, peaked epoch 2, stalled |
| 4 | beb320a | 0.7214 | 0.8724 | 0.9297 | discard | MoCo momentum encoder + 1024-entry queue: oversized denominator hurt convergence |
| 5 | 95b86e4 | 0.4414 | 0.7083 | 0.8086 | discard | Supervised VICReg: no negatives → no cross-modal discriminative signal |
| 6 | 1e1bb76 | 0.8086 | 0.9180 | 0.9518 | keep | T₀=20 single cosine cycle: avoids destructive restart at epoch 10 (+0.004) |
| 7 | 4ac127e | 0.8112 | 0.9128 | 0.9583 | keep | Head LR 2e-5 → 3e-5: faster ramp, peaked epoch 8 (+0.003) |
| **8** | **388d67f** | **0.8255** | **0.9284** | **0.9661** | **keep** | **SmoothAP τ=0.01: peaked epoch 12, R@10 record (+0.014 vs exp7)** |
| **9** | **448c62e** | **0.8581** | **0.9466** | **0.9701** | **keep** | **Patch token re-ranking K=50 α=0.5: +0.033 at zero training cost** |

---

## Key Findings

### 1. SmoothAP outperforms InfoNCE by +0.014 pp R@1

Replacing multi-positive InfoNCE with SmoothAP (τ=0.01) produced the largest single training change gain in this experiment series. The key difference is convergence arc: InfoNCE peaks at epoch 7–8 and degrades; SmoothAP keeps improving through epoch 12. This suggests SmoothAP's direct AP optimisation allows the model to escape sharper loss basins that InfoNCE converges to prematurely.

SmoothAP also consistently achieved higher R@10 (0.9661 vs 0.9583 at best epoch), meaning it learned to retrieve the correct tile in the top-10 more reliably — reflecting its broader AP optimisation objective vs InfoNCE's rank-1 softmax pressure.

The τ=0.01 hyperparameter makes the rank approximation nearly binary (sigmoid is close to a step function). Warmer τ (0.1) would give smoother gradients but less precise rank estimation; τ=0.01 worked well here given a moderate-quality SSL initialisation.

### 2. Patch token re-ranking adds +0.033 pp R@1 at zero training cost

The chamfer similarity between DINOv3 backbone patch tokens (441 patches at 336px input) re-ranks the top-50 CLS candidates at inference time. This is the largest single-step improvement in the series.

The intuition: CLS embeds global scene semantics averaged over all patches; it confuses tiles with similar overall appearance (same land cover, same time of day). Patch tokens retain local spatial structure — road intersections, building outlines, tree arrangements — that uniquely identify a specific location. Chamfer similarity finds the best-matching spatial correspondences between UAV patch tokens and each satellite candidate, which resolves the rank-1 ambiguity for otherwise-similar tiles.

The 0.5/0.5 weighting between CLS and chamfer scores was not tuned; different α values are worth exploring.

### 3. Stronger SSL initialisation raises the Stage-2 ceiling

The SSL4EO-S12 checkpoint (R@1=0.615) consistently outperforms the reference branch's checkpoint (R@1=0.530) at every stage:
- Exp1 baseline: 0.793 vs 0.754 (+3.9 pp)
- Peak InfoNCE: 0.8112 vs 0.7786 (+3.3 pp)
- The gap narrows slightly as Stage-2 training dominates, but the SSL head-start is consistently additive

Multi-positive VICReg's advantage over InfoNCE for SSL is that it avoids false-negative collapse: satellite seasonal variants look visually similar but InfoNCE treats them as hard negatives, while VICReg's variance/covariance terms provide collapse-free learning.

### 4. Single cosine cycle (T₀=20) beats CosineWarmRestarts (T₀=10)

Reference branch dynamics carried over: every LR restart was destructive. With the GPS exclusion zone sharpening the loss, the model finds a sharp basin by epoch 5–7 and a warm restart at epoch 10 consistently overshoots it. Using T₀=20 (one smooth cosine descent over the full budget) allowed the model to continue refining past epoch 10 without disruption. SmoothAP benefited most from this — it peaked at epoch 12, which would have been after two destructive restarts with T₀=10.

### 5. What did not help

| Idea | Result | Why |
|------|--------|-----|
| T₀=6 short schedule (exp3) | 0.7891 | Too aggressive — peaked epoch 2, never built momentum |
| MoCo momentum encoder + queue (exp4) | 0.7214 | 1024-entry denominator overwhelmed the positive signal; needed 3× more epochs to converge |
| Supervised VICReg (exp5) | 0.4414 | VICReg without negatives cannot create cross-modal discriminative structure; variance term spreads embeddings globally but doesn't separate modalities |

---

## Training Dynamics

### InfoNCE (exp7, best InfoNCE result)

```
Epoch  0: R@1 = 0.711   (rapid convergence from SSL init)
Epoch  1: R@1 = 0.755
Epoch  3: R@1 = 0.788
Epoch  8: R@1 = 0.811   ← best (single cosine cycle)
Epoch 14: R@1 = 0.798   (EarlyStopping fires, patience=6)
```

### SmoothAP (exp8, best training result)

```
Epoch  0: R@1 = 0.703   (slower start — SmoothAP gradients sparser early)
Epoch  2: R@1 = 0.790   (rapid recovery)
Epoch  7: R@1 = 0.813   (InfoNCE would plateau here)
Epoch 12: R@1 = 0.826   ← best
Epoch 18: R@1 = 0.819   (EarlyStopping fires, patience=6)
```

SmoothAP loss values (negative AP): epoch 0 ≈ −0.40, epoch 12 ≈ −0.65 (AP improved from 0.40 to 0.65 over training).

---

## Gap Analysis

| Method | R@1 | R@5 | R@10 | Gap to 0.90 |
|--------|-----|-----|------|-------------|
| Exp1 baseline | 0.793 | 0.908 | 0.945 | 0.107 |
| + UAV rotation TTA (exp2) | 0.805 | 0.917 | 0.960 | 0.095 |
| + single cosine cycle (exp6) | 0.809 | 0.918 | 0.952 | 0.091 |
| + head LR 3e-5 (exp7) | 0.811 | 0.913 | 0.958 | 0.089 |
| + SmoothAP loss (exp8) | 0.826 | 0.928 | 0.966 | 0.074 |
| **+ patch re-ranking (exp9)** | **0.858** | **0.947** | **0.970** | **0.042** |

R@10=0.970 means the correct tile is retrieved in the top-10 **97% of the time**. The remaining 0.042 pp gap to R@1=0.90 is a **pure ranking precision problem**.

### Promising directions to close the remaining gap

1. **Distance-weighted SmoothAP positives** — replace binary pos_mask with Gaussian weights (σ=60m): tiles 5m away weight≈1.0, tiles 55m away weight≈0.6. Rewards placing the closest tile at rank 1 specifically, not just any tile within 60m. Expected gain: +0.01–0.02 pp R@1.

2. **Tune patch re-ranking hyperparameters** — K=50 and α=0.5 were not tuned. Larger K (top-100) and higher patch weight (α=0.7) may help; smaller τ_rank for SmoothAP patch scoring is also an option.

3. **Cross-attention re-ranker** — replace the hand-crafted chamfer similarity with a learned cross-attention module over UAV and satellite patch tokens, fine-tuned on hard pairs from the training set. More expressive but requires additional training.

4. **Multi-scale patch tokens** — aggregate patch tokens from multiple ViT layers (e.g., blocks 8, 10, 12) for re-ranking. Lower layers capture texture, higher layers capture semantics; a multi-scale chamfer may be more robust.

5. **Harder negative mining during SmoothAP training** — sample negatives from within the same geographic region (k_flights=4, or add a small GPS-filtered embedding queue) to push the SmoothAP loss to penalise confusable nearby tiles more aggressively.
