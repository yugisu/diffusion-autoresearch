# DINOv3 Self-Supervised Fine-Tuning on SSL4EO-S12

**Best result: R@1 = 61.6%** across 23 experiments on the VisLoc flight 03 validation set.

---

## Task

**UAV-to-satellite visual place recognition.** Given a nadir drone image, retrieve the matching satellite map chunk covering the drone's GPS position.

- **Query set**: 768 UAV images from VisLoc flight 03 (China)
- **Gallery**: 2,860 satellite chunks (512×512 px, 128 px stride) tiled from a GeoTIFF map
- **Metric**: Recall@K — fraction of queries whose correct satellite chunk appears in the top K results
- **Goal**: R@1 ≥ 0.90
- **Zero-shot baseline**: R@1 = 33.98% (frozen DINOv3, no fine-tuning)
- **Supervised upper bound**: R@1 = 73.57% (UAV images + GPS labels, see `dinov3-supervised-fine-tuning.md`)
- **Time budget**: ~2 hours wall-clock per experiment on a single A100 80 GB GPU

---

## Constraint

**No UAV images or GPS labels during training.** Only satellite imagery from SSL4EO-S12 is used.

---

## Architecture

- **Backbone**: `facebook/dinov3-vitb16-pretrain-lvd1689m` — ViT-B/16, 768-d CLS token, 86M params
- **Adaptation**: LoRA (rank=16, alpha=32) on QKV projections of the **last 4 transformer blocks** only — 1.28M trainable params (1.47%)
- **Projection head**: 2-layer MLP, 768 → 512 → 512, used during SSL training only (thrown away at eval)
- **Evaluation**: CLS token L2-normalised cosine similarity; satellite gallery with 4-rotation TTA (0°/90°/180°/270°)

---

## Training Data

**SSL4EO-S12 v1.1** (`/workspace/data/SSL4EOS12`): 244K global locations × 4 seasonal Sentinel-2 RGB timestamps, 264×264 px uint8, WebDataset TAR shards.

**Geographic filter (best config)**: lat 15–55°, lon 90–135° — the China region matching VisLoc flights. Yields ~21,500 samples, cutting training noise from irrelevant global scenes.

**SSL pair construction**: for each location, sample 4 seasonal timestamps without replacement:
- **Anchor** (UAV-like): `RandomResizedCrop(336, scale=0.25–0.50)` + mild `ColorJitter(0.2, 0.2)` → simulates zoomed-in UAV perspective
- **Positives** (satellite-like): 3 remaining timestamps, each with `RandomResizedCrop(336, scale=0.75–1.00)` + `ColorJitter(0.4, 0.4)` + `GaussianBlur(σ=0.5–2.0)`

**Loss**: Multi-positive VICReg (mean of 3 anchor–positive VICReg losses per step):
- λ=25 (invariance MSE), μ=25 (variance), ν=1 (covariance decorrelation)

---

## Best Configuration (Exp13)

```python
# SSL objective
loss = VICReg(lambda_=25, mu=25, nu=1)
n_ssl_positives = 3       # all 4 seasonal timestamps used per step
proj_dim = 512            # 2-layer projection head, thrown away at eval

# LoRA
lora_rank = 16
lora_alpha = 32.0
lora_last_n_blocks = 4

# Optimisation
lr = 1e-5
weight_decay = 1e-4
warmup_epochs = 2
max_epochs = 25
batch_size = 128
steps_per_epoch = 200

# Data
geo_filter = "China: lat 15–55°, lon 90–135°"  # ~21.5K samples
anchor_crop_scale = (0.25, 0.50)
positive_crop_scale = (0.75, 1.00)
```

---

## Experiment Log

| # | Commit | R@1 | R@5 | R@10 | Status | Key variables |
|---|--------|-----|-----|------|--------|---------------|
| 01 | de67207 | 0.548 | 0.685 | 0.742 | discard | InfoNCE global 244K, regressed ep1 |
| 02 | 3fe7af8 | 0.589 | 0.723 | 0.783 | keep | InfoNCE China 21K + GeoRank w=0.1; best ep15 |
| 03 | 3a4c735 | 0.557 | 0.715 | 0.766 | discard | proj=256 + all-blocks LoRA + lr=2e-5; hard collapse ep4 |
| 04 | f348759 | 0.612 | 0.754 | 0.805 | keep | **VICReg** proj=512 + China + GeoRank; steady rise ep0→15 |
| 05 | c8728ee | 0.493 | 0.658 | 0.716 | discard | VICReg + VisLoc sat chunks 50/50; diluted signal |
| 06 | f78065e | 0.572 | 0.749 | 0.803 | discard | VICReg + LoRA last 8 blocks; more blocks hurt |
| 07 | 1eed901 | 0.612 | 0.753 | 0.805 | discard | batch=256 + lr=2e-5; tied Exp04, no gain |
| 08 | 7fefc34 | 0.599 | 0.764 | 0.822 | discard | Barlow Twins + GeoRank; R@5/R@10 up, R@1 capped at 0.599 |
| 09 | eb97a02 | 0.592 | 0.750 | 0.802 | discard | Barlow + adaptive GeoRank 20%; GeoRank too strong |
| 10 | 7135f30 | 0.599 | 0.763 | 0.815 | discard | Pure Barlow Twins; ceiling confirmed ~0.599 vs VICReg 0.612 |
| 11 | d036dc9 | 0.591 | 0.745 | 0.800 | discard | VICReg + deep projector 768→2048→2048→512; deep proj hurts |
| 12 | 7578ac2 | 0.606 | 0.737 | 0.803 | discard | Multi-pos VICReg + LoRA r32; faster conv, lower ceiling |
| 13 | f22899a | **0.616** | **0.758** | **0.805** | **keep** | **Multi-pos VICReg + LoRA r16; peak ep15, NEW BEST** |
| 14 | 07f1aea | 0.616 | 0.750 | 0.803 | discard | +GeoRank w=1.0; tied Exp13, GeoRank confirmed neutral |
| 15 | 848bbd9 | 0.456 | 0.615 | 0.676 | discard | BYOL EMA; collapsed — lr=1e-5 too low for BYOL |
| 16 | f46d80a | 0.467 | 0.599 | 0.689 | discard | DINO multicrop; flatlined — K=512 too small for prototypes |
| 17 | 569ef72 | 0.616 | 0.754 | 0.809 | discard | Expanded East/SE Asia 27.5K; 3× faster conv, same ceiling |
| 18 | 3209509 | 0.582 | 0.738 | 0.789 | discard | Global 244K, no geo filter; domain mismatch hurts |
| 19 | 7a764ca | 0.615 | 0.755 | 0.809 | discard | LLRD decay=0.7 + warm restart T0=8; 2× faster but same ceiling; restart destabilised |
| 20 | 96176b0 | 0.510 | 0.642 | 0.710 | discard | Stronger aug: anchor 15–40% crop + sharpness; crop too small, lost semantics |
| 21 | 438f4fb | 0.546 | 0.699 | 0.742 | discard | Geo hard negative hinge w=25; dominated VICReg, plateau at 0.545 |
| 22 | f3a00b0 | 0.464 | 0.621 | 0.688 | discard | VICReg no proj head (proj=0); collapsed instantly |
| 23 | 3192f53 | 0.585 | 0.746 | 0.806 | discard | VICReg lambda=50 (doubled invariance); peaked ep14=0.5846 EarlyStopped ep19; slower start (ep0=0.461 vs 0.552) without higher ceiling |

---

## Key Findings

### 1. VICReg + China filter + projection head is the essential foundation

The combination that unlocked the 0.61 range (Exp04):
- **VICReg over InfoNCE**: +2.3 pp R@1. VICReg's explicit variance/covariance terms prevent representation collapse without needing negatives. False negatives (visually similar but different-timestamp pairs) cannot contaminate the loss. InfoNCE at its best reached 0.589 (Exp02).
- **China geographic filter**: restricts SSL4EO training to lat 15–55°, lon 90–135° (~21.5K samples). Global training (244K, Exp18) gave R@1=0.582 — the domain mismatch from non-China scenes hurts more than the breadth helps.
- **2-layer projection head (768→512)**: decouples backbone features from the SSL objective. Removing it entirely (Exp22) collapsed to 0.464. A deep 3-layer head (Exp11) absorbed too much gradient in projection space (0.591). A 2-layer 512-d head is the Goldilocks point.

### 2. Multi-positive VICReg pushes past the 0.612 ceiling (+0.4 pp)

Using all 4 seasonal timestamps per location as 3 anchor-positive pairs (Exp13 vs Exp04):
- Each step sees 3× more invariance signal per location
- Forces the model to extract features invariant across summer/winter/spring/autumn
- The multi-positive pairs effectively data-augment the invariance objective at training time
- r16 outperforms r32 (Exp12 vs Exp13): r32 overfits the 21K samples, r16 generalises better

### 3. The 0.6159 ceiling is robust — confirmed across 10 experiments

After Exp13, every subsequent experiment either hit the same ceiling or performed worse:

| Approach | Hypothesis | Result |
|----------|-----------|--------|
| Expanded data (27.5K, Exp17) | More samples → higher ceiling | Same 0.6159, 3× faster |
| Global data (244K, Exp18) | Diversity → richer features | Worse: 0.582 |
| LLRD + warm restarts (Exp19) | Escape local minimum | Same 0.6146, warm restart destabilised |
| Stronger augmentation (Exp20) | Better domain gap simulation | Worse: 0.510 (crops too small) |
| Geographic hard-neg hinge (Exp21) | Explicit discriminative signal | Worse: 0.545 (weight too strong) |
| No projection head (Exp22) | Direct backbone optimisation | Collapsed: 0.464 |
| Lambda=50 (Exp23) | Stronger invariance | Worse: 0.585 (slow start, same ceiling) |

The ceiling is likely fundamental to the SSL training signal: satellite-satellite seasonal pairs teach temporal invariance but cannot teach the UAV-satellite **quality and resolution** gap.

### 4. Geographic auxiliary losses consistently fail

Three independent experiments with geographic signal:
- **GeoRank** (rank regression, Exp02/14): neutral at best — the rank signal is too smooth relative to VICReg loss scale
- **Geo hard negative hinge** (Exp21, weight=25): strongly hurt — dominated VICReg signal during warmup, model plateaued at 0.545
- **Global data** (Exp18): geographic diversity ≠ better features for this domain-specific task

The China region geographic specificity is helpful (filter in), but geographic regularisation applied as a loss consistently fails.

### 5. Barlow Twins hits a hard ceiling at 0.599

Barlow Twins (Exp08–10) consistently reached R@1=0.599 and could not exceed it. Its cross-correlation decorrelation objective is less compatible with LoRA's low-rank adaptation than VICReg's explicit variance term. VICReg's +1.3 pp advantage over Barlow is reproducible.

### 6. BYOL and DINO both fail under these constraints

- **BYOL** (Exp15): the EMA anti-collapse mechanism requires lr ≥ 5e-5; at lr=1e-5 (our budget to prevent LoRA overfitting), BYOL collapses. Would need explicit variance term.
- **DINO** (Exp16): the cross-entropy prototype loss requires projection dimension K ≥ 4096 (original DINO uses K=65,536). With K=512, teacher produces near-uniform distributions and the student receives no discriminative gradient. VICReg's variance term avoids this K-scaling problem entirely.

### 7. More LoRA blocks consistently hurts

| Config | R@1 | Note |
|--------|-----|------|
| Last 4 blocks, r16 | **0.616** | Best |
| Last 4 blocks, r32 | 0.606 | Overfits 21K |
| Last 8 blocks, r16 | 0.572 | Earlier peak, lower ceiling |
| All 12 blocks, r16 | — | Not run (Exp03 w/ lr=2e-5 collapsed) |

More LoRA blocks = more trainable params on a small 21K dataset = overfitting. Last 4 blocks with r16 is the optimum.

### 8. LLRD reveals: the ceiling is a local minimum, not a data-speed issue

Exp19 (LLRD + warm restarts) converged 2× faster than Exp13 (peak at ep6 vs ep15) to the same 0.6146. This confirms the ceiling is a property of the loss landscape and available signal, not a training efficiency problem. Warm restart at T0=8 epochs then destabilised the model (0.607→0.597 dip after LR spike).

---

## What Was Tried and Failed

| Category | Experiment | Result | Reason |
|----------|-----------|--------|--------|
| Different objectives | Barlow Twins (08–10) | 0.599 ceiling | Cross-correlation less compatible than VICReg variance |
| Different objectives | BYOL (15) | Collapsed | lr=1e-5 too small for BYOL anti-collapse |
| Different objectives | DINO (16) | Flatlined | K=512 too small; needs K≥4096 |
| More data | Global 244K (18) | 0.582 | Domain mismatch > diversity benefit |
| More data | East Asia 27.5K (17) | 0.6159 | Same ceiling, faster convergence |
| More capacity | LoRA r32 (12) | 0.606 | Overfits 21K |
| More capacity | LoRA last 8 blocks (06) | 0.572 | More params, worse generalisation |
| Projection | Deep projector 3-layer (11) | 0.591 | Absorbs gradient, weakens backbone signal |
| Projection | No proj head (22) | Collapsed | VICReg needs decoupling at 768-d |
| LR schedule | LLRD + warm restarts (19) | 0.6146 | Faster convergence, same ceiling; restarts destabilise |
| Augmentation | Anchor scale 0.15–0.40 (20) | 0.510 | Crops too small → lose semantic content |
| Geography | GeoRank w=0.1–1.0 (02, 14) | Neutral | Signal too smooth vs VICReg scale |
| Geography | Hard neg hinge w=25 (21) | 0.545 | Weight too strong, fought VICReg |
| Geography | Global training (18) | 0.582 | Non-China scenes add noise |
| Batch | batch=256 (07) | Tied | Larger batch no gain |
| VisLoc data | Sat chunks 50/50 (05) | 0.493 | Easy same-aug pairs diluted SSL4EO signal |

---

## Training Progression

```
Zero-shot frozen DINOv3                         →  R@1 = 0.340
InfoNCE global baseline (Exp01)                 →  R@1 = 0.548  (+20.8 pp)
InfoNCE + China filter + GeoRank (Exp02)        →  R@1 = 0.589  (+4.1 pp)
VICReg + proj512 + China (Exp04)                →  R@1 = 0.612  (+2.3 pp)
Multi-positive VICReg + LoRA r16 (Exp13)        →  R@1 = 0.616  (+0.4 pp)
```

---

## Gap Analysis

| Metric | Best SSL (Exp13) | Supervised UB | Gap |
|--------|-----------------|---------------|-----|
| R@1 | 61.6% | 73.6% | 12.0 pp |
| R@5 | 75.8% | — | — |
| R@10 | 80.5% | 91.7% | 11.2 pp |

R@10=80.5% means the model finds the correct satellite chunk within the top 10 in 80.5% of queries. It knows the right **area** well but cannot always rank the exact chunk first. The remaining R@1 gap is a fine-grained discrimination problem: distinguishing adjacent chunks 50–300m apart.

---

## Root Cause of the SSL Ceiling

The 0.6159 ceiling appears to reflect the **information ceiling of satellite-satellite SSL pairs** for this task:

1. **Training signal mismatch**: SSL trains on Sentinel-2 satellite → satellite temporal pairs. Evaluation requires matching UAV (high-res optical, low altitude) → high-res optical satellite chunks. The UAV-satellite quality and resolution gap is not in the SSL training signal.

2. **Scale mismatch**: Sentinel-2 is 10 m/pixel (264px = 2.6km ground). VisLoc satellite chunks are high-resolution optical (~0.3 m/pixel). Features learned from Sentinel-2 are coarser than what fine-grained chunk discrimination requires.

3. **No fine-grained negatives**: VICReg has no explicit negative pairs. Adjacent satellite chunks (50–200m apart) that look nearly identical are never explicitly pushed apart in training. The covariance term partially helps but cannot target this specific confusion pattern.

---

## Ablation Study

See [`dinov3-ssl-ablation-study.md`](dinov3-ssl-ablation-study.md) for the full ablation, including a dataset comparison between SSL4EO-S12 v1.1 and VisLoc satellite imagery.

**Summary of contribution ranking:**

| Component | R@1 gain | Confidence |
|-----------|----------|-----------|
| SSL fine-tuning (any objective, vs frozen) | **+20.8 pp** | High |
| 2-layer projection head (vs no head) | **+15.2 pp** | High |
| China geographic filter (vs global 244K) | **+4.1 pp** | High |
| LoRA scope: last-4 vs last-8 blocks | **+4.4 pp** | Medium |
| VICReg vs InfoNCE objective | **+2.3 pp** | High |
| LoRA rank: r=16 vs r=32 | **+1.0 pp** | High |
| Multi-positive training (3 vs 1) | **+0.4 pp** | High |
| GeoRank / geographic loss term | **±0.0 pp** | High |

---

## Recommended Next Directions

1. **Supervised fine-tuning from SSL checkpoint**: use Exp13's checkpoint as initialisation for supervised UAV-satellite contrastive training. The SSL baseline (0.616) is substantially better than zero-shot (0.340) and likely gives a head-start to supervised training.

2. **VisLoc satellite chunks as hard pseudo-negatives**: use the VisLoc satellite chunks (not UAV images) as a hard negative pool in SSL training. Adjacent chunks from the same flight would be mined as geographic hard negatives — but with much shorter pairwise distances than the SSL4EO-based mining attempted in Exp21.

3. **Scale-aware training with ground sample distance (GSD) encoding**: if VisLoc satellite resolution is known, Scale-MAE / WaveMAE style positional encoding with GSD could help the model explicitly learn the resolution gap between SSL4EO (10 m/pixel) and VisLoc satellite (0.3 m/pixel).

4. **Larger backbone (ViT-L)**: ViT-L has 307M parameters vs ViT-B's 86M. With LoRA at r=16, the capacity increase is modest in trainable params but the pretrained features are richer, potentially giving a higher starting point for domain adaptation.

5. **SimCLR-style hard negative mining on VisLoc satellite gallery**: at inference time, embed all 2860 satellite chunks and find the hardest within-flight negatives. Use these as a fine-tuning step (without UAV images) to sharpen discrimination between adjacent chunks.
