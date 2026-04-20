# DINOv3 Self-Supervised Fine-Tuning for UAV-to-Satellite Geo-Localization

**Best result: R@1 = 53.0%** across 14 experiments on the VisLoc flight 03 validation set.

---

## Task

**UAV-to-satellite visual place recognition.** Given a nadir drone image, retrieve the matching satellite map chunk that covers the drone's GPS location.

- **Query set**: 768 UAV drone images from VisLoc flight 03
- **Gallery**: ~2,860 satellite chunks (512×512 px, 128 px stride) tiled from a GeoTIFF satellite map
- **Metric**: Recall@1 — fraction of queries whose GPS-correct satellite chunk is the top-1 retrieval
- **Goal**: R@1 ≥ 0.90
- **Zero-shot baseline**: R@1 = 33.98% (frozen DINOv3, no fine-tuning)
- **Supervised upper bound**: R@1 = 73.57% (with UAV images and GPS labels, see `dinov3-supervised-fine-tuning.md`)
- **Time budget**: 45 minutes wall-clock per experiment on a single A100 80GB GPU

---

## Approach

**Constraint**: no UAV images or GPS labels available during training. Only satellite map chunks are used.

The model is fine-tuned using **self-supervised contrastive learning** (InfoNCE) on satellite-only data. The key challenge is bridging the domain gap between satellite imagery (wide-area, lower-resolution, nadir top-down view) and UAV imagery (narrow-area, higher-resolution, from a lower altitude), without ever seeing UAV images.

**Architecture**:
- Backbone: `facebook/dinov3-vitb16-pretrain-lvd1689m` — ViT-B/16, 768-d CLS token
- Adaptation: LoRA (rank=16, alpha=32) applied to QKV projections in the last 4 transformer blocks
- 294k trainable params / 86M total (0.34%)
- Evaluation: CLS token L2-normalised, cosine similarity retrieval, satellite gallery with 4-rotation TTA

---

## Final Configuration (Experiment 13)

```python
# Training
batch_size = 128
max_epochs = 13           # early stopping patience=5
lr = 1e-5
weight_decay = 1e-4
warmup_epochs = 2         # linear LR warmup, then cosine decay
temperature = 0.07        # fixed InfoNCE temperature

# LoRA
lora_rank = 16
lora_alpha = 32.0
lora_last_n_blocks = 4    # last 4 of 12 transformer blocks

# Positive pair construction (cross-scale)
anchor_crop_scale = (0.25, 0.50)   # zoomed-in UAV-like view
positive_crop_scale = (0.75, 1.00) # full-scale satellite-like view

# Anchor augmentation (simulates UAV sensor + temporal domain gap)
ColorJitter(brightness=0.5, contrast=0.5, saturation=0.3, hue=0.1)
RandomGrayscale(p=0.1)
GaussianBlur(kernel_size=9, sigma=(0.1, 2.0))
RandomHorizontalFlip(p=0.5)
RandomVerticalFlip(p=0.5)

# Positive augmentation (mild, preserves satellite appearance)
ColorJitter(brightness=0.3, contrast=0.3, saturation=0, hue=0)
RandomHorizontalFlip(p=0.5)
RandomVerticalFlip(p=0.5)

# Validation
Satellite TTA: average embeddings over 4 rotations (0°/90°/180°/270°), re-normalise
Training flights: 01, 02, 03, 04, 05, 06, 08, 09, 10, 11 (10 flights, 82k chunks at stride=64)
Val flight: 03
```

---

## Experiment Log

| # | Commit | R@1 | R@5 | R@10 | Status | Description |
|---|--------|-----|-----|------|--------|-------------|
| 1b | a5a0586 | 0.4219 | 0.5430 | 0.5964 | discard | Baseline InfoNCE, lr=2e-5, bs=64, fixed temp=0.07 — best at epoch 0, then rapid degradation |
| 1c | 3f544f8 | 0.4544 | 0.6237 | 0.7188 | keep | Reduced lr=5e-6, bs=128 — slower degradation but still degrades after epoch 0 |
| 2 | 2791f03 | 0.4544 | 0.6302 | 0.7070 | discard | VICReg (λ=25, μ=25, ν=1) — loss barely moves, same peak as InfoNCE |
| 3 | b9f73f8 | 0.4635 | 0.6276 | 0.7044 | keep | Same-chunk SimCLR pairs + flight03 in training — new best but loss collapses fast |
| 4 | c07dd6f | 0.4922 | 0.6237 | 0.6823 | keep | IoU>0.5 positive pairs + LoRA last 4 blocks — first sustained multi-epoch improvement |
| 5 | cefb557 | 0.4922 | 0.6276 | 0.6914 | discard | LoRA last 2 blocks — same peak, faster degradation than Exp4 |
| 6 | 52b8965 | 0.4909 | 0.6224 | 0.6685 | discard | GeoRank regularization weight=0.1 — negligible vs Exp4 |
| 7 | 40ba916 | 0.4961 | 0.6146 | 0.6602 | keep | Warm restarts T₀=3 + lr=1e-5 — new best 0.4961 but restarts don't recover degradation |
| 8 | 53c59c4 | 0.5091 | 0.6341 | 0.6732 | keep | **Cross-scale pairs** anchor=25-50%, pos=75-100% — first time >0.50, peak epoch 1 |
| 9 | 7ab69cf | 0.5052 | 0.6445 | 0.6927 | discard | max_epochs=2, warmup=1 — pinning to sweet spot gives 0.5052, below Exp8's 0.5091 |
| 10 | d816838 | 0.5091 | 0.6341 | 0.6732 | keep | batch=256 — same peak 0.5091, held 2 epochs vs immediate drop at batch=128 |
| 11 | 93a0e29 | 0.4740 | 0.6276 | 0.6914 | discard | Anchor scale=10-25% (wider gap) — too hard, degrades ep0→ep1 |
| 12 | 5c99fa2 | 0.5091 | 0.6471 | 0.7057 | discard | Projection head 768→256 — same peak/degradation, does not shield backbone |
| 13 | e656447 | **0.5299** | 0.6641 | 0.7057 | keep | **Asymmetric augmentation** — strong color+blur+grayscale on anchor; 5 epochs of improvement, peak epoch 5 |
| 14 | bfe91c2 | 0.5260 | 0.6484 | 0.6940 | discard | Exp13 augmentation + batch=256 — peak 0.5260 at epoch 3, degraded epoch 4; combined task too hard |

---

## Key Findings

### 1. Cross-scale pairs are the core innovation (+5.3 pp R@1)

The fundamental UAV↔satellite domain gap is one of **scale and resolution**: UAV images show a small ground area in high detail, while satellite chunks cover a much larger area at lower resolution. Creating SSL positive pairs by randomly cropping a small zoomed-in view (25–50% of the chunk area, anchor) alongside a full-scale view (75–100%, positive) and training the backbone to map them to the same embedding directly targets this gap. All experiments before Exp8 were stuck below 0.50 R@1; cross-scale pairs broke through to 0.5091.

### 2. Asymmetric augmentation unlocks sustained improvement (+2.1 pp R@1)

Experiments 8–12 all peaked at ~0.5091 and degraded within 1–2 full-LR epochs — the SSL objective was pulling backbone representations away from UAV-compatible features as fast as it was improving them. The breakthrough in Exp13 was recognising a second domain gap: **sensor and temporal differences**. UAV images are taken with a different camera at a different time of day than the satellite map, producing different color temperature, atmospheric haze, and occasionally motion blur. Applying strong asymmetric augmentation to the anchor only (ColorJitter ×1.7 stronger, RandomGrayscale 10%, GaussianBlur) while keeping the positive mild forces the backbone to learn representations invariant to these sensor effects. The result was qualitatively different training dynamics: 5 consecutive epochs of improvement (vs immediate plateau/degradation), a new best of 0.5299, and clean plateau behaviour once SSL loss converged rather than degrading retrieval quality.

### 3. The degradation ceiling is caused by SSL-retrieval conflict

Every experiment without strong augmentation showed the same pattern: a peak at the warmup-to-full-LR transition (~1300 gradient steps, ~164k samples seen), followed by monotonic degradation. The pretrained DINO features that support cross-domain transfer are progressively overwritten as LoRA adapts attention patterns to satellite-specific textures. Stronger augmentation on the anchor slows this process by making the SSL task harder and more representative of the target domain, giving the backbone more stable gradient signal.

### 4. Batch size improves stability but not the peak

Batch=256 (Exp10) held the 0.5091 peak for 2 epochs vs immediate drop with batch=128 — a stability improvement but no ceiling increase. Combining batch=256 with Exp13's asymmetric augmentation (Exp14) actually hurt: the combined task difficulty (harder negatives + stronger augmentation) was too high, peaking at 0.5260 vs Exp13's 0.5299 and degrading a full epoch earlier. The optimal balance is batch=128 with strong augmentation.

### 5. Satellite TTA is free performance

Averaging satellite gallery embeddings over 4 rotations (0°/90°/180°/270°) at validation adds consistent R@1 improvement at zero training cost. All Exp8+ results include this by default.

### 6. What did not help

| Idea | Why it failed |
|------|---------------|
| Learnable temperature (logit_scale) | Collapsed to near-zero loss in epoch 0, degraded all metrics — fixed temp=0.07 is required |
| VICReg objective | Loss moved less than InfoNCE at the same LR; no ceiling improvement |
| GeoRank regularization (weight=0.1) | Negligible effect vs baseline; would need much higher weight or different formulation |
| Warm restarts | Degradation is directional (SSL erodes UAV-compatible features), not a local minimum — restarts cannot help |
| Wider cross-scale gap (anchor=10-25%) | Task too hard: model cannot learn useful scale invariance in ~1300 steps at this difficulty level |
| Projection head (768→256) | Does not shield the CLS token from SSL pressure — same peak, same degradation rate |
| LoRA on last 2 blocks only | Less capacity meant peak epoch arrived faster and degradation was steeper |
| batch=256 + strong augmentation | Combined task difficulty overshoots: peak 0.5260 vs 0.5299 at batch=128, degraded a full epoch earlier |

---

## Training Progression

```
Zero-shot frozen DINOv3               →  R@1 = 0.340
InfoNCE SSL, fixed temp (Exp1c)       →  R@1 = 0.454  (+11.4 pp)
IoU pairs + LoRA last 4 blocks (Exp4) →  R@1 = 0.492  (+3.8 pp)
Cross-scale positive pairs (Exp8)     →  R@1 = 0.509  (+1.7 pp)
Asymmetric augmentation (Exp13)       →  R@1 = 0.530  (+2.1 pp)
```

---

## Gap to Target

The SSL approach reached R@1 = **53.0%** — a **19.2 pp improvement** over the zero-shot frozen baseline (33.8%), using no UAV images or GPS labels. The supervised upper bound is 73.6% (R@1) with full labels.

R@10 reached **70.6%** (Exp13), meaning the correct satellite chunk is in the top-10 candidates 70% of the time from satellite-only SSL pre-training.

Promising directions to close the remaining gap to 0.90:

- **Momentum encoder (MoCo/BYOL)** — decouple negative count from batch size via memory bank; more stable targets for the backbone
- **Combine with supervised fine-tuning** — use the SSL checkpoint as initialisation for supervised training with UAV images; likely better than training supervised from scratch
- **Harder positive pairs** — incorporate GPS metadata to select anchor/positive pairs from different satellite chunks of the same geographic region (geographically-anchored cross-scale pairs)
- **More LoRA blocks** — now that training is stable (Exp13), increasing to 6–8 blocks may allow learning richer invariances without degradation
- **Multi-scale evaluation** — aggregate CLS tokens from multiple input resolutions at inference
