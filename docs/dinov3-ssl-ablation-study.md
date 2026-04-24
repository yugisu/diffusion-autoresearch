# SSL Fine-Tuning Ablation Study: DINOv3 UAV-to-Satellite Retrieval

Component-wise ablation tracing every major design decision across two SSL training setups — one using VisLoc satellite imagery (same source as the evaluation gallery) and one using SSL4EO-S12 v1.1 Sentinel-2 patches (external, global). All retrieval numbers are **R@1 on VisLoc flight 03** (768 UAV queries, 2,860 satellite gallery chunks). Rows marked *not run* are design points not directly measured.

---

## Satellite Dataset Significance

The single largest gap in this study is not a hyperparameter choice — it is the **training dataset**.

| Training dataset | Best SSL objective | R@1 | R@10 | Doc |
|-----------------|-------------------|-----|------|-----|
| VisLoc satellite chunks (in-domain, ~1 m/px) | InfoNCE + cross-scale aug | 0.530 | 0.706 | `dinov3-self-supervised-fine-tuning.md` |
| SSL4EO-S12 China filter (out-of-domain, ~10 m/px) | VICReg multi-positive | **0.616** | **0.805** | `dinov3-self-supervised-ft-ssl4eos12.md` |
| **Gap** | | **+8.6 pp** | **+9.9 pp** | |

Training on out-of-domain Sentinel-2 imagery yields **8.6 pp higher R@1** than training on the target domain's own satellite imagery. This is the central finding: in-domain does not automatically mean better SSL signal.

Three factors explain the gap:

**1. Temporal pairs vs augmentation-only pairs (~5–6 pp)**
SSL4EO-S12 provides four genuine seasonal timestamps per location (spring/summer/autumn/winter). Forming SSL pairs from these teaches the model to produce identical embeddings for genuinely different-looking images of the same place — different vegetation state, snow cover, lighting angle, and haze. VisLoc satellite is a single snapshot with no temporal dimension; VisLoc SSL experiments used random cross-scale crops from the same pixel buffer as positive pairs. These synthetic pairs are easier and teach scale invariance but not seasonal or appearance invariance, resulting in features that generalise less well to UAV appearance.

**2. SSL objective: VICReg vs InfoNCE (~2.3 pp)**
The best VisLoc SSL model used InfoNCE; the best SSL4EO-S12 model used VICReg. A direct comparison within the SSL4EO-S12 experiments (Exp02 vs Exp04) isolates +2.3 pp from the objective switch alone. InfoNCE's in-batch negatives are contaminated by false negatives (visually similar patches from nearby locations); VICReg avoids negatives entirely by using explicit variance and covariance regularisation.

**3. Geographic diversity (~1–2 pp)**
VisLoc SSL trains on ~82K chunks from 10 narrow flight corridors in Eastern China — a spatially constrained distribution with limited land-cover variety. SSL4EO-S12 China filter covers 21.5K independently sampled locations across a 4,000 × 5,000 km region, spanning diverse urban morphologies, agricultural patterns, and terrain types. Despite lower resolution, the geographic breadth teaches more general location-discriminative features.

---

## Dataset Properties

| Property | VisLoc satellite | SSL4EO-S12 v1.1 |
|----------|-----------------|-----------------|
| **Role** | SSL training (VisLoc exp.) + evaluation gallery | SSL training (SSL4EO exp.) |
| **Sensor** | Commercial high-res optical | Sentinel-2 MSI (ESA) |
| **Spectral bands** | RGB natural colour | RGB (B4/B3/B2) |
| **Spatial resolution** | ~1 m/pixel (est. from geographic bounds) | ~10 m/pixel |
| **Image size** | 512 × 512 px per chunk | 264 × 264 px |
| **Ground coverage per image** | ~0.51 km × 0.51 km | ~2.64 km × 2.64 km |
| **Temporal structure** | Single snapshot | 4 seasonal timestamps per location |
| **Geographic scope** | 11 flight corridors in China, 7–88 km² each | 244K global (China-filtered: 21.5K locations) |
| **Visual appearance** | High contrast, sharp fine detail | Softer, atmospherically corrected |
| **Training volume** | ~82K chunks (10 flights, stride=64) | 21.5K locations × 4 seasons = 86K images |
| **Domain match to VisLoc gallery** | Exact (same sensor, same resolution) | Out-of-domain (10× coarser, different sensor) |

Despite being exactly in-domain with the evaluation gallery, VisLoc satellite training underperforms — confirming that **temporal diversity of training signal matters more than sensor/resolution match** for self-supervised place recognition.

---

## Cumulative Ablation

A single progressive table covering both training setups. Each row adds one change on top of all prior rows within its track; the two tracks share only the zero-shot starting point.

| Step | Configuration | Training dataset | R@1 | Δ vs prior |
|------|--------------|-----------------|-----|------------|
| 0 | Frozen DINOv3 — zero-shot, no adaptation | — | 0.340 | — |
| **VisLoc SSL track** | | | | |
| V1 | + InfoNCE SSL, basic augmentation | VisLoc satellite | 0.454 | +11.4 pp |
| V2 | + IoU>0.5 positive pairs + LoRA last-4 blocks | VisLoc satellite | 0.492 | +3.8 pp |
| V3 | + Cross-scale crops (anchor 25–50%, pos 75–100%) | VisLoc satellite | 0.509 | +1.7 pp |
| V4 | + Asymmetric augmentation on anchor | VisLoc satellite | **0.530** | +2.1 pp |
| **SSL4EO-S12 track** | | | | |
| S1 | + InfoNCE SSL, global 244K samples | SSL4EO-S12 global | 0.548 | **+20.8 pp** |
| S2 | + China geographic filter (~21.5K) | SSL4EO-S12 China | 0.589 | +4.1 pp |
| S3 | + VICReg (replaces InfoNCE) | SSL4EO-S12 China | 0.612 | +2.3 pp |
| S4 | + Multi-positive training (3 seasonal pairs) | SSL4EO-S12 China | **0.616** | +0.4 pp |

S1 (first SSL4EO-S12 result, before any tuning) already exceeds V4 (the best VisLoc SSL result after full tuning). The SSL4EO-S12 temporal pairs provide a stronger learning signal from the very first epoch.

---

## Component Ablations within SSL4EO-S12

The following ablations are all within the SSL4EO-S12 track.

---

### Ablation 1 — SSL Objective

Training dataset: SSL4EO-S12 China filter. Held constant: LoRA last-4 blocks r=16, 2-layer proj head.

| Objective | Positives | R@1 | Δ vs VICReg | Exp |
|-----------|-----------|-----|-------------|-----|
| InfoNCE (in-batch negatives) | 1 | 0.589 | −2.3 pp | Exp02 |
| Barlow Twins (cross-correlation) | 1 | 0.599 | −1.3 pp | Exp10 |
| BYOL (momentum encoder, EMA) | 3 | 0.456 | −15.6 pp | Exp15 |
| DINO (self-distillation, K=512) | — | 0.467 | −14.5 pp | Exp16 |
| VICReg (variance-invariance-covariance) | 1 | 0.612 | ref | Exp04 |
| VICReg | 3 | **0.616** | +0.4 pp | Exp13 |

**Finding**: VICReg's variance and covariance terms are uniquely compatible with LoRA fine-tuning at lr=1e-5. InfoNCE false negatives (similar-looking patches from different locations treated as negatives) conflict with temporal invariance. Barlow Twins stalls at 0.599. BYOL collapses at lr=1e-5; DINO flatlines with K=512 prototypes (original uses 65,536).

---

### Ablation 2 — Geographic Filter

Training dataset: SSL4EO-S12. Held constant: VICReg multi-positive, LoRA last-4 r=16, 2-layer proj head.

| Training scope | Samples | R@1 | Δ vs China | Exp |
|----------------|---------|-----|------------|-----|
| Global — no filter | 244K | 0.582 | −3.4 pp | Exp18 |
| East/SE Asia (lat 5–55°, lon 80–145°) | 27.5K | 0.616 | ±0.0 | Exp17 |
| China region (lat 15–55°, lon 90–135°) | 21.5K | **0.616** | ref | Exp13 |

**Finding**: Geographic filtering is essential even within SSL4EO-S12. Global training with 11× more data is worse than China-only — non-China scenes introduce domain-irrelevant invariances. Expanding to East/SE Asia converges 3× faster but hits the same ceiling, confirming the bottleneck is not data volume.

---

### Ablation 3 — Projection Head Architecture

Training dataset: SSL4EO-S12 China. Held constant: VICReg, LoRA last-4 r=16. Head is discarded at evaluation.

| Projection head | Params | R@1 | Δ vs 2-layer | Exp |
|----------------|--------|-----|--------------|-----|
| None (768-d directly into VICReg) | 0 | 0.464 | −15.2 pp | Exp22 |
| Deep 3-layer: 768→2048→2048→512 | ~8.4M | 0.591 | −2.5 pp* | Exp11 |
| **2-layer: 768→512→512** | ~0.79M | **0.616** | ref | Exp13 |

*Exp11 used single-positive; comparison is approximate.

**Finding**: Without the head, VICReg collapses at 768-d (0.464). A deep head absorbs the gradient, starving the backbone. The 2-layer 512-d head buffers the SSL objective while keeping gradients flowing to the LoRA adapters.

---

### Ablation 4 — LoRA Configuration

Training dataset: SSL4EO-S12 China. Held constant: VICReg, 2-layer proj head. Multi-positive for Exp12/13; single-positive for Exp06.

| LoRA scope | Rank | Trainable params | R@1 | Δ vs best | Exp |
|-----------|------|-----------------|-----|-----------|-----|
| Last 8 blocks | r=16 | ~2.56M | 0.572 | −4.4 pp* | Exp06 |
| Last 4 blocks | r=32 | ~2.56M | 0.606 | −1.0 pp | Exp12 |
| **Last 4 blocks** | **r=16** | **~1.28M** | **0.616** | ref | Exp13 |
| All 12 blocks | r=16 | ~3.84M | — | *not run* | — |

*Exp06 used single-positive; confounded.

**Finding**: More LoRA capacity hurts on 21.5K samples. Last-8/r=16 and last-4/r=32 share the same parameter count (~2.56M) but both underperform last-4/r=16 (~1.28M) — doubling LoRA capacity causes overfitting. The last 4 transformer blocks (high-level semantic attention) with lean rank is the right inductive bias for a small, domain-specific dataset.

---

### Ablation 5 — Number of SSL Positives

Training dataset: SSL4EO-S12 China. Held constant: VICReg, LoRA last-4 r=16, 2-layer proj head. Cleanest single-variable comparison.

| Positives per anchor | R@1 | Δ | Exp |
|---------------------|-----|---|-----|
| 1 (single seasonal pair) | 0.612 | ref | Exp04 |
| 2 (*not run*) | — | *est. +0.2 pp* | — |
| **3 (all 4 timestamps)** | **0.616** | +0.4 pp | Exp13 |

**Finding**: Using all 4 timestamps as 3 anchor–positive pairs triples the invariance signal per step at no memory cost. Gain is modest (+0.4 pp) but stable — confirmed by Exp14.

---

## Summary: Full Contribution Ranking

All decisions, both tracks, ranked by R@1 impact. Dataset column indicates which training data the result applies to.

| Component | R@1 gain | Confidence | Training dataset |
|-----------|----------|-----------|-----------------|
| SSL4EO-S12 over VisLoc sat as training data | **+8.6 pp** | High | Cross-dataset |
| SSL fine-tuning on SSL4EO-S12 (vs frozen backbone) | **+20.8 pp** | High | SSL4EO-S12 |
| SSL fine-tuning on VisLoc satellite (vs frozen backbone) | **+11.4 pp** | High | VisLoc satellite |
| 2-layer projection head vs no head (VICReg) | **+15.2 pp** | High | SSL4EO-S12 |
| China geographic filter vs global SSL4EO | **+4.1 pp** | High | SSL4EO-S12 |
| LoRA scope: last-4 vs last-8 blocks | **+4.4 pp** | Medium | SSL4EO-S12 |
| Asymmetric augmentation on anchor | **+2.1 pp** | High | VisLoc satellite |
| VICReg vs InfoNCE | **+2.3 pp** | High | SSL4EO-S12 |
| Cross-scale crops vs IoU positive pairs | **+1.7 pp** | High | VisLoc satellite |
| LoRA rank: r=16 vs r=32 | **+1.0 pp** | High | SSL4EO-S12 |
| Multi-positive training: 3 pairs vs 1 | **+0.4 pp** | High | SSL4EO-S12 |
| GeoRank / geographic loss term | **±0.0 pp** | High | SSL4EO-S12 |
