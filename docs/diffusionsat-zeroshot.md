# Zero-shot UAV-to-Satellite Geo-localization with DiffusionSat

**Task**: Given a UAV (drone) image, retrieve the correct satellite map tile using only frozen DiffusionSat UNet internal representations. No supervised training; test-time adaptation on unlabelled data only.

**Dataset**: VisLoc flight_03, Taizhou, China — 768 UAV queries, 2860 satellite gallery chunks (512×512 px, stride 128 px). Each query has ~10–16 correct gallery chunks (overlapping tiles).

**Metric**: Recall@1 — fraction of queries where the correct satellite chunk ranks first.

**Model**: DiffusionSat (`finetune_sd21_256_sn-satlas-fmow_snr5_md7norm_bs64_trimmed`) — SD v2.1 fine-tuned on satellite imagery (SatlasPretrain + fMoW) with an optional 7-scalar metadata embedding (GSD, cloud cover, etc.). Metadata is zeroed out during feature extraction.

**Best result**: R@1 = **0.1107** (11.1%), R@5 = 0.171, elapsed ~270s.

---

## Experiment Log — Phase 1: Reaching 10% Target

| commit  | R@1    | R@5    | s     | status  | description |
|---------|--------|--------|-------|---------|-------------|
| b451093 | 0.0117 | 0.0430 | 516   | discard | DDIM inv collect{7,8} high-noise — wrong timestep interpretation |
| 25d6f17 | 0.0026 | 0.0130 | 162   | discard | img256 avg-pool — still wrong timestep (near-random) |
| 2207112 | 0.0260 | 0.0443 | 117   | keep    | baseline: COLLECT={0,1} attn1 GeM img256 (matches prior CSV) |
| c71a156 | 0.0352 | 0.0690 | 145   | keep    | h-flip TTA UAV embedding sum |
| 951b8c0 | 0.0716 | 0.1315 | 177   | keep    | PCA remove=16 keep=1024 whiten=False |
| 9a74881 | 0.0664 | 0.1549 | 200   | discard | avg+max pool (GeM better for R@1) |
| 7565eab | 0.1029 | 0.1888 | 238   | keep    | IMG_SIZE=512 batch=8 — **10% TARGET REACHED** |
| c8e56be | 0.1107 | 0.1706 | 270   | keep    | COLLECT={0,1,2} add t=201 step **(BEST)** |

---

## Experiment Log — Phase 2: Pushing Toward 30% Target

All experiments below start from commit `c8e56be` (R@1=0.1107). Target was raised to 30% R@1.

| R@1    | R@5    | s     | status  | description |
|--------|--------|-------|---------|-------------|
| 0.0990 | 0.1589 | 260   | discard | null prompt — hurts (model trained WITH "A satellite image") |
| 0.1068 | 0.1680 | 326   | discard | PCA_KEEP=2048 — over-keeps noise dims, 1024 better |
| 0.0430 | 0.0964 | 476   | discard | Canny TTA UAV+sat — model can't interpret edge maps, collapsed |
| 0.0573 | 0.1367 | 289   | discard | mid_block attn1 — bottleneck amplifies sensor diffs at low noise |
| 0.0651 | 0.1289 | 262   | discard | PCA_REMOVE=32 — over-strips signal, 16 is sweet spot |
| 0.1042 | 0.1940 | 346   | discard | 4-rotation TTA (0/90/180/270°) — 90° rotations distort satellite-oriented features |
| 0.0521 | 0.1237 | 891   | discard | SPP 1×1+2×2+4×4 — no spatial correspondence across viewpoints; 21× dims; over budget |
| 0.1094 | 0.1836 | 315   | discard | up_blocks+down_blocks — decoder features redundant at low noise |
| 0.0065 | 0.0169 | 356   | discard | img2img UAV→sat translation (TRANSLATE_STEPS=5) — denoising hallucinates, destroys location |
| 0.1094 | 0.1680 | 272   | discard | noise_pred+x0_pred pooled alongside attn1 — 4-channel latent too low-dim |
| 0.0130 | 0.0417 | 259   | discard | per-modality z-score normalization — breaks joint PCA feature space |
| 0.1003 | 0.1484 | 516   | discard | IMG_SIZE=768 batch=4 — model trained at 256px, 512px is resolution sweet spot |

---

## Final Configuration (Best: R@1 = 0.1107)

```python
BATCH_SIZE = 8
IMG_SIZE   = 512
DDIM_STEPS = 10
COLLECT    = {0, 1, 2}   # step_idx 0→t=1, 1→t=101, 2→t=201
PROMPT     = "A satellite image"

# Hooks: attn1 (self-attention) in all down_blocks Transformer2DModel blocks
for block in unet.down_blocks:
    for transformer in block.attentions:
        for tblock in transformer.transformer_blocks:
            tblock.attn1.register_forward_hook(...)

# DDIM inversion with early stop at max(COLLECT)=2
z = vae.encode(imgs).latent_dist.mode() * 0.18215
for step_idx, t in enumerate(inv_timesteps):   # [1, 101, 201, ...]
    noise_pred = unet(z, t, encoder_hidden_states=prompt_embeds)
    if step_idx in COLLECT:
        collect GeM-pooled attn1 features
    if step_idx >= 2: break
    z = ddim_inversion_step(z, noise_pred, t, t_next)

# Pooling: GeM p=3
feat = avg_pool(x.clamp(eps).pow(3), spatial).pow(1/3)   # (B, C)

# UAV TTA: sum original + h-flipped embeddings
uav_embs = extract(original) + extract(hflip)

# PCA: fit on combined UAV+sat, whiten=False, discard top 16 components
pca = PCA(n_components=16 + 1024, whiten=False)
feat = pca.transform(feat)[:, 16:]
```

---

## Critical Finding: Timestep Indexing

The most important discovery in this experiment series was a misreading of the original framework's `save_timesteps` convention.

The SatDiFuser extraction framework (in `ldm_extractor.py`) indexes collected steps using `cur_t = num_timesteps − 1 − i` where `i` is the loop iteration. This means `save_timesteps=[8, 7]` with `num_timesteps=10` does **not** refer to the 8th and 7th timesteps in inversion order (t≈801, t≈701), but to loop iterations `i=1` and `i=2`, which correspond to **t=1 and t=101** — nearly clean latents.

Collecting at high-noise steps (t≈701, 801) produced near-random results (R@1≈0.002–0.011). Collecting at the correct low-noise steps (t=1, t=101) immediately recovered the prior CSV baseline of R@1≈0.026.

This has a clear physical explanation: DiffusionSat was fine-tuned on satellite imagery, so its UNet has learned satellite-specific representations. At low noise (t close to 0), the model's internal features faithfully encode the actual image content in a satellite-adapted feature space. High-noise steps push the latent far from the image manifold, discarding the structure DiffusionSat was trained to represent.

---

## Phase 1 Major Findings

### What worked

**1. Correct timestep collection: low-noise (t=1, t=101, t=201).**
After fixing the timestep indexing bug, the baseline immediately matched the prior CSV result (R@1=0.026).

**2. H-flip TTA gives a solid gain (+0.009, ×1.35).**
Summing UAV embeddings from the original and horizontally-flipped image made descriptors heading-invariant.

**3. PCA component removal is the dominant boost (+0.036, ×2.04).**
Fitting PCA on the combined UAV+satellite embeddings and discarding the first 16 components produced the single largest improvement. DiffusionSat's low-noise features embed strong satellite sensor characteristics in the leading PCA directions — discarding them exposes the residual location signal.

**4. IMG_SIZE=512 crosses the 10% target (+0.031, ×1.44).**
Increasing input resolution from 256 to 512 px doubled the latent spatial resolution (32×32 → 64×64), providing finer-grained location features.

**5. COLLECT={0,1,2} marginally improves (+0.008).**
Adding the t=201 step provides a small amount of additional signal.

### What did not work (Phase 1)

- **High-noise timestep collection**: near-random (R@1≈0.002–0.011)
- **avg+max pool**: R@1 dropped vs GeM (−0.005)

---

## Phase 2 Findings: Hard Ceiling at ~11%

All creative attempts to push past 0.111 failed. The dominant pattern: the pipeline is already well-tuned for this exact architecture, and modifications either do nothing or destroy the carefully-calibrated joint PCA feature space.

### What failed and why

**PCA_REMOVE=32**: Removing more domain components over-strips signal. 16 is the sweet spot between domain removal and location retention.

**4-rotation TTA (0°/90°/180°/270°)**: DiffusionSat was fine-tuned on north-up satellite imagery. Rotating UAV images by 90°/270° creates orientations the model finds unnatural, distorting its features. H-flip is the only safe rotation because it corresponds to the drone flying in the opposite direction.

**Spatial Pyramid Pooling (1×1 + 2×2 + 4×4)**: Failed for a fundamental geometric reason — spatial bins don't correspond across the UAV oblique / satellite top-down viewpoint gap. The same location appears in completely different spatial positions in the two views. Also 21× more dimensions caused PCA quality degradation and pushed runtime to 890s (over budget).

**up_blocks attn1**: Decoder features are redundant with encoder (down_blocks) features at low noise. The UNet decoder refines spatial details already captured in the encoder.

**img2img UAV→satellite translation (TRANSLATE_STEPS=5)**: Catastrophic failure (R@1=0.007). DDIM denoising from a partially-noisy UAV latent causes DiffusionSat to hallucinate plausible-but-location-agnostic satellite content. The generated satellite-style images have identical generic appearances, collapsing all location discriminability. DDIM is not a cross-domain translator.

**noise_pred + x0_pred as additional features**: The UNet output is a 4-channel latent (B, 4, H, W) — GeM-pooled to (B, 4). These 12 extra values (3 steps × 4 channels) are negligible relative to the thousands of attn1 dimensions, and PCA assigns them near-zero weight.

**Per-modality z-score normalization**: Catastrophic failure (R@1=0.013). Normalizing UAV and satellite features independently per dimension breaks the joint PCA feature space. The joint PCA ensures dimension k is meaningful for both modalities simultaneously — independent rescaling destroys this cross-modal alignment.

**IMG_SIZE=768 (batch=4)**: Worse than 512px (R@1=0.100). The DiffusionSat checkpoint name suggests training at 256px input; 512px is already above-training-resolution, and 768px provides no further benefit. The larger latent (96×96) doesn't improve discriminability.

**mid_block attn1**: Bottleneck amplifies fine-grained sensor differences at low noise, widening the domain gap.

### The hard ceiling diagnosis

The R@5/R@1 ratio (~1.54) reveals the nature of the failure: most wrong predictions are ranked far below top-5, not just slightly off. This is a **discriminability failure**, not a precision/re-ranking problem. The features fundamentally cannot distinguish the correct location across the UAV oblique → satellite top-down viewpoint gap for ~89% of queries.

Bridging this gap requires either supervision (cross-view training labels) or a model explicitly trained for cross-view geo-localization. DiffusionSat's features, while satellite-adapted, are not trained to be viewpoint-invariant across the oblique/top-down domain shift.

---

## Comparison with SD v2.1

| | SD v2.1 | DiffusionSat |
|---|---|---|
| Best R@1 | 0.035 | **0.111** |
| Effective timesteps | high-noise (t≈401–801) | low-noise (t=1–201) |
| Hooks | full Transformer2DModel output | attn1 (self-attention) only |
| Pooling | avg+max concat | GeM p=3 |
| PCA boost | +0.001 | **+0.036** |
| Experiments to 10% target | 50 (not reached) | 5 |
| Hard ceiling | ~3.5% | ~11% |

---

## Conclusion

DiffusionSat reaches R@1 = **11.1%** on VisLoc flight_03 in a fully zero-shot regime, a **3.2× improvement** over vanilla SD v2.1. The key insight is that fine-tuning on satellite imagery inverts the optimal noise level for feature extraction: DiffusionSat should be queried at nearly clean inputs (t≈1–200), not at high noise.

The 30% target was not reached. Phase 2 (12 experiments) found a hard ceiling near 11%: the fundamental UAV oblique → satellite top-down viewpoint gap cannot be bridged with unsupervised post-hoc feature engineering. Every approach that modified what DiffusionSat processes or how it's aggregated either produced no improvement or destroyed the joint PCA alignment. The pipeline is essentially at its zero-shot limit; supervised cross-view training would be required to go significantly beyond 11%.
