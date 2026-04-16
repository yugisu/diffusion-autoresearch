# Zero-shot UAV-to-Satellite Geo-localization with DiffusionSat

**Task**: Given a UAV (drone) image, retrieve the correct satellite map tile using only frozen DiffusionSat UNet internal representations. No supervised training; test-time adaptation on unlabelled data only.

**Dataset**: VisLoc flight_03, Taizhou, China — 768 UAV queries, 2860 satellite gallery chunks (512×512 px, stride 128 px). Each query has ~10–16 correct gallery chunks (overlapping tiles).

**Metric**: Recall@1 — fraction of queries where the correct satellite chunk ranks first.

**Model**: DiffusionSat (`finetune_sd21_256_sn-satlas-fmow_snr5_md7norm_bs64_trimmed`) — SD v2.1 fine-tuned on satellite imagery (SatlasPretrain + fMoW) with an optional 7-scalar metadata embedding (GSD, cloud cover, etc.). Metadata is zeroed out during feature extraction.

**Final result**: R@1 = **0.1029** (10.3%), R@5 = 0.189, elapsed ~238s.

---

## Experiment Log

| commit  | R@1    | R@5    | s     | status  | description |
|---------|--------|--------|-------|---------|-------------|
| b451093 | 0.0117 | 0.0430 | 516   | discard | DDIM inv collect{7,8} high-noise — wrong timestep interpretation |
| 25d6f17 | 0.0026 | 0.0130 | 162   | discard | img256 avg-pool — still wrong timestep (near-random) |
| 2207112 | 0.0260 | 0.0443 | 117   | keep    | baseline: COLLECT={0,1} attn1 GeM img256 (matches prior CSV) |
| c71a156 | 0.0352 | 0.0690 | 145   | keep    | h-flip TTA UAV embedding sum |
| 951b8c0 | 0.0716 | 0.1315 | 177   | keep    | PCA remove=16 keep=1024 whiten=False |
| 9a74881 | 0.0664 | 0.1549 | 200   | discard | avg+max pool (GeM better for R@1) |
| 7565eab | 0.1029 | 0.1888 | 238   | keep    | IMG_SIZE=512 batch=8 — **TARGET REACHED** |

---

## Final Configuration

```python
BATCH_SIZE = 8
IMG_SIZE   = 512
DDIM_STEPS = 10
COLLECT    = {0, 1}   # step_idx 0 → t=1, step_idx 1 → t=101 (low-noise)
PROMPT     = "A satellite image"

# Hooks: attn1 (self-attention) inside each Transformer2DModel in down_blocks
# — matches original PoolConcatEmbedder layer_idxs={'down_blocks': {'attn1': 'all'}}
for block in unet.down_blocks:
    for transformer in block.attentions:
        for tblock in transformer.transformer_blocks:
            tblock.attn1.register_forward_hook(...)

# Inversion: only 2 steps needed (early stop after max(COLLECT))
z = vae.encode(imgs).latent_dist.mode() * 0.18215
for step_idx, t in enumerate(inv_timesteps):   # [1, 101, 201, ...]
    noise_pred = unet(z, t, encoder_hidden_states=prompt_embeds)
    if step_idx in COLLECT:
        collect GeM-pooled attn1 features
    if step_idx >= max(COLLECT): break
    z = ddim_inversion_step(z, noise_pred, t, t_next)

# Pooling: GeM p=3
feat = avg_pool(x.clamp(eps).pow(3), spatial).pow(1/3)   # (B, C)

# UAV TTA: sum original + h-flipped embeddings (heading-invariant)
uav_embs = extract(original) + extract(hflip)

# PCA: fit on UAV+sat combined, no whitening, discard top 16 components
pca = PCA(n_components=16 + 1024, whiten=False)
feat = pca.transform(feat)[:, 16:]
```

**Layers**: all `down_blocks` attention blocks (encoder path), `attn1` (self-attention) only.

---

## Critical Finding: Timestep Indexing

The most important discovery in this experiment series was a misreading of the original framework's `save_timesteps` convention.

The SatDiFuser extraction framework (in `ldm_extractor.py`) indexes collected steps using `cur_t = num_timesteps − 1 − i` where `i` is the loop iteration. This means `save_timesteps=[8, 7]` with `num_timesteps=10` does **not** refer to the 8th and 7th timesteps in inversion order (t≈801, t≈701), but to loop iterations `i=1` and `i=2`, which correspond to **t=1 and t=101** — nearly clean latents.

Collecting at high-noise steps (t≈701, 801) produced near-random results (R@1≈0.002–0.011). Collecting at the correct low-noise steps (t=1, t=101) immediately recovered the prior CSV baseline of R@1≈0.026.

This has a clear physical explanation: DiffusionSat was fine-tuned on satellite imagery, so its UNet has learned satellite-specific representations. At low noise (t close to 0), the model's internal features faithfully encode the actual image content in a satellite-adapted feature space. High-noise steps push the latent far from the image manifold, discarding the structure DiffusionSat was trained to represent.

This is the opposite of vanilla SD v2.1 (see `stable-diffusion-zeroshot.md`), where high-noise steps work better because the generic model must suppress sensor-domain differences via noise, while DiffusionSat's domain-adapted features are already useful at low noise.

---

## Major Findings

### What worked

**1. Correct timestep collection: low-noise (t=1, t=101).**
After fixing the timestep indexing bug, the baseline immediately matched the prior CSV result (R@1=0.026). Two UNet forward passes per image suffice — the loop can exit early after collecting both steps, making this the fastest possible inversion regime (~117s total).

**2. H-flip TTA gives a solid gain (+0.009, ×1.35).**
Summing UAV embeddings from the original and horizontally-flipped image made descriptors heading-invariant. UAV flight heading varies; h-flip averages over left/right orientation at no extra inference cost for the satellite gallery.

**3. PCA component removal is the dominant boost (+0.036, ×2.04).**
Fitting PCA on the combined UAV+satellite embeddings and discarding the first 16 components produced the single largest improvement in the series. DiffusionSat was fine-tuned on satellite-only data, so its low-noise features embed strong satellite sensor characteristics in the leading PCA directions. These are domain-discriminative but location-irrelevant — discarding them exposes the residual location signal. The effect is much larger here than in the SD v2.1 experiments (+0.036 vs +0.001), reflecting DiffusionSat's stronger domain imprint.

**4. IMG_SIZE=512 crosses the target (+0.031, ×1.44).**
Increasing input resolution from 256 to 512 px doubled the latent spatial resolution (32×32 → 64×64), providing finer-grained location features. Combined with the earlier improvements this pushed R@1 from 0.072 to 0.103, crossing the 0.10 target.

**5. GeM pooling (p=3) is better than avg+max concat for R@1.**
Avg+max pool hurt R@1 by −0.005 (though R@5 improved by +0.023). GeM's soft aggregation is more robust here, likely because the doubled feature dimension interacts poorly with the fixed PCA_KEEP=1024 budget, diluting the strongest channels.

### What did not work

- **High-noise timestep collection**: collecting at t=701, 801 produced near-random R@1≈0.002–0.011. DiffusionSat's fine-tuning makes low-noise features the informative ones.
- **Avg+max pool**: R@1 dropped from 0.072 to 0.066. The doubled dimension before PCA appears to dilute rather than complement the GeM signal at this PCA budget.

---

## Comparison with SD v2.1

| | SD v2.1 | DiffusionSat |
|---|---|---|
| Best R@1 | 0.035 | **0.103** |
| Effective timesteps | high-noise (t≈401–801) | low-noise (t=1, t=101) |
| Hooks | full Transformer2DModel output | attn1 (self-attention) only |
| Pooling | avg+max concat | GeM p=3 |
| PCA boost | +0.001 | **+0.036** |
| Experiments to target | 50 (target not reached) | 5 (target reached) |

DiffusionSat's satellite fine-tuning provides two compounding advantages: (1) its low-noise features are already in a satellite-adapted space, enabling direct location discrimination without noise-based domain suppression; (2) the strong domain imprint in leading PCA components is precisely what PCA removal eliminates, yielding a much larger relative boost than on vanilla SD v2.1.

---

## Conclusion

DiffusionSat reaches R@1 = **10.3%** on VisLoc flight_03 in a fully zero-shot regime, surpassing the 10% target in 5 effective experiments. The key insight is that fine-tuning on satellite imagery inverts the optimal noise level for feature extraction: DiffusionSat should be queried at nearly clean inputs (t≈1–100), not at high noise. Once that is corrected, standard improvements (TTA, PCA, resolution) transfer directly from the SD v2.1 research, each providing multiplicative gains over a strong baseline.
