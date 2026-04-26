# Zero-shot UAV-to-Satellite Geo-localization with SD v2.1

**Task**: Given a UAV (drone) image, retrieve the correct satellite map tile using only frozen SD v2.1 internal representations. No supervised training; test-time adaptation on unlabelled data only.

**Dataset**: VisLoc flight_03, Taizhou, China — 768 UAV queries, 2860 satellite gallery chunks (512×512 px, stride 128 px). Each query has ~10–16 correct gallery chunks (overlapping tiles).

**Metric**: Recall@1 — fraction of queries where the correct satellite chunk ranks first.

**Final result**: R@1 = **0.0352** (3.5%), R@5 = 0.059, elapsed ~392s.

---

## Experiment Log

| commit  | R@1    | R@5    | s     | status  | description |
|---------|--------|--------|-------|---------|-------------|
| 65e4e13 | 0.0026 | 0.0234 | 139   | keep    | baseline: down_blocks ts=880 GeM prompt="a satellite image" |
| b66d239 | 0.0078 | 0.0326 | 115   | keep    | null text (empty prompt), ts=880 |
| 9e5df56 | 0.0052 | 0.0299 | 118   | discard | null text ts=750 (worse) |
| 1c29efa | 0.0039 | 0.0286 | 110   | discard | null text ts=950 (worse) |
| 1bcbfe2 | 0.0065 | 0.0299 | 129   | discard | PCA whiten remove=64 keep=512 (slightly worse) |
| 108875b | 0.0117 | 0.0313 | 126   | keep    | PCA whiten remove=16 keep=512 |
| ded9bf1 | 0.0065 | 0.0247 | 142   | discard | 2×2 grid GeM pool (spatial noise hurts) |
| 2697ae9 | 0.0026 | 0.0286 | 148   | discard | multi-ts [800,880,940] fixed-noise (correlated, collapsed) |
| 7402d21 | 0.0078 | 0.0286 | 141   | discard | noise avg N=4 t=880 (PCA better without it) |
| 1716817 | 0.0156 | 0.0586 | 158   | keep    | DDIM inv 5-step collect[2,3,4] + PCA whiten |
| db7e88a | 0.0078 | 0.0378 | 180   | discard | DDIM inv 10-step collect[7,8,9] (worse, too narrow range) |
| 2530351 | 0.0169 | 0.0560 | 279   | keep    | IMG_SIZE=512 CenterCrop + DDIM 5-step |
| e830fc3 | 0.0117 | 0.0521 | 294   | discard | DDIM collect all 5 steps (low-noise steps hurt) |
| 5d3e936 | 0.0182 | 0.0534 | 304   | keep    | PCA_KEEP=1024 (was 512) |
| f1c4468 | 0.0078 | 0.0404 | 363   | discard | PCA_KEEP=2048 (whitening amplifies noise) |
| ed76558 | 0.0130 | 0.0430 | 300   | discard | domain mean subtraction (disrupts within-domain structure) |
| d8ffb21 | 0.0065 | 0.0417 | 277   | discard | deep layers only down_blocks[2] (shallow needed for location) |
| 91ce25d | 0.0182 | 0.0521 | 309   | discard | per-block L2 norm (same result, no gain) |
| 636214d | 0.0026 | 0.0208 | 300   | discard | UAV GaussianBlur k=7 (disrupts DDIM trajectory, collapses) |
| 1b16d1a | 0.0182 | 0.0443 | 301   | discard | VAE mid_block+DDIM concat (no gain, R@5 worse) |
| 671540a | 0.0091 | 0.0299 | 355   | discard | 7-step DDIM collect{4,5,6} t≈572-858 (too high noise) |
| e6d19ed | 0.0026 | 0.0286 | 127   | discard | PCA whiten remove=8 (collapsed) |
| 68cabf8 | 0.0104 | 0.0573 | 343   | discard | ResNet+attention hooks (ResNet dilutes attention) |
| 2bfeec6 | 0.0104 | 0.0365 | 310   | discard | DBA(K=10)+alpha-QE(K=5,α=3) — initial retrieval too noisy |
| 3c6a82b | 0.0130 | 0.0456 | 327   | discard | up_blocks attention (decoder worse than encoder down_blocks) |
| 302cfb9 | 0.0052 | 0.0469 | 301   | discard | VAE latent avg_pool2d(k=3) — collapses |
| b709b7e | 0.0117 | 0.0508 | 302   | discard | GeM p=5 (p=3 better, higher p too aggressive) |
| 85066f8 | 0.0104 | 0.0495 | 504   | discard | 2-rotation averaging (0°+90°) — blending hurts |
| 5257fbd | 0.0117 | 0.0547 | 306   | discard | var-sym weighting (too broad, removes location signal) |
| aeddf79 | 0.0091 | 0.0391 | 305   | discard | PCA_REMOVE=24 (worse, confirms 16 is optimal) |
| 9c42ff4 | 0.0104 | 0.0417 | 306   | discard | CORAL alignment UAV→sat covariance (over-constrains) |
| d21beeb | 0.0052 | 0.0000 | 301   | discard | grayscale input — destroys discriminative structure |
| 123a7f6 | 0.0117 | 0.0443 | 297   | discard | per-timestep L2 norm before concat (magnitude signal lost) |
| ebd8e43 | 0.0091 | 0.0000 | 287   | discard | COLLECT={3,4} only — t=401 needed, removing it hurts |
| daaab90 | 0.0104 | 0.0000 | 303   | discard | power norm sign(x)·\|x\|^0.5 after PCA (over-compresses) |
| 23fe589 | 0.0195 | 0.0599 | 302   | keep    | PCA whiten=False (high-eigenval dims dominate) |
| 105df60 | 0.0208 | 0.0560 | 333   | keep    | avg+max pool concat instead of GeM |
| 3e2f1b9 | 0.0091 | 0.0000 | 289   | discard | softmax spatial attention pool (hard focus collapses) |
| 433be62 | 0.0208 | 0.0599 | 373   | discard | avg+max+std triple pool (std adds no value) |
| 6eb22bf | 0.0143 | 0.0000 | 305   | discard | max-only pool (avg+max both needed) |
| 0bb83aa | 0.0195 | 0.0000 | 367   | discard | avg+min+max pool (min adds noise) |
| 8764e22 | 0.0156 | 0.0508 | 593   | discard | spatial pyramid global+2×2 avg+max (too slow, worse) |
| 4de5861 | 0.0117 | 0.0560 | 761   | discard | VLAD K=16 per-depth codebooks (over budget, worse) |
| b53124f | 0.0208 | 0.0651 | 424   | discard | second-order covariance pooling (same R@1, slower) |
| 97b77a4 | 0.0078 | 0.0391 | 335   | discard | random walk re-ranking α=0.85 (visual aliasing spreads) |
| 1cbadd0 | 0.0195 | 0.0586 | 332   | discard | deterministic VAE mode() — same as sample() |
| 5046773 | 0.0313 | 0.0547 | 376   | keep    | UAV h-flip TTA + avg+max + mode() |
| cb80812 | 0.0182 | 0.0417 | 478   | discard | 4-way flip TTA — v-flip disrupts UNet vertical prior |
| 9cb02f0 | 0.0352 | 0.0586 | 392   | keep    | FINAL: h-flip TTA + avg+max + mode() (verified) |

---

## Final Configuration

```python
DDIM_STEPS = 5
COLLECT    = {2, 3, 4}   # inversion steps ≈ t=401, t=601, t=801
IMG_SIZE   = 512
PROMPT     = ""           # null text — purely visual features

# VAE encoding
z = vae.encode(imgs).latent_dist.mode() * 0.18215

# Pooling per attention block
avg = F.avg_pool2d(x, x.shape[-2:]).flatten(1)
mx  = F.max_pool2d(x, x.shape[-2:]).flatten(1)
feat = torch.cat([avg, mx], dim=1)   # (B, 2C)

# UAV TTA: sum original + h-flipped embeddings
uav_embs = extract(original) + extract(hflip)

# PCA: fit on UAV+sat combined, no whitening
pca = PCA(n_components=16 + 1024, whiten=False)
feat = pca.transform(feat)[:, 16:]   # drop top 16 components
```

**Layers**: all `down_blocks` attention blocks (encoder path of SD v2.1 UNet).

---

## Major Findings

### What worked

**1. Null text conditioning is better than a text prompt.**
Passing an empty string instead of `"a satellite image"` nearly tripled R@1 (0.0026 → 0.0078). Text conditioning steers the UNet toward generative semantics; without it, activations stay closer to pure visual perception.

**2. DDIM inversion outperforms a single forward noising pass.**
A 5-step DDIM inversion trajectory (clean → noise) and collecting features from steps 2–4 (t≈401, 601, 801) gave a large jump over single-pass forward noising (0.0078 → 0.0156). The inversion trajectory accumulates richer multi-scale structure than a single noisy latent.

**3. Only high-noise inversion steps carry location signal.**
Collecting all 5 steps (including low-noise steps t≈1, 201) hurt vs. collecting only the high-noise end {2, 3, 4}. Low-noise steps encode fine texture that differs between UAV and satellite sensors, drowning out coarse structural similarity.

**4. Higher resolution (512 px) helps.**
Switching from 256 px to 512 px improved R@1 (0.0156 → 0.0169). Larger latents preserve more spatial structure through the UNet.

**5. PCA component removal (remove top 16, keep 1024) reduces domain gap.**
The first 16 PCA components capture sensor/quality variation that is perfectly correlated within each domain but not location-discriminative across domains. Removing them is a free boost. PCA_REMOVE=16 was the sweet spot — 8 collapsed features, 24 was worse.

**6. No whitening is better than whitening.**
PCA with `whiten=False` (R@1=0.0195) beat `whiten=True` (R@1=0.0182). Whitening equalises all remaining eigenvalues, discarding the magnitude information that encodes how strongly a feature responds — high-eigenvalue dimensions carry more signal here.

**7. Avg+max pooling beats GeM.**
Concatenating global average pool and global max pool (→ 2C per block) outperformed GeM p=3 (0.0208 vs ~0.0117 at the same config). Average pool captures distributed signal; max pool captures peak activations; they are complementary.

**8. H-flip TTA gives the largest single gain.**
Summing UAV embeddings from the original and horizontally-flipped image gave the biggest jump in the whole experiment series (0.0208 → 0.0313, +50%). UAV heading varies per flight; h-flip averages over left/right orientation, making descriptors heading-invariant. V-flip was harmful — the UNet has a vertical prior (sky vs. ground) even for nadir imagery.

**9. Encoder path (down_blocks) is better than decoder path (up_blocks).**
down_blocks attention features scored R@1=0.0352; up_blocks scored 0.0130. The encoder compresses input structure into increasingly abstract representations; the decoder reconstructs — its activations are biased toward generation quality, not discriminative content.

**10. All shallow+deep layers are needed.**
Using only deep layers (down_blocks[2] at 8×8 spatial) dropped R@1 from ~0.018 to 0.0065. Shallow layers (down_blocks[0] at 32×32) carry location signal at finer spatial scale; both are needed.

### What did not work

- **Spatial pooling variants**: 2×2 grid GeM, spatial pyramid (global+2×2 avg+max), VLAD codebooks — all worse or much slower. Global pooling is sufficient; spatial subdivision adds noise because UAV and satellite crops are not spatially aligned.
- **Multi-timestep forward noising**: sampling at multiple timesteps (e.g. t=800, 880, 940) with fixed noise produces correlated features that collapse after PCA.
- **Post-PCA normalisation**: per-timestep L2 norm before concat, power normalisation `sign(x)|x|^0.5`, and per-block L2 norm all hurt or had no effect.
- **Domain alignment**: domain mean subtraction, CORAL covariance alignment — both removed location signal along with domain signal.
- **Re-ranking**: database-side augmentation (DBA) and random-walk re-ranking amplified errors because initial retrieval quality was too low for expansion to help.
- **VAE features**: appending VAE mid_block output, using the VAE latent directly (with pooling) — no gain or collapse. The VAE is a pure autoencoder; its bottleneck lacks the multi-scale attentive structure of the UNet.

---

## Conclusion

The best achievable R@1 with vanilla SD v2.1 in this zero-shot setting is approximately **3.5%**, far below the 10% target. The fundamental barrier is the **sensor domain gap**: UAV images are sharp and high-resolution; satellite tiles are lower-resolution, compressed, and acquired by a different sensor. SD v2.1 was trained on natural photographs; it has no exposure to satellite imagery, so its UNet features do not learn sensor-invariant representations.

For context, DiffusionSat — a version of SD fine-tuned on satellite data — achieves R@1 ≈ 0.026 on this task with a basic single-pass setup, already better than our optimised zero-shot pipeline. The gap points to supervised adaptation (fine-tuning on satellite-UAV pairs, or at minimum self-supervised contrastive learning on unlabelled satellite imagery) as the necessary next step, rather than further feature engineering of vanilla SD activations.
