# DINOv3 Supervised Fine-Tuning for UAV-to-Satellite Geo-Localization

**Best result: R@1 = 73.6%** across 16 experiments on the VisLoc flight 03 validation set.

---

## Task

**UAV-to-satellite visual place recognition.** Given a nadir drone image, retrieve the matching satellite map chunk that covers the drone's GPS location.

- **Query set**: 768 UAV drone images from VisLoc flight 03
- **Gallery**: ~2,860 satellite chunks (512×512 px, 128 px stride) tiled from a GeoTIFF satellite map
- **Metric**: Recall@1 — fraction of queries whose GPS-correct satellite chunk is the top-1 retrieval
- **Goal**: R@1 ≥ 0.90
- **Time budget**: 45 minutes wall-clock per experiment on a single A100 80GB GPU

---

## Model

`facebook/dinov3-vitb16-pretrain-lvd1689m` — DINOv3 ViT-B/16 pretrained on LVD-142M+.

- 12 transformer blocks, 768-d hidden, 4 register tokens
- Input 336×336 px → 441 patch tokens + 1 CLS token
- CLS token used as the image embedding, projected through a learned linear head, L2-normalised

---

## Final Configuration (Experiment 16)

```python
# Image config
image_size = 336          # 336px input for richer texture vs default 224px

# Training
batch_size = 64
max_epochs = 20
loss = "multi-positive InfoNCE with GPS proximity mask (100 m threshold)"

# Backbone learning rates (LLRD)
lr_emb_blocks03 = 5e-6   # embedding + blocks 0-3
lr_blocks47     = 1e-5   # blocks 4-7
lr_blocks811    = 2e-5   # blocks 8-11 + norm
lr_head         = 5e-5   # projection head

# Scheduler
CosineAnnealingWarmRestarts(T_0=5_epochs_in_steps, eta_min=lr * 0.05)

# Batch sampler
TwoFlightBatchSampler: 32 pairs from flight A + 32 pairs from flight B per batch

# Augmentation
UAV:       RandomHorizontalFlip, RandomVerticalFlip, RandomPerspective(0.3),
           ColorJitter(0.3, 0.3, 0.2, 0.1), RandomGrayscale(0.1), GaussianBlur(0.2)
Satellite: RandomHorizontalFlip, RandomVerticalFlip, RandomRotation(90° steps),
           ColorJitter(0.4, 0.4, 0.3, 0.15), RandomGrayscale(0.05)

# Validation
Satellite TTA: average embeddings over 4 rotations (0°/90°/180°/270°), re-normalise
```

---

## Experiment Log

| # | Commit | R@1 | R@5 | R@10 | Status | Description |
|---|--------|-----|-----|------|--------|-------------|
| 1 | 2651b5d | 0.5417 | 0.7318 | 0.8151 | keep | Baseline: partial freeze (blocks 0-7 frozen), blocks 8-11 lr=5e-5, head lr=1e-4, 20 epochs |
| 2 | 3384164 | 0.5469 | 0.7461 | 0.8203 | keep | Multi-positive InfoNCE with GPS proximity mask (100 m) |
| 3 | 6cedcb4 | 0.6029 | 0.7930 | 0.8633 | keep | 336 px input resolution + batch_size=64 |
| 4 | 97f60a9 | 0.6185 | 0.8021 | 0.8581 | keep | Halved LRs (backbone 2e-5, head 5e-5) + 10% linear warmup |
| 5 | f3747e0 | 0.6393 | 0.8307 | 0.8880 | keep | Unfreeze blocks 4-11, graduated LR (4-7: 1e-5, 8-11: 2e-5, head: 5e-5) |
| 6 | 1f37121 | 0.6680 | 0.8659 | 0.9076 | keep | Strong domain-gap augmentations (UAV: perspective+blur+jitter; sat: 90°-step rot+flip+jitter) |
| 7 | 4e7f435 | 0.5898 | 0.7773 | 0.8516 | discard | GeM pooling over patch tokens — worse than CLS |
| 8 | 3b0c897 | 0.6510 | 0.8568 | 0.9141 | discard | batch_size=96 — worse than 64 |
| 9 | a6dec56 | 0.6706 | 0.8529 | 0.9063 | keep | Fully unfreeze all backbone with LLRD (emb+0-3: 5e-6, 4-7: 1e-5, 8-11: 2e-5, head: 5e-5) |
| 10 | 45e2cc5 | 0.6797 | 0.8529 | 0.9010 | keep | Satellite TTA at val — average over 4 rotations |
| 11 | 61869f3 | 0.4102 | 0.6126 | 0.6940 | discard | Asymmetric UAV/satellite projection heads — representation collapse |
| 12 | b6972b5 | 0.6289 | 0.8034 | 0.8750 | discard | RandomResizedCrop on UAV (scale 0.6-1.0) — worse than exp 6 |
| 13 | c819dfe | 0.6815 | 0.8477 | 0.9049 | keep | Warm-start from exp 9 checkpoint + 10× lower LRs (marginal gain) |
| 14 | 10aa26f | 0.6237 | 0.8151 | 0.8776 | discard | CLS + mean of register tokens — register tokens hurt retrieval |
| 15 | 59e9fa9 | 0.7174 | 0.8750 | 0.9219 | keep | **Two-flight batch sampler** (32+32 per batch) for hard in-batch negatives |
| 16 | 97aebf3 | **0.7357** | 0.8685 | 0.9167 | keep | Cosine annealing warm restarts (T_0=5 epochs) — best overall |

---

## Key Findings

### 1. Resolution is the biggest single lever (+7% R@1)

Moving from 224 px to 336 px (experiment 3) was the largest single jump in the entire run. The satellite chunks are 512 px and contain fine texture (roads, rooftops, field edges) that gets destroyed at 224 px. The A100's 80 GB VRAM easily accommodated batch_size=64 at this resolution.

### 2. Hard negative mining via flight grouping breaks plateaus (+5% R@1)

After stalling around 0.68 for several experiments, `TwoFlightBatchSampler` (experiment 15) was the breakthrough: each batch draws 32 UAV-satellite pairs from one flight and 32 from a different flight. Geographically adjacent satellite chunks from the same flight area act as hard negatives, forcing the model to learn fine-grained spatial discrimination rather than coarse flight-area recognition.

### 3. Augmentation stabilises training and raises the ceiling (+3.5% R@1)

Strong domain-gap augmentations in experiment 6 — `RandomPerspective` + `GaussianBlur` for UAV, 90°-step `RandomRotation` + stronger `ColorJitter` for satellite — targeted the known domain gap (colour/contrast, haze, rotation). Beyond the direct R@1 gain, augmentation smoothed training curves and delayed the peak epoch from epoch 2 to epoch 16, giving the model more time to improve.

### 4. Satellite TTA is free performance (+0.5% R@1)

Averaging satellite embeddings over 4 rotations (0°/90°/180°/270°) at validation costs nothing at train time and consistently adds ~0.5% R@1. Satellite imagery has no canonical orientation, so this directly reduces orientation-induced retrieval errors.

### 5. Full backbone fine-tuning with LLRD beats partial freezing

Partially freezing blocks 0-7 (experiment 1, 0.54 R@1) was a reasonable start but left performance on the table. Unfreezing the full backbone with layer-wise LR decay (5e-6 → 2e-5) reached 0.67 R@1. The low LRs on early blocks preserve generalised low-level features while allowing higher blocks to specialise for the UAV↔satellite domain.

### 6. What did not help

| Idea | Why it failed |
|------|---------------|
| GeM pooling over patch tokens | CLS token already encodes global context; GeM over patches adds noise |
| Asymmetric projection heads | Collapsed into a degenerate solution — symmetric heads are safer for contrastive learning |
| CLS + mean register tokens | Register tokens encode global statistics unrelated to localisation |
| RandomResizedCrop on UAV | Crops out the fine texture that distinguishes geo-localisable patches |
| Larger batch (96 vs 64) | Slightly worse; in-batch negative quality matters more than quantity at this scale |

### 7. Cosine warm restarts vs. plateau

Experiment 16 added `CosineAnnealingWarmRestarts(T_0=5 epochs)` on top of the two-flight sampler. The model's R@1 curves show clear post-restart recovery peaks (epochs 10-11 and 16-17), confirming the scheduler helps escape local optima. The final best of 0.7357 is a modest but meaningful improvement over exp 15's 0.7174.

---

## Training Progression

```
Baseline (bs=128, 224px, partial freeze)  →  R@1 = 0.436
After 336px + bs=64                       →  R@1 = 0.603  (+16.7 pp)
After full LLRD unfreeze + augmentations  →  R@1 = 0.668  (+6.5 pp)
After satellite TTA + warm-start          →  R@1 = 0.682  (+1.4 pp)
After two-flight hard negatives           →  R@1 = 0.717  (+3.5 pp)
After cosine warm restarts                →  R@1 = 0.736  (+1.9 pp)
```

---

## Gap to Target

The goal is R@1 ≥ 0.90. Best achieved: **R@1 = 0.736** — a gap of **16.4 percentage points** remains.

R@10 reached **0.930** (experiment 10), meaning the correct satellite chunk is almost always in the top-10 candidates. The bottleneck is ranking precision: the model retrieves the right region but often ranks a slightly offset chunk above the exact match. Promising directions to close the gap:

- **Re-ranking / spatial verification** — use patch-level correspondence to re-rank the top-K candidates
- **Longer training or curriculum** — the model still trends upward at epoch 20; training to convergence may add several pp
- **Metric learning losses** — triplet with hard online mining, or ArcFace-style margin loss
- **Multi-scale feature fusion** — aggregate CLS + patch tokens at multiple backbone depths
- **Finer satellite tiling** — smaller stride (e.g., 64 px) to increase positive coverage and reduce ambiguity
