# autoresearch

This is an experiment to have the LLM do its own research.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `apr17`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current main/master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, dataset loading, and fixed evaluation. Do not modify.
   - `train.py` — the file you modify. Model, optimizer, training loop.
4. **Verify data exists**: Check that VisLoc data exists at `VISLOC_ROOT` (default from `prepare.py`). If missing, tell the human to run data preparation first.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off experimentation.

## Experimentation

Each experiment runs on a single GPU. Each SSL experiment has a **budget of 2 hours on average, 3 hours at max** (wall clock).

Run command:

```bash
uv run train.py > run.log 2>&1
```

Task details:

- Model: `facebook/dinov3-vitb16-pretrain-lvd1689m`
- Framework: PyTorch Lightning
- Logging: Weights & Biases (`autoresearch-ssl-dinov3`, auth via env/WANDB_API_KEY)
- Training approach: **Self-supervised learning on satellite imagery only** (no UAV images during training)
- Training data: **SSL4EO-S12 S2RGB** patches at `/workspace/data/SSL4EOS12` — 244K global locations × 4 seasonal timestamps, 264×264 px RGB uint8. Loaded via `build_ssl4eo_ssl_pipeline` in `train.py` (uses `train_metadata.parquet` for geographic shard pre-filtering). SSL pairs are drawn from two different seasonal timestamps of the same location: anchor = strong crop+jitter (UAV-like), positive = mild crop (satellite-like).
- Optional geographic filter: `--ssl4eo-lat-min 15 --ssl4eo-lat-max 55 --ssl4eo-lon-min 90 --ssl4eo-lon-max 135` restricts to the China / East Asia region matching VisLoc flights (~21K samples).
- Validation flight: `03` (768 UAV queries, 2860 satellite chunks at default eval stride)
- Primary metric: `R@1` on flight 03, evaluated with fixed `evaluate_r1` in `prepare.py`
- Zero-shot baseline (pretrained DINOv3, no training): **R@1 = 0.3398**
- Previous supervised best (for reference only): R@1 = 0.7357

Satellite scale priors (usable with SatChunkDataset if adding VisLoc chunks to training):

```python
sat_scales = {"01": 0.25,"02": 0.25,"03": 0.25,"04": 0.25,"05": 0.4,"06": 0.6,"08": 0.35,"09": 0.25,"10": 0.5,"11": 0.25,}
```

### Evaluation protocol

- Model is trained using self-supervised approaches on satellite chunks only.
- Backbone adaptation uses **LoRA adapters** (no added projection head). This enables fast screening of which SSL approach works best. Further experiments may try unfreezing last N backbone blocks.
- At evaluation, extract embeddings using `last_hidden_state[:, 0]` (CLS token) from the LoRA-adapted backbone, **768-d, no projection head**.
- Evaluation uses the fixed `evaluate_r1` from `prepare.py` on flight 03: 768 UAV queries against 2860 satellite chunks.
- Satellite TTA (averaging embeddings over 0/90/180/270° rotations) may be used at evaluation.

### Experiment plan (execute in order)

Training experiments are conducted **strictly in the given order**. Step 6 unlocks freedom for exploration based on winners from previous experiments.

All experiments use SSL4EO-S12 S2RGB as the base training corpus unless stated otherwise. The canonical SSL pair is: two different seasonal timestamps of the same geographic location, anchor with strong augmentation, positive with mild augmentation.

1. **InfoNCE baseline (global data)**: InfoNCE with SSL4EO-S12, all 244K global locations, no geographic filter. Seasonal timestamp pairs as positives, in-batch negatives. Default config: batch=128, lr=1e-5, LoRA last-4-blocks, max_epochs=13.
2. **China-region filter**: Restrict training data to lat 15–55 / lon 90–135 (~21K samples matching VisLoc flight regions). Compare against Exp1 — more domain-relevant vs less data. Use `--ssl4eo-lat-min 15 --ssl4eo-lat-max 55 --ssl4eo-lon-min 90 --ssl4eo-lon-max 135`.
3. **GeoRank regularization**: Add `georank_weight=0.1` to the best of Exp1/2. GeoRank ties embedding similarity ranks to geographic distance ranks (Burgert et al., WACV 2025).
4. **VICReg objective**: Swap InfoNCE for VICReg (variance–invariance–covariance regularization). May avoid false-negative issues inherent to in-batch contrastive when seasonal pairs have visually similar negatives.
5. **VisLoc satellite chunks mixed in**: Add satellite chunks from all VisLoc flights (stride=64) alongside SSL4EO-S12 to the training pool. This lets the model learn the exact visual domain of the evaluation region. Not cheating — no UAV images or GPS labels used.
6. **Free experimentation**: Try combinations of what worked from experiments 1-5, tune hyperparameters, try alternative sampling strategies. Explore and exploit.
7. **Canny reconstruction** (optional): Add a Canny edge reconstruction objective — the model predicts a Canny-filtered version of the input image. Goal is to learn edge/structure features that transfer across the UAV-satellite domain gap.
8. **DINO self-distillation** (optional, if R@1 < 0.50 after experiments 1-6): Teacher-student setup. Init both from pretrained DINOv3 weights. Teacher is EMA of student (momentum 0.996→1.0 cosine schedule). Student sees local crops, teacher sees global crops. Standard DINO continued pretraining — no architectural novelty needed. **Budget: 4 hours** (DINO requires multi-crop which is compute-intensive).

### Training notes

- **SSL4EO-S12 data loading**: `build_ssl4eo_ssl_pipeline` in `train.py` reads `train_metadata.parquet` to pre-filter shards by lat/lon, then does a per-sample secondary filter inside each shard (shards have global mixing). Batching is done inside the pipeline (webdataset `.batched()`), so the DataLoader uses `batch_size=None`.
- **Seasonal pairs**: Each SSL4EO-S12 sample has 4 seasonal timestamps (Spring/Summer/Autumn/Winter). Two timestamps are sampled without replacement per forward pass; t1 → anchor (strong aug), t2 → positive (mild aug). This teaches temporal/seasonal invariance on top of scale invariance.
- **Training data stride for VisLoc chunks** (if added): Use stride_pixels=64 for training satellite chunks (4× denser than default). Eval stride stays at the default from prepare.py.
- **Augmentation guidance**: Geometric/scale augmentations (flips, 90° rotations, random resized crop) perform better than color augmentations for satellite SSL. Include slight brightness and contrast jitter (`ColorJitter(brightness=0.3, contrast=0.3, saturation=0, hue=0)`) to simulate weather conditions without altering spectral relationships. Avoid hue/saturation jitter.
- **Logging**: Track total samples seen by the model, log LR, elapsed time × samples, and elapsed time × optimizer step metrics. Attach `run.log` to wandb as an artifact.
- **Initial max_epochs**: 13 (inherited from previous best run). Agent can adjust freely if experiments justify it.

### Research context

These findings from the literature should guide experiment design:

- **Smarter pair construction >> more data or architecture changes.** The biggest wins come from what counts as positive, what's a true hard negative, and how to weight ambiguous pairs (GeoRank, Sample4Geo, Semivariogram reweighting papers).
- **False negative danger**: Geographically close chunks that look similar are likely false negatives, not hard negatives. Naive hard negative mining based on visual similarity alone will incorrectly push apart representations of genuinely related locations (Semivariogram paper, 2025).
- **GeoRank** (Burgert et al., WACV 2025): Rank-based geographic regularization validated with DINO. Adds an MSE loss between embedding similarity ranks and geographic distance ranks. Framework-agnostic, consistent gains.
- **Dataset size**: SSL performance saturates at 100-200k images (GeoRank). DINO-MC matched SeCo (1M images) with only 100k. The ~100k chunks from denser stride should be sufficient.
- **GSD encoding** (Scale-MAE, WaveMAE): Encoding ground sample distance into positional embeddings improves performance when mixing scales. At a single fixed scale, less relevant.

**What you CAN do:**

- Modify `train.py` only. Everything in `train.py` is fair game (loss functions, datasets, samplers, augmentations, optimizer/scheduler, LoRA config, precision, scale tuning, etc.).
- Add LoRA adapters, modify the training loop, create new SSL dataset classes.
- Perform significant exploration in step 6 based on findings from steps 1-5.

**What you CANNOT do:**

- Modify `prepare.py`.
- Modify the fixed evaluation logic in `prepare.py`.
- Install new packages or add dependencies during the loop.
- Use UAV images during SSL training (satellite chunks only).

**The goal is simple: maximize val `R@1` on flight 03 using self-supervised learning on satellite imagery.**

**Simplicity criterion**: all else equal, simpler is better. Keep complexity proportional to gains.

## Output format

At run completion, extract key metrics from log:

```bash
grep "R@1\|R@5\|R@10\|Best checkpoint\|Best val/R@1" run.log
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

## Prior experiments findings & project knowledge

### Zero-shot baseline

Pretrained DINOv3 (no training, CLS token embeddings): R@1=0.3398, R@5=0.5872, R@10=0.6732. This is the baseline all SSL experiments must beat.

### Previous supervised experiments (branch: autoresearch/supervised-dinov3)

A prior supervised training run fine-tuned DINOv3 on UAV-satellite pairs with multi-positive InfoNCE and reached R@1=0.7357. Key findings from 16 supervised experiments:
- **Resolution matters**: 336px input gave +7% R@1 over 224px (preserve satellite texture)
- **Hard negative mining is crucial**: Two-flight batch sampler gave +5% R@1 (force spatial precision)
- **Augmentation helps**: Geometric augmentations + domain-gap targeting gave +3.5% R@1
- **Full backbone fine-tuning with LLRD beats partial freezing**: 3-tier LR (5e-6 / 1e-5 / 2e-5 / 5e-5 for head)
- **What failed**: GeM pooling, asymmetric heads, register token concatenation, RandomResizedCrop on UAV, larger batch sizes (96 > 64 was worse)
- **R@10=0.9167**: The model finds the right area but struggles to rank the exact chunk #1

### Dataset description

**VisLoc**: UAV images are high-quality nadir photos from a Mavic-like drone. Satellite imagery is a large GeoTIFF map split into overlapping chunks. The goal is UAV-to-satellite geo-localization through image retrieval. The domain gap comes from camera quality differences (UAV images have better quality, different colors, slight haze, lower contrast) and potential timing differences (satellite images may be outdated). All VisLoc flights are in China (lat 24–41°, lon 100–121°). Training regions include some bodies of water.

**SSL4EO-S12 v1.1** (`/workspace/data/SSL4EOS12`): Global satellite dataset with 246K locations × 4 seasonal Sentinel-2 timestamps. The S2RGB modality (used here) is 264×264 px RGB uint8. Data is organized as WebDataset TAR shards (`train/S2RGB/ssl4eos12_shard_*.tar`, 477 shards × 512 samples). A `train_metadata.parquet` at the root provides per-sample center_lat, center_lon, tar filename, and cloud_cover per season — used for efficient geographic pre-filtering. Source: https://huggingface.co/datasets/embed2scale/SSL4EO-S12-v1.1

### Environment

The experiments loop is conducted on a powerful machine that has an A100 GPU with 80GB of VRAM - keep that in mind and utilize the GPU accordingly.

## The experiment loop

The experiment runs on a dedicated branch (agreed during setup).

Follow the **experiment plan in order** (steps 1-8 above). Within each step, you may run multiple sub-experiments (e.g. trying different hyperparameters for InfoNCE in step 1). Step 6 unlocks free experimentation.

LOOP:

1. Look at git state: current branch/commit.
2. Implement the next experiment from the plan by modifying `train.py`.
3. git commit with a descriptive message.
4. Run experiment: `uv run train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out results from `run.log`.
6. If metrics are missing, run crashed. Use `tail -n 50 run.log` and attempt fix; if not quickly fixable, log crash and move on.
7. Record results in `results.tsv` with a textual description of the attempt.
8. If `R@1` improved (higher than previous best), advance branch and keep commit.
9. If `R@1` is equal/worse, reset to previous best commit.
10. Move to the next experiment in the plan.

**Timeout**: Each experiment should take ~2 hours on average, 3 hours max. If a run exceeds 3.5 hours, kill it, leave a timeout note in the results, and use the best validation metric logged so far. Exception: experiment 8 (DINO self-distillation) has a 4-hour budget.

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working _indefinitely_ until you are manually stopped. You are autonomous. If you run out of ideas in step 6 (free experimentation), think harder — read related papers on the topic (use arxiv and other sources), re-read the in-scope files for new angles, try combining previous near-misses, try more radical approaches. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. Each SSL experiment takes ~2 hours, so you can run about 4-5 experiments overnight. The user then wakes up to experimental results, all completed by you while they slept!
