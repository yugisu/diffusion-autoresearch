# diffusion-autoresearch

Autonomous research into UAV-to-satellite visual geo-localization using Stable Diffusion v2.1 internal representations.

## Context

**Task**: Given a UAV (drone) image, retrieve the matching satellite map tile — zero-shot, using features extracted from a frozen SD v2.1 UNet.

**Dataset**: VisLoc flight_03 (Taizhou, China). 768 UAV queries, 2860 satellite gallery chunks at 512×512 px.

**Metric**: **Recall@1 (R@1)** — the fraction of queries where the correct satellite chunk ranks first. Higher is better.

**Target**: R@1 ≥ 0.10. Current best from naive SD21 baseline: ~0.004. Best from prior manual experiments: ~0.026 (DiffusionSat, down_blocks, ts≈840/999).

**Prior experiments to read before starting**:
- `/root/diffusion-vpr/results/1-baseline-comparison.csv` — cross-model comparison (DINOv3, DINOv2, RemoteCLIP, DiffusionSat, SD21, etc.)
- `/root/diffusion-vpr/results/2-evaluate-timesteps.csv` — timestep and layer sweep for diffusion features

Read both CSVs at the start of setup to absorb what has already been tried.

## Setup

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `apr16`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from main.
3. **Read the in-scope files** — the repo is small, read all of them:
   - `prepare.py` — fixed constants, dataset loading, and the evaluation function. Do not modify.
   - `train.py` — the file you modify. Feature extraction pipeline and optional lightweight training.
4. **Read prior results**: read both CSV files listed above.
5. **Verify setup**: Run `uv run prepare.py` — it should print dataset sizes (768 UAV, 2860 sat) and confirm SD21 is cached.
6. **Initialize results.tsv**: Create with just the header row. The baseline will be recorded after the first run.
7. **Confirm and go**.

## Experimentation

Each experiment runs on a single GPU. The time budget per experiment is **12 minutes wall-clock** (TIME_BUDGET = 720s, excluding Python startup and eval).

**What you CAN do** (modify `train.py` freely):
- Change which UNet layers to hook (down_blocks, mid_block, up_blocks — any combination)
- Change the DDPM timestep (0–999); prior sweep suggests 800–950 for down_blocks
- Change the aggregation method (GeM pool, average pool, spatial descriptors, PCA, etc.)
- Change the text prompt or remove text conditioning entirely (null text)
- Change the image resolution fed into the UNet (currently IMG_SIZE=256)
- Add PCA whitening or other post-processing to decorrelate features
- Add a **lightweight trainable component** (e.g. linear projection, small MLP) trained on GPS-matched UAV–satellite pairs from the same flight. The UAV GPS coordinates and satellite chunk bboxes are available through the dataset classes in prepare.py.
- Combine features from multiple stages/timesteps
- Use the VAE encoder output directly (skip the UNet entirely)
- Try DDIM inversion instead of forward noising
- Anything else that fits in the time budget

**What you CANNOT do**:
- Modify `prepare.py`. It contains the fixed evaluation and dataset split.
- Install new packages or add dependencies.
- Use external models (DINOv3, CLIP, etc.) — the point is to leverage SD v2.1 representations.

**Key insight from prior work**: The main challenge is the domain gap between oblique UAV imagery (3976×2652 px, variable altitude ~400–600m) and nadir satellite tiles. Pure frozen SD features score near-random (R@1≈0.004). The 25× improvement gap to reach R@1=0.10 likely requires either (a) better feature selection, (b) post-processing to remove domain-invariant noise, or (c) a lightweight trained adapter.

**Ideas to explore** (not exhaustive — be creative):
- Mid-block features (the UNet bottleneck): most compressed, highest-level semantics
- Cross-attention activations (attn2 within transformer_blocks) vs. self-attention (attn1)
- PCA whitening: subtract mean, divide by first N principal components (removes illumination/style)
- Linear probe: train a small cross-domain projector on GPS-paired samples
- Null text: set prompt_embeds to the empty string encoding for purely visual features
- Up-block features at high timestep (prior data shows up_blocks are slightly worse than down_blocks overall, but they operate at higher resolution)
- Combination of down_blocks spatial features (low channel, high resolution) and mid_block (high channel, low resolution)

## Output format

When the script finishes it prints:
```
---
R@1:       0.004000
R@5:       0.015625
R@10:      0.029948
elapsed_s: 174.8
emb_dim:   4480
```

Extract the key metric:
```
grep "^R@1:" run.log
```

If the grep is empty, the run crashed. Read the stack trace:
```
tail -n 50 run.log
```

## Logging results

Log to `results.tsv` (tab-separated — commas break inside descriptions). Do not commit this file.

Header and columns:
```
commit	R@1	R@5	elapsed_s	status	description
```

1. `commit` — short git hash (7 chars)
2. `R@1` — Recall@1 (e.g. 0.026042) — use 0.000000 for crashes
3. `R@5` — Recall@5
4. `elapsed_s` — total wall-clock seconds
5. `status` — `keep`, `discard`, or `crash`
6. `description` — short description of what this experiment tried

Example:
```
commit	R@1	R@5	elapsed_s	status	description
a1b2c3d	0.003906	0.015625	180.2	keep	baseline: down_blocks.attn1-2 ts=880 GeM
b2c3d4e	0.012500	0.041667	210.5	keep	added mid_block hook + null text
c3d4e5f	0.009375	0.031250	195.0	discard	PCA 128d (worse)
d4e5f6g	0.000000	0.000000	0.0	crash	linear probe OOM
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/apr16`).

**First run**: always run the baseline first (train.py as written) to establish the reference point.

LOOP FOREVER:

1. Review git state and results.tsv to understand where you are.
2. Form a hypothesis based on what has and hasn't worked. Consult the prior CSV files to avoid re-running known dead-ends.
3. Edit `train.py` with one focused experimental change.
4. `git commit` the change.
5. `git push origin HEAD` — push immediately so results are visible upstream.
6. `uv run train.py > run.log 2>&1`
7. Extract metrics: `grep "^R@1:\|^R@5:\|^elapsed_s:" run.log`
8. If grep is empty → crash. Read `tail -n 50 run.log`, fix if trivial, otherwise log as crash and discard.
9. Log to `results.tsv`.
10. If R@1 **improved** (strictly higher than current best): keep the commit, stay on it.
11. If R@1 is equal or worse: `git reset --hard HEAD~1 && git push --force origin HEAD` — wipe the bad commit locally and from the remote so the branch history only contains kept experiments. Force-push is safe here because autoresearch branches are never shared.

**Timeout**: If a run exceeds 14 minutes total (720s TIME_BUDGET + ~2 min overhead), kill it and treat as crash.

**NEVER STOP**: Once the loop begins, do NOT pause to ask the human whether to continue. The human may be asleep. Run until manually interrupted.

If you run out of obvious ideas: re-read the prior CSVs, look for near-misses in your own results.tsv, try combinations of things that individually helped a little, or try more radical changes (different aggregation, architectural changes to the hook points, trained adapter).
