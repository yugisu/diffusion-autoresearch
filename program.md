# diffusion-autoresearch

Autonomous research into UAV-to-satellite visual geo-localization using Stable Diffusion v2.1 internal representations.

## Context

**Task**: Given a UAV (drone) image, retrieve the matching satellite map tile using features from a frozen SD v2.1 UNet. No supervised training with labels is allowed — only test-time adaptation on unlabelled data.

**Dataset**: VisLoc flight_03 (Taizhou, China). 768 UAV queries, 2860 satellite gallery chunks at 512×512 px.

**Metric**: **Recall@1 (R@1)** — the fraction of queries where the correct satellite chunk ranks first. Higher is better.

**Ground truth**: A satellite chunk is *correct* for a UAV image if the UAV's GPS point falls inside the chunk's bounding box. Because chunks overlap (stride=128 < size=512), each query typically has ~10–16 correct gallery chunks. The baseline output prints `avg_gt_chunks` to confirm this.

**Target**: R@1 ≥ 0.10. Current best from naive SD21 baseline: ~0.004. Best from prior manual experiments: ~0.026 (DiffusionSat, down_blocks, ts≈840/999). **Stop and report when R@1 ≥ 0.10 is reached — do not push higher.**

**Prior experiments to read before starting**:
- `/root/diffusion-vpr/results/1-baseline-comparison.csv` — cross-model comparison (DINOv3, DINOv2, RemoteCLIP, DiffusionSat, SD21, etc.)
- `/root/diffusion-vpr/results/2-evaluate-timesteps.csv` — timestep and layer sweep for diffusion features (**NOTE**: this sweep used DiffusionSat, not vanilla SD21 — the optimal timestep for SD21 may differ and should be validated early)

Read both CSVs at the start of setup to absorb what has already been tried.

## Setup

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `apr16`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from main.
3. **Read the in-scope files** — the repo is small, read all of them:
   - `prepare.py` — fixed constants, dataset loading, and the evaluation function. Do not modify.
   - `train.py` — the file you modify. Feature extraction pipeline and test-time adaptation.
4. **Read prior results**: read both CSV files listed above.
5. **Verify setup**: Run `uv run prepare.py` — it should print dataset sizes (768 UAV, 2860 sat) and confirm SD21 is cached.
6. **Initialize results.tsv**: Create with just the header row. The baseline will be recorded after the first run.
7. **Confirm and go**.

## Experimentation

Each experiment runs on a single GPU. The time budget per experiment is **12 minutes wall-clock** (TIME_BUDGET = 720s, excluding Python startup).

**What you CAN do** (modify `train.py` freely):
- Change which UNet layers to hook (down_blocks, mid_block, up_blocks — any combination)
- Change the DDPM timestep (0–999); the baseline uses 880 but this is unvalidated for vanilla SD21
- Change the aggregation method (GeM pool, average pool, spatial patch descriptors, etc.)
- Change the text prompt or remove text conditioning entirely (null/empty text)
- Change the image resolution fed into the UNet (currently IMG_SIZE=256)
- Apply **test-time adaptation** — any unsupervised method fitted on the unlabelled eval set itself:
  - PCA / PCA whitening (removes dominant style/illumination components)
  - Dimensionality reduction
  - Feature centering / standardisation
  - Any other transform that uses no GPS labels
- Combine features from multiple stages/timesteps
- Use the VAE encoder output directly (skip the UNet entirely — the VAE is part of the SD21 pipeline)
- Try DDIM inversion instead of forward noising
- Anything else that fits in the time budget and uses no label supervision

**What you CANNOT do**:
- Modify `prepare.py`.
- Install new packages or add dependencies.
- Use supervised training of any kind (no GPS-paired positive/negative training, no contrastive loss).
- Use external vision models. Allowed feature sources are: **SD v2.1 UNet activations** and **SD v2.1 VAE encoder output**. The CLIP text encoder is allowed only for producing text conditioning — its visual features are off-limits.

**Strategy**: Start with fast, coarse, exploratory sweeps. Changes can be large and non-minimal. The goal is to cover the search space quickly in early experiments, then drill into what shows promise. Suggested early priorities:
1. **Validate timestep for SD21**: run a quick sweep over a handful of timesteps (e.g. 200, 400, 600, 800, 950) with the current baseline to find SD21's own optimum — don't assume the DiffusionSat sweep transfers.
2. **Try null text conditioning**: remove the CLIP prompt entirely (pass the unconditioned embedding or zero-embed) to get purely visual UNet features.
3. **PCA whitening**: fit PCA on the concatenated UAV+satellite embeddings, drop the first 1–3 components (likely capture illumination/style, not location), re-evaluate. This is a 5-line change and may give a free boost.
4. After finding a good frozen-feature config, try PCA whitening on top of it.

**Key insight from prior work**: The main challenge is the domain gap between oblique UAV imagery (taken at ~400–600m altitude) and nadir satellite tiles. The UNet features at high noise timesteps (~t=800–950) encode coarse structure that partially survives this gap — but they still score near-random for SD21 vs. ~0.026 for DiffusionSat which was fine-tuned on satellite data. Closing the gap without training likely means finding the representations where location-correlated structure dominates illumination/viewpoint variation.

## Output format

When the script finishes it prints:
```
---
R@1:           0.004000
R@5:           0.015625
R@10:          0.029948
elapsed_s:     174.8
emb_dim:       4480
avg_gt_chunks: 12.4
```

Extract the key metrics:
```
grep "^R@[15]\|^elapsed_s" run.log
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
a1b2c3d	0.003906	0.015625	180.2	keep	baseline: down_blocks ts=880 GeM
b2c3d4e	0.002604	0.013021	175.0	discard	ts=400 (worse)
c3d4e5f	0.007813	0.026042	185.0	keep	ts=950 + null text
d4e5f6g	0.015625	0.052083	210.0	keep	PCA whiten 128d on top
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/apr16`).

**First run**: always run the baseline first (train.py as written) to establish the reference point.

LOOP FOREVER (until R@1 ≥ 0.10 or manually interrupted):

1. Review git state and results.tsv to understand where you are.
2. Form a hypothesis. Consult the prior CSV files to avoid re-running known dead-ends.
3. Edit `train.py` — changes can be large and exploratory, especially early on.
4. `git commit` the change.
5. `git push origin HEAD` — push immediately so results are visible upstream.
6. `uv run train.py > run.log 2>&1`
7. Extract metrics: `grep "^R@1:\|^R@5:\|^elapsed_s:" run.log`
8. If grep is empty → crash. Read `tail -n 50 run.log`, fix if trivial, otherwise log as crash and discard.
9. Log to `results.tsv`.
10. If R@1 **improved** (strictly higher than current best): keep the commit, stay on it.
11. If R@1 is equal or worse: `git reset --hard HEAD~1 && git push --force origin HEAD`.

**Halt condition**: When R@1 ≥ 0.10 is first reached, log it, push, and stop. Print a clear summary of what worked.

**Timeout**: If a run exceeds 14 minutes total, kill it and treat as crash.

If you run out of ideas: re-read the prior CSVs, look for near-misses in your own results.tsv, try combinations of things that individually helped a little.
