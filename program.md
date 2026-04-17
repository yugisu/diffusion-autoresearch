# autoresearch

This is an experiment to have the LLM do its own research.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `apr17`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current main/master.
   - For this run, use: **`autoresearch/supervised-dinov3`**.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, dataset loading, and fixed evaluation. Do not modify.
   - `train.py` — the file you modify. Model, optimizer, training loop.
4. **Verify data exists**: Check that VisLoc data exists at `VISLOC_ROOT` (default from `prepare.py`). If missing, tell the human to run data preparation first.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off experimentation.

## Experimentation

Each experiment runs on a single GPU. The training script runs with a **fixed time budget of 45 minutes** per experiment (wall clock budget for a run).

Run command:

```bash
uv run train.py > run.log 2>&1
```

Task details:

- Model: `facebook/dinov3-vitb16-pretrain-lvd1689m`
- Framework: PyTorch Lightning
- Logging: Weights & Biases (`autoresearch-supervised-dinov3`, auth via env/WANDB_API_KEY)
- Train flights: `01, 02, 04, 05, 06, 08, 09, 10, 11`
- Validation flight: `03`
- Primary metric: `R@1` on flight 03, evaluated with fixed `evaluate_r1` in `prepare.py`
- Goal: **R@1 >= 0.90**

Satellite scale priors (usable with SatChunkDataset, adjustable as per your experiments):

```python
sat_scales = {"01": 0.25,"02": 0.25,"03": 0.25,"04": 0.25,"05": 0.4,"06": 0.6,"08": 0.35,"09": 0.25,"10": 0.5,"11": 0.25,}
```

Start point instruction:

- Initial `max_epochs` should be 20.
- During the loop, agent can change `max_epochs` freely if experiments justify it.

**What you CAN do:**

- Modify `train.py` only. Everything in `train.py` is fair game (head, losses, samplers, miners, augments, optimizer/scheduler, freezing, precision, scale tuning, etc.).
- Perform significant architectural shifts to highlight different exploration pathways.

**What you CANNOT do:**

- Modify `prepare.py`.
- Modify the fixed evaluation logic in `prepare.py`.
- Install new packages or add dependencies during the loop.

**The goal is simple: maximize val `R@1` on flight 03.**

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

Baseline version the start of the experiments provided following results: after 10 epochs at bs=128 R@1=0.4362 R@5=0.66 R@10=0.76. No need to re-run this experiment.

VisLoc dataset description: UAV images feature high-quality images taken at nadir with a Mavic-like drone. Satellite imagery features a large satellite map that gets split into chunks. The goal of the resulting system is to implement geo-localization through retrieval of UAV to satellite images. While the domain gap between UAV and satellite imagery is significant, it seems like it mostly comes from camera specifics (UAV images are of better quality, colors are very different, UAV images include slight haze and have lower contrast), or image timings (satellite images might be out-of-date sometimes). Also, the training dataset features some bodies of water which might be irrelevant for training (or fine!).

The experiments loop is conducted on a powerful machine that has an A100 GPU with 80GB of VRAM - keep that in mind and utilize the GPU accordingly.

Also, there are some suggestions for future experiments (explore and exploit promising branches if you find any, regardless if they belong to this list or not!):
- Multi-positive InfoNCE — Replace the arange labels loss with a masked multi-positive formulation so valid GT satellite chunks are not penalized as
  negatives.
- Stronger UAV and satellite augmentations — Add RandomPerspective + aggressive RandomResizedCrop to UAV; add 0/90/180/270° random rotation and
stronger color jitter to satellite. Target the known domain gap (color, contrast, haze).
- Differential LR + partial unfreezing — Freeze the first ~8 backbone blocks, unfreeze the last 4 with lr=5e-5, head at lr=1e-4. Prevents early
feature degradation while letting the upper layers specialize.
- Larger image size (336px or 448px) — Increase input resolution to retain more texture from the 512px satellite chunks. Leverage the A100's 80GB to
keep batch size reasonable.
- Hard negative mining via GPS proximity — During training, sample negatives that are geographically close to the UAV GPS position but fall outside
the GT bbox, forcing finer spatial discrimination.

## The experiment loop

The experiment runs on a dedicated branch (`autoresearch/supervised-dinov3`).

LOOP FOREVER:

1. Look at git state: current branch/commit.
2. Tune `train.py` with one experimental idea by directly hacking code.
3. git commit.
4. Run experiment: `uv run train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out results from `run.log`.
6. If metrics are missing, run crashed. Use `tail -n 50 run.log` and attempt fix; if not quickly fixable, log crash and move on.
7. Record results in `results.tsv`.
8. If `R@1` improved (higher), advance branch and keep commit.
9. If `R@1` is equal/worse, reset to previous best commit.

**Timeout**: Each experiment should take ~50 minutes total (+ a few seconds for startup and eval overhead). If a run exceeds 55 minutes, kill it, leave a timeout note in the results, and use the best validation metric as its result. While the model should be trained for longer under ideal circumstances, the goal of these experiments is to iterate on approach quickly.

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working _indefinitely_ until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read related papers on the topic (use arxiv and other sources), re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~45 minutes then you can run approx 1/hour, for a total of about 10 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!
