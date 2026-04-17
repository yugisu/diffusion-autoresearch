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

Each experiment runs on a single GPU. The training script runs with a **fixed time budget of 20 minutes** per experiment (wall clock budget for a run).

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
- Goal: **R@1 >= 0.70**

Satellite scale priors (usable with SatChunkDataset, adjustable as per your experiments):

```python
sat_scales = {"01": 0.25,"02": 0.25,"03": 0.25,"04": 0.25,"05": 0.4,"06": 0.6,"08": 0.35,"09": 0.25,"10": 0.5,"11": 0.25,}
```

Start point instruction:

- Initial `max_epochs` should be **10**.
- During the loop, agent can change `max_epochs` freely if experiments justify it.

**What you CAN do:**

- Modify `train.py` only. Everything in `train.py` is fair game (head, losses, samplers, miners, augments, optimizer/scheduler, freezing, precision, scale tuning, etc.).

**What you CANNOT do:**

- Modify `prepare.py`.
- Modify the fixed evaluation logic in `prepare.py`.
- Install new packages or add dependencies during the loop.

**The goal is simple: maximize val `R@1` on flight 03.**

**Simplicity criterion**: all else equal, simpler is better. Keep complexity proportional to gains.

**The first run**: Your very first run should always establish baseline with current defaults (`max_epochs=10`).

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

**Timeout**: Each experiment should take ~25 minutes total (+ a few seconds for startup and eval overhead). If a run exceeds 30 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working _indefinitely_ until you are manually stopped. You are autonomous. If you run out of ideas, think harder — re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~30 minutes then you can run approx 2/hour, for a total of about 16 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!
