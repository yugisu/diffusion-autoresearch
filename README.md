# Evaluation of the efficacy of SSL training

train-ssl.py - best SSL training of the base DINOv3
train-supervised.py - best supervised baseline training routine of the base DINOv3
train-st2.py - best supervised stage-2 training routine of the SSL-tuned DINOv3 model

TODO:
- isolate model loading, allow feeding base / SSL-tuned model into train-supervised.py and train-st2.py
- run the best training stage-2 routine on the base model (creating a new supervised baseline)
- run the best training supervised baseline routine on the SSL-tuned model (evaluating efficacy of the supervised baseline training)
- repeat above for different seeds - 42, 122, 234
- for each run report: run_name, method_name, seed, "R@1", "R@5", "R@10", best_epoch, elapsed_s, notes
- total of 6 runs: (supervised routine x {base, SSL}, stage-2 routine x {base, SSL}) x seed {42, 122, 234}
- aggregate the results, report comparison of efficacy of either method on each stage with mean metrics+-std

table of experiments:
- E1-s42-base-model-supervised-routine
- E2-s42-ssl-model-st2-routine
- E3-s42-ssl-model-supervised-routine
- E4-s42-base-model-st2-routine

set variables:
- Dataset split: VisLoc flight 03 validation (same query/gallery as prior reports)
- Input resolution: 336 x 336
- Evaluation script/path: same code path and metric implementation for both stages
- Hardware target: single A100 80GB (or same GPU type across all runs)

how to run all experiments:
```
tmux new -s experiments
bash run_experiments.sh 2>&1 | tee experiments.log
```

## SSL Fine-Tuning Sweep Results (Latest Complete Matrix)

The table below reports the latest complete 12-run matrix from [results_full.tsv](results_full.tsv), using experiment codes E1-E4 across seeds 42, 122, and 234.

| experiment_code | method_name     | seed |    R@1 |  R@1\* |    R@5 |   R@10 | best_epoch | elapsed_s |
| --------------- | --------------- | ---: | -----: | -----: | -----: | -----: | ---------: | --------: |
| E1-S42          | supervised-base |   42 | 0.7578 | 0.7578 | 0.8737 | 0.9167 |         21 |      3320 |
| E2-S42          | st2-ssl         |   42 | 0.8216 | 0.8516 | 0.9245 | 0.9505 |         24 |      3455 |
| E3-S42          | supervised-ssl  |   42 | 0.7760 | 0.7760 | 0.9089 | 0.9375 |         10 |      3656 |
| E4-S42          | st2-base        |   42 | 0.7812 | 0.8164 | 0.8919 | 0.9323 |         24 |      3763 |
| E1-S122         | supervised-base |  122 | 0.7461 | 0.7461 | 0.8867 | 0.9245 |         21 |      3176 |
| E2-S122         | st2-ssl         |  122 | 0.7878 | 0.7878 | 0.8919 | 0.9440 |         20 |      3416 |
| E3-S122         | supervised-ssl  |  122 | 0.7708 | 0.7708 | 0.8815 | 0.9219 |         21 |      3365 |
| E4-S122         | st2-base        |  122 | 0.8060 | 0.8151 | 0.9102 | 0.9453 |         23 |      3461 |
| E1-S234         | supervised-base |  234 | 0.7240 | 0.7240 | 0.8854 | 0.9232 |         18 |      3066 |
| E2-S234         | st2-ssl         |  234 | 0.8529 | 0.8698 | 0.9401 | 0.9701 |         24 |      3436 |
| E3-S234         | supervised-ssl  |  234 | 0.7695 | 0.7695 | 0.8919 | 0.9414 |         24 |      3588 |
| E4-S234         | st2-base        |  234 | 0.7917 | 0.7852 | 0.9089 | 0.9414 |         21 |      3611 |

---

| method_name     | R@1 (mean +/- std)    |  R@1\* (mean +/- std) |                   R@5 |                  R@10 |     best_epoch |          elapsed_s |
| --------------- | --------------------- | --------------------: | --------------------: | --------------------: | -------------: | -----------------: |
| supervised-base | 0.7426 +/- 0.0140     |     0.7426 +/- 0.0140 |     0.8819 +/- 0.0058 |     0.9215 +/- 0.0034 | 20.00 +/- 1.41 | 3187.33 +/- 104.00 |
| st2-ssl         | **0.8208 +/- 0.0266** | **0.8364 +/- 0.0352** | **0.9188 +/- 0.0201** | **0.9549 +/- 0.0111** | 22.67 +/- 1.89 |  3435.67 +/- 15.92 |
| supervised-ssl  | 0.7721 +/- 0.0028     |     0.7721 +/- 0.0028 |     0.8941 +/- 0.0113 |     0.9336 +/- 0.0084 | 18.33 +/- 6.02 | 3536.33 +/- 124.29 |
| st2-base        | _0.7930 +/- 0.0102_   |   _0.8056 +/- 0.0144_ |   _0.9037 +/- 0.0083_ |   _0.9397 +/- 0.0054_ | 22.67 +/- 1.25 | 3611.67 +/- 123.29 |

### Research Summary: Impact of SSL Fine-Tuning on DINOv3

- SSL fine-tuning improves supervised-routine retrieval over base initialization: E3 vs E1 gives +0.0295 R@1, +0.0122 R@5, +0.0121 R@10 (mean across seeds).
- SSL fine-tuning also improves stage-2 routine retrieval over base initialization: E2 vs E4 gives +0.0278 R@1, +0.0308 R@1\* (patch re-ranking), +0.0152 R@5, +0.0152 R@10 (mean across seeds).
- The strongest overall method in this sweep is E2 (st2-ssl), which has the highest mean retrieval metrics among all four methods.
- Variance is method-dependent: E3 (supervised-ssl) is most stable on R@1 (std 0.0028), while E2 (st2-ssl) reaches the best top-line accuracy with larger cross-seed spread.
