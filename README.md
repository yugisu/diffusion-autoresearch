# Evaluation of the efficacy of SSL training

train-ssl.py - best SSL training of the base DINOv3
train-supervised.py - best supervised baseline training routine of the base DINOv3
train-st2.py - best supervised stage-2 training routine of the SSL-tuned DINOv3 model

TODO:
- isolate model loading, allow feeding base / SSL-tuned model into train-supervised.py and train-st2.py
- run the best training stage-2 routine on the base model (creating a new supervised baseline)
- run the best training supervised baseline routine on the SSL-tuned model (evaluating efficacy of the supervised baseline training)
- repeat above for different seeds - 42, 122, 234
- for each run report: run_id, method_name, seed, "R@1", "R@5", "R@10", "Dis@1", "Dis@5", "Dis@10", best_epoch, elapsed_s, notes
- total of 6 runs: (supervised routine x {base, SSL}, stage-2 routine x {base, SSL}) x seed {42, 122, 234}
- aggregate the results, report comparison of efficacy of either method on each stage with mean metrics+-std

set variables:
- Dataset split: VisLoc flight 03 validation (same query/gallery as prior reports)
- Input resolution: 336 x 336
- Evaluation script/path: same code path and metric implementation for both stages
- Hardware target: single A100 80GB (or same GPU type across all runs)
