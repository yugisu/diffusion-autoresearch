Cumulative Gains: DINOv3 Two-Stage Fine-Tuning (SSL4EO-S12)

Here is the cumulative gains breakdown for the two-stage SSL4EO-S12 fine-tuning pipeline, structured from the zero-shot base up through the final configurations, as well as an analysis of the missing experiments required for strict ablations.

Cumulative Gains Breakdown

0. Zero-Shot Baseline
- Zero-shot DINOv3 — R@1: 34.0%
  Note: No training involved; this is the out-of-the-box performance of the ViT-B/16 model.

1. Stage 1: Self-Supervised Learning (SSL) Pre-training
- Reference SSL — R@1: 53.0% (Gain from Zero-shot: +19.0%)
  Note: Fine-tuned on in-domain VisLoc satellite data using InfoNCE. Taught the model scale invariance but struggled with visual similarities between locations.
- SSL4EO-S12 + InfoNCE (ablation) — R@1: 58.9% (Gain from Reference SSL: +5.9%)
  Note: Identical setup to Reference SSL (InfoNCE, same augmentation, LoRA rank=16 last 4 blocks) but training data swapped to SSL4EO-S12 seasonal temporal pairs. Isolates the dataset contribution.
  Script: train-ssl-ssl4eo-infonce.py | R@5: 72.3%, R@10: 78.3%
- SSL4EO-S12 Best Stage-1 SSL — R@1: 61.5% (Gain from SSL4EO-S12 + InfoNCE: +2.6%)
  Note: Switched loss from InfoNCE to VICReg and added a 512-d projection head. This accounts for the remaining gain on top of the dataset swap.
  Decomposition of the original +8.5% gap: ~69% from dataset (SSL4EO-S12 temporal pairs), ~31% from loss (VICReg vs InfoNCE).

2. Stage 2: Supervised Fine-Tuning
- Base Supervised Fine-Tuning (Exp 1) — R@1: 79.3% (Gain from Stage-1 SSL: +17.8%)
  Note: Applied the full baseline supervised configuration from the reference branch to the stronger SSL4EO-S12 checkpoint using InfoNCE and a T0=10 scheduler.
- Added UAV Augmentations (Exp 2) — evaluated alongside TTA below.
  Note: Added random 180-degree rotations during training to help the drone embeddings handle arbitrary heading directions.
- Training Tweaks (Exp 6, 7) — R@1: 81.1% (Cumulative gain over Exp 1: +1.8%)
  Note: Extended to a single cosine cycle (T0=20) to prevent destructive learning rate resets, and slightly raised the head learning rate, pushing the plateau higher.
- SmoothAP Loss (Exp 8) — R@1: 82.6% (Gain over Exp 7: +1.5%)
  Note: Swapped InfoNCE for SmoothAP loss. By directly optimizing average precision, the model escaped sharp local minima and peaked later in training.

3. Inference & Post-Processing Setup (Zero Training Cost)
- Test-Time Augmentation (TTA) (Exp 2) — R@1: 80.5% (Combined gain of Training Augs + TTA over Exp 1: +1.2%)
  Note: Averaged four 90-degree rotations at inference time to smooth out rotational uncertainties.
- Patch Re-ranking (Exp 9) — R@1: 85.8% (Gain over Exp 8: +3.2%)
  Note: Applied on top of the best SmoothAP checkpoint. It leverages spatial patch token chamfer similarity to re-rank the top-50 CLS candidates, preserving local spatial structure (like road intersections) that global pooling loses.

Missing Intermediate Steps & Fair Comparison Experiments
To properly isolate the confounding variables and confirm exactly why certain approaches yielded gains, the following setups are needed:

1. The Stage-1 SSL Disentanglement (Loss vs. Dataset) — RESOLVED
Result: SSL4EO-S12 + InfoNCE = 58.9% R@1. Dataset accounts for +5.9pp of the 8.5pp gap; VICReg accounts for the remaining +2.6pp.
Conclusion: The data domain shift (global temporal pairs vs. in-domain crop pairs) is the dominant factor. VICReg is a real but secondary contributor.

2. Patch Re-ranking Independence
Missing: Apply the patch re-ranking inference mathematically identically on the Exp 1 and Exp 7 InfoNCE checkpoints.
Note: Determines if the massive +3.2% patch-level gain requires SmoothAP's broader embedding setup, or if it is a purely model-agnostic trick.

3. SmoothAP interaction with the LR Scheduler
Missing: Run SmoothAP using the original T0=10 schedule with warm restarts.
Note: Proves whether SmoothAP's gradient structure inherently demands longer, monotonic cool-downs (like in Exp 8), or if it would have succeeded with the reference restart schedule.

4. SmoothAP Hyperparameter Tuning (tau)
Missing: Test tau=0.1 combined with the Gaussian proximity masks.
Note: In Exp 10, Gaussian positive weighting hurt performance, but this might just be a clash with the aggressively sharp tau=0.01 step function. A smoother tau might work better with distance-weighted positives.
