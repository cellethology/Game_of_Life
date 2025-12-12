## Conway Life: Local MLP Predictors, Rollouts, and Dynamics

This repo trains and evaluates MLPs that approximate Conway’s Game of Life.  
We focus on (1) single-step patch prediction for 3×3/5×5/7×7 neighborhoods,  
(2) autoregressive rollouts with/without calibration, and (3) a dynamics
baseline that maps a full 3×3 patch to the next-step center cell.

### What’s here (kept files)
- `data.py` / `game_of_life.py`: dataset generation (toroidal wrap), GoL simulator.
- `models.py`: MLPs for patch classifiers.
- `train_weighted.py`: trains 3×3, 5×5, 7×7 patch models on the general dataset (p=0.5, wrap) with class weighting `w_pos = min(5, sqrt(r))`. Saves to `checkpoints_weighted/`.
- `train_weighted_regimes.py`: train/evaluate per-regime/density if needed.
- `eval_patch_models.py`: evaluate existing checkpoints on test split; can write CSV.
- `batch_autoregressive_eval.py`: batched rollouts (default alignment `current`) with options:
  - `--model_variant {general,match}` selects general ckpt vs regime/density-matched ckpt.
  - `--calibrated` enables temp/threshold scan + optional adaptive thresholding.
  - Outputs snapshots/curves/CSV to `rollout_batch_viz/<calibrated|noncalibrated>/<general|matched>/`.
- `train_dynamics_patch3.py`: 9→1 dynamics model (full 3×3 → next center), with early stopping and board-level rollouts; checkpoints in `checkpoints_dynamics/`.
- `rollout_batch_viz/`: current rollout figures and CSVs (calibrated & noncalibrated, general & matched).
- `legacy_viz/`: older confusion-matrix figures.
- `checkpoints_weighted/`, `checkpoints_dynamics/`, `data/`: trained models and datasets (kept).

### Typical commands
Train general weighted models (wrap data, patch 3/5/7):
```bash
python train_weighted.py
```

Evaluate a checkpoint on the test split:
```bash
python eval_patch_models.py --ckpt checkpoints_weighted/best_model_patch5_life_patches_epoch15.pth
```

Run non-calibrated rollouts (general models, alignment=current):
```bash
python batch_autoregressive_eval.py \
  --patch_sizes 3 5 7 \
  --regimes early mid late \
  --densities 0.2 0.4 0.6 \
  --seeds 0 \
  --steps 50 \
  --model_variant general
# results → rollout_batch_viz/noncalibrated/general/
```

Run calibrated rollouts (temperature + threshold scan, adaptive thresholding enabled):
```bash
python batch_autoregressive_eval.py \
  --patch_sizes 3 5 7 \
  --regimes early mid late \
  --densities 0.2 0.4 0.6 \
  --seeds 0 \
  --steps 50 \
  --model_variant general \
  --calibrated
# results → rollout_batch_viz/calibrated/general/
```

Matched models (if per-regime/density checkpoints are available) use `--model_variant match`
and save under `rollout_batch_viz/.../matched/`.

Train the dynamics 3×3→next-center model and test board rollouts:
```bash
python train_dynamics_patch3.py --generate --max_epochs 10 --patience 2
```

### Data & checkpoints
- Default dataset: `data/life_patches.npz` (wrap extraction, includes 3/5/7 patches).
- General checkpoints: `checkpoints_weighted/best_model_patch{3,5,7}_*.pth`.
- Dynamics checkpoint: `checkpoints_dynamics/best_model_dynamics_patch3.pth`.

### Outputs
- Rollouts: density curves + accuracy/F1 curves + board snapshots per run, plus CSV with
  avg_acc/avg_f1/precision/recall and chosen temp/threshold (for calibrated runs).
- Dynamics script: training/val logs, test metrics, and short board rollouts printed to stdout.

### Notes
- Everything uses toroidal wrap to avoid edge artifacts.
- Class imbalance handled via positive class weight `min(5, sqrt(r))`.
- Calibration (when enabled) scans temps {1, 1.5, 2} and thresholds {0.6, 0.7} on a short horizon, then runs full rollout with optional adaptive threshold tied to predicted density.

### Progress so far
- Early phase: evaluated patch-level predictors across densities/regimes; recall was low, so we adopted the weighted loss (V1) based on F1 trade-off.
- Autoregressive phase: rolled out single-step models; severe “explosion to all-alive” prompted calibration (temp + threshold + optional adaptive tweak). Calibration lifts accuracy but F1 remains modest and drift persists.
- Dynamics probe: trained a 3×3→next-center model using wrapped data; achieved perfect test and board-level rollouts (deterministic mapping is learnable).
- Next steps: use the autoregressive setup to check whether the model reproduces stable GoL motifs (still lifes, oscillators, gliders) or collapses to trivial attractors; refine calibration/thresholding if needed.
