# Conway's Game of Life MLP Prediction

This project trains MLP models to predict the current center cell of Conway's Game of Life using local neighborhoods. It compares the performance of 3×3 versus 5×5 neighborhood sizes.

## Project Overview

1. **Simulation**: Generates Conway's Game of Life trajectories on toroidal grids
2. **Dataset**: Creates paired datasets of 3×3 and 5×5 neighborhoods with the same center cells
3. **Training**: Trains two MLP models to predict center cell states
4. **Comparison**: Evaluates and compares test accuracy between neighborhood sizes

## File Structure

```
life_mlp/
├── __init__.py          # Package initialization
├── game_of_life.py      # Game of Life simulation utilities
├── data.py             # Dataset generation and PyTorch Dataset class
├── models.py           # MLP model definition
├── train.py            # Single experiment training script
├── experiments_mlp.py  # Grid experiment runner
└── README.md           # This file
```

## Dependencies

- Python 3
- PyTorch 2.2.2+ (with CUDA support if available)
- NumPy

## Usage

### From Terminal
```bash
# From the folder containing life_mlp
cd life_mlp
python -m train
```

### From Jupyter Notebook
```python
# In a notebook cell
!python -m life_mlp.train
```

### Grid Experiments

Run systematic experiments across different densities and time regimes:

```bash
# Run all experiments (27 combinations × 3 seeds × 2 patch sizes = 162 models)
python -m life_mlp.experiments_mlp
```

This will:
- Generate datasets with different initial densities (0.2, 0.4, 0.6)
- Test different time regimes (early: burn-in=10, mid: burn-in=60, late: burn-in=160)
- Use multiple random seeds (0, 1, 2) for robustness
- Train both 3×3 and 5×5 models for each configuration
- Save results to `results_mlp_grid.csv`

## Dataset Configuration

The default settings generate a dataset that runs quickly on GPU but still provides good statistics:

- **Board size**: 128×128
- **Training boards**: 16 boards × 60 steps × 200 patches/step = 192,000 samples
- **Test boards**: 4 boards × 60 steps × 200 patches/step = 48,000 samples
- **Patch sizes**: 3×3 (8 features) and 5×5 (24 features)
- **Data file**: `data/life_patches.npz`

## Training Configuration

Default hyperparameters chosen for quick training:

- **Batch size**: 1024
- **Epochs**: 10
- **Learning rate**: 1e-3
- **Optimizer**: Adam
- **Loss**: Binary Cross Entropy with Logits
- **Model**: MLP with hidden layers [128, 128]

The script automatically uses GPU if available (CUDA), otherwise falls back to CPU.

## Expected Runtime

- **Single experiment (train.py)**:
  - NVIDIA A100 GPU: ~2-5 minutes
  - CPU: ~10-30 minutes

- **Full grid experiments (experiments_mlp.py)**:
  - NVIDIA A100 GPU: ~1-2 hours (162 models total)
  - CPU: ~6-12 hours (slower but still feasible)

## Output

### Single Experiment (train.py)
- Training progress for each epoch (loss and accuracy)
- Final test accuracy for both 3×3 and 5×5 models
- Performance improvement achieved by using larger neighborhoods

### Grid Experiments (experiments_mlp.py)
- Progress tracking for all 27 experiment configurations
- Summary table comparing 3×3 vs 5×5 performance across regimes
- CSV file (`results_mlp_grid.csv`) with detailed results for analysis
- Individual datasets saved with descriptive filenames (e.g., `data/life_patches_early_p0.2_burn10_steps40_seed0.npz`)

The datasets are cached for future runs, so subsequent experiments will be faster.