"""Dataset generation and loading for Conway's Game of Life patches."""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Tuple

try:
    from .game_of_life import run_trajectory
except ImportError:
    from game_of_life import run_trajectory


def generate_life_patch_dataset(
    out_path: str = "data/life_patches.npz",
    board_size: int = 128,
    num_train_boards: int = 16,
    num_test_boards: int = 4,
    num_steps: int = 60,
    burn_in: int = 50,
    p_alive: float = 0.5,
    num_patches_per_step: int = 200,
    seed: int = 42,
    include_patch7: bool = True,
    wrap_extraction: bool = True,
) -> None:
    """
    Generate train/test patch datasets and save them as a single .npz file.

    The .npz should contain:
    - X3_train, X5_train, [X7_train], y_train
    - X3_test,  X5_test,  [X7_test],  y_test

    Shapes:
    - X3_*: (N, 8)   float32 or int64
    - X5_*: (N, 24)
    - X7_*: (N, 48)  (only if include_patch7 is True)
    - y_*:  (N,)     int64 (0 or 1)

    Args:
        out_path: Path to save the dataset
        board_size: Size of the square board
        num_train_boards: Number of boards for training
        num_test_boards: Number of boards for testing
        num_steps: Number of steps to store after burn_in
        burn_in: Number of burn-in steps before storing data
        p_alive: Initial probability of a cell being alive
        num_patches_per_step: Number of random patches to sample per step
        seed: Random seed for reproducibility
    """
    # Create output directory
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    # Check if file exists and warn about overwriting
    if Path(out_path).exists():
        print(f"Overwriting existing dataset at {out_path}")

    rng = np.random.default_rng(seed)

    print("Generating training data...")
    train_trajectories = run_trajectory(
        num_boards=num_train_boards,
        size=board_size,
        num_steps=num_steps,
        p_alive=p_alive,
        burn_in=burn_in,
        rng=rng
    )

    print("Generating test data...")
    test_trajectories = run_trajectory(
        num_boards=num_test_boards,
        size=board_size,
        num_steps=num_steps,
        p_alive=p_alive,
        burn_in=burn_in,
        rng=rng
    )

    def extract_patches(trajectories: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
        """Extract 3x3, 5x5, and optionally 7x7 patches from trajectories."""
        num_boards, num_steps, size, _ = trajectories.shape
        total_samples = num_boards * num_steps * num_patches_per_step

        # Initialize arrays
        X3 = np.zeros((total_samples, 8), dtype=np.float32)
        X5 = np.zeros((total_samples, 24), dtype=np.float32)
        X7 = np.zeros((total_samples, 48), dtype=np.float32) if include_patch7 else None
        y = np.zeros(total_samples, dtype=np.int64)

        sample_idx = 0

        for board_idx in range(num_boards):
            for step_idx in range(num_steps):
                board = trajectories[board_idx, step_idx]

                # Sample random centers anywhere on the board
                centers_x = rng.integers(0, size, size=num_patches_per_step)
                centers_y = rng.integers(0, size, size=num_patches_per_step)

                for cx, cy in zip(centers_x, centers_y):
                    if wrap_extraction:
                        padded = np.pad(board, 3, mode="wrap")  # pad by max radius
                        # offsets in padded array
                        ox, oy = cx + 3, cy + 3
                        patch3 = padded[ox-1:ox+2, oy-1:oy+2]
                        patch5 = padded[ox-2:ox+3, oy-2:oy+3]
                        patch7 = padded[ox-3:ox+4, oy-3:oy+4] if include_patch7 else None
                    else:
                        # Fallback to cropping (should not be used if wrap_extraction=True)
                        patch3 = board[max(cx-1,0):min(cx+2,size), max(cy-1,0):min(cy+2,size)]
                        patch5 = board[max(cx-2,0):min(cx+3,size), max(cy-2,0):min(cy+3,size)]
                        patch7 = board[max(cx-3,0):min(cx+4,size), max(cy-3,0):min(cy+4,size)] if include_patch7 else None

                    patch3_center_removed = np.delete(patch3.flatten(), 4)
                    X3[sample_idx] = patch3_center_removed.astype(np.float32)

                    patch5_center_removed = np.delete(patch5.flatten(), 12)
                    X5[sample_idx] = patch5_center_removed.astype(np.float32)

                    if include_patch7 and patch7 is not None:
                        patch7_center_removed = np.delete(patch7.flatten(), 24)
                        X7[sample_idx] = patch7_center_removed.astype(np.float32)

                    # Label is the center cell value
                    y[sample_idx] = int(board[cx, cy])

                    sample_idx += 1

        return X3, X5, X7, y

    print("Extracting patches from training data...")
    X3_train, X5_train, X7_train, y_train = extract_patches(train_trajectories)

    print("Extracting patches from test data...")
    X3_test, X5_test, X7_test, y_test = extract_patches(test_trajectories)

    # Save to npz file with metadata
    print(f"Saving dataset to {out_path}")

    # Create metadata dictionary
    metadata = {
        'board_size': board_size,
        'num_train_boards': num_train_boards,
        'num_test_boards': num_test_boards,
        'num_steps': num_steps,
        'burn_in': burn_in,
        'p_alive': p_alive,
        'num_patches_per_step': num_patches_per_step,
        'seed': seed,
        'wrap_extraction': wrap_extraction,
        'train_samples': len(y_train),
        'test_samples': len(y_test)
    }

    arrays = dict(
        X3_train=X3_train,
        X5_train=X5_train,
        y_train=y_train,
        X3_test=X3_test,
        X5_test=X5_test,
        y_test=y_test,
        **{f'meta_{k}': v for k, v in metadata.items()}
    )
    if include_patch7:
        arrays["X7_train"] = X7_train
        arrays["X7_test"] = X7_test

    np.savez_compressed(out_path, **arrays)

    print(f"Dataset saved successfully!")
    print(f"Train samples: {len(y_train)}")
    print(f"Test samples: {len(y_test)}")


class LifePatchDataset(Dataset):
    """PyTorch Dataset for Game of Life patches."""

    def __init__(self, npz_path: str, split: str = "train", patch_size: int = 3):
        """
        split: "train" or "test"
        patch_size: 3, 5, or 7
        """
        self.split = split
        self.patch_size = patch_size

        # Load data
        data = np.load(npz_path)

        if patch_size == 3:
            self.X = torch.from_numpy(data[f"X3_{split}"])
        elif patch_size == 5:
            self.X = torch.from_numpy(data[f"X5_{split}"])
        elif patch_size == 7:
            key = f"X7_{split}"
            if key not in data:
                raise ValueError(f"{key} not found in {npz_path}. Regenerate dataset with include_patch7=True.")
            self.X = torch.from_numpy(data[key])
        else:
            raise ValueError(f"patch_size must be 3, 5, or 7, got {patch_size}")

        self.y = torch.from_numpy(data[f"y_{split}"])

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]
