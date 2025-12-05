"""
Train V1-weighted MLPs for GoL patches across regimes/densities (early/mid/late; 0.2/0.4/0.6)
for patch sizes 3/5/7. Also keeps the existing "general" p=0.5 dataset untouched.

Datasets are regenerated with include_patch7=True if missing, with regime-specific burn-in.
"""

from pathlib import Path
import numpy as np

from train_weighted import train_one_model_weighted
from data import generate_life_patch_dataset


def main():
    regimes = {
        "early": {"burn_in": 10, "num_steps": 60},
        "mid": {"burn_in": 60, "num_steps": 60},
        "late": {"burn_in": 160, "num_steps": 60},
    }
    densities = [0.2, 0.4, 0.6]
    patch_sizes = [3, 5, 7]
    base_dir = Path("data")
    base_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("TRAINING REGIME/DENSITY-SPECIFIC MODELS (V1 WEIGHTED)")
    print("=" * 80)

    for regime, cfg in regimes.items():
        for density in densities:
            data_path = base_dir / f"life_patches_{regime}_p{density:.1f}.npz"
            regenerate = False
            if not data_path.exists():
                regenerate = True
            else:
                data = np.load(data_path)
                if "X7_train" not in data.files:
                    regenerate = True
            if regenerate:
                print(f"\nGenerating dataset: {data_path} (burn_in={cfg['burn_in']}, p_alive={density})")
                generate_life_patch_dataset(
                    out_path=str(data_path),
                    board_size=128,
                    num_train_boards=16,
                    num_test_boards=4,
                    num_steps=cfg["num_steps"],
                    burn_in=cfg["burn_in"],
                    p_alive=density,
                    num_patches_per_step=200,
                    seed=42,
                    include_patch7=True,
                    wrap_extraction=True,
                )
            for ps in patch_sizes:
                print("\n" + "-" * 60)
                print(f"Training {ps}x{ps} model for {regime}, p={density:.1f}")
                train_one_model_weighted(
                    patch_size=ps,
                    npz_path=str(data_path),
                    checkpoint_dir="checkpoints_weighted",
                )

    print("\n" + "=" * 80)
    print("DONE training regime/density-specific models.")
    print("=" * 80)


if __name__ == "__main__":
    main()
