"""
Autoregressive Rollout Analysis for Conway's Game of Life MLP Prediction

This script extends the existing MLP pipeline to perform autoregressive predictions
and compare them against true Game of Life trajectories.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional
import seaborn as sns

# Import existing modules
try:
    from .models import MLP
    from .game_of_life import step
    from .data import LifePatchDataset
except ImportError:
    from models import MLP
    from game_of_life import step
    from data import LifePatchDataset


def load_trained_model(patch_size: int, device: torch.device) -> nn.Module:
    """
    Load a trained MLP model for a given patch size.

    Args:
        patch_size: Size of the neighborhood (3 or 5)
        device: Device to load the model on

    Returns:
        Trained MLP model in eval mode
    """
    # Input dimension depends on patch size (exclude center cell)
    input_dim = 8 if patch_size == 3 else 24

    # Create model with same architecture as training
    model = MLP(input_dim=input_dim, hidden_dims=[128, 128], dropout=0.0)
    model.to(device)
    model.eval()

    return model


def extract_patches(board: np.ndarray, patch_size: int) -> torch.Tensor:
    """
    Extract patches for all cells in the board.

    Args:
        board: (H, W) numpy array of 0/1 values
        patch_size: Size of the neighborhood (3 or 5)

    Returns:
        (H*W, input_dim) tensor of patch features
    """
    H, W = board.shape
    input_dim = 8 if patch_size == 3 else 24

    # Pad the board to handle boundaries
    pad_size = patch_size // 2
    padded = np.pad(board, pad_size, mode='constant', constant_values=0)

    # Extract patches for each cell
    patches = []
    for i in range(H):
        for j in range(W):
            # Extract patch centered at (i, j)
            patch = padded[i:i+patch_size, j:j+patch_size]
            # Remove center cell and flatten
            patch_flat = patch.flatten()
            center_idx = (patch_size // 2) * patch_size + (patch_size // 2)
            patch_without_center = np.delete(patch_flat, center_idx)
            patches.append(patch_without_center.astype(np.float32))

    return torch.tensor(patches)


def predict_board_mlp(model: nn.Module, board: np.ndarray, patch_size: int,
                     device: torch.device, threshold: float = 0.5) -> np.ndarray:
    """
    Use MLP to predict the next state of all cells on the board.

    Args:
        model: Trained MLP model
        board: Current board state (H, W) numpy array of 0/1
        patch_size: Size of neighborhood (3 or 5)
        device: Device to run computation on
        threshold: Classification threshold

    Returns:
        Predicted next board state (H, W) numpy array of 0/1
    """
    H, W = board.shape

    # Extract patches for all cells
    patches = extract_patches(board, patch_size)
    patches = patches.to(device)

    # Predict with model
    with torch.no_grad():
        logits = model(patches)
        probs = torch.sigmoid(logits)
        predictions = (probs > threshold).float().cpu().numpy()

    # Reshape back to board shape
    return predictions.reshape(H, W).astype(np.uint8)


def run_autoregressive_rollout(model: nn.Module, init_board: np.ndarray,
                             patch_size: int, device: torch.device,
                             T: int = 20, threshold: float = 0.5) -> List[np.ndarray]:
    """
    Run autoregressive prediction for T steps starting from initial board.

    Args:
        model: Trained MLP model
        init_board: Initial board state (H, W)
        patch_size: Size of neighborhood (3 or 5)
        device: Device to run computation on
        T: Number of rollout steps
        threshold: Classification threshold

    Returns:
        List of predicted boards [Ŝ_1, Ŝ_2, ..., Ŝ_T]
    """
    predicted_boards = []
    current_board = init_board.copy()

    for step in range(T):
        # Predict next state
        next_board = predict_board_mlp(model, current_board, patch_size, device, threshold)
        predicted_boards.append(next_board)
        current_board = next_board

    return predicted_boards


def compute_gol_trajectory(init_board: np.ndarray, T: int = 20) -> List[np.ndarray]:
    """
    Compute true Game of Life trajectory for T steps.

    Args:
        init_board: Initial board state (H, W)
        T: Number of steps

    Returns:
        List of true boards [S_1, S_2, ..., S_T]
    """
    true_boards = []
    current_board = init_board.copy()

    for step in range(T):
        next_board = step(current_board)
        true_boards.append(next_board.copy())
        current_board = next_board

    return true_boards


def evaluate_rollout(predicted_boards: List[np.ndarray],
                    true_boards: List[np.ndarray]) -> List[float]:
    """
    Evaluate accuracy of rollout predictions against true trajectory.

    Args:
        predicted_boards: List of predicted boards [Ŝ_1, Ŝ_2, ..., Ŝ_T]
        true_boards: List of true boards [S_1, S_2, ..., S_T]

    Returns:
        List of accuracies [acc_1, acc_2, ..., acc_T]
    """
    accuracies = []

    for pred, true in zip(predicted_boards, true_boards):
        accuracy = np.mean(pred == true)
        accuracies.append(accuracy)

    return accuracies


def plot_rollout_accuracy(accuracies_dict: dict, patch_size: int,
                         save_path: str = None):
    """
    Plot accuracy vs rollout step for a given patch size.

    Args:
        accuracies_dict: Dict mapping configuration names to accuracy lists
        patch_size: Patch size being plotted
        save_path: Path to save the figure
    """
    plt.figure(figsize=(10, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(accuracies_dict)))

    for i, (config_name, accuracies) in enumerate(accuracies_dict.items()):
        steps = list(range(1, len(accuracies) + 1))
        plt.plot(steps, accuracies, marker='o', linewidth=2, markersize=6,
                color=colors[i], label=config_name, alpha=0.8)

    plt.xlabel('Rollout Step', fontsize=12)
    plt.ylabel('Accuracy vs True Game-of-Life', fontsize=12)
    plt.title(f'Rollout Accuracy Analysis — {patch_size}×{patch_size} Neighborhood',
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.ylim(0.4, 1.0)  # Adjust based on expected range

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved accuracy plot: {save_path}")

    plt.tight_layout()
    plt.show()


def plot_rollout_snapshots(true_boards: List[np.ndarray],
                         predicted_boards: List[np.ndarray],
                         save_path: str = None, steps_to_show: List[int] = None):
    """
    Plot side-by-side comparison of true vs predicted boards at selected steps.

    Args:
        true_boards: List of true boards
        predicted_boards: List of predicted boards
        save_path: Path to save the figure
        steps_to_show: List of step indices to visualize (e.g., [0, 5, 10, 15, 20])
    """
    if steps_to_show is None:
        steps_to_show = [0, 5, 10, 15, 20]

    # Filter steps to show to not exceed available data
    steps_to_show = [s for s in steps_to_show if s < len(true_boards)]

    n_steps = len(steps_to_show)
    fig, axes = plt.subplots(2, n_steps, figsize=(4*n_steps, 8))
    if n_steps == 1:
        axes = axes.reshape(2, 1)

    for i, step_idx in enumerate(steps_to_show):
        # True board
        true_board = true_boards[step_idx]
        axes[0, i].imshow(true_board, cmap='binary', interpolation='nearest')
        axes[0, i].set_title(f'True t+{step_idx+1}', fontsize=12)
        axes[0, i].set_xticks([])
        axes[0, i].set_yticks([])

        # Predicted board
        pred_board = predicted_boards[step_idx]
        axes[1, i].imshow(pred_board, cmap='binary', interpolation='nearest')
        axes[1, i].set_title(f'Predicted t+{step_idx+1}', fontsize=12)
        axes[1, i].set_xticks([])
        axes[1, i].set_yticks([])

        # Add accuracy text
        accuracy = np.mean(pred_board == true_board)
        axes[1, i].text(0.5, -0.15, f'Acc: {accuracy:.3f}',
                       transform=axes[1, i].transAxes,
                       ha='center', fontsize=10)

    plt.suptitle('Autoregressive Rollout: True vs Predicted Boards',
                 fontsize=16, fontweight='bold', y=0.95)

    # Add row labels
    fig.text(0.02, 0.75, 'True', ha='center', va='center',
             rotation=90, fontsize=14, fontweight='bold')
    fig.text(0.02, 0.25, 'Predicted', ha='center', va='center',
             rotation=90, fontsize=14, fontweight='bold')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved snapshots plot: {save_path}")

    plt.tight_layout()
    plt.show()


def load_test_trajectory(regime: str, density: float, seed: int,
                       board_idx: int = 0, step_idx: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a test trajectory and return initial board + true trajectory.

    Args:
        regime: 'early', 'mid', or 'late'
        density: 0.2, 0.4, or 0.6
        seed: Random seed
        board_idx: Which test board to use
        step_idx: Starting step index

    Returns:
        Tuple of (initial_board, remaining_trajectory)
    """
    # Determine configuration parameters
    if regime == 'early':
        burn_in, num_steps = 10, 40
    elif regime == 'mid':
        burn_in, num_steps = 60, 40
    else:  # late
        burn_in, num_steps = 160, 40

    # Load the corresponding data file
    data_path = f"data/life_patches_{regime}_p{density:.1f}_burn{burn_in}_steps{num_steps}_seed{seed}.npz"

    # Generate trajectory if needed (reuse existing logic)
    from .game_of_life import run_trajectory
    if not Path(data_path).exists():
        print(f"Generating trajectory for {data_path}")
        trajectories = run_trajectory(
            num_boards=4, size=128, num_steps=num_steps,
            p_alive=density, burn_in=burn_in,
            rng=np.random.default_rng(seed)
        )
    else:
        # For simplicity, regenerate on the fly
        trajectories = run_trajectory(
            num_boards=4, size=128, num_steps=num_steps,
            p_alive=density, burn_in=burn_in,
            rng=np.random.default_rng(seed)
        )

    # Get initial board and remaining trajectory
    init_board = trajectories[board_idx, step_idx]
    remaining_trajectory = trajectories[board_idx, step_idx+1:]

    return init_board, remaining_trajectory


def run_rollout_analysis(patch_size: int, regime: str, density: float, seed: int,
                       T: int = 20, device: torch.device = None,
                       save_plots: bool = True) -> dict:
    """
    Run complete rollout analysis for a single configuration.

    Args:
        patch_size: 3 or 5
        regime: 'early', 'mid', 'late'
        density: 0.2, 0.4, or 0.6
        seed: Random seed
        T: Number of rollout steps
        device: Device to use
        save_plots: Whether to save visualization plots

    Returns:
        Dictionary with results
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Running rollout analysis: patch={patch_size}, regime={regime}, density={density}, seed={seed}")

    # Load model
    model = load_trained_model(patch_size, device)

    # Load test data
    init_board, true_trajectory = load_test_trajectory(regime, density, seed)

    # Truncate true trajectory to T steps
    true_trajectory = true_trajectory[:T]

    # Run autoregressive rollout
    predicted_trajectory = run_autoregressive_rollout(
        model, init_board, patch_size, device, T
    )

    # Evaluate accuracy
    accuracies = evaluate_rollout(predicted_trajectory, true_trajectory)

    # Create results dict
    config_name = f"{regime}_p{density:.1f}_s{seed}"
    results = {
        'patch_size': patch_size,
        'regime': regime,
        'density': density,
        'seed': seed,
        'config_name': config_name,
        'accuracies': accuracies,
        'init_board': init_board,
        'true_trajectory': true_trajectory,
        'predicted_trajectory': predicted_trajectory,
        'final_accuracy': accuracies[-1] if accuracies else 0.0
    }

    # Save plots
    if save_plots:
        # Accuracy plot
        acc_plot_path = f"rollout_accuracy_patch{patch_size}_{config_name}.png"
        plot_rollout_accuracy({config_name: accuracies}, patch_size, acc_plot_path)

        # Snapshot plot (only for one representative configuration)
        if patch_size == 5 and regime == 'mid' and density == 0.4:
            snapshot_path = f"rollout_patch5_example_{config_name}.png"
            plot_rollout_snapshots(true_trajectory, predicted_trajectory, snapshot_path)

    print(f"Final accuracy: {results['final_accuracy']:.4f}")
    return results


def save_rollout_results_to_csv(results_list: List[dict], output_path: str = "rollout_results.csv"):
    """
    Save rollout results to CSV for analysis.

    Args:
        results_list: List of result dictionaries
        output_path: Path to save CSV
    """
    rows = []

    for result in results_list:
        for step, acc in enumerate(result['accuracies'], 1):
            rows.append({
                'patch_size': result['patch_size'],
                'regime': result['regime'],
                'density': result['density'],
                'seed': result['seed'],
                'step': step,
                'accuracy': acc
            })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Saved rollout results to {output_path}")


def main():
    """Main function to run comprehensive rollout analysis."""
    print("="*80)
    print("AUTOREGRESSIVE ROLLOUT ANALYSIS")
    print("="*80)

    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    T = 20  # Rollout steps

    # Test configurations (subset for demonstration)
    configs = [
        {'patch_size': 3, 'regime': 'early', 'density': 0.2, 'seed': 0},
        {'patch_size': 3, 'regime': 'mid', 'density': 0.4, 'seed': 0},
        {'patch_size': 3, 'regime': 'late', 'density': 0.6, 'seed': 0},
        {'patch_size': 5, 'regime': 'early', 'density': 0.2, 'seed': 0},
        {'patch_size': 5, 'regime': 'mid', 'density': 0.4, 'seed': 0},
        {'patch_size': 5, 'regime': 'late', 'density': 0.6, 'seed': 0},
    ]

    # Run analysis for each configuration
    all_results = []

    for config in configs:
        try:
            result = run_rollout_analysis(
                patch_size=config['patch_size'],
                regime=config['regime'],
                density=config['density'],
                seed=config['seed'],
                T=T,
                device=device,
                save_plots=True
            )
            all_results.append(result)
        except Exception as e:
            print(f"Error in {config}: {e}")
            continue

    # Plot combined accuracy comparison
    if all_results:
        # Group by patch size
        patch3_results = {r['config_name']: r['accuracies'] for r in all_results if r['patch_size'] == 3}
        patch5_results = {r['config_name']: r['accuracies'] for r in all_results if r['patch_size'] == 5}

        # Plot comparison
        if patch3_results:
            plot_rollout_accuracy(patch3_results, 3, "rollout_accuracy_comparison_patch3.png")
        if patch5_results:
            plot_rollout_accuracy(patch5_results, 5, "rollout_accuracy_comparison_patch5.png")

        # Save results to CSV
        save_rollout_results_to_csv(all_results)

    print(f"\n{'='*80}")
    print("ROLLOUT ANALYSIS COMPLETE!")
    print(f"Analyzed {len(all_results)} configurations")
    print(f"Results saved to rollout_results.csv")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()