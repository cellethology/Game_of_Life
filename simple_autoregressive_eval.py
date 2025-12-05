"""
Minimal autoregressive rollout test for the current-frame MLP.

What it does:
- Sample a random board, run true GoL burn-in to get t0 (matching training distribution).
- From t0, run:
  * True GoL trajectory (reference).
  * MLP autoregressive trajectory (model output -> next input).
- Report per-step density, collapse-to-zero/one step, and short-horizon agreement with true GoL.

Notes:
- The MLP input matches training: center removed (8 dims for 3x3, 24 for 5x5).
- Boundaries use wrap padding to mirror the toroidal GoL used during data generation.
- Default checkpoints point to the weighted (V1) models in `checkpoints_weighted`.
"""

from pathlib import Path
from typing import Tuple, List
import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt

from models import MLP
from game_of_life import random_board, step as gol_step


def load_model(checkpoint: Path, patch_size: int, device: torch.device) -> torch.nn.Module:
    """Load an MLP checkpoint for the given patch size."""
    if patch_size == 3:
        input_dim = 8
    elif patch_size == 5:
        input_dim = 24
    elif patch_size == 7:
        input_dim = 48
    else:
        raise ValueError(f"Unsupported patch_size {patch_size}")
    model = MLP(input_dim=input_dim, hidden_dims=[128, 128], dropout=0.0)
    state = torch.load(checkpoint, map_location=device)
    # Handle wrapped checkpoints that store a dict with model_state_dict, optimizer, etc.
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def extract_patches_wrap(board: np.ndarray, patch_size: int) -> torch.Tensor:
    """Extract all patches with wrap padding (toroidal), removing the center."""
    h, w = board.shape
    pad = patch_size // 2
    padded = np.pad(board, pad, mode="wrap")
    if patch_size == 3:
        input_dim = 8
        center_idx = 4
    elif patch_size == 5:
        input_dim = 24
        center_idx = 12
    elif patch_size == 7:
        input_dim = 48
        center_idx = 24  # 3*7 + 3
    else:
        raise ValueError(f"Unsupported patch_size {patch_size}")
    patches = np.empty((h * w, input_dim), dtype=np.float32)

    k = 0
    for i in range(h):
        for j in range(w):
            patch = padded[i : i + patch_size, j : j + patch_size].reshape(-1)
            patches[k] = np.delete(patch, center_idx)
            k += 1

    return torch.from_numpy(patches)


def predict_board(model: torch.nn.Module, board: np.ndarray, patch_size: int, device: torch.device,
                  threshold: float = 0.5) -> np.ndarray:
    """Predict an entire board with the MLP."""
    patches = extract_patches_wrap(board, patch_size).to(device)
    with torch.no_grad():
        logits = model(patches)
        probs = torch.sigmoid(logits)
        pred = (probs > threshold).float().cpu().numpy().astype(np.uint8)
    return pred.reshape(board.shape)


def run_rollout(model: torch.nn.Module, init_board: np.ndarray, patch_size: int, device: torch.device,
                steps: int, threshold: float = 0.5) -> List[np.ndarray]:
    """Autoregressive rollout: model output feeds next step."""
    boards = []
    cur = init_board.copy()
    for _ in range(steps):
        nxt = predict_board(model, cur, patch_size, device, threshold)
        boards.append(nxt)
        cur = nxt
    return boards


def run_gol(init_board: np.ndarray, steps: int) -> List[np.ndarray]:
    """True GoL trajectory for reference."""
    boards = []
    cur = init_board.copy()
    for _ in range(steps):
        cur = gol_step(cur)
        boards.append(cur.copy())
    return boards


def compute_density(board: np.ndarray) -> float:
    """Fraction of alive cells."""
    return float(board.mean())


def first_collapse_step(trajectory: List[np.ndarray]) -> Tuple[int | None, int | None]:
    """
    Return the first step index (1-based) when trajectory becomes all-zero or all-one.
    (none if never happens within the trajectory).
    """
    zero_step = one_step = None
    for idx, b in enumerate(trajectory, 1):
        if zero_step is None and b.sum() == 0:
            zero_step = idx
        if one_step is None and b.sum() == b.size:
            one_step = idx
    return zero_step, one_step


def agreement_curve(pred_traj: List[np.ndarray], true_traj: List[np.ndarray]) -> List[float]:
    """Per-step accuracy vs true GoL for the overlapping horizon."""
    steps = min(len(pred_traj), len(true_traj))
    return [float((pred_traj[t] == true_traj[t]).mean()) for t in range(steps)]


def f1_curve(pred_traj: List[np.ndarray], target_traj: List[np.ndarray]) -> List[float]:
    """Per-step F1 vs target trajectory."""
    steps = min(len(pred_traj), len(target_traj))
    f1s = []
    for t in range(steps):
        pred = pred_traj[t]
        targ = target_traj[t]
        tp = np.logical_and(pred == 1, targ == 1).sum()
        fp = np.logical_and(pred == 1, targ == 0).sum()
        fn = np.logical_and(pred == 0, targ == 1).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1s.append(f1)
    return f1s


def plot_density_acc_f1(d_true: List[float], d_pred: List[float], acc: List[float], f1: List[float],
                        out_dir: Path, prefix: str, alignment: str) -> None:
    """Save a line plot for density, accuracy, and F1 curves."""
    out_dir.mkdir(parents=True, exist_ok=True)
    steps = np.arange(len(d_true)) if alignment == "current" else np.arange(1, len(d_true) + 1)
    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.plot(steps, d_true, label="true density", color="tab:blue")
    ax1.plot(steps, d_pred, label="pred density", color="tab:orange")
    ax1.set_xlabel("step" + (" (0 = t0)" if alignment == "current" else ""))
    ax1.set_ylabel("density")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(steps[:len(acc)], acc, label="acc vs target", color="tab:green", linestyle="--")
    ax2.plot(steps[:len(f1)], f1, label="f1 vs target", color="tab:red", linestyle=":")
    ax2.set_ylabel("accuracy / F1")
    ax2.set_ylim(0, 1)
    ax2.legend(loc="upper right")

    fig.tight_layout()
    out_path = out_dir / f"{prefix}_curves.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved curves to {out_path}")


def plot_snapshots(t0: np.ndarray, true_traj: List[np.ndarray], pred_traj: List[np.ndarray],
                   steps_to_show: List[int], out_dir: Path, prefix: str, alignment: str) -> None:
    """
    Save side-by-side snapshots for selected steps.
    For alignment='current', allow step 0 to show t0 (true) vs first prediction.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    if alignment == "next":
        steps = [s for s in steps_to_show if 1 <= s <= len(pred_traj)]
    else:
        max_step = len(pred_traj) + 1  # include step 0 = t0
        steps = [s for s in steps_to_show if 0 <= s < max_step]
    if not steps:
        return
    n = len(steps)
    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6))
    if n == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    def true_board(idx: int) -> np.ndarray:
        if alignment == "next":
            return true_traj[idx - 1]  # step s compares pred s to true s (0-based index s-1)
        return t0 if idx == 0 else true_traj[idx - 1]

    for col, step_id in enumerate(steps):
        t_board = true_board(step_id)
        p_idx = (step_id - 1) if alignment == "next" else (step_id if step_id > 0 else 0)
        p_board = pred_traj[p_idx] if p_idx < len(pred_traj) else pred_traj[-1]
        axes[0, col].imshow(t_board, cmap="binary", interpolation="nearest", vmin=0, vmax=1)
        axes[0, col].set_title(f"True step {step_id}" + (" (t0)" if alignment == "current" and step_id == 0 else ""))
        axes[1, col].imshow(p_board, cmap="binary", interpolation="nearest", vmin=0, vmax=1)
        axes[1, col].set_title(f"Pred step {step_id}")
        for ax in (axes[0, col], axes[1, col]):
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle("Snapshots (top=true target, bottom=pred)")
    fig.tight_layout()
    out_path = out_dir / f"{prefix}_snapshots.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved snapshots to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Simple autoregressive rollout sanity check.")
    parser.add_argument("--patch_size", type=int, choices=[3, 5, 7], default=3)
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Model checkpoint path. If not set, use weighted defaults per patch size.")
    parser.add_argument("--board_size", type=int, default=128)
    parser.add_argument("--p_alive", type=float, default=0.5)
    parser.add_argument("--burn_in", type=int, default=50)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--viz_dir", type=Path, default=Path("rollout_viz"),
                        help="Directory to save plots.")
    parser.add_argument("--snap_steps", type=int, nargs="*", default=[0, 1, 5, 10, 20, 50],
                        help="Steps to visualize; include 0 to show t0 when alignment=current.")
    parser.add_argument("--alignment", choices=["next", "current"], default="next",
                        help="Alignment: 'next' compares to true t+1...; 'current' compares first pred to t0 and starts at step 0.")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Pick default checkpoint if not provided (only for 3/5).
    if args.checkpoint is None:
        if args.patch_size == 3:
            args.checkpoint = Path("checkpoints_weighted/best_model_patch3_life_patches_epoch13.pth")
        elif args.patch_size == 5:
            args.checkpoint = Path("checkpoints_weighted/best_model_patch5_life_patches_epoch19.pth")
        else:
            raise ValueError("No default checkpoint for patch_size=7. Please provide --checkpoint.")

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    print(f"Device: {device}")
    print(f"Patch: {args.patch_size}x{args.patch_size}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Board: {args.board_size}x{args.board_size}, p_alive={args.p_alive}")
    print(f"Burn-in: {args.burn_in}, rollout steps: {args.steps}")
    print(f"Alignment: {args.alignment}")

    model = load_model(args.checkpoint, args.patch_size, device)

    # Burn-in with true GoL to match training distribution.
    board = random_board(args.board_size, args.p_alive, rng=rng).astype(np.uint8)
    for _ in range(args.burn_in):
        board = gol_step(board)
    t0 = board.copy()

    true_traj = run_gol(t0, args.steps)
    pred_traj = run_rollout(model, t0, args.patch_size, device, args.steps, args.threshold)

    # Metrics with alignment handling
    density_true = [compute_density(b) for b in true_traj]
    density_pred = [compute_density(b) for b in pred_traj]
    if args.alignment == "next":
        acc_vs_target = agreement_curve(pred_traj, true_traj)
        f1_vs_target = f1_curve(pred_traj, true_traj)
        plot_true_density = density_true
        plot_pred_density = density_pred
    else:
        targets = [t0] + true_traj[: max(len(pred_traj) - 1, 0)]
        acc_vs_target = [float((p == t).mean()) for p, t in zip(pred_traj, targets)]
        f1_vs_target = f1_curve(pred_traj, targets)
        plot_true_density = [compute_density(t0)] + density_true[: len(density_pred) - 1]
        plot_pred_density = density_pred
    zero_step, one_step = first_collapse_step(pred_traj)
    acc_self_first = float((pred_traj[0] == t0).mean()) if pred_traj else None  # self-consistency: pred vs t0

    print("\n=== Summary ===")
    print(f"Initial density t0: {compute_density(t0):.4f}")
    if acc_self_first is not None:
        print(f"Acc (self-consistency) t0 vs first pred: {acc_self_first:.4f}")
    print(f"Pred density @ step 1/5/10/20/50 (clipped to T):")
    for s in [1, 5, 10, 20, 50]:
        if s <= len(density_pred):
            idx = s - 1
            true_ref = density_true[idx if args.alignment == "next" else (idx - 1 if idx > 0 else 0)]
            print(f"  step {s:>3}: pred={density_pred[idx]:.4f}, true_ref={true_ref:.4f}, acc_vs_target={acc_vs_target[idx]:.4f}, f1={f1_vs_target[idx]:.4f}")
    print(f"First all-zero step: {zero_step}, first all-one step: {one_step}")

    print("\nDensity curves (first 10 steps):")
    for s in range(min(10, len(density_pred))):
        true_ref = density_true[s if args.alignment == "next" else (s - 1 if s > 0 else 0)]
        print(f"  step {s+1:>2}: pred={density_pred[s]:.4f}, true_ref={true_ref:.4f}, acc_vs_target={acc_vs_target[s]:.4f}, f1={f1_vs_target[s]:.4f}")

    # Visualization
    viz_subdir = args.viz_dir / args.alignment
    prefix = f"patch{args.patch_size}_seed{args.seed}"
    plot_density_acc_f1(plot_true_density, plot_pred_density, acc_vs_target, f1_vs_target, viz_subdir, prefix, args.alignment)
    plot_snapshots(t0, true_traj, pred_traj, args.snap_steps, viz_subdir, prefix, args.alignment)


if __name__ == "__main__":
    main()
