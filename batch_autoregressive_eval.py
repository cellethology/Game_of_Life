"""
Batch autoregressive evaluation with visualizations, aligned to existing settings.

- Regimes/densities/burn-in mirror prior experiments:
    early: burn_in=10,  steps=40
    mid:   burn_in=60,  steps=40
    late:  burn_in=160, steps=40
    densities typically 0.2 / 0.4 / 0.6 (can override via CLI).
- For each (patch_size, regime, density, seed):
    * Generate t0 via true GoL burn-in on a random board.
    * Roll true GoL and model autoregressive for T steps (default 50).
    * Save density/accuracy curves and snapshot grids.
    * Record metrics to CSV.

Inputs match training (center removed; 8/24 dims); boundary uses wrap.
Defaults load weighted (V1) checkpoints from checkpoints_weighted/.
"""

from pathlib import Path
from typing import List, Dict, Tuple
import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import itertools

from models import MLP
from game_of_life import random_board, step as gol_step


# ----------------------- Core utilities ----------------------- #

def load_model(checkpoint: Path, patch_size: int, device: torch.device) -> torch.nn.Module:
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
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def extract_patches_wrap(board: np.ndarray, patch_size: int) -> torch.Tensor:
    """Extract all patches with wrap padding, removing center."""
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
            patch = padded[i:i + patch_size, j:j + patch_size].reshape(-1)
            patches[k] = np.delete(patch, center_idx)
            k += 1
    return torch.from_numpy(patches)


def predict_board(model: torch.nn.Module, board: np.ndarray, patch_size: int, device: torch.device,
                  threshold: float, temperature: float = 1.0) -> np.ndarray:
    patches = extract_patches_wrap(board, patch_size).to(device)
    with torch.no_grad():
        logits = model(patches)
        if temperature != 1.0:
            logits = logits / temperature
        probs = torch.sigmoid(logits)
        pred = (probs > threshold).float().cpu().numpy().astype(np.uint8)
    return pred.reshape(board.shape)


def run_rollout(model: torch.nn.Module, t0: np.ndarray, patch_size: int, device: torch.device,
                steps: int, threshold: float, temperature: float = 1.0,
                adaptive: bool = False, target_density: float | None = None,
                adaptive_gain: float = 0.0, thr_min: float = 0.1, thr_max: float = 0.9) -> List[np.ndarray]:
    cur = t0.copy()
    traj = []
    thr = threshold
    tgt = target_density if target_density is not None else density(t0)
    for _ in range(steps):
        nxt = predict_board(model, cur, patch_size, device, thr, temperature)
        traj.append(nxt)
        cur = nxt
        if adaptive:
            dens = density(nxt)
            thr = np.clip(thr + adaptive_gain * (dens - tgt), thr_min, thr_max)
    return traj


def run_gol(t0: np.ndarray, steps: int) -> List[np.ndarray]:
    cur = t0.copy()
    traj = []
    for _ in range(steps):
        cur = gol_step(cur)
        traj.append(cur.copy())
    return traj


def density(board: np.ndarray) -> float:
    return float(board.mean())


def first_collapse(trajectory: List[np.ndarray]) -> Tuple[int | None, int | None]:
    zero = one = None
    for idx, b in enumerate(trajectory, 1):
        if zero is None and b.sum() == 0:
            zero = idx
        if one is None and b.sum() == b.size:
            one = idx
    return zero, one


def agreement(pred_traj: List[np.ndarray], true_traj: List[np.ndarray]) -> List[float]:
    steps = min(len(pred_traj), len(true_traj))
    return [float((pred_traj[t] == true_traj[t]).mean()) for t in range(steps)]


def f1_curve(pred_traj: List[np.ndarray], target_traj: List[np.ndarray]) -> List[float]:
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

# ----------------------- Visualization ----------------------- #

def plot_curves(d_true: List[float], d_pred: List[float], acc: List[float], f1: List[float],
                out_dir: Path, name: str, alignment: str) -> None:
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
    out_path = out_dir / f"{name}_curves.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_snapshots(t0: np.ndarray, true_traj: List[np.ndarray], pred_traj: List[np.ndarray],
                   steps_to_show: List[int], out_dir: Path, name: str, alignment: str) -> None:
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

    for col, s in enumerate(steps):
        t_board = true_board(s)
        p_idx = (s - 1) if alignment == "next" else (s if s > 0 else 0)
        p_board = pred_traj[p_idx] if p_idx < len(pred_traj) else pred_traj[-1]
        axes[0, col].imshow(t_board, cmap="binary", interpolation="nearest", vmin=0, vmax=1)
        axes[0, col].set_title(f"True step {s}" + (" (t0)" if alignment == "current" and s == 0 else ""))
        axes[1, col].imshow(p_board, cmap="binary", interpolation="nearest", vmin=0, vmax=1)
        axes[1, col].set_title(f"Pred step {s}")
        for ax in (axes[0, col], axes[1, col]):
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle("Snapshots (top=true target, bottom=pred)")
    fig.tight_layout()
    out_path = out_dir / f"{name}_snapshots.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# ----------------------- Experiment runner ----------------------- #

def regime_params(regime: str) -> Tuple[int, int]:
    if regime == "early":
        return 10, 40
    if regime == "mid":
        return 60, 40
    if regime == "late":
        return 160, 40
    raise ValueError(f"Unknown regime: {regime}")


def find_checkpoint(patch_size: int, regime: str, density: float, variant: str) -> Path | None:
    """
    Locate a checkpoint based on variant:
    - general: use any best_model_patch{ps}_life_patches*.pth (latest mtime)
    - match:   prefer best_model_patch{ps}_life_patches_{regime}_p{density:.1f}*.pth (latest mtime)
    """
    ckpt_dir = Path("checkpoints_weighted")
    if variant == "match":
        patterns = [
            f"best_model_patch{patch_size}_life_patches_{regime}_p{density:.1f}*.pth",
            f"best_model_patch{patch_size}_{regime}_p{density:.1f}*.pth",  # fallback to files without 'life_patches' in name
        ]
    else:
        patterns = [f"best_model_patch{patch_size}_life_patches*.pth", f"best_model_patch{patch_size}_*.pth"]

    candidates: list[Path] = []
    for pat in patterns:
        candidates.extend(ckpt_dir.glob(pat))
    candidates = sorted(set(candidates), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def run_one(config: Dict, device: torch.device, viz_dir: Path,
            thresholds: List[float], temperatures: List[float],
            steps: int, snap_steps: List[int], alignment: str,
            model_variant: str, tune_short_steps: int = 0,
            adaptive: bool = False, adaptive_gain: float = 0.0,
            thr_min: float = 0.1, thr_max: float = 0.9,
            target_density: float | None = None) -> Dict:
    patch_size = config["patch_size"]
    regime = config["regime"]
    density_init = config["density"]
    seed = config["seed"]
    board_size = config["board_size"]
    burn_in, _ = regime_params(regime)

    ckpt = config.get("checkpoint") or find_checkpoint(patch_size, regime, density_init, model_variant)
    if ckpt is None:
        raise ValueError(
            f"No checkpoint found for patch_size={patch_size} (variant={model_variant}, regime={regime}, density={density_init}). "
            "Provide --checkpoint or place a matching ckpt in checkpoints_weighted/."
        )
    model = load_model(ckpt, patch_size, device)
    rng = np.random.default_rng(seed)

    board = random_board(board_size, p_alive=density_init, rng=rng).astype(np.uint8)
    for _ in range(burn_in):
        board = gol_step(board)
    t0 = board.copy()

    true_traj = run_gol(t0, steps)
    # Hyperparameter selection (temperature/threshold)
    best_temp = temperatures[0]
    best_thr = thresholds[0]

    if tune_short_steps and len(thresholds) * len(temperatures) > 1:
        best_score = -1.0
        best_density_dev = float("inf")
        for temp, thr in itertools.product(temperatures, thresholds):
            short_true = true_traj[:tune_short_steps]
            short_pred = run_rollout(
                model,
                t0,
                patch_size,
                device,
                tune_short_steps,
                thr,
                temp,
                adaptive=adaptive,
                target_density=target_density,
                adaptive_gain=adaptive_gain,
                thr_min=thr_min,
                thr_max=thr_max,
            )
            # targets per alignment
            if alignment == "next":
                targets = short_true
            else:
                targets = [t0] + short_true[: max(len(short_pred) - 1, 0)]
            f1s_short = f1_curve(short_pred, targets)
            score = float(np.mean(f1s_short)) if f1s_short else 0.0
            # density deviation at end of short rollout vs initial
            dens_dev = abs(density(short_pred[-1]) - density(t0)) if short_pred else float("inf")
            if score > best_score or (abs(score - best_score) < 1e-6 and dens_dev < best_density_dev):
                best_score = score
                best_density_dev = dens_dev
                best_temp, best_thr = temp, thr

    pred_traj = run_rollout(
        model,
        t0,
        patch_size,
        device,
        steps,
        best_thr,
        best_temp,
        adaptive=adaptive,
        target_density=target_density,
        adaptive_gain=adaptive_gain,
        thr_min=thr_min,
        thr_max=thr_max,
    )

    d_true_full = [density(b) for b in true_traj]
    d_pred = [density(b) for b in pred_traj]
    if alignment == "next":
        acc = agreement(pred_traj, true_traj)
        f1s = f1_curve(pred_traj, true_traj)
        d_true_plot = d_true_full
    else:
        targets = [t0] + true_traj[: max(len(pred_traj) - 1, 0)]
        acc = [float((p == t).mean()) for p, t in zip(pred_traj, targets)]
        f1s = f1_curve(pred_traj, targets)
        d_true_plot = [density(t0)] + d_true_full[: len(d_pred) - 1]
    zero_step, one_step = first_collapse(pred_traj)
    acc_self_first = float((pred_traj[0] == t0).mean()) if pred_traj else None  # self-consistency vs t0

    name = f"patch{patch_size}_{regime}_p{density_init:.1f}_s{seed}_v{model_variant}_temp{best_temp:.2f}_thr{best_thr:.2f}"
    plot_curves(d_true_plot, d_pred, acc, f1s, viz_dir, name, alignment)
    plot_snapshots(t0, true_traj, pred_traj, snap_steps, viz_dir, name, alignment)

    return {
        "patch_size": patch_size,
        "regime": regime,
        "density": density_init,
        "seed": seed,
        "threshold": best_thr,
        "steps": steps,
        "burn_in": burn_in,
        "board_size": board_size,
        "init_density": density(t0),
        "final_pred_density": d_pred[-1] if d_pred else None,
        "final_true_density": d_true_full[-1] if d_true_full else None,
        "acc_self_first": acc_self_first,
        "acc_first": acc[0] if acc else None,
        "acc_last": acc[-1] if acc else None,
        "f1_first": f1s[0] if f1s else None,
        "f1_last": f1s[-1] if f1s else None,
        "first_zero_step": zero_step,
        "first_one_step": one_step,
        "checkpoint": str(ckpt),
        "model_variant": model_variant,
        "temperature": best_temp,
        "threshold": best_thr,
        "adaptive": adaptive,
        "adaptive_gain": adaptive_gain,
        "thr_min": thr_min,
        "thr_max": thr_max,
        "target_density": target_density if target_density is not None else density(t0),
    }


def main():
    parser = argparse.ArgumentParser(description="Batch autoregressive rollout eval with visuals.")
    parser.add_argument("--patch_sizes", type=int, nargs="*", default=[3, 5],
                        help="Patch sizes to evaluate. 3/5 have defaults; 7 requires --checkpoint.")
    parser.add_argument("--regimes", type=str, nargs="*", default=["early", "mid", "late"])
    parser.add_argument("--densities", type=float, nargs="*", default=[0.2, 0.4, 0.6])
    parser.add_argument("--seeds", type=int, nargs="*", default=[0, 1])
    parser.add_argument("--board_size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.5],
                        help="Decision thresholds to try; if multiple and tune_short_steps>0, will tune.")
    parser.add_argument("--temperatures", type=float, nargs="+", default=[1.0],
                        help="Logit temperatures to try; if multiple and tune_short_steps>0, will tune.")
    parser.add_argument("--adaptive", action="store_true",
                        help="Enable adaptive thresholding per step based on density.")
    parser.add_argument("--adaptive_gain", type=float, default=0.0,
                        help="Gain for adaptive thresholding (thr += gain*(density-target)).")
    parser.add_argument("--thr_min", type=float, default=0.1, help="Min threshold for adaptive mode.")
    parser.add_argument("--thr_max", type=float, default=0.9, help="Max threshold for adaptive mode.")
    parser.add_argument("--target_density", type=float, default=None,
                        help="Target density for adaptive thresholding; default uses initial t0 density.")
    parser.add_argument("--viz_dir", type=Path, default=Path("rollout_batch_viz"))
    parser.add_argument("--snap_steps", type=int, nargs="*", default=[0, 1, 5, 10, 20, 40, 50, 80, 160],
                        help="Steps to visualize; include 0 to show t0 when alignment=current.")
    parser.add_argument("--out_csv", type=Path, default=None,
                        help="(disabled) Output CSV path; metrics are not saved in this run.")
    parser.add_argument("--alignment", choices=["next", "current"], default="next",
                        help="Alignment: 'next' compares to true t+1...; 'current' compares first pred to t0 and starts at step 0.")
    parser.add_argument("--model_variant", choices=["general", "match"], default="general",
                        help="Use general models (p=0.5) or regime/density-matched checkpoints if available.")
    parser.add_argument("--tune_short_steps", type=int, default=0,
                        help="If >0 and multiple temps/thresholds are provided, tune on a short rollout of this length.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Alignment: {args.alignment}")

    configs = []
    for p in args.patch_sizes:
        for r in args.regimes:
            for d in args.densities:
                for s in args.seeds:
                    configs.append({
                        "patch_size": p,
                        "regime": r,
                        "density": d,
                        "seed": s,
                        "board_size": args.board_size,
                        "checkpoint": None,
                    })

    results = []
    viz_dir = args.viz_dir / args.alignment
    for cfg in configs:
        name = f"{cfg['patch_size']}|{cfg['regime']}|{cfg['density']}|{cfg['seed']}"
        try:
            print(f"Running {name} ...")
            res = run_one(
                cfg,
                device,
                viz_dir,
                args.thresholds,
                args.temperatures,
                args.steps,
                args.snap_steps,
                args.alignment,
                args.model_variant,
                args.tune_short_steps,
                args.adaptive,
                args.adaptive_gain,
                args.thr_min,
                args.thr_max,
                args.target_density,
            )
            results.append(res)
        except Exception as e:
            print(f"Failed {name}: {e}")

    # CSV saving disabled per request; focus on visuals
    print(f"Done. Total runs: {len(results)} (metrics not saved to CSV)")


if __name__ == "__main__":
    main()
