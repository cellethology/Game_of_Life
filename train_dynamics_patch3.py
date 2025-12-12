"""
Train a 3x3 patch dynamics predictor: input is full 3x3 (9 dims) at time t, label is center at t+1.
Uses wrap boundary, generates its own dataset, trains an MLP, and reports test metrics.
Supports early stopping, optional extra test sets, and a board-level sanity check (predict full board vs GoL).
"""

import argparse
from pathlib import Path
from typing import Tuple, List
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn

from game_of_life import run_trajectory, step as gol_step, random_board
from models import MLP


def generate_dynamics_dataset(
    out_path: Path = Path("data/dynamics_patch3.npz"),
    board_size: int = 128,
    num_train_boards: int = 16,
    num_test_boards: int = 4,
    num_steps: int = 60,
    burn_in: int = 50,
    p_alive: float = 0.5,
    num_patches_per_step: int = 200,
    seed: int = 42,
) -> None:
    rng = np.random.default_rng(seed)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("Generating training trajectories...")
    train_traj = run_trajectory(
        num_boards=num_train_boards,
        size=board_size,
        num_steps=num_steps,
        p_alive=p_alive,
        burn_in=burn_in,
        rng=rng,
    )
    print("Generating test trajectories...")
    test_traj = run_trajectory(
        num_boards=num_test_boards,
        size=board_size,
        num_steps=num_steps,
        p_alive=p_alive,
        burn_in=burn_in,
        rng=rng,
    )

    def extract(traj: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        num_boards, T, size, _ = traj.shape
        total = num_boards * T * num_patches_per_step
        X = np.zeros((total, 9), dtype=np.float32)
        y = np.zeros(total, dtype=np.int64)
        idx = 0
        for b in range(num_boards):
            for t in range(T):
                board_t = traj[b, t]
                board_tp1 = gol_step(board_t)
                # sample centers anywhere
                cx = rng.integers(0, size, size=num_patches_per_step)
                cy = rng.integers(0, size, size=num_patches_per_step)
                padded_t = np.pad(board_t, 1, mode="wrap")
                for i, j in zip(cx, cy):
                    patch = padded_t[i:i+3, j:j+3].flatten()
                    X[idx] = patch
                    y[idx] = int(board_tp1[i, j])
                    idx += 1
        return X, y

    print("Extracting patches (train)...")
    X_train, y_train = extract(train_traj)
    print("Extracting patches (test)...")
    X_test, y_test = extract(test_traj)

    np.savez_compressed(
        out_path,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        meta_board_size=board_size,
        meta_burn_in=burn_in,
        meta_num_steps=num_steps,
        meta_p_alive=p_alive,
        meta_num_patches_per_step=num_patches_per_step,
        meta_seed=seed,
    )
    print(f"Saved dynamics dataset to {out_path}")


class DynamicsPatch3(Dataset):
    def __init__(self, npz_path: str, split: str = "train"):
        data = np.load(npz_path)
        self.X = torch.from_numpy(data[f"X_{split}"])
        self.y = torch.from_numpy(data[f"y_{split}"])

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train_dynamics(
    npz_path: Path = Path("data/dynamics_patch3.npz"),
    batch_size: int = 2048,
    max_epochs: int = 10,
    lr: float = 1e-3,
    val_split: float = 0.2,
    patience: int = 2,
) -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = DynamicsPatch3(str(npz_path), split="train")
    test_ds = DynamicsPatch3(str(npz_path), split="test")
    val_size = int(len(train_ds) * val_split)
    train_size = len(train_ds) - val_size
    train_ds, val_ds = random_split(train_ds, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = MLP(input_dim=9, hidden_dims=[64, 64], dropout=0.0).to(device)
    crit = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        tr_loss = tr_correct = tr_total = 0
        for x, y in train_dl:
            x = x.to(device)
            y = y.float().to(device)
            opt.zero_grad()
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * y.numel()
            pred = (torch.sigmoid(logits) > 0.5).float()
            tr_correct += (pred == y).sum().item()
            tr_total += y.numel()
        tr_loss /= tr_total
        tr_acc = tr_correct / tr_total

        model.eval()
        val_loss = val_correct = val_total = 0
        with torch.no_grad():
            for x, y in val_dl:
                x = x.to(device)
                y = y.float().to(device)
                logits = model(x)
                loss = crit(logits, y)
                val_loss += loss.item() * y.numel()
                pred = (torch.sigmoid(logits) > 0.5).float()
                val_correct += (pred == y).sum().item()
                val_total += y.numel()
        val_loss /= val_total
        val_acc = val_correct / val_total

        print(f"Epoch {epoch}/{max_epochs} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | val loss {val_loss:.4f} acc {val_acc:.4f}")

        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = model.state_dict()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Test
    model.eval()
    tot = cor = tp = fp = fn = tn = 0
    with torch.no_grad():
        for x, y in test_dl:
            x = x.to(device)
            y = y.float().to(device)
            logits = model(x)
            pred = (torch.sigmoid(logits) > 0.5).float()
            cor += (pred == y).sum().item()
            tot += y.numel()
            tp += ((pred == 1) & (y == 1)).sum().item()
            tn += ((pred == 0) & (y == 0)).sum().item()
            fp += ((pred == 1) & (y == 0)).sum().item()
            fn += ((pred == 0) & (y == 1)).sum().item()
    acc = cor / tot
    prec = tp / (tp + fp) if tp + fp > 0 else 0.0
    rec = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    print(f"Test: acc={acc:.4f} prec={prec:.4f} rec={rec:.4f} f1={f1:.4f}")

    ckpt_dir = Path("checkpoints_dynamics")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "best_model_dynamics_patch3.pth"
    torch.save({"model_state_dict": model.state_dict()}, ckpt_path)
    print(f"Saved dynamics ckpt to {ckpt_path}")
    return ckpt_path


def eval_on_npz(npz_path: Path, ckpt_path: Path, batch_size: int = 4096) -> Tuple[float, float, float, float]:
    device = torch.device("cpu")
    ds = DynamicsPatch3(str(npz_path), split="test")
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    model = MLP(input_dim=9, hidden_dims=[64, 64], dropout=0.0).to(device)
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()
    tot = cor = tp = fp = fn = tn = 0
    with torch.no_grad():
        for x, y in dl:
            x = x.to(device)
            y = y.float().to(device)
            logits = model(x)
            pred = (torch.sigmoid(logits) > 0.5).float()
            cor += (pred == y).sum().item()
            tot += y.numel()
            tp += ((pred == 1) & (y == 1)).sum().item()
            tn += ((pred == 0) & (y == 0)).sum().item()
            fp += ((pred == 1) & (y == 0)).sum().item()
            fn += ((pred == 0) & (y == 1)).sum().item()
    acc = cor / tot
    prec = tp / (tp + fp) if tp + fp > 0 else 0.0
    rec = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    return acc, prec, rec, f1


def predict_full_board(model: torch.nn.Module, board: np.ndarray) -> np.ndarray:
    """Predict full next board using 3x3 model (wrap)."""
    device = next(model.parameters()).device
    size = board.shape[0]
    padded = np.pad(board, 1, mode="wrap")
    patches = []
    for i in range(size):
        for j in range(size):
            patches.append(padded[i:i+3, j:j+3].flatten())
    patches = torch.from_numpy(np.stack(patches).astype(np.float32)).to(device)
    with torch.no_grad():
        logits = model(patches)
        pred = (torch.sigmoid(logits) > 0.5).float().cpu().numpy().astype(np.uint8).reshape(size, size)
    return pred


def board_rollout_eval(ckpt_path: Path, density: float, burn_in: int, steps: int,
                       board_size: int = 128, seed: int = 0) -> Tuple[List[float], List[float]]:
    """Return per-step accuracy vs true GoL and predicted densities."""
    device = torch.device("cpu")
    model = MLP(input_dim=9, hidden_dims=[64, 64], dropout=0.0).to(device)
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()

    rng = np.random.default_rng(seed)
    board = random_board(board_size, density, rng=rng).astype(np.uint8)
    for _ in range(burn_in):
        board = gol_step(board)
    t0 = board.copy()

    accs = []
    dens = []
    cur = t0
    for _ in range(steps):
        true_next = gol_step(cur)
        pred_next = predict_full_board(model, cur)
        accs.append(float((pred_next == true_next).mean()))
        dens.append(float(pred_next.mean()))
        cur = pred_next
    return accs, dens


def board_level_check(ckpt_path: Path, board_size: int = 128, p_alive: float = 0.5, seed: int = 0) -> None:
    """Predict entire next board via patch model and compare to true GoL step."""
    device = torch.device("cpu")
    model = MLP(input_dim=9, hidden_dims=[64, 64], dropout=0.0).to(device)
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()

    rng = np.random.default_rng(seed)
    board = random_board(board_size, p_alive, rng=rng).astype(np.uint8)
    true_next = gol_step(board)
    padded = np.pad(board, 1, mode="wrap")
    patches = []
    for i in range(board_size):
        for j in range(board_size):
            patches.append(padded[i:i+3, j:j+3].flatten())
    patches = torch.from_numpy(np.stack(patches).astype(np.float32)).to(device)
    with torch.no_grad():
        logits = model(patches)
        pred = (torch.sigmoid(logits) > 0.5).float().cpu().numpy().astype(np.uint8).reshape(board_size, board_size)
    acc = (pred == true_next).mean()
    print(f"Board-level check (size={board_size}, p_alive={p_alive}): acc vs true GoL next = {acc:.4f}, "
          f"pred density={pred.mean():.4f}, true density={true_next.mean():.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train/evaluate 3x3 dynamics patch model.")
    parser.add_argument("--npz_path", type=Path, default=Path("data/dynamics_patch3.npz"))
    parser.add_argument("--generate", action="store_true", help="Regenerate dataset before training.")
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--extra_npz", type=str, nargs="*", default=[],
                        help="Optional extra npz paths to evaluate after training.")
    parser.add_argument("--board_regimes", type=str, nargs="*", default=["early", "mid", "late"])
    parser.add_argument("--board_densities", type=float, nargs="*", default=[0.2, 0.4, 0.6])
    parser.add_argument("--board_steps", type=int, default=10)
    parser.add_argument("--board_size", type=int, default=128)
    parser.add_argument("--board_seed", type=int, default=0)
    args = parser.parse_args()

    if args.generate or not args.npz_path.exists():
        generate_dynamics_dataset(out_path=args.npz_path)
    ckpt_path = train_dynamics(npz_path=args.npz_path, max_epochs=args.max_epochs, patience=args.patience)

    # Extra evaluations
    for npz_str in args.extra_npz:
        npz_path = Path(npz_str)
        if not npz_path.exists():
            print(f"Extra npz not found: {npz_path}")
            continue
        acc, prec, rec, f1 = eval_on_npz(npz_path, ckpt_path)
        print(f"Extra eval on {npz_path.name}: acc={acc:.4f} prec={prec:.4f} rec={rec:.4f} f1={f1:.4f}")

    # Board-level rollout checks for given regimes/densities (using standard burn-ins)
    burn_map = {"early": 10, "mid": 60, "late": 160}
    for regime in args.board_regimes:
        if regime not in burn_map:
            continue
        burn = burn_map[regime]
        for d in args.board_densities:
            accs, dens = board_rollout_eval(
                ckpt_path,
                density=d,
                burn_in=burn,
                steps=args.board_steps,
                board_size=args.board_size,
                seed=args.board_seed,
            )
            print(f"Board rollout ({regime}, p={d}, burn={burn}): acc@1={accs[0]:.4f} acc@{args.board_steps}={accs[-1]:.4f} "
                  f"avg_acc={np.mean(accs):.4f} dens_last={dens[-1]:.4f}")
