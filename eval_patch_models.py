"""
Evaluate trained patch models (3/5/7) on stored npz test splits.

Outputs test Accuracy / Precision / Recall / F1 for:
- General dataset: data/life_patches.npz with general ckpts best_model_patch{ps}_life_patches_*.pth
- Regime/density datasets: data/life_patches_{regime}_p{density}.npz with ckpts best_model_patch{ps}_{regime}_p{density}_*.pth

This does NOT train or modify data. Requires checkpoints in checkpoints_weighted/.
"""

import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from data import LifePatchDataset
from models import MLP


def load_model(ckpt_path: Path, patch_size: int, device: torch.device) -> torch.nn.Module:
    if patch_size == 3:
        inp = 8
    elif patch_size == 5:
        inp = 24
    elif patch_size == 7:
        inp = 48
    else:
        raise ValueError(f"Unsupported patch_size {patch_size}")
    model = MLP(input_dim=inp, hidden_dims=[128, 128], dropout=0.0).to(device)
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model


def eval_one(npz_path: Path, ckpt_path: Path, patch_size: int, batch_size: int = 4096):
    device = torch.device("cpu")
    ds = LifePatchDataset(str(npz_path), split="test", patch_size=patch_size)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    model = load_model(ckpt_path, patch_size, device)

    total = correct = tp = fp = fn = tn = 0
    with torch.no_grad():
        for x, y in dl:
            x = x.to(device)
            y = y.float().to(device)
            logits = model(x)
            pred = (torch.sigmoid(logits) > 0.5).float()
            correct += (pred == y).sum().item()
            total += y.numel()
            tp += ((pred == 1) & (y == 1)).sum().item()
            tn += ((pred == 0) & (y == 0)).sum().item()
            fp += ((pred == 1) & (y == 0)).sum().item()
            fn += ((pred == 0) & (y == 1)).sum().item()

    acc = correct / total
    prec = tp / (tp + fp) if tp + fp > 0 else 0.0
    rec = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    return acc, prec, rec, f1


def main():
    parser = argparse.ArgumentParser(description="Evaluate patch models (3/5/7) on test splits.")
    parser.add_argument("--patch_sizes", type=int, nargs="*", default=[3, 5, 7])
    parser.add_argument("--regimes", type=str, nargs="*", default=["early", "mid", "late"])
    parser.add_argument("--densities", type=float, nargs="*", default=[0.2, 0.4, 0.6])
    parser.add_argument("--ckpt_dir", type=Path, default=Path("checkpoints_weighted"))
    parser.add_argument("--data_dir", type=Path, default=Path("data"))
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--out_csv", type=Path, default=None, help="Optional path to save metrics as CSV.")
    args = parser.parse_args()

    rows = []
    # General dataset
    npz_general = args.data_dir / "life_patches.npz"
    if npz_general.exists():
        print("General (p=0.5):")
        for ps in args.patch_sizes:
            cands = sorted(args.ckpt_dir.glob(f"best_model_patch{ps}_life_patches_*.pth"),
                           key=lambda p: p.stat().st_mtime)
            if not cands:
                continue
            ck = cands[-1]
            acc, prec, rec, f1 = eval_one(npz_general, ck, ps, args.batch_size)
            print(f"  patch{ps}: acc={acc:.4f} prec={prec:.4f} rec={rec:.4f} f1={f1:.4f} ck={ck.name}")
            rows.append({
                "regime": "general",
                "density": 0.5,
                "patch_size": ps,
                "acc": acc,
                "prec": prec,
                "rec": rec,
                "f1": f1,
                "ckpt": ck.name,
            })
    else:
        print("General dataset not found; skipping.")

    regimes = args.regimes
    densities = args.densities

    for regime in regimes:
        print(f"\n{regime}:")
        for d in densities:
            npz_path = args.data_dir / f"life_patches_{regime}_p{d:.1f}.npz"
            if not npz_path.exists():
                print(f"  p={d:.1f} npz missing")
                continue
            row = [f"p={d:.1f}"]
            for ps in args.patch_sizes:
                cands = sorted(
                    args.ckpt_dir.glob(f"best_model_patch{ps}_{regime}_p{d:.1f}_*.pth"),
                    key=lambda p: p.stat().st_mtime
                )
                if not cands:
                    row.append(f"patch{ps}: n/a")
                    continue
                ck = cands[-1]
                acc, prec, rec, f1 = eval_one(npz_path, ck, ps, args.batch_size)
                row.append(f"patch{ps}: acc={acc:.4f} f1={f1:.4f} prec={prec:.4f} rec={rec:.4f}")
                rows.append({
                    "regime": regime,
                    "density": d,
                    "patch_size": ps,
                    "acc": acc,
                    "prec": prec,
                    "rec": rec,
                    "f1": f1,
                    "ckpt": ck.name,
                })
            print("  ".join(row))

    if args.out_csv and rows:
        import pandas as pd
        df = pd.DataFrame(rows)
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out_csv, index=False)
        print(f"\nSaved metrics to {args.out_csv}")


if __name__ == "__main__":
    main()
