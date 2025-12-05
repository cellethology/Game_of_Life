"""
Simple test to compare weight strategies.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

try:
    from .data import LifePatchDataset
    from .models import MLP
    from .confusion_matrix_utils import compute_confusion_entries
except ImportError:
    from data import LifePatchDataset
    from models import MLP
    from confusion_matrix_utils import compute_confusion_entries

# Quick test on different densities
datasets = [
    ("p0.2", "data/life_patches_densityp0.2_seed0.npz"),
    ("p0.4", "data/life_patches_densityp0.4_seed0.npz"),
    ("p0.6", "data/life_patches_densityp0.6_seed0.npz"),
]

for density_name, dataset_path in datasets:
    print(f"\n{'='*50}")
    print(f"Testing on {density_name} density")
    print(f"Dataset: {dataset_path}")
    print(f"{'='*50}")

    try:
        dataset = LifePatchDataset(dataset_path, split='train', patch_size=3)
        labels = dataset.y.numpy()
        N_pos = np.sum(labels == 1)
        N_neg = np.sum(labels == 0)
        total = len(labels)
        r = N_neg / max(N_pos, 1)

        print(f"Dataset: N_pos={N_pos}, N_neg={N_neg}, r={r:.2f}, sqrt(r)={np.sqrt(r):.2f}")

        # Test original weight (min(5.0, sqrt(r)))
        pos_weight = min(5.0, np.sqrt(r))
        print(f"Original weight: {pos_weight:.4f}")

        train_dataset = LifePatchDataset(dataset_path, split='train', patch_size=3)
        test_dataset = LifePatchDataset(dataset_path, split='test', patch_size=3)
        train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MLP(input_dim=8, hidden_dims=[128, 128], dropout=0.0).to(device)
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight], dtype=torch.float32).to(device)
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Quick training
        model.train()
        for epoch in range(5):
            for features, labels in train_loader:
                features, labels = features.to(device), labels.float().to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Test
        model.eval()
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for features, labels in test_loader:
                features, labels = features.to(device), labels.float().to(device)
                outputs = model(features)
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                all_predictions.append(outputs.cpu())
                all_labels.append(labels.cpu())

        all_predictions_tensor = torch.cat(all_predictions, dim=0)
        all_labels_tensor = torch.cat(all_labels, dim=0)
        tn, fp, fn, tp = compute_confusion_entries(all_labels_tensor, all_predictions_tensor)

        # Metrics
        total = tn + fp + fn + tp
        accuracy = (tn + tp) / total
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        print(f"Results: Acc={accuracy:.4f}, Prec={precision:.4f}, Rec={recall:.4f}")
        print(f"Confusion: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

print(f"\n{'='*50}")
print("Testing completed!")