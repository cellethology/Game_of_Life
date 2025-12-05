"""MLP models for Conway's Game of Life prediction."""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Multi-layer perceptron for binary classification."""

    def __init__(self, input_dim: int, hidden_dims: list[int] = [128, 128], dropout: float = 0.0):
        super().__init__()
        layers = []
        prev_dim = input_dim

        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h

        layers.append(nn.Linear(prev_dim, 1))  # binary classification (logit)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch_size, input_dim) tensor of input features

        Returns:
            (batch_size,) tensor of logits
        """
        return self.net(x).squeeze(-1)  # logits