"""
Neural network models for quantile regression.
"""

import torch
import torch.nn as nn


class QuantileNN(nn.Module):
    """
    Feedforward neural network for single quantile estimation.

    Architecture: input -> Linear -> ReLU -> Linear -> ReLU -> Linear -> output

    Args:
        input_dim: Input feature dimension (d)
        hidden_dim: Width of hidden layers
        n_layers: Number of hidden layers (default 2)
    """

    def __init__(self, input_dim: int = 1, hidden_dim: int = 64, n_layers: int = 2):
        super().__init__()

        layers = []

        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())

        # Hidden layers
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        # Output layer
        layers.append(nn.Linear(hidden_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def quantile_loss(
    preds: torch.Tensor, target: torch.Tensor, tau: float
) -> torch.Tensor:
    """
    Pinball loss for quantile regression.

    L_τ(y, ŷ) = τ * max(y - ŷ, 0) + (1 - τ) * max(ŷ - y, 0)
             = max(τ * (y - ŷ), (τ - 1) * (y - ŷ))

    Args:
        preds: Model predictions, shape (N, 1)
        target: True values, shape (N, 1)
        tau: Quantile level in (0, 1)

    Returns:
        Mean pinball loss (scalar)
    """
    err = target - preds
    return torch.mean(torch.max(tau * err, (tau - 1) * err))


def train_quantile_models(
    X_train: torch.Tensor,
    Y_train: torch.Tensor,
    tau_low: float,
    tau_high: float,
    input_dim: int = 1,
    hidden_dim: int = 64,
    epochs: int = 300,
    lr: float = 0.01,
    verbose: bool = False,
) -> tuple:
    """
    Train two quantile regression networks for lower and upper quantiles.

    Args:
        X_train: Training features, shape (n, d)
        Y_train: Training targets, shape (n, 1)
        tau_low: Lower quantile level (e.g., 0.025)
        tau_high: Upper quantile level (e.g., 0.975)
        input_dim: Feature dimension
        hidden_dim: Hidden layer width
        epochs: Number of training epochs
        lr: Learning rate
        verbose: Print training progress

    Returns:
        (model_lo, model_hi): Trained models for lower and upper quantiles
    """
    import torch.optim as optim

    model_lo = QuantileNN(input_dim=input_dim, hidden_dim=hidden_dim)
    model_hi = QuantileNN(input_dim=input_dim, hidden_dim=hidden_dim)

    opt_lo = optim.Adam(model_lo.parameters(), lr=lr)
    opt_hi = optim.Adam(model_hi.parameters(), lr=lr)

    for epoch in range(epochs):
        # Train low quantile
        opt_lo.zero_grad()
        loss_lo = quantile_loss(model_lo(X_train), Y_train, tau_low)
        loss_lo.backward()
        opt_lo.step()

        # Train high quantile
        opt_hi.zero_grad()
        loss_hi = quantile_loss(model_hi(X_train), Y_train, tau_high)
        loss_hi.backward()
        opt_hi.step()

        if verbose and (epoch + 1) % 50 == 0:
            print(
                f"Epoch {epoch + 1}/{epochs}: loss_lo={loss_lo.item():.4f}, loss_hi={loss_hi.item():.4f}"
            )

    return model_lo, model_hi
