"""
Neural network models for quantile regression using ReQU activation.

ReQU (Rectified Quadratic Unit): f(x) = max(0, x)^2
This is a smooth activation that is particularly well-suited for
quantile regression — it is C^1 smooth, which can lead to smoother
quantile estimates compared to ReLU.
"""

import torch
import torch.nn as nn
from .models import quantile_loss


class ReQU(nn.Module):
    """
    Rectified Quadratic Unit activation: f(x) = max(0, x)^2.

    Properties:
        - C^1 smooth (unlike ReLU which has a kink at 0)
        - Non-negative output
        - Zero gradient for x < 0, gradient = 2x for x > 0
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x) ** 2


class QuantileReQUNN(nn.Module):
    """
    Feedforward neural network for quantile estimation using ReQU activations.

    Architecture: input -> [Linear -> ReQU] x n_layers -> Linear -> output

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
        layers.append(ReQU())

        # Hidden layers
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(ReQU())

        # Output layer
        layers.append(nn.Linear(hidden_dim, 1))

        self.net = nn.Sequential(*layers)

        # Initialize weights with smaller values since ReQU squares the output
        self._init_weights()

    def _init_weights(self):
        """Xavier init scaled down for ReQU to prevent output explosion."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_requ_quantile_models(
    X_train: torch.Tensor,
    Y_train: torch.Tensor,
    tau_low: float,
    tau_high: float,
    input_dim: int = 1,
    hidden_dim: int = 64,
    n_layers: int = 2,
    epochs: int = 300,
    lr: float = 0.001,
    batch_size: int = 0,
    weight_decay: float = 1e-5,
    verbose: bool = False,
    grad_clip: float = 1.0,
) -> tuple:
    """
    Train two ReQU quantile regression networks for lower and upper quantiles.

    Includes mini-batch training, gradient clipping, and weight decay —
    important for ReQU since the squared activation can amplify gradients.

    Args:
        X_train: Training features, shape (n, d)
        Y_train: Training targets, shape (n, 1)
        tau_low: Lower quantile level (e.g., 0.025 or 0.05)
        tau_high: Upper quantile level (e.g., 0.975 or 0.95)
        input_dim: Feature dimension
        hidden_dim: Hidden layer width
        n_layers: Number of hidden layers
        epochs: Number of training epochs
        lr: Learning rate (smaller than ReLU due to ReQU squaring)
        batch_size: Mini-batch size (0 = full batch)
        weight_decay: L2 regularization strength
        verbose: Print training progress
        grad_clip: Max gradient norm for clipping

    Returns:
        (model_lo, model_hi): Trained models for lower and upper quantiles
    """
    import torch.optim as optim

    model_lo = QuantileReQUNN(input_dim=input_dim, hidden_dim=hidden_dim, n_layers=n_layers)
    model_hi = QuantileReQUNN(input_dim=input_dim, hidden_dim=hidden_dim, n_layers=n_layers)

    opt_lo = optim.Adam(model_lo.parameters(), lr=lr, weight_decay=weight_decay)
    opt_hi = optim.Adam(model_hi.parameters(), lr=lr, weight_decay=weight_decay)

    n = X_train.shape[0]
    use_batches = batch_size > 0 and batch_size < n

    for epoch in range(epochs):
        if use_batches:
            # Shuffle indices each epoch
            perm = torch.randperm(n)
            epoch_loss_lo = 0.0
            epoch_loss_hi = 0.0
            n_batches = 0

            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                idx = perm[start:end]
                X_batch = X_train[idx]
                Y_batch = Y_train[idx]

                opt_lo.zero_grad()
                loss_lo = quantile_loss(model_lo(X_batch), Y_batch, tau_low)
                loss_lo.backward()
                torch.nn.utils.clip_grad_norm_(model_lo.parameters(), grad_clip)
                opt_lo.step()

                opt_hi.zero_grad()
                loss_hi = quantile_loss(model_hi(X_batch), Y_batch, tau_high)
                loss_hi.backward()
                torch.nn.utils.clip_grad_norm_(model_hi.parameters(), grad_clip)
                opt_hi.step()

                epoch_loss_lo += loss_lo.item()
                epoch_loss_hi += loss_hi.item()
                n_batches += 1

            if verbose and (epoch == 0 or (epoch + 1) % 50 == 0):
                print(
                    f"Epoch {epoch + 1}/{epochs}: "
                    f"loss_lo={epoch_loss_lo / n_batches:.4f}, "
                    f"loss_hi={epoch_loss_hi / n_batches:.4f}"
                )
        else:
            # Full-batch training
            opt_lo.zero_grad()
            loss_lo = quantile_loss(model_lo(X_train), Y_train, tau_low)
            loss_lo.backward()
            torch.nn.utils.clip_grad_norm_(model_lo.parameters(), grad_clip)
            opt_lo.step()

            opt_hi.zero_grad()
            loss_hi = quantile_loss(model_hi(X_train), Y_train, tau_high)
            loss_hi.backward()
            torch.nn.utils.clip_grad_norm_(model_hi.parameters(), grad_clip)
            opt_hi.step()

            if verbose and (epoch == 0 or (epoch + 1) % 50 == 0):
                print(
                    f"Epoch {epoch + 1}/{epochs}: "
                    f"loss_lo={loss_lo.item():.4f}, "
                    f"loss_hi={loss_hi.item():.4f}"
                )

    return model_lo, model_hi

