"""
Unified training for quantile regression — supports both ReLU and ReQU activations.

Provides a single entry point ``train_quantile_models_unified`` that uses the
same training protocol (mini-batching, gradient clipping, weight decay) for
both activation families, so that comparisons are fair.

The original per-activation trainers in ``models.py`` and ``models_requ.py``
are preserved for backward compatibility.
"""

import torch
import torch.nn as nn
import torch.optim as optim

from .models import QuantileNN, quantile_loss
from .models_requ import QuantileReQUNN


_VALID_ACTIVATIONS = {"relu", "requ"}


def _make_model(activation: str, input_dim: int, hidden_dim: int, n_layers: int) -> nn.Module:
    """Instantiate a quantile network with the requested activation."""
    act = activation.lower().strip()
    if act not in _VALID_ACTIVATIONS:
        raise ValueError(
            f"Unknown activation '{activation}'. Choose from {sorted(_VALID_ACTIVATIONS)}."
        )
    if act == "relu":
        return QuantileNN(input_dim=input_dim, hidden_dim=hidden_dim, n_layers=n_layers)
    else:
        return QuantileReQUNN(input_dim=input_dim, hidden_dim=hidden_dim, n_layers=n_layers)


def train_quantile_models_unified(
    X_train: torch.Tensor,
    Y_train: torch.Tensor,
    tau_low: float,
    tau_high: float,
    input_dim: int = 1,
    hidden_dim: int = 64,
    n_layers: int = 2,
    epochs: int = 300,
    lr: float = 0.01,
    batch_size: int = 0,
    weight_decay: float = 1e-5,
    grad_clip: float = 1.0,
    activation: str = "relu",
    verbose: bool = False,
    seed: int = 0,
) -> tuple:
    """
    Train two quantile regression networks (lower & upper) with a unified
    training loop.

    The same protocol — mini-batching, gradient clipping, weight decay — is
    applied regardless of the activation, so comparisons between ReLU and
    ReQU are fair.

    Args:
        X_train:      Training features, shape (n, d).
        Y_train:      Training targets,  shape (n, 1).
        tau_low:      Lower quantile level  (e.g. 0.025).
        tau_high:     Upper quantile level  (e.g. 0.975).
        input_dim:    Feature dimension *d*.
        hidden_dim:   Width of hidden layers.
        n_layers:     Number of hidden layers.
        epochs:       Number of training epochs.
        lr:           Learning rate for Adam.
        batch_size:   Mini-batch size (0 → full batch).
        weight_decay: L2 regularisation strength.
        grad_clip:    Max gradient norm for clipping (0 → no clipping).
        activation:   ``"relu"`` or ``"requ"``.
        verbose:      Print training progress every 50 epochs.
        seed:         Random seed for reproducible initialisation & batching.

    Returns:
        ``(model_lo, model_hi)`` — trained models for the two quantiles.
    """
    # Seed for reproducibility (same seed → same init + batch order)
    torch.manual_seed(seed)
    model_lo = _make_model(activation, input_dim, hidden_dim, n_layers)
    model_hi = _make_model(activation, input_dim, hidden_dim, n_layers)

    opt_lo = optim.Adam(model_lo.parameters(), lr=lr, weight_decay=weight_decay)
    opt_hi = optim.Adam(model_hi.parameters(), lr=lr, weight_decay=weight_decay)

    n = X_train.shape[0]
    use_batches = batch_size > 0 and batch_size < n
    do_clip = grad_clip > 0

    for epoch in range(epochs):
        if use_batches:
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
                if do_clip:
                    nn.utils.clip_grad_norm_(model_lo.parameters(), grad_clip)
                opt_lo.step()

                opt_hi.zero_grad()
                loss_hi = quantile_loss(model_hi(X_batch), Y_batch, tau_high)
                loss_hi.backward()
                if do_clip:
                    nn.utils.clip_grad_norm_(model_hi.parameters(), grad_clip)
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
            if do_clip:
                nn.utils.clip_grad_norm_(model_lo.parameters(), grad_clip)
            opt_lo.step()

            opt_hi.zero_grad()
            loss_hi = quantile_loss(model_hi(X_train), Y_train, tau_high)
            loss_hi.backward()
            if do_clip:
                nn.utils.clip_grad_norm_(model_hi.parameters(), grad_clip)
            opt_hi.step()

            if verbose and (epoch == 0 or (epoch + 1) % 50 == 0):
                print(
                    f"Epoch {epoch + 1}/{epochs}: "
                    f"loss_lo={loss_lo.item():.4f}, "
                    f"loss_hi={loss_hi.item():.4f}"
                )

    return model_lo, model_hi
