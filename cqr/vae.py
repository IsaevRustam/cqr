"""
VAE-based dimensionality reduction for the LocalConformalOptimizer kernel space.

Trains a small MLP variational autoencoder on the (already-standardized)
training features and exposes the posterior mean as a deterministic, low-d
embedding suitable for kernel weighting.

The training loop mirrors ``cqr/training.py``: torch.manual_seed for
reproducibility, Adam with weight decay, mini-batches via torch.randperm,
optional gradient clipping. Inference returns ``mu(x)`` (no sampling) so
calibration features are run-to-run identical for a given seed.
"""

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class VAE(nn.Module):
    """
    Symmetric MLP variational autoencoder.

    Encoder: input_dim -> [hidden_dim, ReLU] x n_layers -> (mu, logvar)
    Decoder: latent_dim -> [hidden_dim, ReLU] x n_layers -> input_dim
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        latent_dim: int = 3,
        n_layers: int = 2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        enc: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_layers - 1):
            enc += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        self.encoder = nn.Sequential(*enc)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        dec: list[nn.Module] = [nn.Linear(latent_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_layers - 1):
            dec += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        dec += [nn.Linear(hidden_dim, input_dim)]
        self.decoder = nn.Sequential(*dec)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar


def vae_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
) -> torch.Tensor:
    """Mean ELBO loss: MSE reconstruction + beta * KL(N(mu, sigma) || N(0, I))."""
    recon = torch.mean((x_hat - x) ** 2)
    kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
    return recon + beta * kl


def train_vae_encoder(
    X_train: np.ndarray,
    *,
    latent_dim: int = 3,
    hidden_dim: int = 64,
    n_layers: int = 2,
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 256,
    weight_decay: float = 1e-5,
    grad_clip: float = 1.0,
    beta: float = 1.0,
    seed: int = 0,
    verbose: bool = False,
) -> VAE:
    """
    Train a VAE on standardized training features.

    Args:
        X_train: (n, d) numpy array — already standardized upstream.
        latent_dim: Latent (kernel) dimension.
        hidden_dim, n_layers: Encoder/decoder MLP capacity.
        epochs, lr, batch_size, weight_decay, grad_clip: Optimizer settings.
        beta: KL weight (1.0 = standard VAE).
        seed: Random seed for init + batching.
        verbose: Print loss every 50 epochs.

    Returns:
        Trained VAE in eval() mode.
    """
    torch.manual_seed(seed)

    X = torch.from_numpy(np.ascontiguousarray(X_train)).float()
    n, d = X.shape
    k = min(latent_dim, d)

    model = VAE(input_dim=d, hidden_dim=hidden_dim, latent_dim=k, n_layers=n_layers)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    use_batches = batch_size > 0 and batch_size < n
    do_clip = grad_clip > 0

    model.train()
    for epoch in range(epochs):
        if use_batches:
            perm = torch.randperm(n)
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                idx = perm[start:end]
                x_batch = X[idx]
                opt.zero_grad()
                x_hat, mu, logvar = model(x_batch)
                loss = vae_loss(x_batch, x_hat, mu, logvar, beta=beta)
                loss.backward()
                if do_clip:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()
                epoch_loss += loss.item()
                n_batches += 1
            if verbose and (epoch == 0 or (epoch + 1) % 50 == 0):
                print(f"[VAE] epoch {epoch + 1}/{epochs}  loss={epoch_loss / n_batches:.4f}")
        else:
            opt.zero_grad()
            x_hat, mu, logvar = model(X)
            loss = vae_loss(X, x_hat, mu, logvar, beta=beta)
            loss.backward()
            if do_clip:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            if verbose and (epoch == 0 or (epoch + 1) % 50 == 0):
                print(f"[VAE] epoch {epoch + 1}/{epochs}  loss={loss.item():.4f}")

    model.eval()
    return model


def encode_mean(model: VAE, X: np.ndarray) -> np.ndarray:
    """Deterministic encoding: returns posterior mean mu(X) as numpy."""
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(np.ascontiguousarray(X)).float()
        mu, _ = model.encode(x)
    return mu.numpy()
