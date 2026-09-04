"""
Variational Autoencoder (VAE) models for unsupervised anomaly detection
in industrial machinery vibration signals.

References:
  - Kingma & Welling (2013), arXiv 1312.6114
  - Xu et al. (2019), arXiv 1912.01096  (Semi-Supervised VAE)
  - Hendrycks et al. (2020), arXiv 2007.05314  (ID-Conditioned AE)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Base VAE
# ---------------------------------------------------------------------------

class VibrationVAE(nn.Module):
    """Variational Autoencoder trained on normal-only vibration feature vectors.

    Architecture
    ------------
    Encoder : 1024 → 512 → 256 → (mu: 64, log_var: 64)
    Decoder : 64 → 256 → 512 → 1024

    Each hidden layer uses BatchNorm + ReLU.  The decoder output uses Sigmoid
    to keep reconstructions in [0, 1] (assumes input is pre-normalised to that
    range; use MinMaxScaler or StandardScaler + clip before passing data in).

    Loss
    ----
    L = MSE(x, x_hat) + beta * KL(q(z|x) || N(0,I))

    The KL term is computed analytically:
        KL = -0.5 * sum(1 + log_var - mu^2 - exp(log_var))

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input feature vector (default 1024).
    latent_dim : int
        Dimensionality of the latent space (default 64).
    hidden_dims : list[int]
        Hidden layer sizes for encoder (reversed for decoder).
        Default: [512, 256].
    beta : float
        Weight on the KL divergence term (default 1.0).
        Lower = better reconstruction; higher = more regularised latent space.
        Recommended range for anomaly detection: 0.5–2.0.
    dropout : float
        Dropout probability applied after each hidden layer (default 0.1).
    """

    def __init__(
        self,
        input_dim: int = 1024,
        latent_dim: int = 64,
        hidden_dims: Optional[list] = None,
        beta: float = 1.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.beta = beta

        if hidden_dims is None:
            hidden_dims = [512, 256]

        # --- Encoder ---
        encoder_layers: list[nn.Module] = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers += [
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
            ]
            in_dim = h_dim
        self.encoder = nn.Sequential(*encoder_layers)

        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_log_var = nn.Linear(hidden_dims[-1], latent_dim)

        # --- Decoder ---
        decoder_layers: list[nn.Module] = []
        in_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers += [
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
            ]
            in_dim = h_dim
        decoder_layers.append(nn.Linear(hidden_dims[0], input_dim))
        decoder_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder_layers)

    # ------------------------------------------------------------------
    # Forward passes
    # ------------------------------------------------------------------

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (mu, log_var) for the latent distribution q(z|x)."""
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_log_var(h)

    def reparameterize(
        self, mu: torch.Tensor, log_var: torch.Tensor
    ) -> torch.Tensor:
        """Sample z via the reparameterisation trick: z = mu + eps * sigma.

        During evaluation (model.eval()), falls back to the mean for
        deterministic, lower-variance anomaly scores.
        """
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct feature vector from latent sample z."""
        return self.decoder(z)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass.

        Returns
        -------
        x_hat : torch.Tensor  — reconstructed input
        mu    : torch.Tensor  — latent mean
        log_var : torch.Tensor — latent log-variance
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decode(z)
        return x_hat, mu, log_var

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def loss(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute VAE loss = MSE + beta * KL.

        Returns
        -------
        total_loss       : scalar tensor
        recon_loss       : scalar tensor (MSE, un-reduced mean)
        kl_loss          : scalar tensor (KL per sample, mean over batch)
        """
        # Reconstruction: mean MSE over all elements, then mean over batch
        recon_loss = F.mse_loss(x_hat, x, reduction="mean")

        # KL divergence: -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
        kl_per_sample = -0.5 * torch.sum(
            1 + log_var - mu.pow(2) - log_var.exp(), dim=1
        )
        kl_loss = kl_per_sample.mean()

        total_loss = recon_loss + self.beta * kl_loss
        return total_loss, recon_loss, kl_loss

    # ------------------------------------------------------------------
    # Anomaly scoring
    # ------------------------------------------------------------------

    @torch.no_grad()
    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """Compute reconstruction error for each sample in x.

        Parameters
        ----------
        x : torch.Tensor, shape (N, input_dim) or (input_dim,)
            Pre-processed feature vector(s), values in [0, 1].

        Returns
        -------
        scores : torch.Tensor, shape (N,)
            Per-sample MSE reconstruction errors.  Higher = more anomalous.
        """
        was_1d = x.ndim == 1
        if was_1d:
            x = x.unsqueeze(0)

        self.eval()
        x_hat, _, _ = self(x)
        # MSE per sample (mean over feature dimensions)
        scores = F.mse_loss(x_hat, x, reduction="none").mean(dim=1)

        return scores.squeeze() if was_1d else scores


# ---------------------------------------------------------------------------
# MachineID-Conditioned VAE
# ---------------------------------------------------------------------------

class MachineIDConditionedVAE(VibrationVAE):
    """VAE conditioned on machine ID for per-unit personalized baselines.

    Adds a learned embedding (``embed_dim``-dimensional) for each machine ID
    that is concatenated to the feature vector before encoding.  This allows
    the model to learn separate normal profiles for each asset, correcting for
    unit-to-unit manufacturing variation, installation differences, and
    long-term drift.

    Reference: arXiv 2007.05314 — *ID-Conditioned Expert AE for Anomaly
    Detection*

    Parameters
    ----------
    num_machines : int
        Total number of distinct machine IDs (vocabulary size).
    embed_dim : int
        Dimensionality of the machine ID embedding (default 128).
    All other parameters inherited from VibrationVAE.

    Usage
    -----
    >>> model = MachineIDConditionedVAE(num_machines=50, embed_dim=128)
    >>> x = torch.randn(32, 1024)            # batch of 32 feature vectors
    >>> machine_ids = torch.randint(0, 50, (32,))
    >>> x_hat, mu, log_var = model(x, machine_ids)
    >>> scores = model.anomaly_score(x, machine_ids)
    """

    def __init__(
        self,
        num_machines: int,
        embed_dim: int = 128,
        input_dim: int = 1024,
        latent_dim: int = 64,
        hidden_dims: Optional[list] = None,
        beta: float = 1.0,
        dropout: float = 0.1,
    ) -> None:
        # The encoder receives [feature_vector || machine_embedding]
        conditioned_input_dim = input_dim + embed_dim
        super().__init__(
            input_dim=conditioned_input_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            beta=beta,
            dropout=dropout,
        )
        # Store original feature dim separately
        self.feature_dim = input_dim
        self.embed_dim = embed_dim
        self.num_machines = num_machines

        self.machine_embedding = nn.Embedding(num_machines, embed_dim)

    def _condition(
        self, x: torch.Tensor, machine_ids: torch.Tensor
    ) -> torch.Tensor:
        """Concatenate machine embedding to feature vector."""
        emb = self.machine_embedding(machine_ids)          # (N, embed_dim)
        return torch.cat([x, emb], dim=1)                  # (N, 1024 + 128)

    def forward(
        self,
        x: torch.Tensor,
        machine_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with machine conditioning.

        Parameters
        ----------
        x           : (N, feature_dim) feature vectors
        machine_ids : (N,) integer IDs in [0, num_machines)

        Returns
        -------
        x_hat     : (N, feature_dim)   reconstructed feature vectors
        mu        : (N, latent_dim)
        log_var   : (N, latent_dim)
        """
        x_cond = self._condition(x, machine_ids)
        x_hat_cond, mu, log_var = super().forward(x_cond)
        # Decoder reconstructs conditioned vector; strip embedding dims
        x_hat = x_hat_cond[:, : self.feature_dim]
        return x_hat, mu, log_var

    def loss(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute loss against original (unconditioned) feature vector."""
        # x is the raw feature vector (not conditioned); compare to x_hat
        return super().loss(x, x_hat, mu, log_var)

    @torch.no_grad()
    def anomaly_score(
        self, x: torch.Tensor, machine_ids: torch.Tensor
    ) -> torch.Tensor:
        """Per-sample reconstruction error conditioned on machine ID."""
        was_1d = x.ndim == 1
        if was_1d:
            x = x.unsqueeze(0)
            machine_ids = machine_ids.unsqueeze(0)

        self.eval()
        x_hat, _, _ = self(x, machine_ids)
        scores = F.mse_loss(x_hat, x, reduction="none").mean(dim=1)

        return scores.squeeze() if was_1d else scores


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("VibrationVAE sanity check...")
    vae = VibrationVAE(input_dim=1024, latent_dim=64, beta=1.0)
    x = torch.rand(8, 1024)
    x_hat, mu, log_var = vae(x)
    total, recon, kl = vae.loss(x, x_hat, mu, log_var)
    scores = vae.anomaly_score(x)
    print(f"  x_hat shape    : {x_hat.shape}")
    print(f"  total loss     : {total.item():.4f}")
    print(f"  recon loss     : {recon.item():.4f}")
    print(f"  kl loss        : {kl.item():.4f}")
    print(f"  anomaly scores : {scores}")

    print("\nMachineIDConditionedVAE sanity check...")
    cond_vae = MachineIDConditionedVAE(num_machines=50, embed_dim=128)
    machine_ids = torch.randint(0, 50, (8,))
    x_hat_c, mu_c, lv_c = cond_vae(x, machine_ids)
    scores_c = cond_vae.anomaly_score(x, machine_ids)
    print(f"  x_hat shape    : {x_hat_c.shape}")
    print(f"  anomaly scores : {scores_c}")
    print("All checks passed.")
