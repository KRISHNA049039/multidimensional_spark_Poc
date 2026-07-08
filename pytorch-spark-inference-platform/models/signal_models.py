"""
Additional EW/Signal Processing Models:
- Signal Denoiser (Autoencoder)
- Threat Prioritizer (Multi-head attention)
- RF Fingerprinter (1D-CNN)
- Anomaly Detector (Variational Autoencoder)

All operate on 128-dim IQ signal vectors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

INPUT_DIM = 128


class SignalDenoiser(nn.Module):
    """
    Autoencoder for signal denoising/preprocessing.
    Input: 128-dim noisy IQ vector
    Output: 128-dim denoised IQ vector
    Params: ~1.2M, ~100MB GPU
    """

    def __init__(self, input_dim=INPUT_DIM, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, latent_dim), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, input_dim), nn.Tanh(),
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed


class ThreatPrioritizer(nn.Module):
    """
    Multi-head attention model for threat priority scoring.
    Input: 128-dim signal features
    Output: scalar priority score [0, 1]
    Params: ~8M, ~350MB GPU
    """

    def __init__(self, input_dim=INPUT_DIM, num_heads=4, hidden_dim=512):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        # Project input to hidden dim
        h = self.input_proj(x).unsqueeze(1)  # (N, 1, hidden)
        # Self-attention (treating features as sequence of 1)
        attn_out, _ = self.attention(h, h, h)
        attn_out = attn_out.squeeze(1)  # (N, hidden)
        priority = self.fc(attn_out)
        return priority.squeeze(-1)


class RFFingerprinter(nn.Module):
    """
    1D-CNN for RF emitter fingerprinting/identification.
    Input: 128-dim IQ signal
    Output: 32-dim embedding (for emitter matching)
    Params: ~2.5M, ~120MB GPU
    """

    def __init__(self, input_dim=INPUT_DIM, embedding_dim=32):
        super().__init__()
        # Treat 128-dim as 1D signal with 1 channel
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3), nn.ReLU(), nn.BatchNorm1d(32),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.ReLU(), nn.BatchNorm1d(64),
            nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm1d(128),
            nn.AdaptiveAvgPool1d(8),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 8, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, embedding_dim),
        )

    def forward(self, x):
        # Reshape (N, 128) → (N, 1, 128) for Conv1d
        x = x.unsqueeze(1)
        features = self.conv_layers(x)
        features = features.view(features.size(0), -1)
        embedding = self.fc(features)
        return F.normalize(embedding, p=2, dim=1)


class AnomalyDetector(nn.Module):
    """
    Variational Autoencoder for unknown/anomalous signal detection.
    Input: 128-dim IQ signal
    Output: scalar anomaly score (higher = more anomalous)
    Params: ~1.8M, ~100MB GPU
    """

    def __init__(self, input_dim=INPUT_DIM, latent_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
        )
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, input_dim),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        # Anomaly score = reconstruction error
        recon_error = F.mse_loss(recon, x, reduction='none').sum(dim=1)
        return recon_error
