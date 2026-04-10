import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.encoder import Encoder
from modules.decoder import Decoder


class DiagonalGaussianDistribution:
    def __init__(self, moments: torch.Tensor):
        self.mean, self.logvar = torch.chunk(moments, 2, dim=1)
        self.logvar = self.logvar.clamp(-30.0, 20.0)

    def sample(self) -> torch.Tensor:
        std = torch.exp(0.5 * self.logvar)
        return self.mean + std * torch.randn_like(std)

    def mode(self) -> torch.Tensor:
        return self.mean

    def kl(self) -> torch.Tensor:
        return 0.5 * torch.sum(torch.exp(self.logvar) + self.mean ** 2 - 1.0 - self.logvar, dim=1)


class VAE(nn.Module):
    """Light VAE tokenizer for MNIST."""

    def __init__(self, in_channels: int = 1, latent_dim: int = 8, base_channels: int = 32, kl_weight: float = 1e-4):
        super().__init__()
        self.encoder = Encoder(in_channels=in_channels, latent_dim=base_channels, base_channels=base_channels)
        self.to_moments = nn.Conv2d(base_channels, latent_dim * 2, kernel_size=1)
        self.decoder = Decoder(out_channels=in_channels, latent_dim=latent_dim, base_channels=base_channels)
        self.kl_weight = kl_weight

    def encode(self, x: torch.Tensor) -> DiagonalGaussianDistribution:
        h = self.encoder(x)
        moments = self.to_moments(h)
        return DiagonalGaussianDistribution(moments)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        posterior = self.encode(x)
        z = posterior.sample()
        x_rec = self.decode(z)

        rec_loss = F.mse_loss(x_rec, x)
        kl_loss = posterior.kl().mean()
        loss = rec_loss + self.kl_weight * kl_loss
        stats = {"loss": loss.item(), "rec": rec_loss.item(), "kl": kl_loss.item()}
        return loss, stats
