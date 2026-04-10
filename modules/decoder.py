import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8 if out_ch >= 8 else 1, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8 if out_ch >= 8 else 1, out_ch),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Decoder(nn.Module):
    """Lightweight decoder from 7x7 latent map back to 28x28 image."""

    def __init__(self, out_channels: int = 1, latent_dim: int = 8, base_channels: int = 32):
        super().__init__()
        self.stem = nn.Conv2d(latent_dim, base_channels * 2, kernel_size=3, stride=1, padding=1)
        self.block1 = ConvBlock(base_channels * 2, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1)  # 7->14
        self.block2 = ConvBlock(base_channels, base_channels)
        self.up2 = nn.ConvTranspose2d(base_channels, base_channels, kernel_size=4, stride=2, padding=1)  # 14->28
        self.head = nn.Sequential(
            nn.Conv2d(base_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = self.stem(z)
        z = self.block1(z)
        z = self.up1(z)
        z = self.block2(z)
        z = self.up2(z)
        return self.head(z)
