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


class Encoder(nn.Module):
    """Lightweight encoder for 28x28 MNIST -> 7x7 latent map."""

    def __init__(self, in_channels: int = 1, latent_dim: int = 8, base_channels: int = 32):
        super().__init__()
        self.stem = nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1)
        self.block1 = ConvBlock(base_channels, base_channels)
        self.down1 = nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1)  # 28->14
        self.block2 = ConvBlock(base_channels * 2, base_channels * 2)
        self.down2 = nn.Conv2d(base_channels * 2, latent_dim, kernel_size=4, stride=2, padding=1)  # 14->7

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.block1(x)
        x = self.down1(x)
        x = self.block2(x)
        x = self.down2(x)
        return x
