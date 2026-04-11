import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(0, half, device=timesteps.device).float() / max(half - 1, 1))
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, tdim: int):
        super().__init__()
        self.in_layers = nn.Sequential(
            nn.GroupNorm(8 if in_ch >= 8 else 1, in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
        )
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(tdim, out_ch),
        )
        self.out_layers = nn.Sequential(
            nn.GroupNorm(8 if out_ch >= 8 else 1, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.in_layers(x)
        h = h + self.emb_layers(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = self.out_layers(h)
        return h + self.skip(x)


class TinyUNet(nn.Module):
    def __init__(self, in_channels: int = 8, model_channels: int = 64, out_channels: int = 8, time_embed_dim: int = 128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        self.in_conv = nn.Conv2d(in_channels, model_channels, 3, padding=1)
        self.down1 = ResBlock(model_channels, model_channels, time_embed_dim)
        self.downsample = nn.Conv2d(model_channels, model_channels * 2, 3, stride=2, padding=1)
        self.mid = ResBlock(model_channels * 2, model_channels * 2, time_embed_dim)
        self.up = nn.ConvTranspose2d(model_channels * 2, model_channels, 4, stride=2, padding=1)
        self.up1 = ResBlock(model_channels * 2, model_channels, time_embed_dim)
        self.out = nn.Sequential(
            nn.GroupNorm(8 if model_channels >= 8 else 1, model_channels),
            nn.SiLU(),
            nn.Conv2d(model_channels, out_channels, 3, padding=1),
        )
        self.time_embed_dim = time_embed_dim

    def forward(self, x: torch.Tensor, t: torch.Tensor, context=None, y=None) -> torch.Tensor:
        del context, y
        t_emb = timestep_embedding(t, self.time_embed_dim)
        t_emb = self.time_mlp(t_emb)

        h0 = self.in_conv(x)
        h1 = self.down1(h0, t_emb)
        h2 = self.downsample(h1)
        h3 = self.mid(h2, t_emb)
        hu = self.up(h3)

        if hu.shape[-2:] != h1.shape[-2:]:
            hu = F.interpolate(hu, size=h1.shape[-2:], mode="nearest")

        h = torch.cat([hu, h1], dim=1)
        h = self.up1(h, t_emb)
        return self.out(h)
