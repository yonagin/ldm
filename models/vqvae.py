import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.cnn_models import Encoder, Decoder
from modules.quantize import VectorQuantizer


class VQVAE(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        latent_dim: int = 8,
        num_embeddings: int = 65536,
        beta: float = 0.25,
    ):
        super().__init__()
        self.encoder = Encoder(in_channels, z_channels=latent_dim)
        self.quantize = VectorQuantizer(n_e=num_embeddings, e_dim=latent_dim, beta=beta)
        self.decoder = Decoder(in_channels, z_channels=latent_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        z_q, _, _ = self.quantize(h)
        return z_q

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z_q, _, _ = self.quantize(z)
        return self.decoder(z_q)

    def forward(self, x: torch.Tensor):
        h = self.encoder(x)
        z_q, q_loss, info = self.quantize(h)
        x_rec = self.decoder(z_q)

        rec_loss = F.mse_loss(x_rec, x)
        loss = rec_loss + q_loss
        stats = {
            "loss": loss.item(),
            "rec": rec_loss.item(),
            "q": q_loss.item(),
            "perplexity": float(info["perplexity"].item()),
        }
        return loss, stats
