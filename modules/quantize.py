import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    def __init__(self, n_e: int, e_dim: int, beta: float = 0.25):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        self.embedding = nn.Embedding(n_e, e_dim)
        nn.init.normal_(self.embedding.weight, mean=0, std=self.e_dim**-0.5)

    def forward(self, z: torch.Tensor):
        # z: [B, C, H, W]
        if z.ndim != 4:
            raise ValueError(f"VectorQuantizer expects 4D input [B,C,H,W], got shape={tuple(z.shape)}")

        z_perm = z.permute(0, 2, 3, 1).contiguous()
        z_flat = z_perm.view(-1, self.e_dim)

        emb = self.embedding.weight
        distances = (
            z_flat.pow(2).sum(dim=1, keepdim=True)
            + emb.pow(2).sum(dim=1)
            - 2 * z_flat @ emb.t()
        )

        indices = torch.argmin(distances, dim=1)
        z_q_flat = self.embedding(indices)
        z_q = z_q_flat.view_as(z_perm)

        codebook_loss = F.mse_loss(z_q, z_perm.detach())
        commit_loss = F.mse_loss(z_q.detach(), z_perm)
        q_loss = codebook_loss + self.beta * commit_loss

        z_q = z_perm + (z_q - z_perm).detach()
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        one_hot = F.one_hot(indices, num_classes=self.n_e).float()
        avg_probs = one_hot.mean(dim=0)
        perplexity = torch.exp(-(avg_probs * torch.log(avg_probs + 1e-10)).sum())

        info = {
            "indices": indices.view(z.shape[0], z.shape[2], z.shape[3]),
            "perplexity": perplexity,
        }
        return z_q, q_loss, info
