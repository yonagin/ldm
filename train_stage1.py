import argparse
import os
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid

from dataset_utils import build_dataset, supported_datasets, unpack_batch
from models.vae import VAE
from models.rankae import RankAE
from models.vqvae import VQVAE


@dataclass
class Stage1Config:
    tokenizer: str
    dataset: str
    id: str
    img_size: int
    data_root: str
    save_dir: str
    epochs: int
    batch_size: int
    lr: float
    latent_dim: int
    device: str


def build_model(cfg: Stage1Config, in_channels: int):
    if cfg.tokenizer == "vae":
        return VAE(in_channels=in_channels, latent_dim=cfg.latent_dim)
    if cfg.tokenizer == "rankae":
        return RankAE(in_channels=in_channels, latent_dim=cfg.latent_dim)
    if cfg.tokenizer == "vqvae":
        return VQVAE(in_channels=in_channels, latent_dim=cfg.latent_dim)
    raise ValueError(f"Unsupported tokenizer: {cfg.tokenizer}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str, default="vae", choices=["vae", "rankae", "vqvae"])
    parser.add_argument("--dataset", type=str, default="mnist", choices=supported_datasets())
    parser.add_argument("--id", type=str, default=None)
    parser.add_argument("--img-size", type=int, default=32)
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--save-dir", type=str, default="./checkpoints/stage1")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--latent-dim", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = Stage1Config(**vars(args))
    os.makedirs(cfg.save_dir, exist_ok=True)

    ds, in_channels = build_dataset(
        cfg.dataset,
        root=cfg.data_root,
        train=True,
        img_size=cfg.img_size,
        id=cfg.id,
    )
    dl = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        
    )

    model = build_model(cfg, in_channels=in_channels).to(cfg.device)
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    fixed, _ = unpack_batch(next(iter(dl)))
    fixed = fixed[:64].to(cfg.device)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0
        for batch in dl:
            x, _ = unpack_batch(batch)
            x = x.to(cfg.device)
            optim.zero_grad(set_to_none=True)

            if cfg.tokenizer in ["vae", "vqvae"]:
                loss, stats = model(x)
            else:
                loss = model(x)
                stats = {"loss": loss.item()}

            loss.backward()
            optim.step()
            running += stats["loss"]

        avg = running / len(dl)
        print(f"[Stage1][{cfg.tokenizer}] dataset={cfg.dataset} epoch={epoch}/{cfg.epochs} loss={avg:.6f}")

        model.eval()
        with torch.no_grad():
            if cfg.tokenizer == "vae":
                z = model.encode(fixed).mode()
                rec = model.decode(z)
            else:
                z = model.encode(fixed)
                rec = model.decode(z)

            rec_vis = (rec.clamp(-1, 1) + 1.0) / 2.0
            grid = make_grid(rec_vis, nrow=8)
            save_image(grid, os.path.join(cfg.save_dir, f"recon_epoch_{epoch:03d}.png"))

        ckpt = {
            "tokenizer": cfg.tokenizer,
            "dataset": cfg.dataset,
            "id": cfg.id,
            "img_size": cfg.img_size,
            "in_channels": in_channels,
            "latent_dim": cfg.latent_dim,
            "model": model.state_dict(),
            "epoch": epoch,
        }
        torch.save(ckpt, os.path.join(cfg.save_dir, f"{cfg.tokenizer}_{cfg.dataset}_ep{epoch:03d}.pt"))

    torch.save(
        {
            "tokenizer": cfg.tokenizer,
            "dataset": cfg.dataset,
            "id": cfg.id,
            "img_size": cfg.img_size,
            "in_channels": in_channels,
            "latent_dim": cfg.latent_dim,
            "model": model.state_dict(),
            "epoch": cfg.epochs,
        },
        os.path.join(cfg.save_dir, f"{cfg.tokenizer}_{cfg.dataset}_last.pt"),
    )


if __name__ == "__main__":
    main()
