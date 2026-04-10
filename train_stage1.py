import argparse
import os
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid

from models.vae import VAE
from models.rankae import RankAE


@dataclass
class Stage1Config:
    tokenizer: str
    data_root: str
    save_dir: str
    epochs: int
    batch_size: int
    lr: float
    latent_dim: int
    device: str


def build_model(cfg: Stage1Config):
    if cfg.tokenizer == "vae":
        return VAE(in_channels=1, latent_dim=cfg.latent_dim)
    if cfg.tokenizer == "rankae":
        return RankAE(in_channels=1, latent_dim=cfg.latent_dim)
    raise ValueError(f"Unsupported tokenizer: {cfg.tokenizer}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str, default="vae", choices=["vae", "rankae"])
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

    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    ds = datasets.MNIST(root=cfg.data_root, train=True, download=True, transform=tfm)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=2, pin_memory=True)

    model = build_model(cfg).to(cfg.device)
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    fixed, _ = next(iter(dl))
    fixed = fixed[:64].to(cfg.device)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0
        for x, _ in dl:
            x = x.to(cfg.device)
            optim.zero_grad(set_to_none=True)

            if cfg.tokenizer == "vae":
                loss, stats = model(x)
            else:
                loss = model(x)
                stats = {"loss": loss.item()}

            loss.backward()
            optim.step()
            running += stats["loss"]

        avg = running / len(dl)
        print(f"[Stage1][{cfg.tokenizer}] epoch={epoch}/{cfg.epochs} loss={avg:.6f}")

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
            "latent_dim": cfg.latent_dim,
            "model": model.state_dict(),
            "epoch": epoch,
        }
        torch.save(ckpt, os.path.join(cfg.save_dir, f"{cfg.tokenizer}_mnist_ep{epoch:03d}.pt"))

    torch.save(
        {
            "tokenizer": cfg.tokenizer,
            "latent_dim": cfg.latent_dim,
            "model": model.state_dict(),
            "epoch": cfg.epochs,
        },
        os.path.join(cfg.save_dir, f"{cfg.tokenizer}_mnist_last.pt"),
    )


if __name__ == "__main__":
    main()
