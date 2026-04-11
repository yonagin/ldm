import argparse
import os

import torch
from torch.utils.data import DataLoader

from dataset_utils import build_dataset, unpack_batch
from models.diffusion import DDPM
from models.vae import VAE
from models.rankae import RankAE
from models.vqvae import VQVAE
from modules.unet import UNetModel


def load_tokenizer(path: str, device: str):
    ckpt = torch.load(path, map_location=device)
    tokenizer_type = ckpt["tokenizer"]
    latent_dim = ckpt["latent_dim"]
    in_channels = ckpt.get("in_channels", 1)

    if tokenizer_type == "vae":
        tok = VAE(in_channels=in_channels, latent_dim=latent_dim)
    elif tokenizer_type == "rankae":
        tok = RankAE(in_channels=in_channels, latent_dim=latent_dim)
    elif tokenizer_type == "vqvae":
        tok = VQVAE(in_channels=in_channels, latent_dim=latent_dim)
    else:
        raise ValueError(f"Unsupported tokenizer in checkpoint: {tokenizer_type}")

    tok.load_state_dict(ckpt["model"])
    tok.to(device).eval()
    for p in tok.parameters():
        p.requires_grad_(False)

    return tok, tokenizer_type, latent_dim, in_channels


def encode_with_tokenizer(tokenizer, tokenizer_type: str, x: torch.Tensor):
    if tokenizer_type == "vae":
        return tokenizer.encode(x).mode()
    return tokenizer.encode(x)


@torch.no_grad()
def infer_latent_size(tokenizer, tokenizer_type: str, dataloader, device: str) -> int:
    x, _ = unpack_batch(next(iter(dataloader)))
    x = x.to(device)
    z = encode_with_tokenizer(tokenizer, tokenizer_type, x)
    if z.ndim != 4 or z.shape[-1] != z.shape[-2]:
        raise ValueError(f"Expected square latent map (B,C,H,W), got shape={tuple(z.shape)}")
    return int(z.shape[-1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer-ckpt", type=str, required=True)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--id", type=str, default=None)
    parser.add_argument("--img-size", type=int, default=None)
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--save-dir", type=str, default="./checkpoints/stage2")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--timesteps", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = args.device

    tokenizer, tokenizer_type, latent_dim, _ = load_tokenizer(args.tokenizer_ckpt, device)

    tok_meta = torch.load(args.tokenizer_ckpt, map_location="cpu")
    dataset_name = args.dataset or tok_meta.get("dataset", "mnist")
    id = args.id or tok_meta.get("id")
    img_size = args.img_size or tok_meta.get("img_size", 32)

    ds, _ = build_dataset(
        dataset_name,
        root=args.data_root,
        train=True,
        img_size=img_size,
        id=id,
    )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    latent_size = infer_latent_size(tokenizer, tokenizer_type, dl, device)

    unet = UNetModel(in_channels=latent_dim, out_channels=latent_dim)
    ddpm = DDPM(
        unet=unet,
        timesteps=args.timesteps,
        image_size=latent_size,
        channels=latent_dim,
        parameterization="eps",
        loss_type="l2",
    ).to(device)

    optim = torch.optim.AdamW(ddpm.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        ddpm.train()
        running = 0.0

        for batch in dl:
            x, _ = unpack_batch(batch)
            x = x.to(device)
            with torch.no_grad():
                z = encode_with_tokenizer(tokenizer, tokenizer_type, x)

            optim.zero_grad(set_to_none=True)
            loss, stats = ddpm(z)
            loss.backward()
            optim.step()
            running += stats["loss"]

        avg = running / len(dl)
        print(f"[Stage2][{tokenizer_type}] dataset={dataset_name} epoch={epoch}/{args.epochs} loss={avg:.6f}")

        torch.save(
            {
                "model": ddpm.state_dict(),
                "tokenizer_type": tokenizer_type,
                "dataset": dataset_name,
                "id": id,
                "img_size": img_size,
                "latent_dim": latent_dim,
                "latent_size": latent_size,
                "timesteps": args.timesteps,
                "epoch": epoch,
            },
            os.path.join(args.save_dir, f"ldm_{tokenizer_type}_{dataset_name}_ep{epoch:03d}.pt"),
        )

    torch.save(
        {
            "model": ddpm.state_dict(),
            "tokenizer_type": tokenizer_type,
            "dataset": dataset_name,
            "id": id,
            "img_size": img_size,
            "latent_dim": latent_dim,
            "latent_size": latent_size,
            "timesteps": args.timesteps,
            "epoch": args.epochs,
        },
        os.path.join(args.save_dir, f"ldm_{tokenizer_type}_{dataset_name}_last.pt"),
    )


if __name__ == "__main__":
    main()
