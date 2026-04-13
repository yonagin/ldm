import argparse
import os

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid

from dataset_utils import build_dataset, supported_datasets, unpack_batch
from models.vae import VAE
from models.rankae import RankAE
from models.vqvae import VQVAE


def load_tokenizer(path: str, device: str):
    ckpt = torch.load(path, map_location=device)

    tokenizer_type = ckpt["tokenizer"]
    latent_dim = ckpt["latent_dim"]
    in_channels = ckpt.get("in_channels", 1)

    if tokenizer_type == "vae":
        model = VAE(in_channels=in_channels, latent_dim=latent_dim)
    elif tokenizer_type == "rankae":
        model = RankAE(in_channels=in_channels, latent_dim=latent_dim)
    elif tokenizer_type == "vqvae":
        model = VQVAE(in_channels=in_channels, latent_dim=latent_dim)
    else:
        raise ValueError(f"Unsupported tokenizer: {tokenizer_type}")

    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    return model, tokenizer_type, in_channels


@torch.no_grad()
def reconstruct(tokenizer, tokenizer_type, x):
    if tokenizer_type == "vae":
        z = tokenizer.encode(x).mode()
    else:
        z = tokenizer.encode(x)

    rec = tokenizer.decode(z)
    return rec


def main():
    parser = argparse.ArgumentParser()

    # tokenizer
    parser.add_argument("--tokenizer-ckpt", type=str, required=True)

    # dataset
    parser.add_argument("--dataset", type=str, default="mnist", choices=supported_datasets())
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--img-size", type=int, default=32)
    parser.add_argument("--id", type=str, default=None)

    # runtime
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--out", type=str, default="./recon.png")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    device = args.device

    # === load tokenizer ===
    tokenizer, tokenizer_type, in_channels = load_tokenizer(
        args.tokenizer_ckpt, device
    )

    # === build dataset ===
    ds, _ = build_dataset(
        args.dataset,
        root=args.data_root,
        train=False,   # 测试集更合理
        img_size=args.img_size,
        id=args.id,
    )

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # === get one batch ===
    batch = next(iter(dl))
    x, _ = unpack_batch(batch)
    x = x.to(device)

    # === reconstruct ===
    rec = reconstruct(tokenizer, tokenizer_type, x)

    # === normalize to [0,1] ===
    x_vis = (x.clamp(-1, 1) + 1.0) / 2.0
    rec_vis = (rec.clamp(-1, 1) + 1.0) / 2.0

    vis = torch.cat([x_vis, rec_vis], dim=0)

    grid = make_grid(vis, nrow=int(args.batch_size ** 0.5))

    save_image(grid, args.out)

    print(f"Saved reconstruction grid to: {args.out}")
    print(f"Tokenizer: {tokenizer_type}")


if __name__ == "__main__":
    main()