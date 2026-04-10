import argparse
import os

import torch
from torchvision.utils import save_image, make_grid

from models.diffusion import DDPM
from models.vae import VAE
from models.rankae import RankAE
from modules.unet import SimpleUNet


def load_tokenizer(path: str, device: str):
    ckpt = torch.load(path, map_location=device)
    tokenizer_type = ckpt["tokenizer"]
    latent_dim = ckpt["latent_dim"]

    if tokenizer_type == "vae":
        tok = VAE(in_channels=1, latent_dim=latent_dim)
    elif tokenizer_type == "rankae":
        tok = RankAE(in_channels=1, latent_dim=latent_dim)
    else:
        raise ValueError(f"Unsupported tokenizer: {tokenizer_type}")

    tok.load_state_dict(ckpt["model"])
    tok.to(device).eval()
    return tok, tokenizer_type, latent_dim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer-ckpt", type=str, required=True)
    parser.add_argument("--ldm-ckpt", type=str, required=True)
    parser.add_argument("--out", type=str, default="./samples/generated.png")
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    device = args.device

    tokenizer, tokenizer_type, latent_dim = load_tokenizer(args.tokenizer_ckpt, device)

    ldm_ckpt = torch.load(args.ldm_ckpt, map_location=device)
    timesteps = ldm_ckpt.get("timesteps", 200)

    unet = SimpleUNet(in_channels=latent_dim, out_channels=latent_dim)
    ddpm = DDPM(
        unet=unet,
        timesteps=timesteps,
        image_size=7,
        channels=latent_dim,
        parameterization="eps",
        loss_type="l2",
    ).to(device)
    ddpm.load_state_dict(ldm_ckpt["model"])
    ddpm.eval()

    with torch.no_grad():
        z = ddpm.sample(batch_size=args.n)
        x = tokenizer.decode(z)
        x = (x.clamp(-1, 1) + 1.0) / 2.0
        grid = make_grid(x, nrow=int(args.n ** 0.5))
        save_image(grid, args.out)

    print(f"Saved samples to: {args.out} (tokenizer={tokenizer_type})")


if __name__ == "__main__":
    main()
