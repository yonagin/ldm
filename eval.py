import argparse
import json
import os
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader, Subset

from dataset_utils import build_dataset, unpack_batch
from models.diffusion import DDPM
from models.vae import VAE
from models.rankae import RankAE
from models.vqvae import VQVAE
from modules.unet import UNetModel


def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
        raise ValueError(f"Unsupported tokenizer: {tokenizer_type}")

    tok.load_state_dict(ckpt["model"])
    tok.to(device).eval()
    for p in tok.parameters():
        p.requires_grad_(False)
    return tok, tokenizer_type, latent_dim


def encode_with_tokenizer(tokenizer, tokenizer_type: str, x: torch.Tensor):
    if tokenizer_type == "vae":
        return tokenizer.encode(x).mode()
    return tokenizer.encode(x)


def load_ldm(ldm_ckpt: str, latent_dim: int, device: str) -> Tuple[DDPM, Dict]:
    ckpt = torch.load(ldm_ckpt, map_location=device)
    timesteps = ckpt.get("timesteps", 200)
    latent_size = ckpt.get("latent_size")
    if latent_size is None:
        raise ValueError("latent_size not found in ldm checkpoint")

    unet = UNetModel(in_channels=latent_dim, out_channels=latent_dim)
    ddpm = DDPM(
        unet=unet,
        timesteps=timesteps,
        image_size=latent_size,
        channels=latent_dim,
        parameterization="eps",
        loss_type="l2",
    ).to(device)
    ddpm.load_state_dict(ckpt["model"])
    ddpm.eval()
    return ddpm, ckpt


@torch.no_grad()
def generate_fake_images(ddpm: DDPM, tokenizer, n: int, batch_size: int, device: str):
    chunks = []
    left = n
    while left > 0:
        cur = min(left, batch_size)
        z = ddpm.sample(batch_size=cur)
        x = tokenizer.decode(z).clamp(-1, 1)
        chunks.append(x)
        left -= cur
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def collect_real_images(
    dataset_name: str,
    data_root: str,
    n: int,
    split: str,
    batch_size: int,
    img_size: int,
    id: str = None,
):
    is_train = split == "train"
    ds, _ = build_dataset(
        dataset_name,
        root=data_root,
        train=is_train,
        img_size=img_size,
        id=id,
    )
    n = min(n, len(ds))
    ds = Subset(ds, list(range(n)))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    xs = []
    for batch in dl:
        x, _ = unpack_batch(batch)
        xs.append(x)
    return torch.cat(xs, dim=0)


@torch.no_grad()
def extract_features(x: torch.Tensor, tokenizer, tokenizer_type: str, device: str, batch_size: int):
    feats = []
    for i in range(0, x.shape[0], batch_size):
        xb = x[i:i + batch_size].to(device)
        z = encode_with_tokenizer(tokenizer, tokenizer_type, xb)
        feats.append(z.flatten(start_dim=1).detach().cpu())
    return torch.cat(feats, dim=0)


def pairwise_dists(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.cdist(a, b, p=2)


def manifold_radii(x: torch.Tensor, k: int) -> torch.Tensor:
    d = pairwise_dists(x, x)
    d.fill_diagonal_(float("inf"))
    kth = torch.topk(d, k=k, dim=1, largest=False).values[:, -1]
    return kth


def precision_recall_coverage(real_f: torch.Tensor, fake_f: torch.Tensor, k: int):
    rr = manifold_radii(real_f, k=k)
    rf = manifold_radii(fake_f, k=k)
    d_fr = pairwise_dists(fake_f, real_f)
    d_rf = d_fr.t()

    precision = (d_fr <= rr.unsqueeze(0)).any(dim=1).float().mean().item()
    recall = (d_rf <= rf.unsqueeze(0)).any(dim=1).float().mean().item()

    nearest_fake_to_real = d_rf.min(dim=1).values
    coverage = (nearest_fake_to_real <= rr).float().mean().item()

    return precision, recall, coverage


def polynomial_mmd_kid(x: torch.Tensor, y: torch.Tensor, degree: int = 3, gamma=None, coef0: float = 1.0):
    if gamma is None:
        gamma = 1.0 / x.shape[1]

    k_xx = (gamma * (x @ x.t()) + coef0).pow(degree)
    k_yy = (gamma * (y @ y.t()) + coef0).pow(degree)
    k_xy = (gamma * (x @ y.t()) + coef0).pow(degree)

    m = x.shape[0]
    n = y.shape[0]

    sum_xx = (k_xx.sum() - k_xx.diag().sum()) / (m * (m - 1))
    sum_yy = (k_yy.sum() - k_yy.diag().sum()) / (n * (n - 1))
    sum_xy = k_xy.mean()
    return (sum_xx + sum_yy - 2.0 * sum_xy).item()


def diversity_score(fake_f: torch.Tensor, max_pairs: int = 10000):
    n = fake_f.shape[0]
    if n < 2:
        return 0.0

    idx_i = torch.randint(0, n, (max_pairs,))
    idx_j = torch.randint(0, n, (max_pairs,))
    keep = idx_i != idx_j
    idx_i = idx_i[keep]
    idx_j = idx_j[keep]
    if idx_i.numel() == 0:
        return 0.0
    d = torch.norm(fake_f[idx_i] - fake_f[idx_j], dim=1)
    return d.mean().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer-ckpt", type=str, required=True)
    parser.add_argument("--ldm-ckpt", type=str, required=True)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--id", type=str, default=None)
    parser.add_argument("--img-size", type=int, default=None)
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--num-real", type=int, default=2000)
    parser.add_argument("--num-fake", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-json", type=str, default="./samples/quality_metrics.json")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.k < 1:
        raise ValueError("--k must be >= 1")

    set_seed(args.seed)
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)

    tokenizer, tokenizer_type, latent_dim = load_tokenizer(args.tokenizer_ckpt, args.device)
    ddpm, ldm_meta = load_ldm(args.ldm_ckpt, latent_dim=latent_dim, device=args.device)

    dataset_name = args.dataset or ldm_meta.get("dataset", "mnist")
    id = args.id or ldm_meta.get("id")
    img_size = args.img_size or ldm_meta.get("img_size", 28)

    real_x = collect_real_images(
        dataset_name,
        args.data_root,
        args.num_real,
        args.split,
        args.batch_size,
        img_size,
        id,
    )
    fake_x = generate_fake_images(ddpm, tokenizer, args.num_fake, args.batch_size, args.device).cpu()

    real_f = extract_features(real_x, tokenizer, tokenizer_type, args.device, args.batch_size)
    fake_f = extract_features(fake_x, tokenizer, tokenizer_type, args.device, args.batch_size)

    k_eff = min(args.k, real_f.shape[0] - 1, fake_f.shape[0] - 1)
    if k_eff < 1:
        raise ValueError("Not enough samples for k-NN metrics, increase --num-real/--num-fake")

    precision, recall, coverage = precision_recall_coverage(real_f, fake_f, k=k_eff)
    kid = polynomial_mmd_kid(real_f, fake_f)
    diversity = diversity_score(fake_f)

    pixel_mean_real = real_x.mean().item()
    pixel_mean_fake = fake_x.mean().item()
    pixel_std_real = real_x.std().item()
    pixel_std_fake = fake_x.std().item()

    metrics = {
        "tokenizer": tokenizer_type,
        "dataset": dataset_name,
        "id": id,
        "img_size": img_size,
        "latent_dim": latent_dim,
        "timesteps": ldm_meta.get("timesteps", 200),
        "latent_size": ldm_meta.get("latent_size"),
        "num_real": int(real_x.shape[0]),
        "num_fake": int(fake_x.shape[0]),
        "k": int(k_eff),
        "precision": precision,
        "recall": recall,
        "coverage": coverage,
        "kid_poly3": kid,
        "diversity_l2": diversity,
        "pixel_mean_real": pixel_mean_real,
        "pixel_mean_fake": pixel_mean_fake,
        "pixel_std_real": pixel_std_real,
        "pixel_std_fake": pixel_std_fake,
    }

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Saved metrics to: {args.out_json}")


if __name__ == "__main__":
    main()
