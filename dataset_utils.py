from typing import Dict

from datasets import load_dataset
from PIL import Image
import torch
from torchvision import datasets, transforms


DATASET_CHANNELS: Dict[str, int] = {
    "mnist": 1,
    "fashionmnist": 1,
    "kmnist": 1,
    "cifar10": 3,
    "cifar100": 3,
    "svhn": 3,
    "stl10": 3,
    "custom": -1,
}


def supported_datasets():
    return sorted(DATASET_CHANNELS.keys())


def infer_in_channels(dataset_name: str) -> int:
    name = dataset_name.lower()
    if name not in DATASET_CHANNELS:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Supported: {supported_datasets()}")
    if name == "custom":
        raise ValueError("in_channels for custom dataset should be inferred from samples.")
    return DATASET_CHANNELS[name]


def build_transform(in_channels: int, img_size: int):
    mean = tuple([0.5] * in_channels)
    std = tuple([0.5] * in_channels)
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def _resolve_split_kwargs(dataset_name: str, train: bool) -> Dict:
    name = dataset_name.lower()
    if name in {"mnist", "fashionmnist", "kmnist", "cifar10", "cifar100"}:
        return {"train": train}
    if name in {"svhn", "stl10"}:
        return {"split": "train" if train else "test"}
    raise ValueError(f"Unsupported dataset: {dataset_name}. Supported: {supported_datasets()}")


def _as_pil_image(value):
    if isinstance(value, Image.Image):
        return value
    if torch.is_tensor(value):
        return transforms.ToPILImage()(value)
    if isinstance(value, dict) and "path" in value:
        return Image.open(value["path"]).convert("RGB")
    raise ValueError("Unsupported image type in HF sample.")


class HFBatchTransform:
    def __init__(self, tfm, in_channels: int):
        self.tfm = tfm
        self.mode = "L" if in_channels == 1 else "RGB"

    def __call__(self, batch):
        images = None
        for key in ["image", "img", "pixel_values"]:
            if key in batch:
                images = batch[key]
                break
        if images is None:
            raise ValueError(
                "No image-like field found in HF batch. "
                "Expected one of: image, img, pixel_values."
            )

        if not isinstance(images, list):
            images = [images]

        xs = [self.tfm(_as_pil_image(img).convert(self.mode)) for img in images]

        labels = batch.get("label")
        if labels is None:
            labels = [0] * len(xs)
        elif not isinstance(labels, list):
            labels = [int(labels)]
        else:
            label_set = list(filter(None, set(labels)))
            labels = [0 if y is None else label_set.index(y) for y in labels]

        return {"pixel_values": xs, "label": labels}


def _build_hf_custom_dataset(hf_ds, tfm, in_channels: int):
    transform_fn = HFBatchTransform(tfm=tfm, in_channels=in_channels)
    return hf_ds.with_transform(transform_fn)


def _infer_hf_channels(hf_ds) -> int:
    sample = hf_ds[0]
    for key in ["image", "img", "pixel_values"]:
        if key in sample:
            value = sample[key]
            if isinstance(value, Image.Image):
                bands = value.getbands()
                return 1 if len(bands) == 1 else 3
            if torch.is_tensor(value):
                if value.ndim == 2:
                    return 1
                if value.ndim == 3:
                    c = value.shape[0] if value.shape[0] <= 4 else value.shape[-1]
                    return 1 if c == 1 else 3
    return 3


def unpack_batch(batch):
    if isinstance(batch, (list, tuple)):
        if len(batch) < 2:
            raise ValueError("Expected batch tuple/list as (x, y).")
        return batch[0], batch[1]

    if isinstance(batch, dict):
        if "pixel_values" in batch:
            x = batch["pixel_values"]
        elif "x" in batch:
            x = batch["x"]
        else:
            raise ValueError(
                "Batch dict missing image tensor. Expected key 'pixel_values' or 'x'."
            )

        y = batch.get("label")
        if y is None:
            if torch.is_tensor(x):
                y = torch.zeros(x.shape[0], dtype=torch.long)
            else:
                y = 0
        return x, y

    raise TypeError(f"Unsupported batch type: {type(batch)}")


def build_dataset(dataset_name: str, root: str, train: bool, img_size: int, id: str = None):
    name = dataset_name.lower()
    if name == "custom":
        if not id:
            raise ValueError("dataset=custom requires --id")
        split = "train" if train else "test"
        try:
            hf_ds = load_dataset(id, split=split, cache_dir=root, trust_remote_code=True)
        except Exception:
            hf_ds = load_dataset(id, split="train", cache_dir=root, trust_remote_code=True)
        in_channels = _infer_hf_channels(hf_ds)
        tfm = build_transform(in_channels=in_channels, img_size=img_size)
        ds = _build_hf_custom_dataset(hf_ds=hf_ds, tfm=tfm, in_channels=in_channels)
        return ds, in_channels

    in_channels = infer_in_channels(name)
    tfm = build_transform(in_channels=in_channels, img_size=img_size)
    split_kwargs = _resolve_split_kwargs(name, train=train)

    if name == "mnist":
        ds = datasets.MNIST(root=root, download=True, transform=tfm, **split_kwargs)
    elif name == "fashionmnist":
        ds = datasets.FashionMNIST(root=root, download=True, transform=tfm, **split_kwargs)
    elif name == "kmnist":
        ds = datasets.KMNIST(root=root, download=True, transform=tfm, **split_kwargs)
    elif name == "cifar10":
        ds = datasets.CIFAR10(root=root, download=True, transform=tfm, **split_kwargs)
    elif name == "cifar100":
        ds = datasets.CIFAR100(root=root, download=True, transform=tfm, **split_kwargs)
    elif name == "svhn":
        ds = datasets.SVHN(root=root, download=True, transform=tfm, **split_kwargs)
    elif name == "stl10":
        ds = datasets.STL10(root=root, download=True, transform=tfm, **split_kwargs)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Supported: {supported_datasets()}")

    return ds, in_channels