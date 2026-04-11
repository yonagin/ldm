from typing import Dict

from datasets import load_dataset
from PIL import Image
import torch
from torch.utils.data import Dataset
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


class HFDatasetWrapper(Dataset):
    def __init__(self, hf_ds, tfm, in_channels: int):
        self.hf_ds = hf_ds
        self.tfm = tfm
        self.in_channels = in_channels

    def __len__(self):
        return len(self.hf_ds)

    def _extract_image(self, sample):
        for key in ["image", "img", "pixel_values"]:
            if key in sample:
                value = sample[key]
                if isinstance(value, Image.Image):
                    return value
                if torch.is_tensor(value):
                    return transforms.ToPILImage()(value)
                if isinstance(value, dict) and "path" in value:
                    return Image.open(value["path"]).convert("RGB")
        raise ValueError("No image-like field found in sample. Expected one of: image, img, pixel_values.")

    def __getitem__(self, idx):
        sample = self.hf_ds[idx]
        image = self._extract_image(sample)
        image = image.convert("L" if self.in_channels == 1 else "RGB")
        x = self.tfm(image)
        y = sample.get("label", 0)
        if y is None:
            y = 0
        return x, int(y)


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
                    # CHW or HWC
                    c = value.shape[0] if value.shape[0] <= 4 else value.shape[-1]
                    return 1 if c == 1 else 3
    return 3


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
        ds = HFDatasetWrapper(hf_ds=hf_ds, tfm=tfm, in_channels=in_channels)
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
