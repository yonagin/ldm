from typing import Dict, Tuple

from torchvision import datasets, transforms


DATASET_CHANNELS: Dict[str, int] = {
    "mnist": 1,
    "fashionmnist": 1,
    "kmnist": 1,
    "cifar10": 3,
    "cifar100": 3,
    "svhn": 3,
    "stl10": 3,
}


def supported_datasets():
    return sorted(DATASET_CHANNELS.keys())


def infer_in_channels(dataset_name: str) -> int:
    name = dataset_name.lower()
    if name not in DATASET_CHANNELS:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Supported: {supported_datasets()}")
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


def build_dataset(dataset_name: str, root: str, train: bool, img_size: int):
    name = dataset_name.lower()
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

