from typing import Dict, Optional
from datasets import load_dataset
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import numpy as np


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
    raise ValueError(f"Unsupported dataset: {dataset_name}.")


class HFDatasetWrapper(Dataset):
    """
    关键修复：
    1. 构造时探测一次 image_key，避免 __getitem__ 每次都遍历 key
    2. 用 hf_ds.with_format(None) 确保返回 Python 原生对象而非 Arrow
    3. 支持 numpy array 输入（HF 经常返回 np.ndarray）
    """

    def __init__(self, hf_ds, tfm, in_channels: int):
        # 关键：显式设置格式，避免 Arrow 反序列化开销
        self.hf_ds = hf_ds.with_format(None)  # 返回纯 Python/PIL
        self.tfm = tfm
        self.in_channels = in_channels
        self.image_key = self._detect_image_key()
        self.mode = "L" if in_channels == 1 else "RGB"

    def _detect_image_key(self) -> str:
        """只在初始化时探测一次"""
        sample = self.hf_ds[0]
        for key in ["image", "img", "pixel_values"]:
            if key in sample:
                return key
        raise ValueError(
            f"No image field found. Available keys: {list(sample.keys())}"
        )

    def __len__(self):
        return len(self.hf_ds)

    def _to_pil(self, value) -> Image.Image:
        """统一转换为 PIL Image"""
        if isinstance(value, Image.Image):
            return value
        if isinstance(value, np.ndarray):
            # HF 很多数据集返回 numpy array，直接用 PIL 转，避免绕一圈
            return Image.fromarray(value)
        if torch.is_tensor(value):
            return transforms.ToPILImage()(value)
        if isinstance(value, dict) and "path" in value:
            return Image.open(value["path"])
        raise TypeError(f"Cannot convert type {type(value)} to PIL Image")

    def __getitem__(self, idx):
        sample = self.hf_ds[idx]
        image = self._to_pil(sample[self.image_key]).convert(self.mode)
        x = self.tfm(image)
        y = sample.get("label", 0) or 0
        return x, int(y)


def _infer_hf_channels(hf_ds) -> int:
    sample = hf_ds[0]
    for key in ["image", "img", "pixel_values"]:
        if key in sample:
            value = sample[key]
            if isinstance(value, Image.Image):
                return 1 if len(value.getbands()) == 1 else 3
            if isinstance(value, np.ndarray):
                if value.ndim == 2:
                    return 1
                return 1 if value.shape[-1] == 1 else 3
            if torch.is_tensor(value):
                if value.ndim == 2:
                    return 1
                c = value.shape[0] if value.shape[0] <= 4 else value.shape[-1]
                return 1 if c == 1 else 3
    return 3


def build_dataset(
    dataset_name: str,
    root: str,
    train: bool,
    img_size: int,
    id: Optional[str] = None,
    num_proc: int = 4,  # 新增：预处理并行数
):
    name = dataset_name.lower()

    if name == "custom":
        if not id:
            raise ValueError("dataset=custom requires --id")
        split = "train" if train else "test"
        try:
            hf_ds = load_dataset(
                id,
                split=split,
                cache_dir=root,
                trust_remote_code=True,
                num_proc=num_proc,  # 并行下载/处理
            )
        except Exception:
            hf_ds = load_dataset(
                id,
                split="train",
                cache_dir=root,
                trust_remote_code=True,
                num_proc=num_proc,
            )

        in_channels = _infer_hf_channels(hf_ds)
        tfm = build_transform(in_channels=in_channels, img_size=img_size)
        ds = HFDatasetWrapper(hf_ds=hf_ds, tfm=tfm, in_channels=in_channels)
        return ds, in_channels

    # 以下 torchvision 原生数据集不变
    in_channels = infer_in_channels(name)
    tfm = build_transform(in_channels=in_channels, img_size=img_size)
    split_kwargs = _resolve_split_kwargs(name, train=train)

    DS_MAP = {
        "mnist": datasets.MNIST,
        "fashionmnist": datasets.FashionMNIST,
        "kmnist": datasets.KMNIST,
        "cifar10": datasets.CIFAR10,
        "cifar100": datasets.CIFAR100,
        "svhn": datasets.SVHN,
        "stl10": datasets.STL10,
    }
    ds = DS_MAP[name](root=root, download=True, transform=tfm, **split_kwargs)
    return ds, in_channels