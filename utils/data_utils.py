from typing import Tuple, Union, Dict, Optional
import logging
import torch
from torch.utils.data import DataLoader, random_split

try:
    from utils.dataloader import ByteDataset
except Exception:
    from dataloader import ByteDataset

__all__ = [
    "create_train_val_loaders",
    "create_test_loader",
    "create_streaming_loader",
]

logger = logging.getLogger(__name__)


def _get(args: Union[object, Dict], name: str, default=None):
    """同时兼容 Namespace / dict 的取参。"""
    if isinstance(args, dict):
        return args.get(name, default)
    return getattr(args, name, default)

def _infer_pin_memory(args: Union[object, Dict]) -> bool:
    """简单规则：若可用 CUDA 且没有显式指定 cpu，则启用 pin_memory。"""
    device = str(_get(args, "device", "") or "").lower()
    return torch.cuda.is_available() and not device.startswith("cpu")

def _dataset_kwargs_from_args(args: Union[object, Dict], augment: bool) -> dict:
    kwargs = dict(
        augment=augment or bool(_get(args, "use_augmentation", False)),
        corruption_prob=float(_get(args, "corruption_prob", 0.0) or 0.0),
        packet_drop_prob=float(_get(args, "packet_drop_prob", 0.0) or 0.0),
        mask_prob=float(_get(args, "mask_prob", 0.0) or 0.0),
        noise_std=float(_get(args, "noise_std", 0.0) or 0.0),
        seed=int(_get(args, "seed", 42) or 42),
    )
    # 可选：num_packets
    npk = _get(args, "num_packets", None)
    if npk is not None:
        kwargs["num_packets"] = int(npk)
    return kwargs

def _make_loader(dataset, args: Union[object, Dict], *, shuffle: bool, drop_last: bool) -> DataLoader:
    bs = int(_get(args, "batch_size", 128) or 128)
    nw = int(_get(args, "num_workers", 0) or 0)
    pin = _infer_pin_memory(args)
    return DataLoader(
        dataset,
        batch_size=bs,
        shuffle=shuffle,
        num_workers=nw,
        pin_memory=pin,
        drop_last=drop_last,
    )


def create_train_val_loaders(
    data_path: str,
    args: Union[object, Dict],
    val_split: float = 0.2,
    seed: int = 42,
    augment: bool = False
) -> Tuple[DataLoader, DataLoader]:
    """
    返回 (train_loader, val_loader)。
    - 数据文件支持：.json / .npz / .pt(.pth)
    - 需要包含键：
        * 'packets' 形如 [N, P, L]，或
        * 'headers' + 'payloads'（会自动在最后一维拼接）
      标签可选：'labels' / 'y' / 'targets'，缺省则全 0。
    """
    # 构建完整的 Dataset 参数
    ds_kwargs = _dataset_kwargs_from_args(args, augment=augment)
    dataset = ByteDataset(data_path, **ds_kwargs)
    total = len(dataset)
    if total < 2:
        raise ValueError(f"Dataset too small (N={total}). Need at least 2 samples for train/val split.")

    # 计算切分尺寸
    val_size = int(round(total * float(val_split)))
    val_size = max(1, min(val_size, total - 1))  # 确保 [1, total-1]
    train_size = total - val_size

    # 确保可复现
    g = torch.Generator()
    g.manual_seed(int(seed))

    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=g)

    # DataLoader
    train_loader = _make_loader(train_ds, args, shuffle=True, drop_last=True)
    val_loader = _make_loader(val_ds, args, shuffle=False, drop_last=False)

    logger.info(
        f"[create_train_val_loaders] N={total}, split={train_size}/{val_size}, "
        f"batch_size={_get(args,'batch_size',128)}, workers={_get(args,'num_workers',0)}, "
        f"augment={ds_kwargs['augment']}"
    )
    return train_loader, val_loader


def create_test_loader(
    data_path: str,
    args: Union[object, Dict],
    batch_size: Optional[int] = None,
    augment: bool = False
) -> DataLoader:
    """
    构建仅用于评估/推理的 DataLoader（不 shuffle、不丢尾）。
    """
    ds_kwargs = _dataset_kwargs_from_args(args, augment=augment)
    dataset = ByteDataset(data_path, **ds_kwargs)

    # 若传入了自定义 batch_size，临时覆盖
    tmp = dict(args.__dict__) if hasattr(args, "__dict__") else dict(args)
    if batch_size is not None:
        tmp["batch_size"] = int(batch_size)

    loader = _make_loader(dataset, tmp, shuffle=False, drop_last=False)
    logger.info(
        f"[create_test_loader] N={len(dataset)}, batch_size={_get(tmp,'batch_size',128)}, "
        f"workers={_get(tmp,'num_workers',0)}, augment={ds_kwargs['augment']}"
    )
    return loader


def create_streaming_loader(
    data_path: str,
    args: Union[object, Dict],
    batch_size: Optional[int] = None
) -> DataLoader:
    """
    “流式”接口占位：此处等价于 test loader（按需你可以换成分块读）。
    """
    return create_test_loader(data_path, args, batch_size=batch_size, augment=False)
