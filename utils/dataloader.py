import os
import json
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = [
    "AddNoise", "RandomMask", "RandomByteCorruption", "PacketDrop",
    "ByteDataset"
]

class AddNoise:
    def __init__(self, std: float = 0.0):
        self.std = float(std)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.std <= 0:
            return x
        xf = x.float()
        noise = torch.randn_like(xf) * self.std * 255.0
        xf = torch.clamp(torch.round(xf + noise), 0, 255)
        return xf.long()

class RandomMask:
    def __init__(self, mask_prob: float = 0.0, mask_value: int = 0):
        assert 0.0 <= mask_prob <= 1.0
        self.mask_prob = mask_prob
        self.mask_value = int(mask_value)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.mask_prob <= 0:
            return x
        mask = torch.rand_like(x.float()) < self.mask_prob
        x = x.clone()
        x[mask] = self.mask_value
        return x

class RandomByteCorruption:
    def __init__(self, corruption_prob: float = 0.0):
        assert 0.0 <= corruption_prob <= 1.0
        self.corruption_prob = corruption_prob

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.corruption_prob <= 0:
            return x
        mask = torch.rand_like(x.float()) < self.corruption_prob
        rand_bytes = torch.randint(0, 256, x.shape, device=x.device)
        out = x.clone()
        out[mask] = rand_bytes[mask]
        return out.long()

class PacketDrop:
    def __init__(self, drop_prob: float = 0.0):
        assert 0.0 <= drop_prob <= 1.0
        self.drop_prob = drop_prob

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob <= 0:
            return x
        assert x.dim() == 2 and x.size(1) == 256, "PacketDrop expects [num_packets, 256]"
        num_packets = x.size(0)
        drop_mask = torch.rand(num_packets, device=x.device) < self.drop_prob
        if drop_mask.any():
            x = x.clone()
            x[drop_mask] = 0
        return x.long()

# ---------------------- Dataset ----------------------

def _ensure_uint8(arr: np.ndarray) -> np.ndarray:
    """将任意数值数组裁剪到 0..255 并转为 np.uint8。"""
    arr = np.nan_to_num(arr, nan=0.0, posinf=255.0, neginf=0.0)
    arr = np.clip(arr, 0, 255)
    return arr.astype(np.uint8, copy=False)

def _pad_or_trunc_lastdim(arr: np.ndarray, target: int) -> np.ndarray:
    cur = arr.shape[-1]
    if cur == target:
        return arr
    if cur > target:
        return arr[..., :target]
    pad = target - cur
    pad_width = [(0, 0)] * arr.ndim
    pad_width[-1] = (0, pad)
    return np.pad(arr, pad_width, mode='constant')

def _pad_or_trunc_packets(arr: np.ndarray, target_packets: int) -> np.ndarray:
    """第二维（num_packets）填充/截断到 target_packets。"""
    cur = arr.shape[1]
    if cur == target_packets:
        return arr
    if cur > target_packets:
        return arr[:, :target_packets, :]
    pad = target_packets - cur
    pad_width = [(0, 0)] * arr.ndim
    pad_width[1] = (0, pad)
    return np.pad(arr, pad_width, mode='constant')

class ByteDataset(Dataset):
    def __init__(self,
                 data_path: str,
                 augment: bool = False,
                 corruption_prob: float = 0.0,
                 packet_drop_prob: float = 0.0,
                 mask_prob: float = 0.0,
                 noise_std: float = 0.0,
                 num_packets: Optional[int] = None,
                 packet_len: int = 256,
                 preload_in_memory: bool = True,
                 seed: int = 42):
        super().__init__()
        self.data_path = data_path
        self.augment = augment
        self.num_packets = num_packets
        self.packet_len = packet_len
        self.preload_in_memory = preload_in_memory
        self.rng = np.random.RandomState(seed)

        # 组装 transforms（顺序：mask -> corruption -> noise -> packet_drop）
        self.t_mask = RandomMask(mask_prob=mask_prob) if augment and mask_prob > 0 else None
        self.t_corrupt = RandomByteCorruption(corruption_prob=corruption_prob) if augment and corruption_prob > 0 else None
        self.t_noise = AddNoise(std=noise_std) if augment and noise_std > 0 else None
        self.t_drop = PacketDrop(drop_prob=packet_drop_prob) if augment and packet_drop_prob > 0 else None

        # 读取数据
        X, y = self._load_any(self.data_path)

        assert X.ndim == 3, f"Expect 3-D array [N, P, L], got shape {X.shape}"
        if self.num_packets is not None:
            X = _pad_or_trunc_packets(X, self.num_packets)
        else:
            self.num_packets = X.shape[1]

        X = _pad_or_trunc_lastdim(X, self.packet_len)
        X = _ensure_uint8(X)

        self.N = X.shape[0]
        self.X = torch.from_numpy(X) if self.preload_in_memory else None
        self.y = torch.from_numpy(y.astype(np.int64)) if y is not None else torch.zeros(self.N, dtype=torch.long)

        if not self.preload_in_memory:
            raise NotImplementedError("Set preload_in_memory=True for simplicity.")

        print(f"[ByteDataset] Loaded: N={self.N}, num_packets={self.num_packets}, packet_len={self.packet_len}")

    # ------- load helpers -------

    # def _load_any(self, path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    #     ext = os.path.splitext(path)[1].lower()
    #     if ext == ".json":
    #         with open(path, "r") as f:
    #             d = json.load(f)
    #     elif ext == ".npz":
    #         d = dict(np.load(path, allow_pickle=True))
    #     elif ext in (".pt", ".pth"):
    #         d = torch.load(path, map_location="cpu")
    #     else:
    #         raise ValueError(f"Unsupported file: {path}")
    #
    #     if "packets" in d:
    #         X = np.array(d["packets"])
    #     elif "headers" in d and "payloads" in d:
    #         headers = np.array(d["headers"])
    #         payloads = np.array(d["payloads"])
    #         assert headers.shape[:2] == payloads.shape[:2], "headers/payloads batch&packet dims mismatch"
    #         X = np.concatenate([headers, payloads], axis=-1)
    #     else:
    #         raise KeyError("Expected keys 'packets' or 'headers'+'payloads' in data file.")
    #
    #     y = None
    #     for key in ("labels", "y", "targets"):
    #         if key in d:
    #             y = np.array(d[key]).reshape(-1)
    #             break
    #     if y is None:
    #         y = np.zeros((X.shape[0],), dtype=np.int64)
    #
    #     return X, y

    def _load_any(self, path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".json":
            with open(path, "r") as f:
                d = json.load(f)
        elif ext == ".npz":
            d = dict(np.load(path, allow_pickle=True))
        elif ext in (".pt", ".pth"):
            d = torch.load(path, map_location="cpu")
        else:
            raise ValueError(f"Unsupported file: {path}")

        if "packets" in d:
            # X = np.array(d["packets"])
            X = np.array(d["packets"], dtype=np.uint8)  # <--- 修正: 节省 8 倍内存

        elif "headers" in d and "payloads" in d:
            # headers = np.array(d["headers"])
            # payloads = np.array(d["payloads"])

            # <--- 修正: 强制使用 uint8 节省 8 倍内存 ---
            headers = np.array(d["headers"], dtype=np.uint8)
            payloads = np.array(d["payloads"], dtype=np.uint8)
            # ----------------------------------------

            assert headers.shape[:2] == payloads.shape[:2], "headers/payloads batch&packet dims mismatch"
            X = np.concatenate([headers, payloads], axis=-1)
        else:
            raise KeyError("Expected keys 'packets' or 'headers'+'payloads' in data file.")

        y = None
        for key in ("labels", "y", "targets"):
            if key in d:
                y = np.array(d[key]).reshape(-1)
                break
        if y is None:
            y = np.zeros((X.shape[0],), dtype=np.int64)  # 标签用 int64 没问题，数据量很小

        return X, y

    def __len__(self) -> int:
        return self.N

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        x = self.X[idx]  # [P, L], uint8
        y = int(self.y[idx].item())

        x = x.long()
        if self.t_mask is not None:
            x = self.t_mask(x)
        if self.t_corrupt is not None:
            x = self.t_corrupt(x)
        if self.t_noise is not None:
            x = self.t_noise(x)
        if self.t_drop is not None:
            x = self.t_drop(x)

        return x, y