from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


def _load_array(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(path)
    elif suffix == ".csv":
        arr = pd.read_csv(path).select_dtypes(include=["number"]).to_numpy()
    else:
        raise ValueError(f"Unsupported file type: {path}")
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    elif arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    return np.nan_to_num(arr)


def _find_file(data_path: Path, dataset: str, split: str) -> Path:
    names = {
        "train": [f"{dataset}_train.npy", f"{dataset}_train.csv", "train.npy", "train.csv"],
        "test": [f"{dataset}_test.npy", f"{dataset}_test.csv", "test.npy", "test.csv"],
        "test_label": [
            f"{dataset}_test_label.npy",
            f"{dataset}_test_label.csv",
            "test_label.npy",
            "test_label.csv",
        ],
    }[split]
    lower = {p.name.lower(): p for p in data_path.iterdir() if p.is_file()}
    for name in names:
        hit = lower.get(name.lower())
        if hit is not None:
            return hit
    raise FileNotFoundError(f"No {split} file found in {data_path}")


class SegmentLoader(Dataset):
    def __init__(self, data_path, dataset, win_size, step=1, mode="train"):
        self.mode = mode
        self.step = int(step)
        self.win_size = int(win_size)
        self.dataset = dataset
        self.data_path = Path(data_path)

        train = _load_array(_find_file(self.data_path, dataset, "train"))
        test = _load_array(_find_file(self.data_path, dataset, "test"))
        labels = _load_array(_find_file(self.data_path, dataset, "test_label"))
        labels = labels.reshape(labels.shape[0], -1)
        if labels.shape[1] > 1:
            labels = labels.max(axis=1, keepdims=True)

        self.scaler = StandardScaler()
        self.train = self.scaler.fit_transform(train).astype(np.float32)
        self.test = self.scaler.transform(test).astype(np.float32)
        self.test_labels = labels.astype(np.float32)
        self.num_features = int(self.train.shape[1])

        self.val = self.test

    def __len__(self):
        data = self.train if self.mode == "train" else self.test
        if data.shape[0] < self.win_size:
            return 0
        return (data.shape[0] - self.win_size) // self.step + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            window = self.train[index : index + self.win_size]
            labels = np.zeros((self.win_size, 1), dtype=np.float32)
        else:
            window = self.test[index : index + self.win_size]
            labels = self.test_labels[index : index + self.win_size]
        return np.float32(window), np.float32(labels)


def get_loader_segment(
    index,
    data_path,
    batch_size,
    win_size=100,
    step=1,
    mode="train",
    dataset="SKAB",
    num_workers=0,
):
    del index
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset path not found: {data_path}")
    ds = SegmentLoader(data_path, dataset=dataset, win_size=win_size, step=step, mode=mode)
    shuffle = mode == "train"
    return DataLoader(
        dataset=ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
    )
