"""TimeBase-compatible ETT-hour/minute and custom benchmark data loading.

ETT follows chronological 12/4/4 month splits; Exchange, Solar, Electricity,
Traffic and Weather use row-order 70/10/20 splits. Statistics are fitted on the training portion only and
then reused for every split.  Each sample contains a history window and its
forecast horizon; no decoder labels or time-feature tensors are returned.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


SUPPORTED_DATASETS = (
    "ETTh1", "ETTh2", "ETTm1", "ETTm2", "Exchange", "Solar",
    "Electricity", "Traffic", "Weather",
)
_OT_CUSTOM_DATASETS = {"Electricity", "Traffic", "Weather"}
_ROW_SPLIT_DATASETS = {"Exchange", "Solar", *_OT_CUSTOM_DATASETS}
Split = Literal["train", "val", "test"]
_SAMPLES_PER_MONTH = 30 * 24
_TRAIN_MONTHS = 12
_EVAL_MONTHS = 4


def _canonical_dataset(dataset: str) -> str:
    """Return the canonical dataset name or raise a useful error."""

    normalized = str(dataset).strip().lower()
    mapping = {name.lower(): name for name in SUPPORTED_DATASETS}
    try:
        return mapping[normalized]
    except KeyError as exc:
        supported = ", ".join(SUPPORTED_DATASETS)
        raise ValueError(f"dataset must be one of {supported}; received {dataset!r}") from exc


def _dataset_path(data_root: Path, dataset: str, data_file: str | None = None) -> Path:
    """Resolve the configured CSV without silently accepting another dataset."""

    canonical = _canonical_dataset(dataset)
    root = Path(data_root).expanduser()
    basename = data_file or (canonical.lower() + ".csv" if canonical in {"Electricity", "Traffic", "Weather"} else canonical + ".csv")
    if Path(basename).name != basename:
        raise ValueError("data_file must be a filename, not an absolute or relative path")
    path = root / basename
    if not path.is_file():
        raise FileNotFoundError(
            f"Could not find {path}. Put {canonical}.csv directly in data_root."
        )
    return path.resolve()


def _read_numeric_frame(path: Path, *, require_chronological_dates: bool = True) -> np.ndarray:
    """Read numerical observations without changing their original CSV row order.

    For the original ETT protocol, timestamps must be unique and monotonic.
    TimeBase custom benchmarks (Weather, Traffic, Electricity) use *row-index*
    chronological splits, and do not require unique or monotonic timestamps.
    Forecasting in TimeDO does not use the timestamp as a model feature.
    """

    frame = pd.read_csv(path)
    if "date" not in frame.columns:
        raise ValueError(f"{path} must contain a column named 'date'")
    dates = pd.to_datetime(frame["date"], errors="coerce")
    if dates.isna().any():
        bad = int(dates.isna().sum())
        raise ValueError(f"{path} contains {bad} invalid date value(s)")
    if require_chronological_dates and (not dates.is_monotonic_increasing or dates.duplicated().any()):
        raise ValueError(f"{path} dates must be unique and ordered chronologically")

    # Explicitly remove only the date column.  Every remaining column is a
    # measured variable and must be numeric for the multivariate benchmark.
    value_frame = frame.drop(columns=["date"])
    if value_frame.shape[1] == 0:
        raise ValueError(f"{path} contains no numeric series after removing 'date'")
    converted = value_frame.apply(pd.to_numeric, errors="coerce")
    if converted.isna().any().any():
        bad_columns = converted.columns[converted.isna().any()].tolist()
        raise ValueError(f"Non-numeric or missing values found in columns: {bad_columns}")
    values = converted.to_numpy(dtype=np.float64, copy=True)
    if not np.isfinite(values).all():
        raise ValueError(f"{path} contains non-finite numeric values")
    return values


def _load_scaled(path_string: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load one CSV and return scaled values plus scaler statistics.

    The file is read afresh for each dataset instance, so editing a CSV never
    leaves an old in-memory snapshot active in an interactive experiment.
    """

    path = Path(path_string)
    values = _read_numeric_frame(path)
    train_end = _TRAIN_MONTHS * _SAMPLES_PER_MONTH
    if len(values) < (_TRAIN_MONTHS + 2 * _EVAL_MONTHS) * _SAMPLES_PER_MONTH:
        raise ValueError(
            f"{path} has {len(values)} rows; the fixed 12/4/4 split requires "
            f"at least {(_TRAIN_MONTHS + 2 * _EVAL_MONTHS) * _SAMPLES_PER_MONTH}"
        )

    scaler = StandardScaler(copy=True)
    scaler.fit(values[:train_end])
    # Fit on training rows only, then transform the complete series.  The
    # explicit float64 conversion preserves the historical loader's dtype.
    scaled_all = scaler.transform(values).astype(np.float64, copy=False)
    return scaled_all, scaler.mean_.astype(np.float64), scaler.scale_.astype(np.float64)


class ETTHourDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Fixed-split ETTh1/ETTh2 windows with train-only normalization.

    Parameters
    ----------
    data_root:
        Directory containing ``ETTh1.csv`` and/or ``ETTh2.csv``.
    dataset:
        One of ``ETTh1`` or ``ETTh2`` (case-insensitive).
    split:
        ``train``, ``val``, or ``test`` under the standard 12/4/4 split.
    seq_len, pred_len:
        Number of history and forecast rows returned by each sample.
    """

    def __init__(
        self,
        data_root: Path,
        dataset: str,
        split: Split,
        seq_len: int,
        pred_len: int,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"split must be train, val, or test; received {split!r}")
        self.dataset = _canonical_dataset(dataset)
        self.split: Split = split
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        if self.seq_len <= 0 or self.pred_len <= 0:
            raise ValueError("seq_len and pred_len must be positive")
        max_history = _TRAIN_MONTHS * _SAMPLES_PER_MONTH
        if self.seq_len > max_history:
            raise ValueError(
                f"seq_len={self.seq_len} exceeds the 12-month history overlap "
                f"supported by the fixed split ({max_history})"
            )

        self.path = _dataset_path(Path(data_root), self.dataset)
        values, mean, scale = _load_scaled(str(self.path))
        self.scaler_mean = mean
        self.scaler_scale = scale
        start, end = self._split_bounds()
        if start < 0 or end > len(values) or end <= start:
            raise ValueError(f"invalid {split} split bounds {(start, end)} for {len(values)} rows")
        split_values = values[start:end]
        required = self.seq_len + self.pred_len
        if len(split_values) < required:
            raise ValueError(
                f"{split} split has {len(split_values)} rows but requires at least "
                f"seq_len + pred_len = {required}"
            )
        self.values = split_values
        self.channels = int(values.shape[1])

    def _split_bounds(self) -> tuple[int, int]:
        """Return half-open row bounds, retaining history at split edges."""

        month = _SAMPLES_PER_MONTH
        if self.split == "train":
            return 0, _TRAIN_MONTHS * month
        if self.split == "val":
            return _TRAIN_MONTHS * month - self.seq_len, (_TRAIN_MONTHS + _EVAL_MONTHS) * month
        return (
            (_TRAIN_MONTHS + _EVAL_MONTHS) * month - self.seq_len,
            (_TRAIN_MONTHS + 2 * _EVAL_MONTHS) * month,
        )

    def __len__(self) -> int:
        return len(self.values) - self.seq_len - self.pred_len + 1

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        history = self.values[index : index + self.seq_len]
        target_start = index + self.seq_len
        target = self.values[target_start : target_start + self.pred_len]
        # Keep float64 here to preserve the original data-loader contract;
        # the training engine converts batches to float32 on the target device.
        return torch.from_numpy(history), torch.from_numpy(target)


class LongForecastDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """ETTm fixed splits and 70/10/20 splits for the remaining CSV benchmarks.

    ETTm follows the standard chronological 12/4/4 month split. Electricity,
    Traffic and Weather put ``OT`` last as in the TimeBase custom protocol;
    Exchange and Solar keep every measured column in its original order.
    Scaling is fitted on training rows only.
    """
    def __init__(self, data_root: Path, dataset: str, split: Split,
                 seq_len: int, pred_len: int, data_file: str | None = None) -> None:
        self.dataset = _canonical_dataset(dataset)
        if split not in ("train", "val", "test"):
            raise ValueError(f"invalid split: {split}")
        self.split = split
        self.seq_len, self.pred_len = int(seq_len), int(pred_len)
        self.path = _dataset_path(data_root, dataset, data_file)
        frame = pd.read_csv(self.path)
        if "date" not in frame.columns:
            raise ValueError(f"{self.path} is missing a date column")
        dates = pd.to_datetime(frame["date"], errors="coerce")
        if dates.isna().any():
            raise ValueError(f"{self.path} date values contain invalid timestamps")
        # TimeBase's Dataset_Custom keeps the CSV row order for its 70/10/20
        # protocol; it does not sort or deduplicate custom-dataset timestamps.
        if self.dataset not in _OT_CUSTOM_DATASETS:
            if not dates.is_monotonic_increasing or dates.duplicated().any():
                raise ValueError(f"{self.path} date values must be unique and chronological")
        if self.dataset in _OT_CUSTOM_DATASETS:
            if "OT" not in frame.columns:
                raise ValueError(f"{self.path}: TimeBase custom protocol requires OT column")
            cols = [c for c in frame.columns if c not in ("date", "OT")]
            cols += ["OT"]
        else:
            cols = [c for c in frame.columns if c != "date"]

        if self.dataset in _ROW_SPLIT_DATASETS:
            n = len(frame)
            train_end, test_count = int(n * 0.7), int(n * 0.2)
            val_end = n - test_count
            bounds = {"train": (0, train_end), "val": (train_end-self.seq_len, val_end),
                      "test": (val_end-self.seq_len, n)}
        else:
            steps_per_month = 30 * 24 * 4 if self.dataset.startswith("ETTm") else 30 * 24
            train_end = 12 * steps_per_month
            val_end = 16 * steps_per_month
            test_end = 20 * steps_per_month
            if len(frame) < test_end:
                raise ValueError(f"{self.path} requires >= {test_end} rows for the ETT split")
            bounds = {"train": (0, train_end), "val": (train_end-self.seq_len, val_end),
                      "test": (val_end-self.seq_len, test_end)}
        numeric = frame[cols].apply(pd.to_numeric, errors="coerce")
        if len(cols) == 0 or numeric.isna().any().any():
            raise ValueError(f"{self.path} contains missing/non-numeric features")
        values = numeric.to_numpy(dtype=np.float64, copy=True)
        if not np.isfinite(values).all():
            raise ValueError(f"{self.path} contains nonfinite values")
        scaler = StandardScaler(copy=True)
        scaler.fit(values[:train_end])
        scaled = scaler.transform(values).astype(np.float64, copy=False)
        start, end = bounds[split]
        if start < 0 or end > len(values) or end-start < seq_len+pred_len:
            raise ValueError(f"not enough rows in {split} split of {self.path}")
        self.values = scaled[start:end]
        self.channels = len(cols)
        self.scaler_mean = scaler.mean_.copy()
        self.scaler_scale = scaler.scale_.copy()

    def __len__(self) -> int:
        return len(self.values) - self.seq_len - self.pred_len + 1

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        a = self.values[index:index+self.seq_len]
        b = self.values[index+self.seq_len:index+self.seq_len+self.pred_len]
        return torch.from_numpy(a), torch.from_numpy(b)


def infer_channels(data_root: Path, dataset: str, data_file: str | None = None) -> int:
    """Return the number of measured series in an ETTh CSV."""

    path = _dataset_path(Path(data_root), dataset, data_file)
    # Custom benchmarks may legitimately contain repeated/out-of-order date
    # labels. Channel inference must obey the same policy as the data loader.
    enforce = _canonical_dataset(dataset) not in _OT_CUSTOM_DATASETS
    return int(_read_numeric_frame(path, require_chronological_dates=enforce).shape[1])


def build_loaders(
    data_root: Path,
    dataset: str,
    seq_len: int,
    pred_len: int,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = False,
    data_file: str | None = None,
) -> dict[str, DataLoader]:
    """Build train/validation/test loaders under the fixed ETT protocol."""

    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    if int(num_workers) < 0:
        raise ValueError("num_workers cannot be negative")
    canonical = _canonical_dataset(dataset)
    datasets = {
        split: (ETTHourDataset(Path(data_root), canonical, split, seq_len, pred_len)
                if canonical in ("ETTh1", "ETTh2") and data_file in (None, canonical + ".csv")
                else LongForecastDataset(Path(data_root), canonical, split, seq_len, pred_len, data_file))
        for split in ("train", "val", "test")
    }
    return {
        split: DataLoader(
            dataset_obj,
            batch_size=int(batch_size),
            shuffle=split == "train",
            drop_last=False,
            num_workers=int(num_workers),
            pin_memory=bool(pin_memory),
        )
        for split, dataset_obj in datasets.items()
    }


__all__ = ["ETTHourDataset", "LongForecastDataset", "SUPPORTED_DATASETS", "build_loaders", "infer_channels"]
