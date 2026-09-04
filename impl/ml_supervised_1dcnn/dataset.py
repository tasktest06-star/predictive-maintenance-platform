"""
PyTorch Dataset classes for CWRU and MFPT bearing fault datasets.

CRITICAL: All train/test splits are bearing-wise to prevent data leakage.
Correlated windows from the same bearing run must not appear in both sets.
See arXiv 2509.22267 for motivation and methodology.

References:
    arXiv 2509.22267 - Data leakage in bearing fault diagnosis
    https://engineering.case.edu/bearingdatacenter
    https://www.mfpt.org/fault-data-sets/
"""

from __future__ import annotations

import logging
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Optional, Union

import numpy as np
import scipy.io
import torch
from torch import Tensor
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

from model import FAULT_CLASSES, NUM_CLASSES

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Label maps
# ---------------------------------------------------------------------------

# CWRU: maps (location, fault_diameter_inches) → fault class index
# location: 'normal', 'outer', 'inner', 'ball'
CWRU_LABEL_MAP: dict[tuple[str, str], int] = {
    ("normal", "0"):     FAULT_CLASSES.index("Normal"),
    ("outer", "0.007"):  FAULT_CLASSES.index("BPFO"),
    ("outer", "0.014"):  FAULT_CLASSES.index("BPFO"),
    ("outer", "0.021"):  FAULT_CLASSES.index("BPFO"),
    ("inner", "0.007"):  FAULT_CLASSES.index("BPFI"),
    ("inner", "0.014"):  FAULT_CLASSES.index("BPFI"),
    ("inner", "0.021"):  FAULT_CLASSES.index("BPFI"),
    ("ball",  "0.007"):  FAULT_CLASSES.index("BSF"),
    ("ball",  "0.014"):  FAULT_CLASSES.index("BSF"),
    ("ball",  "0.021"):  FAULT_CLASSES.index("BSF"),
}

# MFPT: maps condition string → fault class index
MFPT_LABEL_MAP: dict[str, int] = {
    "baseline":          FAULT_CLASSES.index("Normal"),
    "outer_race":        FAULT_CLASSES.index("BPFO"),
    "inner_race":        FAULT_CLASSES.index("BPFI"),
    "ball":              FAULT_CLASSES.index("BSF"),
}

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

WINDOW_SIZE = 2048
OVERLAP = 0.5  # 50% overlap between consecutive windows


def segment_and_fft(
    signal: np.ndarray,
    window_size: int = WINDOW_SIZE,
    overlap: float = OVERLAP,
) -> np.ndarray:
    """
    Slice a 1D signal into overlapping windows and compute FFT magnitude per window.

    Args:
        signal:      1D vibration signal array.
        window_size: Number of samples per window.
        overlap:     Fractional overlap between consecutive windows [0, 1).

    Returns:
        windows_fft: (N_windows, window_size) float32 array of FFT magnitudes.
                     Only the one-sided spectrum (first window_size//2 bins) is kept
                     and then zero-padded back to window_size for a uniform shape.
    """
    step = int(window_size * (1.0 - overlap))
    n_windows = max(0, (len(signal) - window_size) // step + 1)
    windows = np.lib.stride_tricks.sliding_window_view(signal, window_size)[::step]
    windows = windows[:n_windows].astype(np.float32)

    # Hann window to reduce spectral leakage
    hann = np.hanning(window_size).astype(np.float32)
    windows = windows * hann[np.newaxis, :]

    # FFT magnitude — one-sided, normalised
    fft_mags = np.abs(np.fft.rfft(windows, n=window_size, axis=1)).astype(np.float32)
    # rfft returns window_size//2 + 1 bins; pad to window_size for uniform shape
    half = fft_mags.shape[1]
    result = np.zeros((n_windows, window_size), dtype=np.float32)
    result[:, :half] = fft_mags

    # Per-window standardisation
    std = result.std(axis=1, keepdims=True) + 1e-8
    result = (result - result.mean(axis=1, keepdims=True)) / std
    return result


# ---------------------------------------------------------------------------
# CWRU Dataset
# ---------------------------------------------------------------------------

# Expected .mat variable names for the drive-end channel
CWRU_DE_KEYS = [
    "X097_DE_time", "X098_DE_time", "X099_DE_time", "X100_DE_time",
    "X105_DE_time", "X106_DE_time", "X107_DE_time", "X108_DE_time",
    "X109_DE_time", "X110_DE_time", "X111_DE_time", "X112_DE_time",
    "X118_DE_time", "X119_DE_time", "X120_DE_time", "X121_DE_time",
    "X122_DE_time", "X123_DE_time", "X124_DE_time", "X125_DE_time",
    "X130_DE_time", "X131_DE_time", "X132_DE_time", "X133_DE_time",
    "X169_DE_time", "X170_DE_time", "X171_DE_time", "X172_DE_time",
    "X173_DE_time", "X174_DE_time", "X175_DE_time", "X176_DE_time",
    "X177_DE_time", "X178_DE_time", "X179_DE_time", "X180_DE_time",
    "X185_DE_time", "X186_DE_time", "X187_DE_time", "X188_DE_time",
    "X189_DE_time", "X190_DE_time", "X191_DE_time", "X192_DE_time",
    "X197_DE_time", "X198_DE_time", "X199_DE_time", "X200_DE_time",
    "X201_DE_time", "X202_DE_time", "X203_DE_time", "X204_DE_time",
    "X209_DE_time", "X210_DE_time", "X211_DE_time", "X212_DE_time",
    "X213_DE_time", "X214_DE_time", "X215_DE_time", "X217_DE_time",
]


def _load_cwru_mat(mat_path: Path, channel: str = "DE") -> Optional[np.ndarray]:
    """Load drive-end (DE) or fan-end (FE) signal from a CWRU .mat file."""
    try:
        mat = scipy.io.loadmat(str(mat_path))
    except Exception as exc:
        logger.warning("Failed to load %s: %s", mat_path, exc)
        return None

    key_suffix = f"_{channel}_time"
    for key in mat:
        if key.endswith(key_suffix) and not key.startswith("__"):
            signal = mat[key].flatten().astype(np.float64)
            return signal

    logger.warning("No %s channel found in %s", channel, mat_path)
    return None


class CWRUBearingDataset(Dataset):
    """
    PyTorch Dataset for the CWRU Bearing Dataset.

    Directory structure expected::

        data_root/
            Normal_0/     (normal baseline)
                97.mat, 98.mat, 99.mat, 100.mat
            12k_Drive_End_Bearing_Fault_Data/
                outer/
                    0.007/  105.mat 169.mat 197.mat 209.mat
                    0.014/  ...
                    0.021/  ...
                inner/
                    0.007/  ...
                ball/
                    0.007/  ...
            ...

    Alternatively, pass a flat list of (mat_path, label_int, bearing_id) tuples
    via the ``records`` argument.

    Args:
        data_root:   Root directory of the CWRU dataset.
        records:     Pre-built list of (path, label, bearing_id) — overrides data_root scan.
        channel:     'DE' (drive-end) or 'FE' (fan-end).
        window_size: FFT window size in samples.
        overlap:     Overlap fraction between consecutive windows.
        transform:   Optional callable applied to each (1, window_size) numpy array.
    """

    def __init__(
        self,
        data_root: Optional[Union[str, Path]] = None,
        records: Optional[list[tuple[Path, int, str]]] = None,
        channel: str = "DE",
        window_size: int = WINDOW_SIZE,
        overlap: float = OVERLAP,
        transform=None,
    ) -> None:
        super().__init__()
        self.channel = channel
        self.window_size = window_size
        self.overlap = overlap
        self.transform = transform

        # List of (windows_array, label, bearing_id)
        # windows_array: (N_windows, window_size)
        self._samples: list[tuple[np.ndarray, int, str]] = []

        # bearing_id → list of global window indices
        self._bearing_index: dict[str, list[int]] = defaultdict(list)

        if records is not None:
            self._load_from_records(records)
        elif data_root is not None:
            self._scan_directory(Path(data_root))
        else:
            raise ValueError("Provide either data_root or records.")

        self._build_flat_index()

    # ------------------------------------------------------------------
    # Data loading helpers
    # ------------------------------------------------------------------

    def _scan_directory(self, root: Path) -> None:
        """
        Walk the CWRU directory tree and build the record list.

        Recognises the standard CWRU folder naming convention:
          - Folders containing 'Normal' → label Normal
          - Folders named 'outer'/'inner'/'ball' → respective fault
          - Subfolder name = fault diameter (e.g. '0.007')
        """
        bearing_counter = 0
        for mat_file in sorted(root.rglob("*.mat")):
            parts = [p.lower() for p in mat_file.parts]
            fault_type = "normal"
            fault_diam = "0"

            if "normal" in parts or mat_file.stem.startswith("97") or mat_file.stem.startswith("98"):
                fault_type = "normal"
                fault_diam = "0"
            else:
                for loc in ("outer", "inner", "ball"):
                    if loc in parts:
                        fault_type = loc
                        break
                # Find diameter folder (e.g. '0.007')
                for part in parts:
                    try:
                        float(part)
                        fault_diam = part
                        break
                    except ValueError:
                        continue

            label = CWRU_LABEL_MAP.get((fault_type, fault_diam))
            if label is None:
                logger.debug("Skipping %s — no matching label for (%s, %s)", mat_file, fault_type, fault_diam)
                continue

            bearing_id = f"cwru_{bearing_counter:04d}_{mat_file.stem}"
            bearing_counter += 1

            signal = _load_cwru_mat(mat_file, channel=self.channel)
            if signal is None or len(signal) < self.window_size:
                continue

            windows = segment_and_fft(signal, self.window_size, self.overlap)
            if len(windows) == 0:
                continue
            self._samples.append((windows, label, bearing_id))

    def _load_from_records(self, records: list[tuple[Path, int, str]]) -> None:
        for mat_path, label, bearing_id in records:
            signal = _load_cwru_mat(mat_path, channel=self.channel)
            if signal is None or len(signal) < self.window_size:
                continue
            windows = segment_and_fft(signal, self.window_size, self.overlap)
            if len(windows) > 0:
                self._samples.append((windows, label, bearing_id))

    def _build_flat_index(self) -> None:
        """Build flat list mapping global index → (sample_idx, window_idx)."""
        self._flat: list[tuple[int, int]] = []
        for sample_idx, (windows, label, bearing_id) in enumerate(self._samples):
            start = len(self._flat)
            for w in range(len(windows)):
                self._flat.append((sample_idx, w))
            end = len(self._flat)
            for idx in range(start, end):
                self._bearing_index[bearing_id].append(idx)

    # ------------------------------------------------------------------
    # Bearing-wise split — arXiv 2509.22267
    # ------------------------------------------------------------------

    def split_by_bearing(
        self,
        test_fraction: float = 0.2,
        seed: int = 42,
    ) -> tuple["CWRUBearingDataset", "CWRUBearingDataset"]:
        """
        Split the dataset into train and test subsets ensuring no bearing
        appears in both sets (bearing-wise partitioning).

        This is the ONLY valid evaluation split for bearing datasets. Random
        window-level splits cause data leakage that can inflate accuracy by
        up to 40 pp. See arXiv 2509.22267.

        Args:
            test_fraction: Fraction of bearings to hold out for testing.
            seed:          Random seed for reproducibility.

        Returns:
            train_dataset, test_dataset: Two Subset-wrapped datasets.
        """
        rng = random.Random(seed)
        bearing_ids = sorted(self._bearing_index.keys())
        rng.shuffle(bearing_ids)

        n_test = max(1, int(len(bearing_ids) * test_fraction))
        test_bearings = set(bearing_ids[:n_test])
        train_bearings = set(bearing_ids[n_test:])

        train_indices: list[int] = []
        test_indices: list[int] = []
        for bid in bearing_ids:
            indices = self._bearing_index[bid]
            if bid in test_bearings:
                test_indices.extend(indices)
            else:
                train_indices.extend(indices)

        logger.info(
            "Bearing-wise split: %d train bearings (%d windows), %d test bearings (%d windows)",
            len(train_bearings), len(train_indices),
            len(test_bearings), len(test_indices),
        )
        return Subset(self, train_indices), Subset(self, test_indices)

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._flat)

    def __getitem__(self, idx: int) -> tuple[Tensor, int]:
        sample_idx, window_idx = self._flat[idx]
        windows, label, _ = self._samples[sample_idx]
        window = windows[window_idx]  # (window_size,)

        x = torch.from_numpy(window).unsqueeze(0)  # (1, window_size)
        if self.transform is not None:
            x = self.transform(x)
        return x, label

    @property
    def bearing_ids(self) -> list[str]:
        return list(self._bearing_index.keys())

    @property
    def labels(self) -> list[int]:
        """Return label for every window in dataset order."""
        return [self._samples[s][1] for s, _ in self._flat]


# ---------------------------------------------------------------------------
# MFPT Dataset
# ---------------------------------------------------------------------------

class MFPTBearingDataset(Dataset):
    """
    PyTorch Dataset for the MFPT Bearing Fault Dataset.

    The MFPT dataset ships as .mat files with a struct containing:
        - gs: vibration signal (1D array)
        - sr: sample rate

    Directory structure::

        data_root/
            baseline/        normal condition (3 files)
            outer_race/      outer race fault (7 variable load files)
            inner_race/      inner race fault (optional)
            ball/            ball fault (optional)

    Args:
        data_root:   Root directory of the MFPT dataset.
        window_size: FFT window size in samples.
        overlap:     Overlap fraction.
        transform:   Optional callable applied to (1, window_size) tensor.
    """

    def __init__(
        self,
        data_root: Union[str, Path],
        window_size: int = WINDOW_SIZE,
        overlap: float = OVERLAP,
        transform=None,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.overlap = overlap
        self.transform = transform

        self._samples: list[tuple[np.ndarray, int, str]] = []
        self._bearing_index: dict[str, list[int]] = defaultdict(list)
        self._flat: list[tuple[int, int]] = []

        self._scan_directory(Path(data_root))
        self._build_flat_index()

    def _scan_directory(self, root: Path) -> None:
        for condition, label in MFPT_LABEL_MAP.items():
            condition_dir = root / condition
            if not condition_dir.exists():
                logger.debug("MFPT condition dir not found: %s", condition_dir)
                continue
            for i, mat_file in enumerate(sorted(condition_dir.glob("*.mat"))):
                try:
                    mat = scipy.io.loadmat(str(mat_file))
                except Exception as exc:
                    logger.warning("Failed loading %s: %s", mat_file, exc)
                    continue

                # MFPT struct layout
                signal = None
                for key in ("bearing", "Bearing"):
                    if key in mat:
                        try:
                            signal = mat[key]["gs"][0, 0].flatten().astype(np.float64)
                        except Exception:
                            pass
                        break
                if signal is None:
                    # Fallback: look for numeric arrays
                    for key, val in mat.items():
                        if not key.startswith("__") and isinstance(val, np.ndarray) and val.ndim <= 2:
                            signal = val.flatten().astype(np.float64)
                            break

                if signal is None or len(signal) < self.window_size:
                    continue

                bearing_id = f"mfpt_{condition}_{i:03d}"
                windows = segment_and_fft(signal, self.window_size, self.overlap)
                if len(windows) > 0:
                    self._samples.append((windows, label, bearing_id))

    def _build_flat_index(self) -> None:
        self._flat = []
        for sample_idx, (windows, label, bearing_id) in enumerate(self._samples):
            start = len(self._flat)
            for w in range(len(windows)):
                self._flat.append((sample_idx, w))
            end = len(self._flat)
            for idx in range(start, end):
                self._bearing_index[bearing_id].append(idx)

    def split_by_bearing(
        self,
        test_fraction: float = 0.2,
        seed: int = 42,
    ) -> tuple["MFPTBearingDataset", "MFPTBearingDataset"]:
        """Bearing-wise split — see CWRUBearingDataset.split_by_bearing docstring."""
        rng = random.Random(seed)
        bearing_ids = sorted(self._bearing_index.keys())
        rng.shuffle(bearing_ids)
        n_test = max(1, int(len(bearing_ids) * test_fraction))
        test_bearings = set(bearing_ids[:n_test])

        train_indices, test_indices = [], []
        for bid in bearing_ids:
            (test_indices if bid in test_bearings else train_indices).extend(self._bearing_index[bid])
        return Subset(self, train_indices), Subset(self, test_indices)

    def __len__(self) -> int:
        return len(self._flat)

    def __getitem__(self, idx: int) -> tuple[Tensor, int]:
        sample_idx, window_idx = self._flat[idx]
        windows, label, _ = self._samples[sample_idx]
        x = torch.from_numpy(windows[window_idx]).unsqueeze(0)
        if self.transform is not None:
            x = self.transform(x)
        return x, label

    @property
    def labels(self) -> list[int]:
        return [self._samples[s][1] for s, _ in self._flat]


# ---------------------------------------------------------------------------
# DatasetFactory
# ---------------------------------------------------------------------------

class DatasetFactory:
    """
    Factory for creating combined CWRU + MFPT train/test DataLoader pairs.

    All splits are bearing-wise to prevent data leakage (arXiv 2509.22267).
    """

    @staticmethod
    def combined_cwru_mfpt(
        cwru_root: Union[str, Path],
        mfpt_root: Union[str, Path],
        test_fraction: float = 0.2,
        batch_size: int = 256,
        num_workers: int = 4,
        pin_memory: bool = True,
        seed: int = 42,
    ) -> tuple[DataLoader, DataLoader, list[float]]:
        """
        Build combined CWRU + MFPT train and test DataLoaders.

        The train/test split is performed independently per dataset using
        bearing-wise partitioning, then the splits are concatenated.

        Args:
            cwru_root:       Path to CWRU dataset root.
            mfpt_root:       Path to MFPT dataset root.
            test_fraction:   Fraction of bearings held out per dataset.
            batch_size:      DataLoader batch size.
            num_workers:     DataLoader worker processes.
            pin_memory:      Pin memory for CUDA transfers.
            seed:            Random seed.

        Returns:
            train_loader, test_loader, class_weights
            class_weights is a list of length NUM_CLASSES for use with
            nn.CrossEntropyLoss(weight=...).
        """
        logger.info("Loading CWRU dataset from %s ...", cwru_root)
        cwru_ds = CWRUBearingDataset(data_root=cwru_root)
        cwru_train, cwru_test = cwru_ds.split_by_bearing(test_fraction=test_fraction, seed=seed)

        logger.info("Loading MFPT dataset from %s ...", mfpt_root)
        mfpt_ds = MFPTBearingDataset(data_root=mfpt_root)
        mfpt_train, mfpt_test = mfpt_ds.split_by_bearing(test_fraction=test_fraction, seed=seed)

        train_ds = ConcatDataset([cwru_train, mfpt_train])
        test_ds = ConcatDataset([cwru_test, mfpt_test])

        # Class weights for imbalanced dataset
        class_weights = DatasetFactory._compute_class_weights(
            cwru_train, cwru_ds,
            mfpt_train, mfpt_ds,
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        logger.info(
            "DataLoaders ready: %d train windows, %d test windows",
            len(train_ds), len(test_ds),
        )
        return train_loader, test_loader, class_weights

    @staticmethod
    def _compute_class_weights(
        cwru_train: Subset, cwru_ds: CWRUBearingDataset,
        mfpt_train: Subset, mfpt_ds: MFPTBearingDataset,
    ) -> list[float]:
        """
        Compute inverse-frequency class weights from the training subset.
        """
        label_counts = np.zeros(NUM_CLASSES, dtype=np.float64)
        for idx in cwru_train.indices:
            sample_idx, _ = cwru_ds._flat[idx]
            label = cwru_ds._samples[sample_idx][1]
            label_counts[label] += 1
        for idx in mfpt_train.indices:
            sample_idx, _ = mfpt_ds._flat[idx]
            label = mfpt_ds._samples[sample_idx][1]
            label_counts[label] += 1

        # Inverse-frequency weighting; classes with zero examples get weight 0
        total = label_counts.sum()
        weights = np.where(
            label_counts > 0,
            total / (NUM_CLASSES * label_counts),
            0.0,
        )
        return weights.tolist()
