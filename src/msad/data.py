"""Datasets. Each one lists clips and loads raw signals for the modalities it has.

DCASE 2024 Task 2 (sound only):
    <data_root>/<machine>/train/section_00_source_train_normal_0000_<attrs>.wav
    <data_root>/<machine>/test/section_00_target_test_anomaly_0012_<attrs>.wav
    Unlabelled evaluation files (e.g. section_00_0000.wav) get label=None.

MaFaulDa (sound + vibration, recorded together on one machine fault simulator):
    <data_root>/normal/<rotation_hz>.csv
    <data_root>/imbalance/10g/<rotation_hz>.csv
    <data_root>/underhang/outer_race/6g/<rotation_hz>.csv
    CSV columns: 0 tachometer, 1-3 underhang accelerometer (axial, radial, tangential),
    4-6 overhang accelerometer (axial, radial, tangential), 7 microphone. 50 kHz, 5 s.
"""
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

MODALITIES = ("sound", "vibration")


@dataclass(frozen=True)
class Clip:
    path: Path
    machine: str
    section: str
    domain: str | None  # "source" | "target" | None
    label: int | None   # 0 = normal, 1 = anomaly, None = unknown
    attrs: str = ""
    fault: str = ""     # fault type for anomalous clips, when the dataset provides it
    segment: int | None = None  # index of a fixed-length segment within the file


# ---------------------------------------------------------------- DCASE 2024 Task 2

_DCASE_PATTERN = re.compile(
    r"section_(?P<section>\d+)_(?P<domain>source|target)_(?P<split>train|test)"
    r"_(?P<label>normal|anomaly)_(?P<idx>\d+)(?:_(?P<attrs>.*))?\.wav$"
)


def parse_clip(path: Path, machine: str) -> Clip:
    m = _DCASE_PATTERN.search(path.name)
    if m is None:
        section = re.search(r"section_(\d+)", path.name)
        return Clip(path, machine, section.group(1) if section else "00", None, None)
    return Clip(
        path=path,
        machine=machine,
        section=m["section"],
        domain=m["domain"],
        label=int(m["label"] == "anomaly"),
        attrs=m["attrs"] or "",
    )


def list_clips(data_root: str | Path, machine: str, split: str) -> list[Clip]:
    folder = Path(data_root) / machine / split
    if not folder.is_dir():
        raise FileNotFoundError(
            f"{folder} not found. Download data first: python scripts/download_dcase2024.py --machines {machine}"
        )
    clips = [parse_clip(p, machine) for p in sorted(folder.glob("*.wav"))]
    if not clips:
        raise FileNotFoundError(f"no .wav files in {folder}")
    return clips


def list_machines(data_root: str | Path) -> list[str]:
    root = Path(data_root)
    return sorted(p.name for p in root.iterdir() if (p / "train").is_dir()) if root.is_dir() else []


class DCASE:
    modalities = ("sound",)

    def __init__(self, cfg: dict):
        self.root = Path(cfg["data_root"])

    def machines(self) -> list[str]:
        return list_machines(self.root)

    def clips(self, machine: str, split: str) -> list[Clip]:
        return list_clips(self.root, machine, split)

    def file_clips(self, path: str | Path) -> list[Clip]:
        path = Path(path)
        return [parse_clip(path, path.parent.parent.name)]

    def load(self, clip: Clip, modality: str) -> tuple[np.ndarray, int]:
        """Return (channels, samples) signal and its sample rate."""
        y, sr = librosa.load(clip.path, sr=None, mono=True)
        return y[None, :], sr


# ---------------------------------------------------------------- MaFaulDa

_BEARING_GROUPS = ("underhang", "overhang")


@lru_cache(maxsize=None)
def _count_rows(path: Path) -> int:
    """Number of samples in a CSV without parsing it (much faster than reading the numbers)."""
    data = path.read_bytes()
    return data.count(b"\n") + (0 if data.endswith(b"\n") else 1)


def _fault_type(rel: Path) -> str:
    """normal / imbalance / horizontal-misalignment / vertical-misalignment / underhang/outer_race / ..."""
    parts = rel.parts
    return "/".join(parts[:2]) if parts[0] in _BEARING_GROUPS else parts[0]


class MaFaulDa:
    modalities = ("sound", "vibration")
    machine = "mafaulda"
    sample_rate = 50000

    def __init__(self, cfg: dict):
        self.root = Path(cfg["data_root"])
        self.cfg = cfg["mafaulda"]
        self.seed = cfg.get("seed", 0)
        self._splits: dict[str, list[tuple[Path, int, str]]] | None = None

    def machines(self) -> list[str]:
        return [self.machine]

    def _make_splits(self) -> dict[str, list[tuple[Path, int, str]]]:
        """Split by file (never by segment) so no recording appears in both train and test."""
        if not (self.root / "normal").is_dir():
            raise FileNotFoundError(
                f"{self.root / 'normal'} not found. Download data first: python scripts/download_mafaulda.py"
            )
        rng = np.random.default_rng(self.seed)
        normal = sorted((self.root / "normal").glob("*.csv"))
        order = rng.permutation(len(normal))
        n_test = max(1, round(len(normal) * self.cfg["test_fraction"]))
        test = [(normal[i], 0, "") for i in sorted(order[:n_test])]
        train = [(normal[i], 0, "") for i in sorted(order[n_test:])]

        by_fault: dict[str, list[Path]] = {}
        for p in sorted(self.root.rglob("*.csv")):
            fault = _fault_type(p.relative_to(self.root))
            if fault != "normal":
                by_fault.setdefault(fault, []).append(p)
        limit = self.cfg.get("files_per_fault")
        for fault, files in sorted(by_fault.items()):
            if limit and len(files) > limit:
                files = [files[i] for i in sorted(rng.choice(len(files), limit, replace=False))]
            test += [(p, 1, fault) for p in files]
        return {"train": train, "test": test}

    def _segments(self, path: Path, label: int | None, fault: str) -> list[Clip]:
        n_samples = _count_rows(path)
        seg_len = int(self.cfg["segment_seconds"] * self.sample_rate)
        speed = path.stem
        return [Clip(path, self.machine, "00", None, label, f"{speed}Hz", fault, i)
                for i in range(max(1, n_samples // seg_len))]

    def clips(self, machine: str, split: str) -> list[Clip]:
        if self._splits is None:
            self._splits = self._make_splits()
        return [c for p, label, fault in self._splits[split] for c in self._segments(p, label, fault)]

    def file_clips(self, path: str | Path) -> list[Clip]:
        return self._segments(Path(path), None, "")

    @staticmethod
    @lru_cache(maxsize=4)  # segments of one file are loaded one after another
    def _read(path: Path) -> np.ndarray:
        return pd.read_csv(path, header=None, dtype=np.float32, engine="c").to_numpy()

    def load(self, clip: Clip, modality: str) -> tuple[np.ndarray, int]:
        data = self._read(clip.path)
        if clip.segment is not None:
            seg_len = int(self.cfg["segment_seconds"] * self.sample_rate)
            data = data[clip.segment * seg_len : (clip.segment + 1) * seg_len]
        cols = self.cfg["vibration_channels"] if modality == "vibration" else [self.cfg["sound_channel"]]
        return data[:, cols].T.copy(), self.sample_rate


DATASETS = {"dcase2024": DCASE, "mafaulda": MaFaulDa}


def get_dataset(cfg: dict):
    name = cfg.get("dataset", "dcase2024")
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}; choose from {sorted(DATASETS)}")
    return DATASETS[name](cfg)
