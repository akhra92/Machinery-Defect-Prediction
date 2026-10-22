"""Find and parse DCASE Task 2 (2022-2024) style audio clips.

Expected layout:
    <data_root>/<machine>/train/section_00_source_train_normal_0000_<attrs>.wav
    <data_root>/<machine>/test/section_00_target_test_anomaly_0012_<attrs>.wav

Unlabelled evaluation files (e.g. section_00_0000.wav) get label=None.
"""
import re
from dataclasses import dataclass
from pathlib import Path

_PATTERN = re.compile(
    r"section_(?P<section>\d+)_(?P<domain>source|target)_(?P<split>train|test)"
    r"_(?P<label>normal|anomaly)_(?P<idx>\d+)(?:_(?P<attrs>.*))?\.wav$"
)


@dataclass(frozen=True)
class Clip:
    path: Path
    machine: str
    section: str
    domain: str | None  # "source" | "target" | None
    label: int | None   # 0 = normal, 1 = anomaly, None = unknown
    attrs: str = ""


def parse_clip(path: Path, machine: str) -> Clip:
    m = _PATTERN.search(path.name)
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
