"""Log-mel spectrogram features, with an on-disk cache so each clip is processed once."""
import hashlib
import json
from pathlib import Path

import librosa
import numpy as np
from tqdm import tqdm


def log_mel(path: str | Path, cfg: dict) -> np.ndarray:
    """Return a (n_mels, n_time) log-mel spectrogram in dB, as in the DCASE baseline."""
    y, _ = librosa.load(path, sr=cfg["sample_rate"], mono=True)
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=cfg["sample_rate"],
        n_fft=cfg["n_fft"],
        hop_length=cfg["hop_length"],
        n_mels=cfg["n_mels"],
        power=cfg["power"],
    )
    return (20.0 / cfg["power"] * np.log10(mel + np.finfo(np.float32).eps)).astype(np.float32)


class FeatureCache:
    def __init__(self, cache_dir: str | Path, cfg: dict):
        self.cfg = cfg
        key = {k: cfg[k] for k in ("sample_rate", "n_fft", "hop_length", "n_mels", "power")}
        tag = hashlib.md5(json.dumps(key, sort_keys=True).encode()).hexdigest()[:8]
        self.dir = Path(cache_dir) / tag
        self.dir.mkdir(parents=True, exist_ok=True)

    def get(self, path: str | Path) -> np.ndarray:
        path = Path(path)
        name = hashlib.md5(str(path.resolve()).encode()).hexdigest() + ".npy"
        cached = self.dir / name
        if cached.exists():
            return np.load(cached)
        spec = log_mel(path, self.cfg)
        np.save(cached, spec)
        return spec

    def get_many(self, paths, desc: str = "features") -> list[np.ndarray]:
        return [self.get(p) for p in tqdm(paths, desc=desc, leave=False)]


def stack_frames(spec: np.ndarray, n_frames: int) -> np.ndarray:
    """(n_mels, T) -> (T - n_frames + 1, n_mels * n_frames) context windows for the autoencoder."""
    n_mels, t = spec.shape
    n_vectors = t - n_frames + 1
    if n_vectors < 1:
        raise ValueError(f"clip too short: {t} frames < n_frames={n_frames}")
    out = np.empty((n_vectors, n_mels * n_frames), dtype=np.float32)
    for i in range(n_frames):
        out[:, n_mels * i : n_mels * (i + 1)] = spec[:, i : i + n_vectors].T
    return out


def clip_stats(spec: np.ndarray) -> np.ndarray:
    """One vector per clip: mean and std of each mel band over time."""
    return np.concatenate([spec.mean(axis=1), spec.std(axis=1)]).astype(np.float32)
