"""Anomaly detectors. All are trained on normal clips only and return one score per clip
(higher = more anomalous).

- "gmm": Gaussian mixture on clip-level log-mel statistics; score = negative log-likelihood
- "knn": distance to the k-th nearest normal training clip in the same feature space
- "ae":  DCASE-style dense autoencoder on stacked log-mel frames; score = reconstruction error
"""
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from torch import nn
from tqdm import tqdm

from .features import clip_stats, stack_frames


class Detector:
    def fit(self, specs: list[np.ndarray]) -> "Detector":
        raise NotImplementedError

    def score(self, specs: list[np.ndarray]) -> np.ndarray:
        raise NotImplementedError

    def save(self, path: Path) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> "Detector":
        return joblib.load(path)


class ClipStatsDetector(Detector):
    def _features(self, specs):
        return np.stack([clip_stats(s) for s in specs])

    def fit(self, specs):
        x = self._features(specs)
        self.scaler = StandardScaler().fit(x)
        self._fit(self.scaler.transform(x))
        return self

    def score(self, specs):
        return self._score(self.scaler.transform(self._features(specs)))


class GMMDetector(ClipStatsDetector):
    def __init__(self, n_components=4, covariance_type="diag", seed=0):
        self.gmm = GaussianMixture(n_components, covariance_type=covariance_type, random_state=seed)

    def _fit(self, x):
        self.gmm.fit(x)

    def _score(self, x):
        return -self.gmm.score_samples(x)


class KNNDetector(ClipStatsDetector):
    def __init__(self, k=1):
        self.k = k

    def _fit(self, x):
        self.nn = NearestNeighbors(n_neighbors=self.k).fit(x)

    def _score(self, x):
        dist, _ = self.nn.kneighbors(x)
        return dist[:, -1]


class _DenseAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_hidden_layers, bottleneck_dim):
        super().__init__()

        def block(i, o):
            return [nn.Linear(i, o), nn.BatchNorm1d(o), nn.ReLU()]

        enc, dec = [], []
        dims = [input_dim] + [hidden_dim] * n_hidden_layers
        for i, o in zip(dims[:-1], dims[1:]):
            enc += block(i, o)
        enc += block(hidden_dim, bottleneck_dim)
        dec += block(bottleneck_dim, hidden_dim)
        for _ in range(n_hidden_layers - 1):
            dec += block(hidden_dim, hidden_dim)
        dec.append(nn.Linear(hidden_dim, input_dim))
        self.net = nn.Sequential(*enc, *dec)

    def forward(self, x):
        return self.net(x)


class AutoencoderDetector(Detector):
    def __init__(self, n_frames=5, hidden_dim=128, n_hidden_layers=4, bottleneck_dim=8,
                 epochs=100, batch_size=512, lr=1e-3, val_fraction=0.1, seed=0, device=None):
        self.n_frames = n_frames
        self.arch = dict(hidden_dim=hidden_dim, n_hidden_layers=n_hidden_layers, bottleneck_dim=bottleneck_dim)
        self.epochs, self.batch_size, self.lr = epochs, batch_size, lr
        self.val_fraction, self.seed = val_fraction, seed
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.history: list[dict] = []

    def _vectors(self, spec):
        return (stack_frames(spec, self.n_frames) - self.mean) / self.std

    def fit(self, specs):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        order = rng.permutation(len(specs))
        n_val = int(len(specs) * self.val_fraction)
        val_idx, train_idx = order[:n_val], order[n_val:]

        raw = np.concatenate([stack_frames(specs[i], self.n_frames) for i in train_idx])
        self.mean, self.std = raw.mean(axis=0), raw.std(axis=0) + 1e-6
        x_train = torch.from_numpy((raw - self.mean) / self.std).to(self.device)
        del raw
        x_val = (torch.from_numpy(np.concatenate([self._vectors(specs[i]) for i in val_idx])).to(self.device)
                 if n_val else None)

        self.model = _DenseAE(x_train.shape[1], **self.arch).to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        best_val, best_state = float("inf"), None

        bar = tqdm(range(self.epochs), desc=f"AE on {self.device}", leave=False)
        for epoch in bar:
            self.model.train()
            perm = torch.randperm(len(x_train), device=self.device)
            total = 0.0
            for start in range(0, len(x_train), self.batch_size):
                batch = x_train[perm[start : start + self.batch_size]]
                if len(batch) < 2:  # BatchNorm needs >1 sample
                    continue
                loss = loss_fn(self.model(batch), batch)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item() * len(batch)
            row = {"epoch": epoch + 1, "train_loss": total / len(x_train)}
            if x_val is not None:
                row["val_loss"] = self._mse(x_val).mean().item()
                if row["val_loss"] < best_val:
                    best_val = row["val_loss"]
                    best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
            self.history.append(row)
            bar.set_postfix({k: f"{v:.4f}" for k, v in row.items() if k != "epoch"})

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    @torch.no_grad()
    def _mse(self, x):
        self.model.eval()
        return torch.cat([((self.model(x[i : i + 8192]) - x[i : i + 8192]) ** 2).mean(dim=1)
                          for i in range(0, len(x), 8192)])

    def score(self, specs):
        return np.array([
            self._mse(torch.from_numpy(self._vectors(s)).to(self.device)).mean().item() for s in specs
        ])

    def save(self, path):
        self.model.cpu()
        device, self.device = self.device, "cpu"
        super().save(path)
        self.device = device
        self.model.to(device)


def build(name: str, cfg: dict) -> Detector:
    seed = cfg.get("seed", 0)
    if name == "gmm":
        return GMMDetector(**cfg["gmm"], seed=seed)
    if name == "knn":
        return KNNDetector(**cfg["knn"])
    if name == "ae":
        return AutoencoderDetector(n_frames=cfg["features"]["n_frames"], **cfg["ae"], seed=seed)
    raise ValueError(f"unknown model {name!r}; choose gmm, knn or ae")
