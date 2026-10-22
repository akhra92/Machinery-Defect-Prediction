"""DCASE 2023/2024 Task 2 style evaluation.

- AUC per domain d: normal test clips from domain d vs. all anomalous test clips
- pAUC: all test clips, false-positive rate limited to [0, max_fpr]
- hmean: harmonic mean of AUC(source), AUC(target) and pAUC; for datasets without domains
  (e.g. MaFaulDa) it is the harmonic mean of overall AUC and pAUC
- auc_fault/<type>: normal test clips vs. anomalies of one fault type, when fault types are known
Also reports precision/recall/F1 at the decision threshold.
"""
import numpy as np
from scipy.stats import hmean
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from .data import Clip


def evaluate(clips: list[Clip], scores: np.ndarray, threshold: float, max_fpr: float = 0.1) -> dict:
    labels = np.array([c.label for c in clips])
    if any(l is None for l in labels):
        raise ValueError("test clips have no normal/anomaly labels; cannot evaluate")
    labels = labels.astype(int)
    domains = np.array([c.domain for c in clips])
    faults = np.array([c.fault for c in clips])
    anomalous = labels == 1

    out: dict[str, float] = {}
    for d in ("source", "target"):
        mask = ((domains == d) & ~anomalous) | anomalous
        if (labels[mask] == 0).any() and anomalous.any():
            out[f"auc_{d}"] = roc_auc_score(labels[mask], scores[mask])
    out["auc_all"] = roc_auc_score(labels, scores)
    out["pauc"] = roc_auc_score(labels, scores, max_fpr=max_fpr)
    domain_aucs = [out[k] for k in ("auc_source", "auc_target") if k in out]
    out["hmean"] = float(hmean((domain_aucs or [out["auc_all"]]) + [out["pauc"]]))

    for fault in sorted(set(faults[anomalous]) - {""}):
        mask = ~anomalous | (faults == fault)
        out[f"auc_fault/{fault}"] = roc_auc_score(labels[mask], scores[mask])

    pred = (scores > threshold).astype(int)
    out["precision"] = precision_score(labels, pred, zero_division=0)
    out["recall"] = recall_score(labels, pred, zero_division=0)
    out["f1"] = f1_score(labels, pred, zero_division=0)
    return {k: float(v) for k, v in out.items()}
