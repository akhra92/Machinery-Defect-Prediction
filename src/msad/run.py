"""Train an anomaly detector on normal clips and evaluate it on the labelled test set.

    python -m msad.run --machine fan --model ae
    python -m msad.run --machine all --model gmm knn ae
    python -m msad.run --machine fan --model ae --set ae.epochs=20 data_root=data/other
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import yaml

from .data import list_clips, list_machines
from .features import FeatureCache
from .metrics import evaluate
from .models import build


def load_config(path: str, overrides: list[str]) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    for item in overrides:
        key, _, value = item.partition("=")
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node[p]
        node[leaf] = yaml.safe_load(value)
    return cfg


def run_one(machine: str, model_name: str, cfg: dict, cache: FeatureCache) -> dict:
    out_dir = Path(cfg["output_dir"]) / machine / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    train = [c for c in list_clips(cfg["data_root"], machine, "train") if c.label in (0, None)]
    test = list_clips(cfg["data_root"], machine, "test")
    train_specs = cache.get_many([c.path for c in train], desc=f"{machine} train features")
    test_specs = cache.get_many([c.path for c in test], desc=f"{machine} test features")

    t0 = time.time()
    model = build(model_name, cfg).fit(train_specs)
    fit_seconds = time.time() - t0

    threshold = float(np.percentile(model.score(train_specs), cfg["threshold_percentile"]))
    test_scores = model.score(test_specs)

    model.save(out_dir / "model.joblib")
    (out_dir / "threshold.json").write_text(json.dumps({"threshold": threshold}))
    with open(out_dir / "test_scores.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "domain", "label", "score", "predicted_anomaly"])
        for c, s in zip(test, test_scores):
            w.writerow([c.path.name, c.domain, c.label, f"{s:.6f}", int(s > threshold)])
    if getattr(model, "history", None):
        (out_dir / "history.json").write_text(json.dumps(model.history, indent=1))

    result = {"machine": machine, "model": model_name, "n_train": len(train), "n_test": len(test),
              "fit_seconds": round(fit_seconds, 1)}
    if all(c.label is not None for c in test):
        result |= evaluate(test, test_scores, threshold, cfg["max_fpr"])
    (out_dir / "results.json").write_text(json.dumps(result, indent=1))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--machine", nargs="+", required=True, help="machine type(s) or 'all'")
    parser.add_argument("--model", nargs="+", default=["ae"], choices=["gmm", "knn", "ae"])
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config values")
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    np.random.seed(cfg["seed"])
    machines = list_machines(cfg["data_root"]) if args.machine == ["all"] else args.machine
    if not machines:
        parser.error(f"no machine folders found under {cfg['data_root']}")
    cache = FeatureCache(cfg["cache_dir"], cfg["features"])

    results = []
    for machine in machines:
        for model_name in args.model:
            r = run_one(machine, model_name, cfg, cache)
            results.append(r)
            metrics = "  ".join(f"{k}={r[k]:.3f}" for k in ("auc_source", "auc_target", "pauc", "hmean", "f1") if k in r)
            print(f"[{machine:>9} | {model_name:>3}] {metrics}  ({r['fit_seconds']}s)")

    summary = Path(cfg["output_dir"]) / "results.csv"
    fields = ["machine", "model", "auc_source", "auc_target", "auc_all", "pauc", "hmean",
              "precision", "recall", "f1", "n_train", "n_test", "fit_seconds"]
    new_file = not summary.exists()
    with open(summary, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if new_file:
            w.writeheader()
        w.writerows(results)
    print(f"\nAppended {len(results)} result row(s) to {summary}")


if __name__ == "__main__":
    main()
