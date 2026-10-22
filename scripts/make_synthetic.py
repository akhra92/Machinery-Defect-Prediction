"""Generate a tiny synthetic dataset in DCASE layout to smoke-test the pipeline without downloading.

Normal = motor hum (harmonics) + noise. Anomaly = the same plus a bearing-like whine or periodic clicks.
Target-domain clips run at a different speed (domain shift).

    python scripts/make_synthetic.py --out data/synthetic
    python -m msad.run --machine synth --model gmm knn ae --set data_root=data/synthetic ae.epochs=5
"""
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16000


def machine_sound(rng, f0, seconds, anomaly):
    t = np.arange(int(SR * seconds)) / SR
    y = sum(np.sin(2 * np.pi * f0 * h * t + rng.uniform(0, 6.3)) / h for h in range(1, 8))
    y = y * (1 + 0.05 * np.sin(2 * np.pi * 3 * t)) + rng.normal(0, 0.3, t.size)
    if anomaly:
        if rng.random() < 0.5:
            y += 0.25 * np.sin(2 * np.pi * rng.uniform(2500, 4000) * t)
        else:
            clicks = np.zeros_like(t)
            clicks[:: int(SR / rng.uniform(15, 30))] = 3.0
            y += np.convolve(clicks, np.hanning(64), mode="same")
    return (0.1 * y / np.abs(y).max()).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/synthetic")
    parser.add_argument("--machine", default="synth")
    parser.add_argument("--seconds", type=float, default=2.0)
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    root = Path(args.out) / args.machine
    speeds = {"source": 50.0, "target": 60.0}
    plan = [("train", "source", "normal", 90), ("train", "target", "normal", 10)] + [
        ("test", d, lab, 25) for d in speeds for lab in ("normal", "anomaly")
    ]
    for split, domain, label, n in plan:
        (root / split).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            f0 = speeds[domain] * rng.uniform(0.97, 1.03)
            y = machine_sound(rng, f0, args.seconds, label == "anomaly")
            name = f"section_00_{domain}_{split}_{label}_{i:04d}_speed_{int(speeds[domain])}.wav"
            sf.write(root / split / name, y, SR)
    print(f"wrote synthetic dataset to {root}")


if __name__ == "__main__":
    main()
