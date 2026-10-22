"""Score new recordings with a trained detector.

    python -m msad.predict --model-dir outputs/fan/ae recordings/*.wav
"""
import argparse
import json
from pathlib import Path

import yaml

from .features import log_mel
from .models import Detector


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("files", nargs="+")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    cfg = yaml.safe_load(Path(args.config).read_text())
    model = Detector.load(model_dir / "model.joblib")
    threshold = json.loads((model_dir / "threshold.json").read_text())["threshold"]

    specs = [log_mel(f, cfg["features"]) for f in args.files]
    for f, s in zip(args.files, model.score(specs)):
        print(f"{'ANOMALY' if s > threshold else 'normal ':7}  score={s:.4f}  {f}")


if __name__ == "__main__":
    main()
