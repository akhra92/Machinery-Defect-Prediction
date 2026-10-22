"""Download and unzip the DCASE 2024 Task 2 development dataset from Zenodo.

Dataset: https://doi.org/10.5281/zenodo.10902294  (license: CC BY-NC-SA 4.0, non-commercial)
Each machine type is ~300-400 MB zipped.

Usage:
    python scripts/download_dcase2024.py --machines fan valve
    python scripts/download_dcase2024.py --machines all
"""
import argparse
import urllib.request
import zipfile
from pathlib import Path

from tqdm import tqdm

RECORD_URL = "https://zenodo.org/api/records/10902294/files/dev_{name}.zip/content"
MACHINES = ["bearing", "fan", "gearbox", "slider", "valve", "ToyCar", "ToyTrain"]


def download(url: str, dest: Path) -> None:
    with urllib.request.urlopen(url) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        tmp = dest.with_suffix(".part")
        with open(tmp, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar:
            while chunk := resp.read(1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    tmp.rename(dest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--machines", nargs="+", default=["fan"], help=f"any of {MACHINES} or 'all'")
    parser.add_argument("--out", default="data/dcase2024_dev")
    parser.add_argument("--keep-zip", action="store_true")
    args = parser.parse_args()

    machines = MACHINES if args.machines == ["all"] else args.machines
    unknown = set(machines) - set(MACHINES)
    if unknown:
        parser.error(f"unknown machine types: {sorted(unknown)}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in machines:
        if (out / name / "train").is_dir():
            print(f"{name}: already present, skipping")
            continue
        zip_path = out / f"dev_{name}.zip"
        if not zip_path.exists():
            download(RECORD_URL.format(name=name), zip_path)
        print(f"{name}: extracting")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(out)
        if not args.keep_zip:
            zip_path.unlink()


if __name__ == "__main__":
    main()
