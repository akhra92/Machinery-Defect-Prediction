"""Download and unzip MaFaulDa (Machinery Fault Database, UFRJ) - sound + vibration.

Site: https://www02.smt.ufrj.br/~offshore/mfs/page_01.html  (cite the authors when publishing results)
Compressed sizes: normal 0.3 GB, imbalance 2.2 GB, horizontal-misalignment 1.3 GB,
vertical-misalignment 2.0 GB, underhang 3.7 GB, overhang 3.4 GB (12.9 GB total, ~35 GB unzipped).

Usage:
    python scripts/download_mafaulda.py                       # everything
    python scripts/download_mafaulda.py --classes normal imbalance
"""
import argparse
import urllib.request
import zipfile
from pathlib import Path

from tqdm import tqdm

URL = "https://www02.smt.ufrj.br/~offshore/mfs/database/mafaulda/{name}.zip"
CLASSES = ["normal", "imbalance", "horizontal-misalignment", "vertical-misalignment", "underhang", "overhang"]


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
    parser.add_argument("--classes", nargs="+", default=CLASSES, choices=CLASSES)
    parser.add_argument("--out", default="data/mafaulda")
    parser.add_argument("--keep-zip", action="store_true")
    args = parser.parse_args()
    if "normal" not in args.classes:
        print("note: 'normal' is required for training; add it unless you already have it")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in args.classes:
        if (out / name).is_dir():
            print(f"{name}: already present, skipping")
            continue
        zip_path = out / f"{name}.zip"
        if not zip_path.exists():
            download(URL.format(name=name), zip_path)
        print(f"{name}: extracting")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(out)  # archives contain a top-level <class>/ folder
        if not args.keep_zip:
            zip_path.unlink()


if __name__ == "__main__":
    main()
