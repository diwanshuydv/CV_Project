"""Unzip the Kaggle archives into the layout expected by sam2unet/datasets.py.

    data/DRIVE/training/{images,1st_manual,mask}, data/DRIVE/test/...
    data/roads/tiff/{train,val,test}{,_labels}

Usage: python scripts/prepare_data.py [--data_root /scratch/b23cs1017/cv/data]
"""
import argparse
import zipfile
from pathlib import Path


def extract(zip_path: Path, dest: Path, name: str) -> None:
    marker = dest / f".{name}.extracted"   # written only after a complete extraction
    if marker.exists():
        print(f"[skip] {zip_path.name} already extracted")
        return
    print(f"[unzip] {zip_path.name} -> {dest}", flush=True)
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)
    marker.touch()


def count(d: Path) -> int:
    return len(list(d.glob("*"))) if d.exists() else 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="/scratch/b23cs1017/cv/data")
    root = Path(ap.parse_args().data_root)

    extract(root / "drive-digital-retinal-images-for-vessel-extraction.zip", root, "drive")
    extract(root / "massachusetts-roads-dataset.zip", root / "roads", "roads")

    for sub in ["DRIVE/training/images", "DRIVE/training/1st_manual", "DRIVE/training/mask",
                "DRIVE/test/images", "roads/tiff/train", "roads/tiff/train_labels",
                "roads/tiff/val", "roads/tiff/val_labels", "roads/tiff/test", "roads/tiff/test_labels"]:
        print(f"{sub:32s} {count(root / sub)}")


if __name__ == "__main__":
    main()
