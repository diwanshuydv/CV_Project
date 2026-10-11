"""
Step 1 -- Datasets & dataloaders for thin-structure binary segmentation.

Supported datasets
------------------
* DRIVE (retinal vessels, Kaggle mirror layout)
      DRIVE/training/images/21_training.tif
      DRIVE/training/1st_manual/21_manual1.gif      <- vessel ground truth
      DRIVE/training/mask/21_training_mask.gif      <- field-of-view (FOV) mask
  The public test split ships *without* vessel labels, so the 20 labelled
  training images are split deterministically into train/val/test
  (default 14/2/4, or k-fold cross-validation via ``fold``).

* Massachusetts Roads (Kaggle mirror layout)
      roads/tiff/{train,val,test}/<id>.tiff
      roads/tiff/{train,val,test}_labels/<id>.tif
  Images are 1500x1500 aerial tiles; some training tiles contain large white
  "no-data" regions which are treated as outside the FOV.

Every sample is returned as a dict::

    image : float32 [3, H, W]  ImageNet-normalised RGB
    mask  : float32 [1, H, W]  binary ground truth {0, 1}
    dist  : float32 [1, H, W]  signed Euclidean distance map of the GT boundary,
                               negative inside the structure, positive outside,
                               clipped to +-max_dist and divided by max_dist -> [-1, 1]
    fov   : float32 [1, H, W]  valid-pixel mask (DRIVE FOV / non-white road pixels)
    name  : str

The distance map is computed on the CPU (inside the DataLoader workers) with
``scipy.ndimage.distance_transform_edt`` *after* all augmentations, so it is
always consistent with the mask the network actually sees.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import distance_transform_edt
from torch.utils.data import DataLoader, Dataset

try:  # tifffile reads the large road TIFFs faster/more reliably than PIL
    import tifffile
except ImportError:  # pragma: no cover
    tifffile = None

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

DEFAULT_DATA_ROOT = os.environ.get("SAM2UNET_DATA", "/scratch/b23cs1017/cv/data")


# --------------------------------------------------------------------------- #
#                               Image I/O helpers                             #
# --------------------------------------------------------------------------- #
def _imread(path: Path) -> np.ndarray:
    """tifffile for TIFFs (LZW-compressed DRIVE TIFFs need ``imagecodecs``), PIL otherwise/fallback."""
    if path.suffix.lower() in (".tif", ".tiff") and tifffile is not None:
        try:
            return tifffile.imread(str(path))
        except ValueError:  # missing codec -> let PIL decode it
            pass
    return np.array(Image.open(path))


def _read_image(path: Path) -> np.ndarray:
    """Read an image as uint8 HxWx3 (RGB)."""
    arr = _imread(path)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[-1] == 4:  # drop alpha
        arr = arr[..., :3]
    if arr.dtype != np.uint8:  # e.g. 16-bit TIFF -> rescale to 8 bit
        arr = arr.astype(np.float32)
        arr = (255.0 * (arr - arr.min()) / max(arr.max() - arr.min(), 1e-6)).astype(np.uint8)
    return arr


def _read_binary(path: Path) -> np.ndarray:
    """Read a label / FOV image as uint8 HxW in {0, 1}."""
    arr = _imread(path)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return (arr > 127).astype(np.uint8) if arr.max() > 1 else (arr > 0).astype(np.uint8)


# --------------------------------------------------------------------------- #
#                         Euclidean distance transform                        #
# --------------------------------------------------------------------------- #
def signed_distance_map(mask: np.ndarray, max_dist: float = 20.0) -> np.ndarray:
    """Signed, clipped, normalised Euclidean distance to the GT boundary.

    phi(x) =  EDT(1 - g)(x)          for background pixels  (> 0)
    phi(x) = -(EDT(g)(x) - 1)        for foreground pixels  (<= 0)

    ``distance_transform_edt`` returns, for every non-zero pixel, the distance to
    the nearest zero pixel.  For a background pixel EDT(1-g) is therefore the
    distance to the nearest structure pixel (>= 1).  For foreground pixels we
    subtract 1 so that the outermost ring of a vessel/road has phi = 0, i.e. the
    zero level-set lies exactly on the structure's boundary pixels -- important
    for 1-2 px wide capillaries where every pixel is a boundary pixel.

    Degenerate cases (empty or full mask, common for random crops) have no
    boundary; every pixel is then assigned the maximal distance with the
    appropriate sign.

    Returns float32 HxW in [-1, 1].
    """
    g = mask.astype(bool)
    if not g.any():
        return np.ones(g.shape, dtype=np.float32)
    if g.all():
        return -np.ones(g.shape, dtype=np.float32)
    d_out = distance_transform_edt(~g)            # distance of bg pixels to fg
    d_in = distance_transform_edt(g) - 1.0         # depth of fg pixels inside fg
    phi = np.where(g, -d_in, d_out).astype(np.float32)
    return np.clip(phi, -max_dist, max_dist) / max_dist


# --------------------------------------------------------------------------- #
#                         Dataset "sources" (file lists)                      #
# --------------------------------------------------------------------------- #
@dataclass
class SampleRecord:
    name: str
    image: Path
    label: Path
    fov: Optional[Path] = None   # None -> computed from the image (roads)


class DriveSource:
    """File lists for DRIVE with a deterministic split of the 20 labelled images."""

    def __init__(self, root: str, split: str, n_val: int = 2, n_test: int = 4,
                 seed: int = 0, fold: Optional[int] = None, n_folds: int = 5):
        base = Path(root) / "DRIVE" / "training"
        ids = sorted(p.name.split("_")[0] for p in (base / "images").glob("*_training.tif"))
        if not ids:
            raise FileNotFoundError(f"No DRIVE images found under {base}")
        records = {
            i: SampleRecord(
                name=f"drive_{i}",
                image=base / "images" / f"{i}_training.tif",
                label=base / "1st_manual" / f"{i}_manual1.gif",
                fov=base / "mask" / f"{i}_training_mask.gif",
            )
            for i in ids
        }
        rng = random.Random(seed)
        order = ids[:]
        rng.shuffle(order)
        if fold is None:
            test_ids = order[:n_test]
            val_ids = order[n_test:n_test + n_val]
        else:  # k-fold: fold k is the test set, next n_val images are validation
            k = len(order) // n_folds
            test_ids = order[fold * k:(fold + 1) * k]
            rest = [i for i in order if i not in test_ids]
            val_ids = rest[:n_val]
        train_ids = [i for i in order if i not in test_ids and i not in val_ids]
        self.splits = {"train": sorted(train_ids), "val": sorted(val_ids), "test": sorted(test_ids)}
        self.records = [records[i] for i in self.splits[split]]

        # Persist the split for reproducibility / the thesis appendix.
        split_file = Path(root) / "splits" / f"drive_split_seed{seed}_fold{fold}.json"
        try:
            split_file.parent.mkdir(parents=True, exist_ok=True)
            if not split_file.exists():
                split_file.write_text(json.dumps(self.splits, indent=2))
        except OSError:
            pass


class RoadsSource:
    """File lists for the Massachusetts Roads dataset (official train/val/test)."""

    def __init__(self, root: str, split: str):
        base = Path(root) / "roads" / "tiff"
        img_dir, lbl_dir = base / split, base / f"{split}_labels"
        imgs = sorted(img_dir.glob("*.tif*"))
        if not imgs:
            raise FileNotFoundError(f"No road images found under {img_dir}")
        lbl_by_stem = {p.stem: p for p in lbl_dir.glob("*.tif*")}
        self.records = [
            SampleRecord(name=f"roads_{p.stem}", image=p, label=lbl_by_stem[p.stem])
            for p in imgs if p.stem in lbl_by_stem
        ]


def roads_fov(image: np.ndarray) -> np.ndarray:
    """Valid-pixel mask for aerial tiles: pure-white pixels are 'no data'."""
    return (~np.all(image >= 250, axis=-1)).astype(np.uint8)


# --------------------------------------------------------------------------- #
#                                   Dataset                                   #
# --------------------------------------------------------------------------- #
@dataclass
class DataConfig:
    data_root: str = DEFAULT_DATA_ROOT
    crop_size: int = 512                 # training patch size (multiple of 32 for Hiera)
    samples_per_epoch: Optional[int] = None  # None -> one crop per image per epoch
    max_dist: float = 20.0               # EDT clipping distance (pixels)
    min_fg_frac: float = 0.01            # reject training crops with less foreground
    max_nodata_frac: float = 0.25        # reject training crops with more no-data
    crop_retries: int = 10
    color_jitter: float = 0.2
    cache_images: bool = True            # keep decoded images in RAM (per worker)
    seed: int = 0
    drive_fold: Optional[int] = None


class BinarySegDataset(Dataset):
    """Binary segmentation dataset yielding (image, mask, signed EDT, FOV).

    ``mode='train'`` -> random crops + augmentation, ``mode='eval'`` -> full images
    (inference is done with a sliding window in ``engine.py``).
    """

    def __init__(self, records: Sequence[SampleRecord], mode: str, cfg: DataConfig):
        assert mode in ("train", "eval")
        self.records = list(records)
        self.mode = mode
        self.cfg = cfg
        self._cache: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    # -- length: decouple epoch length from #images (DRIVE has only 14 train images)
    def __len__(self) -> int:
        if self.mode == "train" and self.cfg.samples_per_epoch:
            return self.cfg.samples_per_epoch
        return len(self.records)

    def _load(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if idx in self._cache:
            return self._cache[idx]
        rec = self.records[idx]
        img = _read_image(rec.image)
        lbl = _read_binary(rec.label)
        fov = _read_binary(rec.fov) if rec.fov is not None else roads_fov(img)
        lbl = lbl * fov  # labels outside the FOV are meaningless
        out = (img, lbl, fov)
        if self.cfg.cache_images:
            self._cache[idx] = out
        return out

    # -- training-time augmentation -------------------------------------------
    def _random_crop(self, img, lbl, fov):
        cs = self.cfg.crop_size
        img, lbl, fov = _pad_to_min(img, lbl, fov, cs)
        H, W = lbl.shape
        best = None
        for _ in range(self.cfg.crop_retries):
            y = random.randint(0, H - cs)
            x = random.randint(0, W - cs)
            l = lbl[y:y + cs, x:x + cs]
            f = fov[y:y + cs, x:x + cs]
            best = (y, x)
            if l.mean() >= self.cfg.min_fg_frac and (1 - f.mean()) <= self.cfg.max_nodata_frac:
                break
        y, x = best
        return img[y:y + cs, x:x + cs], lbl[y:y + cs, x:x + cs], fov[y:y + cs, x:x + cs]

    def _augment(self, img, lbl, fov):
        # Geometric: flips + 90-degree rotations (exact, no interpolation of the labels).
        if random.random() < 0.5:
            img, lbl, fov = img[:, ::-1], lbl[:, ::-1], fov[:, ::-1]
        if random.random() < 0.5:
            img, lbl, fov = img[::-1], lbl[::-1], fov[::-1]
        k = random.randint(0, 3)
        img, lbl, fov = np.rot90(img, k), np.rot90(lbl, k), np.rot90(fov, k)
        # Photometric: brightness / contrast / saturation jitter.
        img = img.astype(np.float32) / 255.0
        cj = self.cfg.color_jitter
        if cj > 0:
            b = 1 + random.uniform(-cj, cj)
            c = 1 + random.uniform(-cj, cj)
            s = 1 + random.uniform(-cj, cj)
            gray = img.mean(axis=-1, keepdims=True)
            img = gray + (img - gray) * s
            img = (img - img.mean()) * c + img.mean()
            img = np.clip(img * b, 0, 1)
        return img, lbl, fov

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        idx = random.randrange(len(self.records)) if (
            self.mode == "train" and self.cfg.samples_per_epoch) else i
        img, lbl, fov = self._load(idx)
        name = self.records[idx].name

        if self.mode == "train":
            img, lbl, fov = self._random_crop(img, lbl, fov)
            img, lbl, fov = self._augment(img, lbl, fov)
        else:
            img = img.astype(np.float32) / 255.0

        # EDT is computed AFTER augmentation/cropping on the CPU.
        dist = signed_distance_map(lbl, self.cfg.max_dist)

        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        return {
            "image": torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).float(),
            "mask": torch.from_numpy(np.ascontiguousarray(lbl))[None].float(),
            "dist": torch.from_numpy(np.ascontiguousarray(dist))[None].float(),
            "fov": torch.from_numpy(np.ascontiguousarray(fov))[None].float(),
            "name": name,
        }


def _pad_to_min(img, lbl, fov, size):
    """Zero-pad (outside FOV) so both spatial dims are >= size."""
    H, W = lbl.shape
    ph, pw = max(0, size - H), max(0, size - W)
    if ph == 0 and pw == 0:
        return img, lbl, fov
    pad2 = ((0, ph), (0, pw))
    return (np.pad(img, pad2 + ((0, 0),)), np.pad(lbl, pad2), np.pad(fov, pad2))


# --------------------------------------------------------------------------- #
#                                  Factories                                  #
# --------------------------------------------------------------------------- #
DATASETS = ("drive", "roads")


def build_dataset(name: str, split: str, cfg: DataConfig) -> BinarySegDataset:
    """``name`` in {'drive', 'roads'}; ``split`` in {'train', 'val', 'test'}."""
    name = name.lower()
    if name == "drive":
        src = DriveSource(cfg.data_root, split, seed=cfg.seed, fold=cfg.drive_fold)
    elif name == "roads":
        src = RoadsSource(cfg.data_root, split)
    else:
        raise ValueError(f"Unknown dataset '{name}'. Choose from {DATASETS}.")
    mode = "train" if split == "train" else "eval"
    return BinarySegDataset(src.records, mode, cfg)


def _worker_init(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    random.seed(seed + worker_id)
    np.random.seed((seed + worker_id) % 2**32)


def build_loader(ds: BinarySegDataset, batch_size: int, num_workers: int = 8,
                 shuffle: Optional[bool] = None) -> DataLoader:
    train = ds.mode == "train"
    return DataLoader(
        ds,
        batch_size=batch_size if train else 1,   # eval: full images of varying size
        shuffle=train if shuffle is None else shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=train,
        persistent_workers=num_workers > 0,
        worker_init_fn=_worker_init,
    )
