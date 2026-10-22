"""
Step 1 -- Datasets & dataloaders for thin-structure binary segmentation.

Supported datasets (``build_dataset`` names; provenance in data/raw/SOURCES.md)
------------------
Retinal vessels
* ``drive``      official 20 train / 20 test (test labels in DRIVE/test/1st_manual)
* ``drive_dev``  the 20 labelled training images split 14/2/4 (or k-fold via ``fold``);
                 used by the first experiments, kept for reproducibility
* ``chase``      CHASE_DB1, 20 train (subjects 01-10) / 8 test (11-14), 1st observer
* ``stare``      STARE, 10 train / 10 test, Hoover labels
* ``hrf``        HRF, images 01-05 of each category train (15) / 06-15 test (30)
Roads
* ``roads``      Massachusetts Roads, official train/val/test (1500x1500 tiles; white
                 "no-data" regions are treated as outside the FOV)
* ``deepglobe``  DeepGlobe Roads, 4696 train / 1530 test split of Batra et al. 2019
Cracks
* ``crack500``   CRACK500 crops, official train/val/test lists
* ``deepcrack``  DeepCrack, 300 train / 237 test
* ``cracktree200``, ``cfd``, ``gaps384``   test-only (FPHBN cross-dataset protocol)

Validation: datasets without an official validation split hold out a few training
images (``n_val``).  ``DataConfig.trainval`` merges them back into the training
set for final runs (model selection is then the last epoch).

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

import gc
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor
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


def tubed_skeleton(mask: np.ndarray) -> np.ndarray:
    """1-px skeleton of the mask dilated by one pixel (3x3), as in Skeleton Recall (ECCV 2024):
    the tube makes the target robust to the exact centreline position."""
    from scipy.ndimage import binary_dilation
    from skimage.morphology import skeletonize

    if not mask.any():
        return np.zeros(mask.shape, np.uint8)
    skel = skeletonize(np.ascontiguousarray(mask, dtype=bool))
    return (binary_dilation(skel, structure=np.ones((3, 3), bool)) & mask.astype(bool)).astype(np.uint8)


# --------------------------------------------------------------------------- #
#                         Dataset "sources" (file lists)                      #
# --------------------------------------------------------------------------- #
@dataclass
class SampleRecord:
    name: str
    image: Path
    label: Path
    fov: Optional[Path] = None   # None -> computed from the image with ``fov_mode``
    fov_mode: str = "roads"      # 'roads' (non-white), 'fundus' (non-black disc), 'all'


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


def _hold_out(records: List[SampleRecord], n_val: int, seed: int) -> Tuple[list, list]:
    """Deterministically move ``n_val`` records of a training list into a validation list."""
    order = list(range(len(records)))
    random.Random(seed).shuffle(order)
    val = set(order[:n_val])
    return ([r for i, r in enumerate(records) if i not in val],
            [r for i, r in enumerate(records) if i in val])


class OfficialDriveSource:
    """DRIVE with the official protocol: 20 training images, 20 labelled test images."""

    def __init__(self, root: str, split: str, n_val: int = 2, seed: int = 0):
        base = Path(root) / "DRIVE"
        train = [SampleRecord(f"drive_{i}", base / "training" / "images" / f"{i}_training.tif",
                              base / "training" / "1st_manual" / f"{i}_manual1.gif",
                              base / "training" / "mask" / f"{i}_training_mask.gif")
                 for i in range(21, 41)]
        test = [SampleRecord(f"drive_{i:02d}", base / "test" / "images" / f"{i:02d}_test.tif",
                             base / "test" / "1st_manual" / f"{i:02d}_manual1.png",
                             base / "test" / "mask" / f"{i:02d}_test_mask.gif")
                for i in range(1, 21)]
        tr, va = _hold_out(train, n_val, seed)
        self.splits = {"train": tr, "val": va, "trainval": train, "test": test}


class ChaseSource:
    """CHASE_DB1: subjects 01-10 train (20 images), 11-14 test (8); both eyes stay together."""

    def __init__(self, root: str, split: str, n_val: int = 2, seed: int = 0):
        base = Path(root) / "raw" / "chase"
        recs = sorted(
            (SampleRecord(f"chase_{p.stem}", p, p.parent.parent / "label" / f"{p.stem}_1stHO.png",
                          fov_mode="fundus")
             for p in base.glob("*/input/Image_*.png")), key=lambda r: r.name)
        if len(recs) != 28:
            raise FileNotFoundError(f"Expected 28 CHASE_DB1 images under {base}, found {len(recs)}")
        subj = lambda r: int(r.name.split("_")[2][:2])
        train = [r for r in recs if subj(r) <= 10]
        # validation = one whole subject (both eyes), never split across sets
        va = [r for r in train if subj(r) == 10] if n_val else []
        tr = [r for r in train if r not in va]
        self.splits = {"train": tr, "val": va, "trainval": train,
                       "test": [r for r in recs if subj(r) > 10]}


class StareSource:
    """STARE: the 10/10 split used by the HF mirror (im0001-im0082 train, im0139-im0324 test)."""

    def __init__(self, root: str, split: str, n_val: int = 1, seed: int = 0):
        base = Path(root) / "raw" / "stare"

        def recs(d):
            return [SampleRecord(f"stare_{p.stem}", p, base / d / "label" / f"{p.stem}.ah.ppm",
                                 fov_mode="fundus")
                    for p in sorted((base / d / "input").glob("*.ppm"))]

        train = recs("train")
        tr, va = _hold_out(train, n_val, seed)
        self.splits = {"train": tr, "val": va, "trainval": train, "test": recs("val")}


class HrfSource:
    """HRF (3504x2336): images 01-05 of each category (dr/g/h) train, 06-15 test."""

    def __init__(self, root: str, split: str, n_val: int = 3, seed: int = 0):
        base = Path(root) / "raw" / "hrf"
        recs = [SampleRecord(f"hrf_{p.stem}", p, base / "manual1" / f"{p.stem}.tif",
                             base / "mask" / f"{p.stem}_mask.tif")
                for p in sorted((base / "images").iterdir())]
        num = lambda r: int(r.name.split("_")[1])
        train = [r for r in recs if num(r) <= 5]
        va = [r for r in train if num(r) == 5] if n_val else []   # one image per category
        self.splits = {"train": [r for r in train if r not in va], "val": va, "trainval": train,
                       "test": [r for r in recs if num(r) > 5]}


class DeepGlobeSource:
    """DeepGlobe Roads with the 4696/1530 split of Batra et al. (CVPR 2019)."""

    def __init__(self, root: str, split: str, n_val: int = 100, seed: int = 0):
        base = Path(root) / "raw" / "deepglobe"
        by_id = {p.name.split("_")[0]: p for p in base.glob("*/images/*_sat.jpg")}

        def recs(lst):
            ids = [l.strip() for l in (base / "batra_split" / lst).read_text().split() if l.strip()]
            return [SampleRecord(f"deepglobe_{i}", by_id[i],
                                 by_id[i].parent.parent / "masks" / f"{i}_mask.png", fov_mode="all")
                    for i in ids]

        train = recs("train.txt")
        tr, va = _hold_out(train, n_val, seed)
        self.splits = {"train": tr, "val": va, "trainval": train, "test": recs("val.txt")}


class CrackListSource:
    """Crack datasets of the FPHBN bundle, defined by '<image> <label>' list files."""

    LISTS = {  # name -> (folder, {split: list file}, label dir, label suffix)
        "crack500": ("CRACK500", {"train": "train.txt", "val": "val.txt", "test": "test.txt"}, None, ".png"),
        "cracktree200": ("cracktree200", {"test": "test.txt"}, "cracktree200_gt", ".png"),
        "cfd": ("CFD", {"test": "test.txt"}, "seg_gt", ".png"),
        "gaps384": ("GAPS384", {"test": "test.txt"}, "croppedgt", ".png"),
    }

    def __init__(self, root: str, name: str, split: str):
        folder, lists, lbl_dir, suffix = self.LISTS[name]
        base = Path(root) / "raw" / "pavement" / folder
        self.splits = {}
        for s, lst in lists.items():
            recs = []
            for line in (base / lst).read_text().splitlines():
                if not line.strip():
                    continue
                img = base / line.split()[0]
                lbl = (img.parent if lbl_dir is None else base / lbl_dir) / f"{img.stem}{suffix}"
                if lbl.exists():   # CFD's list contains a few images without a label
                    recs.append(SampleRecord(f"{name}_{img.stem}", img, lbl, fov_mode="all"))
            self.splits[s] = recs
        if "train" in self.splits:
            self.splits["trainval"] = self.splits["train"] + self.splits["val"]


class DeepCrackSource:
    """DeepCrack (Liu et al. 2019): 300 train / 237 test."""

    def __init__(self, root: str, split: str, n_val: int = 20, seed: int = 0):
        base = Path(root) / "raw" / "deepcrack"

        def recs(s):
            return [SampleRecord(f"deepcrack_{p.stem}", p, base / f"{s}_lab" / f"{p.stem}.png",
                                 fov_mode="all")
                    for p in sorted((base / f"{s}_img").glob("*.jpg"))]

        train = recs("train")
        tr, va = _hold_out(train, n_val, seed)
        self.splits = {"train": tr, "val": va, "trainval": train, "test": recs("test")}


def roads_fov(image: np.ndarray) -> np.ndarray:
    """Valid-pixel mask for aerial tiles: pure-white pixels are 'no data'."""
    return (~np.all(image >= 250, axis=-1)).astype(np.uint8)


def fundus_fov(image: np.ndarray, erode: int = 3) -> np.ndarray:
    """Circular field of view of a fundus photograph without an official mask (CHASE, STARE).

    The retina is strongly red while the background is dark but noisy (STARE's reaches
    ~45 grey levels, so a fixed threshold leaks into it): threshold the smoothed red
    channel at half its Otsu level, keep the largest component, fill holes and erode a
    few pixels off the rim.
    """
    from scipy.ndimage import binary_erosion as erosion, binary_fill_holes, gaussian_filter
    from scipy.ndimage import label as cc_label
    from skimage.filters import threshold_otsu

    red = gaussian_filter(image[..., 0].astype(np.float32), 3)
    m = red > 0.5 * threshold_otsu(red)
    lab, n = cc_label(m)
    if n > 1:
        m = lab == (np.bincount(lab.ravel())[1:].argmax() + 1)
    m = binary_fill_holes(m)
    if erode:
        m = erosion(m, iterations=erode)
    return m.astype(np.uint8)


FOV_FUNCS = {"roads": roads_fov, "fundus": fundus_fov,
             "all": lambda img: np.ones(img.shape[:2], np.uint8)}


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
    cache_images: bool = True            # keep decoded images in RAM (train: preloaded, shared by workers)
    seed: int = 0
    drive_fold: Optional[int] = None
    trainval: bool = False               # train on train+val (final runs; select last epoch)
    scale: float = 1.0                   # resize *training* images (e.g. 2 for small fundus images);
                                         # evaluation runs at the same scale via evaluate(test_scale=)


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

    def _read(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        rec = self.records[idx]
        img = _read_image(rec.image)
        lbl = _read_binary(rec.label)
        fov = _read_binary(rec.fov) if rec.fov is not None else FOV_FUNCS[rec.fov_mode](img)
        lbl = lbl * fov  # labels outside the FOV are meaningless
        if self.mode == "train" and self.cfg.scale != 1.0:
            img, lbl, fov = _rescale(img, lbl, fov, self.cfg.scale)
        return img, lbl, fov

    def _load(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if idx in self._cache:
            return self._cache[idx]
        out = self._read(idx)
        if self.cfg.cache_images:
            self._cache[idx] = out
        return out

    def preload(self, num_threads: int = 8) -> None:
        """Decode every image into the cache up front.

        Must run in the main process *before* the DataLoader workers fork: they then
        share one copy-on-write copy of the arrays instead of each filling its own
        cache (Roads: 1108 x ~11 MB per worker -> host OOM after a few epochs).
        """
        missing = [i for i in range(len(self.records)) if i not in self._cache]
        with ThreadPoolExecutor(num_threads) as ex:
            for i, out in zip(missing, ex.map(self._read, missing)):
                self._cache[i] = out

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
        out = {
            "image": torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).float(),
            "mask": torch.from_numpy(np.ascontiguousarray(lbl))[None].float(),
            "dist": torch.from_numpy(np.ascontiguousarray(dist))[None].float(),
            "fov": torch.from_numpy(np.ascontiguousarray(fov))[None].float(),
            "name": name,
        }
        if self.mode == "train":  # centreline target for the topology losses
            out["skel"] = torch.from_numpy(tubed_skeleton(lbl))[None].float()
        return out


def _rescale(img, lbl, fov, scale: float):
    """Bilinear image; label resampled bilinearly and re-thresholded at 0.5 (keeps 1-px vessels
    connected when upsampling); FOV nearest."""
    H, W = lbl.shape
    size = (round(W * scale), round(H * scale))
    img = np.array(Image.fromarray(img).resize(size, Image.BILINEAR))
    lbl = (np.array(Image.fromarray(lbl.astype(np.float32)).resize(size, Image.BILINEAR)) >= 0.5)
    fov = np.array(Image.fromarray(fov).resize(size, Image.NEAREST))
    return img, (lbl & (fov > 0)).astype(np.uint8), fov


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
DATASETS = ("drive", "drive_dev", "chase", "stare", "hrf", "roads", "deepglobe",
            "crack500", "deepcrack", "cracktree200", "cfd", "gaps384")
DEFAULT_SAMPLES_PER_EPOCH = {"drive": 500, "drive_dev": 500, "chase": 500, "stare": 500, "hrf": 500,
                             "roads": 2000, "deepglobe": 2000, "crack500": 2000, "deepcrack": 1000}


def build_records(name: str, split: str, cfg: DataConfig) -> List[SampleRecord]:
    """File list of one split; ``split`` in {'train', 'val', 'test'}."""
    name = name.lower()
    if name == "drive_dev":
        return DriveSource(cfg.data_root, split, seed=cfg.seed, fold=cfg.drive_fold).records
    if name == "roads":
        return RoadsSource(cfg.data_root, split).records
    sources = {"drive": OfficialDriveSource, "chase": ChaseSource, "stare": StareSource,
               "hrf": HrfSource, "deepglobe": DeepGlobeSource, "deepcrack": DeepCrackSource}
    if name in sources:
        splits = sources[name](cfg.data_root, split, seed=cfg.seed).splits
    elif name in CrackListSource.LISTS:
        splits = CrackListSource(cfg.data_root, name, split).splits
    else:
        raise ValueError(f"Unknown dataset '{name}'. Choose from {DATASETS}.")
    if split == "train" and cfg.trainval:
        split = "trainval"
    if split not in splits:
        raise ValueError(f"Dataset '{name}' has no '{split}' split (test-only: {list(splits)})")
    return splits[split]


def build_dataset(name: str, split: str, cfg: DataConfig) -> BinarySegDataset:
    """``name`` in DATASETS; ``split`` in {'train', 'val', 'test'}."""
    mode = "train" if split == "train" else "eval"
    return BinarySegDataset(build_records(name, split, cfg), mode, cfg)


def _worker_init(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    random.seed(seed + worker_id)
    np.random.seed((seed + worker_id) % 2**32)


def build_loader(ds: BinarySegDataset, batch_size: int, num_workers: int = 8,
                 shuffle: Optional[bool] = None) -> DataLoader:
    train = ds.mode == "train"
    # Random crops make every train worker touch every image, so share one preloaded
    # cache. gc.freeze() keeps the workers' GC from writing to (and thereby copying)
    # the pages holding the cached objects.
    if train and num_workers > 0 and ds.cfg.cache_images:
        ds.preload(num_workers)
        gc.freeze()
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
