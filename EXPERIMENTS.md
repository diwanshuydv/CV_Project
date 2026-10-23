# Experiment log — SAM2-UNet for thin-structure segmentation

Everything tried so far, in the order it was done, with the numbers and what each result
means. All numbers are on held-out **test** sets and come from `runs/<run>/eval_test_<dataset>.json`
(TTA / memory-off variants: `eval_test_<dataset>_tta.json`, `..._memoff.json`).
Last updated 2026-10-23.

## 1. Summary

| | Finding |
|---|---|
| Best results | **Your SAM2-UNet baseline at 2× input scale** reaches the published level: DRIVE pooled F1 **0.830 ± 0.002** (3 seeds; 0.833 single seed with flip TTA) vs FR-UNet 0.832; CHASE_DB1 **0.823 ± 0.002** vs 0.815; STARE **0.851 ± 0.002** vs 0.833; Massachusetts Roads IoU 0.659 vs ≈ 0.66. |
| What made the difference | **Input resolution.** Upsampling the image 2× before the network (+0.8 F1 on DRIVE) — SAM2's finest features are at 1/4 resolution, so 1–2 px vessels are lost at 1×. |
| What did not help | Distance-transform boundary loss; tile-sequence memory (SAM2 video memory across tiles); skeleton-recall loss (trades F1 for connectivity); fixed high-frequency (Sobel / Haar-DWT) edge prior in the decoder. None gives a consistent, significant gain across seeds and datasets. |
| Error analysis | Remaining errors are mostly **whole missed segments**, not imprecise edges: 45% of all error pixels on Roads (57% of missed pixels) and 17% on DRIVE lie > 2 px from any correct structure. |
| Status for a paper | Competitive results, **no novel method that beats the baseline yet**. Defensible content: resolution-is-the-bottleneck finding, negative results for context memory / boundary losses / edge priors, error decomposition. |

## 2. Setup

**Model (baseline).** Frozen SAM 2.1 Hiera-L encoder + LoRA (r = 8, 192 layers) + residual U-Net
decoder with a small learned CNN stem on the raw image at strides 1 and 2; deep supervision.
219.96 M parameters, 7.81 M trainable. Loss BCE + Dice unless stated. AdamW (LoRA 2e-4,
decoder 1e-3), cosine schedule, 100 epochs, 512 px random crops, batch 4, bf16.
Inference: 512 px sliding window, 25 % overlap, Gaussian blending.

**Protocol** (changed during the project, see E1).
- Official splits: DRIVE 20 train / 20 test; CHASE_DB1 20 / 8 (subjects 01–10 / 11–14);
  STARE 10 / 10; HRF 15 / 30 (images 01–05 of each category for training); Massachusetts
  Roads official train / val / test.
- Validation: a few training images are held out (DRIVE 2, CHASE 2 = subject 10, STARE 1,
  HRF 3); the best-validation checkpoint is tested.
- Threshold: the F1-optimal threshold on the validation set, not a fixed 0.5.
- Seeds: 42 by default; seeds 1 and 2 also change which training images are held out.

**Metrics.** Pooled F1 / Se / Sp / Acc / AUC over all test pixels inside the field of view (FOV),
which is the vessel-segmentation convention. Per-image Dice, IoU and clDice (centreline Dice).
Betti-0 error per 64 px patch (fragmentation), HD95, relaxed F1 (ρ = 3 px). Error decomposition
splits error pixels into *fringe* (≤ 2 px from the other mask) and *segment* (farther: whole
branch missed or spurious fragment). Significance is a Wilcoxon signed-rank test on per-image
values, averaged over seeds.

**Compute.** A30 24 GB nodes. Long runs are split into 1 h jobs on the `test` partition,
which has no per-user GPU cap (`scripts/chain.sh`).

**Datasets** are in `data/`; provenance is in `data/raw/SOURCES.md`. Also downloaded but not yet used:
DeepGlobe Roads (Batra 4696 / 1530 split), CRACK500, DeepCrack, CrackTree200, CFD and GAPs384.

## 3. Experiments

### E0 — Boundary-aware loss (original repo, before this log's protocol)
`L = BCE + Dice + λ(t)·L_boundary`, a distance-transform boundary loss with λ ramped to 0.5.
This used the old DRIVE split (4 test images taken from the training set) and seed 42.

| Run | Test | IoU | Dice | HD95 |
|---|---|---|---|---|
| drive_base (λ = 0) | DRIVE (4 img) | 0.679 | 0.808 | 9.44 |
| drive_bnd (λ = 0.5) | DRIVE (4 img) | 0.674 | 0.805 | 8.38 |
| roads_base | Roads | 0.659 | 0.792 | 98.7 |
| roads_bnd | Roads | 0.655 | 0.789 | 87.7 |

Cross-domain IoU: DRIVE→Roads 0.065 (baseline) and 0.070 (boundary loss); Roads→DRIVE 0.000 for both.

**Result:** no effect on overlap. The HD95 change is within noise; the visual overlays
(`runs/figures/`) show it comes from removing a few stray fragments.

**Roads→DRIVE = 0 is not a bug** (`scripts/diag_cross.py`): a Roads model predicts p ≈ 0 on
DRIVE even after the image is inverted and upscaled ×2 / ×3, while scoring Dice 0.83 on Roads
through the same code.

### E1 — Protocol fix and new metrics
- Found the official DRIVE test labels (HF mirror; the 20 test images are pixel-identical to
  the official ones) and switched to official splits for every dataset.
- Added clDice, Betti errors, pooled metrics, AUC, ODS/OIS, relaxed F1, error decomposition
  and validation-selected thresholds.
- **Bug found and fixed:** the AUC histogram used uniform probability bins, which under-estimates
  the AUC of saturated (Dice-trained) predictions. It now uses logit-uniform bins, checked against
  scikit-learn.
- **Bug found and fixed:** the STARE FOV mask leaked into the noisy dark background (FOV 93 %
  vs 77 %). It now uses an Otsu threshold on the smoothed red channel (FOV 72–73 %).

Official-protocol baselines (1×, seed 42):

| Dataset | pooled F1 | Dice | clDice | Betti-0 / patch | segment-error share | HD95 |
|---|---|---|---|---|---|---|
| DRIVE (`drive_off_base`) | 0.822 | 0.822 | 0.816 | 1.14 | 17 % | 6.2 |
| Roads (`roads_base`) | 0.797 | 0.792 | 0.887 | 0.24 | **45 %** | 98.7 |

**Note on AUC / Acc vs the literature:** we compute them inside the FOV. Published DRIVE
Acc ≈ 0.97 together with F1 ≈ 0.83 is only possible if the black background outside the FOV
is included. F1 is unaffected by this convention, so it is the comparable number.

### E2 — Tile-sequence memory (TSM): the planned novel method
**Idea:** reuse SAM2's pretrained *video* memory as *spatial* memory. A training sample is a
region of 2 × 2 tiles. Pass 1 decodes a downsampled global view and every tile without memory.
SAM2's MemoryEncoder encodes each frame (features plus its pass-1 mask). In pass 2 each tile
cross-attends to all memories through SAM2's MemoryAttention with LoRA. SAM2's temporal
encodings are replaced by zero-initialised encodings of absolute region coordinates, so the
model starts out identical to the baseline. Code: `sam2unet/memory.py`, `--model tsm`.
+1.74 M trainable parameters. Inference processes about 5× more frames per pixel.

**Prior art check:** no published work reuses SAM2 memory across the spatial tiles of a single
image. Close work on topology and LoRA-SAM: TopoLoRA-SAM (arXiv 2601.02273), SA-SAM2 (Springer
PAA 2026), SAM2MS (Remote Sensing 2025), TAPDecoderFT (arXiv 2609.26189).

| Comparison | pooled F1 base → TSM | other | significance (per image) |
|---|---|---|---|
| DRIVE 1× (seed 42) | 0.822 → 0.821 | HD95 6.2 → 9.2 | — |
| Roads (seed 42) | 0.797 → 0.795 | HD95 98.7 → 83.4 | Dice p = 0.10, HD95 p = 0.39 |
| DRIVE 2× (3 seeds) | 0.830 → 0.830 | Betti-0 0.97 → 0.96 | Dice p = 0.76, clDice p = 0.70, Betti-0 p = 0.65 |
| CHASE 2× (1 TSM seed vs 3 base) | 0.823 → 0.822 | | n.s. |
| STARE 2× (1 TSM seed vs 3 base) | 0.851 → 0.851 | vs seed 42: Dice p = 0.04, clDice p = 0.01, **worse** | |
| HRF 1× (seed 42) | 0.822 → 0.817 | clDice 0.830 → 0.818 | Dice p = 0.001, **worse** |

**Memory-off analysis** (the same trained TSM weights, predicting with the pass-1 outputs):
F1 is unchanged (DRIVE 2× 0.832 / 0.832, Roads 0.795 / 0.795), while Betti-0 is 0.85 → 1.06 on
DRIVE and 0.24 → 0.27 on Roads. So the memory reduces fragmentation by 13–20 % within one model,
but that effect did not survive across seeds: seed 2 TSM had Betti-0 1.17 against the
baseline's 0.90.

**Design issue found:** at 1× a DRIVE image (584 px) is smaller than one 1024 px region, so
three of the four training tiles are mostly padding. The 2× runs fix this.

**Result:** no reliable gain; worse on HRF, the dataset with the most context. The context
projection does train (weight norm 10.8 vs 23 for the decoder's own stride-16 adapter), so the
memory is used but not useful. Hiera-L's attention within a 512 px tile already provides most
of the context the decoder needs.

### E3 — Skeleton-recall loss (Kirchhoff et al., ECCV 2024)
`--w_skel 1`, tubed skeleton targets computed in the dataloader. Soft clDice is implemented but untested.

| Dataset | pooled F1 base → skel | Betti-0 | relaxed recall / precision |
|---|---|---|---|
| DRIVE 1× | 0.822 → 0.816 | 1.14 → **0.77** | |
| Roads | 0.797 → 0.791 | 0.24 → 0.25 | 0.878 / 0.949 → **0.904** / 0.932 |

**Result:** it shifts errors from missed structure to false positives, with fewer fragments on
DRIVE and fewer wholly missed road segments (57 % → 52 % of missed pixels), but F1 drops
0.6 points. Single seed.

### E4 — 2× input scale (the gain)
`--scale 2`: training images are resized ×2 (labels bilinear then thresholded at 0.5, which keeps
both the vessel fraction and the connectivity exactly). At test time the prediction is made at
×2 and resized back, so metrics are computed at the original resolution.

| Dataset | 1× | 2× (mean ± std over 3 seeds) |
|---|---|---|
| DRIVE pooled F1 | 0.822 | **0.830 ± 0.002** (0.8311, 0.8312, 0.8271) |
| DRIVE clDice | 0.816 | 0.832 ± 0.004 |
| DRIVE AUC (FOV) | 0.976 | 0.981 |
| CHASE pooled F1 | — | **0.823 ± 0.002** (0.8239, 0.8201, 0.8247) |
| STARE pooled F1 | — | **0.851 ± 0.002** (0.8536, 0.8487, 0.8505) |

HRF and Roads stay at 1× because their images are already large: HRF pooled F1 0.822, Roads 0.797.

### E5 — Flip test-time augmentation
Average over the identity, horizontal, vertical and both flips (`evaluate.py --tta`).
DRIVE 1×: 0.822 → 0.827. DRIVE 2× (seed 42): 0.831 → **0.833**. TSM 2× (seed 42): 0.832 → 0.833.
It has only been applied to the seed-42 DRIVE runs so far.

### E6 — Fixed high-frequency prior in the decoder
**Idea:** feed deterministic edge information past the smoothing of the SAM2 encoder. The maps are:
- Sobel gradient magnitude (R, G, B, luminance) concatenated into the stride-1 skip;
- Haar-DWT level-1 detail bands (LH, HL, HH) into the stride-2 skip;
- level-2 detail bands into the stride-4 skip.

Each map is normalised per image with log1p. There are no learned filters (+≈10 k decoder
weights). Canny was not used because its hard thresholds remove faint capillaries.
Code: `HighFrequencyPrior` in `sam2unet/model.py`, `--hf_prior`.

| Dataset | seeds | pooled F1 base → HF | other | significance (per image) |
|---|---|---|---|---|
| DRIVE 2× | 3 | 0.8298 → 0.8302 | Betti-0 0.97 → 0.92; F1 std across seeds 0.0019 → 0.0007 | Betti-0 p = 0.003 better; clDice p = 0.09 |
| CHASE 2× | 3 | 0.8229 → 0.8238 | clDice 0.840 → 0.842; HD95 20.7 → 17.4 | clDice p = 0.02 better; Dice n.s. |
| STARE 2× | 3 | 0.8509 → 0.8480 | Betti-0 0.55 → 0.60 | Dice p = 0.05 **worse**; Betti-0 p = 0.03 worse |
| HRF 1× | 1 | 0.8218 → 0.8196 | | Dice p < 0.001 **worse** |
| Roads | 1 | 0.7974 → 0.7920 | recall 0.793 → 0.777 | Dice / clDice p < 0.001 **worse** |

**Result:** small connectivity gains on DRIVE and CHASE, losses on STARE, HRF and Roads. The
2× upsampling already removes most of the smoothing the prior targets. On aerial images the
edge maps mostly mark non-road edges (roofs, cars, field boundaries).

**Open test (not run yet):** can the prior *replace* the 2× upsampling? DRIVE at 1× with the
prior against 0.830 at 2× would show whether it is an efficiency win (4× fewer pixels).

## 4. Comparison with published results

| Dataset | Ours (protocol) | Published | Caveat |
|---|---|---|---|
| DRIVE | 0.830 ± 0.002 pooled F1, 3 seeds (0.833 with TTA, 1 seed) | FR-UNet 0.8316 | same split; FR-UNet trains on all 20 images |
| CHASE_DB1 | 0.823 ± 0.002 | FR-UNet 0.8151 | 20 / 8 split, the same as FR-UNet |
| STARE | 0.851 ± 0.002 | FR-UNet 0.833 (as re-run by LFRA-Net) | STARE splits differ across papers |
| HRF | 0.822 (1 seed) | ≈ 0.80–0.82 | protocols vary (full vs half resolution) |
| Massachusetts Roads | IoU 0.659, F1 0.797 (1 seed) | ≈ 0.66 IoU | splits / crops vary |

These published numbers are taken from the papers. Before they can be claimed as state of the art,
FR-UNet (and ideally TopoLoRA-SAM) should be re-run under our exact splits and thresholds.

## 5. Lessons

1. For SAM2-based thin-structure segmentation, **resolution matters more than context or
   boundary supervision**: one change to the input (×2) beat every architectural or loss change.
2. Several apparent wins came from a single seed and disappeared with more: the TSM
   connectivity gain on DRIVE, and TSM on CHASE / STARE when compared with one baseline seed.
   Every claim needs ≥ 3 seeds and a per-image test.
3. Most remaining errors are whole missed segments (Roads 57 % of missed pixels). Methods that
   refine edges (boundary losses, edge priors) cannot fix them.
4. Topology losses (skeleton recall) and memory move errors around (fewer fragments ↔ more
   false positives) rather than removing them.

## 6. Possible next steps

- **Consolidate the results:** 3 seeds for HRF and Roads, flip TTA for every run, `--trainval`
  (all training images, last epoch) for the final numbers, and FR-UNet re-run under our splits.
- **HF prior at 1×** (DRIVE, 3 seeds, ≈ 1 h): does it replace upsampling?
- **Method aimed at the resolution bottleneck**, e.g. a learned high-resolution path into the
  Hiera encoder instead of plain upsampling. This is untested.
- Unused datasets: DeepGlobe and the crack sets (CRACK500 → CrackTree200 / CFD / GAPs384
  cross-dataset protocol).

## 7. Reproducing

```bash
# baseline at 2x (the best configuration), one 1 h chunk chain on the test partition
bash scripts/chain.sh 3 --exp_name drive_base_x2 --train_dataset drive --lambda_max 0 --scale 2
# tile-sequence memory / high-frequency prior / skeleton recall variants
...  --model tsm --window 512 --batch_size 2 --grad_ckpt --samples_per_epoch 250 --epochs 60
...  --hf_prior
...  --w_skel 1
# re-score runs (validation threshold, optional TTA / memory-off), then tabulate
python evaluate.py --runs runs/drive_base_x2 runs/drive_hf_x2 --tta --tag _tta --out runs/eval.md
python scripts/collect_results.py --runs runs/drive_* --tag _tta
```

| File | Role |
|---|---|
| `sam2unet/datasets.py` | 12 datasets, official splits, FOV masks, skeleton targets, `--scale` |
| `sam2unet/model.py` | SAM2-UNet, `HighFrequencyPrior` |
| `sam2unet/memory.py` | tile-sequence memory (`SAM2MemUNet`) |
| `sam2unet/losses.py` | BCE + Dice, boundary loss, skeleton recall, soft clDice |
| `sam2unet/metrics.py` | all metrics above, `PixelStats` |
| `scripts/chain.sh`, `slurm/train_test.sbatch` | 1 h chained jobs on the `test` partition |
| `scripts/queue.sh` | serial job feeder for `small` / `btech` (2-job QoS) |
| `scripts/check_datasets.py`, `scripts/diag_cross.py`, `scripts/collect_results.py` | data checks, cross-domain diagnosis, result tables |
| `runs/eval_*.md` | evaluation tables (baselines, round 1, TTA, memory-off) |
