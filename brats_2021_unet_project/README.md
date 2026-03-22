# BraTS 2021 U-Net Project

This folder contains a complete PyTorch training pipeline for the **RSNA-ASNR-MICCAI BraTS 2021 brain tumor segmentation task** using a **2D U-Net**.

## Features
- Reads BraTS 2021 subject folders with `.nii.gz` files.
- Uses the four BraTS modalities: `t1`, `t1ce`, `t2`, `flair`.
- Automatically remaps BraTS labels from `{0, 1, 2, 4}` to contiguous class ids `{0, 1, 2, 3}`.
- Splits subjects into train/test with an approximate **70/30 ratio**.
- Trains a memory-friendly **2D U-Net** that works well on a single GPU such as an RTX 4090.
- Computes Dice score **without the background class**.
- Saves:
  - subject split files,
  - checkpoints,
  - training loss curve,
  - sample prediction visualizations,
  - final metrics JSON.

## Expected dataset structure
After downloading and extracting the BraTS 2021 data from Kaggle, the dataset root should look like this:

```text
BraTS2021/
├── BraTS2021_00000/
│   ├── BraTS2021_00000_flair.nii.gz
│   ├── BraTS2021_00000_t1.nii.gz
│   ├── BraTS2021_00000_t1ce.nii.gz
│   ├── BraTS2021_00000_t2.nii.gz
│   └── BraTS2021_00000_seg.nii.gz
├── BraTS2021_00001/
│   └── ...
└── ...
```

## Recommended environment
```bash
conda create -n brats-unet python=3.10 -y
conda activate brats-unet
pip install -r brats_2021_unet_project/requirements.txt
```

## Quick start
Run one epoch first to verify the pipeline:

```bash
python brats_2021_unet_project/train.py \
  --data-root /path/to/BraTS2021 \
  --output-dir /path/to/outputs/brats_unet_run \
  --epochs 1 \
  --batch-size 8 \
  --num-workers 8 \
  --device cuda
```

## Main outputs
After training finishes, the output directory contains:

```text
outputs/brats_unet_run/
├── checkpoints/
│   ├── best_model.pt
│   └── last_model.pt
├── figures/
│   ├── training_loss.png
│   └── sample_predictions.png
├── splits/
│   ├── train_subjects.txt
│   └── test_subjects.txt
├── history.csv
├── metrics.json
└── train.log
```

## Notes
- Default training uses **all axial slices** from each subject.
- Empty slices can be skipped with `--skip-empty-slices`.
- Mixed precision is enabled automatically on CUDA unless `--disable-amp` is passed.
- The script defaults to `--epochs 1`, matching the assignment suggestion to prioritize a successful end-to-end run.
