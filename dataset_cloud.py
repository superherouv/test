"""
Multi-temporal cloud removal dataset loader.

Supported dataset layouts
--------------------------

Layout A – Sen2 MTC New  (recommended)
    root/
        train/
            s2_t1/   <stem>.png   ← temporal image 1 (cloudy)
            s2_t2/   <stem>.png   ← temporal image 2 (cloudy)
            s2_t3/   <stem>.png   ← temporal image 3 (cloudy)
            s2_target/ <stem>.png ← cloud-free reference
            masks/   <stem>.png   ← cloud masks  (optional)
                                     pixel=0 → cloud, pixel=255 → clear
        val/
            (same structure)

Layout B – Flat folders  (simpler, any L)
    root/
        cloudy_t1/   <stem>.png
        cloudy_t2/   <stem>.png
        ...
        cloudy_tL/   <stem>.png
        target/      <stem>.png
        masks/       <stem>.png  (optional)

Usage
-----
    from dataset_cloud import MultiTemporalCloudDataset

    ds = MultiTemporalCloudDataset(
        root_dir   = '/data/Sen2MTC/train',
        L          = 3,
        patch_size = 256,
        layout     = 'Sen2MTC',   # or 'flat'
        augment    = True,
    )
    # ds[i] → (cloudy_seq, target, cloud_masks)
    #          cloudy_seq  : [L, C, H, W]  float32 in [0, 1]
    #          target      : [C, H, W]     float32 in [0, 1]
    #          cloud_masks : [L, 1, H, W]  float32  1=cloud / 0=clear
    #                        (zeros if no mask folder found)
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


def _load_img(path: str) -> np.ndarray:
    """Load image as float32 [H, W, C] in [0, 1]."""
    img = Image.open(path).convert('RGB')
    return np.array(img, dtype=np.float32) / 255.0


def _load_mask(path: str) -> np.ndarray:
    """Load cloud mask as float32 [H, W, 1] where 1=cloud, 0=clear."""
    img = Image.open(path).convert('L')
    arr = np.array(img, dtype=np.float32) / 255.0   # 255=clear → 1.0=clear
    arr = 1.0 - arr                                  # invert: 1.0 = cloud
    return arr[:, :, np.newaxis]                     # [H, W, 1]


class MultiTemporalCloudDataset(Dataset):
    """
    Parameters
    ----------
    root_dir   : str   path to the split directory (e.g. '.../train')
    L          : int   number of temporal images per sample
    patch_size : int   random crop size (set 0 to use full image)
    layout     : str   'Sen2MTC' or 'flat'
    augment    : bool  random horizontal/vertical flip + 90° rotation
    """

    # Folder name templates for each layout
    _SEN2MTC_CLOUDY  = ['s2_t1', 's2_t2', 's2_t3']   # extend if L > 3
    _SEN2MTC_TARGET  = 's2_target'
    _SEN2MTC_MASKS   = 'masks'

    def __init__(
        self,
        root_dir:   str,
        L:          int  = 3,
        patch_size: int  = 256,
        layout:     str  = 'Sen2MTC',
        augment:    bool = True,
    ):
        super().__init__()
        self.root_dir   = root_dir
        self.L          = L
        self.patch_size = patch_size
        self.augment    = augment
        self.layout     = layout

        # ── Build folder paths ───────────────────────────────────────────
        if layout == 'Sen2MTC':
            # Use the first L entries from the template list
            self._cloudy_dirs = [
                os.path.join(root_dir, f's2_t{l+1}') for l in range(L)
            ]
            self._target_dir = os.path.join(root_dir, self._SEN2MTC_TARGET)
            self._mask_dir   = os.path.join(root_dir, self._SEN2MTC_MASKS)
        elif layout == 'flat':
            self._cloudy_dirs = [
                os.path.join(root_dir, f'cloudy_t{l+1}') for l in range(L)
            ]
            self._target_dir = os.path.join(root_dir, 'target')
            self._mask_dir   = os.path.join(root_dir, 'masks')
        else:
            raise ValueError(f"Unknown layout '{layout}'. Use 'Sen2MTC' or 'flat'.")

        # ── Verify required dirs exist ───────────────────────────────────
        for d in self._cloudy_dirs + [self._target_dir]:
            if not os.path.isdir(d):
                raise FileNotFoundError(
                    f"Directory not found: {d}\n"
                    f"Check --train_dir / --val_dir and --layout."
                )

        self._has_masks = os.path.isdir(self._mask_dir)
        if not self._has_masks:
            print(f"[CloudDataset] No mask dir found at {self._mask_dir}. "
                  "Cloud detection BCE loss will be disabled.")

        # ── Build file list from target dir ──────────────────────────────
        self._stems = sorted(
            os.path.splitext(f)[0]
            for f in os.listdir(self._target_dir)
            if f.lower().endswith(('.png', '.jpg', '.tif', '.tiff'))
        )
        assert len(self._stems) > 0, f"No images found in {self._target_dir}"
        print(f"[CloudDataset] {layout} | {len(self._stems)} samples | "
              f"L={L} | patch={patch_size} | masks={'yes' if self._has_masks else 'no'}")

    def __len__(self):
        return len(self._stems)

    # ── Internal helpers ─────────────────────────────────────────────────────
    def _find_file(self, directory: str, stem: str) -> str:
        """Find image file with given stem, trying common extensions."""
        for ext in ('.png', '.jpg', '.tif', '.tiff', '.PNG', '.JPG'):
            path = os.path.join(directory, stem + ext)
            if os.path.isfile(path):
                return path
        raise FileNotFoundError(
            f"Cannot find '{stem}.*' in {directory}"
        )

    def _sync_augment(self, arrays: list) -> list:
        """Apply same random flip/rotation to all arrays [H, W, C]."""
        if random.random() > 0.5:
            arrays = [np.flipud(a).copy() for a in arrays]
        if random.random() > 0.5:
            arrays = [np.fliplr(a).copy() for a in arrays]
        k = random.randint(0, 3)
        if k > 0:
            arrays = [np.rot90(a, k).copy() for a in arrays]
        return arrays

    def _crop(self, arrays: list, h: int, w: int) -> list:
        """Random crop – same region for all arrays."""
        ps = self.patch_size
        r  = random.randint(0, h - ps)
        c  = random.randint(0, w - ps)
        return [a[r:r+ps, c:c+ps] for a in arrays]

    # ── __getitem__ ──────────────────────────────────────────────────────────
    def __getitem__(self, idx: int):
        stem = self._stems[idx]

        # Load target
        target_path = self._find_file(self._target_dir, stem)
        target_img  = _load_img(target_path)          # [H, W, C]
        H, W, _     = target_img.shape

        # Load L temporal cloudy images
        cloudy_imgs = [
            _load_img(self._find_file(d, stem))
            for d in self._cloudy_dirs
        ]  # list of L × [H, W, C]

        # Load L cloud masks (or zeros)
        if self._has_masks:
            cloud_masks = [
                _load_mask(self._find_file(self._mask_dir, stem))
                for _ in range(self.L)
            ]  # list of L × [H, W, 1]
            # If only one mask file per scene, replicate across L
        else:
            cloud_masks = [np.zeros((H, W, 1), dtype=np.float32)
                           for _ in range(self.L)]

        # ── Random patch crop ────────────────────────────────────────────
        if self.patch_size > 0 and H > self.patch_size and W > self.patch_size:
            all_arrays = cloudy_imgs + [target_img] + cloud_masks
            all_arrays = self._crop(all_arrays, H, W)
            cloudy_imgs  = all_arrays[:self.L]
            target_img   = all_arrays[self.L]
            cloud_masks  = all_arrays[self.L + 1:]

        # ── Augmentation ─────────────────────────────────────────────────
        if self.augment:
            all_arrays   = cloudy_imgs + [target_img] + cloud_masks
            all_arrays   = self._sync_augment(all_arrays)
            cloudy_imgs  = all_arrays[:self.L]
            target_img   = all_arrays[self.L]
            cloud_masks  = all_arrays[self.L + 1:]

        # ── To tensors ───────────────────────────────────────────────────
        # [H, W, C] → [C, H, W]
        cloudy_tensors = torch.stack([
            torch.from_numpy(np.transpose(img, (2, 0, 1)))
            for img in cloudy_imgs
        ], dim=0)                                      # [L, C, H, W]

        target_tensor = torch.from_numpy(
            np.transpose(target_img, (2, 0, 1))
        )                                              # [C, H, W]

        mask_tensors = torch.stack([
            torch.from_numpy(np.transpose(m, (2, 0, 1)))
            for m in cloud_masks
        ], dim=0)                                      # [L, 1, H, W]

        return cloudy_tensors, target_tensor, mask_tensors


# ─── SingleTemporalCloudDataset ───────────────────────────────────────────────
class SingleTemporalCloudDataset(Dataset):
    """
    Single-temporal cloud removal dataset  (CUHK-CR1 / CUHK-CR2).

    Expected folder layout
    ----------------------
        root_dir/
            input/    <stem>.png   ← cloud-degraded image
            target/   <stem>.png   ← cloud-free reference

    Cloud masks are synthesised on-the-fly as the absolute difference
    |input − target| (thresholded, then blurred) — no explicit mask files
    are required for CUHK-CR1 or CUHK-CR2.

    Alternatively, if a ``masks/`` sub-directory exists it will be loaded
    and used directly (same format as MultiTemporalCloudDataset masks:
    pixel 0 = cloud, pixel 255 = clear, stored as float 1=cloud / 0=clear).

    Returns
    -------
    (input_tensor, target_tensor, cloud_mask)
        input_tensor : [C, H, W]  float32 in [0, 1]
        target_tensor: [C, H, W]  float32 in [0, 1]
        cloud_mask   : [1, H, W]  float32  1=cloud, 0=clear
    """

    def __init__(
        self,
        root_dir:   str,
        patch_size: int  = 256,
        augment:    bool = True,
        mask_thresh: float = 0.05,   # |input-target| threshold for auto-mask
    ):
        super().__init__()
        self.root_dir    = root_dir
        self.patch_size  = patch_size
        self.augment     = augment
        self.mask_thresh = mask_thresh

        self._input_dir  = os.path.join(root_dir, 'input')
        self._target_dir = os.path.join(root_dir, 'target')
        self._mask_dir   = os.path.join(root_dir, 'masks')

        for d in (self._input_dir, self._target_dir):
            if not os.path.isdir(d):
                raise FileNotFoundError(
                    f"Directory not found: {d}\n"
                    f"Check --train_dir / --val_dir."
                )

        self._has_masks = os.path.isdir(self._mask_dir)
        if self._has_masks:
            print(f"[SingleCloudDataset] Using explicit masks from {self._mask_dir}")
        else:
            print(f"[SingleCloudDataset] No mask dir; synthesising masks from "
                  f"|input−target| > {mask_thresh}")

        self._stems = sorted(
            os.path.splitext(f)[0]
            for f in os.listdir(self._target_dir)
            if f.lower().endswith(('.png', '.jpg', '.tif', '.tiff'))
        )
        assert len(self._stems) > 0, f"No images found in {self._target_dir}"
        print(f"[SingleCloudDataset] {len(self._stems)} samples | "
              f"patch={patch_size} | augment={augment}")

    def __len__(self):
        return len(self._stems)

    def _find_file(self, directory: str, stem: str) -> str:
        for ext in ('.png', '.jpg', '.tif', '.tiff', '.PNG', '.JPG'):
            path = os.path.join(directory, stem + ext)
            if os.path.isfile(path):
                return path
        raise FileNotFoundError(f"Cannot find '{stem}.*' in {directory}")

    def _sync_augment(self, arrays: list) -> list:
        if random.random() > 0.5:
            arrays = [np.flipud(a).copy() for a in arrays]
        if random.random() > 0.5:
            arrays = [np.fliplr(a).copy() for a in arrays]
        k = random.randint(0, 3)
        if k > 0:
            arrays = [np.rot90(a, k).copy() for a in arrays]
        return arrays

    def _crop(self, arrays: list, h: int, w: int) -> list:
        ps = self.patch_size
        r  = random.randint(0, h - ps)
        c  = random.randint(0, w - ps)
        return [a[r:r+ps, c:c+ps] for a in arrays]

    def __getitem__(self, idx: int):
        stem       = self._stems[idx]
        input_img  = _load_img(self._find_file(self._input_dir,  stem))  # [H,W,C]
        target_img = _load_img(self._find_file(self._target_dir, stem))  # [H,W,C]
        H, W, _    = target_img.shape

        if self._has_masks:
            cloud_mask = _load_mask(self._find_file(self._mask_dir, stem))  # [H,W,1]
        else:
            # Auto-synthesis: pixels where input differs from target are cloudy
            diff       = np.abs(input_img - target_img).mean(axis=2, keepdims=True)
            cloud_mask = (diff > self.mask_thresh).astype(np.float32)       # [H,W,1]

        if self.patch_size > 0 and H > self.patch_size and W > self.patch_size:
            input_img, target_img, cloud_mask = self._crop(
                [input_img, target_img, cloud_mask], H, W)

        if self.augment:
            input_img, target_img, cloud_mask = self._sync_augment(
                [input_img, target_img, cloud_mask])

        input_t  = torch.from_numpy(np.transpose(input_img,  (2, 0, 1)))
        target_t = torch.from_numpy(np.transpose(target_img, (2, 0, 1)))
        mask_t   = torch.from_numpy(np.transpose(cloud_mask, (2, 0, 1)))

        return input_t, target_t, mask_t
