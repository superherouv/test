from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import nibabel as nib
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from losses import remap_mask


MODALITIES = ("t1", "t1ce", "t2", "flair")


@dataclass(frozen=True)
class SubjectPaths:
    subject_id: str
    subject_dir: Path
    image_paths: dict[str, Path]
    seg_path: Path


class BraTSSubjectDiscovery:
    def __init__(self, data_root: str | Path) -> None:
        self.data_root = Path(data_root)

    def discover(self) -> list[SubjectPaths]:
        if not self.data_root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.data_root}")

        subjects: list[SubjectPaths] = []
        for subject_dir in sorted(path for path in self.data_root.iterdir() if path.is_dir()):
            subject_id = subject_dir.name
            image_paths = {m: subject_dir / f"{subject_id}_{m}.nii.gz" for m in MODALITIES}
            seg_path = subject_dir / f"{subject_id}_seg.nii.gz"
            if seg_path.exists() and all(path.exists() for path in image_paths.values()):
                subjects.append(
                    SubjectPaths(
                        subject_id=subject_id,
                        subject_dir=subject_dir,
                        image_paths=image_paths,
                        seg_path=seg_path,
                    )
                )
        if not subjects:
            raise RuntimeError(
                "No BraTS subjects were found. Make sure each subject folder contains t1/t1ce/t2/flair/seg .nii.gz files."
            )
        return subjects


def split_subjects(
    subjects: Sequence[SubjectPaths],
    test_size: float = 0.3,
    random_state: int = 42,
) -> tuple[list[SubjectPaths], list[SubjectPaths]]:
    train_subjects, test_subjects = train_test_split(
        list(subjects), test_size=test_size, random_state=random_state, shuffle=True
    )
    return sorted(train_subjects, key=lambda item: item.subject_id), sorted(test_subjects, key=lambda item: item.subject_id)


class BraTSSliceDataset(Dataset):
    def __init__(
        self,
        subjects: Sequence[SubjectPaths],
        skip_empty_slices: bool = False,
        slice_axis: int = 2,
    ) -> None:
        self.subjects = list(subjects)
        self.skip_empty_slices = skip_empty_slices
        self.slice_axis = slice_axis
        self.index: list[tuple[int, int]] = []
        self.cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        for subject_idx, subject in enumerate(self.subjects):
            seg = nib.load(str(subject.seg_path)).get_fdata().astype(np.int16)
            total_slices = seg.shape[self.slice_axis]
            for slice_idx in range(total_slices):
                mask_slice = np.take(seg, slice_idx, axis=self.slice_axis)
                if self.skip_empty_slices and np.max(mask_slice) == 0:
                    continue
                self.index.append((subject_idx, slice_idx))

        if not self.index:
            raise RuntimeError("No valid slices were found for the selected subjects.")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        subject_idx, slice_idx = self.index[idx]
        subject = self.subjects[subject_idx]
        image_volume, mask_volume = self._load_subject(subject)

        image_slice = np.take(image_volume, slice_idx, axis=self.slice_axis + 1)
        mask_slice = np.take(mask_volume, slice_idx, axis=self.slice_axis)

        image_tensor = torch.from_numpy(image_slice.astype(np.float32))
        mask_tensor = torch.from_numpy(mask_slice.astype(np.int64))
        mask_tensor = remap_mask(mask_tensor)
        return image_tensor, mask_tensor

    def _load_subject(self, subject: SubjectPaths) -> tuple[np.ndarray, np.ndarray]:
        if subject.subject_id in self.cache:
            return self.cache[subject.subject_id]

        modalities = []
        for modality in MODALITIES:
            volume = nib.load(str(subject.image_paths[modality])).get_fdata().astype(np.float32)
            modalities.append(self._normalize(volume))

        image_volume = np.stack(modalities, axis=0)
        mask_volume = nib.load(str(subject.seg_path)).get_fdata().astype(np.int16)
        self.cache[subject.subject_id] = (image_volume, mask_volume)
        return image_volume, mask_volume

    @staticmethod
    def _normalize(volume: np.ndarray) -> np.ndarray:
        non_zero = volume != 0
        if not np.any(non_zero):
            return volume
        mean = volume[non_zero].mean()
        std = volume[non_zero].std()
        std = std if std > 0 else 1.0
        output = volume.copy()
        output[non_zero] = (output[non_zero] - mean) / std
        return output
