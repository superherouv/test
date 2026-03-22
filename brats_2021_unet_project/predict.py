from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from tqdm import tqdm

from dataset import MODALITIES
from losses import INV_LABEL_MAP
from model import UNet2D
from utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference for a BraTS 2021 subject using a trained 2D U-Net.")
    parser.add_argument("--subject-dir", type=str, required=True, help="Path to one BraTS subject folder.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the saved .pt checkpoint.")
    parser.add_argument("--output-path", type=str, required=True, help="Path to save the predicted segmentation NIfTI file.")
    parser.add_argument("--base-channels", type=int, default=32, help="Base channel count used during training.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    return parser.parse_args()


def normalize(volume: np.ndarray) -> np.ndarray:
    non_zero = volume != 0
    if not np.any(non_zero):
        return volume
    mean = volume[non_zero].mean()
    std = volume[non_zero].std()
    std = std if std > 0 else 1.0
    result = volume.copy()
    result[non_zero] = (result[non_zero] - mean) / std
    return result


def restore_brats_labels(mask: np.ndarray) -> np.ndarray:
    restored = np.zeros_like(mask)
    for new_value, original_value in INV_LABEL_MAP.items():
        restored[mask == new_value] = original_value
    return restored.astype(np.int16)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    subject_dir = Path(args.subject_dir)
    subject_id = subject_dir.name

    image_paths = {m: subject_dir / f"{subject_id}_{m}.nii.gz" for m in MODALITIES}
    reference_path = image_paths["flair"]

    for modality, path in image_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing modality {modality}: {path}")

    channels = []
    for modality in MODALITIES:
        volume = nib.load(str(image_paths[modality])).get_fdata().astype(np.float32)
        channels.append(normalize(volume))
    stacked = np.stack(channels, axis=0)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = UNet2D(in_channels=4, num_classes=4, base_channels=args.base_channels).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    predictions = []
    with torch.no_grad():
        for slice_idx in tqdm(range(stacked.shape[-1]), desc="Predicting"):
            image_slice = stacked[:, :, :, slice_idx]
            tensor = torch.from_numpy(image_slice).unsqueeze(0).to(device)
            logits = model(tensor)
            pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
            predictions.append(pred)

    pred_volume = np.stack(predictions, axis=-1)
    pred_volume = restore_brats_labels(pred_volume)

    reference = nib.load(str(reference_path))
    output_path = Path(args.output_path)
    ensure_dir(output_path.parent)
    nib.save(nib.Nifti1Image(pred_volume, affine=reference.affine, header=reference.header), str(output_path))
    print(f"Saved prediction to {output_path}")


if __name__ == "__main__":
    main()
