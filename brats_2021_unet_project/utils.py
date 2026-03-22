from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch

from losses import INV_LABEL_MAP


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def setup_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger(str(log_file))
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def save_lines(path: Path, values: Iterable[str]) -> None:
    path.write_text("\n".join(values) + "\n", encoding="utf-8")


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def plot_training_loss(history: list[dict], output_path: Path) -> None:
    epochs = [row["epoch"] for row in history]
    train_losses = [row["train_loss"] for row in history]
    val_losses = [row["val_loss"] for row in history]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, marker="o", label="Train loss")
    plt.plot(epochs, val_losses, marker="s", label="Test loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("BraTS 2021 U-Net loss")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


@torch.no_grad()
def save_sample_predictions(
    images: torch.Tensor,
    logits: torch.Tensor,
    masks: torch.Tensor,
    output_path: Path,
) -> None:
    pred = torch.argmax(logits, dim=1).cpu().numpy()
    images = images.cpu().numpy()
    masks = masks.cpu().numpy()

    num_rows = min(3, images.shape[0])
    fig, axes = plt.subplots(num_rows, 3, figsize=(12, 4 * num_rows))
    if num_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for row in range(num_rows):
        flair = images[row, 3]
        axes[row, 0].imshow(flair, cmap="gray")
        axes[row, 0].set_title("FLAIR slice")
        axes[row, 1].imshow(_restore_brats_labels(masks[row]), cmap="viridis", vmin=0, vmax=4)
        axes[row, 1].set_title("Ground truth")
        axes[row, 2].imshow(_restore_brats_labels(pred[row]), cmap="viridis", vmin=0, vmax=4)
        axes[row, 2].set_title("Prediction")
        for col in range(3):
            axes[row, col].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def _restore_brats_labels(mask: np.ndarray) -> np.ndarray:
    restored = np.zeros_like(mask)
    for new_value, original_value in INV_LABEL_MAP.items():
        restored[mask == new_value] = original_value
    return restored
