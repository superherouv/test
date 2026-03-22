from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import BraTSSubjectDiscovery, BraTSSliceDataset, split_subjects
from losses import DiceCrossEntropyLoss, multiclass_dice_score
from model import UNet2D
from utils import ensure_dir, plot_training_loss, save_json, save_lines, save_sample_predictions, setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a 2D U-Net on BraTS 2021.")
    parser.add_argument("--data-root", type=str, required=True, help="Root directory of the BraTS 2021 subject folders.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory used to save checkpoints, metrics, figures, and splits.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs. Default is 1 to match the assignment's quick-run suggestion.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for 2D slices.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for AdamW.")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of dataloader workers.")
    parser.add_argument("--test-size", type=float, default=0.3, help="Test split ratio.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed used for splitting and training.")
    parser.add_argument("--base-channels", type=int, default=32, help="Base number of U-Net channels.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Training device.")
    parser.add_argument("--skip-empty-slices", action="store_true", help="Skip axial slices with no tumor labels.")
    parser.add_argument("--disable-amp", action="store_true", help="Disable automatic mixed precision on CUDA.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, float, dict[str, float], tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None]:
    model.eval()
    running_loss = 0.0
    running_dice = 0.0
    per_class_accumulator = {"NCR": 0.0, "ED": 0.0, "ET": 0.0}
    sample_batch = None

    for batch_idx, (images, masks) in enumerate(tqdm(loader, desc="Evaluating", leave=False)):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, masks)
        dice, per_class = multiclass_dice_score(logits, masks)

        running_loss += loss.item()
        running_dice += dice
        for key, value in per_class.items():
            per_class_accumulator[key] += value

        if sample_batch is None:
            sample_batch = (images[:3].detach().cpu(), logits[:3].detach().cpu(), masks[:3].detach().cpu())

    num_batches = max(len(loader), 1)
    mean_loss = running_loss / num_batches
    mean_dice = running_dice / num_batches
    mean_per_class = {key: value / num_batches for key, value in per_class_accumulator.items()}
    return mean_loss, mean_dice, mean_per_class, sample_batch


def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: AdamW, scaler: torch.amp.GradScaler | None, device: torch.device, use_amp: bool) -> float:
    model.train()
    running_loss = 0.0

    for images, masks in tqdm(loader, desc="Training", leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.autocast(device_type=device.type, enabled=True):
                logits = model(images)
                loss = criterion(logits, masks)
            assert scaler is not None
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

        running_loss += loss.item()

    return running_loss / max(len(loader), 1)


def main() -> None:
    args = parse_args()
    set_seed(args.random_state)

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.disable_amp

    output_dir = ensure_dir(args.output_dir)
    checkpoints_dir = ensure_dir(output_dir / "checkpoints")
    figures_dir = ensure_dir(output_dir / "figures")
    splits_dir = ensure_dir(output_dir / "splits")
    logger = setup_logger(output_dir / "train.log")

    discovery = BraTSSubjectDiscovery(args.data_root)
    subjects = discovery.discover()
    train_subjects, test_subjects = split_subjects(subjects, test_size=args.test_size, random_state=args.random_state)

    save_lines(splits_dir / "train_subjects.txt", [subject.subject_id for subject in train_subjects])
    save_lines(splits_dir / "test_subjects.txt", [subject.subject_id for subject in test_subjects])

    train_dataset = BraTSSliceDataset(train_subjects, skip_empty_slices=args.skip_empty_slices)
    test_dataset = BraTSSliceDataset(test_subjects, skip_empty_slices=args.skip_empty_slices)

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    model = UNet2D(in_channels=4, num_classes=4, base_channels=args.base_channels).to(device)
    criterion = DiceCrossEntropyLoss(num_classes=4)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    logger.info("Device: %s", device)
    logger.info("Train subjects: %d | Test subjects: %d", len(train_subjects), len(test_subjects))
    logger.info("Train slices: %d | Test slices: %d", len(train_dataset), len(test_dataset))
    logger.info("AMP enabled: %s", use_amp)

    history: list[dict] = []
    best_dice = -1.0
    best_metrics: dict | None = None

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, use_amp)
        test_loss, test_dice, test_per_class, sample_batch = evaluate(model, test_loader, criterion, device)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": test_loss,
            "val_dice": test_dice,
            "dice_ncr": test_per_class["NCR"],
            "dice_ed": test_per_class["ED"],
            "dice_et": test_per_class["ET"],
        }
        history.append(row)

        logger.info(
            "Epoch %d/%d | train_loss=%.4f | test_loss=%.4f | test_dice=%.4f | NCR=%.4f | ED=%.4f | ET=%.4f",
            epoch,
            args.epochs,
            train_loss,
            test_loss,
            test_dice,
            test_per_class["NCR"],
            test_per_class["ED"],
            test_per_class["ET"],
        )

        torch.save({"model_state": model.state_dict(), "epoch": epoch, "args": vars(args)}, checkpoints_dir / "last_model.pt")
        if test_dice > best_dice:
            best_dice = test_dice
            best_metrics = {
                "epoch": epoch,
                "mean_dice": test_dice,
                "test_loss": test_loss,
                "per_class_dice": test_per_class,
                "device": str(device),
                "train_subject_count": len(train_subjects),
                "test_subject_count": len(test_subjects),
                "train_slice_count": len(train_dataset),
                "test_slice_count": len(test_dataset),
            }
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "args": vars(args)}, checkpoints_dir / "best_model.pt")
            if sample_batch is not None:
                save_sample_predictions(*sample_batch, output_path=figures_dir / "sample_predictions.png")

    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    plot_training_loss(history, figures_dir / "training_loss.png")
    save_json(output_dir / "metrics.json", best_metrics or {})
    logger.info("Training finished. Best mean Dice: %.4f", best_dice)


if __name__ == "__main__":
    main()
