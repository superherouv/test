from __future__ import annotations

import torch
import torch.nn.functional as F


LABEL_MAP = {0: 0, 1: 1, 2: 2, 4: 3}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


def remap_mask(mask: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(mask, dtype=torch.long)
    for old_value, new_value in LABEL_MAP.items():
        out[mask == old_value] = new_value
    return out


def one_hot_encode(mask: torch.Tensor, num_classes: int) -> torch.Tensor:
    return F.one_hot(mask.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()


class DiceCrossEntropyLoss(torch.nn.Module):
    def __init__(self, num_classes: int = 4, smooth: float = 1e-5, ce_weight: float = 1.0, dice_weight: float = 1.0) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target)
        probs = torch.softmax(logits, dim=1)
        target_one_hot = one_hot_encode(target, self.num_classes)

        probs_fg = probs[:, 1:]
        target_fg = target_one_hot[:, 1:]

        dims = (0, 2, 3)
        intersection = torch.sum(probs_fg * target_fg, dim=dims)
        denominator = torch.sum(probs_fg + target_fg, dim=dims)
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        dice_loss = 1.0 - dice.mean()
        return self.ce_weight * ce + self.dice_weight * dice_loss


@torch.no_grad()
def multiclass_dice_score(logits: torch.Tensor, target: torch.Tensor, smooth: float = 1e-5) -> tuple[float, dict[str, float]]:
    pred = torch.argmax(logits, dim=1)
    class_names = {1: "NCR", 2: "ED", 3: "ET"}
    per_class: dict[str, float] = {}
    dice_values = []

    for class_index, class_name in class_names.items():
        pred_mask = (pred == class_index).float()
        target_mask = (target == class_index).float()
        intersection = (pred_mask * target_mask).sum()
        denominator = pred_mask.sum() + target_mask.sum()
        dice = ((2.0 * intersection + smooth) / (denominator + smooth)).item()
        per_class[class_name] = dice
        dice_values.append(dice)

    mean_dice = float(sum(dice_values) / len(dice_values))
    return mean_dice, per_class
