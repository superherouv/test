"""
viz_utils.py – Rich visualization utilities for SNN cloud-removal / deraining training.

Public API
----------
error_heatmap(pred, target)             → [3,H,W] jet-colored error tensor
cloud_heatmap(cloud_map)               → [3,H,W] inferno cloud-prob tensor
save_rich_comparison_multi(...)        → matplotlib figure (multi-temporal cloud removal)
save_rich_comparison_single(...)       → matplotlib figure (single-temporal cloud removal)
save_rich_comparison_derain(...)       → matplotlib figure (deraining)
log_images_tensorboard(writer, ...)    → log named image grids to TensorBoard
save_metric_curves(history, path)      → matplotlib multi-panel metric history chart
save_cloud_coverage_plot(hist, path)   → cloud coverage area-fill line chart
"""

import os
import numpy as np
import torch
import torchvision.utils as vutils
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec

__all__ = [
    'error_heatmap', 'cloud_heatmap',
    'save_rich_comparison_multi', 'save_rich_comparison_single',
    'save_rich_comparison_derain',
    'log_images_tensorboard',
    'save_metric_curves', 'save_cloud_coverage_plot',
]

# ─── internal helpers ──────────────────────────────────────────────────────────

def _np(t: torch.Tensor) -> np.ndarray:
    """[C,H,W] float tensor → [H,W,C] uint8 ndarray."""
    return (t.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def _psnr_float(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = ((pred.detach().cpu().clamp(0, 1) -
             target.detach().cpu().clamp(0, 1)) ** 2).mean().item()
    return 10 * np.log10(1.0 / (mse + 1e-8))


def _safe_mkdir(path: str):
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)


# ─── colormap helpers ──────────────────────────────────────────────────────────

def error_heatmap(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return a [3,H,W] jet-colorized absolute-error tensor (values in [0,1])."""
    err = (pred.detach().cpu().clamp(0, 1) -
           target.detach().cpu().clamp(0, 1)).abs().mean(0).numpy()
    err = (err - err.min()) / (err.max() - err.min() + 1e-8)
    rgb = cm.jet(err)[:, :, :3].astype(np.float32)
    return torch.from_numpy(rgb).permute(2, 0, 1)


def cloud_heatmap(cloud_map: torch.Tensor) -> torch.Tensor:
    """Return a [3,H,W] inferno-colorized cloud-probability tensor."""
    arr = cloud_map.detach().cpu().squeeze(0).numpy()
    arr = np.clip(arr, 0, 1)
    rgb = cm.inferno(arr)[:, :, :3].astype(np.float32)
    return torch.from_numpy(rgb).permute(2, 0, 1)


# ─── shared subplot helper ────────────────────────────────────────────────────

def _show(ax, t: torch.Tensor, title: str = '', badge: str = '',
          title_color: str = 'black'):
    """Imshow [C,H,W] tensor; optional column header and PSNR badge overlay."""
    ax.imshow(_np(t))
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.4)
    if title:
        ax.set_title(title, fontsize=6.8, color=title_color, pad=2,
                     fontweight='bold')
    if badge:
        ax.text(3, 3, badge, fontsize=6.5, color='#00ff88', va='top',
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.58))


# ─── rich comparison grids ────────────────────────────────────────────────────

def save_rich_comparison_multi(cloudy_seq, target, restored, cloud_maps,
                                path: str, n: int = 4):
    """
    Multi-temporal cloud removal comparison grid.

    Columns: Cloudy_T1 … Cloudy_TL | Target(GT) | Restored | Error Map |
             CloudMap_T1 … CloudMap_TL

    Args:
        cloudy_seq : [L, B, C, H, W]
        target     : [B, C, H, W]
        restored   : [B, C, H, W]  (clamped or raw)
        cloud_maps : [L, B, 1, H, W]  predicted cloud probability maps
        path       : output PNG path
        n          : max number of samples (rows) to display
    """
    L = cloudy_seq.shape[0]
    B = target.shape[0]
    n = min(n, B)
    ncols = L + 3 + L          # L cloudy | target | restored | error | L clouds

    fig, axes = plt.subplots(
        n, ncols,
        figsize=(2.25 * ncols, 2.5 * n),
        gridspec_kw={'wspace': 0.04, 'hspace': 0.30},
    )
    if n == 1:
        axes = axes[np.newaxis, :]

    col_labels = (
        [f'Cloudy T{i+1}' for i in range(L)] +
        ['Target (GT)', 'Restored', 'Error Map'] +
        [f'Cloud Map T{i+1}' for i in range(L)]
    )
    # Column headers (only on first row)
    for c, lbl in enumerate(col_labels):
        axes[0, c].set_title(lbl, fontsize=7, pad=2, fontweight='bold')

    for r in range(n):
        p = _psnr_float(restored[r], target[r])
        c = 0
        for l in range(L):
            _show(axes[r, c], cloudy_seq[l, r])
            c += 1
        _show(axes[r, c], target[r]);                               c += 1
        _show(axes[r, c], restored[r].clamp(0, 1),
              badge=f'{p:.2f} dB');                                 c += 1
        _show(axes[r, c], error_heatmap(restored[r].clamp(0, 1), target[r]))
        c += 1
        for l in range(L):
            _show(axes[r, c], cloud_heatmap(cloud_maps[l, r]));    c += 1
        # Row label on y-axis of first column
        axes[r, 0].set_ylabel(f'#{r+1}', fontsize=7, rotation=0,
                               labelpad=14, va='center', color='#555555')

    fig.suptitle('Multi-temporal Cloud Removal — Epoch Snapshot',
                 fontsize=9, fontweight='bold', y=1.01)
    _safe_mkdir(path)
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def save_rich_comparison_single(inp, target, restored, cloud_map,
                                 path: str, n: int = 4):
    """
    Single-temporal cloud removal comparison grid.

    Columns: Input (Cloudy) | Target (GT) | Restored | Error Map | Cloud Map

    Args:
        inp        : [B, C, H, W]
        target     : [B, C, H, W]
        restored   : [B, C, H, W]
        cloud_map  : [B, 1, H, W]
        path       : output PNG path
        n          : max number of samples (rows)
    """
    B = target.shape[0]
    n = min(n, B)
    ncols = 5

    fig, axes = plt.subplots(
        n, ncols,
        figsize=(2.25 * ncols, 2.5 * n),
        gridspec_kw={'wspace': 0.04, 'hspace': 0.30},
    )
    if n == 1:
        axes = axes[np.newaxis, :]

    col_labels = ['Input (Cloudy)', 'Target (GT)', 'Restored',
                  'Error Map', 'Cloud Map']
    for c, lbl in enumerate(col_labels):
        axes[0, c].set_title(lbl, fontsize=7, pad=2, fontweight='bold')

    for r in range(n):
        p = _psnr_float(restored[r], target[r])
        _show(axes[r, 0], inp[r])
        _show(axes[r, 1], target[r])
        _show(axes[r, 2], restored[r].clamp(0, 1), badge=f'{p:.2f} dB')
        _show(axes[r, 3], error_heatmap(restored[r].clamp(0, 1), target[r]))
        _show(axes[r, 4], cloud_heatmap(cloud_map[r]))
        axes[r, 0].set_ylabel(f'#{r+1}', fontsize=7, rotation=0,
                               labelpad=14, va='center', color='#555555')

    fig.suptitle('Single-temporal Cloud Removal — Epoch Snapshot',
                 fontsize=9, fontweight='bold', y=1.01)
    _safe_mkdir(path)
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def save_rich_comparison_derain(inp, target, restored, path: str, n: int = 4):
    """
    Image deraining comparison grid.

    Columns: Rainy Input | Clean Target | Restored | Error Map

    Args:
        inp/target/restored : [B, C, H, W]
        path                : output PNG path
        n                   : max number of samples (rows)
    """
    B = target.shape[0]
    n = min(n, B)
    ncols = 4

    fig, axes = plt.subplots(
        n, ncols,
        figsize=(2.6 * ncols, 2.7 * n),
        gridspec_kw={'wspace': 0.04, 'hspace': 0.30},
    )
    if n == 1:
        axes = axes[np.newaxis, :]

    col_labels = ['Rainy Input', 'Clean Target', 'Restored', 'Error Map']
    for c, lbl in enumerate(col_labels):
        axes[0, c].set_title(lbl, fontsize=7, pad=2, fontweight='bold')

    for r in range(n):
        p = _psnr_float(restored[r], target[r])
        _show(axes[r, 0], inp[r])
        _show(axes[r, 1], target[r])
        _show(axes[r, 2], restored[r].clamp(0, 1), badge=f'{p:.2f} dB')
        _show(axes[r, 3], error_heatmap(restored[r].clamp(0, 1), target[r]))
        axes[r, 0].set_ylabel(f'#{r+1}', fontsize=7, rotation=0,
                               labelpad=14, va='center', color='#555555')

    fig.suptitle('Image Deraining — Epoch Snapshot',
                 fontsize=9, fontweight='bold', y=1.01)
    _safe_mkdir(path)
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)


# ─── TensorBoard image logging ────────────────────────────────────────────────

def log_images_tensorboard(writer, tag: str, images_dict: dict,
                            step: int, nrow: int = 4):
    """
    Log named image tensors as TensorBoard image grids.

    Args:
        writer      : SummaryWriter instance
        tag         : top-level TensorBoard tag prefix
        images_dict : {suffix_str: tensor [B,C,H,W] or [C,H,W]}
        step        : global step (epoch)
        nrow        : images per row in the grid
    """
    for suffix, imgs in images_dict.items():
        if imgs.dim() == 3:
            imgs = imgs.unsqueeze(0)
        imgs = imgs[:nrow].detach().cpu().clamp(0, 1)
        grid = vutils.make_grid(imgs, nrow=nrow, padding=2, normalize=False)
        writer.add_image(f'{tag}/{suffix}', grid, step)


# ─── metric history charts ────────────────────────────────────────────────────

def save_metric_curves(history: dict, path: str,
                       title: str = 'Training Metrics'):
    """
    Save a multi-panel matplotlib figure with one subplot per tracked metric.

    Args:
        history : {metric_name: [(epoch, value), ...]}
        path    : output PNG path
        title   : figure super-title
    """
    metrics = {k: v for k, v in history.items() if v}
    if not metrics:
        return

    n = len(metrics)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(5 * ncols, 3.5 * nrows),
        squeeze=False,
    )
    fig.suptitle(title, fontsize=11, y=1.01, fontweight='bold')

    palette = plt.cm.tab10.colors

    for idx, (name, vals) in enumerate(metrics.items()):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        epochs = [v[0] for v in vals]
        values = [v[1] for v in vals]
        color = palette[idx % len(palette)]

        ax.plot(epochs, values, color=color, linewidth=1.7,
                marker='o', markersize=2.5,
                markevery=max(1, len(epochs) // 20))
        ax.fill_between(epochs, values, alpha=0.13, color=color)

        # Highlight best value
        is_higher_better = any(k in name.lower() for k in ('psnr', 'ssim'))
        best_idx = (np.argmax(values) if is_higher_better
                    else np.argmin(values))
        ax.axvline(x=epochs[best_idx], color='crimson',
                   linestyle='--', alpha=0.55, linewidth=0.9)
        ax.scatter([epochs[best_idx]], [values[best_idx]],
                   color='crimson', s=35, zorder=5,
                   label=f'best={values[best_idx]:.4f} @ ep{epochs[best_idx]}')

        ax.set_title(name, fontsize=9, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=8)
        ax.grid(True, alpha=0.25, linestyle='--')
        ax.legend(fontsize=6.5, loc='best')

    # Hide extra subplots
    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].set_visible(False)

    plt.tight_layout(pad=0.6)
    _safe_mkdir(path)
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(fig)


# ─── cloud coverage area chart ────────────────────────────────────────────────

def save_cloud_coverage_plot(cloud_coverage_history: list, path: str):
    """
    Area-fill line chart of mean cloud-coverage ratio per epoch.

    Args:
        cloud_coverage_history : [(epoch, mean_coverage_ratio), ...]
        path                   : output PNG path
    """
    if not cloud_coverage_history:
        return
    epochs = [v[0] for v in cloud_coverage_history]
    vals   = [v[1] for v in cloud_coverage_history]

    fig, ax = plt.subplots(figsize=(9, 3))
    ax.fill_between(epochs, vals, alpha=0.32, color='steelblue')
    ax.plot(epochs, vals, color='steelblue', linewidth=1.7)
    ax.set_xlabel('Epoch', fontsize=9)
    ax.set_ylabel('Mean Cloud Coverage', fontsize=9)
    ax.set_title('Training Cloud Coverage per Epoch',
                 fontsize=10, fontweight='bold')
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f'{x:.1%}'))
    ax.grid(True, alpha=0.25, linestyle='--')
    plt.tight_layout()
    _safe_mkdir(path)
    fig.savefig(path, dpi=110, bbox_inches='tight')
    plt.close(fig)
