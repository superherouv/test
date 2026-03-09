"""
Training script for single-temporal cloud removal with CloudGate-SNN.

Supports CUHK-CR1 (thin clouds) and CUHK-CR2 (thick clouds).

Dataset layout expected
-----------------------
    root/
        input/   <stem>.png   ← cloud-degraded image
        target/  <stem>.png   ← cloud-free reference
        masks/   <stem>.png   ← optional cloud masks
                                 (pixel 0=cloud, pixel 255=clear)

Key differences from train_cloud.py
-------------------------------------
  - Input is [B, C, H, W] per batch (single image, no temporal stacking)
  - Model is VLIFNet with cloud_mode='single'
  - CloudGate applies a learnable per-step curriculum gating
  - Returns (restored, cloud_map) tuple
  - Loss = CloudRemovalLoss (SSIM+L1 + optional cloud-BCE)
  - Metrics: PSNR, SSIM; cPSNR over synthesised cloud mask

Example commands
----------------
    # CUHK-CR1 (thin clouds)
    python train_cloud_single.py \\
        --train_dir  /data/CUHK-CR1/train \\
        --val_dir    /data/CUHK-CR1/val \\
        --session    cukh_cr1_T4

    # CUHK-CR2 (thick clouds)
    python train_cloud_single.py \\
        --train_dir  /data/CUHK-CR2/train \\
        --val_dir    /data/CUHK-CR2/val \\
        --lambda_cloud 0.2 \\
        --session    cuhk_cr2_T4
"""

import os
os.environ["CUDA_DEVICE_ORDER"]   = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import torch
torch.backends.cudnn.benchmark = True

import torch.optim as optim
from torch.utils.data import DataLoader
import random
import numpy as np
import time
import argparse
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import torchvision.utils as vutils

from spikingjelly.activation_based import functional

import utils
from model            import model_cloud_single
from losses           import CloudRemovalLoss
from warmup_scheduler import GradualWarmupScheduler
from dataset_cloud    import SingleTemporalCloudDataset


# ─── helpers ──────────────────────────────────────────────────────────────────
def psnr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = ((pred - target) ** 2).mean()
    return 10 * torch.log10(1.0 / (mse + 1e-8))


def ssim_metric(pred: torch.Tensor, target: torch.Tensor,
                win: int = 11) -> torch.Tensor:
    """Mean SSIM (uniform kernel) over a batch."""
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    pad     = win // 2
    ch      = pred.shape[1]
    kernel  = torch.ones(ch, 1, win, win, device=pred.device,
                         dtype=pred.dtype) / (win * win)
    mu_x    = torch.nn.functional.conv2d(pred,   kernel, padding=pad, groups=ch)
    mu_y    = torch.nn.functional.conv2d(target, kernel, padding=pad, groups=ch)
    sig_x2  = torch.nn.functional.conv2d(pred   * pred,   kernel, padding=pad, groups=ch) - mu_x * mu_x
    sig_y2  = torch.nn.functional.conv2d(target * target, kernel, padding=pad, groups=ch) - mu_y * mu_y
    sig_xy  = torch.nn.functional.conv2d(pred   * target, kernel, padding=pad, groups=ch) - mu_x * mu_y
    num = (2 * mu_x * mu_y + C1) * (2 * sig_xy + C2)
    den = (mu_x * mu_x + mu_y * mu_y + C1) * (sig_x2 + sig_y2 + C2)
    return (num / den).mean()


def cpsnr(pred: torch.Tensor, target: torch.Tensor,
          cloud_mask: torch.Tensor) -> torch.Tensor:
    """PSNR restricted to cloud-covered pixels."""
    if cloud_mask.sum() < 1:
        return psnr(pred, target)
    err       = (pred - target) ** 2
    cloud_mse = (err * cloud_mask).sum() / (cloud_mask.sum() * pred.shape[1] + 1e-8)
    return 10 * torch.log10(1.0 / (cloud_mse + 1e-8))


def save_comparison(inp, target, restored, cloud_map, path):
    """Grid: (input | target | restored | cloud_map)."""
    B = target.shape[0]
    n = min(B, 4)
    rows = []
    for i in range(n):
        c_img  = inp[i].cpu()
        tgt    = target[i].cpu()
        res    = restored[i].cpu().clamp(0, 1)
        cmap   = cloud_map[i].cpu().expand(3, -1, -1)
        rows.append(torch.stack([c_img, tgt, res, cmap], dim=0))
    grid = vutils.make_grid(torch.cat(rows, dim=0), nrow=4, padding=2)
    arr  = (grid.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


# ─── main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # ── Seed ──────────────────────────────────────────────────────────────────
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)

    # ── Args ──────────────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description='CloudGate-SNN Single-temporal Cloud Removal')

    # Paths
    parser.add_argument('--train_dir',        default='', type=str)
    parser.add_argument('--val_dir',          default='', type=str)
    parser.add_argument('--model_save_dir',   default='./checkpoints/', type=str)
    parser.add_argument('--pretrain_weights', default='', type=str)
    parser.add_argument('--session',          default='CloudGateSNN_single', type=str)

    # Data
    parser.add_argument('--inp_channels',     default=3, type=int)
    parser.add_argument('--patch_size_train', default=256, type=int)
    parser.add_argument('--patch_size_test',  default=256, type=int)
    parser.add_argument('--mask_thresh',      default=0.05, type=float,
                        help='|input-target| threshold for auto cloud mask synthesis')
    parser.add_argument('--num_workers',      default=4, type=int)

    # Model
    parser.add_argument('--T',                default=4, type=int,
                        help='SNN time steps (CloudGate curriculum length)')

    # Training
    parser.add_argument('--num_epochs',       default=200, type=int)
    parser.add_argument('--batch_size',       default=8, type=int)
    parser.add_argument('--lr',               default=2e-4, type=float)
    parser.add_argument('--min_lr',           default=1e-7, type=float)
    parser.add_argument('--warmup_epochs',    default=5, type=int)
    parser.add_argument('--clip_grad',        default=1.0, type=float)
    parser.add_argument('--val_epochs',       default=5, type=int)

    # Loss
    parser.add_argument('--lambda_cloud',     default=0.1, type=float,
                        help='Weight of cloud-detection BCE loss')

    # Misc
    parser.add_argument('--use_refinement',   default=False, type=bool)
    args = parser.parse_args()

    # ── Directories ───────────────────────────────────────────────────────────
    model_dir  = os.path.join(args.model_save_dir,
                              'CloudGateSNN_single', 'models', args.session)
    sample_dir = os.path.join(args.model_save_dir,
                              'CloudGateSNN_single', 'samples', args.session)
    utils.mkdir(model_dir)
    utils.mkdir(sample_dir)
    utils.mkdir(os.path.join(sample_dir, 'train'))
    utils.mkdir(os.path.join(sample_dir, 'val'))

    # ── Model ─────────────────────────────────────────────────────────────────
    net = model_cloud_single(inp_channels=args.inp_channels,
                             T=args.T,
                             use_refinement=args.use_refinement)
    net.cuda()
    functional.set_step_mode(net, step_mode='m')
    functional.set_backend(net, backend='cupy')

    device_ids = list(range(torch.cuda.device_count()))
    print(f"[Init] GPUs: {device_ids}  |  T={args.T}  |  cloud_mode=single")

    if args.pretrain_weights and os.path.isfile(args.pretrain_weights):
        state = torch.load(args.pretrain_weights, map_location='cpu')
        net.load_state_dict(state.get('state_dict', state), strict=False)
        print(f"[Init] Loaded pretrain weights from {args.pretrain_weights}")

    if len(device_ids) > 1:
        net = torch.nn.DataParallel(net, device_ids=device_ids)

    # ── Optimiser & scheduler ─────────────────────────────────────────────────
    optimizer = optim.AdamW(net.parameters(), lr=args.lr,
                            betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4)

    scheduler_cos = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.num_epochs - args.warmup_epochs, eta_min=args.min_lr)
    scheduler = GradualWarmupScheduler(
        optimizer, multiplier=1,
        total_epoch=args.warmup_epochs, after_scheduler=scheduler_cos)

    # ── Loss ──────────────────────────────────────────────────────────────────
    criterion = CloudRemovalLoss(lambda_cloud=args.lambda_cloud,
                                 lambda_tmc=0.0).cuda()

    # ── Data ──────────────────────────────────────────────────────────────────
    train_ds = SingleTemporalCloudDataset(
        root_dir    = args.train_dir,
        patch_size  = args.patch_size_train,
        augment     = True,
        mask_thresh = args.mask_thresh,
    )
    val_ds = SingleTemporalCloudDataset(
        root_dir    = args.val_dir,
        patch_size  = args.patch_size_test,
        augment     = False,
        mask_thresh = args.mask_thresh,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=args.num_workers,
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=2,
                              pin_memory=True, drop_last=False)

    # ── TensorBoard ───────────────────────────────────────────────────────────
    writer = SummaryWriter(model_dir)

    print(f"[Train] epochs={args.num_epochs}  batch={args.batch_size}  "
          f"lr={args.lr}  λ_cloud={args.lambda_cloud}")
    print(f"[Train] train={len(train_ds)}  val={len(val_ds)}")
    print('=' * 80)

    best_psnr  = 0.0
    best_epoch = 0
    step_iter  = 0
    start_epoch = 1

    for epoch in range(start_epoch, args.num_epochs + 1):
        t0          = time.time()
        net.train()
        epoch_loss  = 0.0
        train_psnrs = []

        for batch_idx, (inp_img, target, cloud_mask) in enumerate(
                tqdm(train_loader, desc=f'Epoch {epoch}', unit='batch')):

            inp_img    = inp_img.cuda()      # [B, C, H, W]
            target     = target.cuda()       # [B, C, H, W]
            cloud_mask = cloud_mask.cuda()   # [B, 1, H, W]

            for p in net.parameters():
                p.grad = None

            # Forward
            restored, pred_cloud_map = net(inp_img)

            # cloud_maps expected as [L, B, 1, H, W] by CloudRemovalLoss;
            # for single-temporal: L=1
            pred_maps_loss = pred_cloud_map.unsqueeze(0)   # [1, B, 1, H, W]
            gt_masks_loss  = cloud_mask.unsqueeze(0)       # [1, B, 1, H, W]
            gt_masks_for_loss = gt_masks_loss if train_ds._has_masks else None

            loss, log_dict = criterion(
                restored, target,
                pred_maps_loss,
                gt_masks_for_loss,
            )

            loss.backward()
            if args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip_grad)
            optimizer.step()
            functional.reset_net(net)

            with torch.no_grad():
                p_val = psnr(restored.clamp(0, 1), target)
            train_psnrs.append(p_val.item())
            epoch_loss += loss.item()

            for k, v in log_dict.items():
                writer.add_scalar(f'train/{k}', v, step_iter)
            step_iter += 1

        train_psnr_mean = np.mean(train_psnrs)
        writer.add_scalar('train/psnr',       train_psnr_mean, epoch)
        writer.add_scalar('train/epoch_loss', epoch_loss / len(train_loader), epoch)
        writer.add_scalar('lr', scheduler.get_lr()[0], epoch)

        if epoch % 10 == 0:
            with torch.no_grad():
                sample_path = os.path.join(
                    sample_dir, 'train', f'epoch_{epoch}.png')
                save_comparison(inp_img, target, restored.clamp(0, 1),
                                pred_cloud_map, sample_path)

        # ── Validation ────────────────────────────────────────────────────
        if epoch % args.val_epochs == 0:
            net.eval()
            val_psnrs   = []
            val_ssims   = []
            val_cpsnrs  = []
            val_sample_saved = False

            for inp_v, target_v, mask_v in tqdm(
                    val_loader, desc='  Val', leave=False):
                inp_v    = inp_v.cuda()
                target_v = target_v.cuda()
                mask_v   = mask_v.cuda()

                with torch.no_grad():
                    restored_v, pred_map_v = net(inp_v)
                functional.reset_net(net)

                restored_v = restored_v.clamp(0, 1)
                val_psnrs.append(psnr(restored_v, target_v).item())
                val_ssims.append(ssim_metric(restored_v, target_v).item())
                val_cpsnrs.append(cpsnr(restored_v, target_v, mask_v).item())

                if not val_sample_saved:
                    save_comparison(
                        inp_v, target_v, restored_v, pred_map_v,
                        os.path.join(sample_dir, 'val', f'epoch_{epoch}.png'))
                    val_sample_saved = True

            val_psnr_mean  = np.mean(val_psnrs)
            val_ssim_mean  = np.mean(val_ssims)
            val_cpsnr_mean = np.mean(val_cpsnrs)

            writer.add_scalar('val/psnr',  val_psnr_mean,  epoch)
            writer.add_scalar('val/ssim',  val_ssim_mean,  epoch)
            writer.add_scalar('val/cpsnr', val_cpsnr_mean, epoch)

            if val_psnr_mean > best_psnr:
                best_psnr  = val_psnr_mean
                best_epoch = epoch
                torch.save(net.state_dict(),
                           os.path.join(model_dir, 'model_best.pth'))

            print(f"  Val PSNR={val_psnr_mean:.3f}  "
                  f"SSIM={val_ssim_mean:.4f}  "
                  f"cPSNR={val_cpsnr_mean:.3f}  "
                  f"(best={best_psnr:.3f} @ ep{best_epoch})")

        # ── Checkpoint ────────────────────────────────────────────────────
        if epoch % 50 == 0:
            torch.save({'epoch': epoch, 'state_dict': net.state_dict(),
                        'optimizer': optimizer.state_dict()},
                       os.path.join(model_dir, f'model_epoch_{epoch}.pth'))

        torch.save({'epoch': epoch, 'state_dict': net.state_dict(),
                    'optimizer': optimizer.state_dict()},
                   os.path.join(model_dir, 'model_last.pth'))

        scheduler.step()
        elapsed = time.time() - t0
        print(f"Epoch {epoch:04d} | loss={epoch_loss/len(train_loader):.4f} | "
              f"psnr={train_psnr_mean:.3f} | lr={scheduler.get_lr()[0]:.2e} | "
              f"t={elapsed:.1f}s")
        print('-' * 80)

    writer.close()
    print(f"\nTraining complete. Best val PSNR = {best_psnr:.3f} at epoch {best_epoch}")
