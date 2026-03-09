"""
Training script for multi-temporal cloud removal with Kalman-SNN.

Key differences from train.py
------------------------------
  - Input is [L, B, C, H, W] per batch (L temporal images stacked on dim 0)
  - Model returns (restored, cloud_maps) tuple
  - Loss = CloudRemovalLoss (SSIM+L1 + optional cloud-BCE + optional TMC)
  - Metrics: PSNR on cloud-covered regions (cPSNR) in addition to full PSNR

Visualizations (saved to checkpoints/<session>/samples/ and TensorBoard)
------------------------------------------------------------------------
  - Rich matplotlib grids: all L cloudy frames | target | restored |
    jet error-map | inferno cloud-maps  (saved every 10 epochs, train+val)
  - TensorBoard image panels: individual named channels per split
  - TensorBoard scalars: loss components, PSNR, cPSNR, grad-norm,
    cloud-coverage, LR
  - TensorBoard histogram: val cloud-prediction distribution per epoch
  - PNG metric-history curves: multi-panel chart saved to samples/curves/
  - PNG cloud-coverage area chart saved to samples/curves/

Example command
---------------
    python train_cloud.py \\
        --train_dir  /data/Sen2MTC/train \\
        --val_dir    /data/Sen2MTC/val \\
        --layout     Sen2MTC \\
        --L          3 \\
        --inp_channels 3 \\
        --patch_size_train 256 \\
        --patch_size_test  256 \\
        --batch_size 4 \\
        --num_epochs 200 \\
        --lr         2e-4 \\
        --lambda_cloud 0.1 \\
        --session    kalman_snn_L3
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

from spikingjelly.activation_based import functional

import utils
import viz_utils
from model      import model_cloud
from losses     import CloudRemovalLoss
from warmup_scheduler import GradualWarmupScheduler
from dataset_cloud    import MultiTemporalCloudDataset


# ─── helpers ──────────────────────────────────────────────────────────────────
def psnr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = ((pred - target) ** 2).mean()
    return 10 * torch.log10(1.0 / (mse + 1e-8))


def cpsnr(pred: torch.Tensor, target: torch.Tensor,
          cloud_masks: torch.Tensor) -> torch.Tensor:
    """
    Cloud-region PSNR: only evaluate pixels where ANY temporal mask = cloud.

    pred, target : [B, C, H, W]
    cloud_masks  : [L, B, 1, H, W]  (1 = cloud)
    """
    cloud_union = cloud_masks.max(dim=0).values   # [B, 1, H, W]

    if cloud_union.sum() < 1:
        return psnr(pred, target)

    err       = (pred - target) ** 2
    cloud_mse = (err * cloud_union).sum() / (cloud_union.sum() * pred.shape[1] + 1e-8)
    return 10 * torch.log10(1.0 / (cloud_mse + 1e-8))


def _build_tb_images(cloudy_seq, target, restored, pred_cloud_maps, n=4):
    """Build a dict of named image tensors for TensorBoard logging."""
    L = cloudy_seq.shape[0]
    n = min(n, target.shape[0])
    imgs = {}
    for l in range(L):
        imgs[f'cloudy_T{l+1}'] = cloudy_seq[l, :n]
    imgs['target']   = target[:n]
    imgs['restored'] = restored[:n].clamp(0, 1)
    imgs['error_map'] = torch.stack([
        viz_utils.error_heatmap(restored[i].clamp(0, 1), target[i])
        for i in range(n)
    ])
    for l in range(L):
        imgs[f'cloud_map_T{l+1}'] = torch.stack([
            viz_utils.cloud_heatmap(pred_cloud_maps[l, i])
            for i in range(n)
        ])
    return imgs


# ─── main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # ── Seed ──────────────────────────────────────────────────────────────────
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)

    # ── Args ──────────────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(description='Kalman-SNN Multi-temporal Cloud Removal')

    # Paths
    parser.add_argument('--train_dir',        default='', type=str)
    parser.add_argument('--val_dir',          default='', type=str)
    parser.add_argument('--model_save_dir',   default='./checkpoints/', type=str)
    parser.add_argument('--pretrain_weights', default='', type=str)
    parser.add_argument('--session',          default='KalmanSNN_cloud', type=str)

    # Data
    parser.add_argument('--layout',           default='Sen2MTC', type=str,
                        choices=['Sen2MTC', 'flat'],
                        help='Dataset folder layout (Sen2MTC or flat)')
    parser.add_argument('--L',                default=3, type=int,
                        help='Number of temporal images per sample')
    parser.add_argument('--inp_channels',     default=3, type=int,
                        help='Input channels per image (3=RGB, 4=RGB+NIR)')
    parser.add_argument('--patch_size_train', default=256, type=int)
    parser.add_argument('--patch_size_test',  default=256, type=int)
    parser.add_argument('--num_workers',      default=4, type=int)

    # Training
    parser.add_argument('--num_epochs',       default=200, type=int)
    parser.add_argument('--batch_size',       default=4, type=int)
    parser.add_argument('--lr',               default=2e-4, type=float)
    parser.add_argument('--min_lr',           default=1e-7, type=float)
    parser.add_argument('--warmup_epochs',    default=5, type=int)
    parser.add_argument('--clip_grad',        default=1.0, type=float)
    parser.add_argument('--val_epochs',       default=5, type=int)

    # Loss weights
    parser.add_argument('--lambda_cloud',     default=0.1, type=float,
                        help='Weight of cloud-detection BCE loss')
    parser.add_argument('--lambda_tmc',       default=0.0, type=float,
                        help='Weight of TMC temporal calibration loss')

    # Misc
    parser.add_argument('--use_refinement',   default=False, type=bool)
    args = parser.parse_args()

    # ── Directories ───────────────────────────────────────────────────────────
    model_dir  = os.path.join(args.model_save_dir, 'KalmanSNN_cloud', 'models', args.session)
    sample_dir = os.path.join(args.model_save_dir, 'KalmanSNN_cloud', 'samples', args.session)
    curves_dir = os.path.join(sample_dir, 'curves')
    utils.mkdir(model_dir)
    utils.mkdir(sample_dir)
    utils.mkdir(os.path.join(sample_dir, 'train'))
    utils.mkdir(os.path.join(sample_dir, 'val'))
    utils.mkdir(curves_dir)

    # ── Model ─────────────────────────────────────────────────────────────────
    net = model_cloud(L=args.L, inp_channels=args.inp_channels,
                      use_refinement=args.use_refinement)
    net.cuda()
    functional.set_step_mode(net, step_mode='m')
    functional.set_backend(net, backend='cupy')

    device_ids = list(range(torch.cuda.device_count()))
    print(f"[Init] GPUs: {device_ids}  |  L={args.L}  |  layout={args.layout}")

    # Optional pretrain
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
                                 lambda_tmc=args.lambda_tmc).cuda()

    # ── Data ──────────────────────────────────────────────────────────────────
    train_ds = MultiTemporalCloudDataset(
        root_dir   = args.train_dir,
        L          = args.L,
        patch_size = args.patch_size_train,
        layout     = args.layout,
        augment    = True,
    )
    val_ds = MultiTemporalCloudDataset(
        root_dir   = args.val_dir,
        L          = args.L,
        patch_size = args.patch_size_test,
        layout     = args.layout,
        augment    = False,
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
          f"lr={args.lr}  λ_cloud={args.lambda_cloud}  λ_tmc={args.lambda_tmc}")
    print(f"[Train] train={len(train_ds)}  val={len(val_ds)}")
    print('=' * 80)

    best_psnr  = 0.0
    best_epoch = 0
    step_iter  = 0
    start_epoch = 1

    # ── Metric history for curve plots ────────────────────────────────────────
    history = {
        'Train PSNR':     [],
        'Val PSNR':       [],
        'Val cPSNR':      [],
        'Epoch Loss':     [],
        'Grad Norm':      [],
        'LR':             [],
        'Cloud Coverage': [],
    }

    for epoch in range(start_epoch, args.num_epochs + 1):
        t0                    = time.time()
        net.train()
        epoch_loss            = 0.0
        train_psnrs           = []
        epoch_grad_norms      = []
        epoch_cloud_coverages = []

        for batch_idx, (cloudy_seq, target, cloud_masks) in enumerate(
                tqdm(train_loader, desc=f'Epoch {epoch}', unit='batch')):

            cloudy_seq  = cloudy_seq.cuda().permute(1, 0, 2, 3, 4)  # [L,B,C,H,W]
            target      = target.cuda()                               # [B, C, H, W]
            cloud_masks = cloud_masks.cuda().permute(1, 0, 2, 3, 4) # [L,B,1,H,W]

            for p in net.parameters():
                p.grad = None

            # Forward
            restored, pred_cloud_maps = net(cloudy_seq)

            # Loss
            gt_masks_for_loss = cloud_masks if train_ds._has_masks else None
            loss, log_dict    = criterion(
                restored, target,
                pred_cloud_maps,
                gt_masks_for_loss,
            )

            loss.backward()

            # Gradient clipping + track norm
            if args.clip_grad > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    net.parameters(), args.clip_grad).item()
            else:
                grad_norm = sum(
                    p.grad.data.norm(2).item() ** 2
                    for p in net.parameters() if p.grad is not None
                ) ** 0.5
            optimizer.step()
            functional.reset_net(net)

            # Per-batch metrics
            with torch.no_grad():
                p_val = psnr(restored.clamp(0, 1), target)
            train_psnrs.append(p_val.item())
            epoch_loss += loss.item()
            epoch_grad_norms.append(grad_norm)

            # Cloud coverage ratio
            cloud_cov = cloud_masks.float().mean().item()
            epoch_cloud_coverages.append(cloud_cov)

            # TensorBoard per-step scalars
            for k, v in log_dict.items():
                writer.add_scalar(f'train/{k}', v, step_iter)
            writer.add_scalar('train/grad_norm',       grad_norm, step_iter)
            writer.add_scalar('train/cloud_coverage',  cloud_cov, step_iter)
            step_iter += 1

        # ── Per-epoch scalars ──────────────────────────────────────────────
        train_psnr_mean   = np.mean(train_psnrs)
        grad_norm_mean    = np.mean(epoch_grad_norms)
        cloud_cov_mean    = np.mean(epoch_cloud_coverages)
        epoch_loss_mean   = epoch_loss / len(train_loader)

        writer.add_scalar('train/psnr',            train_psnr_mean, epoch)
        writer.add_scalar('train/epoch_loss',      epoch_loss_mean, epoch)
        writer.add_scalar('train/grad_norm_epoch', grad_norm_mean,  epoch)
        writer.add_scalar('train/cloud_cov_epoch', cloud_cov_mean,  epoch)
        writer.add_scalar('lr', scheduler.get_lr()[0], epoch)

        history['Train PSNR'].append((epoch, train_psnr_mean))
        history['Epoch Loss'].append((epoch, epoch_loss_mean))
        history['Grad Norm'].append((epoch, grad_norm_mean))
        history['LR'].append((epoch, scheduler.get_lr()[0]))
        history['Cloud Coverage'].append((epoch, cloud_cov_mean))

        # ── Training sample visualization (every 10 epochs) ───────────────
        if epoch % 10 == 0:
            with torch.no_grad():
                # Rich matplotlib comparison grid
                sample_path = os.path.join(
                    sample_dir, 'train', f'epoch_{epoch}.png')
                viz_utils.save_rich_comparison_multi(
                    cloudy_seq, target, restored.clamp(0, 1),
                    pred_cloud_maps, sample_path)

                # TensorBoard image panels
                tb_imgs = _build_tb_images(
                    cloudy_seq, target, restored, pred_cloud_maps)
                viz_utils.log_images_tensorboard(
                    writer, 'train_images', tb_imgs, epoch)

        # ── Validation ────────────────────────────────────────────────────
        if epoch % args.val_epochs == 0:
            net.eval()
            val_psnrs        = []
            val_cpsnrs       = []
            val_sample_saved = False
            all_cloud_preds  = []

            for cloudy_seq_v, target_v, masks_v in tqdm(
                    val_loader, desc='  Val', leave=False):
                cloudy_seq_v = cloudy_seq_v.cuda().permute(1, 0, 2, 3, 4)
                target_v     = target_v.cuda()
                masks_v      = masks_v.cuda().permute(1, 0, 2, 3, 4)

                with torch.no_grad():
                    restored_v, pred_maps_v = net(cloudy_seq_v)
                functional.reset_net(net)

                restored_v = restored_v.clamp(0, 1)
                val_psnrs.append(psnr(restored_v, target_v).item())
                val_cpsnrs.append(cpsnr(restored_v, target_v, masks_v).item())
                all_cloud_preds.append(pred_maps_v.detach().cpu())

                if not val_sample_saved:
                    # Rich matplotlib grid
                    viz_utils.save_rich_comparison_multi(
                        cloudy_seq_v, target_v, restored_v, pred_maps_v,
                        os.path.join(sample_dir, 'val', f'epoch_{epoch}.png'))
                    # TensorBoard image panels
                    tb_imgs_v = _build_tb_images(
                        cloudy_seq_v, target_v, restored_v, pred_maps_v)
                    viz_utils.log_images_tensorboard(
                        writer, 'val_images', tb_imgs_v, epoch)
                    val_sample_saved = True

            val_psnr_mean  = np.mean(val_psnrs)
            val_cpsnr_mean = np.mean(val_cpsnrs)
            writer.add_scalar('val/psnr',  val_psnr_mean,  epoch)
            writer.add_scalar('val/cpsnr', val_cpsnr_mean, epoch)

            # Cloud prediction distribution histogram
            flat_preds = torch.cat(
                [x.flatten() for x in all_cloud_preds])
            writer.add_histogram('val/cloud_pred_distribution', flat_preds, epoch)

            history['Val PSNR'].append((epoch, val_psnr_mean))
            history['Val cPSNR'].append((epoch, val_cpsnr_mean))

            if val_psnr_mean > best_psnr:
                best_psnr  = val_psnr_mean
                best_epoch = epoch
                torch.save(net.state_dict(),
                           os.path.join(model_dir, 'model_best.pth'))

            print(f"  Val PSNR={val_psnr_mean:.3f}  "
                  f"cPSNR={val_cpsnr_mean:.3f}  "
                  f"(best={best_psnr:.3f} @ ep{best_epoch})")

            # ── Save metric-history charts ─────────────────────────────────
            viz_utils.save_metric_curves(
                history,
                os.path.join(curves_dir, 'metrics.png'),
                title=f'Kalman-SNN Training Metrics  (session={args.session})')
            viz_utils.save_cloud_coverage_plot(
                history['Cloud Coverage'],
                os.path.join(curves_dir, 'cloud_coverage.png'))

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
        print(f"Epoch {epoch:04d} | loss={epoch_loss_mean:.4f} | "
              f"psnr={train_psnr_mean:.3f} | grad={grad_norm_mean:.3f} | "
              f"cloud={cloud_cov_mean:.2%} | lr={scheduler.get_lr()[0]:.2e} | "
              f"t={elapsed:.1f}s")
        print('-' * 80)

    # ── Final metric curves ────────────────────────────────────────────────────
    viz_utils.save_metric_curves(
        history,
        os.path.join(curves_dir, 'metrics_final.png'),
        title=f'Kalman-SNN Training Metrics — Final  (session={args.session})')
    viz_utils.save_cloud_coverage_plot(
        history['Cloud Coverage'],
        os.path.join(curves_dir, 'cloud_coverage_final.png'))

    writer.close()
    print(f"\nTraining complete. Best val PSNR = {best_psnr:.3f} at epoch {best_epoch}")
