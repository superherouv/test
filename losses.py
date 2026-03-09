import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import functools


# ─────────────────────────────────────────────────────────────────────────────
# CloudRemovalLoss
# ─────────────────────────────────────────────────────────────────────────────
class CloudRemovalLoss(nn.Module):
    """
    Combined loss for multi-temporal Kalman-SNN cloud removal.

    Components
    ----------
    L_rec   : SSIM + L1  reconstruction loss on the final output.
    L_cloud : Binary-cross-entropy supervision for the cloud estimation head.
              Only applied when ground-truth cloud masks are available.
              λ_cloud weights this term (default 0.1).
    L_tmc   : Temporal Model Calibration loss (ICML 2025 style).
              Encourages gradient diversity across L temporal steps by
              penalising early steps more when cumulative confidence is low.
              λ_tmc weights this term (default 0.0 → off by default).

    Usage
    -----
        criterion = CloudRemovalLoss(lambda_cloud=0.1, lambda_tmc=0.0)
        loss, log = criterion(restored, target, cloud_maps, gt_masks)
        loss.backward()

    Parameters
    ----------
    lambda_cloud : float  weight of cloud detection BCE loss (0 to disable)
    lambda_tmc   : float  weight of TMC temporal calibration  (0 to disable)
    ssim_win     : int    SSIM window size (default 11)
    """

    def __init__(self, lambda_cloud: float = 0.1,
                 lambda_tmc: float = 0.0,
                 ssim_win: int = 11):
        super().__init__()
        self.lambda_cloud = lambda_cloud
        self.lambda_tmc   = lambda_tmc
        self.ssim_win     = ssim_win
        self.bce          = nn.BCELoss()

    # ── SSIM helper ──────────────────────────────────────────────────────────
    @staticmethod
    def _ssim(x: torch.Tensor, y: torch.Tensor, win: int = 11) -> torch.Tensor:
        """Mean SSIM over a batch; returns scalar."""
        C1, C2 = 0.01**2, 0.03**2
        pad    = win // 2
        ch     = x.shape[1]

        kernel = torch.ones(ch, 1, win, win, device=x.device,
                            dtype=x.dtype) / (win * win)
        mu_x   = F.conv2d(x, kernel, padding=pad, groups=ch)
        mu_y   = F.conv2d(y, kernel, padding=pad, groups=ch)
        mu_x2  = mu_x * mu_x
        mu_y2  = mu_y * mu_y
        mu_xy  = mu_x * mu_y
        sig_x2 = F.conv2d(x * x, kernel, padding=pad, groups=ch) - mu_x2
        sig_y2 = F.conv2d(y * y, kernel, padding=pad, groups=ch) - mu_y2
        sig_xy = F.conv2d(x * y, kernel, padding=pad, groups=ch) - mu_xy

        num = (2 * mu_xy + C1) * (2 * sig_xy + C2)
        den = (mu_x2 + mu_y2 + C1) * (sig_x2 + sig_y2 + C2)
        return (num / den).mean()

    # ── TMC temporal calibration ─────────────────────────────────────────────
    @staticmethod
    def _tmc_loss(per_step_outputs: list, target: torch.Tensor,
                  base_criterion) -> torch.Tensor:
        """
        Temporal Model Calibration loss (adapted for regression / image restoration).

        per_step_outputs : list of L tensors, each [B, C, H, W]
                           intermediate reconstructions (one per temporal step)
        target           : [B, C, H, W]  ground-truth cloud-free image
        base_criterion   : callable, e.g. F.l1_loss

        Formulation (adapted from ICML 2025 TMC for regression)
        --------------------------------------------------------
        For step l = 0 … L-1:
            β_l  = 1 − SSIM(cumulative_mean_l.detach(), target)  ∈ (0,1)
                   (1 − SSIM used so β≈0 means good reconstruction)
            θ_l  = β_l / (1 − β_l + ε)   (confidence ratio, ∞ if perfect)
            w_l  = (L − l) / L            (temporal exponent: earlier = larger)
            L_l  = base_loss(step_l, target) + θ_l^w_l

        Total = mean over L steps.
        """
        L     = len(per_step_outputs)
        total = torch.tensor(0.0, device=target.device)

        _cumsum = torch.zeros_like(per_step_outputs[0])

        for l, out_l in enumerate(per_step_outputs):
            # Cumulative mean up to step l (history detached, current step live)
            _cumsum_detached = _cumsum.detach()
            cum_mean = (_cumsum_detached + out_l) / (l + 1)

            # SSIM-based quality proxy (stop-gradient on history)
            with torch.no_grad():
                C1, C2  = 0.01**2, 0.03**2
                mu_p    = cum_mean.detach().mean(dim=[2, 3], keepdim=True)
                mu_t    = target.mean(dim=[2, 3], keepdim=True)
                beta_l  = 1.0 - ((2 * mu_p * mu_t + C1) /
                                  (mu_p**2 + mu_t**2 + C1)).mean()
                beta_l  = beta_l.clamp(1e-4, 1 - 1e-4)

            theta_l = beta_l / (1.0 - beta_l + 1e-6)
            w_l     = (L - l) / L                            # temporal weight

            step_loss = base_criterion(out_l, target)
            total    += step_loss + (theta_l ** w_l)

            # Advance cumsum (all detached for next iteration)
            _cumsum = _cumsum.detach() + out_l.detach()

        return total / L

    # ── Main forward ─────────────────────────────────────────────────────────
    def forward(
        self,
        restored:    torch.Tensor,          # [B, C, H, W]  final output
        target:      torch.Tensor,          # [B, C, H, W]  ground-truth
        cloud_maps:  torch.Tensor,          # [L, B, 1, H, W]  estimated cloud prob
        gt_masks:    torch.Tensor = None,   # [L, B, 1, H, W]  GT masks (0=cloud)
                                            #   or None if unavailable
        per_step_outputs: list = None,      # list[L] of [B,C,H,W] for TMC
    ):
        """
        Returns
        -------
        total_loss : scalar tensor (call .backward() on this)
        log        : dict  {component_name: float}  for TensorBoard logging
        """
        log = {}

        # ── Reconstruction loss (SSIM + L1) ──────────────────────────────
        ssim_val  = self._ssim(restored, target, self.ssim_win)
        l1_val    = F.l1_loss(restored, target)
        L_rec     = (1.0 - ssim_val) + 0.1 * l1_val
        log['L_rec']  = L_rec.item()
        log['ssim']   = ssim_val.item()
        log['l1']     = l1_val.item()

        total = L_rec

        # ── Cloud detection BCE ───────────────────────────────────────────
        if self.lambda_cloud > 0 and gt_masks is not None:
            # gt_masks: [L, B, 1, H, W], 1 = cloud, 0 = clear
            # cloud_maps (predicted): [L, B, 1, H, W]  ∈ (0,1)
            L_cloud = self.bce(cloud_maps, gt_masks.float())
            log['L_cloud'] = L_cloud.item()
            total = total + self.lambda_cloud * L_cloud
        else:
            log['L_cloud'] = 0.0

        # ── TMC temporal calibration ──────────────────────────────────────
        if self.lambda_tmc > 0 and per_step_outputs is not None:
            L_tmc = self._tmc_loss(per_step_outputs, target, F.l1_loss)
            log['L_tmc'] = L_tmc.item()
            total = total + self.lambda_tmc * L_tmc
        else:
            log['L_tmc'] = 0.0

        log['total'] = total.item()
        return total, log

_reduction_modes = ['none', 'mean', 'sum']

def reduce_loss(loss, reduction):
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are 'none', 'mean' and 'sum'.

    Returns:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    else:
        return loss.sum()

def weight_reduce_loss(loss, weight=None, reduction='mean'):
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights. Default: None.
        reduction (str): Same as built-in losses of PyTorch. Options are
            'none', 'mean' and 'sum'. Default: 'mean'.

    Returns:
        Tensor: Loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        assert weight.dim() == loss.dim()
        assert weight.size(1) == 1 or weight.size(1) == loss.size(1)
        loss = loss * weight

    # if weight is not specified or reduction is sum, just reduce the loss
    if weight is None or reduction == 'sum':
        loss = reduce_loss(loss, reduction)
    # if reduction is mean, then compute mean over weight region
    elif reduction == 'mean':
        if weight.size(1) > 1:
            weight = weight.sum()
        else:
            weight = weight.sum() * loss.size(1)
        loss = loss.sum() / weight

    return loss

def weighted_loss(loss_func):
    """Create a weighted version of a given loss function.

    To use this decorator, the loss function must have the signature like
    `loss_func(pred, target, **kwargs)`. The function only needs to compute
    element-wise loss without any reduction. This decorator will add weight
    and reduction arguments to the function. The decorated function will have
    the signature like `loss_func(pred, target, weight=None, reduction='mean',
    **kwargs)`.

    :Example:

    >>> import torch
    >>> @weighted_loss
    >>> def l1_loss(pred, target):
    >>>     return (pred - target).abs()

    >>> pred = torch.Tensor([0, 2, 3])
    >>> target = torch.Tensor([1, 1, 1])
    >>> weight = torch.Tensor([1, 0, 1])

    >>> l1_loss(pred, target)
    tensor(1.3333)
    >>> l1_loss(pred, target, weight)
    tensor(1.5000)
    >>> l1_loss(pred, target, reduction='none')
    tensor([1., 1., 2.])
    >>> l1_loss(pred, target, weight, reduction='sum')
    tensor(3.)
    """

    @functools.wraps(loss_func)
    def wrapper(pred, target, weight=None, reduction='mean', **kwargs):
        # get element-wise loss
        loss = loss_func(pred, target, **kwargs)
        loss = weight_reduce_loss(loss, weight, reduction)
        return loss

    return wrapper

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x.to('cuda:0') - y.to('cuda:0')
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss

class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(),k).unsqueeze(0).repeat(3,1,1,1)
        if torch.cuda.is_available():
            self.kernel = self.kernel.to('cuda:0')
        self.loss = CharbonnierLoss()

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered    = self.conv_gauss(current)
        down        = filtered[:,:,::2,::2]
        new_filter  = torch.zeros_like(filtered)
        new_filter[:,:,::2,::2] = down*4
        filtered    = self.conv_gauss(new_filter)
        diff = current - filtered
        return diff

    def forward(self, x, y):
        loss = self.loss(self.laplacian_kernel(x.to('cuda:0')), self.laplacian_kernel(y.to('cuda:0')))
        return loss

class fftLoss(nn.Module):
    def __init__(self):
        super(fftLoss, self).__init__()

    def forward(self, x, y):
        diff = torch.fft.fft2(x.to('cuda:0')) - torch.fft.fft2(y.to('cuda:0'))
        loss = torch.mean(abs(diff))
        return loss

@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')

@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')

class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * l1_loss(
            pred, target, weight, reduction=self.reduction)

class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * mse_loss(
            pred, target, weight, reduction=self.reduction)

class PSNRLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4

        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()
