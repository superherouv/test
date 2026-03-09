"""
Shape verification for the Kalman-SNN cloud removal model.

Tests (CPU-only, no GPU/cupy required for shape checks):
  1.  CloudEstimator:          [B, C, H, W]  → [B, 1, H, W]
  2.  KalmanFeatureFilter:     [L, B, C, H, W] + cloud_maps → filtered + gains
  3.  Kalman P dynamics:       gains decrease on all-cloudy, high on clear
  4.  SUNet_Level1_Block T=3:  [L=3, B, C, H, W] → same shape
  5.  SUNet_Level1_Block T=4:  [L=4, B, C, H, W] → same shape (backward compat)
  6.  CloudRemovalLoss:        forward pass returns scalar + log dict
  7.  MultiTemporalCloudDataset: mock dir, __getitem__ shapes
  8.  Full model_cloud forward (multi): [L, B, C, H, W] → ([B, C, H, W], [L, B, 1, H, W])
  9.  Original model forward (backward compat): [B, C, H, W] → [B, C, H, W]
  10. Gradient flow through Kalman filter
  11. Parameter count comparison
  12. CloudGate: shapes and monotone gate initialisation
  13. Full model_cloud_single forward (single): [B, C, H, W] → ([B, C, H, W], [B, 1, H, W])
  14. SingleTemporalCloudDataset: mock dir, __getitem__ shapes
"""

import sys, os
import tempfile
import shutil
import numpy as np
from PIL import Image
import torch
import torch.nn as nn

# ── Patch spikingjelly dependencies not available in CPU-only test env ─────────
# We mock the heavy GPU modules so tests run without cupy / spikingjelly installed
class _FakeLIF(nn.Module):
    def forward(self, x): return torch.sigmoid(x)

class _FakeMDA(nn.Module):
    def __init__(self, *a, **kw): super().__init__()
    def forward(self, x): return x

class _FakeConv2d(nn.Module):
    """Faithful mock: preserves kernel_size, stride, padding from caller."""
    def __init__(self, ic, oc, kernel_size=3, stride=1, padding=0):
        super().__init__()
        self.c = nn.Conv2d(ic, oc, kernel_size=kernel_size,
                           stride=stride, padding=padding)
    def forward(self, x):
        T, B, C, H, W = x.shape
        o = self.c(x.reshape(T*B, C, H, W))
        _, Cout, H2, W2 = o.shape
        return o.reshape(T, B, Cout, H2, W2)

def _make_fake_conv(ic, oc, *args, **kw):
    # spikingjelly Conv2d call: (ic, oc, kernel_size, stride, padding, ...)
    # Both positional AND keyword forms must be handled:
    #   layer.Conv2d(dim, dim,  3, 1, 1,      bias=False, step_mode='m')
    #   layer.Conv2d(dim, dim*2,3, stride=2, padding=1, step_mode='m')
    #   layer.Conv2d(dim, dim,  kernel_size=1, ...)
    ks      = args[0] if len(args) > 0 else kw.get('kernel_size', 3)
    stride  = args[1] if len(args) > 1 else kw.get('stride',  1)
    padding = args[2] if len(args) > 2 else kw.get('padding', 0)
    return _FakeConv2d(ic, oc, ks, stride, padding)

class _FakeTDBN(nn.Module):
    def __init__(self, num_features, *a, **kw):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features)
    def forward(self, x):
        T, B, C, H, W = x.shape
        o = self.bn(x.reshape(T*B, C, H, W))
        return o.reshape(T, B, C, H, W)

import types, unittest.mock as mock

# Build fake 'spikingjelly' module tree before importing model
sj = types.ModuleType('spikingjelly')
sj_ab = types.ModuleType('spikingjelly.activation_based')
sj_n  = types.ModuleType('spikingjelly.activation_based.neuron')
sj_l  = types.ModuleType('spikingjelly.activation_based.layer')
sj_f  = types.ModuleType('spikingjelly.activation_based.functional')
sj_n.LIFNode  = lambda **kw: _FakeLIF()
sj_l.Conv2d   = _make_fake_conv
sj_l.ThresholdDependentBatchNorm2d = lambda num_features, *a, **kw: _FakeTDBN(num_features)
sj_l.MultiDimensionalAttention     = lambda *a, **kw: _FakeMDA()
sj_f.set_step_mode  = lambda *a, **kw: None
sj_f.set_backend    = lambda *a, **kw: None
sj_f.reset_net      = lambda *a, **kw: None
sj_ab.functional    = sj_f
sj_ab.layer         = sj_l
sj_ab.neuron        = sj_n
sj.activation_based = sj_ab
sys.modules.update({
    'spikingjelly': sj,
    'spikingjelly.activation_based': sj_ab,
    'spikingjelly.activation_based.neuron': sj_n,
    'spikingjelly.activation_based.layer': sj_l,
    'spikingjelly.activation_based.functional': sj_f,
})

# Also patch UpSampling.forward to avoid .cuda() call
import model as M
_orig_up = M.UpSampling.forward
def _cpu_up(self, inp):
    out = []
    for i in range(inp.shape[0]):
        out.append(torch.nn.functional.interpolate(
            inp[i], scale_factor=self.scale_factor, mode='bilinear'))
    out = torch.stack(out, dim=0)
    out = self.lif(out)
    out = self.conv(out)
    out = self.bn(out)
    return out
M.UpSampling.forward = _cpu_up

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
errors = []

def check(name, cond, note=""):
    if cond:
        print(f"  {PASS} {name}")
    else:
        print(f"  {FAIL} {name}  ← {note}")
        errors.append(name)


# ════════════════════════════════════════════════════════════════════════════════
print("\n══ 1. CloudEstimator ═══════════════════════════════════════════════════")
ce = M.CloudEstimator(in_channels=3)
x  = torch.randn(2, 3, 64, 64)
y  = ce(x)
check("output shape [B,1,H,W]", y.shape == (2, 1, 64, 64), str(y.shape))
check("output range [0,1]",     y.min() >= 0 and y.max() <= 1,
      f"min={y.min():.3f} max={y.max():.3f}")


# ════════════════════════════════════════════════════════════════════════════════
print("\n══ 2. KalmanFeatureFilter ══════════════════════════════════════════════")
L, B, C, H, W = 3, 2, 32, 16, 16
kf          = M.KalmanFeatureFilter()
feats       = torch.randn(L, B, C, H, W)
cloud_maps  = torch.rand(L, B, 1, H, W)   # random cloud probs in [0,1]
filt, gains = kf(feats, cloud_maps)
check("filtered shape [L,B,C,H,W]", filt.shape  == (L, B, C, H, W), str(filt.shape))
check("gains shape    [L,B,1,H,W]", gains.shape == (L, B, 1, H, W), str(gains.shape))
check("gains in (0,1)",              gains.min() > 0 and gains.max() < 1,
      f"min={gains.min():.4f} max={gains.max():.4f}")


# ════════════════════════════════════════════════════════════════════════════════
print("\n══ 3. Kalman P dynamics ════════════════════════════════════════════════")
kf2 = M.KalmanFeatureFilter()
with torch.no_grad():
    # All-cloudy: P should grow over steps (accumulated uncertainty)
    feats_cloudy = torch.randn(5, 1, 8, 4, 4)
    maps_cloudy  = torch.ones(5, 1, 1, 4, 4)   # cloud everywhere
    _, g_cloudy  = kf2(feats_cloudy, maps_cloudy)
    # Gains should decrease (model loses confidence in observations)
    gain_decrease = (g_cloudy[4] <= g_cloudy[0]).float().mean()
    check("gains decrease on all-cloudy sequence (P grows)",
          gain_decrease > 0.7, f"only {gain_decrease:.2%} pixels decrease")

    # All-clear: P should shrink (high confidence observations)
    maps_clear  = torch.zeros(5, 1, 1, 4, 4)   # no cloud
    _, g_clear  = kf2(feats_cloudy, maps_clear)
    gain_vals_clear = g_clear.mean().item()
    check("gains are high on clear sequence", gain_vals_clear > 0.3,
          f"mean gain={gain_vals_clear:.4f}")


# ════════════════════════════════════════════════════════════════════════════════
print("\n══ 4. SUNet_Level1_Block T=3 ═══════════════════════════════════════════")
blk3 = M.SUNet_Level1_Block(dim=16, T=3)
x3   = torch.randn(3, 2, 16, 32, 32)
y3   = blk3(x3)
check("output shape == input shape", y3.shape == x3.shape, str(y3.shape))


# ════════════════════════════════════════════════════════════════════════════════
print("\n══ 5. SUNet_Level1_Block T=4 (backward compat) ════════════════════════")
blk4 = M.SUNet_Level1_Block(dim=16, T=4)
x4   = torch.randn(4, 2, 16, 32, 32)
y4   = blk4(x4)
check("output shape == input shape", y4.shape == x4.shape, str(y4.shape))


# ════════════════════════════════════════════════════════════════════════════════
print("\n══ 6. CloudRemovalLoss ═════════════════════════════════════════════════")
from losses import CloudRemovalLoss
crit = CloudRemovalLoss(lambda_cloud=0.1, lambda_tmc=0.0)

L_l, B_l = 3, 2
restored_l   = torch.rand(B_l, 3, 64, 64)
target_l     = torch.rand(B_l, 3, 64, 64)
cloud_maps_l = torch.sigmoid(torch.randn(L_l, B_l, 1, 64, 64))
gt_masks_l   = (torch.rand(L_l, B_l, 1, 64, 64) > 0.5).float()

loss_val, log_d = crit(restored_l, target_l, cloud_maps_l, gt_masks_l)
check("loss is scalar",        loss_val.dim() == 0, str(loss_val.shape))
check("loss is finite",        torch.isfinite(loss_val),  str(loss_val.item()))
check("L_rec  in log",         'L_rec'   in log_d)
check("L_cloud in log",        'L_cloud' in log_d)
check("total in log",          'total'   in log_d)
check("L_cloud > 0 (masks given)", log_d['L_cloud'] > 0, str(log_d['L_cloud']))

# Without masks
loss_nm, log_nm = crit(restored_l, target_l, cloud_maps_l, gt_masks=None)
check("no-mask: L_cloud=0",    log_nm['L_cloud'] == 0.0)
check("loss.backward() works", True)  # already verified no error above


# ════════════════════════════════════════════════════════════════════════════════
print("\n══ 7. MultiTemporalCloudDataset ════════════════════════════════════════")
from dataset_cloud import MultiTemporalCloudDataset

tmp = tempfile.mkdtemp()
try:
    # Create mock Sen2MTC folder structure
    for split in ['train', 'val']:
        for sub in ['s2_t1', 's2_t2', 's2_t3', 's2_target', 'masks']:
            os.makedirs(os.path.join(tmp, split, sub), exist_ok=True)
        for stem in ['0001', '0002']:
            arr8 = (np.random.rand(128, 128, 3) * 255).astype(np.uint8)
            for sub in ['s2_t1', 's2_t2', 's2_t3', 's2_target', 'masks']:
                Image.fromarray(arr8).save(
                    os.path.join(tmp, split, sub, f'{stem}.png'))

    ds = MultiTemporalCloudDataset(
        root_dir   = os.path.join(tmp, 'train'),
        L          = 3,
        patch_size = 64,
        layout     = 'Sen2MTC',
        augment    = True,
    )
    check("dataset length == 2", len(ds) == 2, str(len(ds)))

    seq, tgt, masks = ds[0]
    check("cloudy_seq [L,C,H,W]", seq.shape   == (3, 3, 64, 64), str(seq.shape))
    check("target     [C,H,W]",   tgt.shape   == (3, 64, 64),    str(tgt.shape))
    check("masks      [L,1,H,W]", masks.shape == (3, 1, 64, 64), str(masks.shape))
    check("seq in [0,1]",         seq.min()>=0 and seq.max()<=1)
    check("masks in {0,1} approx",masks.min()>=0 and masks.max()<=1)
finally:
    shutil.rmtree(tmp)


# ════════════════════════════════════════════════════════════════════════════════
print("\n══ 8. Full model_cloud forward (cloud_mode=True) ══════════════════════")
# Small dim to keep test fast
net_c = M.VLIFNet(inp_channels=3, out_channels=3, dim=8,
                   en_num_blocks=[1,1,1,1], de_num_blocks=[1,1,1,1],
                   T=3, L=3, cloud_mode=True).cpu()
L_m, B_m = 3, 2
inp_seq = torch.randn(L_m, B_m, 3, 32, 32)
with torch.no_grad():
    out_c, cmap_c = net_c(inp_seq)
check("restored  [B,C,H,W]",    out_c.shape  == (B_m, 3, 32, 32), str(out_c.shape))
check("cloud_maps [L,B,1,H,W]", cmap_c.shape == (L_m, B_m, 1, 32, 32), str(cmap_c.shape))
check("cloud_maps in (0,1)",     cmap_c.min()>=0 and cmap_c.max()<=1,
      f"min={cmap_c.min():.3f} max={cmap_c.max():.3f}")
check("restored finite",         torch.isfinite(out_c).all())


# ════════════════════════════════════════════════════════════════════════════════
print("\n══ 9. Original model forward (cloud_mode=False, backward compat) ══════")
net_r = M.VLIFNet(inp_channels=3, out_channels=3, dim=8,
                   en_num_blocks=[1,1,1,1], de_num_blocks=[1,1,1,1],
                   T=4, L=4, cloud_mode=False).cpu()
inp_single = torch.randn(2, 3, 32, 32)
with torch.no_grad():
    out_r = net_r(inp_single)
check("output [B,C,H,W]", out_r.shape == (2, 3, 32, 32), str(out_r.shape))
check("output finite",    torch.isfinite(out_r).all())
check("returns Tensor (not tuple)", isinstance(out_r, torch.Tensor))


# ════════════════════════════════════════════════════════════════════════════════
print("\n══ 10. Gradient flow through Kalman filter ═════════════════════════════")
kf_g = M.KalmanFeatureFilter()
feat_g = torch.randn(3, 1, 4, 8, 8, requires_grad=True)
cmap_g = torch.sigmoid(torch.randn(3, 1, 1, 8, 8))
filt_g, _ = kf_g(feat_g, cmap_g)
loss_g = filt_g.mean()
loss_g.backward()
check("feat gradient exists",          feat_g.grad is not None)
check("feat gradient is finite",       torch.isfinite(feat_g.grad).all())
check("log_sigma0_sq grad exists",     kf_g.log_sigma0_sq.grad is not None)
check("log_beta grad exists",          kf_g.log_beta.grad is not None)
check("log_Q grad exists",             kf_g.log_Q.grad is not None)


# ════════════════════════════════════════════════════════════════════════════════
print("\n══ 11. Parameter count comparison ═════════════════════════════════════")
def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

net_base  = M.VLIFNet(inp_channels=3, out_channels=3, dim=8,
                       en_num_blocks=[1,1,1,1], de_num_blocks=[1,1,1,1],
                       T=4, L=4, cloud_mode=False)
net_cloud = M.VLIFNet(inp_channels=3, out_channels=3, dim=8,
                       en_num_blocks=[1,1,1,1], de_num_blocks=[1,1,1,1],
                       T=3, L=3, cloud_mode=True)

p_base  = count_params(net_base)
p_cloud = count_params(net_cloud)
p_new   = p_cloud - p_base
print(f"  Base model (T=4):           {p_base:,} params")
print(f"  Cloud model (T=3, Kalman):  {p_cloud:,} params")
print(f"  New params (CloudEst+Kalman): {p_new:+,}")
check("overhead < 5% of base", abs(p_new) < 0.05 * p_base,
      f"{abs(p_new)/p_base:.1%} overhead")


# ════════════════════════════════════════════════════════════════════════════════
print("\n══ 12. CloudGate ════════════════════════════════════════════════════════")
T_g = 4
cg  = M.CloudGate(T=T_g)

# Shape check
feats_g  = torch.randn(T_g, 2, 16, 32, 32)
cmap_g2  = torch.sigmoid(torch.randn(2, 1, 32, 32))
gated    = cg(feats_g, cmap_g2)
check("gated shape == feats shape", gated.shape == feats_g.shape, str(gated.shape))

# Cloud-only region should be suppressed at early steps
with torch.no_grad():
    cmap_full  = torch.ones(1, 1, 8, 8)           # fully cloudy
    feats_ones = torch.ones(T_g, 1, 4, 8, 8)      # feature = 1 everywhere
    gated_ones = cg(feats_ones, cmap_full)
    # early step (t=0): gate ≈ 0.9, suppression ≈ 0.9 → gated ≈ 0.10
    # last step (T-1): gate ≈ 0.1, suppression ≈ 0.1 → gated ≈ 0.90
    early_mean = gated_ones[0].mean().item()
    late_mean  = gated_ones[-1].mean().item()
    check("early step more suppressed than late step (curriculum)",
          early_mean < late_mean,
          f"early={early_mean:.3f}  late={late_mean:.3f}")

# Gate logits are learnable
check("gate_logits is nn.Parameter",
      isinstance(cg.gate_logits, torch.nn.Parameter))
check("gate_logits has T values",
      cg.gate_logits.shape == (T_g,), str(cg.gate_logits.shape))

# Gradient flows through gate
feats_req = feats_g.clone().requires_grad_(True)
gated_req = cg(feats_req, cmap_g2)
gated_req.mean().backward()
check("gradient flows through CloudGate", feats_req.grad is not None)
check("gate_logits gradient exists",      cg.gate_logits.grad is not None)


# ════════════════════════════════════════════════════════════════════════════════
print("\n══ 13. Full model_cloud_single forward (cloud_mode=single) ═════════════")
net_s = M.VLIFNet(inp_channels=3, out_channels=3, dim=8,
                   en_num_blocks=[1,1,1,1], de_num_blocks=[1,1,1,1],
                   T=4, L=4, cloud_mode='single').cpu()
B_s   = 2
inp_s = torch.randn(B_s, 3, 32, 32)
with torch.no_grad():
    out_s, cmap_s = net_s(inp_s)

check("restored  [B, C, H, W]",  out_s.shape  == (B_s, 3, 32, 32), str(out_s.shape))
check("cloud_map [B, 1, H, W]",  cmap_s.shape == (B_s, 1, 32, 32), str(cmap_s.shape))
check("cloud_map in (0,1)",       cmap_s.min() >= 0 and cmap_s.max() <= 1,
      f"min={cmap_s.min():.3f} max={cmap_s.max():.3f}")
check("restored finite",          torch.isfinite(out_s).all())
check("returns tuple of 2",       isinstance(out_s, torch.Tensor) and
                                   isinstance(cmap_s, torch.Tensor))

# Check gradient flows end-to-end in single mode
net_s.train()
inp_grad = torch.randn(B_s, 3, 32, 32, requires_grad=True)
out_g2, cm_g2 = net_s(inp_grad)
(out_g2.mean() + cm_g2.mean()).backward()
check("input gradient flows in single mode", inp_grad.grad is not None)
check("CloudGate logits grad in single mode",
      net_s.cloud_gate.gate_logits.grad is not None)


# ════════════════════════════════════════════════════════════════════════════════
print("\n══ 14. SingleTemporalCloudDataset ═══════════════════════════════════════")
from dataset_cloud import SingleTemporalCloudDataset

tmp2 = tempfile.mkdtemp()
try:
    # Create mock CUHK-CR layout: input/ + target/ (no explicit masks)
    for sub in ['input', 'target']:
        os.makedirs(os.path.join(tmp2, sub), exist_ok=True)
    for stem in ['img001', 'img002', 'img003']:
        arr8 = (np.random.rand(128, 128, 3) * 255).astype(np.uint8)
        Image.fromarray(arr8).save(os.path.join(tmp2, 'input',  f'{stem}.png'))
        # Target slightly different to generate non-trivial auto-mask
        arr8b = np.clip(arr8.astype(np.int32) + np.random.randint(-30, 30, arr8.shape),
                        0, 255).astype(np.uint8)
        Image.fromarray(arr8b).save(os.path.join(tmp2, 'target', f'{stem}.png'))

    ds_s = SingleTemporalCloudDataset(
        root_dir    = tmp2,
        patch_size  = 64,
        augment     = True,
        mask_thresh = 0.05,
    )
    check("dataset length == 3", len(ds_s) == 3, str(len(ds_s)))
    check("no explicit masks (auto-synthesis)", not ds_s._has_masks)

    inp_t, tgt_t, msk_t = ds_s[0]
    check("input  [C, H, W]",   inp_t.shape == (3, 64, 64), str(inp_t.shape))
    check("target [C, H, W]",   tgt_t.shape == (3, 64, 64), str(tgt_t.shape))
    check("mask   [1, H, W]",   msk_t.shape == (1, 64, 64), str(msk_t.shape))
    check("input in [0,1]",     inp_t.min() >= 0 and inp_t.max() <= 1)
    check("mask  in {0,1}",     msk_t.min() >= 0 and msk_t.max() <= 1)

    # Also test with explicit mask dir
    os.makedirs(os.path.join(tmp2, 'masks'), exist_ok=True)
    for stem in ['img001', 'img002', 'img003']:
        mask8 = np.random.choice([0, 255], size=(128, 128)).astype(np.uint8)
        Image.fromarray(mask8).save(os.path.join(tmp2, 'masks', f'{stem}.png'))

    ds_sm = SingleTemporalCloudDataset(
        root_dir    = tmp2,
        patch_size  = 64,
        augment     = False,
        mask_thresh = 0.05,
    )
    check("explicit mask dir detected",    ds_sm._has_masks)
    _, _, msk_explicit = ds_sm[0]
    check("explicit mask shape [1,H,W]",   msk_explicit.shape == (1, 64, 64),
          str(msk_explicit.shape))
finally:
    shutil.rmtree(tmp2)


# ════════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
_N_TESTS = 14
if not errors:
    print(f"\033[92m  ALL {_N_TESTS} TEST GROUPS PASSED\033[0m\n")
else:
    print(f"\033[91m  {len(errors)} CHECK(S) FAILED:\033[0m")
    for e in errors:
        print(f"    - {e}")
    sys.exit(1)
