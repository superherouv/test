from spikingjelly.activation_based.neuron import LIFNode
from spikingjelly.activation_based import functional, layer
import torch
import torch.nn as nn
import torch.nn.functional as F

v_th = 0.15

alpha = 1 / (2 ** 0.5)

# ─── Constants ────────────────────────────────────────────────────────────────
decay = 0.25          # fixed SNN membrane decay (used only in mem_update)
_r = 2                # spatial fold factor in SUNet_Level1_Block (r²=4 steps per image)


# ─── MultiSpike4 ──────────────────────────────────────────────────────────────
class MultiSpike4(nn.Module):
    class quant4(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            ctx.save_for_backward(input)
            quantized = torch.round(torch.clamp(input, min=0, max=4))
            return quantized / 4.0

        @staticmethod
        def backward(ctx, grad_output):
            input, = ctx.saved_tensors
            grad_input = grad_output.clone()
            grad_input[input < 0] = 0
            grad_input[input > 4] = 0
            return grad_input / 4.0

    def forward(self, x):
        return self.quant4.apply(x)


# ─── mem_update ───────────────────────────────────────────────────────────────
class mem_update(nn.Module):
    """
    LIF membrane update with optional per-step adaptive decay.

    adaptive_decay: None  → use global fixed decay=0.25 (original behaviour)
                    Tensor [T, B, 1, H, W] → Kalman-derived pixel-wise decay
    """
    def __init__(self):
        super(mem_update, self).__init__()
        self.qtrick = MultiSpike4()

    def forward(self, x, adaptive_decay=None):
        # x: [T, B, C, H, W]
        spike = torch.zeros_like(x[0]).to(x.device)
        output = torch.zeros_like(x)
        mem_old = 0
        T = x.shape[0]

        for i in range(T):
            if i >= 1:
                if adaptive_decay is not None:
                    d = adaptive_decay[i]          # [B, 1, H, W], broadcasts over C
                else:
                    d = decay
                mem = (mem_old - spike.detach()) * d + x[i]
            else:
                mem = x[i]
            spike = self.qtrick(mem)
            mem_old = mem.clone()
            output[i] = spike
        return output


# ─── CloudEstimator ───────────────────────────────────────────────────────────
class CloudEstimator(nn.Module):
    """
    Lightweight per-image cloud probability estimator.

    Input : [B, C_in, H, W]  (single temporal image, raw pixel values in [0,1])
    Output: [B, 1,    H, W]  cloud probability map in [0, 1]
                              1 = fully cloudy, 0 = clear

    Design: 3 conv layers (no pooling, preserves spatial resolution).
    Only ~500 params – negligible overhead.
    """
    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─── KalmanFeatureFilter ──────────────────────────────────────────────────────
class KalmanFeatureFilter(nn.Module):
    """
    Theoretically-optimal multi-temporal feature fusion via Kalman filtering.

    Replaces heuristic α+(1-α)c decay with MMSE-optimal state estimation.

    State-space model
    -----------------
        s_l  : true (cloud-free) surface feature (hidden state)
        f_l  : embedded feature of temporal image l (noisy observation)
        c_l  : cloud probability map            (controls observation noise)

    Kalman recursion (pixel-wise, all operations broadcast over channels C)
    -----------------------------------------------------------------------
        R_l    = σ₀² + β · c_l              observation noise variance
        K_l    = P_{l-1} / (P_{l-1} + R_l)  Kalman gain   ∈ (0, 1)
        s_l    = (1 - K_l) · s_{l-1}        prior keep
               + K_l · f_l                  current update
        P_l    = (1 - K_l) · P_{l-1} + Q   posterior variance

    Parameters (all log-parameterised to ensure positivity, all learnable)
    -----------------------------------------------------------------------
        σ₀²   base observation noise  (init exp(0) = 1.0)
        β     cloud noise coefficient (init exp(0) = 1.0)
        Q     process noise           (init exp(-2) ≈ 0.135, surface changes slowly)
        P_init initial uncertainty    (init exp(0) = 1.0)

    Why this matters for all-cloudy sequences
    ------------------------------------------
    When every temporal image is cloudy, c_l ≈ 1 for all l, so R_l is large.
    But as l increases, P_{l-1} also grows (accumulated uncertainty), causing
    K_l to gradually rise and the model to attempt reconstruction from faint
    residual signals rather than endlessly repeating the initial state.
    A heuristic linear map (decay = α+(1-α)c) cannot reproduce this behaviour.
    """

    def __init__(self):
        super().__init__()
        # Log-parameterised: ensures all noise variances > 0 during training
        self.log_sigma0_sq = nn.Parameter(torch.tensor(0.0))   # ln(σ₀²)
        self.log_beta      = nn.Parameter(torch.tensor(0.0))   # ln(β)
        self.log_Q         = nn.Parameter(torch.tensor(-2.0))  # ln(Q)
        self.log_P_init    = nn.Parameter(torch.tensor(0.0))   # ln(P₀)

    def forward(self, feats: torch.Tensor, cloud_maps: torch.Tensor):
        """
        Parameters
        ----------
        feats      : [L, B, C, H, W]  embedded temporal features
        cloud_maps : [L, B, 1, H, W]  cloud probability per temporal image

        Returns
        -------
        filtered   : [L, B, C, H, W]  Kalman-filtered feature sequence
        gains      : [L, B, 1, H, W]  Kalman gains (= 1 - optimal decay)
                     High gain  → clear frame, feature update trusted.
                     Low  gain  → cloudy frame, prior state preserved.
        """
        L, B, C, H, W = feats.shape

        sigma0_sq = torch.exp(self.log_sigma0_sq)          # scalar > 0
        beta      = torch.exp(self.log_beta)               # scalar > 0
        Q         = torch.exp(self.log_Q)                  # scalar > 0
        P_init    = torch.exp(self.log_P_init)             # scalar > 0

        # Initialise state and variance  [B, 1, H, W]
        P     = P_init.expand(B, 1, H, W).clone()
        state = torch.zeros_like(feats[0])                 # [B, C, H, W]

        filtered_list: list = []
        gain_list:     list = []

        for l in range(L):
            c_l = cloud_maps[l]                            # [B, 1, H, W]

            # Observation noise: cloudy → large R → small K → trust prior
            R_l = sigma0_sq + beta * c_l                   # [B, 1, H, W]

            # Optimal Kalman gain
            K_l = P / (P + R_l)                            # [B, 1, H, W]

            # State update (K_l broadcasts over C)
            state = (1.0 - K_l) * state + K_l * feats[l]  # [B, C, H, W]

            # Posterior variance update
            P = (1.0 - K_l) * P + Q                        # [B, 1, H, W]

            filtered_list.append(state)
            gain_list.append(K_l)

        filtered = torch.stack(filtered_list, dim=0)       # [L, B, C, H, W]
        gains    = torch.stack(gain_list,     dim=0)       # [L, B, 1, H, W]
        return filtered, gains


# ─── PixelShuffleLIFBlock ─────────────────────────────────────────────────────
class PixelShuffleLIFBlock(nn.Module):
    def __init__(self, in_channels, downsample_factor=2):
        super().__init__()
        self.downsample_factor = downsample_factor
        self.r = downsample_factor
        self.pixel_unshuffle = nn.PixelUnshuffle(downsample_factor)
        self.lif_node = mem_update()
        self.channel_adjust = layer.Conv2d(
            in_channels * (downsample_factor ** 2),
            in_channels,
            kernel_size=1,
            bias=False,
            step_mode='m'
        )
        functional.set_step_mode(self.channel_adjust, step_mode='m')

    def forward(self, x):
        T, B, C, H, W = x.shape
        x_flat        = x.reshape(T * B, C, H, W)
        x_downsampled = self.pixel_unshuffle(x_flat)
        x_reshaped    = x_downsampled.reshape(T, B, C * self.r**2, H//self.r, W//self.r)
        x_patch_sep   = x_reshaped.reshape(T, B, C, self.r**2, H//self.r, W//self.r)
        x_temporal    = x_patch_sep.permute(0, 3, 1, 2, 4, 5).contiguous()
        x_temporal    = x_temporal.reshape(T * self.r**2, B, C, H//self.r, W//self.r)
        x_lif_output  = self.lif_node(x_temporal)
        x_reorg       = x_lif_output.reshape(T, self.r**2, B, C, H//self.r, W//self.r)
        x_reorg       = x_reorg.permute(0, 2, 3, 1, 4, 5).contiguous()
        x_reorg       = x_reorg.reshape(T, B, C * self.r**2, H//self.r, W//self.r)
        x_flat_r      = x_reorg.reshape(T * B, C * self.r**2, H//self.r, W//self.r)
        x_up          = F.interpolate(x_flat_r, size=(H, W), mode='bilinear', align_corners=False)
        x_restored    = x_up.reshape(T, B, C * self.r**2, H, W)
        return self.channel_adjust(x_restored)


# ─── TimeAttention ────────────────────────────────────────────────────────────
class TimeAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(TimeAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        bottleneck = max(1, in_planes // ratio)
        self.sharedMLP = nn.Sequential(
            nn.Conv3d(in_planes, bottleneck, 1, bias=False),
            nn.ReLU(),
            nn.Conv3d(bottleneck, in_planes, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


# ─── SUNet_Level1_Block ───────────────────────────────────────────────────────
class SUNet_Level1_Block(nn.Module):
    """
    SUNet Level-1 block with parameterised T (number of temporal images).

    Two-level temporal hierarchy
    ----------------------------
    Coarse (L images)  : handled by Kalman filter *before* this block.
    Fine   (r²=4 steps): spatial-fold each image into r²=4 SNN sub-steps.
    Total SNN steps inside block = T * r² = T * 4.

    When T=L=3 : total steps = 12 (was 16 for T=4).
    When T=4   : total steps = 16 (original, backward-compatible).
    """

    def __init__(self, dim: int, T: int = 4):
        super(SUNet_Level1_Block, self).__init__()
        functional.set_step_mode(self, step_mode='m')

        self.T = T
        self.downsample_factor = 2
        self.r = 2                          # spatial fold factor
        _r2 = self.r ** 2                   # = 4
        T_ext = T * _r2                     # extended SNN steps after folding

        # Reduction for MultiDimensionalAttention: keep bottleneck = 1
        _rt = max(1, T)                     # reduction_t = T  →  T // T = 1

        # 1. Initial residual block
        self.initial_residual = Spiking_Residual_Block(dim=dim, T=T)

        # 2. Spatial fold (no learnable params here)
        self.pixel_unshuffle = nn.PixelUnshuffle(self.downsample_factor)

        # LIF on extended time axis (T_ext steps, fixed decay)
        self.lif_node = mem_update()

        # 3. TimeAttention on T_ext = T*4 extended steps
        self.time_attention = TimeAttention(in_planes=T_ext, ratio=4)

        # 4. 3D conv compresses T_ext back to T (kernel=r²=4, stride=r²=4)
        self.temporal_compress = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=(_r2, 1, 1),
                      stride=(_r2, 1, 1), padding=0, bias=False),
            nn.ReLU(inplace=True),
        )

        # 5. Two LIF-Conv-BN-Attention blocks (T steps at this point)
        self.lif_1  = LIFNode(v_threshold=v_th, backend='cupy', step_mode='m', decay_input=False)
        self.conv_1 = layer.Conv2d(dim, dim, 3, 1, 1, bias=False, step_mode='m')
        self.bn_1   = layer.ThresholdDependentBatchNorm2d(dim, alpha=alpha, v_th=v_th, affine=True)
        self.attn_1 = layer.MultiDimensionalAttention(T=T, reduction_t=_rt, reduction_c=16,
                                                       kernel_size=3, C=dim)

        self.lif_2  = LIFNode(v_threshold=v_th, backend='cupy', step_mode='m', decay_input=False)
        self.conv_2 = layer.Conv2d(dim, dim, 3, 1, 1, bias=False, step_mode='m')
        self.bn_2   = layer.ThresholdDependentBatchNorm2d(dim, alpha=alpha, v_th=v_th, affine=True)
        self.attn_2 = layer.MultiDimensionalAttention(T=T, reduction_t=_rt, reduction_c=16,
                                                       kernel_size=3, C=dim)

        # 6. Upsample + skip
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        # 7. Final attention after skip connection
        self.final_attn = layer.MultiDimensionalAttention(T=T, reduction_t=_rt, reduction_c=16,
                                                           kernel_size=3, C=dim)

    def forward(self, x):
        # x: [T, B, C, H, W]
        residual_out  = self.initial_residual(x)
        skip_features = residual_out.clone()

        T, B, C, H, W = residual_out.shape

        # ── Spatial fold: T → T * r² extended time steps ──────────────────
        x_flat       = residual_out.reshape(T * B, C, H, W)
        x_down       = self.pixel_unshuffle(x_flat)          # [T*B, C*4, H/2, W/2]
        x_reshaped   = x_down.reshape(T, B, C * self.r**2, H//self.r, W//self.r)
        x_patch_sep  = x_reshaped.reshape(T, B, C, self.r**2, H//self.r, W//self.r)
        x_temporal   = x_patch_sep.permute(0, 3, 1, 2, 4, 5).contiguous()
        downsampled  = x_temporal.reshape(T * self.r**2, B, C, H//self.r, W//self.r)

        # ── LIF over T_ext steps ────────────────────────────────────────────
        downsampled = self.lif_node(downsampled)             # [T*4, B, C, H/2, W/2]

        # ── TimeAttention on T_ext steps ────────────────────────────────────
        T_ext, B, C, H_d, W_d = downsampled.shape
        ds_t = downsampled.transpose(0, 1)                   # [B, T_ext, C, H/2, W/2]
        time_att = self.time_attention(ds_t)                 # [B, T_ext, C, H/2, W/2]
        attended = (ds_t * time_att).transpose(0, 1)         # [T_ext, B, C, H/2, W/2]

        # ── 3D conv: T_ext → T ──────────────────────────────────────────────
        attended_3d = attended.permute(1, 2, 0, 3, 4)       # [B, C, T_ext, H/2, W/2]
        compressed  = self.temporal_compress(attended_3d)    # [B, C, T,     H/2, W/2]
        out         = compressed.permute(2, 0, 1, 3, 4)     # [T, B, C, H/2, W/2]

        # ── Two LIF-Conv-BN-Attention blocks ────────────────────────────────
        out = self.lif_1(out)
        out = self.conv_1(out)
        out = self.bn_1(out)
        out = self.attn_1(out)

        out = self.lif_2(out)
        out = self.conv_2(out)
        out = self.bn_2(out)
        out = self.attn_2(out)

        # ── Upsample back to original H, W ──────────────────────────────────
        T2, B2, C2, H2, W2 = out.shape
        out_flat = out.reshape(T2 * B2, C2, H2, W2)
        up_flat  = self.upsample(out_flat)
        upsampled = up_flat.reshape(T2, B2, C2, up_flat.shape[2], up_flat.shape[3])

        # ── Skip connection + final attention ───────────────────────────────
        combined  = upsampled + skip_features
        final_out = self.final_attn(combined)
        return final_out


# ─── OverlapPatchEmbed ────────────────────────────────────────────────────────
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=32, spike_mode="lif",
                 LayerNorm_type='WithBias', bias=False, T=4):
        super(OverlapPatchEmbed, self).__init__()
        functional.set_step_mode(self, step_mode='m')
        self.proj = layer.Conv2d(in_c, embed_dim, kernel_size=3, stride=1,
                                 padding=1, bias=bias)

    def forward(self, x):
        return self.proj(x)


# ─── Spiking_Residual_Block ───────────────────────────────────────────────────
class Spiking_Residual_Block(nn.Module):
    def __init__(self, dim: int, T: int = 4):
        super(Spiking_Residual_Block, self).__init__()
        functional.set_step_mode(self, step_mode='m')

        _rt = max(1, T)   # reduction_t = T → bottleneck = T//T = 1

        self.lif_1 = LIFNode(v_threshold=v_th, backend='cupy', step_mode='m', decay_input=False)
        self.conv1 = layer.Conv2d(dim, dim, 3, 1, 1, bias=False, step_mode='m')
        self.bn1   = layer.ThresholdDependentBatchNorm2d(dim, alpha=alpha, v_th=v_th, affine=True)

        self.high_freq_scale_1 = nn.Parameter(torch.ones(1))
        self.low_freq_scale_1  = nn.Parameter(torch.ones(1))

        self.lif_2 = PixelShuffleLIFBlock(in_channels=dim, downsample_factor=2)
        self.conv2 = layer.Conv2d(dim, dim, 3, 1, 1, bias=False, step_mode='m')
        self.bn2   = layer.ThresholdDependentBatchNorm2d(dim, alpha=alpha,
                                                          v_th=v_th * 0.2, affine=True)

        self.high_freq_scale_2 = nn.Parameter(torch.ones(1))
        self.low_freq_scale_2  = nn.Parameter(torch.ones(1))

        self.shortcut = nn.Sequential(
            layer.Conv2d(dim, dim, 3, 1, 1, bias=False, step_mode='m'),
            layer.ThresholdDependentBatchNorm2d(dim, alpha=alpha, v_th=v_th, affine=True),
        )

        self.attn = layer.MultiDimensionalAttention(T=T, reduction_t=_rt,
                                                     reduction_c=16, kernel_size=3, C=dim)

    def forward(self, x):
        x_h_1 = self.lif_1(x)
        x_l_1 = x - x_h_1
        x_h_1_scaled = self.high_freq_scale_1 * x_h_1
        x_l_1_scaled = self.low_freq_scale_1  * x_l_1
        x_enhanced_1 = x * x_h_1
        combined_1   = x_h_1_scaled + x_l_1_scaled + x_enhanced_1
        out          = self.conv1(combined_1)
        out          = self.bn1(out)

        x_h_2 = self.lif_2(out)
        x_l_2 = out - x_h_2
        x_h_2_scaled = self.high_freq_scale_2 * x_h_2
        x_l_2_scaled = self.low_freq_scale_2  * x_l_2
        x_enhanced_2 = out * x_h_2
        combined_2   = x_h_2_scaled + x_l_2_scaled + x_enhanced_2
        out          = self.conv2(combined_2)
        out          = self.bn2(out)

        shortcut = torch.clone(x)
        out      = out + self.shortcut(shortcut)
        out      = self.attn(out) + shortcut
        return out


# ─── DownSampling / UpSampling ───────────────────────────────────────────────
class DownSampling(nn.Module):
    def __init__(self, dim):
        super(DownSampling, self).__init__()
        functional.set_step_mode(self, step_mode='m')
        self.lif  = LIFNode(v_threshold=v_th, backend='cupy', step_mode='m', decay_input=False)
        self.conv = layer.Conv2d(dim, dim * 2, 3, stride=2, padding=1, step_mode='m', bias=False)
        self.bn   = layer.ThresholdDependentBatchNorm2d(alpha=alpha, v_th=v_th,
                                                         num_features=dim * 2, affine=True)

    def forward(self, x):
        x = self.lif(x)
        x = self.conv(x)
        x = self.bn(x)
        return x


class UpSampling(nn.Module):
    def __init__(self, dim):
        super(UpSampling, self).__init__()
        self.scale_factor = 2
        self.lif  = LIFNode(v_threshold=v_th, backend='cupy', step_mode='m', decay_input=False)
        self.conv = layer.Conv2d(dim, dim // 2, 3, 1, 1, step_mode='m', bias=False)
        self.bn   = layer.ThresholdDependentBatchNorm2d(alpha=alpha, v_th=v_th,
                                                         num_features=dim // 2, affine=True)

    def forward(self, input):
        temp = torch.zeros((input.shape[0], input.shape[1], input.shape[2],
                            input.shape[3] * self.scale_factor,
                            input.shape[4] * self.scale_factor)).cuda()
        output = []
        for i in range(input.shape[0]):
            temp[i] = F.interpolate(input[i], scale_factor=self.scale_factor, mode='bilinear')
            output.append(temp[i])
        out = torch.stack(output, dim=0)
        out = self.lif(out)
        out = self.conv(out)
        out = self.bn(out)
        return out


# ─── VLIFNet ─────────────────────────────────────────────────────────────────
class VLIFNet(nn.Module):
    """
    VLIFNet: Spiking U-Net for image restoration.

    Two operating modes
    -------------------
    cloud_mode=False  (default, backward-compatible)
        Single input [B, C, H, W] is repeated T times.
        Returns: restored image [B, C, H, W]

    cloud_mode=True   (multi-temporal cloud removal)
        Input is a sequence [L, B, C, H, W] of L temporal images.
        CloudEstimator estimates per-image cloud probability maps.
        KalmanFeatureFilter fuses the L embedded feature maps with
        theoretically-optimal Kalman weights before the SNN encoder.
        Returns: (restored [B, C, H, W], cloud_maps [L, B, 1, H, W])
        The cloud_maps can be used to compute cloud detection auxiliary loss.

    Two-level temporal hierarchy (cloud_mode=True)
    -----------------------------------------------
    Level 1 – Kalman   (coarse, L images):
        Builds running state estimate, discarding cloudy observations.
    Level 2 – SNN r²   (fine, r²=4 spatial sub-steps per image):
        Folded spatial patches supply intra-image feature diversity.
    """

    def __init__(
        self,
        inp_channels: int = 3,
        out_channels: int = 3,
        dim: int = 24,
        en_num_blocks: list = None,
        de_num_blocks: list = None,
        bias: bool = False,
        T: int = 4,          # SNN time steps (= L when cloud_mode=True)
        L: int = 4,          # number of temporal images (= T when cloud_mode=True)
        cloud_mode: bool = False,
        use_refinement: bool = False,
    ):
        super(VLIFNet, self).__init__()

        if en_num_blocks is None:
            en_num_blocks = [4, 4, 6, 6]
        if de_num_blocks is None:
            de_num_blocks = [4, 4, 6, 6]

        functional.set_backend(self, backend='cupy')
        functional.set_step_mode(self, step_mode='m')

        self.T           = T
        self.L           = L
        self.cloud_mode  = cloud_mode
        self.use_refinement = use_refinement

        # ── Cloud modules (only for cloud_mode) ────────────────────────────
        if cloud_mode:
            self.cloud_estimator = CloudEstimator(in_channels=inp_channels)
            self.kalman_filter   = KalmanFeatureFilter()

        # ── Patch embedding ────────────────────────────────────────────────
        self.patch_embed = OverlapPatchEmbed(in_c=inp_channels, embed_dim=dim, T=T)

        # ── Encoder ────────────────────────────────────────────────────────
        self.encoder_level1 = SUNet_Level1_Block(dim=int(dim * 1), T=T)

        self.down1_2        = DownSampling(dim)
        self.encoder_level2 = nn.Sequential(*[
            Spiking_Residual_Block(dim=int(dim * 2), T=T)
            for _ in range(en_num_blocks[1])
        ])

        self.down2_3        = DownSampling(int(dim * 2))
        self.encoder_level3 = nn.Sequential(*[
            Spiking_Residual_Block(dim=int(dim * 4), T=T)
            for _ in range(en_num_blocks[2])
        ])

        # ── Decoder ────────────────────────────────────────────────────────
        self.decoder_level3 = nn.Sequential(*[
            Spiking_Residual_Block(dim=int(dim * 4), T=T)
            for _ in range(de_num_blocks[2])
        ])

        self.up3_2 = UpSampling(int(dim * 4))

        self.lif_level2         = LIFNode(v_threshold=v_th, backend='cupy',
                                           step_mode='m', decay_input=False)
        self.reduce_conv_level2 = layer.Conv2d(int(dim * 4), int(dim * 2),
                                               kernel_size=1, bias=bias, step_mode='m')
        self.reduce_bn_level2   = layer.ThresholdDependentBatchNorm2d(
                                      num_features=int(dim * 2), alpha=alpha, v_th=v_th)

        self.decoder_level2 = nn.Sequential(*[
            Spiking_Residual_Block(dim=int(dim * 2), T=T)
            for _ in range(de_num_blocks[1])
        ])

        self.up2_1 = UpSampling(int(dim * 2))

        self.lif_level1         = LIFNode(v_threshold=v_th, backend='cupy',
                                           step_mode='m', decay_input=False)
        self.reduce_conv_level1 = layer.Conv2d(int(dim * 2), int(dim * 1),
                                               kernel_size=1, bias=bias, step_mode='m')
        self.reduce_bn_level1   = layer.ThresholdDependentBatchNorm2d(
                                      num_features=int(dim * 1), alpha=alpha, v_th=v_th)

        self.decoder_level1            = SUNet_Level1_Block(dim=int(dim * 1), T=T)
        self.additional_sunet_level1   = SUNet_Level1_Block(dim=int(dim * 1), T=T)

        if self.use_refinement:
            self.refinement_blocks = nn.Sequential(*[
                Spiking_Residual_Block(dim=int(dim * 1), T=T) for _ in range(4)
            ])

        self.output = nn.Sequential(
            nn.Conv2d(int(dim * 1), out_channels, kernel_size=3, stride=1, padding=1)
        )

    # ── Forward ───────────────────────────────────────────────────────────────
    def forward(self, inp_img: torch.Tensor):
        """
        Parameters
        ----------
        inp_img : [B, C, H, W]          when cloud_mode=False
                  [L, B, C, H, W]       when cloud_mode=True

        Returns
        -------
        cloud_mode=False : restored  [B, C, H, W]
        cloud_mode=True  : (restored [B, C, H, W],
                            cloud_maps [L, B, 1, H, W])
        """

        # ── Branch: original single-image mode (backward compat) ───────────
        if not self.cloud_mode:
            short = inp_img.clone()
            if inp_img.dim() == 4:
                inp_img = inp_img.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)

            inp_enc_level1 = self.patch_embed(inp_img)
            out_enc_level1 = self.encoder_level1(inp_enc_level1)
            inp_enc_level2 = self.down1_2(out_enc_level1)
            out_enc_level2 = self.encoder_level2(inp_enc_level2)
            inp_enc_level3 = self.down2_3(out_enc_level2)
            out_enc_level3 = self.encoder_level3(inp_enc_level3)

            out_dec_level3 = self.decoder_level3(out_enc_level3)
            inp_dec_level2 = self.up3_2(out_dec_level3)
            inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], dim=2)
            inp_dec_level2 = self.lif_level2(inp_dec_level2)
            inp_dec_level2 = self.reduce_conv_level2(inp_dec_level2)
            inp_dec_level2 = self.reduce_bn_level2(inp_dec_level2)
            out_dec_level2 = self.decoder_level2(inp_dec_level2)

            inp_dec_level1 = self.up2_1(out_dec_level2)
            inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], dim=2)
            inp_dec_level1 = self.lif_level1(inp_dec_level1)
            inp_dec_level1 = self.reduce_conv_level1(inp_dec_level1)
            inp_dec_level1 = self.reduce_bn_level1(inp_dec_level1)
            out_dec_level1 = self.decoder_level1(inp_dec_level1)
            out_dec_level1 = self.additional_sunet_level1(out_dec_level1)

            if self.use_refinement:
                out_dec_level1 = self.refinement_blocks(out_dec_level1)

            return self.output(out_dec_level1.mean(0)) + short

        # ── Branch: multi-temporal cloud removal ───────────────────────────
        # inp_img: [L, B, C, H, W]
        assert inp_img.dim() == 5, (
            "cloud_mode=True expects input [L, B, C, H, W], "
            f"got shape {tuple(inp_img.shape)}"
        )
        L, B, C, H, W = inp_img.shape

        # Residual reference: clearness-weighted mean of raw input frames
        # (computed here before cloud maps are available; replaced below)
        short = inp_img.mean(0)          # simple mean as placeholder [B, C, H, W]

        # ── 1. Cloud probability estimation ────────────────────────────────
        cloud_maps = torch.stack(
            [self.cloud_estimator(inp_img[l]) for l in range(L)], dim=0
        )  # [L, B, 1, H, W]  ∈ (0, 1)

        # Clearness-weighted mean for residual connection (more informative)
        clearness = 1.0 - cloud_maps                    # [L, B, 1, H, W]
        w_sum     = clearness.sum(0, keepdim=True)      # [1, B, 1, H, W]
        weights   = clearness / (w_sum + 1e-6)          # normalised [L, B, 1, H, W]
        short     = (inp_img * weights).sum(0)          # [B, C, H, W]

        # ── 2. Patch embedding (shared weights across L images) ─────────────
        inp_enc = self.patch_embed(inp_img)             # [L, B, dim, H, W]

        # ── 3. Kalman feature filtering ─────────────────────────────────────
        #    Replaces heuristic repeat(T) with MMSE-optimal temporal fusion.
        inp_enc, kalman_gains = self.kalman_filter(inp_enc, cloud_maps)
        # inp_enc:      [L, B, dim, H, W]  – Kalman-filtered features
        # kalman_gains: [L, B, 1,   H, W]  – can be logged for analysis

        # ── 4. Encoder (exactly as original, T=L steps flow through) ────────
        out_enc_level1 = self.encoder_level1(inp_enc)
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)
        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        # ── 5. Decoder ───────────────────────────────────────────────────────
        out_dec_level3 = self.decoder_level3(out_enc_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], dim=2)
        inp_dec_level2 = self.lif_level2(inp_dec_level2)
        inp_dec_level2 = self.reduce_conv_level2(inp_dec_level2)
        inp_dec_level2 = self.reduce_bn_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], dim=2)
        inp_dec_level1 = self.lif_level1(inp_dec_level1)
        inp_dec_level1 = self.reduce_conv_level1(inp_dec_level1)
        inp_dec_level1 = self.reduce_bn_level1(inp_dec_level1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        out_dec_level1 = self.additional_sunet_level1(out_dec_level1)

        if self.use_refinement:
            out_dec_level1 = self.refinement_blocks(out_dec_level1)

        # ── 6. Output: clearness-weighted temporal aggregation ───────────────
        #    Frames with clearer sky contribute more to the final prediction.
        #    weights: [L, B, 1, H, W] (same normalised clearness as above,
        #             but now applied to decoder feature maps)
        aggregated = (out_dec_level1 * weights).sum(0)   # [B, dim, H, W]

        restored = self.output(aggregated) + short        # [B, C, H, W]
        return restored, cloud_maps


# ─── Factory functions ────────────────────────────────────────────────────────
def model(use_refinement: bool = False) -> VLIFNet:
    """Original single-image deraining model. Backward-compatible."""
    return VLIFNet(
        inp_channels=3,
        out_channels=3,
        dim=48,
        en_num_blocks=[4, 4, 8, 8],
        de_num_blocks=[2, 2, 2, 2],
        T=4,
        L=4,
        cloud_mode=False,
        use_refinement=use_refinement,
    ).cuda()


def model_cloud(L: int = 3, inp_channels: int = 3,
                use_refinement: bool = False) -> VLIFNet:
    """
    Multi-temporal cloud removal model.

    Parameters
    ----------
    L            : number of temporal images per sample (default 3 for Sen2 MTC)
    inp_channels : 3 for RGB, 4 for RGB+NIR
    """
    return VLIFNet(
        inp_channels=inp_channels,
        out_channels=3,
        dim=48,
        en_num_blocks=[4, 4, 8, 8],
        de_num_blocks=[2, 2, 2, 2],
        T=L,            # dual-time unification: SNN steps = temporal images
        L=L,
        cloud_mode=True,
        use_refinement=use_refinement,
    ).cuda()
