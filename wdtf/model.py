import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class WDTFNetConfig:
    in_channels: int = 3
    num_classes: int = 1
    base_channels: int = 32
    window_size: int = 7  # compatibility only
    max_residual_offset: float = 0.35
    use_skeleton_head: bool = False
    adapter_reduction: int = 4
    freeze_backbone_in_stage2: bool = True
    num_templates: int = 6
    wavelet_basis: str = "haar"


class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNAct(in_ch, out_ch, 3, 1, 1),
            ConvBNAct(out_ch, out_ch, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.conv = DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class FixedDWT2d(nn.Module):
    """Fixed four-band analysis transform used by the frequency modules.

    ``haar`` preserves the original implementation exactly.  The other modes
    exist solely for controlled basis ablations and keep four output bands at
    half spatial resolution.
    """
    def __init__(self, basis: str = "haar"):
        super().__init__()
        self.basis = basis
        if basis == "haar":
            ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.float32)
            lh = torch.tensor([[-0.5, -0.5], [0.5, 0.5]], dtype=torch.float32)
            hl = torch.tensor([[-0.5, 0.5], [-0.5, 0.5]], dtype=torch.float32)
            hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]], dtype=torch.float32)
            padding = 0
        elif basis == "db2":
            root3 = math.sqrt(3.0)
            low = torch.tensor([(1 + root3), (3 + root3), (3 - root3), (1 - root3)], dtype=torch.float32) / (4 * math.sqrt(2.0))
            high = torch.tensor([-low[3], low[2], -low[1], low[0]], dtype=torch.float32)
            ll, lh, hl, hh = torch.outer(low, low), torch.outer(high, low), torch.outer(low, high), torch.outer(high, high)
            padding = 1
        elif basis == "sobel":
            ll = torch.ones((3, 3), dtype=torch.float32) / 3.0
            lh = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]) / 4.0
            hl = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]) / 4.0
            hh = torch.tensor([[0., -1., 0.], [-1., 4., -1.], [0., -1., 0.]]) / 4.0
            padding = 1
        elif basis == "lowpass":
            ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.float32)
            lh = torch.zeros((2, 2), dtype=torch.float32)
            hl = torch.zeros((2, 2), dtype=torch.float32)
            hh = torch.zeros((2, 2), dtype=torch.float32)
            padding = 0
        else:
            raise ValueError(f"Unsupported analysis basis: {basis}")
        filt = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)
        self.register_buffer("filt", filt)
        self.padding = padding

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        b, c, h, w = x.shape
        if h < 2 or w < 2:
            return x, x, x, x
        weight = self.filt.repeat(c, 1, 1, 1)
        y = F.conv2d(x, weight, stride=2, padding=self.padding, groups=c)
        y = y.view(b, c, 4, y.shape[-2], y.shape[-1])
        return y[:, :, 0], y[:, :, 1], y[:, :, 2], y[:, :, 3]


class HaarDWT2d(FixedDWT2d):
    """Backward-compatible name for the original Haar transform."""
    def __init__(self):
        super().__init__("haar")


class WaveletPriorAnalyzer(nn.Module):
    def __init__(self, in_ch: int, num_templates: int = 6, basis: str = "haar"):
        super().__init__()
        self.dwt = FixedDWT2d(basis)
        self.pre = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(7, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.template_head = nn.Conv2d(32, num_templates, kernel_size=1)
        self.theta_head = nn.Conv2d(32, 1, kernel_size=1)
        self.rho_head = nn.Conv2d(32, 1, kernel_size=1)
        self.scale_head = nn.Conv2d(32, 1, kernel_size=1)
        self.bfreq_head = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x0 = self.pre(x)
        ll, lh, hl, hh = self.dwt(x0)

        e_h = lh.abs().mean(dim=1, keepdim=True)
        e_v = hl.abs().mean(dim=1, keepdim=True)
        e_d = hh.abs().mean(dim=1, keepdim=True)
        e_l = ll.abs().mean(dim=1, keepdim=True)

        def up(t: torch.Tensor) -> torch.Tensor:
            return F.interpolate(t, size=x.shape[-2:], mode="bilinear", align_corners=False)

        e_h = up(e_h)
        e_v = up(e_v)
        e_d = up(e_d)
        e_l = up(e_l)
        e_sum = e_h + e_v + e_d + 1e-6
        dir_hv = (e_h - e_v) / e_sum
        dir_diag = e_d / e_sum
        hi = (e_h + e_v + e_d) / (e_l + e_h + e_v + e_d + 1e-6)

        feat = torch.cat([e_l, e_h, e_v, e_d, dir_hv, dir_diag, hi], dim=1)
        feat = self.fuse(feat)

        alpha = F.softmax(self.template_head(feat), dim=1)
        theta = math.pi * torch.tanh(self.theta_head(feat))
        rho = torch.sigmoid(self.rho_head(feat))
        scale = 0.8 + 0.8 * torch.sigmoid(self.scale_head(feat))
        bfreq = torch.sigmoid(self.bfreq_head(feat))
        return {"alpha": alpha, "theta": theta, "rho": rho, "scale": scale, "bfreq": bfreq}


class WGDC2d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, num_templates: int = 6, max_residual_offset: float = 0.35):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.n_points = 9
        self.num_templates = num_templates
        self.max_residual_offset = max_residual_offset

        base = torch.tensor([
            [-1.0, -1.0], [0.0, -1.0], [1.0, -1.0],
            [-1.0, 0.0], [0.0, 0.0], [1.0, 0.0],
            [-1.0, 1.0], [0.0, 1.0], [1.0, 1.0],
        ], dtype=torch.float32)
        large = 1.5 * base
        horiz = torch.tensor([
            [-1.5, 0.0], [-1.0, 0.0], [-0.5, 0.0],
            [-0.25, 0.0], [0.0, 0.0], [0.25, 0.0],
            [0.5, 0.0], [1.0, 0.0], [1.5, 0.0],
        ], dtype=torch.float32)
        vert = torch.tensor([
            [0.0, -1.5], [0.0, -1.0], [0.0, -0.5],
            [0.0, -0.25], [0.0, 0.0], [0.0, 0.25],
            [0.0, 0.5], [0.0, 1.0], [0.0, 1.5],
        ], dtype=torch.float32)
        diag = torch.tensor([
            [-1.2, -1.2], [-0.8, -0.8], [-0.4, -0.4],
            [-0.2, -0.2], [0.0, 0.0], [0.2, 0.2],
            [0.4, 0.4], [0.8, 0.8], [1.2, 1.2],
        ], dtype=torch.float32)
        anti = torch.tensor([
            [1.2, -1.2], [0.8, -0.8], [0.4, -0.4],
            [0.2, -0.2], [0.0, 0.0], [-0.2, 0.2],
            [-0.4, 0.4], [-0.8, 0.8], [-1.2, 1.2],
        ], dtype=torch.float32)
        bank = torch.stack([base, large, horiz, vert, diag, anti], dim=0)
        if num_templates != 6:
            raise ValueError("This implementation expects num_templates=6.")
        self.register_buffer("template_bank", bank)

        self.residual_head = nn.Sequential(
            nn.Conv2d(in_ch + 4, in_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, self.n_points * 2, kernel_size=3, padding=1),
        )
        self.mask_head = nn.Sequential(
            nn.Conv2d(in_ch + 3, in_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, self.n_points, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch * self.n_points, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def _make_normalized_base_coords(self, b: int, h: int, w: int, device, dtype) -> torch.Tensor:
        ys = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
        xs = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        base = torch.stack([xx, yy], dim=-1)
        return base.unsqueeze(0).repeat(b, 1, 1, 1)

    @staticmethod
    def _pixel_to_normalized(dx: torch.Tensor, dy: torch.Tensor, h: int, w: int) -> Tuple[torch.Tensor, torch.Tensor]:
        dx_n = dx * (2.0 / max(w - 1, 1))
        dy_n = dy * (2.0 / max(h - 1, 1))
        return dx_n, dy_n

    def forward(self, x: torch.Tensor, prior: Dict[str, torch.Tensor]) -> torch.Tensor:
        b, c, h, w = x.shape
        alpha = prior["alpha"]
        theta = prior["theta"]
        rho = prior["rho"]
        scale = prior["scale"]
        bfreq = prior["bfreq"]

        bank = self.template_bank.to(device=x.device, dtype=x.dtype)
        base_offsets = torch.einsum("bkhw,knd->bndhw", alpha, bank).contiguous()

        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        px = base_offsets[:, :, 0]
        py = base_offsets[:, :, 1]
        qx = scale * (px * cos_t - py * sin_t)
        qy = scale * (px * sin_t + py * cos_t)
        base_offsets = torch.stack([qx, qy], dim=2)

        residual_in = torch.cat([x, theta, rho, scale, bfreq], dim=1)
        residual = self.max_residual_offset * torch.tanh(self.residual_head(residual_in))
        residual = residual.view(b, self.n_points, 2, h, w)

        mask_in = torch.cat([x, rho, scale, bfreq], dim=1)
        masks = self.mask_head(mask_in).view(b, self.n_points, 1, h, w)

        final_offsets = base_offsets + residual
        base_coords = self._make_normalized_base_coords(b, h, w, x.device, x.dtype)

        sampled_list: List[torch.Tensor] = []
        for n in range(self.n_points):
            dx = final_offsets[:, n, 0]
            dy = final_offsets[:, n, 1]
            dx_n, dy_n = self._pixel_to_normalized(dx, dy, h, w)
            grid = base_coords.clone()
            grid[..., 0] = torch.clamp(grid[..., 0] + dx_n, -1.0, 1.0)
            grid[..., 1] = torch.clamp(grid[..., 1] + dy_n, -1.0, 1.0)
            sampled = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=True)
            sampled = sampled * masks[:, n]
            sampled_list.append(sampled)

        out = torch.cat(sampled_list, dim=1)
        return self.proj(out)


class FrequencyDecoderGate(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        )
        self.refine = ConvBNAct(channels, channels, 3, 1, 1)

    def forward(self, x: torch.Tensor, rho: torch.Tensor, bfreq: torch.Tensor) -> torch.Tensor:
        freq = torch.cat([rho, bfreq], dim=1)
        if freq.shape[-2:] != x.shape[-2:]:
            freq = F.interpolate(freq, size=x.shape[-2:], mode="bilinear", align_corners=False)
        g = self.gate(freq)
        return self.refine(x * (1.0 + g))


class WaveletAdapter(nn.Module):
    def __init__(self, channels: int, reduction: int = 4, basis: str = "haar"):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.down = nn.Conv2d(channels, hidden, 1, bias=False)
        self.dw = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False)
        self.act = nn.GELU()
        self.up = nn.Conv2d(hidden, channels, 1, bias=False)
        self.alpha = nn.Parameter(torch.zeros(1))

        self.dwt = FixedDWT2d(basis)
        self.wavelet_scale = nn.Parameter(torch.tensor(1.0))
        self.detail_scale = nn.Parameter(torch.tensor(1.0))

    def _wavelet_gate(self, x: torch.Tensor) -> torch.Tensor:
        ll, lh, hl, hh = self.dwt(x)
        if ll.shape[-2:] != x.shape[-2:]:
            def up(t: torch.Tensor) -> torch.Tensor:
                return F.interpolate(t, size=x.shape[-2:], mode="bilinear", align_corners=False)
            e_l = up(ll.abs().mean(dim=1, keepdim=True))
            e_h = up(lh.abs().mean(dim=1, keepdim=True))
            e_v = up(hl.abs().mean(dim=1, keepdim=True))
            e_d = up(hh.abs().mean(dim=1, keepdim=True))
        else:
            e_l = ll.abs().mean(dim=1, keepdim=True)
            e_h = lh.abs().mean(dim=1, keepdim=True)
            e_v = hl.abs().mean(dim=1, keepdim=True)
            e_d = hh.abs().mean(dim=1, keepdim=True)

        hi = (e_h + e_v + e_d) / (e_l + e_h + e_v + e_d + 1e-6)
        ani = torch.abs(e_h - e_v) / (e_h + e_v + e_d + 1e-6)
        edge = (e_h + e_v).clamp(min=0.0)
        gate = torch.sigmoid(self.wavelet_scale * hi + self.detail_scale * 0.5 * (ani + edge))
        return gate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self._wavelet_gate(x)
        xw = x * (1.0 + gate)
        y = self.up(self.act(self.dw(self.down(xw))))
        return x + self.alpha * y


class WDTFBlock(nn.Module):
    def __init__(self, channels: int, window_size: int = 7, max_residual_offset: float = 0.35, num_templates: int = 6, wavelet_basis: str = "haar"):
        super().__init__()
        self.prior = WaveletPriorAnalyzer(channels, num_templates=num_templates, basis=wavelet_basis)
        self.wgdc = WGDC2d(channels, channels, num_templates=num_templates, max_residual_offset=max_residual_offset)
        self.refine = DoubleConv(channels, channels)
        self.res_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        prior = self.prior(x)
        y = self.wgdc(x, prior)
        y = self.refine(y)
        y = x + self.res_scale * y
        aux = {
            "alpha": prior["alpha"],
            "theta": prior["theta"],
            "rho": prior["rho"],
            "scale": prior["scale"],
            "bfreq": prior["bfreq"],
        }
        return y, aux


class WDTFNetOptimized(nn.Module):
    def __init__(self, cfg: WDTFNetConfig):
        super().__init__()
        c = cfg.base_channels
        self.cfg = cfg

        self.stem = DoubleConv(cfg.in_channels, c)
        self.down1 = Down(c, c * 2)
        self.down2 = Down(c * 2, c * 4)
        self.down3 = Down(c * 4, c * 8)

        self.wdtf2 = WDTFBlock(c * 2, window_size=cfg.window_size, max_residual_offset=cfg.max_residual_offset, num_templates=cfg.num_templates, wavelet_basis=cfg.wavelet_basis)
        self.wdtf3 = WDTFBlock(c * 4, window_size=cfg.window_size, max_residual_offset=cfg.max_residual_offset, num_templates=cfg.num_templates, wavelet_basis=cfg.wavelet_basis)
        self.wdtf4 = WDTFBlock(c * 8, window_size=cfg.window_size, max_residual_offset=cfg.max_residual_offset, num_templates=cfg.num_templates, wavelet_basis=cfg.wavelet_basis)

        self.up3 = Up(c * 8, c * 4, c * 4)
        self.up2 = Up(c * 4, c * 2, c * 2)
        self.up1 = Up(c * 2, c, c)

        self.dec_gate3 = FrequencyDecoderGate(c * 4)
        self.dec_gate2 = FrequencyDecoderGate(c * 2)
        self.dec_gate1 = FrequencyDecoderGate(c)

        self.adapter_wdtf2 = WaveletAdapter(c * 2, reduction=cfg.adapter_reduction, basis=cfg.wavelet_basis)
        self.adapter_wdtf3 = WaveletAdapter(c * 4, reduction=cfg.adapter_reduction, basis=cfg.wavelet_basis)
        self.adapter_wdtf4 = WaveletAdapter(c * 8, reduction=cfg.adapter_reduction, basis=cfg.wavelet_basis)
        self.adapter_up3 = WaveletAdapter(c * 4, reduction=cfg.adapter_reduction, basis=cfg.wavelet_basis)
        self.adapter_up2 = WaveletAdapter(c * 2, reduction=cfg.adapter_reduction, basis=cfg.wavelet_basis)

        self.seg_head = nn.Conv2d(c, cfg.num_classes, kernel_size=1)
        self.skel_head = nn.Conv2d(c, 1, kernel_size=1) if cfg.use_skeleton_head else None

    def forward(self, x: torch.Tensor, use_adapters: bool = False) -> dict:
        e1 = self.stem(x)
        e2 = self.down1(e1)
        e2, aux2 = self.wdtf2(e2)
        if use_adapters:
            e2 = self.adapter_wdtf2(e2)

        e3 = self.down2(e2)
        e3, aux3 = self.wdtf3(e3)
        if use_adapters:
            e3 = self.adapter_wdtf3(e3)

        e4 = self.down3(e3)
        e4, aux4 = self.wdtf4(e4)
        if use_adapters:
            e4 = self.adapter_wdtf4(e4)

        d3 = self.up3(e4, e3)
        d3 = self.dec_gate3(d3, aux3["rho"], aux3["bfreq"])
        if use_adapters:
            d3 = self.adapter_up3(d3)

        d2 = self.up2(d3, e2)
        d2 = self.dec_gate2(d2, aux2["rho"], aux2["bfreq"])
        if use_adapters:
            d2 = self.adapter_up2(d2)

        d1 = self.up1(d2, e1)
        d1 = self.dec_gate1(d1, aux2["rho"], aux2["bfreq"])

        out = {
            "logits": self.seg_head(d1),
            "feat": d1,  # 用于 stage2 的查询特征
            "cluster_feat": d2,  # 用于聚类的中层语义特征（比 d1 更稳一些）
            "aux": {"stage2": aux2, "stage3": aux3, "stage4": aux4},
            "use_adapters": use_adapters,
        }
        if self.skel_head is not None:
            out["skel_logits"] = self.skel_head(d1)
        return out

    def freeze_for_stage2(self, train_seg_head: bool = True, tune_geometry: bool = False) -> None:
        for name, p in self.named_parameters():
            p.requires_grad = False
            if "adapter_" in name:
                p.requires_grad = True
            if train_seg_head and name.startswith("seg_head"):
                p.requires_grad = True
            if tune_geometry and any(k in name for k in ["prior", "wgdc", "dec_gate"]):
                p.requires_grad = True

    def trainable_parameter_summary(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "ratio_pct": round(100.0 * trainable / max(total, 1), 3)}

    def set_stage2_train_mode(self, freeze_all_bn: bool = False) -> None:
        """Set Stage-2 training mode while protecting BatchNorm statistics.

        ``freeze_all_bn`` is useful when a small decoder tail is unfrozen: its
        affine weights may still learn, but running statistics should not drift
        on the small weak-label validation split.
        """
        self.train()
        for module in self.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                params = list(module.parameters(recurse=False))
                if freeze_all_bn or (params and not any(p.requires_grad for p in params)):
                    module.eval()


def dice_loss_with_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    num = 2.0 * (probs * targets).sum(dim=(1, 2, 3))
    den = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) + eps
    return 1.0 - (num + eps) / den


