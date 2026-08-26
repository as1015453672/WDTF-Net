
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_per_channel(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mean = x.mean(dim=(2, 3), keepdim=True)
    std = x.std(dim=(2, 3), keepdim=True).clamp_min(eps)
    return (x - mean) / std


def l2norm(x: torch.Tensor, dim: int = 1, eps: float = 1e-6) -> torch.Tensor:
    return x / (torch.norm(x, p=2, dim=dim, keepdim=True) + eps)


def reduce_channels_group_mean(feat: torch.Tensor, out_dim: int = 4) -> torch.Tensor:
    b, c, h, w = feat.shape
    if out_dim >= c:
        return feat
    splits = torch.linspace(0, c, out_dim + 1, device=feat.device).long().tolist()
    outs = []
    for i in range(out_dim):
        s, e = splits[i], splits[i + 1]
        if e <= s:
            e = min(s + 1, c)
        outs.append(feat[:, s:e].mean(dim=1, keepdim=True))
    return torch.cat(outs, dim=1)


def sobel_features(gray: torch.Tensor) -> torch.Tensor:
    device, dtype = gray.device, gray.dtype
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=dtype, device=device).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=dtype, device=device).view(1, 1, 3, 3)
    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    mag = torch.sqrt(gx * gx + gy * gy + 1e-6)
    return torch.cat([gx, gy, mag], dim=1)


def local_std(gray: torch.Tensor, k: int = 5) -> torch.Tensor:
    pad = k // 2
    mean = F.avg_pool2d(gray, kernel_size=k, stride=1, padding=pad)
    mean2 = F.avg_pool2d(gray * gray, kernel_size=k, stride=1, padding=pad)
    var = (mean2 - mean * mean).clamp_min(1e-6)
    return torch.sqrt(var)


def make_coords(b: int, h: int, w: int, device, dtype) -> torch.Tensor:
    ys = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx, yy], dim=0).unsqueeze(0).repeat(b, 1, 1, 1)


def pad_prototypes(proto: torch.Tensor, target_k: int) -> torch.Tensor:
    if proto.shape[0] == target_k:
        return proto
    if proto.shape[0] == 0:
        return proto.new_zeros(target_k, proto.shape[1])
    if proto.shape[0] > target_k:
        return proto[:target_k]
    pad = proto[-1:].repeat(target_k - proto.shape[0], 1)
    return torch.cat([proto, pad], dim=0)


class Stage2AdaptiveGraphPrototypeLoss(nn.Module):
    """
    Simplified Stage-2 loss:
        1) Prototype-guided Alignment Loss (PA)
        2) Guidance Consistency Loss (GC)
        3) Edge-aware Smoothness Loss (SR)

    Notes:
    - Keep class name and most init args for compatibility with existing training code.
    - Old loss terms (uncertain / prior / absent / spec / batch_proto) are absorbed or disabled.
    - Returned dict still contains the old keys so hr_training.py can run without modification.
    """

    def __init__(
        self,
        num_water_proto: int = 2,
        num_nonwater_proto: int = 3,
        tau_water: float = 0.90,
        tau_nonwater: float = 0.10,
        uncertain_low: float = 0.40,
        uncertain_high: float = 0.60,
        mid_low: float = 0.25,
        mid_high: float = 0.75,
        presence_threshold: float = 0.003,
        min_proto_points: int = 10,
        max_points_per_class: int = 96,
        semantic_dim: int = 4,
        cluster_temperature: float = 0.18,
        proto_temperature: float = 0.12,
        cluster_iters: int = 2,
        uncertain_margin: float = 0.22,
        affinity_sigma_img: float = 0.18,   # kept for compatibility, unused
        affinity_sigma_coord: float = 0.40, # kept for compatibility, unused
        graph_num_samples: int = 64,        # kept for compatibility, unused
        lambda_proto: float = 1.0,
        lambda_uncertain: float = 0.10,     # kept for compatibility, absorbed into PA weight
        lambda_cons: float = 0.40,
        lambda_smooth: float = 0.08,
        lambda_spec: float = 0.04,          # kept for compatibility, unused
        lambda_prior: float = 0.22,
        lambda_absent: float = 0.32,
        lambda_batch_proto: float = 0.05,   # kept for compatibility, unused
        proto_uncertain_weight: float = 0.50,
        gc_stage1_weight: float = 0.70,
        gc_prior_weight: float = 0.30,
        gc_strong_weight: float = 1.00,
        gc_mid_weight: float = 0.35,
        nonwater_strong_weight: float = 1.00,
        enable_recovery: bool = False,
        recovery_proto_threshold: float = 0.72,
        recovery_weight: float = 0.20,
        lambda_boundary: float = 0.0,
        boundary_band_radius: int = 3,
        spectral_negative_gate: bool = False,
        spectral_ndwi_threshold: float = 0.0,
        spectral_water_negative_weight: float = 0.10,
        lambda_spectral_seed: float = 0.0,
        spectral_seed_threshold: float = 0.05,
    ):
        super().__init__()
        self.num_water_proto = num_water_proto
        self.num_nonwater_proto = num_nonwater_proto

        self.tau_water = tau_water
        self.tau_nonwater = tau_nonwater
        self.uncertain_low = uncertain_low
        self.uncertain_high = uncertain_high
        self.mid_low = mid_low
        self.mid_high = mid_high
        self.presence_threshold = presence_threshold

        self.min_proto_points = min_proto_points
        self.max_points_per_class = max_points_per_class
        self.semantic_dim = semantic_dim
        self.cluster_temperature = cluster_temperature
        self.proto_temperature = proto_temperature
        self.cluster_iters = cluster_iters
        self.uncertain_margin = uncertain_margin

        # simplified top-level weights
        self.lambda_proto = lambda_proto
        self.lambda_cons = lambda_cons
        self.lambda_smooth = lambda_smooth

        # internal GC / PA shaping weights
        self.lambda_prior = lambda_prior
        self.lambda_absent = lambda_absent
        self.lambda_uncertain = lambda_uncertain
        self.proto_uncertain_weight = proto_uncertain_weight
        self.gc_stage1_weight = gc_stage1_weight
        self.gc_prior_weight = gc_prior_weight
        self.gc_strong_weight = gc_strong_weight
        self.gc_mid_weight = gc_mid_weight
        self.nonwater_strong_weight = nonwater_strong_weight
        # Optional experimental mode.  It never consumes manual labels: a
        # candidate must be weak-label negative, low-confidence in Stage 1,
        # and strongly supported by the water/non-water prototype contrast.
        self.enable_recovery = enable_recovery
        self.recovery_proto_threshold = recovery_proto_threshold
        self.recovery_weight = recovery_weight
        self.lambda_boundary = lambda_boundary
        self.boundary_band_radius = boundary_band_radius
        self.spectral_negative_gate = spectral_negative_gate
        self.spectral_ndwi_threshold = spectral_ndwi_threshold
        self.spectral_water_negative_weight = spectral_water_negative_weight
        self.lambda_spectral_seed = lambda_spectral_seed
        self.spectral_seed_threshold = spectral_seed_threshold

    @torch.no_grad()
    def build_masks(self, stage1_prob: torch.Tensor):
        water = (stage1_prob > self.tau_water).float()
        nonwater = (stage1_prob < self.tau_nonwater).float()
        uncertain = ((stage1_prob >= self.uncertain_low) & (stage1_prob <= self.uncertain_high)).float()
        mid = ((stage1_prob >= self.mid_low) & (stage1_prob <= self.mid_high)).float()
        return water, nonwater, uncertain, mid

    def build_joint_features(self, image: torch.Tensor, prob: torch.Tensor, semantic_feat: torch.Tensor):
        b, _, h, w = image.shape
        img_n = normalize_per_channel(image)
        gray = img_n.mean(dim=1, keepdim=True)
        tex = torch.cat([sobel_features(gray), local_std(gray, 5)], dim=1)
        coord = make_coords(b, h, w, image.device, image.dtype)
        if semantic_feat.shape[-2:] != (h, w):
            semantic_feat = F.interpolate(semantic_feat, size=(h, w), mode="bilinear", align_corners=False)
        sem = reduce_channels_group_mean(semantic_feat, out_dim=self.semantic_dim)
        sem = normalize_per_channel(sem)
        joint = torch.cat([img_n, tex, prob, coord, sem], dim=1)
        joint = l2norm(joint, dim=1)
        return joint, coord

    @torch.no_grad()
    def upsample_prior(self, prior_lr: torch.Tensor, size_hw):
        if prior_lr is None:
            return None
        if prior_lr.ndim == 3:
            prior_lr = prior_lr.unsqueeze(1)
        prior_hr = F.interpolate(prior_lr.float(), size=size_hw, mode="bilinear", align_corners=False)
        return prior_hr.clamp(0.0, 1.0)

    @torch.no_grad()
    def determine_presence(self, water_mask: torch.Tensor, prior_hr: torch.Tensor):
        stage1_area = water_mask.mean(dim=(1, 2, 3))
        if prior_hr is None:
            prior_area = torch.zeros_like(stage1_area)
        else:
            prior_area = prior_hr.mean(dim=(1, 2, 3))
        presence_score = 0.65 * stage1_area + 0.35 * prior_area
        water_present = (presence_score > self.presence_threshold).float()
        return water_present, stage1_area, prior_area

    def _sample_indices(self, idx: torch.Tensor) -> torch.Tensor:
        if idx.numel() > self.max_points_per_class:
            perm = torch.randperm(idx.numel(), device=idx.device)[: self.max_points_per_class]
            idx = idx[perm]
        return idx

    def _weighted_fps_init(self, x: torch.Tensor, weight: torch.Tensor, k: int) -> torch.Tensor:
        m, d = x.shape
        if m == 0:
            return x.new_zeros(k, d)
        x_n = F.normalize(x, dim=1)
        weight = weight.clamp_min(1e-6)
        actual_k = min(k, m)
        first = torch.argmax(weight)
        centers = [x_n[first:first + 1]]
        dist = 1.0 - torch.mm(x_n, centers[0].t()).squeeze(1)
        for _ in range(1, actual_k):
            idx = torch.argmax(dist * weight)
            c = x_n[idx:idx + 1]
            centers.append(c)
            dist = torch.minimum(dist, 1.0 - torch.mm(x_n, c.t()).squeeze(1))
        centers = torch.cat(centers, dim=0)
        return pad_prototypes(centers, k)

    def _soft_cluster_single(self, x: torch.Tensor, weight: torch.Tensor, k: int) -> torch.Tensor:
        m, d = x.shape
        if m == 0:
            return x.new_zeros(k, d)
        x = F.normalize(x, dim=1)
        centers = self._weighted_fps_init(x, weight, k)
        centers = F.normalize(centers, dim=1)
        for _ in range(self.cluster_iters):
            sim = torch.mm(x, centers.t())
            q = F.softmax(sim / self.cluster_temperature, dim=1)
            q = q * weight.unsqueeze(1)
            denom = q.sum(dim=0, keepdim=False).unsqueeze(1).clamp_min(1e-6)
            centers = torch.mm(q.t(), x) / denom
            centers = F.normalize(centers, dim=1)
        return pad_prototypes(centers, k)

    def extract_multi_prototypes(self, joint_feat, stage1_prob, water_mask, nonwater_mask, water_present):
        b, d, h, w = joint_feat.shape
        feat_flat = joint_feat.flatten(2).transpose(1, 2)  # [B,N,D]
        prob_flat = stage1_prob.flatten(2).squeeze(1)
        water_flat = water_mask.flatten(2).squeeze(1)
        nonwater_flat = nonwater_mask.flatten(2).squeeze(1)

        water_protos, nonwater_protos = [], []
        water_valid = []

        for bi in range(b):
            x = feat_flat[bi]

            idx_w = self._sample_indices(torch.where(water_flat[bi] > 0.5)[0])
            xw = x[idx_w]
            ww = prob_flat[bi][idx_w] if idx_w.numel() > 0 else x.new_zeros(0)
            valid_w = (water_present[bi] > 0.5) and (xw.shape[0] >= self.min_proto_points)
            if valid_w:
                proto_w = self._soft_cluster_single(xw, ww, self.num_water_proto)
                water_valid.append(1.0)
            else:
                proto_w = x.new_zeros(self.num_water_proto, d)
                water_valid.append(0.0)

            idx_n = self._sample_indices(torch.where(nonwater_flat[bi] > 0.5)[0])
            xn = x[idx_n]
            wn = (1.0 - prob_flat[bi][idx_n]) if idx_n.numel() > 0 else x.new_zeros(0)
            if xn.shape[0] >= max(4, self.min_proto_points // 2):
                proto_n = self._soft_cluster_single(xn, wn, self.num_nonwater_proto)
            else:
                proto_n = x.new_zeros(self.num_nonwater_proto, d)

            water_protos.append(pad_prototypes(proto_w, self.num_water_proto))
            nonwater_protos.append(pad_prototypes(proto_n, self.num_nonwater_proto))

        water_protos = torch.stack(water_protos, dim=0)
        nonwater_protos = torch.stack(nonwater_protos, dim=0)
        water_valid = joint_feat.new_tensor(water_valid)
        return water_protos, nonwater_protos, water_valid

    def compute_proto_scores(self, query_feat, water_protos, nonwater_protos, water_valid):
        q = F.normalize(query_feat.flatten(2).transpose(1, 2), dim=-1)  # [B,N,D]
        wp = F.normalize(water_protos, dim=-1)
        np_ = F.normalize(nonwater_protos, dim=-1)

        sim_w = torch.einsum("bnd,bkd->bnk", q, wp)
        sim_n = torch.einsum("bnd,bkd->bnk", q, np_)

        score_w = torch.logsumexp(sim_w / self.proto_temperature, dim=-1)
        score_n = torch.logsumexp(sim_n / self.proto_temperature, dim=-1)

        # disable water score for absent-water images
        score_w = score_w + (water_valid[:, None] - 1.0) * 1e4
        return score_w, score_n

    def prototype_alignment_loss(
        self,
        stage2_logits: torch.Tensor,
        score_w: torch.Tensor,
        score_n: torch.Tensor,
        water_mask: torch.Tensor,
        nonwater_mask: torch.Tensor,
        uncertain_mask: torch.Tensor,
        water_valid: torch.Tensor,
        recovery_candidates: torch.Tensor = None,
        nonwater_weight: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Unified prototype-guided alignment:
            - strong water/nonwater regions use hard labels
            - uncertain regions use prototype soft targets if margin is confident
        """
        stage2_prob = torch.sigmoid(stage2_logits).flatten(2).squeeze(1)  # [B,N]
        water_flat = water_mask.flatten(2).squeeze(1)
        nonwater_flat = nonwater_mask.flatten(2).squeeze(1)
        uncertain_flat = uncertain_mask.flatten(2).squeeze(1)

        proto_prob = torch.sigmoid((score_w - score_n) / self.proto_temperature).detach()
        margin = (score_w - score_n).abs().detach()

        strong_target = water_flat  # water->1, nonwater->0
        if recovery_candidates is None:
            recovery_flat = torch.zeros_like(nonwater_flat)
        else:
            recovery_flat = recovery_candidates.flatten(2).squeeze(1)
        if nonwater_weight is None:
            nonwater_weight_flat = self.nonwater_strong_weight
        else:
            nonwater_weight_flat = nonwater_weight.flatten(2).squeeze(1)
        strong_weight = water_flat * water_valid[:, None] + \
                        nonwater_weight_flat * nonwater_flat * (1.0 - recovery_flat)
        strong_loss = F.binary_cross_entropy(stage2_prob, strong_target, reduction="none")

        uncertain_gate = (margin > self.uncertain_margin).float() * uncertain_flat * water_valid[:, None]
        uncertain_loss = F.binary_cross_entropy(stage2_prob, proto_prob, reduction="none")

        recovery_loss = F.binary_cross_entropy(stage2_prob, torch.ones_like(stage2_prob), reduction="none")
        num = (strong_loss * strong_weight).sum() + self.proto_uncertain_weight * (uncertain_loss * uncertain_gate).sum() + self.recovery_weight * (recovery_loss * recovery_flat).sum()
        den = strong_weight.sum() + self.proto_uncertain_weight * uncertain_gate.sum() + self.recovery_weight * recovery_flat.sum()
        return num / den.clamp_min(1.0)

    def guidance_consistency_loss(
        self,
        stage2_logits: torch.Tensor,
        stage1_prob: torch.Tensor,
        prior_hr: torch.Tensor,
        water_present: torch.Tensor,
        water_mask: torch.Tensor,
        nonwater_mask: torch.Tensor,
        mid_mask: torch.Tensor,
        recovery_candidates: torch.Tensor = None,
        nonwater_weight: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Merge:
            - stage1 consistency
            - LR prior consistency
            - absent-water suppression
        """
        stage2_prob = torch.sigmoid(stage2_logits)
        strong_support = ((water_mask + nonwater_mask) > 0).float()
        if nonwater_weight is None:
            nonwater_weight = self.nonwater_strong_weight
        strong = water_mask * water_present[:, None, None, None] + nonwater_weight * nonwater_mask
        if recovery_candidates is not None:
            strong = strong * (1.0 - recovery_candidates)
            strong_support = strong_support * (1.0 - recovery_candidates)
        weak = (mid_mask * (1.0 - strong_support)).float()

        # stage1 target
        target_stage1 = stage1_prob.detach()
        weight_stage1 = self.gc_strong_weight * strong + self.gc_mid_weight * weak
        loss_stage1 = F.binary_cross_entropy(stage2_prob, target_stage1, reduction="none")

        # prior / absent target
        if prior_hr is None:
            target_prior = torch.zeros_like(stage2_prob)
            conf_prior = torch.ones_like(stage2_prob) * 0.5
        elif self.enable_recovery:
            # Weak products are treated as positive-only evidence here.  Their
            # zeros may include omitted water and must not veto a prototype-
            # supported recovery candidate.
            target_prior = torch.ones_like(stage2_prob)
            conf_prior = (prior_hr >= 0.5).float()
        else:
            conf_prior = (2.0 * torch.abs(prior_hr - 0.5)).clamp(0.0, 1.0)
            target_prior = prior_hr

        # absent-water images: strongly suppress positives
        absent = (1.0 - water_present)[:, None, None, None]
        target_prior = water_present[:, None, None, None] * target_prior + absent * torch.zeros_like(target_prior)
        conf_prior = water_present[:, None, None, None] * conf_prior + absent * (1.0 + self.lambda_absent) * torch.ones_like(conf_prior)

        loss_prior = F.binary_cross_entropy(stage2_prob, target_prior.detach(), reduction="none")

        num = self.gc_stage1_weight * (loss_stage1 * weight_stage1).sum() + \
              self.gc_prior_weight * self.lambda_prior * (loss_prior * conf_prior).sum()
        den = self.gc_stage1_weight * weight_stage1.sum() + \
              self.gc_prior_weight * self.lambda_prior * conf_prior.sum()
        return num / den.clamp_min(1.0)

    def smoothness_loss(self, stage2_logits: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        prob = torch.sigmoid(stage2_logits)
        img_n = normalize_per_channel(image)
        dx_img = (img_n[:, :, :, 1:] - img_n[:, :, :, :-1]).pow(2).mean(dim=1, keepdim=True)
        dy_img = (img_n[:, :, 1:, :] - img_n[:, :, :-1, :]).pow(2).mean(dim=1, keepdim=True)
        wx = torch.exp(-dx_img / 0.18)
        wy = torch.exp(-dy_img / 0.18)
        dx_p = torch.abs(prob[:, :, :, 1:] - prob[:, :, :, :-1])
        dy_p = torch.abs(prob[:, :, 1:, :] - prob[:, :, :-1, :])
        return (wx * dx_p).mean() + (wy * dy_p).mean()

    @torch.no_grad()
    def stage1_boundary_band(self, stage1_prob: torch.Tensor) -> torch.Tensor:
        """A fixed narrow band around the Stage-1 contour, not a label-derived mask."""
        binary = (stage1_prob >= 0.5).float()
        radius = self.boundary_band_radius
        kernel = 2 * radius + 1
        dilated = F.max_pool2d(binary, kernel, stride=1, padding=radius)
        eroded = 1.0 - F.max_pool2d(1.0 - binary, kernel, stride=1, padding=radius)
        return (dilated - eroded).clamp(0.0, 1.0)

    def boundary_image_alignment_loss(
        self, stage2_logits: torch.Tensor, image: torch.Tensor, stage1_prob: torch.Tensor
    ) -> torch.Tensor:
        """Match output contours to image edges only near an existing Stage-1 contour.

        This deliberately cannot introduce a full-image edge prior: shadows,
        roads and shore texture outside the narrow Stage-1 band have zero
        weight.  It is intended as a small geometric refinement on top of the
        relaxed-negative setting.
        """
        band = self.stage1_boundary_band(stage1_prob)
        prob = torch.sigmoid(stage2_logits)
        pred_dx = F.pad(torch.abs(prob[:, :, :, 1:] - prob[:, :, :, :-1]), (0, 1, 0, 0))
        pred_dy = F.pad(torch.abs(prob[:, :, 1:, :] - prob[:, :, :-1, :]), (0, 0, 0, 1))
        pred_edge = torch.sqrt(pred_dx.square() + pred_dy.square() + 1e-6)
        gray = normalize_per_channel(image).mean(dim=1, keepdim=True)
        image_edge = sobel_features(gray)[:, 2:3]
        scale = image_edge.flatten(2).amax(dim=2, keepdim=True).view(-1, 1, 1, 1).clamp_min(1e-6)
        image_edge = (image_edge / scale).clamp(0.0, 1.0)
        return ((pred_edge - image_edge).square() * band).sum() / band.sum().clamp_min(1.0)

    @torch.no_grad()
    def spectral_nonwater_weight(self, image: torch.Tensor) -> Optional[torch.Tensor]:
        """Use Sentinel green/NIR evidence only to protect relaxed negatives.

        Pixels with NDWI above the fixed training-data threshold receive the
        lower negative anchor; all other weak-negative pixels retain the
        S2-B weight. No manual label is involved.
        """
        if not self.spectral_negative_gate or image.shape[1] < 4:
            return None
        green, nir = image[:, 1:2], image[:, 3:4]
        ndwi = (green - nir) / (green + nir + 1e-6)
        likely_water = (ndwi >= self.spectral_ndwi_threshold).float()
        return self.nonwater_strong_weight * (1.0 - likely_water) + \
               self.spectral_water_negative_weight * likely_water

    @torch.no_grad()
    def spectral_recovery_seeds(self, image: torch.Tensor, stage1_prob: torch.Tensor, prior_hr: torch.Tensor, water_valid: torch.Tensor) -> torch.Tensor:
        if self.lambda_spectral_seed <= 0 or image.shape[1] < 4 or prior_hr is None:
            return torch.zeros_like(stage1_prob)
        green, nir = image[:, 1:2], image[:, 3:4]
        ndwi = (green - nir) / (green + nir + 1e-6)
        return ((ndwi >= self.spectral_seed_threshold) & (stage1_prob < 0.5) & (prior_hr < 0.5) &
                (water_valid[:, None, None, None] > 0.5)).float()

    def forward(self, image, stage1_out, stage2_out, prior_lr=None, compute_spec=True):
        del compute_spec  # kept for compatibility

        stage1_prob = torch.sigmoid(stage1_out["logits"]).detach()
        stage2_logits = stage2_out["logits"]
        stage1_sem = stage1_out.get("cluster_feat", stage1_out["feat"]).detach()
        stage2_sem = stage2_out.get("cluster_feat", stage2_out["feat"])

        water_mask, nonwater_mask, uncertain_mask, mid_mask = self.build_masks(stage1_prob)
        prior_hr = self.upsample_prior(prior_lr, stage1_prob.shape[-2:])
        water_present, stage1_area, prior_area = self.determine_presence(water_mask, prior_hr)

        joint_stage1, coord = self.build_joint_features(image, stage1_prob, stage1_sem)
        stage2_prob = torch.sigmoid(stage2_logits)
        joint_stage2, _ = self.build_joint_features(image, stage2_prob, stage2_sem)

        water_protos, nonwater_protos, water_valid = self.extract_multi_prototypes(
            joint_stage1, stage1_prob, water_mask, nonwater_mask, water_present
        )
        score_w, score_n = self.compute_proto_scores(
            joint_stage2, water_protos, nonwater_protos, water_valid
        )
        proto_prob = torch.sigmoid((score_w - score_n) / self.proto_temperature).view_as(stage1_prob)
        proto_margin = (score_w - score_n).abs().view_as(stage1_prob)
        if self.enable_recovery and prior_hr is not None:
            recovery_candidates = ((prior_hr < 0.5) & (stage1_prob < self.uncertain_low) &
                                   (proto_prob > self.recovery_proto_threshold) &
                                   (proto_margin > self.uncertain_margin) &
                                   (water_valid[:, None, None, None] > 0.5)).float()
        else:
            recovery_candidates = torch.zeros_like(stage1_prob)
        nonwater_weight = self.spectral_nonwater_weight(image)
        spectral_seed = self.spectral_recovery_seeds(image, stage1_prob, prior_hr, water_valid)

        loss_pa = self.prototype_alignment_loss(
            stage2_logits, score_w, score_n,
            water_mask, nonwater_mask, uncertain_mask, water_valid, recovery_candidates, nonwater_weight
        )
        loss_gc = self.guidance_consistency_loss(
            stage2_logits, stage1_prob, prior_hr, water_present,
            water_mask, nonwater_mask, mid_mask, recovery_candidates, nonwater_weight
        )
        loss_sr = self.smoothness_loss(stage2_logits, image)
        loss_boundary = self.boundary_image_alignment_loss(stage2_logits, image, stage1_prob) \
            if self.lambda_boundary > 0 else stage2_logits.new_tensor(0.0)
        seed_loss = F.binary_cross_entropy(torch.sigmoid(stage2_logits), torch.ones_like(stage2_logits), reduction="none")
        loss_spectral_seed = (seed_loss * spectral_seed).sum() / spectral_seed.sum().clamp_min(1.0)

        total = (
            self.lambda_proto * loss_pa
            + self.lambda_cons * loss_gc
            + self.lambda_smooth * loss_sr
            + self.lambda_boundary * loss_boundary
            + self.lambda_spectral_seed * loss_spectral_seed
        )

        # compatibility: keep old logging keys
        zero = stage2_logits.new_tensor(0.0)
        return {
            "loss": total,
            "loss_proto": loss_pa.detach(),      # now means PA
            "loss_uncertain": zero.detach(),     # absorbed into PA
            "loss_cons": loss_gc.detach(),       # now means GC
            "loss_prior": zero.detach(),         # absorbed into GC
            "loss_absent": zero.detach(),        # absorbed into GC
            "loss_smooth": loss_sr.detach(),     # SR
            "loss_boundary": loss_boundary.detach(),
            "loss_spectral_seed": loss_spectral_seed.detach(),
            "loss_spec": zero.detach(),          # removed
            "loss_batch_proto": zero.detach(),   # removed
            "water_ratio": water_mask.mean().detach(),
            "nonwater_ratio": nonwater_mask.mean().detach(),
            "uncertain_ratio": uncertain_mask.mean().detach(),
            "water_present_ratio": water_present.mean().detach(),
            "stage1_area_mean": stage1_area.mean().detach(),
            "prior_area_mean": prior_area.mean().detach(),
            "recovery_candidate_ratio": recovery_candidates.mean().detach(),
        }


