"""Stage-1 weak-supervision loss used by the paper training protocol."""
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================
def normalize_per_channel(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mean = x.mean(dim=(2, 3), keepdim=True)
    std = x.std(dim=(2, 3), keepdim=True).clamp_min(eps)
    return (x - mean) / std


def l2norm(x: torch.Tensor, dim: int = 1, eps: float = 1e-6) -> torch.Tensor:
    return x / (torch.norm(x, p=2, dim=dim, keepdim=True) + eps)


def reduce_channels_group_mean(feat: torch.Tensor, out_dim: int = 8) -> torch.Tensor:
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


class Stage1WeakPrototypeLoss(nn.Module):
    """
    Stage-1 weak-supervised loss for noisy HR masks.

    Components:
      1) confidence-weighted weak BCE + Dice
      2) lightweight prototype correction
      3) absent-water suppression
      4) edge-aware smoothness
      5) weak boundary consistency (avoid over-sharpening noisy mask boundary)
    """

    def __init__(
        self,
        warmup_epochs: int = 5,
        ramp_epochs: int = 10,
        beta_max: float = 0.25,
        tau_pos: float = 0.90,
        tau_neg: float = 0.10,
        proto_temperature: float = 0.15,
        semantic_dim: int = 8,
        min_pos_points: int = 6,
        min_neg_points: int = 12,
        presence_threshold: float = 0.003,
        lambda_proto: float = 0.30,
        lambda_absent: float = 0.20,
        lambda_smooth: float = 0.05,
        lambda_boundary: float = 0.05,
    ):
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.ramp_epochs = ramp_epochs
        self.beta_max = beta_max
        self.tau_pos = tau_pos
        self.tau_neg = tau_neg
        self.proto_temperature = proto_temperature
        self.semantic_dim = semantic_dim
        self.min_pos_points = min_pos_points
        self.min_neg_points = min_neg_points
        self.presence_threshold = presence_threshold
        self.lambda_proto = lambda_proto
        self.lambda_absent = lambda_absent
        self.lambda_smooth = lambda_smooth
        self.lambda_boundary = lambda_boundary

    def _ramp(self, epoch: int) -> float:
        if epoch < self.warmup_epochs:
            return 0.0
        x = min(epoch - self.warmup_epochs + 1, self.ramp_epochs)
        return float(x) / float(max(self.ramp_epochs, 1))

    def build_joint_features(self, image: torch.Tensor, prob: torch.Tensor, semantic_feat: torch.Tensor) -> torch.Tensor:
        h, w = prob.shape[-2:]
        img_n = normalize_per_channel(image)
        if semantic_feat.shape[-2:] != (h, w):
            semantic_feat = F.interpolate(semantic_feat, size=(h, w), mode="bilinear", align_corners=False)
        sem = reduce_channels_group_mean(semantic_feat, out_dim=self.semantic_dim)
        sem = normalize_per_channel(sem)
        joint = torch.cat([img_n, sem, prob], dim=1)
        return l2norm(joint, dim=1)

    def weighted_soft_dice_loss(self, logits: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        inter = (weights * probs * targets).sum(dim=(1, 2, 3))
        den = (weights * probs).sum(dim=(1, 2, 3)) + (weights * targets).sum(dim=(1, 2, 3))
        dice = (2.0 * inter + eps) / (den + eps)
        return 1.0 - dice.mean()

    def weak_supervised_loss(self, logits: torch.Tensor, mask: torch.Tensor, conf: torch.Tensor, epoch: int):
        prob_detach = torch.sigmoid(logits).detach()
        ramp = self._ramp(epoch)
        beta = self.beta_max * ramp

        # let model slightly correct noisy weak mask after warmup
        soft_target = (1.0 - beta) * mask + beta * prob_detach

        reliability = (1.0 - torch.abs(prob_detach - mask)).clamp(0.15, 1.0)
        pos_ratio = mask.mean(dim=(1, 2, 3), keepdim=True)
        pos_w = ((1.0 - pos_ratio) / (pos_ratio + 1e-6)).clamp(1.0, 6.0)
        cls_w = torch.where(mask > 0.5, pos_w, torch.ones_like(mask))

        weights = conf * reliability * cls_w

        bce = F.binary_cross_entropy_with_logits(logits, soft_target, reduction="none")
        loss_bce = (bce * weights).sum() / weights.sum().clamp_min(1.0)
        loss_dice = self.weighted_soft_dice_loss(logits, soft_target, weights=weights)
        return loss_bce, loss_dice, soft_target, prob_detach, weights

    @torch.no_grad()
    def extract_prototypes(self, joint_feat: torch.Tensor, prob_detach: torch.Tensor, mask: torch.Tensor, conf: torch.Tensor):
        b, d, _, _ = joint_feat.shape
        feat_flat = joint_feat.flatten(2).transpose(1, 2)
        prob_flat = prob_detach.flatten(2).squeeze(1)
        mask_flat = mask.flatten(2).squeeze(1)
        conf_flat = conf.flatten(2).squeeze(1)

        pos_protos, neg_protos, water_valid = [], [], []
        for bi in range(b):
            x = feat_flat[bi]
            pos_idx = torch.where((mask_flat[bi] > 0.5) & (prob_flat[bi] > self.tau_pos) & (conf_flat[bi] > 0.5))[0]
            neg_idx = torch.where((mask_flat[bi] < 0.5) & (prob_flat[bi] < self.tau_neg) & (conf_flat[bi] > 0.5))[0]

            if pos_idx.numel() >= self.min_pos_points:
                xp = x[pos_idx]
                wp = (prob_flat[bi][pos_idx] * conf_flat[bi][pos_idx]).unsqueeze(1)
                proto_pos = (xp * wp).sum(dim=0) / wp.sum().clamp_min(1e-6)
                proto_pos = F.normalize(proto_pos.unsqueeze(0), dim=1).squeeze(0)
                water_valid.append(1.0)
            else:
                proto_pos = x.new_zeros(d)
                water_valid.append(0.0)

            if neg_idx.numel() >= self.min_neg_points:
                xn = x[neg_idx]
                wn = ((1.0 - prob_flat[bi][neg_idx]) * conf_flat[bi][neg_idx]).unsqueeze(1)
                proto_neg = (xn * wn).sum(dim=0) / wn.sum().clamp_min(1e-6)
                proto_neg = F.normalize(proto_neg.unsqueeze(0), dim=1).squeeze(0)
            else:
                fallback_idx = torch.where((prob_flat[bi] < 0.3) & (conf_flat[bi] > 0.3))[0]
                if fallback_idx.numel() > 0:
                    xn = x[fallback_idx]
                    proto_neg = F.normalize(xn.mean(dim=0, keepdim=True), dim=1).squeeze(0)
                else:
                    proto_neg = x.new_zeros(d)

            pos_protos.append(proto_pos)
            neg_protos.append(proto_neg)

        return torch.stack(pos_protos, dim=0), torch.stack(neg_protos, dim=0), joint_feat.new_tensor(water_valid)

    def prototype_correction_loss(
        self,
        logits: torch.Tensor,
        joint_feat: torch.Tensor,
        prob_detach: torch.Tensor,
        mask: torch.Tensor,
        conf: torch.Tensor,
        pos_protos: torch.Tensor,
        neg_protos: torch.Tensor,
        water_valid: torch.Tensor,
    ):
        q = F.normalize(joint_feat.flatten(2).transpose(1, 2), dim=-1)
        pos_protos = F.normalize(pos_protos, dim=-1)
        neg_protos = F.normalize(neg_protos, dim=-1)

        sim_pos = torch.einsum("bnd,bd->bn", q, pos_protos)
        sim_neg = torch.einsum("bnd,bd->bn", q, neg_protos)
        proto_prob = torch.sigmoid((sim_pos - sim_neg) / self.proto_temperature).detach()

        stage_prob = torch.sigmoid(logits).flatten(2).squeeze(1)
        mask_flat = mask.flatten(2).squeeze(1)
        prob_flat = prob_detach.flatten(2).squeeze(1)
        conf_flat = conf.flatten(2).squeeze(1)

        mid_mask = ((prob_flat > 0.30) & (prob_flat < 0.70)).float()
        disagree = (torch.abs(prob_flat - mask_flat) > 0.40).float()
        gate = torch.clamp(mid_mask + disagree, 0.0, 1.0) * conf_flat * water_valid[:, None]

        if gate.sum() == 0:
            return logits.new_tensor(0.0)

        loss = F.binary_cross_entropy(stage_prob, proto_prob, reduction="none")
        return (loss * gate).sum() / gate.sum().clamp_min(1.0)

    def absence_suppression_loss(self, logits: torch.Tensor, mask: torch.Tensor, conf: torch.Tensor, water_valid: torch.Tensor):
        mask_area = (mask * conf).sum(dim=(1, 2, 3)) / conf.sum(dim=(1, 2, 3)).clamp_min(1.0)
        absent = ((mask_area < self.presence_threshold) & (water_valid < 0.5)).float()
        if absent.sum() == 0:
            return logits.new_tensor(0.0)
        prob = torch.sigmoid(logits)
        target = torch.zeros_like(prob)
        loss = F.binary_cross_entropy(prob, target, reduction="none")
        weight = absent[:, None, None, None].expand_as(loss)
        return (loss * weight).sum() / weight.sum().clamp_min(1.0)

    def smoothness_loss(self, logits: torch.Tensor, image: torch.Tensor):
        prob = torch.sigmoid(logits)
        img_n = normalize_per_channel(image)
        dx_img = (img_n[:, :, :, 1:] - img_n[:, :, :, :-1]).pow(2).mean(dim=1, keepdim=True)
        dy_img = (img_n[:, :, 1:, :] - img_n[:, :, :-1, :]).pow(2).mean(dim=1, keepdim=True)
        wx = torch.exp(-dx_img / 0.18)
        wy = torch.exp(-dy_img / 0.18)
        dx_p = torch.abs(prob[:, :, :, 1:] - prob[:, :, :, :-1])
        dy_p = torch.abs(prob[:, :, 1:, :] - prob[:, :, :-1, :])
        return (wx * dx_p).mean() + (wy * dy_p).mean()

    def boundary_consistency_loss(self, logits: torch.Tensor, mask: torch.Tensor, boundary: torch.Tensor):
        if boundary.sum() < 1:
            return logits.new_tensor(0.0)
        prob = torch.sigmoid(logits)
        # weakly keep average boundary activation close to weak mask, without forcing exact hard edge
        loss = F.binary_cross_entropy(prob, mask, reduction="none")
        return (loss * boundary).sum() / boundary.sum().clamp_min(1.0)

    def forward(self, outputs: Dict[str, torch.Tensor], image: torch.Tensor, mask: torch.Tensor, conf: torch.Tensor, boundary: torch.Tensor, epoch: int) -> Dict[str, torch.Tensor]:
        logits = outputs["logits"]
        semantic_feat = outputs.get("cluster_feat", outputs["feat"])

        loss_bce, loss_dice, _, prob_detach, _ = self.weak_supervised_loss(logits, mask, conf, epoch)
        joint_feat = self.build_joint_features(image, prob_detach, semantic_feat)
        pos_protos, neg_protos, water_valid = self.extract_prototypes(joint_feat, prob_detach, mask, conf)

        ramp = self._ramp(epoch)
        loss_proto = self.prototype_correction_loss(logits, joint_feat, prob_detach, mask, conf, pos_protos, neg_protos, water_valid)
        loss_absent = self.absence_suppression_loss(logits, mask, conf, water_valid)
        loss_smooth = self.smoothness_loss(logits, image)
        loss_boundary = self.boundary_consistency_loss(logits, mask, boundary)

        loss_sup = loss_bce + loss_dice
        total = loss_sup + ramp * self.lambda_proto * loss_proto + self.lambda_absent * loss_absent + self.lambda_smooth * loss_smooth + self.lambda_boundary * loss_boundary

        return {
            "loss": total,
            "loss_sup": loss_sup.detach(),
            "loss_bce": loss_bce.detach(),
            "loss_dice": loss_dice.detach(),
            "loss_proto": loss_proto.detach(),
            "loss_absent": loss_absent.detach(),
            "loss_smooth": loss_smooth.detach(),
            "loss_boundary": loss_boundary.detach(),
            "water_valid_ratio": water_valid.mean().detach(),
            "conf_mean": conf.mean().detach(),
            "boundary_ratio": boundary.mean().detach(),
        }


