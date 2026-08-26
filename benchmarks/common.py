import copy
import csv
import math
import os
import random
import time
from collections import deque
from dataclasses import asdict, dataclass
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from models.deeplabv3plus import DeepLabV3Plus
except ModuleNotFoundError:
    from deeplabv3plus import DeepLabV3Plus


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class AverageMeter:
    def __init__(self):
        self.sum = 0.0
        self.count = 0

    @property
    def avg(self):
        return self.sum / max(1, self.count)

    def update(self, val, n=1):
        self.sum += float(val) * n
        self.count += n


@dataclass
class TrainConfig:
    method: str = 'unimatch'
    root_dir: str = './data/train'
    save_dir: str = './experiment'
    backbone: str = 'resnet101'
    output_stride: int = 16
    pretrained_backbone: bool = True
    num_epochs: int = 60
    batch_size: int = 8
    num_workers: int = 4
    lr: float = 1e-4
    backbone_lr_mult: float = 1.0
    weight_decay: float = 1e-4
    ema_momentum: float = 0.99
    seed: int = 42
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    val_ratio: float = 0.1
    hr_divisor: float = 1024.0
    lr_divisor: float = 1024.0
    mask_threshold: float = 50.0
    conf_thresh: float = 0.95
    rank_conf_thresh: float = 0.7
    u2pl_conf_thresh: float = 0.95
    temperature: float = 0.5
    rampup_epochs: int = 15
    unsup_weight: float = 1.0
    rank_weight: float = 0.5
    edge_weight: float = 0.15
    proto_weight: float = 0.2
    neg_weight: float = 0.2
    rank_pairs: int = 2048
    rank_margin: float = 0.0
    u2pl_queue_size: int = 512
    u2pl_alpha0: float = 0.20
    u2pl_delta_p: float = 0.30
    u2pl_contrast_weight: float = 0.5
    u2pl_low_rank: int = 1
    u2pl_high_rank: int = 2
    u2pl_num_anchors: int = 256
    u2pl_num_negatives: int = 512
    contrast_temperature: float = 0.1


class PolyOptimizer(torch.optim.SGD):
    def __init__(self, params, lr, momentum=0.9, weight_decay=1e-4, max_iters=1000, power=0.9):
        super().__init__(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
        self.base_lrs = [g['lr'] for g in self.param_groups]
        self.max_iters = max_iters
        self.power = power
        self.iter = 0

    def step(self, closure=None):
        out = super().step(closure)
        self.iter += 1
        coef = (1 - min(self.iter, self.max_iters) / max(1, self.max_iters)) ** self.power
        for base_lr, group in zip(self.base_lrs, self.param_groups):
            group['lr'] = base_lr * coef
        return out


class PrototypeQueue:
    def __init__(self, feat_dim=256, max_size=512):
        self.pos = deque(maxlen=max_size)
        self.neg = deque(maxlen=max_size)
        self.feat_dim = feat_dim

    @torch.no_grad()
    def enqueue(self, feat, pseudo, conf, thresh):
        feat = feat.permute(0, 2, 3, 1).reshape(-1, feat.shape[1])
        pseudo = pseudo.reshape(-1)
        conf = conf.reshape(-1)
        keep = conf >= thresh
        if keep.sum() == 0:
            return
        feat = feat[keep]
        pseudo = pseudo[keep]
        pos = feat[pseudo > 0.5]
        neg = feat[pseudo <= 0.5]
        for v in pos[:64].detach().cpu():
            self.pos.append(v)
        for v in neg[:64].detach().cpu():
            self.neg.append(v)

    def get(self, device):
        pos = neg = None
        if len(self.pos) > 0:
            pos = F.normalize(torch.stack(list(self.pos), 0).to(device).mean(0), dim=0)
        if len(self.neg) > 0:
            neg = F.normalize(torch.stack(list(self.neg), 0).to(device).mean(0), dim=0)
        return pos, neg


def build_model(in_channels, cfg):
    return DeepLabV3Plus(in_channels=in_channels, num_classes=1, backbone=cfg.backbone,
                         output_stride=cfg.output_stride, pretrained=cfg.pretrained_backbone, feat_dim=256)


def make_optimizer(model, cfg, total_iters):
    backbone_params, head_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith('backbone.'):
            backbone_params.append(p)
        else:
            head_params.append(p)
    return torch.optim.AdamW([
        {'params': backbone_params, 'lr': cfg.lr * cfg.backbone_lr_mult},
        {'params': head_params, 'lr': cfg.lr},
    ], lr=cfg.lr, weight_decay=cfg.weight_decay)


bce_logits = nn.BCEWithLogitsLoss()


def sigmoid_dice_loss(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    num = 2.0 * (probs * targets).sum(dim=(1, 2, 3))
    den = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) + eps
    return (1.0 - num / den).mean()


def supervised_loss_lr(logits_hr, target_lr):
    logits_lr = F.interpolate(logits_hr, size=target_lr.shape[-2:], mode='bilinear', align_corners=False)
    return bce_logits(logits_lr, target_lr) + sigmoid_dice_loss(logits_lr, target_lr), logits_lr


def binary_iou_from_logits(logits, target, thr=0.5, eps=1e-6):
    pred = (torch.sigmoid(logits) > thr).float()
    inter = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - inter
    return ((inter + eps) / (union + eps)).mean().item()


def gradient_mag(prob):
    dx = torch.abs(prob[:, :, :, 1:] - prob[:, :, :, :-1])
    dy = torch.abs(prob[:, :, 1:, :] - prob[:, :, :-1, :])
    dx = F.pad(dx, (0, 1, 0, 0))
    dy = F.pad(dy, (0, 0, 0, 1))
    return torch.clamp(dx + dy, 0.0, 1.0)


def masked_mean(x, mask, eps=1e-6):
    return (x * mask).sum() / (mask.sum() + eps)


def sharpen(prob, temperature):
    prob = prob.clamp(1e-6, 1 - 1e-6)
    p = prob ** (1.0 / temperature)
    q = (1.0 - prob) ** (1.0 / temperature)
    return p / (p + q)


def ramp_weight(epoch, cfg, max_weight):
    if epoch >= cfg.rampup_epochs:
        return max_weight
    p = epoch / max(1, cfg.rampup_epochs)
    return max_weight * math.exp(-5.0 * (1.0 - p) ** 2)


@torch.no_grad()
def update_ema(student, teacher, momentum):
    for ps, pt in zip(student.parameters(), teacher.parameters()):
        pt.data.mul_(momentum).add_(ps.data, alpha=1.0 - momentum)


def unsup_bce(student_logits, teacher_prob, valid_mask):
    pseudo = (teacher_prob > 0.5).float()
    loss_map = F.binary_cross_entropy_with_logits(student_logits, pseudo, reduction='none')
    return masked_mean(loss_map, valid_mask)


def rank_loss(student_prob, teacher_prob, valid_mask, pairs, margin):
    total = 0.0
    used = 0
    for b in range(student_prob.shape[0]):
        s = student_prob[b, 0].reshape(-1)
        t = teacher_prob[b, 0].reshape(-1)
        m = valid_mask[b, 0].reshape(-1) > 0.5
        if m.sum() < 8:
            continue
        s = s[m]
        t = t[m]
        n = s.numel()
        idx1 = torch.randint(0, n, (min(pairs, n),), device=s.device)
        idx2 = torch.randint(0, n, (min(pairs, n),), device=s.device)
        ds = s[idx1] - s[idx2]
        dt = torch.sign(t[idx1] - t[idx2])
        nz = dt != 0
        if nz.sum() == 0:
            continue
        total = total + F.softplus(-(ds[nz] - margin) * dt[nz]).mean()
        used += 1
    if used == 0:
        return student_prob.new_tensor(0.0)
    return total / used


def edge_loss(prob, edge_gt):
    pred_edge = gradient_mag(prob)
    return F.binary_cross_entropy(pred_edge, edge_gt)


def proto_losses(feat, prob, queue, cfg):
    pos_proto, neg_proto = queue.get(feat.device)
    if pos_proto is None and neg_proto is None:
        return feat.new_tensor(0.0), feat.new_tensor(0.0)
    feat_flat = feat.permute(0, 2, 3, 1).reshape(-1, feat.shape[1])
    prob_flat = prob.reshape(-1)
    loss_proto = feat.new_tensor(0.0)
    loss_neg = feat.new_tensor(0.0)
    pos_mask = (prob_flat > 0.5).float()
    if pos_proto is not None and pos_mask.sum() > 0:
        sim = F.cosine_similarity(feat_flat, pos_proto.unsqueeze(0), dim=1)
        loss_proto = ((1.0 - sim) * pos_mask).sum() / (pos_mask.sum() + 1e-6)
    if neg_proto is not None and pos_mask.sum() > 0:
        sim = F.cosine_similarity(feat_flat, neg_proto.unsqueeze(0), dim=1)
        loss_neg = (F.relu(sim - 0.2) * pos_mask).sum() / (pos_mask.sum() + 1e-6)
    return loss_proto, loss_neg


def format_sec(sec):
    sec = int(max(sec, 0))
    return f'{sec//3600:02d}:{(sec%3600)//60:02d}:{sec%60:02d}'


def save_ckpt(path, models, optimizers, cfg, epoch, best_iou):
    state = {'epoch': epoch, 'best_iou': best_iou, 'cfg': asdict(cfg)}
    for k, v in models.items():
        state[f'model_{k}'] = v.state_dict()
    for k, v in optimizers.items():
        state[f'optim_{k}'] = v.state_dict()
    torch.save(state, path)


def validate(models, loader, cfg):
    if cfg.method == 'cps':
        models['student1'].eval(); models['student2'].eval()
    else:
        models['student'].eval()
    loss_meter = AverageMeter(); iou_meter = AverageMeter()
    with torch.no_grad():
        for batch in loader:
            img = batch['img'].to(cfg.device)
            target_lr = batch['mask_lr'].to(cfg.device)
            if cfg.method == 'cps':
                logits = 0.5 * (models['student1'](img) + models['student2'](img))
            else:
                logits = models['student'](img)
            loss, logits_lr = supervised_loss_lr(logits, target_lr)
            iou = binary_iou_from_logits(logits_lr, target_lr)
            bs = img.size(0)
            loss_meter.update(loss.item(), bs)
            iou_meter.update(iou, bs)
    return {'loss': loss_meter.avg, 'iou_lr': iou_meter.avg}


def write_log_header(csv_path):
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow([
            'epoch', 'train_loss', 'train_loss_lr', 'train_loss_unsup', 'train_loss_rank',
            'train_loss_edge', 'train_loss_proto', 'train_loss_neg', 'train_iou_lr', 'train_valid_ratio',
            'val_loss', 'val_iou_lr', 'epoch_sec', 'avg_epoch_sec', 'eta_sec'
        ])


def append_log(csv_path, epoch, train_stats, val_stats, epoch_sec, avg_epoch_sec, eta_sec):
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow([
            epoch, train_stats['loss'], train_stats['loss_lr'], train_stats['loss_unsup'], train_stats['loss_rank'],
            train_stats['loss_edge'], train_stats['loss_proto'], train_stats['loss_neg'], train_stats['iou_lr'], train_stats['valid_ratio'],
            val_stats['loss'], val_stats['iou_lr'], epoch_sec, avg_epoch_sec, eta_sec
        ])


def run_training(cfg, train_loader, val_loader, build_models_fn, train_epoch_fn):
    os.makedirs(cfg.save_dir, exist_ok=True)
    sample = next(iter(train_loader))
    in_channels = sample['img'].shape[1]
    models, aux = build_models_fn(in_channels, cfg)
    total_iters = cfg.num_epochs * max(1, len(train_loader))
    if cfg.method == 'cps':
        optimizers = {
            'student1': make_optimizer(models['student1'], cfg, total_iters),
            'student2': make_optimizer(models['student2'], cfg, total_iters),
        }
    else:
        optimizers = {'student': make_optimizer(models['student'], cfg, total_iters)}

    csv_path = os.path.join(cfg.save_dir, 'train_log.csv')
    write_log_header(csv_path)
    with open(os.path.join(cfg.save_dir, 'config.txt'), 'w', encoding='utf-8') as f:
        for k, v in asdict(cfg).items():
            f.write(f'{k}: {v}\n')

    best_iou = -1.0
    timer = AverageMeter()
    print(f'method={cfg.method} backbone={cfg.backbone} input_channels={in_channels} device={cfg.device}')
    print(f'train={len(train_loader.dataset)} val={len(val_loader.dataset)} save_dir={cfg.save_dir}')
    for epoch in range(1, cfg.num_epochs + 1):
        t0 = time.time()
        train_stats = train_epoch_fn(models, optimizers, train_loader, epoch, cfg, aux)
        val_stats = validate(models, val_loader, cfg) if len(val_loader.dataset) > 0 else {'loss': 0.0, 'iou_lr': 0.0}
        epoch_sec = time.time() - t0
        timer.update(epoch_sec)
        eta_sec = timer.avg * (cfg.num_epochs - epoch)
        print(
            f"[{cfg.method}] {epoch:03d}/{cfg.num_epochs:03d} | train={train_stats['loss']:.4f} "
            f"(lr={train_stats['loss_lr']:.4f}, unsup={train_stats['loss_unsup']:.4f}, rank={train_stats['loss_rank']:.4f}, "
            f"edge={train_stats['loss_edge']:.4f}, proto={train_stats['loss_proto']:.4f}, neg={train_stats['loss_neg']:.4f}, "
            f"iou={train_stats['iou_lr']:.4f}, valid={train_stats['valid_ratio']:.3f}) | "
            f"val={val_stats['loss']:.4f} (iou={val_stats['iou_lr']:.4f}) | time={epoch_sec:.1f}s avg={timer.avg:.1f}s eta={format_sec(eta_sec)}"
        )
        append_log(csv_path, epoch, train_stats, val_stats, epoch_sec, timer.avg, eta_sec)
        save_ckpt(os.path.join(cfg.save_dir, 'latest.pth'), models, optimizers, cfg, epoch, best_iou)
        if val_stats['iou_lr'] > best_iou:
            best_iou = val_stats['iou_lr']
            save_ckpt(os.path.join(cfg.save_dir, 'best.pth'), models, optimizers, cfg, epoch, best_iou)
            print(f'  -> save best, best_iou={best_iou:.4f}')
    print(f'Finished. Best val IoU(LR) = {best_iou:.4f}')
