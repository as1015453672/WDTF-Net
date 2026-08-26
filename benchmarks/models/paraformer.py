
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RPBlock(nn.Module):
    """Resolution-preserving block inspired by Paraformer/L2HNet V1.
    Parallel 1x1, 3x3, 5x5 convs with channels 128/64/32, concatenate -> reduce to 128.
    """
    def __init__(self, in_ch: int, out_ch: int = 128):
        super().__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(in_ch, 128, 1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.b3 = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.b5 = nn.Sequential(
            nn.Conv2d(in_ch, 32, 5, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.reduce = nn.Sequential(
            nn.Conv2d(224, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.short = nn.Identity() if in_ch == out_ch else nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.out_act = nn.ReLU(inplace=True)

    def forward(self, x):
        y = torch.cat([self.b1(x), self.b3(x), self.b5(x)], dim=1)
        y = self.reduce(y)
        y = y + self.short(x)
        return self.out_act(y)


class SimpleTransformerBranch(nn.Module):
    def __init__(self, in_ch=128*5, embed_dim=256, depth=12, num_heads=8, mlp_ratio=4.0, token_pool=4):
        super().__init__()
        self.embed = nn.Conv2d(in_ch, embed_dim, 1, bias=False)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.token_pool = int(token_pool)
        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # x: B,C,H,W -> downsample for tractable global modeling
        x = self.embed(x)
        x_small = F.avg_pool2d(x, kernel_size=self.token_pool, stride=self.token_pool, ceil_mode=False)
        b, c, h, w = x_small.shape
        seq = x_small.flatten(2).transpose(1, 2)  # B,HW,C
        seq = self.encoder(seq)
        feat = seq.transpose(1, 2).reshape(b, c, h, w)
        feat = F.interpolate(feat, size=x.shape[-2:], mode='bilinear', align_corners=False)
        feat = self.proj(feat)
        return feat


class CNNPrimalClassifier(nn.Module):
    def __init__(self, in_ch=128, num_classes=1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes, 1),
        )

    def forward(self, x):
        return self.head(x)


class HybridClassifier(nn.Module):
    def __init__(self, in_ch=128*6, num_classes=1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes, 1),
        )

    def forward(self, x):
        return self.head(x)


class Paraformer(nn.Module):
    """Task-adapted Paraformer.
    - CNN branch: five serial resolution-preserving blocks
    - Transformer branch: 12-layer transformer on concatenated CNN features
    - PLAT-ready outputs: cnn primal prediction and final hybrid prediction
    """
    def __init__(self, in_channels=3, num_classes=1, base_ch=128, trans_embed_dim=256, trans_depth=12, trans_heads=8, token_pool=4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
        )
        self.rp_blocks = nn.ModuleList([
            RPBlock(base_ch if i == 0 else base_ch, base_ch) for i in range(5)
        ])
        self.transformer = SimpleTransformerBranch(
            in_ch=base_ch * 5,
            embed_dim=trans_embed_dim,
            depth=trans_depth,
            num_heads=trans_heads,
            token_pool=token_pool,
        )
        self.cnn_classifier = CNNPrimalClassifier(in_ch=base_ch, num_classes=num_classes)
        self.hybrid_classifier = HybridClassifier(in_ch=base_ch * 6, num_classes=num_classes)

    def forward(self, x, return_feat=False):
        x0 = self.stem(x)
        feats = []
        xk = x0
        for blk in self.rp_blocks:
            xk = blk(xk)
            feats.append(xk)

        cnn_feat = feats[-1]
        primal_logits = self.cnn_classifier(cnn_feat)

        concat_feats = torch.cat(feats, dim=1)
        trans_feat = self.transformer(concat_feats)
        hybrid_feat = torch.cat(feats + [trans_feat], dim=1)
        final_logits = self.hybrid_classifier(hybrid_feat)

        if return_feat:
            return primal_logits, final_logits, hybrid_feat
        return primal_logits, final_logits


