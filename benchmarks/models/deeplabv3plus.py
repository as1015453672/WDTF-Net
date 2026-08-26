import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, resnet101
from torchvision.models._utils import IntermediateLayerGetter


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        super().__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels=256):
        super().__init__()
        modules = [
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        ]
        modules.extend([ASPPConv(in_channels, out_channels, rate) for rate in atrous_rates])
        modules.append(ASPPPooling(in_channels, out_channels))
        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(len(modules) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        xs = [conv(x) for conv in self.convs]
        x = torch.cat(xs, dim=1)
        return self.project(x)


class Decoder(nn.Module):
    def __init__(self, low_level_channels, num_classes, feat_dim=256):
        super().__init__()
        self.low_proj = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(feat_dim + 48, feat_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_dim, feat_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Conv2d(feat_dim, num_classes, 1)

    def forward(self, x, low_level):
        low = self.low_proj(low_level)
        x = F.interpolate(x, size=low.shape[-2:], mode='bilinear', align_corners=False)
        feat = self.fuse(torch.cat([x, low], dim=1))
        logits = self.classifier(feat)
        return logits, feat


class DeepLabV3Plus(nn.Module):
    def __init__(self, in_channels=3, num_classes=1, backbone='resnet50', output_stride=16, pretrained=True, feat_dim=256):
        super().__init__()
        if backbone == 'resnet50':
            backbone_net = resnet50(weights='DEFAULT' if pretrained else None, replace_stride_with_dilation=[False, output_stride == 8, True])
            high_ch, low_ch = 2048, 256
        elif backbone == 'resnet101':
            backbone_net = resnet101(weights='DEFAULT' if pretrained else None, replace_stride_with_dilation=[False, output_stride == 8, True])
            high_ch, low_ch = 2048, 256
        else:
            raise ValueError(backbone)

        if in_channels != 3:
            old = backbone_net.conv1
            backbone_net.conv1 = nn.Conv2d(in_channels, old.out_channels, kernel_size=old.kernel_size,
                                           stride=old.stride, padding=old.padding, bias=False)
            with torch.no_grad():
                if in_channels > 3:
                    backbone_net.conv1.weight[:, :3] = old.weight
                    nn.init.kaiming_normal_(backbone_net.conv1.weight[:, 3:], mode='fan_out', nonlinearity='relu')
                else:
                    backbone_net.conv1.weight[:] = old.weight[:, :in_channels]

        return_layers = {'layer1': 'low_level', 'layer4': 'out'}
        self.backbone = IntermediateLayerGetter(backbone_net, return_layers=return_layers)
        atrous_rates = [6, 12, 18] if output_stride == 16 else [12, 24, 36]
        self.aspp = ASPP(high_ch, atrous_rates, out_channels=feat_dim)
        self.decoder = Decoder(low_ch, num_classes, feat_dim=feat_dim)

    def logits_from_feat(self, feat):
        return self.decoder.classifier(feat)

    def forward(self, x, return_feat=False, normalize_feat=True):
        size = x.shape[-2:]
        features = self.backbone(x)
        x = self.aspp(features['out'])
        logits, feat = self.decoder(x, features['low_level'])
        logits = F.interpolate(logits, size=size, mode='bilinear', align_corners=False)
        feat = F.interpolate(feat, size=size, mode='bilinear', align_corners=False)
        if return_feat:
            feat_out = F.normalize(feat, dim=1) if normalize_feat else feat
            return logits, feat_out
        return logits


