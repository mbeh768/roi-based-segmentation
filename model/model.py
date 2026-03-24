import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from base import BaseModel


class _DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ResNetSegmentation(BaseModel):
    """
    ImageNet-pretrained ResNet-50 encoder + simple convolutional decoder
    for dense segmentation.

    Encoder: ResNet-50 without avgpool/fc → (B, 2048, H/32, W/32)
    Decoder: 5× bilinear upsample + conv blocks  → (B, num_classes, H, W)
    Output:  raw logits — use BCEWithLogitsLoss for num_classes=1,
             CrossEntropyLoss for num_classes>1

    Config:
        "arch": {"type": "ResNetSegmentation", "args": {"num_classes": 1, "freeze_encoder": false}}
    """
    def __init__(self, num_classes=1, pretrained=True, freeze_encoder=False):
        super().__init__()

        backbone = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)

        # Encoder: strip the classification head, keep spatial layers
        self.encoder = nn.Sequential(
            backbone.conv1,    # → (B,   64, H/2,  W/2)
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,  # → (B,   64, H/4,  W/4)
            backbone.layer1,   # → (B,  256, H/4,  W/4)
            backbone.layer2,   # → (B,  512, H/8,  W/8)
            backbone.layer3,   # → (B, 1024, H/16, W/16)
            backbone.layer4,   # → (B, 2048, H/32, W/32)
        )
        if freeze_encoder:
            self.encoder.requires_grad_(False)

        self.decoder = nn.Sequential(
            _DecoderBlock(2048, 1024),  # → H/16
            _DecoderBlock(1024,  512),  # → H/8
            _DecoderBlock( 512,  256),  # → H/4
            _DecoderBlock( 256,  128),  # → H/2
            _DecoderBlock( 128,   64),  # → H
        )

        self.head = nn.Conv2d(64, num_classes, kernel_size=1)

    def train(self, mode=True):
        super().train(mode)
        # if encoder is frozen, keep its BN layers in eval mode so pretrained
        # running stats aren't corrupted by the new domain's batch statistics
        encoder_frozen = not any(p.requires_grad for p in self.encoder.parameters())
        if encoder_frozen:
            self.encoder.eval()
        return self

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return self.head(x)


class _CrossChannelAttention(nn.Module):
    """
    Global-to-local cross-channel attention.
    Global features are spatially pooled to produce per-channel weights
    that modulate the local feature map — global context selects which
    local channels are relevant.
    No learned parameters.
    """
    def forward(self, local_feat, global_feat):
        B, C, _, _ = global_feat.shape
        weights = torch.sigmoid(global_feat.mean(dim=(2, 3)))  # (B, C)
        return local_feat * weights.view(B, C, 1, 1)           # (B, C, H, W)


class _ChannelAttention(nn.Module):
    """SE-style channel attention."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        B, C, _, _ = x.shape
        scale = self.fc(x.mean(dim=(2, 3)))   # (B, C)
        return x * scale.view(B, C, 1, 1)


def _make_encoder(pretrained=True, freeze=False):
    backbone = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
    enc = nn.Sequential(
        backbone.conv1,
        backbone.bn1,
        backbone.relu,
        backbone.maxpool,
        backbone.layer1,
        backbone.layer2,
        backbone.layer3,
        backbone.layer4,
    )
    if freeze:
        enc.requires_grad_(False)
    return enc


class DualBranchSegmentation(BaseModel):
    """
    Shared ResNet-50 encoder processes both a local tile and a downsampled global
    image. Spatial cross-attention localises the global context to the tile region,
    SE channel attention re-weights the concatenated features, a 1×1 fuse conv
    reduces channels, and a shallow decoder up-samples to the input resolution.

    Config:
        "arch": {"type": "DualBranchSegmentation",
                 "args": {"num_classes": 1, "freeze_encoder": false}}
    """
    def __init__(self, num_classes=1, pretrained=True, freeze_encoder=False):
        super().__init__()
        self.encoder = _make_encoder(pretrained, freeze=freeze_encoder)

        self.cross_channel_attn = _CrossChannelAttention()
        self.channel_attn       = _ChannelAttention(4096, reduction=16)

        self.fuse = nn.Sequential(
            nn.Conv2d(4096, 2048, kernel_size=1, bias=False),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            _DecoderBlock(2048, 1024),
            _DecoderBlock(1024,  512),
            _DecoderBlock( 512,  256),
            _DecoderBlock( 256,  128),
            _DecoderBlock( 128,   64),
        )

        self.head = nn.Conv2d(64, num_classes, kernel_size=1)

    def train(self, mode=True):
        super().train(mode)
        encoder_frozen = not any(p.requires_grad for p in self.encoder.parameters())
        if encoder_frozen:
            self.encoder.eval()
        return self

    def forward(self, local, glob):
        local_feat  = self.encoder(local)              # (B, 2048, h, w)
        global_feat = self.encoder(glob)               # (B, 2048, h, w)
        local_attended = self.cross_channel_attn(local_feat, global_feat)
        x = torch.cat([local_attended, global_feat], dim=1)  # (B, 4096, h, w)
        x = self.channel_attn(x)
        x = self.fuse(x)                               # (B, 2048, h, w)
        x = self.decoder(x)                            # (B, 64, H, W)
        return self.head(x)                            # (B, num_classes, H, W)

