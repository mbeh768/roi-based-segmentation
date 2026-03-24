import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from base import BaseModel


class _DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, skip_ch=0):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


def _double_conv(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
    )


class SimpleUNet(BaseModel):
    """
    Vanilla UNet for global-image-only segmentation.

    Input:  (B, 3, H, W)
    Output: (B, num_classes, H, W) raw logits

    Designed for overfitting demonstrations — no pretrained weights,
    enough capacity to memorise a small dataset.

    Config:
        "arch": {"type": "SimpleUNet", "args": {"num_classes": 1}}
    """
    def __init__(self, num_classes=1):
        super().__init__()

        self.enc1 = _double_conv(3,    64)
        self.enc2 = _double_conv(64,  128)
        self.enc3 = _double_conv(128, 256)
        self.enc4 = _double_conv(256, 512)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = _double_conv(512, 1024)

        self.up4   = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4  = _double_conv(1024, 512)

        self.up3   = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3  = _double_conv(512, 256)

        self.up2   = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2  = _double_conv(256, 128)

        self.up1   = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1  = _double_conv(128, 64)

        self.head  = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        s1 = self.enc1(x)                          # (B,  64, H,    W)
        s2 = self.enc2(self.pool(s1))              # (B, 128, H/2,  W/2)
        s3 = self.enc3(self.pool(s2))              # (B, 256, H/4,  W/4)
        s4 = self.enc4(self.pool(s3))              # (B, 512, H/8,  W/8)

        x  = self.bottleneck(self.pool(s4))        # (B,1024, H/16, W/16)

        x  = self.dec4(torch.cat([self.up4(x), s4], dim=1))   # (B, 512, H/8,  W/8)
        x  = self.dec3(torch.cat([self.up3(x), s3], dim=1))   # (B, 256, H/4,  W/4)
        x  = self.dec2(torch.cat([self.up2(x), s2], dim=1))   # (B, 128, H/2,  W/2)
        x  = self.dec1(torch.cat([self.up1(x), s1], dim=1))   # (B,  64, H,    W)

        return self.head(x)                        # (B, num_classes, H, W)


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


def _make_region_bias(pos, h, w, device):
    """
    Gaussian heatmap (B, 1, h, w) centred on the tile's region in the global feature map.
    pos: (B, 4) = [y0, x0, y1, x1] normalised to [0, 1]
    """
    B = pos.shape[0]
    ys = torch.linspace(0, 1, h, device=device).view(1, h, 1)
    xs = torch.linspace(0, 1, w, device=device).view(1, 1, w)
    cy = ((pos[:, 0] + pos[:, 2]) / 2).view(B, 1, 1)
    cx = ((pos[:, 1] + pos[:, 3]) / 2).view(B, 1, 1)
    sy = ((pos[:, 2] - pos[:, 0]) / 2).clamp(min=1.0 / h).view(B, 1, 1)
    sx = ((pos[:, 3] - pos[:, 1]) / 2).clamp(min=1.0 / w).view(B, 1, 1)
    gauss = torch.exp(-0.5 * (((ys - cy) / sy) ** 2 + ((xs - cx) / sx) ** 2))
    return gauss.unsqueeze(1)   # (B, 1, h, w)


class DualBranchSegmentation(BaseModel):
    """
    Shared ResNet-50 encoder processes a local tile and a downsampled global image.

    - A Gaussian regional bias focuses the global feature map on the tile's location.
    - Cross-channel attention (global → local) selects which local channels matter.
    - SE channel attention re-weights the concatenated 4096-ch features.
    - A 1×1 fuse conv reduces to 2048.
    - Decoder with U-Net skip connections from the local encoder stages.

    Inputs: local (B,3,H,W), glob (B,3,H,W), pos (B,4) = [y0,x0,y1,x1] normalised

    Config:
        "arch": {"type": "DualBranchSegmentation",
                 "args": {"num_classes": 1, "freeze_encoder": false}}
    """
    def __init__(self, num_classes=1, pretrained=True, freeze_encoder=False):
        super().__init__()

        bb = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)

        # shared encoder — split into stages so local branch can expose skip features
        # stem split before maxpool to get a H/2 skip for dec4
        self.enc_pre    = nn.Sequential(bb.conv1, bb.bn1, bb.relu)  # → (B,  64, H/2, W/2)
        self.enc_pool   = bb.maxpool                                 # → (B,  64, H/4, W/4)
        self.enc_layer1 = bb.layer1                                  # → (B, 256, H/4, W/4)
        self.enc_layer2 = bb.layer2                                  # → (B, 512, H/8, W/8)
        self.enc_layer3 = bb.layer3                                  # → (B,1024, H/16,W/16)
        self.enc_layer4 = bb.layer4                                  # → (B,2048, H/32,W/32)

        if freeze_encoder:
            for m in self._encoder_stages():
                m.requires_grad_(False)

        self.cross_channel_attn = _CrossChannelAttention()
        self.channel_attn       = _ChannelAttention(4096, reduction=16)

        self.fuse = nn.Sequential(
            nn.Conv2d(4096, 2048, kernel_size=1, bias=False),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
        )

        # decoder with skip connections from local encoder
        # skips: s3=layer3(1024@H/16), s2=layer2(512@H/8),
        #        s1=layer1(256@H/4),   s_pre=pre-pool(64@H/2)
        self.dec1 = _DecoderBlock(2048, 1024, skip_ch=1024)  # → H/16
        self.dec2 = _DecoderBlock(1024,  512, skip_ch=512)   # → H/8
        self.dec3 = _DecoderBlock( 512,  256, skip_ch=256)   # → H/4
        self.dec4 = _DecoderBlock( 256,  128, skip_ch=64)    # → H/2  (s_pre)
        self.dec5 = _DecoderBlock( 128,   64)                 # → H

        self.head = nn.Conv2d(64, num_classes, kernel_size=1)

    def _encoder_stages(self):
        return [self.enc_pre, self.enc_pool, self.enc_layer1,
                self.enc_layer2, self.enc_layer3, self.enc_layer4]

    def _encode(self, x):
        s_pre = self.enc_pre(x)          # (B,  64, H/2,  W/2)
        s0    = self.enc_pool(s_pre)     # (B,  64, H/4,  W/4)
        s1    = self.enc_layer1(s0)      # (B, 256, H/4,  W/4)
        s2    = self.enc_layer2(s1)      # (B, 512, H/8,  W/8)
        s3    = self.enc_layer3(s2)      # (B,1024, H/16, W/16)
        s4    = self.enc_layer4(s3)      # (B,2048, H/32, W/32)
        return s4, (s_pre, s1, s2, s3)

    def train(self, mode=True):
        super().train(mode)
        enc_frozen = not any(
            p.requires_grad for m in self._encoder_stages() for p in m.parameters()
        )
        if enc_frozen:
            for m in self._encoder_stages():
                m.eval()
        return self

    def forward(self, local, glob, pos):
        local_feat,  local_skips = self._encode(local)   # (B,2048,h,w) + 4 skips
        global_feat, _           = self._encode(glob)    # (B,2048,h,w)

        # focus global features on the tile's region before attention
        bias = _make_region_bias(pos, *global_feat.shape[2:], global_feat.device)
        global_feat = global_feat * (1 + bias)

        local_attended = self.cross_channel_attn(local_feat, global_feat)
        x = torch.cat([local_attended, global_feat], dim=1)  # (B,4096,h,w)
        x = self.channel_attn(x)
        x = self.fuse(x)                                 # (B,2048,h,w)

        s_pre, s1, s2, s3 = local_skips
        x = self.dec1(x,   s3)    # (B,1024,H/16)
        x = self.dec2(x,   s2)    # (B, 512,H/8)
        x = self.dec3(x,   s1)    # (B, 256,H/4)
        x = self.dec4(x,   s_pre) # (B, 128,H/2)
        x = self.dec5(x)         # (B,  64,H)
        return self.head(x)      # (B,num_classes,H,W)

