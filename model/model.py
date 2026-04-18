import math

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel


def _inverse_geo_feat(x, g):
    """Unwarp a (B, C, H, W) feature map back to canonical frame per-sample.

    g: (B, 3) long tensor where each row is (hflip, vflip, rot90_k) — the
    forward geometric aug applied to produce the view this feature came from.
    The forward order was hflip -> vflip -> rot90_k, so the inverse applies
    rot90_{-k} -> vflip -> hflip.
    """
    out = []
    for i in range(x.shape[0]):
        xi = x[i:i+1]
        hf, vf, k = int(g[i, 0]), int(g[i, 1]), int(g[i, 2])
        if k:
            xi = torch.rot90(xi, -k, dims=[-2, -1])
        if vf:
            xi = torch.flip(xi, dims=[-2])
        if hf:
            xi = torch.flip(xi, dims=[-1])
        out.append(xi)
    return torch.cat(out, dim=0)


def _crop_and_resize(feat, pos, out_h, out_w):
    """
    Differentiably crop feat (B, C, H, W) at the normalised region
    pos (B, 4) = [y0, x0, y1, x1] in [0, 1], then resize to (out_h, out_w)
    using bilinear interpolation via F.grid_sample.
    """
    B, _, H, W = feat.shape
    y0, x0, y1, x1 = pos[:, 0], pos[:, 1], pos[:, 2], pos[:, 3]

    # build a dense sampling grid (B, out_h, out_w, 2) in [-1, 1]
    ts = torch.linspace(0, 1, out_h, device=feat.device)  # (out_h,)
    us = torch.linspace(0, 1, out_w, device=feat.device)  # (out_w,)

    # interpolate positions inside the crop box
    gy = (y0.view(B, 1) + ts.view(1, out_h) * (y1 - y0).view(B, 1)) * 2 - 1  # (B, out_h)
    gx = (x0.view(B, 1) + us.view(1, out_w) * (x1 - x0).view(B, 1)) * 2 - 1  # (B, out_w)

    # grid_sample expects (x, y) order in last dim
    grid = torch.stack([
        gx.view(B, 1, out_w).expand(B, out_h, out_w),
        gy.view(B, out_h, 1).expand(B, out_h, out_w),
    ], dim=-1)  # (B, out_h, out_w, 2)

    return F.grid_sample(feat, grid, mode='bilinear', align_corners=True, padding_mode='border')


def _double_conv(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.GroupNorm(8, out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.GroupNorm(8, out_ch),
        nn.ReLU(inplace=True),
    )


class _GlobalUNet(nn.Module):
    """
    Small UNet for the global branch.

    Encoder: 3 × (_double_conv + pool), 3→32→64→64 channels.
    Bottleneck at H/8 (RF ~61 px on a 256-px input).
    Decoder: 3 × (upsample + skip + _double_conv) back to full resolution.

    Output: (B, out_ch, H, W) — dense spatial features for _crop_and_resize.

    Pretrain without labels: attach Conv2d(out_ch, 3, 1), crop g_feat at tile
    position with _crop_and_resize, and reconstruct a downsampled local tile.
    """
    def __init__(self, in_ch=3, out_ch=64):
        super().__init__()
        self.pool = nn.MaxPool2d(2)

        self.enc1 = _double_conv(in_ch, 32)
        self.enc2 = _double_conv(32,  64)
        self.enc3 = _double_conv(64,  64)
        self.bot  = _double_conv(64,  64)           # H/8

        self.up3  = nn.ConvTranspose2d(64,     64,     2, 2)
        self.dec3 = _double_conv(64  + 64,     64)   # +enc3 skip → H/4

        self.up2  = nn.ConvTranspose2d(64,     64,     2, 2)
        self.dec2 = _double_conv(64  + 64,     64)   # +enc2 skip → H/2

        self.up1  = nn.ConvTranspose2d(64,     out_ch, 2, 2)
        self.dec1 = _double_conv(out_ch + 32,  out_ch) # +enc1 skip → H

    def forward(self, x):
        s1 = self.enc1(x)                                        # (B, 32, H,   W)
        s2 = self.enc2(self.pool(s1))                            # (B, 64, H/2, W/2)
        s3 = self.enc3(self.pool(s2))                            # (B, 64, H/4, W/4)
        x  = self.bot(self.pool(s3))                             # (B, 64, H/8, W/8)
        x  = self.dec3(torch.cat([self.up3(x), s3], dim=1))     # (B, 64, H/4)
        x  = self.dec2(torch.cat([self.up2(x), s2], dim=1))     # (B, 64, H/2)
        x  = self.dec1(torch.cat([self.up1(x), s1], dim=1))     # (B, out_ch, H)
        return x


class PretrainedLocalGlobalUNet(BaseModel):
    """
    Identical to LocalGlobalUNet but loads SSL-pretrained weights into global_enc.

    Config:
        "arch": {"type": "PretrainedLocalGlobalUNet", "args": {
            "num_classes": 1,
            "ssl_checkpoint": "saved/models/SSL_.../checkpoint-epoch50.pth"
        }}
    """
    def __init__(self, num_classes=1, in_ch=3, global_ch=64, ssl_checkpoint=None,
                 bottleneck_dropout=0.1):
        super().__init__()

        self.pool = nn.MaxPool2d(2)
        self.enc1 = _double_conv(in_ch, 64)
        self.enc2 = _double_conv(64,  128)
        self.enc3 = _double_conv(128, 256)
        self.enc4 = _double_conv(256, 512)
        self.bot  = _double_conv(512, 512)           # H/16
        self.bot_drop = nn.Dropout2d(bottleneck_dropout)

        self.global_enc = _GlobalUNet(in_ch=in_ch, out_ch=global_ch)

        if ssl_checkpoint is not None:
            ckpt = torch.load(ssl_checkpoint, map_location='cpu', weights_only=False)
            ssl_sd = ckpt['state_dict']
            global_enc_sd = {k.replace('unet.', ''): v
                             for k, v in ssl_sd.items() if k.startswith('unet.')}
            self.global_enc.load_state_dict(global_enc_sd)
            for p in self.global_enc.parameters():
                p.requires_grad = False

        self.up4  = nn.ConvTranspose2d(512 + global_ch, 256, kernel_size=2, stride=2)
        self.dec4 = _double_conv(256 + 512, 256)     # up + s4 → H/8

        self.up3  = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = _double_conv(128 + 256, 128)     # up + s3 → H/4

        self.up2  = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = _double_conv(64  + 128,  64)     # up + s2 → H/2

        self.up1  = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec1 = _double_conv(64  +  64,  64)     # up + s1 → H

        self.head     = nn.Conv2d(64, num_classes, kernel_size=1)
        self.aux_head = nn.Conv2d(global_ch, num_classes, kernel_size=1)

    def forward(self, local, glob, pos):
        s1 = self.enc1(local)                                        # (B,  64, H)
        s2 = self.enc2(self.pool(s1))                                # (B, 128, H/2)
        s3 = self.enc3(self.pool(s2))                                # (B, 256, H/4)
        s4 = self.enc4(self.pool(s3))                                # (B, 512, H/8)
        x  = self.bot_drop(self.bot(self.pool(s4)))                  # (B, 512, H/16)

        g_feat = self.global_enc(glob)
        g_ctx  = _crop_and_resize(g_feat, pos, *x.shape[2:])
        x = torch.cat([x, g_ctx], dim=1)                            # (B, 512+global_ch, H/16)

        x = self.dec4(torch.cat([self.up4(x), s4], dim=1))          # (B, 256, H/8)
        x = self.dec3(torch.cat([self.up3(x), s3], dim=1))          # (B, 128, H/4)
        x = self.dec2(torch.cat([self.up2(x), s2], dim=1))          # (B,  64, H/2)
        x = self.dec1(torch.cat([self.up1(x), s1], dim=1))          # (B,  64, H)

        main = self.head(x)

        if self.training:
            aux = F.interpolate(
                self.aux_head(g_feat),
                size=glob.shape[2:],
                mode='bilinear',
                align_corners=True,
            )
            return main, aux

        return main


class ImageNetPretrainedLocalGlobalUNet(BaseModel):
    """
    Like PretrainedLocalGlobalUNet, but the local encoder is a timm
    ImageNet-pretrained CNN with the classification head stripped. The global
    branch is still _GlobalUNet and can load SSL-pretrained weights.

    freeze_local_encoder accepts either a bool (True = freeze all stages,
    False = freeze none) or a list of stage indices in [0..4] to freeze
    individually. Stages map to ResNet modules as:
        0 -> conv1 + bn1   (stem, /2)
        1 -> layer1        (/4)
        2 -> layer2        (/8)
        3 -> layer3        (/16)
        4 -> layer4        (/32)

    Config:
        "arch": {"type": "ImageNetPretrainedLocalGlobalUNet", "args": {
            "num_classes": 1,
            "backbone": "resnet18",
            "backbone_pretrained": true,
            "freeze_local_encoder": [0, 1],
            "ssl_checkpoint": "saved/models/SSL_.../checkpoint-epoch50.pth"
        }}
    """
    _STAGE_MODULES = {
        0: ['conv1', 'bn1'],
        1: ['layer1'],
        2: ['layer2'],
        3: ['layer3'],
        4: ['layer4'],
    }

    def __init__(self, num_classes=1, in_ch=3, global_ch=64,
                 backbone='resnet18', backbone_pretrained=True,
                 freeze_local_encoder=False,
                 ssl_checkpoint=None, bottleneck_dropout=0.1):
        super().__init__()

        if ssl_checkpoint is None:
            raise ValueError("ssl_checkpoint is required for ImageNetPretrainedLocalGlobalUNet")

        # 5-stage encoder at reductions [2, 4, 8, 16, 32]
        self.backbone = timm.create_model(
            backbone, pretrained=backbone_pretrained, features_only=True,
            in_chans=in_ch, out_indices=(0, 1, 2, 3, 4),
        )
        enc_ch = self.backbone.feature_info.channels()   # [c1, c2, c3, c4, c5]
        assert len(enc_ch) == 5, (
            f"backbone {backbone} exposes {len(enc_ch)} stages; expected 5"
        )
        c1, c2, c3, c4, c5 = enc_ch

        if freeze_local_encoder is True:
            frozen_stages = [0, 1, 2, 3, 4]
        elif freeze_local_encoder:
            frozen_stages = list(freeze_local_encoder)
        else:
            frozen_stages = []
        for s in frozen_stages:
            if s not in self._STAGE_MODULES:
                raise ValueError(
                    f"freeze_local_encoder stage {s} invalid; must be in 0..4"
                )
            for mod_name in self._STAGE_MODULES[s]:
                if not hasattr(self.backbone, mod_name):
                    raise ValueError(
                        f"backbone '{backbone}' does not expose '{mod_name}'; "
                        f"stage-level freezing assumes ResNet-family structure"
                    )
                for p in getattr(self.backbone, mod_name).parameters():
                    p.requires_grad = False

        self.bot_drop = nn.Dropout2d(bottleneck_dropout)

        self.global_enc = _GlobalUNet(in_ch=in_ch, out_ch=global_ch)
        if ssl_checkpoint is not None:
            ckpt = torch.load(ssl_checkpoint, map_location='cpu', weights_only=False)
            ssl_sd = ckpt['state_dict']
            global_enc_sd = {k.replace('unet.', ''): v
                             for k, v in ssl_sd.items() if k.startswith('unet.')}
            self.global_enc.load_state_dict(global_enc_sd)
            for p in self.global_enc.parameters():
                p.requires_grad = False

        # Decoder: upsample ×2 at each step, concat with matching skip.
        # bottleneck (/32) → /16 → /8 → /4 → /2 → H
        self.up5  = nn.ConvTranspose2d(c5 + global_ch, c4, kernel_size=2, stride=2)
        self.dec5 = _double_conv(c4 + c4, c4)     # + s4 (/16)

        self.up4  = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.dec4 = _double_conv(c3 + c3, c3)     # + s3 (/8)

        self.up3  = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec3 = _double_conv(c2 + c2, c2)     # + s2 (/4)

        self.up2  = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec2 = _double_conv(c1 + c1, c1)     # + s1 (/2)

        self.up1  = nn.ConvTranspose2d(c1, c1, kernel_size=2, stride=2)
        self.dec1 = _double_conv(c1, c1)          # no skip at full res

        self.head     = nn.Conv2d(c1, num_classes, kernel_size=1)
        self.aux_head = nn.Conv2d(global_ch, num_classes, kernel_size=1)

    def forward(self, local, glob, pos):
        s1, s2, s3, s4, x = self.backbone(local)     # /2, /4, /8, /16, /32
        x = self.bot_drop(x)

        g_feat = self.global_enc(glob)
        g_ctx  = _crop_and_resize(g_feat, pos, *x.shape[2:])
        x = torch.cat([x, g_ctx], dim=1)

        x = self.dec5(torch.cat([self.up5(x), s4], dim=1))
        x = self.dec4(torch.cat([self.up4(x), s3], dim=1))
        x = self.dec3(torch.cat([self.up3(x), s2], dim=1))
        x = self.dec2(torch.cat([self.up2(x), s1], dim=1))
        x = self.dec1(self.up1(x))

        main = self.head(x)

        if self.training:
            aux = F.interpolate(
                self.aux_head(g_feat),
                size=glob.shape[2:],
                mode='bilinear',
                align_corners=True,
            )
            return main, aux

        return main


class SSLGlobalUNet(BaseModel):
    """
    Self-supervised wrapper around _GlobalUNet for pretraining.

    The UNet produces dense 64-ch spatial features. Two lightweight 1x1 conv
    heads project those features to RGB for two reconstruction tasks:
      1) aug_head:   reverse augmentation on the full global image
      2) local_head: reconstruct a local tile from features cropped at its position

    After pretraining, transfer self.unet weights to LocalGlobalUNet.global_enc.

    Config:
        "arch": {"type": "SSLGlobalUNet", "args": {"in_ch": 1, "out_ch": 64}}
    """
    def __init__(self, in_ch=3, out_ch=64):
        super().__init__()
        self.unet = _GlobalUNet(in_ch=in_ch, out_ch=out_ch)
        # throwaway decoder heads — discarded after pretraining
        self.global_head = nn.Sequential(
            _double_conv(out_ch, out_ch),
            _double_conv(out_ch, out_ch),
            nn.Conv2d(out_ch, in_ch, kernel_size=1),
        )
        self.local_head = nn.Sequential(
            _double_conv(out_ch, out_ch),
            _double_conv(out_ch, out_ch),
            nn.Conv2d(out_ch, in_ch, kernel_size=1),
        )

    def forward(self, aug_global, pos, local_target):
        feat = self.unet(aug_global)                                    # (B, out_ch, H, W)
        global_recon = self.global_head(feat)                           # (B, in_ch, H, W)
        cs = local_target.shape[-1]
        B, N = pos.shape[:2]                                            # pos: (B, N, 4)
        feat_exp = feat.repeat_interleave(N, dim=0)                    # (B*N, out_ch, H, W)
        local_feat = _crop_and_resize(feat_exp, pos.reshape(B * N, 4), cs, cs)  # (B*N, out_ch, cs, cs)
        local_recon = self.local_head(local_feat)                       # (B*N, in_ch, cs, cs)
        local_recon = local_recon.reshape(B, N, *local_recon.shape[1:]) # (B, N, in_ch, cs, cs)
        return global_recon, local_recon, feat


