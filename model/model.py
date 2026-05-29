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


class _GlobalUNetHD(nn.Module):
    """
    High-res global UNet (EXPERIMENT 8C). Ingests a high-res global image
    (e.g. 1024) and emits a feature map at input/4 (e.g. 256) — the working
    resolution the rest of the pipeline expects.

    A 2-stage strided stem learns the 4x downsample that used to be a fixed
    bilinear preprocess step ("learned downsampling"); the body is the same
    pool/skip UNet as _GlobalUNet running at the working resolution, so it
    keeps the multi-scale bottleneck. Output: (B, out_ch, in/4, in/4).
    """
    def __init__(self, in_ch=3, out_ch=64):
        super().__init__()
        self.pool = nn.MaxPool2d(2)

        # learned downsampling stem: in -> in/2 -> in/4
        self.stem1 = _double_conv(in_ch, 16)        # @in
        self.stem2 = _double_conv(16,    32)        # @in/2

        # UNet body at the working resolution (in/4)
        self.enc1 = _double_conv(32, 32)            # @in/4
        self.enc2 = _double_conv(32, 64)            # /8
        self.enc3 = _double_conv(64, 64)            # /16
        self.bot  = _double_conv(64, 64)            # /32

        self.up3  = nn.ConvTranspose2d(64, 64, 2, 2)
        self.dec3 = _double_conv(64 + 64, 64)       # +enc3 → /16
        self.up2  = nn.ConvTranspose2d(64, 64, 2, 2)
        self.dec2 = _double_conv(64 + 64, 64)       # +enc2 → /8
        self.up1  = nn.ConvTranspose2d(64, out_ch, 2, 2)
        self.dec1 = _double_conv(out_ch + 32, out_ch)  # +enc1 → /4 (working res)

    def forward(self, x):
        x  = self.pool(self.stem1(x))               # in/2
        x  = self.pool(self.stem2(x))               # in/4
        s1 = self.enc1(x)                            # in/4
        s2 = self.enc2(self.pool(s1))               # /8
        s3 = self.enc3(self.pool(s2))               # /16
        x  = self.bot(self.pool(s3))                # /32
        x  = self.dec3(torch.cat([self.up3(x), s3], dim=1))   # /16
        x  = self.dec2(torch.cat([self.up2(x), s2], dim=1))   # /8
        x  = self.dec1(torch.cat([self.up1(x), s1], dim=1))   # /4
        return x


class _GlobalCNNHD(nn.Module):
    """
    High-res global CNN (EXPERIMENT 8D). Same high-res input and same input/4
    output as _GlobalUNetHD, but no bottleneck and no skips — features flow
    straight down 1024 → 256 and are then refined at the working resolution.

    Tests whether the UNet bottleneck (multi-scale context) matters versus
    plain high-res-aware downsampling.  Output: (B, out_ch, in/4, in/4).
    """
    def __init__(self, in_ch=3, out_ch=64):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.down1 = _double_conv(in_ch, 32)        # @in    → pool /2
        self.down2 = _double_conv(32,    64)        # @in/2  → pool /4
        self.body1 = _double_conv(64,    64)        # @in/4
        self.body2 = _double_conv(64,    out_ch)    # @in/4

    def forward(self, x):
        x = self.pool(self.down1(x))                # in/2
        x = self.pool(self.down2(x))                # in/4
        x = self.body1(x)
        x = self.body2(x)
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

    When use_roi=True, the forward pass accepts an extra (B, roi_ch, H_g, W_g)
    mask tensor (typically produced by annotate_globals.py). The ROI is cropped
    at the same normalised pos as the local patch via _crop_and_resize and
    concatenated alongside the global context at the bottleneck.

    Config:
        "arch": {"type": "ImageNetPretrainedLocalGlobalUNet", "args": {
            "num_classes": 1,
            "backbone": "resnet18",
            "backbone_pretrained": true,
            "freeze_local_encoder": [0, 1],
            "ssl_checkpoint": "saved/models/SSL_.../checkpoint-epoch50.pth",
            "use_roi": true,
            "roi_ch": 1
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
                 ssl_checkpoint=None, bottleneck_dropout=0.1,
                 use_roi=False, roi_ch=1, disable_global=False):
        super().__init__()

        if ssl_checkpoint is None:
            raise ValueError("ssl_checkpoint is required for ImageNetPretrainedLocalGlobalUNet")

        self.use_roi = use_roi
        self.roi_ch = roi_ch if use_roi else 0
        self.disable_global = disable_global
        self.global_ch = global_ch

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

        self.global_enc = self._build_global_enc(in_ch, global_ch)
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
        self.up5  = nn.ConvTranspose2d(c5 + global_ch + self.roi_ch, c4, kernel_size=2, stride=2)
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

    def _build_global_enc(self, in_ch, global_ch):
        """Global branch encoder. Subclasses override to swap the encoder while
        reusing the SSL-load + freeze logic in __init__."""
        return _GlobalUNet(in_ch=in_ch, out_ch=global_ch)

    def forward(self, local, glob, pos, roi=None):
        s1, s2, s3, s4, x = self.backbone(local)     # /2, /4, /8, /16, /32

        if self.disable_global:
            g_feat = None
            g_ctx = torch.zeros(x.shape[0], self.global_ch, *x.shape[2:],
                                device=x.device, dtype=x.dtype)
        else:
            g_feat = self.global_enc(glob)
            g_ctx  = _crop_and_resize(g_feat, pos, *x.shape[2:])
        if self.use_roi:
            assert roi is not None, "use_roi=True but no roi tensor was passed to forward()"
            roi_ctx = _crop_and_resize(roi, pos, *x.shape[2:])
            x = torch.cat([x, g_ctx, roi_ctx], dim=1)
        else:
            x = torch.cat([x, g_ctx], dim=1)

        x = self.dec5(torch.cat([self.up5(x), s4], dim=1))
        x = self.bot_drop(x)
        x = self.dec4(torch.cat([self.up4(x), s3], dim=1))
        x = self.dec3(torch.cat([self.up3(x), s2], dim=1))
        x = self.dec2(torch.cat([self.up2(x), s1], dim=1))
        x = self.dec1(self.up1(x))

        main = self.head(x)

        if self.training and g_feat is not None:
            aux = F.interpolate(
                self.aux_head(g_feat),
                size=glob.shape[2:],
                mode='bilinear',
                align_corners=True,
            )
            return main, aux

        return main


class ImageNetPretrainedLocalUNet(BaseModel):
    """
    Streamlined ablation of ImageNetPretrainedLocalGlobalUNet: ImageNet local
    encoder + optional ROI concatenated at the bottleneck, no global branch and
    no SSL checkpoint.

    Formalizes the disable_global=True ablation by physically removing the
    global encoder and the global_ch slice of up5, so parameter count drops by
    the global branch's weights + global_ch*c4*4 in up5.

    Config:
        "arch": {"type": "ImageNetPretrainedLocalUNet", "args": {
            "num_classes": 1,
            "in_ch": 1,
            "backbone": "resnet18",
            "backbone_pretrained": true,
            "freeze_local_encoder": true,
            "bottleneck_dropout": 0.1,
            "use_roi": true,
            "roi_ch": 1
        }}
    """
    _STAGE_MODULES = ImageNetPretrainedLocalGlobalUNet._STAGE_MODULES

    def __init__(self, num_classes=1, in_ch=3,
                 backbone='resnet18', backbone_pretrained=True,
                 freeze_local_encoder=False,
                 bottleneck_dropout=0.1,
                 use_roi=False, roi_ch=1):
        super().__init__()

        self.use_roi = use_roi
        self.roi_ch = roi_ch if use_roi else 0

        self.backbone = timm.create_model(
            backbone, pretrained=backbone_pretrained, features_only=True,
            in_chans=in_ch, out_indices=(0, 1, 2, 3, 4),
        )
        enc_ch = self.backbone.feature_info.channels()
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

        self.up5  = nn.ConvTranspose2d(c5 + self.roi_ch, c4, kernel_size=2, stride=2)
        self.dec5 = _double_conv(c4 + c4, c4)
        self.up4  = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.dec4 = _double_conv(c3 + c3, c3)
        self.up3  = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec3 = _double_conv(c2 + c2, c2)
        self.up2  = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec2 = _double_conv(c1 + c1, c1)
        self.up1  = nn.ConvTranspose2d(c1, c1, kernel_size=2, stride=2)
        self.dec1 = _double_conv(c1, c1)

        self.head = nn.Conv2d(c1, num_classes, kernel_size=1)

    def forward(self, local, glob, pos, roi=None):
        # glob is accepted for signature compatibility with LocalGlobalDataLoader batches
        # but not used; same for pos when use_roi=False.
        s1, s2, s3, s4, x = self.backbone(local)

        if self.use_roi:
            assert roi is not None, "use_roi=True but no roi tensor was passed to forward()"
            roi_ctx = _crop_and_resize(roi, pos, *x.shape[2:])
            x = torch.cat([x, roi_ctx], dim=1)

        x = self.dec5(torch.cat([self.up5(x), s4], dim=1))
        x = self.bot_drop(x)
        x = self.dec4(torch.cat([self.up4(x), s3], dim=1))
        x = self.dec3(torch.cat([self.up3(x), s2], dim=1))
        x = self.dec2(torch.cat([self.up2(x), s1], dim=1))
        x = self.dec1(self.up1(x))

        return self.head(x)


class ImageNetPretrainedLocalGlobalUNetLightDecoder(ImageNetPretrainedLocalGlobalUNet):
    """
    Lighter decoder variant of ImageNetPretrainedLocalGlobalUNet.

    Skip connections are taken from every OTHER ResNet stage — specifically
    s4 (/16) and s2 (/4). The intermediate stages (s3, s1) are skipped, and the
    upsample between skip fusions is 4× rather than 2×. A single double_conv
    follows each skip-concat.

    decoder_ch (optional) collapses the entire decoder to a single narrow
    width. When set, up5 outputs decoder_ch channels and the s4 / s2 skips
    are projected to decoder_ch via 1×1 convs before concat. This is the big
    param-saver because up5 + dec5 dominate the trainable parameter count
    (resnet18: c4=256, c5=512 → ~2.4M of ~3.6M).

    Layout:
        bottleneck (/32)
          → up5 (×2)          → concat s4 (/16)  → dec5 double_conv
          → up3 (×4)          → concat s2 (/4)   → dec3 double_conv
          → up1 (×4, no skip) → dec1 double_conv → head
    """
    def __init__(self, num_classes=1, in_ch=3, global_ch=64,
                 backbone='resnet18', backbone_pretrained=True,
                 freeze_local_encoder=False,
                 ssl_checkpoint=None, bottleneck_dropout=0.1,
                 use_roi=False, roi_ch=1,
                 decoder_ch=None):
        super().__init__(
            num_classes=num_classes, in_ch=in_ch, global_ch=global_ch,
            backbone=backbone, backbone_pretrained=backbone_pretrained,
            freeze_local_encoder=freeze_local_encoder,
            ssl_checkpoint=ssl_checkpoint, bottleneck_dropout=bottleneck_dropout,
            use_roi=use_roi, roi_ch=roi_ch,
        )
        c1, c2, c3, c4, c5 = self.backbone.feature_info.channels()
        d = decoder_ch if decoder_ch is not None else c4

        # Replace the full-depth decoder with a 3-step version.
        del self.up5, self.dec5, self.up4, self.dec4, self.up3, self.dec3
        del self.up2, self.dec2, self.up1, self.dec1

        self.up5     = nn.ConvTranspose2d(c5 + global_ch + self.roi_ch, d,
                                          kernel_size=2, stride=2)
        self.s4_proj = nn.Conv2d(c4, d, kernel_size=1) if d != c4 else nn.Identity()
        self.dec5    = _double_conv(d + d, d)                   # + s4 (/16)

        self.up3     = nn.ConvTranspose2d(d, d, kernel_size=4, stride=4)
        self.s2_proj = nn.Conv2d(c2, d, kernel_size=1) if d != c2 else nn.Identity()
        self.dec3    = _double_conv(d + d, d)                   # + s2 (/4)

        self.up1     = nn.ConvTranspose2d(d, d, kernel_size=4, stride=4)
        self.dec1    = _double_conv(d, d)                        # no skip at full res

        self.head = nn.Conv2d(d, num_classes, kernel_size=1)

    def forward(self, local, glob, pos, roi=None):
        s1, s2, s3, s4, x = self.backbone(local)     # /2, /4, /8, /16, /32

        g_feat = self.global_enc(glob)
        g_ctx  = _crop_and_resize(g_feat, pos, *x.shape[2:])
        if self.use_roi:
            assert roi is not None, "use_roi=True but no roi tensor was passed to forward()"
            roi_ctx = _crop_and_resize(roi, pos, *x.shape[2:])
            x = torch.cat([x, g_ctx, roi_ctx], dim=1)
        else:
            x = torch.cat([x, g_ctx], dim=1)

        x = self.dec5(torch.cat([self.up5(x), self.s4_proj(s4)], dim=1))    # /16
        x = self.bot_drop(x)
        x = self.dec3(torch.cat([self.up3(x), self.s2_proj(s2)], dim=1))    # /4
        x = self.dec1(self.up1(x))                                          # full

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


class ImageNetPretrainedLocalGlobalUNetMinimalDecoder(ImageNetPretrainedLocalGlobalUNet):
    """
    Maximally stripped decoder for the ROI-primed setup.

    Design choices vs. the light decoder:
      - Same 2 skip connections (s4 at /16, s2 at /4)
      - decoder_ch defaults to 8 (chosen so a single GN group of 8 channels
        still works if we ever switch back; we use BatchNorm2d here instead to
        allow going even lower without tripping GroupNorm)
      - Single conv + BN + ReLU after each skip (no more double_conv)
      - Bilinear upsample instead of ConvTranspose2d — zero parameters
      - A 1×1 projection collapses the (c5 + global_ch + roi_ch) bottleneck
        stack to decoder_ch before the first upsample
      - No trailing dec1 block — final bilinear up + head only

    Trainable decoder params at decoder_ch=8 collapse from ~3.6M (full)
    / ~2.9M (light) / ~1.5M-ish (narrow light) to under 10k.
    """
    def __init__(self, num_classes=1, in_ch=3, global_ch=64,
                 backbone='resnet18', backbone_pretrained=True,
                 freeze_local_encoder=False,
                 ssl_checkpoint=None, bottleneck_dropout=0.1,
                 use_roi=False, roi_ch=1,
                 decoder_ch=8):
        super().__init__(
            num_classes=num_classes, in_ch=in_ch, global_ch=global_ch,
            backbone=backbone, backbone_pretrained=backbone_pretrained,
            freeze_local_encoder=freeze_local_encoder,
            ssl_checkpoint=ssl_checkpoint, bottleneck_dropout=bottleneck_dropout,
            use_roi=use_roi, roi_ch=roi_ch,
        )
        _, c2, _, c4, c5 = self.backbone.feature_info.channels()
        d = decoder_ch

        # Replace the full-depth decoder wholesale.
        del self.up5, self.dec5, self.up4, self.dec4, self.up3, self.dec3
        del self.up2, self.dec2, self.up1, self.dec1

        def _block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        self.bot_proj = nn.Conv2d(c5 + global_ch + self.roi_ch, d, kernel_size=1)
        self.s4_proj  = nn.Conv2d(c4, d, kernel_size=1)
        self.s2_proj  = nn.Conv2d(c2, d, kernel_size=1)

        self.dec5 = _block(d + d, d)
        self.dec3 = _block(d + d, d)

        self.head = nn.Conv2d(d, num_classes, kernel_size=1)

    def forward(self, local, glob, pos, roi=None):
        _, s2, _, s4, x = self.backbone(local)       # /4, /16, /32

        g_feat = self.global_enc(glob)
        g_ctx  = _crop_and_resize(g_feat, pos, *x.shape[2:])
        if self.use_roi:
            assert roi is not None, "use_roi=True but no roi tensor was passed to forward()"
            roi_ctx = _crop_and_resize(roi, pos, *x.shape[2:])
            x = torch.cat([x, g_ctx, roi_ctx], dim=1)
        else:
            x = torch.cat([x, g_ctx], dim=1)

        x = self.bot_proj(x)                                                      # /32, d
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True) # /16
        x = self.dec5(torch.cat([x, self.s4_proj(s4)], dim=1))                    # /16
        x = self.bot_drop(x)
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=True) # /4
        x = self.dec3(torch.cat([x, self.s2_proj(s2)], dim=1))                    # /4
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=True) # full

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


class ImageNetPretrainedLocalGlobalUNetHDGlobalUNet(ImageNetPretrainedLocalGlobalUNet):
    """
    EXPERIMENT 8C. Identical to ImageNetPretrainedLocalGlobalUNet but the global
    branch is _GlobalUNetHD: it ingests a high-res (e.g. 1024) global image and
    learns the downsample to the 256 working resolution, keeping the UNet
    bottleneck. ssl_checkpoint must come from SSLGlobalUNetHD pretraining.

    Config:
        "arch": {"type": "ImageNetPretrainedLocalGlobalUNetHDGlobalUNet", "args": {...}}
    """
    def _build_global_enc(self, in_ch, global_ch):
        return _GlobalUNetHD(in_ch=in_ch, out_ch=global_ch)


class ImageNetPretrainedLocalGlobalUNetHDGlobalCNN(ImageNetPretrainedLocalGlobalUNet):
    """
    EXPERIMENT 8D. Like 8C, but the global branch is _GlobalCNNHD — a plain
    strided CNN with no bottleneck (features flow straight 1024 → 256).
    ssl_checkpoint must come from SSLGlobalCNNHD pretraining.

    Config:
        "arch": {"type": "ImageNetPretrainedLocalGlobalUNetHDGlobalCNN", "args": {...}}
    """
    def _build_global_enc(self, in_ch, global_ch):
        return _GlobalCNNHD(in_ch=in_ch, out_ch=global_ch)


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


class _SSLGlobalHD(BaseModel):
    """
    SSL wrapper for the HD (1024-in / 256-out) global encoders (8C/8D).

    Same two-head reconstruction scheme as SSLGlobalUNet. Because these encoders
    emit features at input/4, the global reconstruction is produced at feature
    resolution and bilinearly upsampled back to the input size so the existing
    SSLTrainer (which compares against full-res clean_global + erase_mask) is
    unchanged. The encoder is stored as self.unet so the downstream `unet.`
    state-dict prefix used by the supervised models still matches.

    After pretraining, transfer self.unet weights to the matching supervised
    model's global_enc (8C → _GlobalUNetHD, 8D → _GlobalCNNHD).
    """
    def __init__(self, encoder, in_ch, out_ch):
        super().__init__()
        self.unet = encoder
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
        feat = self.unet(aug_global)                                    # (B, out_ch, H/4, W/4)
        global_recon = self.global_head(feat)                           # (B, in_ch, H/4, W/4)
        global_recon = F.interpolate(global_recon, size=aug_global.shape[2:],
                                     mode='bilinear', align_corners=True)
        cs = local_target.shape[-1]
        B, N = pos.shape[:2]                                            # pos: (B, N, 4)
        feat_exp = feat.repeat_interleave(N, dim=0)                    # (B*N, out_ch, H, W)
        local_feat = _crop_and_resize(feat_exp, pos.reshape(B * N, 4), cs, cs)
        local_recon = self.local_head(local_feat)                       # (B*N, in_ch, cs, cs)
        local_recon = local_recon.reshape(B, N, *local_recon.shape[1:]) # (B, N, in_ch, cs, cs)
        return global_recon, local_recon, feat


class SSLGlobalUNetHD(_SSLGlobalHD):
    """SSL pretraining wrapper for EXPERIMENT 8C (_GlobalUNetHD).

    Config:
        "arch": {"type": "SSLGlobalUNetHD", "args": {"in_ch": 1, "out_ch": 64}}
    """
    def __init__(self, in_ch=3, out_ch=64):
        super().__init__(_GlobalUNetHD(in_ch=in_ch, out_ch=out_ch), in_ch, out_ch)


class SSLGlobalCNNHD(_SSLGlobalHD):
    """SSL pretraining wrapper for EXPERIMENT 8D (_GlobalCNNHD).

    Config:
        "arch": {"type": "SSLGlobalCNNHD", "args": {"in_ch": 1, "out_ch": 64}}
    """
    def __init__(self, in_ch=3, out_ch=64):
        super().__init__(_GlobalCNNHD(in_ch=in_ch, out_ch=out_ch), in_ch, out_ch)


