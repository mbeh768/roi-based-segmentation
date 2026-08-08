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


def _crop_and_resize(feat, pos, out_h, out_w, mode='bilinear'):
    """
    Differentiably crop feat (B, C, H, W) at the normalised region
    pos (B, 4) = [y0, x0, y1, x1] in [0, 1], then resize to (out_h, out_w)
    via F.grid_sample. mode='bilinear' (default) for smooth feature maps;
    mode='nearest' keeps a low-res binary ROI crisp when upsampled.
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

    # align_corners only applies to bilinear; grid_sample ignores it for nearest.
    return F.grid_sample(feat, grid, mode=mode, align_corners=True, padding_mode='border')


def _double_conv(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.GroupNorm(8, out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.GroupNorm(8, out_ch),
        nn.ReLU(inplace=True),
    )


class ImageNetPretrainedLocalUNet(BaseModel):
    """
    ImageNet local encoder + optional ROI concatenated at the bottleneck
    (and, with roi_inject='skips', at every decoder skip connection).

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

    freeze_local_encoder accepts either a bool (True = freeze all stages,
    False = freeze none) or a list of stage indices in [0..4] to freeze
    individually. Stages map to ResNet modules as:
        0 -> conv1 + bn1   (stem, /2)
        1 -> layer1        (/4)
        2 -> layer2        (/8)
        3 -> layer3        (/16)
        4 -> layer4        (/32)
    """
    _STAGE_MODULES = {
        0: ['conv1', 'bn1'],
        1: ['layer1'],
        2: ['layer2'],
        3: ['layer3'],
        4: ['layer4'],
    }

    def __init__(self, num_classes=1, in_ch=3,
                 backbone='resnet18', backbone_pretrained=True,
                 freeze_local_encoder=False,
                 bottleneck_dropout=0.1,
                 use_roi=False, roi_ch=1,
                 roi_inject='bottleneck', roi_interp='bilinear',
                 decoder_half_depth=False, decoder_ch=None):
        super().__init__()

        self.use_roi = use_roi
        self.roi_ch = roi_ch if use_roi else 0

        # roi_inject controls where the ROI guidance enters the decoder:
        #   'bottleneck' (default) — single coarse crop at /32, original behaviour.
        #   'skips'                — additionally fused at every decoder concat
        #                            (/16,/8,/4,/2,/1) so fine ROI detail reaches
        #                            the resolution where pixel decisions are made.
        # roi_interp is the resampling mode for ROI crops ('nearest' keeps a
        # low-res binary ROI crisp when upsampled; 'bilinear' for the legacy path).
        if roi_inject not in ('bottleneck', 'skips'):
            raise ValueError(f"roi_inject must be 'bottleneck' or 'skips', got {roi_inject!r}")
        if roi_interp not in ('bilinear', 'nearest'):
            raise ValueError(f"roi_interp must be 'bilinear' or 'nearest', got {roi_interp!r}")
        self.roi_inject = roi_inject
        self.roi_interp = roi_interp
        skip_roi_ch = self.roi_ch if (use_roi and roi_inject == 'skips') else 0
        self.skip_roi_ch = skip_roi_ch

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

        # ---- decoder ablations (EXPERIMENT 5D-5I) --------------------------
        # decoder_half_depth (5D/5E): drop every other decoder level, so each
        #   remaining ConvTranspose upsamples by 4 instead of 2 and skips are
        #   consumed every other stage (s4 and s2 survive; s3 and s1 do not).
        # decoder_ch (5F-5I): run the whole decoder at one fixed width, with 1x1
        #   projections on the surviving skips. Most trainable parameters live in
        #   up5+dec5 because c5=512/c4=256 dominate, so this is where the cuts
        #   actually land. Stacks on half-depth, matching 5F ("identical to 5D
        #   otherwise").
        self.decoder_half_depth = bool(decoder_half_depth)
        self.decoder_ch = decoder_ch
        d = decoder_ch      # None => keep the original per-level widths

        if self.decoder_half_depth:
            # Levels kept: /32 -> /16 -> /4 -> /1, so the surviving skips are s4
            # and s2 -- the pair 5F names ("1x1 projections on the s4 and s2
            # skips"). s3 and s1 are dropped, and the two later ConvTransposes
            # upsample 4x to span the gap.
            w5 = d if d else c4
            self.up5 = nn.ConvTranspose2d(c5 + self.roi_ch, w5, kernel_size=2, stride=2)
            self.skip4 = nn.Conv2d(c4, w5, 1) if d else nn.Identity()
            self.dec5 = _double_conv(w5 + (d if d else c4) + skip_roi_ch, w5)

            w3 = d if d else c2
            self.up3 = nn.ConvTranspose2d(w5, w3, kernel_size=4, stride=4)
            self.skip2 = nn.Conv2d(c2, w3, 1) if d else nn.Identity()
            self.dec3 = _double_conv(w3 + (d if d else c2) + skip_roi_ch, w3)

            w1 = d if d else c1
            self.up1 = nn.ConvTranspose2d(w3, w1, kernel_size=4, stride=4)
            self.dec1 = _double_conv(w1 + skip_roi_ch, w1)
            head_in = w1
        else:
            w5, w4, w3, w2, w1 = (d, d, d, d, d) if d else (c4, c3, c2, c1, c1)
            self.up5  = nn.ConvTranspose2d(c5 + self.roi_ch, w5, kernel_size=2, stride=2)
            self.skip4 = nn.Conv2d(c4, w5, 1) if d else nn.Identity()
            self.dec5 = _double_conv(w5 + w5 + skip_roi_ch, w5)
            self.up4  = nn.ConvTranspose2d(w5, w4, kernel_size=2, stride=2)
            self.skip3 = nn.Conv2d(c3, w4, 1) if d else nn.Identity()
            self.dec4 = _double_conv(w4 + w4 + skip_roi_ch, w4)
            self.up3  = nn.ConvTranspose2d(w4, w3, kernel_size=2, stride=2)
            self.skip2 = nn.Conv2d(c2, w3, 1) if d else nn.Identity()
            self.dec3 = _double_conv(w3 + w3 + skip_roi_ch, w3)
            self.up2  = nn.ConvTranspose2d(w3, w2, kernel_size=2, stride=2)
            self.skip1 = nn.Conv2d(c1, w2, 1) if d else nn.Identity()
            self.dec2 = _double_conv(w2 + w2 + skip_roi_ch, w2)
            self.up1  = nn.ConvTranspose2d(w2, w1, kernel_size=2, stride=2)
            self.dec1 = _double_conv(w1 + skip_roi_ch, w1)
            head_in = w1

        self.head = nn.Conv2d(head_in, num_classes, kernel_size=1)

    def forward(self, local, glob, pos, roi=None):
        # glob is accepted for signature compatibility with LocalGlobalDataLoader batches
        # but not used; same for pos when use_roi=False.
        s1, s2, s3, s4, x = self.backbone(local)

        inject_skips = self.use_roi and self.roi_inject == 'skips'
        if self.use_roi:
            assert roi is not None, "use_roi=True but no roi tensor was passed to forward()"

        def roi_at(h, w):
            # crop the per-image ROI to this tile (pos) and resample to (h, w)
            return _crop_and_resize(roi, pos, h, w, mode=self.roi_interp)

        if self.use_roi:
            x = torch.cat([x, roi_at(*x.shape[2:])], dim=1)

        def fuse(up, skip):
            parts = [up, skip]
            if inject_skips:
                parts.append(roi_at(*up.shape[2:]))
            return torch.cat(parts, dim=1)

        if self.decoder_half_depth:
            # s3 and s1 are deliberately unused: half-depth keeps every other skip
            x = self.dec5(fuse(self.up5(x), self.skip4(s4)))
            x = self.bot_drop(x)
            x = self.dec3(fuse(self.up3(x), self.skip2(s2)))
        else:
            x = self.dec5(fuse(self.up5(x), self.skip4(s4)))
            x = self.bot_drop(x)
            x = self.dec4(fuse(self.up4(x), self.skip3(s3)))
            x = self.dec3(fuse(self.up3(x), self.skip2(s2)))
            x = self.dec2(fuse(self.up2(x), self.skip1(s1)))
        u1 = self.up1(x)
        if inject_skips:
            u1 = torch.cat([u1, roi_at(*u1.shape[2:])], dim=1)
        x = self.dec1(u1)

        return self.head(x)


# EXPERIMENT 13 lives in its own module so this one stays exactly as 9I had it,
# but config-driven instantiation resolves arch types by getattr on THIS module.
# Importing at the bottom (rather than the top) is safe despite neighbour_model
# importing from here: by this point the classes above are defined, and this
# module is already in sys.modules, so the cycle resolves.
from model.neighbour_model import NeighbourContextUNet  # noqa: E402,F401
from model.rebalanced_model import RebalancedDecoderUNet  # noqa: E402,F401
