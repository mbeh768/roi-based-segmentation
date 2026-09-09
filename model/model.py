import timm
import torch
import torch.nn as nn
from base import BaseModel


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
    ImageNet local encoder, U-Net decoder, with the ROI concatenated onto the
    bottleneck as roi_ch extra channels. That concatenation is the only point at
    which the network sees the ROI.

    Config:
        "arch": {"type": "ImageNetPretrainedLocalUNet", "args": {
            "num_classes": 1,
            "in_ch": 1,
            "backbone": "resnet18",
            "backbone_pretrained": true,
            "freeze_local_encoder": true,
            "bottleneck_dropout": 0.1,
            "use_roi": true,
            "roi_ch": 1,
            "skip_proj": true
        }}

    skip_proj adds an identity-initialised 1x1 convolution to each encoder skip
    before it is concatenated, as an adapter for the frozen ImageNet features.
    With it off the skips are passed through raw.

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
                 skip_proj=False):
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
        # Channel COUNTS per encoder stage, coarsest last. For resnet18:
        # c1=64 (/2), c2=64 (/4), c3=128 (/8), c4=256 (/16), c5=512 (/32).
        # The decoder is written in these same counts: each ConvTranspose emits
        # the count of the skip it is about to concatenate with, so the levels
        # are offset by one (up5 lifts the /32 bottleneck to /16, where skip4
        # lives) and no separate decoder namespace is needed.
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

        # ---- decoder --------------------------------------------------------
        # skip_proj puts a 1x1 convolution on each skip before concatenation.
        # The encoder is FROZEN ImageNet ResNet18, so the skips carry features
        # optimised for natural-image classification and reach a histology
        # decoder raw; a 1x1 is the cheapest adapter that lets the decoder
        # reweight and recombine those frozen channels.
        #
        # Every projection is SQUARE, so each is initialised to the exact
        # identity -- dirac_ weights, no bias. At initialisation this model IS
        # the unprojected one, tensor for tensor; the projections leave identity
        # only if training moves them. That is what makes it safe to enable
        # without prior validation: the starting point is the known-good
        # architecture. Cost is ~89k parameters, 2.8% on top of 3.23M.
        self.skip_proj = bool(skip_proj)

        def _skip(ch):
            if not self.skip_proj:
                return nn.Identity()
            # IDENTITY INITIALISED. dirac_ on a square 1x1 kernel writes a 1 on
            # the diagonal and 0 everywhere else, and bias=False leaves no
            # offset, so at step 0 this conv returns its input UNCHANGED --
            # the skip is passed through exactly as nn.Identity() would.
            conv = nn.Conv2d(ch, ch, kernel_size=1, bias=False)
            nn.init.dirac_(conv.weight)
            return conv

        # Each ConvTranspose emits the channel count of the skip it is about to
        # concatenate with, so `2 * cN` below is that upsampled tensor plus the
        # skip, not a widened level.
        self.up5   = nn.ConvTranspose2d(c5 + self.roi_ch, c4, kernel_size=2, stride=2)
        self.skip4 = _skip(c4)
        self.dec5  = _double_conv(2 * c4, c4)

        self.up4   = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.skip3 = _skip(c3)
        self.dec4  = _double_conv(2 * c3, c3)

        self.up3   = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.skip2 = _skip(c2)
        self.dec3  = _double_conv(2 * c2, c2)

        self.up2   = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.skip1 = _skip(c1)
        self.dec2  = _double_conv(2 * c1, c1)

        # up1 reaches full resolution, where there is no encoder skip, so dec1
        # is the one level whose input is not doubled.
        self.up1  = nn.ConvTranspose2d(c1, c1, kernel_size=2, stride=2)
        self.dec1 = _double_conv(c1, c1)

        self.head = nn.Conv2d(c1, num_classes, kernel_size=1)

    def forward(self, local, pos, roi=None):
        # pos is unused when use_roi=False: it exists to locate the ROI crop.
        s1, s2, s3, s4, x = self.backbone(local)

        if self.use_roi:
            assert roi is not None, "use_roi=True but no roi tensor was passed to forward()"

        def roi_at(h, w):
            """Exact ROI crop for each tile, no resampling.

            The pipeline is built so one ROI pixel maps to one bottleneck cell:
            a tile_size tile covers tile_size/tiling_size of the slide, so it
            spans roi_dim * tile_size / tiling_size ROI pixels, and the encoder
            reduces the same tile to tile_size/32 cells. At tiling 8192 with a
            512 tile and a 256 ROI both are 16, so the crop is already the
            bottleneck's size and any resample would be identity in shape while
            still blurring a binary mask through interpolation. Slicing instead
            keeps the ROI exactly {0, 1}.

            The assert is the guard on that symmetry: change tiling_size, tile
            size or ROI resolution without adjusting the others and it fires
            here rather than silently degrading the prior.
            """
            B, C, H, W = roi.shape
            y0 = (pos[:, 0] * H).round().long()
            x0 = (pos[:, 1] * W).round().long()
            y1 = (pos[:, 2] * H).round().long()
            x1 = (pos[:, 3] * W).round().long()
            assert torch.all(y1 - y0 == h) and torch.all(x1 - x0 == w), (
                f"ROI crop is {(y1 - y0).unique().tolist()}x{(x1 - x0).unique().tolist()} "
                f"but the target is {h}x{w}. The ROI/tile/bottleneck symmetry is "
                f"broken: with tile_size T, tiling_size S and an R-pixel ROI, "
                f"R*T/S must equal T/32."
            )
            ar = torch.arange(B, device=roi.device)
            iy = y0.view(B, 1) + torch.arange(h, device=roi.device).view(1, h)
            ix = x0.view(B, 1) + torch.arange(w, device=roi.device).view(1, w)
            # (B, h, w, C) from broadcast advanced indexing, then back to (B, C, h, w)
            return roi[ar.view(B, 1, 1), :, iy.view(B, h, 1), ix.view(B, 1, w)].permute(0, 3, 1, 2)

        if self.use_roi:
            x = torch.cat([x, roi_at(*x.shape[2:])], dim=1)

        def fuse(up, skip):
            return torch.cat([up, skip], dim=1)

        x = self.dec5(fuse(self.up5(x), self.skip4(s4)))
        x = self.bot_drop(x)
        x = self.dec4(fuse(self.up4(x), self.skip3(s3)))
        x = self.dec3(fuse(self.up3(x), self.skip2(s2)))
        x = self.dec2(fuse(self.up2(x), self.skip1(s1)))
        x = self.dec1(self.up1(x))

        return self.head(x)