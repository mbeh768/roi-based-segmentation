"""Model for EXPERIMENT 14: capacity rebalanced toward high resolution.

Separate module so model.py's champion class is untouched.

Motivation. The champion decoder tapers 256, 128, 64, 64, 64 as resolution
doubles, which puts 71% of the trainable parameters in up5+dec5 at 32x32 and
only 3% in up1+dec1 at 512x512 -- where boundary decisions are actually made.
EXPERIMENT 11A measured 41.2% of the residual held-out error as boundary
imprecision (within +/-8px of the true edge), the single largest bucket and
three times the island error the topology work chased.

So the question is not "is there enough capacity" -- EXPERIMENT 12 answered
that, with 5G reaching within 0.0096 of the champion on 18% of the parameters --
but "is the capacity in the wrong place".

The test is deliberately parameter-MATCHED rather than larger:

    champion   256, 128,  64,  64,  64   3,231,617 params,  6.7% at 256^2+512^2
    this       96,   96, 128, 192, 192   3,002,369 params, 56.1% at 256^2+512^2

93% of the champion's parameter count, so a difference cannot be attributed to
capacity in either direction, while the share of parameters operating at the two
finest resolutions goes from 6.7% to 56%. A win says placement matters; a null
says the boundary error is not a capacity-placement problem -- most likely it is
annotation precision, which no architecture recovers.

Everything else is EXPERIMENT 9I: frozen ImageNet resnet18 local encoder, ROI at
the bottleneck, bce_dice, same 14-fold leave-one-case-out protocol.
"""
import torch
import torch.nn as nn

from model.model import ImageNetPretrainedLocalUNet, _double_conv, _crop_and_resize


class RebalancedDecoderUNet(ImageNetPretrainedLocalUNet):
    """Champion architecture with per-level decoder widths.

    decoder_widths is (dec5, dec4, dec3, dec2, dec1), coarsest first. Skips are
    1x1-projected onto the level width where the encoder channel count does not
    already match, which is what makes arbitrary per-level widths possible at
    all -- the champion relies on w_k happening to equal the skip's channels.
    """

    def __init__(self, *args, decoder_widths=(96, 96, 128, 192, 192), **kwargs):
        # Build the parent first (backbone, freezing, ROI config), then discard
        # and replace its decoder. Wasteful by a few MB at construction, but it
        # keeps every other behaviour inherited rather than duplicated.
        super().__init__(*args, **kwargs)
        if len(decoder_widths) != 5:
            raise ValueError(f"decoder_widths needs 5 entries, got {decoder_widths}")
        if self.decoder_half_depth:
            raise ValueError("decoder_half_depth is not supported here; this "
                             "experiment varies width at full depth")
        self.decoder_widths = tuple(decoder_widths)

        c1, c2, c3, c4, c5 = self.backbone.feature_info.channels()
        w5, w4, w3, w2, w1 = self.decoder_widths
        s = self.skip_roi_ch

        for w in self.decoder_widths:
            if w % 8:
                raise ValueError(f"decoder width {w} must be divisible by 8 "
                                 f"(GroupNorm(8, ch) in _double_conv)")

        self.up5 = nn.ConvTranspose2d(c5 + self.roi_ch, w5, kernel_size=2, stride=2)
        self.skip4 = nn.Conv2d(c4, w5, 1) if c4 != w5 else nn.Identity()
        self.dec5 = _double_conv(w5 + w5 + s, w5)

        self.up4 = nn.ConvTranspose2d(w5, w4, kernel_size=2, stride=2)
        self.skip3 = nn.Conv2d(c3, w4, 1) if c3 != w4 else nn.Identity()
        self.dec4 = _double_conv(w4 + w4 + s, w4)

        self.up3 = nn.ConvTranspose2d(w4, w3, kernel_size=2, stride=2)
        self.skip2 = nn.Conv2d(c2, w3, 1) if c2 != w3 else nn.Identity()
        self.dec3 = _double_conv(w3 + w3 + s, w3)

        self.up2 = nn.ConvTranspose2d(w3, w2, kernel_size=2, stride=2)
        self.skip1 = nn.Conv2d(c1, w2, 1) if c1 != w2 else nn.Identity()
        self.dec2 = _double_conv(w2 + w2 + s, w2)

        self.up1 = nn.ConvTranspose2d(w2, w1, kernel_size=2, stride=2)
        self.dec1 = _double_conv(w1 + s, w1)

        self.head = nn.Conv2d(w1, self.head.out_channels, kernel_size=1)

    # forward() is inherited verbatim: the parent's full-depth path already
    # routes through skip4/skip3/skip2/skip1, so only the module widths change.
