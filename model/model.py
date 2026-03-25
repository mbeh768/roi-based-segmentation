import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel


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


def _double_conv(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.GroupNorm(8, out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.GroupNorm(8, out_ch),
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


class _UNetEncoder(nn.Module):
    """Shared encoder backbone for LocalGlobalUNet."""
    def __init__(self):
        super().__init__()
        self.enc1       = _double_conv(3,    64)
        self.enc2       = _double_conv(64,  128)
        self.enc3       = _double_conv(128, 256)
        self.enc4       = _double_conv(256, 512)
        self.pool       = nn.MaxPool2d(2)
        self.bottleneck = _double_conv(512, 1024)

    def forward(self, x):
        s1  = self.enc1(x)                   # (B,   64, H,    W)
        s2  = self.enc2(self.pool(s1))       # (B,  128, H/2,  W/2)
        s3  = self.enc3(self.pool(s2))       # (B,  256, H/4,  W/4)
        s4  = self.enc4(self.pool(s3))       # (B,  512, H/8,  W/8)
        bot = self.bottleneck(self.pool(s4)) # (B, 1024, H/16, W/16)
        return bot, (s1, s2, s3, s4)


class _GlobalShallowEncoder(nn.Module):
    """
    Shallow stack of 5×5 convolutions (no padding, no pooling) for the global branch.

    Each layer shrinks spatial dims by 4 pixels. With num_layers=4 the receptive
    field is 17×17 pixels — small enough that features remain spatially specific,
    making the positional crop meaningful.

    Input:  (B, 3,       H,          W)
    Output: (B, out_ch,  H-4*layers, W-4*layers)
    """
    def __init__(self, out_ch=64, num_layers=32):
        super().__init__()
        layers = []
        in_ch = 3
        mid_ch = out_ch
        for i in range(num_layers):
            out = out_ch if i == num_layers - 1 else mid_ch
            layers += [
                nn.Conv2d(in_ch, out, kernel_size=5, padding=0, bias=False),
                nn.GroupNorm(8, out),
                nn.ReLU(inplace=True),
            ]
            in_ch = out
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)  # (B, out_ch, H', W')


class LocalGlobalUNet(BaseModel):
    """
    UNet that jointly encodes a local tile and a downsampled global image.

    Global context injected only at the bottleneck — global skips are dropped
    because at fine scales the global image (full slide compressed to 256×256)
    encodes different spatial detail than the local tile and adds noise.

    pos (B, 4) = [y0, x0, y1, x1] normalised is used to crop the global
    bottleneck to the tile's exact region, then resize to match the local
    bottleneck via bilinear grid_sample. Without pos the full global
    bottleneck is used unchanged.

    Decoder uses only local skip connections, same channel counts as SimpleUNet.

    Inputs: local (B, 3, H, W), glob (B, 3, H, W)
    Output: (B, num_classes, H, W) raw logits

    Config:
        "arch": {"type": "LocalGlobalUNet", "args": {"num_classes": 1}}
    """
    def __init__(self, num_classes=1, global_ch=64, global_layers=32):
        super().__init__()

        self.local_enc  = _UNetEncoder()
        self.global_enc = _GlobalShallowEncoder(out_ch=global_ch, num_layers=global_layers)

        # bottleneck: cat(local 1024, global global_ch) → 1024+global_ch
        self.up4  = nn.ConvTranspose2d(1024 + global_ch, 512, kernel_size=2, stride=2)
        self.dec4 = _double_conv(512 + 512, 512)    # up + local_s4

        self.up3  = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = _double_conv(256 + 256, 256)    # up + local_s3

        self.up2  = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = _double_conv(128 + 128, 128)    # up + local_s2

        self.up1  = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = _double_conv(64  + 64,  64)     # up + local_s1

        self.head     = nn.Conv2d(64, num_classes, kernel_size=1)
        self.aux_head = nn.Conv2d(global_ch, num_classes, kernel_size=1)

    def forward(self, local, glob, pos):
        l_bot, (l_s1, l_s2, l_s3, l_s4) = self.local_enc(local)
        g_feat = self.global_enc(glob)                               # (B, global_ch, H', W')

        g_ctx = _crop_and_resize(g_feat, pos, *l_bot.shape[2:])

        x = torch.cat([l_bot, g_ctx], dim=1)                        # (B, 1024+global_ch, H/16, W/16)

        x = self.dec4(torch.cat([self.up4(x), l_s4], dim=1))        # (B,  512, H/8)
        x = self.dec3(torch.cat([self.up3(x), l_s3], dim=1))        # (B,  256, H/4)
        x = self.dec2(torch.cat([self.up2(x), l_s2], dim=1))        # (B,  128, H/2)
        x = self.dec1(torch.cat([self.up1(x), l_s1], dim=1))        # (B,   64, H)

        main = self.head(x)                                          # (B, num_classes, H, W)

        if self.training:
            aux = F.interpolate(
                self.aux_head(g_feat),
                size=glob.shape[2:],
                mode='bilinear',
                align_corners=True,
            )                                                        # (B, num_classes, globalH, globalW)
            return main, aux

        return main
