import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel


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


class LocalGlobalUNet(BaseModel):
    """
    UNet that jointly encodes a local tile and a downsampled global image.

    Both inputs pass through independent encoders. The decoder follows the
    local encoder's skip connections, with global encoder features concatenated
    at every scale (bottleneck + each decoder stage).

    If pos (B, 4) = [y0, x0, y1, x1] normalised is provided, a Gaussian bias
    is applied to every global skip connection to focus them on the tile's
    approximate location in the global feature map.

    Decoder input channels are up + local_skip + global_skip at each level,
    roughly tripling vs a plain UNet decoder but keeping the output channels
    the same as SimpleUNet.

    Inputs: local (B, 3, H, W), glob (B, 3, H, W)
    Output: (B, num_classes, H, W) raw logits

    Config:
        "arch": {"type": "LocalGlobalUNet", "args": {"num_classes": 1}}
    """
    def __init__(self, num_classes=1):
        super().__init__()

        self.local_enc  = _UNetEncoder()
        self.global_enc = _UNetEncoder()

        # bottleneck: cat(local 1024, global 1024) → 2048
        self.up4  = nn.ConvTranspose2d(2048, 512, kernel_size=2, stride=2)
        self.dec4 = _double_conv(512 + 512 + 512, 512)   # up + local_s4 + global_s4

        self.up3  = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = _double_conv(256 + 256 + 256, 256)   # up + local_s3 + global_s3

        self.up2  = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = _double_conv(128 + 128 + 128, 128)   # up + local_s2 + global_s2

        self.up1  = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = _double_conv(64  + 64  + 64,  64)    # up + local_s1 + global_s1

        self.head = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, local, glob, pos=None):
        l_bot, (l_s1, l_s2, l_s3, l_s4) = self.local_enc(local)
        g_bot, (g_s1, g_s2, g_s3, g_s4) = self.global_enc(glob)

        if pos is not None:
            def bias(feat):
                b = _make_region_bias(pos, *feat.shape[2:], feat.device)
                return feat * (1 + b)
            g_bot = bias(g_bot)
            g_s4  = bias(g_s4)
            g_s3  = bias(g_s3)
            g_s2  = bias(g_s2)
            g_s1  = bias(g_s1)

        x = torch.cat([l_bot, g_bot], dim=1)                           # (B, 2048, H/16, W/16)

        x = self.dec4(torch.cat([self.up4(x), l_s4, g_s4], dim=1))    # (B,  512, H/8)
        x = self.dec3(torch.cat([self.up3(x), l_s3, g_s3], dim=1))    # (B,  256, H/4)
        x = self.dec2(torch.cat([self.up2(x), l_s2, g_s2], dim=1))    # (B,  128, H/2)
        x = self.dec1(torch.cat([self.up1(x), l_s1, g_s1], dim=1))    # (B,   64, H)

        return self.head(x)                                             # (B, num_classes, H, W)
