import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
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
        self.pool       = nn.MaxPool2d(2)
        self.bottleneck = _double_conv(256, 512)

    def forward(self, x):
        s1  = self.enc1(x)                   # (B,  64, H,   W)
        s2  = self.enc2(self.pool(s1))       # (B, 128, H/2, W/2)
        s3  = self.enc3(self.pool(s2))       # (B, 256, H/4, W/4)
        bot = self.bottleneck(self.pool(s3)) # (B, 512, H/8, W/8)
        return bot, (s1, s2, s3)


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
    def __init__(self, out_ch=64):
        super().__init__()
        self.pool = nn.MaxPool2d(2)

        self.enc1 = _double_conv(3,   32)
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


class LocalGlobalUNet3Layer(BaseModel):
    """
    3-encoder-layer variant of LocalGlobalUNet. Bottleneck at H/8.

    Config:
        "arch": {"type": "LocalGlobalUNet3Layer", "args": {"num_classes": 1}}
    """
    def __init__(self, num_classes=1, global_ch=64):
        super().__init__()

        self.local_enc  = _UNetEncoder()
        self.global_enc = _GlobalUNet(out_ch=global_ch)

        # bottleneck: cat(local 512, global global_ch) → 512+global_ch
        self.up3  = nn.ConvTranspose2d(512 + global_ch, 256, kernel_size=2, stride=2)
        self.dec3 = _double_conv(256 + 256, 256)    # up + local_s3

        self.up2  = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = _double_conv(128 + 128, 128)    # up + local_s2

        self.up1  = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = _double_conv(64  + 64,  64)     # up + local_s1

        self.head     = nn.Conv2d(64, num_classes, kernel_size=1)
        self.aux_head = nn.Conv2d(global_ch, num_classes, kernel_size=1)

    def forward(self, local, glob, pos):
        l_bot, (l_s1, l_s2, l_s3) = self.local_enc(local)
        g_feat = self.global_enc(glob)

        g_ctx = _crop_and_resize(g_feat, pos, *l_bot.shape[2:])
        x = torch.cat([l_bot, g_ctx], dim=1)                        # (B, 512+global_ch, H/8)

        x = self.dec3(torch.cat([self.up3(x), l_s3], dim=1))        # (B, 256, H/4)
        x = self.dec2(torch.cat([self.up2(x), l_s2], dim=1))        # (B, 128, H/2)
        x = self.dec1(torch.cat([self.up1(x), l_s1], dim=1))        # (B,  64, H)

        main = self.head(x)

        if self.training:
            aux = F.interpolate(
                self.aux_head(g_feat),
                size=glob.shape[2:],
                mode='bilinear', align_corners=True,
            )
            return main, aux
        return main


class LocalGlobalUNet(BaseModel):
    """
    4-encoder-layer LocalGlobalUNet. Bottleneck at H/16.

    Encoder: 3→64→128→256→512, bottleneck 512→512 @ H/16.
    Global context (_GlobalUNet output, H×W) cropped to the bottleneck spatial
    size and concatenated before decoding.

    Inputs: local (B, 3, H, W), glob (B, 3, H, W), pos (B, 4)
    Output: (B, num_classes, H, W) raw logits

    Config:
        "arch": {"type": "LocalGlobalUNet", "args": {"num_classes": 1}}
    """
    def __init__(self, num_classes=1, global_ch=64):
        super().__init__()

        self.pool = nn.MaxPool2d(2)
        self.enc1 = _double_conv(3,    64)
        self.enc2 = _double_conv(64,  128)
        self.enc3 = _double_conv(128, 256)
        self.enc4 = _double_conv(256, 512)
        self.bot  = _double_conv(512, 512)           # H/16

        self.global_enc = _GlobalUNet(out_ch=global_ch)

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
        x  = self.bot(self.pool(s4))                                 # (B, 512, H/16)

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
            )                                                        # (B, num_classes, globalH, globalW)
            return main, aux

        return main


class ImageNetUNetShallow(BaseModel):
    """
    Identical to ImageNetUNet but uses _GlobalShallowEncoder for the global branch.

    _GlobalShallowEncoder applies num_layers of 5×5 no-padding convolutions,
    preserving near-full spatial resolution (output: H - 4*num_layers).
    For a 256×256 global image with num_layers=32 → 128×128 feature map,
    giving a tile crop of ~16×16 before _crop_and_resize to H/32 bottleneck.

    Config:
        "arch": {"type": "ImageNetUNetShallow", "args": {"num_classes": 1}}
    """
    def __init__(self, num_classes=1, global_ch=64, global_layers=32):
        super().__init__()

        self.encoder = timm.create_model(
            'resnet18', pretrained=True,
            features_only=True, out_indices=(0, 1, 2, 3, 4),
        )
        for p in self.encoder.parameters():
            p.requires_grad = False

        self.global_enc = _GlobalShallowEncoder(out_ch=global_ch, num_layers=global_layers)

        self.bottleneck = _double_conv(512, 512)       # refine s4 @ H/32

        self.up4  = nn.ConvTranspose2d(512 + global_ch, 256, kernel_size=2, stride=2)
        self.dec4 = _double_conv(256 + 256, 256)       # up + s3(256ch) → H/16

        self.up3  = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = _double_conv(128 + 128, 128)       # up + s2(128ch) → H/8

        self.up2  = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = _double_conv(64  +  64,  64)       # up + s1(64ch)  → H/4

        self.up1  = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec1 = _double_conv(64  +  64,  64)       # up + s0(64ch)  → H/2

        self.up0  = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec0 = _double_conv(32, 32)               # no skip        → H

        self.head     = nn.Conv2d(32, num_classes, kernel_size=1)
        self.aux_head = nn.Conv2d(global_ch, num_classes, kernel_size=1)

    def forward(self, local, glob, pos):
        with torch.no_grad():
            s0, s1, s2, s3, s4 = self.encoder(local)  # H/2, H/4, H/8, H/16, H/32

        bot   = self.bottleneck(s4)                    # (B, 512, H/32, W/32)

        g_feat = self.global_enc(glob)
        g_ctx  = _crop_and_resize(g_feat, pos, *bot.shape[2:])

        x = torch.cat([bot, g_ctx], dim=1)             # (B, 512+global_ch, H/32)

        x = self.dec4(torch.cat([self.up4(x), s3], dim=1))  # (B, 256, H/16)
        x = self.dec3(torch.cat([self.up3(x), s2], dim=1))  # (B, 128, H/8)
        x = self.dec2(torch.cat([self.up2(x), s1], dim=1))  # (B,  64, H/4)
        x = self.dec1(torch.cat([self.up1(x), s0], dim=1))  # (B,  64, H/2)
        x = self.dec0(self.up0(x))                          # (B,  32, H)

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


class ImageNetUNet(BaseModel):
    """
    UNet with a frozen ImageNet-pretrained ResNet-18 encoder.

    ResNet-18 produces 5 feature stages:
        s0:  64 ch @ H/2   (stem conv)
        s1:  64 ch @ H/4   (layer1)
        s2: 128 ch @ H/8   (layer2)
        s3: 256 ch @ H/16  (layer3)
        s4: 512 ch @ H/32  (layer4 — bottleneck)

    Global context is cropped to the bottleneck spatial size (H/32) and
    concatenated before decoding. Skips from s0–s3 feed the decoder;
    a final no-skip step (H/2 → H) completes the output.

    Inputs: local (B, 3, H, W), glob (B, 3, H, W), pos (B, 4)
    Output: (main_logits, aux_logits) during training, main_logits during eval

    Config:
        "arch": {"type": "ImageNetUNet", "args": {"num_classes": 1}}
    """
    def __init__(self, num_classes=1, global_ch=64, n_down=2):
        super().__init__()

        self.encoder = timm.create_model(
            'resnet18', pretrained=True,
            features_only=True, out_indices=(0, 1, 2, 3, 4),
        )
        for p in self.encoder.parameters():
            p.requires_grad = False

        self.global_enc = _GlobalUNet(out_ch=global_ch)

        self.bottleneck = _double_conv(512, 512)       # refine s4 @ H/32

        # Decoder: inject global at bottleneck, step through s3 → s0 → final up
        self.up4  = nn.ConvTranspose2d(512 + global_ch, 256, kernel_size=2, stride=2)
        self.dec4 = _double_conv(256 + 256, 256)       # up + s3(256ch) → H/16

        self.up3  = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = _double_conv(128 + 128, 128)       # up + s2(128ch) → H/8

        self.up2  = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = _double_conv(64  +  64,  64)       # up + s1(64ch)  → H/4

        self.up1  = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec1 = _double_conv(64  +  64,  64)       # up + s0(64ch)  → H/2

        self.up0  = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec0 = _double_conv(32, 32)               # no skip        → H

        self.head     = nn.Conv2d(32, num_classes, kernel_size=1)
        self.aux_head = nn.Conv2d(global_ch, num_classes, kernel_size=1)

    def forward(self, local, glob, pos):
        with torch.no_grad():
            s0, s1, s2, s3, s4 = self.encoder(local)  # H/2, H/4, H/8, H/16, H/32

        bot   = self.bottleneck(s4)                    # (B, 512, H/32, W/32)

        g_feat = self.global_enc(glob)
        g_ctx  = _crop_and_resize(g_feat, pos, *bot.shape[2:])

        x = torch.cat([bot, g_ctx], dim=1)             # (B, 512+global_ch, H/32)

        x = self.dec4(torch.cat([self.up4(x), s3], dim=1))  # (B, 256, H/16)
        x = self.dec3(torch.cat([self.up3(x), s2], dim=1))  # (B, 128, H/8)
        x = self.dec2(torch.cat([self.up2(x), s1], dim=1))  # (B,  64, H/4)
        x = self.dec1(torch.cat([self.up1(x), s0], dim=1))  # (B,  64, H/2)
        x = self.dec0(self.up0(x))                          # (B,  32, H)

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


class PretrainedLocalGlobalUNet(BaseModel):
    """
    Identical to LocalGlobalUNet but loads SSL-pretrained weights into global_enc.

    Config:
        "arch": {"type": "PretrainedLocalGlobalUNet", "args": {
            "num_classes": 1,
            "ssl_checkpoint": "saved/models/SSL_.../checkpoint-epoch50.pth"
        }}
    """
    def __init__(self, num_classes=1, global_ch=64, ssl_checkpoint=None):
        super().__init__()

        self.pool = nn.MaxPool2d(2)
        self.enc1 = _double_conv(3,    64)
        self.enc2 = _double_conv(64,  128)
        self.enc3 = _double_conv(128, 256)
        self.enc4 = _double_conv(256, 512)
        self.bot  = _double_conv(512, 512)           # H/16

        self.global_enc = _GlobalUNet(out_ch=global_ch)

        if ssl_checkpoint is not None:
            ckpt = torch.load(ssl_checkpoint, map_location='cpu', weights_only=False)
            ssl_sd = ckpt['state_dict']
            global_enc_sd = {k.replace('unet.', ''): v
                             for k, v in ssl_sd.items() if k.startswith('unet.')}
            self.global_enc.load_state_dict(global_enc_sd)
            # freeze all, then unfreeze first encoder + last decoder
            for p in self.global_enc.parameters():
                p.requires_grad = False
            for p in self.global_enc.enc1.parameters():
                p.requires_grad = True
            for p in self.global_enc.dec1.parameters():
                p.requires_grad = False
            for p in self.global_enc.up1.parameters():
                p.requires_grad = False
            print(f"[PretrainedLocalGlobalUNet] Loaded SSL global_enc from {ssl_checkpoint} "
                  f"(enc1/dec1/up1 unfrozen)")

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
        x  = self.bot(self.pool(s4))                                 # (B, 512, H/16)

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


class SSLGlobalUNet(BaseModel):
    """
    Self-supervised wrapper around _GlobalUNet for pretraining.

    The UNet produces dense 64-ch spatial features. Two lightweight 1x1 conv
    heads project those features to RGB for two reconstruction tasks:
      1) aug_head:   reverse augmentation on the full global image
      2) local_head: reconstruct a local tile from features cropped at its position

    After pretraining, transfer self.unet weights to LocalGlobalUNet.global_enc.

    Config:
        "arch": {"type": "SSLGlobalUNet", "args": {"out_ch": 64}}
    """
    def __init__(self, out_ch=64):
        super().__init__()
        self.unet = _GlobalUNet(out_ch=out_ch)
        # throwaway decoder heads — discarded after pretraining
        self.global_head = nn.Sequential(
            _double_conv(out_ch, out_ch),
            _double_conv(out_ch, out_ch),
            nn.Conv2d(out_ch, 3, kernel_size=1),
        )
        self.local_head = nn.Sequential(
            _double_conv(out_ch, out_ch),
            _double_conv(out_ch, out_ch),
            nn.Conv2d(out_ch, 3, kernel_size=1),
        )

    def forward(self, aug_global, pos, local_target):
        feat = self.unet(aug_global)                                    # (B, out_ch, H, W)
        global_recon = self.global_head(feat)                           # (B, 3, H, W)
        cs = local_target.shape[-1]
        local_feat = _crop_and_resize(feat, pos, cs, cs)               # (B, out_ch, cs, cs)
        local_recon  = self.local_head(local_feat)                      # (B, 3, cs, cs)
        return global_recon, local_recon, feat
