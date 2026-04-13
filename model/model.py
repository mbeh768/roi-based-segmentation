import math

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
        global_recon = self.global_head(feat)                           # (B, 3, H, W)
        cs = local_target.shape[-1]
        local_feat = _crop_and_resize(feat, pos, cs, cs)               # (B, out_ch, cs, cs)
        local_recon  = self.local_head(local_feat)                      # (B, 3, cs, cs)
        return global_recon, local_recon, feat


def _dense_mlp(in_ch, hidden_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, hidden_ch, kernel_size=1, bias=False),
        nn.GroupNorm(8, hidden_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(hidden_ch, out_ch, kernel_size=1),
    )


class BYOLDenseUNet(BaseModel):
    """
    Dense BYOL wrapper around _GlobalUNet for pretraining.

    Two augmented views of the global image drive the main representation loss:
    online (unet + projector + predictor) vs EMA target (unet + projector,
    stop-grad), with per-pixel normalized-MSE symmetrized across views.

    An auxiliary crop-to-local decoder keeps the cropped global features
    semantically grounded: features cropped from feat1 at pos are upsampled by
    a throwaway local_head to predict the full-resolution local tile (pixel
    MSE). This mirrors the downstream task where the bottleneck crop feeds a
    decoder that reconstructs the local tile. Only unet, projector, predictor
    and local_head receive gradients; the local tile is never pushed through
    the unet.

    After pretraining, transfer self.unet weights to LocalGlobalUNet.global_enc
    (same convention as SSLGlobalUNet — keys start with "unet.").

    Config:
        "arch": {"type": "BYOLDenseUNet", "args": {
            "in_ch": 1, "out_ch": 64,
            "proj_hidden": 128, "proj_dim": 128,
            "momentum": 0.996,
            "crop_size": 16, "local_size": 256
        }}
    """
    def __init__(self, in_ch=1, out_ch=64, proj_hidden=128, proj_dim=128,
                 momentum=0.996, crop_size=16, local_size=256):
        super().__init__()
        self.momentum = momentum
        self.crop_size = crop_size
        self.local_size = local_size

        self.unet = _GlobalUNet(in_ch=in_ch, out_ch=out_ch)
        self.projector = _dense_mlp(out_ch, proj_hidden, proj_dim)
        self.predictor = _dense_mlp(proj_dim, proj_hidden, proj_dim)

        ratio = local_size / crop_size
        n_up = int(round(math.log2(ratio)))
        assert 2 ** n_up == int(ratio), \
            f"local_size/crop_size must be a power of two (got {local_size}/{crop_size})"
        layers = []
        for _ in range(n_up):
            layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
            layers.append(_double_conv(out_ch, out_ch))
        layers.append(nn.Conv2d(out_ch, in_ch, kernel_size=1))
        self.local_head = nn.Sequential(*layers)

        self.target_unet = _GlobalUNet(in_ch=in_ch, out_ch=out_ch)
        self.target_projector = _dense_mlp(out_ch, proj_hidden, proj_dim)
        self._copy_to_target()
        for p in self.target_unet.parameters():
            p.requires_grad = False
        for p in self.target_projector.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def _copy_to_target(self):
        for p_o, p_t in zip(self.unet.parameters(), self.target_unet.parameters()):
            p_t.data.copy_(p_o.data)
        for p_o, p_t in zip(self.projector.parameters(), self.target_projector.parameters()):
            p_t.data.copy_(p_o.data)

    @torch.no_grad()
    def ema_update(self, tau=None):
        m = self.momentum if tau is None else tau
        for p_o, p_t in zip(self.unet.parameters(), self.target_unet.parameters()):
            p_t.data.mul_(m).add_(p_o.data, alpha=1 - m)
        for p_o, p_t in zip(self.projector.parameters(), self.target_projector.parameters()):
            p_t.data.mul_(m).add_(p_o.data, alpha=1 - m)

    def forward(self, v1, v2, pos, g1, g2):
        feat1 = self.unet(v1)
        feat2 = self.unet(v2)
        z1 = self.predictor(self.projector(feat1))
        z2 = self.predictor(self.projector(feat2))

        # local_head decodes in v1's frame (pos and local_target are aligned
        # with v1 by the dataset).
        cs = self.crop_size
        local_feat = _crop_and_resize(feat1, pos, cs, cs)
        local_recon = self.local_head(local_feat)

        with torch.no_grad():
            t1 = self.target_projector(self.target_unet(v1))
            t2 = self.target_projector(self.target_unet(v2))

        # Each view had an independent geometric aug, so z_i / t_i live in
        # different frames. Unwarp to canonical so per-pixel dense BYOL
        # aligns corresponding spatial locations across views.
        z1 = _inverse_geo_feat(z1, g1)
        z2 = _inverse_geo_feat(z2, g2)
        t1 = _inverse_geo_feat(t1, g1)
        t2 = _inverse_geo_feat(t2, g2)

        return z1, z2, t1, t2, local_recon, feat1


class MAEGlobalUNet(BaseModel):
    """
    Masked-input UNet SSL: the UNet encodes a GPU-augmented + sparsely-masked
    global, producing dense features. For every tile in the global, those
    features are cropped at the tile's pos and decoded by a throwaway
    local_head to predict the full-res (un-augmented) local tile. Photometric
    aug and masking happen on the GPU here (not in the dataset) so targets
    stay clean and CPU work is minimal.

    After pretraining, transfer self.unet weights to LocalGlobalUNet.global_enc
    (keys start with "unet.", same convention as BYOLDenseUNet / SSLGlobalUNet).

    Config:
        "arch": {"type": "MAEGlobalUNet", "args": {
            "in_ch": 1, "out_ch": 64,
            "crop_size": 16, "local_size": 256,
            "mask_ratio": 0.5, "mask_patch_size": 16,
            "noise_std": 0.1, "color_scale_range": 0.15,
            "blur_kernel": 7, "p_noise": 0.3, "p_color": 0.3, "p_blur": 0.3,
            "p_intensity_invert": 0.0
        }}
    """
    def __init__(self, in_ch=1, out_ch=64,
                 crop_size=16, local_size=256, target_multiplier=2,
                 mask_ratio=0.5, mask_patch_size=16,
                 noise_std=0.1, color_scale_range=0.15, blur_kernel=7,
                 p_noise=0.3, p_color=0.3, p_blur=0.3, p_intensity_invert=0.0):
        super().__init__()
        self.crop_size = crop_size
        self.local_size = local_size
        self.target_multiplier = target_multiplier
        self.target_size = crop_size * target_multiplier
        self.mask_ratio = mask_ratio
        self.mask_patch_size = mask_patch_size
        self.noise_std = noise_std
        self.color_scale_range = color_scale_range
        self.blur_kernel = blur_kernel
        self.p_noise = p_noise
        self.p_color = p_color
        self.p_blur = p_blur
        self.p_intensity_invert = p_intensity_invert

        self.unet = _GlobalUNet(in_ch=in_ch, out_ch=out_ch)

        n_up = int(round(math.log2(target_multiplier)))
        assert 2 ** n_up == int(target_multiplier), \
            f"target_multiplier must be a power of two (got {target_multiplier})"
        layers = []
        for _ in range(n_up):
            layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
            layers.append(_double_conv(out_ch, out_ch))
        layers.append(nn.Conv2d(out_ch, in_ch, kernel_size=1))
        self.local_head = nn.Sequential(*layers)

    def _photo_augment(self, x):
        """GPU-side photometric augmentation on (B, C, H, W)."""
        B, C = x.shape[0], x.shape[1]
        if self.p_noise > 0:
            m = (torch.rand(B, 1, 1, 1, device=x.device) < self.p_noise).float()
            x = x + m * torch.randn_like(x) * self.noise_std
        if self.p_color > 0:
            m = (torch.rand(B, 1, 1, 1, device=x.device) < self.p_color).float()
            scale = 1.0 + (torch.rand(B, C, 1, 1, device=x.device) * 2 - 1) * self.color_scale_range
            x = x * (m * scale + (1 - m))
        if self.p_blur > 0:
            m = (torch.rand(B, 1, 1, 1, device=x.device) < self.p_blur).float()
            x_blur = self._gaussian_blur(x)
            x = m * x_blur + (1 - m) * x
        if self.p_intensity_invert > 0:
            m = (torch.rand(B, 1, 1, 1, device=x.device) < self.p_intensity_invert).float()
            x = (1 - m) * x + m * (-x)
        return x

    def _gaussian_blur(self, x):
        k = self.blur_kernel
        sigma = torch.empty(1, device=x.device).uniform_(0.1, 2.0).item()
        ax = torch.arange(k, device=x.device, dtype=x.dtype) - (k - 1) / 2
        g = torch.exp(-0.5 * (ax / sigma) ** 2)
        g = g / g.sum()
        kx = g.view(1, 1, 1, k).expand(x.shape[1], 1, 1, k)
        ky = g.view(1, 1, k, 1).expand(x.shape[1], 1, k, 1)
        pad = k // 2
        x = F.conv2d(x, kx, padding=(0, pad), groups=x.shape[1])
        x = F.conv2d(x, ky, padding=(pad, 0), groups=x.shape[1])
        return x

    def _sparse_mask(self, x):
        """Per-sample MAE-style patch masking: zero mask_ratio fraction of
        mask_patch_size x mask_patch_size blocks independently per sample."""
        if self.mask_ratio <= 0:
            return x
        B, _, H, W = x.shape
        ps = self.mask_patch_size
        gh, gw = H // ps, W // ps
        n_patches = gh * gw
        n_mask = int(round(n_patches * self.mask_ratio))
        if n_mask == 0:
            return x
        noise = torch.rand(B, n_patches, device=x.device)
        idx = noise.argsort(dim=1)[:, :n_mask]
        patch_mask = torch.ones(B, n_patches, device=x.device, dtype=x.dtype)
        patch_mask.scatter_(1, idx, 0)
        patch_mask = patch_mask.view(B, 1, gh, gw)
        pixel_mask = patch_mask.repeat_interleave(ps, dim=-2).repeat_interleave(ps, dim=-1)
        return x * pixel_mask

    def prepare_input(self, glob):
        """Photo-aug + sparse mask on GPU. Returns the tensor the encoder sees."""
        x = self._photo_augment(glob)
        x = self._sparse_mask(x)
        return x

    def encode(self, glob):
        """Photo-aug + mask on GPU, then run the UNet. Returns (feat, x_viewed)."""
        x = self.prepare_input(glob)
        return self.unet(x), x

    def decode(self, feat, pos):
        """Crop feat at pos and upsample to target_size via local_head.

        feat: (M, C, H, W), pos: (M, 4) — caller aligns batch dims.
        """
        local_feat = _crop_and_resize(feat, pos, self.crop_size, self.crop_size)
        return self.local_head(local_feat)

    def forward(self, glob):
        feat, _ = self.encode(glob)
        return feat
