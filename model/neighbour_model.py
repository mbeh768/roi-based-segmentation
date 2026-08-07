"""Model for EXPERIMENT 13: neighbour features fused at the bottleneck.

Subclasses ImageNetPretrainedLocalUNet so the 9I / champion class is untouched.
Everything below the first decoder convolution is inherited verbatim -- skips,
dec5..dec1, the head. Only two things change:

  bottleneck   the centre tile's s5 is concatenated with the 8 neighbours'
               pre-extracted s5, each with its own cropped ROI:
               9 x (512 + 1) = 4617 channels at the same 16x16 spatial size.
  reduction    a 1x1 conv takes 4617 -> 512 before up5, so up5 stays exactly
               the (513 -> 256) layer 9I had. Doing the reduction inside up5
               instead would cost 4.7M parameters -- more than the entire
               champion model -- and EXPERIMENT 12 already showed decoder
               capacity is not the binding constraint, so any gain here should
               come from context rather than from capacity.

Neighbours contribute through the same frozen backbone that produced the
centre's features, so the fusion is comparing like with like.
"""
import torch
import torch.nn as nn

from model.model import ImageNetPretrainedLocalUNet, _crop_and_resize


class NeighbourContextUNet(ImageNetPretrainedLocalUNet):
    def __init__(self, *args, n_neighbours=8, neighbour_proj_ch=512,
                 zero_neighbours=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_neighbours = n_neighbours
        # Capacity control. The 1x1 projection adds ~2.6M parameters, so a win
        # for this architecture would otherwise confound "context helped" with
        # "a bigger model helped". With zero_neighbours=True the parameter count
        # and every code path are identical but the neighbours contribute
        # nothing, so the difference between the two arms is context alone.
        self.zero_neighbours = zero_neighbours

        c5 = self.backbone.feature_info.channels()[-1]
        per_tile = c5 + self.roi_ch                     # one tile's features + its ROI
        fused_in = (n_neighbours + 1) * per_tile
        self.neighbour_proj = nn.Sequential(
            nn.Conv2d(fused_in, neighbour_proj_ch, kernel_size=1, bias=False),
            nn.GroupNorm(8, neighbour_proj_ch),
            nn.ReLU(inplace=True),
        )
        # up5 expects c5 + roi_ch; project back to exactly that so the rest of
        # the decoder is bit-identical to 9I's.
        self.neighbour_out = nn.Conv2d(neighbour_proj_ch, per_tile, kernel_size=1)
        self.fused_in = fused_in

    def forward(self, local, glob, pos, roi=None,
                nb_feats=None, nb_pos=None, nb_valid=None):
        s1, s2, s3, s4, x = self.backbone(local)

        inject_skips = self.use_roi and self.roi_inject == 'skips'
        if self.use_roi:
            assert roi is not None, "use_roi=True but no roi passed to forward()"

        def roi_at(h, w, p):
            return _crop_and_resize(roi, p, h, w, mode=self.roi_interp)

        h, w = x.shape[2:]
        if self.use_roi:
            x = torch.cat([x, roi_at(h, w, pos)], dim=1)

        if nb_feats is not None:
            if self.zero_neighbours:
                nb_valid = torch.zeros_like(nb_valid)
            parts = [x]
            for i in range(nb_feats.shape[1]):
                f = nb_feats[:, i]
                if self.use_roi:
                    # zero the ROI crop for absent neighbours too, so an edge tile
                    # contributes nothing rather than the ROI of a tile that is
                    # not there.
                    r = roi_at(h, w, nb_pos[:, i]) * nb_valid[:, i].view(-1, 1, 1, 1)
                    f = torch.cat([f, r], dim=1)
                parts.append(f * nb_valid[:, i].view(-1, 1, 1, 1))
            fused = torch.cat(parts, dim=1)
            assert fused.shape[1] == self.fused_in, (
                f"bottleneck got {fused.shape[1]}ch, expected {self.fused_in}")
            x = self.neighbour_out(self.neighbour_proj(fused))

        def fuse(up, skip):
            parts = [up, skip]
            if inject_skips:
                parts.append(roi_at(*up.shape[2:], pos))
            return torch.cat(parts, dim=1)

        x = self.dec5(fuse(self.up5(x), self.skip4(s4)))
        x = self.bot_drop(x)
        x = self.dec4(fuse(self.up4(x), self.skip3(s3)))
        x = self.dec3(fuse(self.up3(x), self.skip2(s2)))
        x = self.dec2(fuse(self.up2(x), self.skip1(s1)))
        u1 = self.up1(x)
        if inject_skips:
            u1 = torch.cat([u1, roi_at(*u1.shape[2:], pos)], dim=1)
        x = self.dec1(u1)
        return self.head(x)
