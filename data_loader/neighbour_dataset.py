"""Dataset for EXPERIMENT 13: tiles served with their 8 neighbours' features.

Kept separate from LocalGlobalDataset so the 9I / champion path is untouched.
Subclasses it, so the tile inventory, ROI handling and roi_exclusion filtering
are inherited unchanged; only __getitem__ is replaced, to additionally serve the
pre-extracted bottleneck features of the 8 surrounding tiles.

Three things have to line up, and none of them fails loudly:

  orientation  the augmenter draws (hflip, vflip, rot90); the cached features
               must be read at the dihedral element that those parameters
               actually compose to (utils.dihedral.geo_params_to_dihedral),
               not at a freshly sampled one, or the neighbours disagree with
               the centre tile.
  slot order   rotating the grid moves a tile's west neighbour to the south, so
               the 9 slots must be permuted by the same element.
  pos          every neighbour needs its OWN normalised box, taken from its own
               coordinates.json entry -- reusing the centre's would hand all 9
               slots the same ROI crop and silently destroy the spatial
               correspondence the experiment is testing.

Returns (local, glob, pos, roi, nb_feats, nb_pos, nb_valid, mask, global_mask):
  nb_feats [8, C, h, w]  neighbour bottleneck features, slot order with the
                         centre removed (slots 0,1,2,3,5,6,7,8)
  nb_pos   [8, 4]        each neighbour's context window, augmented frame
  nb_valid [8]           0 where the neighbour falls outside the slide; its
                         features and ROI are zero-filled
"""
import json
from pathlib import Path

import torch

from data_loader.datasets import (LocalGlobalDataset, _sample_geo_params,
                                  _apply_geo_tensor, _apply_geo_pos)
from utils.dihedral import (NEIGHBOUR_OFFSETS, CENTRE_SLOT, neighbour_perm,
                            geo_params_to_dihedral)


class NeighbourContextDataset(LocalGlobalDataset):
    def __init__(self, *args, neighbour_offset=512, feature_dirname='extracted',
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.neighbour_offset = neighbour_offset
        self.feature_dirname = feature_dirname

        # (stem, y, x) -> filename, for O(1) neighbour lookup. Built from the
        # full coordinates.json rather than self.items, because roi_exclusion
        # drops tiles from training that are still perfectly good CONTEXT for a
        # tile that survived.
        self.grid = {}
        for coords_file in sorted(self.image_dir.glob('*/coordinates.json')):
            stem = coords_file.parent.name
            for e in json.loads(coords_file.read_text()):
                self.grid[(stem, e['y'], e['x'])] = e

        probe = next(iter(self.items))
        d = probe[0].parent / self.feature_dirname
        if not d.exists():
            raise FileNotFoundError(
                f"no pre-extracted features at {d}. Run extract_all.py first.")
        self.feat_shape = tuple(json.loads((d / 'manifest.json').read_text())
                                ['feature_shape'])

    def _feat_path(self, stem, entry, k):
        name = entry['filename'][:-3]
        return self.image_dir / stem / self.feature_dirname / f'{name}__d{k}.pt'

    def _neighbour_slots(self, stem, entry):
        """The 9 grid entries around (and including) this tile; None if absent."""
        off = self.neighbour_offset
        out = []
        for dy, dx in NEIGHBOUR_OFFSETS:
            out.append(self.grid.get((stem, entry['y'] + dy * off,
                                      entry['x'] + dx * off)))
        return out

    @staticmethod
    def _pos_of(entry):
        """Normalised [y0,x0,y1,x1] of an entry's context window (pre-augment)."""
        H, W = entry['image_H'], entry['image_W']
        y, x = entry['y'], entry['x']
        h, w = entry['height'], entry['width']
        ctx = entry.get('context_size', w)
        pad = (ctx - w) // 2
        cy, cx = y - pad, x - pad
        return torch.tensor([cy / H, cx / W, (cy + ctx) / H, (cx + ctx) / W],
                            dtype=torch.float32)

    def __getitem__(self, idx):
        patch_path, stem, entry = self.items[idx]

        local = torch.load(patch_path, weights_only=True)
        glob = torch.load(self.image_dir / stem / 'global.pt', weights_only=True)
        pos = self._pos_of(entry)

        mask_path = self.mask_dir / stem / patch_path.name
        mask = (torch.load(mask_path, weights_only=True) if mask_path.exists()
                else torch.zeros(1, local.shape[-2], local.shape[-1]))
        gmask_path = self.mask_dir / stem / 'global.pt'
        if not gmask_path.exists():
            raise FileNotFoundError(f"Global mask not found: {gmask_path}")
        global_mask = torch.load(gmask_path, weights_only=True)

        roi = None
        if self.require_roi:
            import random
            roi = torch.load(random.choice(self.stem_to_roi[stem]), weights_only=True)
            if self.roi_jitter > 0 and random.random() < self.roi_jitter:
                roi = torch.ones_like(roi)
            if self.roi_dropout > 0 and random.random() < self.roi_dropout:
                roi = torch.zeros_like(roi)

        # One draw, shared by the tile, its mask, its ROI and its neighbours.
        params = _sample_geo_params(self.aug_flags) if self.augment else (0, 0, 0)
        k = geo_params_to_dihedral(params)

        slots = self._neighbour_slots(stem, entry)
        perm = neighbour_perm(k)
        slots = [slots[perm[i]] for i in range(9)]          # rearrange, centre stays put

        nb_feats, nb_pos, nb_valid = [], [], []
        for i in range(9):
            if i == CENTRE_SLOT:
                continue                                     # centre comes from the live backbone
            e = slots[i]
            if e is None:                                    # off the edge of the slide
                nb_feats.append(torch.zeros(self.feat_shape, dtype=torch.float32))
                nb_pos.append(torch.zeros(4, dtype=torch.float32))
                nb_valid.append(0.0)
            else:
                f = torch.load(self._feat_path(stem, e, k), weights_only=True)
                nb_feats.append(f.float())
                nb_pos.append(_apply_geo_pos(self._pos_of(e), params))
                nb_valid.append(1.0)

        if self.augment:
            local = _apply_geo_tensor(local, params)
            glob = _apply_geo_tensor(glob, params)
            mask = _apply_geo_tensor(mask, params)
            global_mask = _apply_geo_tensor(global_mask, params)
            if roi is not None:
                roi = _apply_geo_tensor(roi, params)
            pos = _apply_geo_pos(pos, params)
        if self.photo_transform is not None:
            local = self.photo_transform(local)

        return (local, glob, pos, roi,
                torch.stack(nb_feats), torch.stack(nb_pos),
                torch.tensor(nb_valid, dtype=torch.float32),
                mask, global_mask)
