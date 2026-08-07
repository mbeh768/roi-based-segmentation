"""Pre-extract frozen-backbone bottleneck features for every tile (EXPERIMENT 13).

Runs the same ImageNet resnet18 local encoder as EXPERIMENT 9I over every tile and
caches the bottleneck (s5) feature map to <tile_dir>/extracted/. The model then
concatenates a tile's 8 neighbours at the bottleneck, giving each tile a
1536x1536 receptive field instead of 512x512 without touching the skips.

Why caching is valid here: the local encoder is frozen (freeze_local_encoder,
established in 4D/9A), and the pipeline applies NO photometric augmentation --
datasets.LocalGlobalDataset sets photo_transform = None and nothing assigns it.
So a tile's pixels are deterministic up to the geometric augmentations, which
form the dihedral group of the square. All 8 elements are extracted, so training
reads the features the backbone would genuinely have produced on the augmented
tile rather than a feature-space approximation of one.

Output layout, one file per tile per orientation:
    <data_dir>/images/<stem>/extracted/<patch_name>__d<k>.pt      float16 [512,16,16]
    <data_dir>/images/<stem>/extracted/manifest.json

Orientation k indexes DIHEDRAL below: k = 0..3 are rot90 by k with no flip,
k = 4..7 are the same rotations after a horizontal flip. This ordering is
mirrored in data_loader/datasets.py, which must also permute the neighbour
slots -- rotating a tile moves its west neighbour to the south, and getting the
orientation right while leaving slot order alone would train the model on a
scrambled arrangement that still looks plausible. See DIHEDRAL/NEIGHBOUR_PERM.

Usage:
    python extract_all.py -c config.json                 # all stems
    python extract_all.py -c config.json --stems A B     # subset
    python extract_all.py -c config.json --verify        # self-check, no writing
"""
import argparse
import json
from pathlib import Path

import torch

# Single source of truth, shared with the data loader -- see utils/dihedral.py
# for why the orientation index, the neighbour permutation and the augmenter
# parameter mapping must all be derived rather than hand-written.
from utils.dihedral import (DIHEDRAL, NEIGHBOUR_OFFSETS, apply_dihedral,
                            neighbour_perm, self_test)


def build_encoder(cfg, device):
    import model.model as module_arch
    arch = dict(cfg['arch'])
    m = getattr(module_arch, arch['type'])(**arch['args']).to(device)
    m.eval()
    for p in m.parameters():
        p.requires_grad = False
    return m


def verify(device=None):
    """Check the shared geometry contracts before writing 40GB of cache."""
    return self_test(verbose=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('-c', '--config', default='config.json')
    ap.add_argument('--data-dir', default=None, help='override config data_dir')
    ap.add_argument('--stems', nargs='*', default=None)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--verify', action='store_true', help='run the self-check and exit')
    ap.add_argument('--overwrite', action='store_true')
    a = ap.parse_args()

    if a.verify:
        raise SystemExit(0 if verify(torch.device('cpu')) else 1)

    cfg = json.loads(Path(a.config).read_text())
    data_dir = Path(a.data_dir or cfg['data_loader']['args']['data_dir'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("checking dihedral/permutation contract first:")
    if not verify(device):
        raise SystemExit("refusing to extract: neighbour permutation is wrong")
    print()

    enc = build_encoder(cfg, device)
    img_root = data_dir / 'images'
    stems = a.stems or sorted(p.name for p in img_root.iterdir() if p.is_dir())
    print(f"extracting {len(stems)} stems from {data_dir}")

    total = 0
    for si, stem in enumerate(stems, 1):
        sd = img_root / stem
        entries = json.loads((sd / 'coordinates.json').read_text())
        out = sd / 'extracted'
        out.mkdir(exist_ok=True)
        done = 0
        for s in range(0, len(entries), a.batch_size):
            chunk = entries[s:s + a.batch_size]
            names = [e['filename'] for e in chunk]
            if not a.overwrite and all((out / f"{n[:-3]}__d7.pt").exists() for n in names):
                done += len(chunk)
                continue
            tiles = torch.stack([torch.load(sd / n, weights_only=True) for n in names]).to(device)
            for k in range(8):
                with torch.no_grad():
                    feats = enc.backbone(apply_dihedral(tiles, k))[-1]   # s5 bottleneck
                feats = feats.to(torch.float16).cpu()
                for j, n in enumerate(names):
                    torch.save(feats[j].clone(), out / f"{n[:-3]}__d{k}.pt")
            done += len(chunk)
        (out / 'manifest.json').write_text(json.dumps({
            'stem': stem, 'n_tiles': len(entries), 'orientations': 8,
            'dihedral': [list(x) for x in DIHEDRAL],
            'neighbour_offsets': NEIGHBOUR_OFFSETS,
            'feature_shape': list(feats.shape[1:]), 'dtype': 'float16',
            'source_config': str(a.config),
        }, indent=2))
        total += done
        print(f"  [{si}/{len(stems)}] {stem:<24} {done} tiles x 8 orientations")
    print(f"\ndone: {total} tiles, {total*8} feature files under */extracted/")


if __name__ == '__main__':
    main()
