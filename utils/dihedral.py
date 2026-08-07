"""Shared geometry for pre-extracted neighbour features (EXPERIMENT 13).

Both the extractor and the data loader need to agree on three things, and a
disagreement in any of them produces a model that trains to a plausible-looking
but wrong result rather than an error:

  1. which of the 8 dihedral orientations index k means,
  2. where each neighbour tile lands when the grid is transformed by k,
  3. which k corresponds to the (hflip, vflip, rot90) the augmenter drew.

(3) matters because the pipeline samples hflip/vflip/rot90 independently -- 16
combinations over the group's 8 elements -- and EXPERIMENT 13 must keep that
exact distribution to stay comparable to 9I. So the drawn parameters are mapped
onto a cache index rather than resampled from the group.

Everything here is derived by construction and checked by self_test(), not
hand-written, so the three cannot fall out of sync.
"""
import torch

# (k_rot90, hflip) applied as: hflip first, then rot90. Index = cache suffix __d<k>.
DIHEDRAL = [(0, False), (1, False), (2, False), (3, False),
            (0, True),  (1, True),  (2, True),  (3, True)]

# Neighbour slots, row-major over (dy, dx) in units of the neighbour offset:
#   0 NW  1 N  2 NE
#   3 W   4 C  5 E
#   6 SW  7 S  8 SE
NEIGHBOUR_OFFSETS = [(-1, -1), (-1, 0), (-1, 1),
                     ( 0, -1), ( 0, 0), ( 0, 1),
                     ( 1, -1), ( 1, 0), ( 1, 1)]
CENTRE_SLOT = 4


def apply_dihedral(t, k):
    """Apply dihedral element k to a [..., H, W] tensor."""
    rot, flip = DIHEDRAL[k]
    if flip:
        t = torch.flip(t, dims=[-1])
    if rot:
        t = torch.rot90(t, k=rot, dims=[-2, -1])
    return t


def apply_geo_params(t, params):
    """Apply the augmenter's (hflip, vflip, k) to a [..., H, W] tensor.

    Mirrors data_loader.datasets._apply_geo_tensor exactly (hflip -> vflip ->
    rot90); duplicated here so the mapping below can be verified without
    importing the dataset module.
    """
    hflip, vflip, k = params
    if hflip:
        t = torch.flip(t, [-1])
    if vflip:
        t = torch.flip(t, [-2])
    if k:
        t = torch.rot90(t, k, [-2, -1])
    return t


def neighbour_perm(k):
    """out_slot[i] = in_slot[perm[i]] under dihedral element k.

    Derived by transforming the slot-index grid itself, so it cannot drift out
    of step with apply_dihedral. A rotation moves a tile's west neighbour to the
    south; leaving slot order alone while transforming orientation would train
    the model on a scrambled spatial arrangement.
    """
    grid = torch.arange(9).reshape(3, 3)
    return apply_dihedral(grid, k).reshape(9).tolist()


def _build_param_map():
    """(hflip, vflip, k) -> dihedral index, found by matching actual transforms.

    hflip+vflip is a 180 degree rotation, so the 16 sampled combinations are
    2-to-1 onto the group's 8 elements. Matching empirically avoids a
    hand-derived composition rule being subtly wrong.
    """
    probe = torch.arange(36).reshape(1, 6, 6)
    canon = {}
    for d in range(8):
        canon[apply_dihedral(probe, d).flatten().tolist()] = d
    out = {}
    for h in (0, 1):
        for v in (0, 1):
            for k in range(4):
                key = apply_geo_params(probe, (h, v, k)).flatten().tolist()
                if key not in canon:
                    raise RuntimeError(f"(h={h},v={v},k={k}) is not a dihedral element")
                out[(h, v, k)] = canon[key]
    return out


PARAM_TO_DIHEDRAL = _build_param_map()


def geo_params_to_dihedral(params):
    """Cache index for the augmenter's (hflip, vflip, k)."""
    return PARAM_TO_DIHEDRAL[(int(params[0]), int(params[1]), int(params[2]))]


def self_test(verbose=True):
    """Check all three contracts. Cheap enough to run before any extraction."""
    ok = True

    # 1) transforming tiles individually then reordering == transforming the whole grid
    torch.manual_seed(0)
    T = 4
    tiles = [torch.randn(1, T, T) for _ in range(9)]
    big = torch.cat([torch.cat([tiles[r * 3 + c] for c in range(3)], -1)
                     for r in range(3)], -2)
    for k in range(8):
        perm = neighbour_perm(k)
        got_t = [apply_dihedral(tiles[perm[i]], k) for i in range(9)]
        got = torch.cat([torch.cat([got_t[r * 3 + c] for c in range(3)], -1)
                         for r in range(3)], -2)
        same = torch.equal(apply_dihedral(big, k), got)
        ok &= same
        if verbose:
            print(f"  k={k} rot={DIHEDRAL[k][0]} flip={str(DIHEDRAL[k][1]):<5} "
                  f"perm={perm}  {'OK' if same else 'MISMATCH'}")

    # 2) the centre never moves
    for k in range(8):
        if neighbour_perm(k)[CENTRE_SLOT] != CENTRE_SLOT:
            ok = False
            if verbose:
                print(f"  centre slot moved under k={k}  MISMATCH")

    # 3) the sampled params map to a transform that really is that group element
    probe = torch.arange(36).reshape(1, 6, 6)
    bad = 0
    for (h, v, k), d in sorted(PARAM_TO_DIHEDRAL.items()):
        if not torch.equal(apply_geo_params(probe, (h, v, k)), apply_dihedral(probe, d)):
            bad += 1
    ok &= bad == 0
    if verbose:
        hit = sorted(set(PARAM_TO_DIHEDRAL.values()))
        print(f"  param map: {len(PARAM_TO_DIHEDRAL)} combos -> {len(hit)} elements {hit}, "
              f"{bad} mismatched")
        print("\n  contracts:", "VERIFIED" if ok else "*** BROKEN ***")
    return ok


if __name__ == '__main__':
    raise SystemExit(0 if self_test() else 1)
