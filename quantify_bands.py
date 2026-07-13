"""
Generate transparent-background region maps from a ground-truth mask:
  subendocardium.png  -- the mask itself, blue
  100um.png            -- mask + everything within 100um outward, light blue
  200um.png            -- mask + everything within 200um outward, pale blue

Physical distances need microns/pixel. Do NOT trust TIFF resolution tags here --
verified against known CZI metadata they're bogus placeholders (e.g. a flat 300 DPI
that doesn't match the real optical calibration). Instead:
  - pass --microns-per-pixel directly, or
  - pass --czi pointing at the source .czi for this slide (voxel size is read from
    its Scaling metadata automatically), or
  - pass a .czi directly as --mask (voxel size auto-read from the same file).

Regions are cumulative (100um band already contains the mask; 200um already
contains the 100um band), matching "ground truth PLUS Xum offset outward".

Usage:
    python quantify_bands.py --mask <mask.tif> --microns-per-pixel 2.5562774308232 --outdir out/
    python quantify_bands.py --mask <mask.tif> --czi <source.czi> --outdir out/
    python quantify_bands.py --mask <mask.czi> --outdir out/   # voxel auto-read from the mask czi itself
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import tifffile

from pipeline import czi_native_dims_and_scale

# RGB (not BGR), straight alpha, 0-255 -- converted to BGRA at write time in save_rgba()
COLOR_SUBENDOCARDIUM = (20, 80, 200)    # blue
COLOR_100UM = (100, 170, 255)           # light blue
COLOR_200UM = (190, 220, 255)           # pale blue


def _load_mask_tif(path):
    arr = tifffile.imread(str(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    # Some source masks aren't clean 0/255 (e.g. a low near-zero "tissue present"
    # fill value alongside the true annotation) -- threshold at the midpoint rather
    # than >0 so stray low values don't get counted as foreground.
    return arr >= 128


def _load_mask_czi(path):
    from pylibCZIrw import czi as pyczi
    with pyczi.open_czi(str(path)) as czidoc:
        plane = czidoc.read(plane={'C': 0})
    arr = np.asarray(plane)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr > 0


def load_mask(path):
    suffix = Path(path).suffix.lower()
    if suffix == '.czi':
        return _load_mask_czi(path)
    return _load_mask_tif(path)


def resolve_microns_per_pixel(args):
    if args.microns_per_pixel is not None:
        print(f"microns/pixel: {args.microns_per_pixel:.6f}  (CLI override)")
        return float(args.microns_per_pixel)

    czi_source = args.czi or (args.mask if Path(args.mask).suffix.lower() == '.czi' else None)
    if czi_source is None:
        raise ValueError(
            "no microns/pixel available: pass --microns-per-pixel explicitly, "
            "or --czi <source.czi>, or point --mask directly at a .czi."
        )
    _, _, um_x, um_y = czi_native_dims_and_scale(czi_source)
    if um_x is None:
        raise ValueError(f"could not read microns/pixel from CZI metadata of {czi_source}; "
                         f"pass --microns-per-pixel explicitly.")
    if um_y is not None and abs(um_x - um_y) / um_x > 0.01:
        print(f"WARNING: anisotropic scaling (X={um_x:.4f}, Y={um_y:.4f} um/px); using X.")
    print(f"microns/pixel: {um_x:.6f}  (read from {Path(czi_source).name})")
    return float(um_x)


def save_rgba(mask_bool, color_rgb, path):
    """Write a foreground-only RGBA PNG: color where mask_bool, alpha 0 elsewhere."""
    h, w = mask_bool.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[mask_bool, 0:3] = color_rgb
    rgba[mask_bool, 3] = 255
    # cv2 expects BGRA
    bgra = rgba[..., [2, 1, 0, 3]]
    cv2.imwrite(str(path), bgra)
    print(f"saved -> {path}  ({int(mask_bool.sum())} px, {100*mask_bool.mean():.2f}% of frame)")


def main(args):
    mask = load_mask(args.mask)
    print(f"mask: {args.mask}  shape={mask.shape}  foreground={int(mask.sum())}px "
          f"({100*mask.mean():.2f}%)")

    um_per_px = resolve_microns_per_pixel(args)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # subendocardium: the ground-truth mask itself
    save_rgba(mask, COLOR_SUBENDOCARDIUM, outdir / "subendocardium.png")

    # cumulative outward distance bands via Euclidean distance transform
    dist_px = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5)
    dist_um = dist_px * um_per_px

    band_100 = mask | (dist_um <= 100.0)
    save_rgba(band_100, COLOR_100UM, outdir / "100um.png")

    band_200 = mask | (dist_um <= 200.0)
    save_rgba(band_200, COLOR_200UM, outdir / "200um.png")

    print(f"\ndone -> {outdir}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--mask', required=True, help='ground-truth mask (.tif/.tiff or .czi)')
    parser.add_argument('--czi', default=None,
                        help='source .czi to auto-read microns/pixel from (ignored if '
                             '--microns-per-pixel is given or --mask is itself a .czi)')
    parser.add_argument('--microns-per-pixel', type=float, default=None,
                        help='override/skip auto voxel-size lookup entirely')
    parser.add_argument('--outdir', required=True)
    main(parser.parse_args())
