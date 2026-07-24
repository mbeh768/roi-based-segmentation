"""
Plain intensity histogram of the red channel (cardiomyocyte stain) for each
case, full resolution, log-scale y-axis. No exclusion, no clipping, no
thresholding -- just the raw distribution, for looking at where tissue and
noise actually sit before picking a thresholding method.

Log-scale y-axis is used because the artifact zero-bin (see
epicardium_otsu_preview.py's docstring: scanner/stitching gaps, exactly
R=G=B=0) and the background population both vastly outnumber tissue pixels;
on a linear axis the rest of the histogram would be invisible. The exact
zero-bin count is annotated separately since even log-scale can make it hard
to read off precisely.

Reuses the full-res TIFF loading from epicardium_otsu_preview.py (the CASES
path table and the vips-based downsampler that handles this dataset's three
on-disk layouts) rather than re-deriving it.

Writes one PNG per case to red_channel_histograms/ in the project root.

Usage:
    python quant/red_channel_histogram.py
"""
import argparse
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from epicardium_otsu_preview import CASES, RGB_ROOT, DEFAULT_MAX_DIM, load_rgb_downsampled

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTDIR = REPO_ROOT / 'red_channel_histograms'


def process_case(stem, rel_path, outdir, max_dim, tmpdir):
    src = RGB_ROOT / rel_path
    if not src.exists():
        return f'{stem}: NOT FOUND ({src}) -- skipped'

    case_tmp = tmpdir / stem
    case_tmp.mkdir(parents=True, exist_ok=True)
    bgr, native_hw = load_rgb_downsampled(src, max_dim, case_tmp)
    for f in case_tmp.iterdir():
        f.unlink()
    case_tmp.rmdir()

    red = bgr[..., 2].ravel()
    zero_frac = float((red == 0).mean())

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(red, bins=256, range=(0, 255), color='#d62728')
    ax.set_yscale('log')
    ax.set_xlabel('red channel intensity (0-255)')
    ax.set_ylabel('pixel count (log scale)')
    ax.set_title(f'{stem}: red channel histogram  (native {native_hw[1]}x{native_hw[0]}, '
                f'preview {bgr.shape[1]}x{bgr.shape[0]})', fontsize=10)
    ax.text(0.98, 0.95, f'{100 * zero_frac:.1f}% of pixels at 0\n(scanner/stitching artifact)',
           transform=ax.transAxes, ha='right', va='top', fontsize=8,
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    fig.tight_layout()
    out_path = outdir / f'{stem}.png'
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return f'{stem}: native {native_hw[1]}x{native_hw[0]}, {100*zero_frac:.1f}% zero, wrote {out_path.name}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', type=Path, default=DEFAULT_OUTDIR)
    ap.add_argument('--max-dim', type=int, default=DEFAULT_MAX_DIM)
    args = ap.parse_args()

    if args.outdir.exists():
        for f in args.outdir.glob('*.png'):
            f.unlink()
    args.outdir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix='red_hist_vips_') as tmp:
        tmpdir = Path(tmp)
        for stem, rel_path in CASES.items():
            print(process_case(stem, rel_path, args.outdir, args.max_dim, tmpdir), flush=True)
    print(f'\nwrote histograms to {args.outdir}')


if __name__ == '__main__':
    main()
