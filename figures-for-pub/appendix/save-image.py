"""Save a slide TIFF as a PNG, at full resolution or under a file-size budget.

--size full writes the native grid. --size <N>MB downsamples to the largest
whole-image scale whose encoded PNG fits in N MB: PNG size depends on content,
not just pixel count, so the scale is found by encoding and measuring rather
than predicted. A budget buys resolution, not field of view.

Usage:
    python save-image.py --path <slide.tif> --filename out.png --size full
    python save-image.py --path <slide.tif> --filename out.png --size 50MB
"""
import argparse
import io
import sys
from pathlib import Path

import cv2
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / 'train.py').exists())
sys.path.insert(0, str(ROOT / 'utils'))

from slide_render import load_slide_rgb    # noqa: E402

Image.MAX_IMAGE_PIXELS = None     # these are ~850 MPx; the bomb guard stops at 89

MB = 1024 * 1024
COMPRESS_LEVEL = 6
TOLERANCE = 0.90                  # accept a fit once it uses this much of the budget
MAX_PROBES = 8


def parse_size(text):
    """'full' -> None, '50' / '50MB' / '50 mb' -> bytes."""
    t = text.strip().lower()
    if t == 'full':
        return None
    t = t.removesuffix('mb').strip()
    try:
        mb = float(t)
    except ValueError:
        raise argparse.ArgumentTypeError(f"--size takes 'full' or a size in MB, not {text!r}")
    if mb <= 0:
        raise argparse.ArgumentTypeError('--size in MB must be positive')
    return int(mb * MB)


def encode(rgb, scale):
    """PNG bytes for `rgb` downsampled by `scale`, with the scaled dimensions."""
    if scale < 1.0:
        h, w = rgb.shape[:2]
        size = (max(1, round(w * scale)), max(1, round(h * scale)))
        small = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
    else:
        small = rgb
    buf = io.BytesIO()
    Image.fromarray(small).save(buf, format='PNG', compress_level=COMPRESS_LEVEL)
    return buf.getvalue(), small.shape[1], small.shape[0]


def fit_to_budget(rgb, budget):
    """Largest scale whose encoded PNG fits in `budget` bytes.

    Every probe resizes from the full-resolution master rather than from the
    previous probe, so a long search does not compound resampling loss. The
    first guess assumes bytes scale with pixel count -- true enough for a
    starting point, wrong enough to need the refinement that follows.
    """
    data, w, h = encode(rgb, 1.0)
    print(f'  probe 1: full {w}x{h} -> {len(data)/MB:.1f} MB')
    if len(data) <= budget:
        return data, w, h

    best = None
    scale = min(1.0, (budget / len(data)) ** 0.5)
    for probe in range(2, MAX_PROBES + 1):
        data, w, h = encode(rgb, scale)
        fit = 'fits' if len(data) <= budget else 'over'
        print(f'  probe {probe}: scale {scale:.4f} {w}x{h} -> {len(data)/MB:.1f} MB ({fit})')
        if len(data) <= budget:
            if best is None or len(data) > len(best[0]):
                best = (data, w, h)
            if len(data) >= TOLERANCE * budget:
                break
        scale *= (budget / len(data)) ** 0.5 * 0.99   # 0.99 biases the step under
        scale = min(scale, 1.0)

    if best is None:
        raise SystemExit(f'could not fit {budget/MB:.0f} MB in {MAX_PROBES} probes')
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--path', required=True, help='source slide TIFF')
    ap.add_argument('--filename', required=True, help='output PNG (path or bare name)')
    ap.add_argument('--size', required=True, type=parse_size,
                    help="'full' for native resolution, or a budget like 50MB")
    a = ap.parse_args()

    src = Path(a.path)
    if not src.is_file():
        raise SystemExit(f'no such file: {src}')
    out = Path(a.filename)
    if out.parent == Path('.') and not a.filename.startswith('.'):
        out = src.parent / out.name if out.is_absolute() else Path.cwd() / out.name
    out = out.with_suffix('.png')

    print(f'reading {src.name}')
    rgb = load_slide_rgb(src)

    if a.size is None:
        data, w, h = encode(rgb, 1.0)
    else:
        print(f'fitting {a.size/MB:.0f} MB')
        data, w, h = fit_to_budget(rgb, a.size)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    print(f'wrote {out}  {w}x{h}  {len(data)/MB:.1f} MB')


if __name__ == '__main__':
    main()
