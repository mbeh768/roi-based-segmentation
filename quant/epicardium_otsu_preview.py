"""
Otsu-tissue preview at full resolution, downsampled for display.

Earlier version of this script worked off the 256x256 preprocessed `global.pt`
tensors and subtracted the hand-drawn ROI to approximate "epicardium = tissue
- ventricle". Both of those turned out to be the wrong call: the 256px globals
needed workarounds (percentile scaling, keeping zero pixels in the Otsu
histogram) that only existed because of that downsampling, and the ROI
subtraction was usually a near no-op since most ROIs trace the dark
ventricular cavity, which Otsu already excludes on its own -- see
quantify_intensity.py's docstring for the same finding applied to the real
quantification pipeline.

This version reads the real full-resolution RGB TIFFs from the "RGB FULL -
dataset" share, downsamples each to at most 2000px on the long side, and runs
a plain global Otsu threshold on the red channel (the cardiomyocyte stain) --
no ROI involved anywhere. Writes one 2-panel PNG per case to
epicardium_previews/ in the project root (replaces the old contents).

Downsampling goes through the `vipsthumbnail` CLI (libvips), not
tifffile+cv2: these files are 1.5-5 GB uncompressed with a single strip per
page, so tifffile.imread pulls the entire page into RAM before anything can
shrink it, and the naive uint16->uint8 rescale used elsewhere in this repo
(quantify_intensity.load_rgb, done on the full-res array) upcasts to float64
first -- for the largest slide here that's ~20 GB for one array, on a machine
with 31 GB total, and it (along with a couple of dtype copies alongside it)
took the whole system down once already. libvips streams the file and
shrinks region-by-region, so peak memory stays near the thumbnail size
regardless of source resolution.

Three on-disk layouts show up in this dataset (confirmed via `vipsheader -f
bands` / `-f n-pages`):
  1. a standard interleaved RGB TIFF (1 page, 3 bands), which vips reads as a
     single 3-band image in one call;
  2. an ImageJ-style stack of 3 separate single-band grayscale pages, one per
     channel in R,G,B order (vips reports 1 band, 3 pages -- ordinary
     `vipsthumbnail` on this layout silently thumbnails page 0 only, i.e. just
     the red channel);
  3. the same ImageJ 3-channel stack, but large enough that ImageJ wrote it as
     a single classic-TIFF IFD with the 3 channel planes packed contiguously
     after it (raw, uncompressed) rather than as 3 separate IFDs -- vips (and
     tifffile) both see this as "1 page, 1 band" with no way to address planes
     1 and 2 through normal page indexing. Confirmed via the file's ImageJ
     description tag (`images`/`channels` = 3) plus dataoffsets/databytecounts
     from tifffile (header-only, no pixel data read) matching a simple
     `offset0 + i * plane_bytes` layout. Each plane is then pulled with `vips
     rawload` at its computed byte offset.
This script picks the right path per file based on that header.

Usage:
    python quant/epicardium_otsu_preview.py
    python quant/epicardium_otsu_preview.py --outdir <dir> --max-dim 2000
"""
import argparse
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import tifffile

VIPS_FORMAT = {'uint8': 'uchar', 'uint16': 'ushort'}

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTDIR = REPO_ROOT / 'epicardium_previews'
RGB_ROOT = Path("/mnt/8TBhdd/datasets/mayo_datasets/"
                "Max B_files_from Behfar_Shared folder(formatted)/"
                "Segmentation_Project/RGB FULL - dataset")

# stem -> path to the full-res RGB TIFF, relative to RGB_ROOT. Folder/file
# naming in this share is inconsistent (spaces, "of2" vs "1of2", parenthetical
# suffixes, .tif vs .tiff vs .ome.tiff), so this is hand-resolved per stem
# rather than pattern-matched.
CASES = {
    '17P391_7A': '17P391_7A/17P391_7A_image.tif',
    '17P645-1': '17P645-1 (validated) (channel issue)/17P645-1_image.ome.tiff',
    '21PControl-1': '21PControl-1/21PControl-1_image.tiff',
    '21PControl-11-1of2': '21PControl-11-1of2/21PControl-11-1of2_image.ome.tiff',
    '21PControl-11-2of2': '21PControl-11-2of2/21PControl-11-2of2_image.ome.tiff',
    '21PControl-2': '21PControl-2/21PControl-2_image.tiff',
    '21PControl_4': '21PControl_4/21PControl_4_image.tif',
    '21PControl-9-1of2': '21PControl-9-1of2/21PControl-9-1of2_image.tif',
    '21PControl-9-2of2': '21PControl-9-2of2/21PControl-9-2of2_image.ome.tiff',
    'AJK056-10-2of2': 'AJK056-10-2of2/AJK056-10-2of2_image.tiff',
    'AJKV056-10-1-of-2': 'AJK056-10-1of2/AJKV056-10-1-of-2_image.tiff',
    'AJKV056-7': 'AJKV056_7/AJKV056-7_image.tif',
    'AJKV056-8': 'AJKV056_8/AJKV056-8_image.tiff',
    'AJKY121-7-2': 'AJKY121_7/AJKY121-7-2_image.tif',
    'AJKY121-8-1': 'AJKY121_8/AJKY121-8-1_image.tif',
    'AJKY121-8-2': 'AJKY121_8/AJKY121-8-2_image.tif',
    'AJKY121-9-1': 'AJKY121_9/AJKY121-9-1_image.tif',
    'AJKY121-9-2': 'AJKY121_9/AJKY121-9-2_image.tif',
    'FL_22P003-6_1of2': 'FL_22P003-6_1of2/FL_22P003-6_1of2_image.tif',
    'FL_22P003-6_2of2': 'FL_22P003-6_2of2/FL_22P003-6_2of2_image.tif',
}

TISSUE_COLOR = (255, 0, 255)  # BGR magenta
ALPHA = 0.45
DEFAULT_MAX_DIM = 2000


def vips_field(path, field):
    out = subprocess.run(['vipsheader', '-f', field, str(path)],
                         capture_output=True, text=True, check=True)
    return out.stdout.strip()


def native_dims(path):
    wh = vips_field(path, 'width'), vips_field(path, 'height')
    return int(wh[1]), int(wh[0])  # (H, W)


def vips_thumb(spec, max_dim, out_png):
    """spec is a path, optionally with a vips filename suffix like '[page=0]'."""
    subprocess.run(['vipsthumbnail', spec, '--size', f'{max_dim}x{max_dim}',
                    '-o', str(out_png)],
                   check=True, capture_output=True, text=True)


def packed_plane_layout(path):
    """If `path` is an ImageJ multi-channel TIFF written as a single IFD with
    the channel planes packed contiguously after it (see module docstring,
    layout 3), return (n_planes, plane_offsets, (H, W), vips_format).
    Otherwise return None. Reads only the header/first-IFD metadata via
    tifffile, never the pixel data."""
    with tifffile.TiffFile(str(path)) as tf:
        if len(tf.pages) != 1:
            return None
        meta = tf.imagej_metadata
        if not meta or meta.get('images', 1) <= 1:
            return None
        pg = tf.pages[0]
        n_planes = int(meta['images'])
        offset0, nbytes0 = pg.dataoffsets[0], pg.databytecounts[0]
        h, w = pg.shape
        expected = h * w * pg.dtype.itemsize
        if nbytes0 != expected:
            return None  # not the simple one-strip-per-plane case we know how to address
        vfmt = VIPS_FORMAT.get(pg.dtype.name)
        if vfmt is None:
            return None
        offsets = [offset0 + i * nbytes0 for i in range(n_planes)]
        return n_planes, offsets, (h, w), vfmt


def vips_thumb_raw_plane(path, offset, h, w, vfmt, max_dim, tmpdir, tag):
    """Thumbnail one packed plane by pointing `vips rawload` at its byte
    offset, then shrinking with vipsthumbnail -- same streaming approach as
    the normal paths, just with the offset computed by hand instead of read
    from a real IFD."""
    raw_v = tmpdir / f'raw_{tag}.v'
    subprocess.run(['vips', 'rawload', str(path), str(raw_v), str(w), str(h), '1',
                    '--offset', str(offset), '--format', vfmt],
                   check=True, capture_output=True, text=True)
    out_png = tmpdir / f'raw_{tag}_thumb.png'
    vips_thumb(str(raw_v), max_dim, out_png)
    raw_v.unlink()
    return cv2.imread(str(out_png), cv2.IMREAD_UNCHANGED)


def load_rgb_downsampled(path, max_dim, tmpdir):
    """Downsample the full-res RGB TIFF to at most max_dim on the long side via
    vipsthumbnail (see module docstring for why: avoids ever materializing the
    full-resolution array in this process). Returns a BGR uint8 image."""
    h, w = native_dims(path)
    bands = int(vips_field(path, 'bands'))
    n_pages = int(vips_field(path, 'n-pages'))
    packed = packed_plane_layout(path) if (bands == 1 and n_pages == 1) else None

    if bands == 3:
        # Standard interleaved RGB -- one vipsthumbnail call handles it, and
        # vips already resolves the R/G/B band order correctly on its own.
        out_png = tmpdir / 'rgb.png'
        vips_thumb(str(path), max_dim, out_png)
        bgr = cv2.imread(str(out_png), cv2.IMREAD_COLOR)
    elif bands == 1 and n_pages >= 3:
        # ImageJ-style stack: 3 separate single-band pages in R,G,B order.
        # Thumbnail each page independently, then stack into BGR ourselves --
        # plain vipsthumbnail on this layout would silently take page 0 only.
        chans = []
        for i in range(3):
            out_png = tmpdir / f'ch{i}.png'
            vips_thumb(f'{path}[page={i}]', max_dim, out_png)
            chans.append(cv2.imread(str(out_png), cv2.IMREAD_UNCHANGED))
        r, g, b = chans
        bgr = np.stack([b, g, r], axis=-1)
    elif packed is not None:
        # Layout 3 (see module docstring): 3 channel planes packed
        # contiguously behind a single IFD -- address each by hand.
        n_planes, offsets, (ph, pw), vfmt = packed
        chans = [vips_thumb_raw_plane(path, offsets[i], ph, pw, vfmt, max_dim, tmpdir, i)
                 for i in range(3)]
        r, g, b = chans
        bgr = np.stack([b, g, r], axis=-1)
    else:
        raise ValueError(f"unrecognized layout for {path}: bands={bands} n_pages={n_pages}")

    if bgr.dtype == np.uint16:
        bgr = np.round(bgr.astype(np.float32) / 257.0).clip(0, 255).astype(np.uint8)
    elif bgr.dtype != np.uint8:
        raise ValueError(f"unexpected dtype {bgr.dtype} for {path}")

    return bgr, (h, w)


def otsu_tissue(bgr, raw=False):
    """Global Otsu on the red channel (cardiomyocyte stain), foreground = the
    brighter class -- same definition as quantify_intensity.tissue_foreground,
    duplicated here rather than imported since that module requires a mask
    argument this preview doesn't have.

    Completely black pixels (R=G=B=0) are excluded from the histogram before
    thresholding. These aren't background -- background still has some low
    fluorescence -- they're scanner/stitching artifacts (gaps between scan
    tiles, unscanned borders), visible in the previews as a black grid
    overlaid on the tissue. Left in, they stack up as a spike at 0 that
    drags Otsu's split down, and since the grid lines cut across tissue
    everywhere (not just at the true tissue/background edge), they degrade
    the threshold's quality generally, not just at the footprint boundary.

    Unless raw=True, the remaining values are then cleaned up (see
    clip_tails) before the split: a handful of very bright, spatially
    localized artifact pixels otherwise dominate Otsu's between-class
    variance criterion and drag the threshold up (capped at the 95th
    percentile), and uniform low-level background noise sits right next to
    real dark tissue and adds a fuzzy low-end tail that blurs the split
    (floored using a statistical estimate of the noise level, not a fixed
    cutoff). raw=True skips that step entirely -- bare Otsu on the
    real-pixel values, nothing else."""
    red = bgr[..., 2]
    real = np.any(bgr > 0, axis=-1)
    if not real.any():
        return np.zeros(red.shape, dtype=bool)
    adjusted = red if raw else clip_tails(red, real)
    thresh, _ = cv2.threshold(adjusted[real].reshape(-1, 1), 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return real & (adjusted > thresh)


def estimate_noise_floor(vals, k=3.0):
    """Statistically estimate the uniform low-level background noise floor,
    instead of assuming a fixed fraction of pixels is noise: a first-pass Otsu
    split gives a rough tissue/background boundary, the pixels below it are
    treated as the noise population, and the floor is set k robust standard
    deviations (median absolute deviation, scaled to be std-equivalent under
    a normal assumption) above their median. This adapts per slide -- a
    quieter scan gets a lower floor, a noisier one a higher floor -- rather
    than a percentile that's blind to how much noise is actually there.

    This targets uniform low-level noise specifically, not the extremely
    localized bright hotspots (scanner artifacts) -- those are handled
    separately in clip_tails by capping the high percentile."""
    thresh0, _ = cv2.threshold(vals.reshape(-1, 1).astype(np.uint8), 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    noise = vals[vals <= thresh0]
    if noise.size == 0:
        return 0.0
    median = np.median(noise)
    mad = np.median(np.abs(noise.astype(np.float64) - median))
    robust_std = mad * 1.4826
    return float(median + k * robust_std)


def clip_tails(values, real, hi_pct=95, noise_k=3.0):
    """Floor uniform low-level noise to 0 (see estimate_noise_floor) and cap
    the top hi_pct at its value (localized bright outliers/hotspots), both
    computed from the `real` (non-artifact) pixels only. Non-real pixels are
    always 0."""
    vals = values[real]
    hi = np.percentile(vals, hi_pct)
    floor = estimate_noise_floor(vals, k=noise_k)
    out = values.copy()
    if hi > floor:
        hi_u8 = int(round(hi))
        out[out <= floor] = 0
        out[out > hi_u8] = hi_u8
    out[~real] = 0
    return out


def tint(base_bgr, mask, color, alpha=ALPHA):
    out = base_bgr.copy()
    if not mask.any():
        return out
    patch = out[mask].astype(np.float32)
    out[mask] = np.clip(patch * (1 - alpha) + np.array(color, np.float32) * alpha,
                        0, 255).astype(np.uint8)
    return out


def label_panel(bgr, text):
    bgr = bgr.copy()
    cv2.rectangle(bgr, (0, 0), (bgr.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(bgr, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 1, cv2.LINE_AA)
    return bgr


def process_case(stem, rel_path, outdir, max_dim, tmpdir, raw=False):
    src = RGB_ROOT / rel_path
    if not src.exists():
        return f'{stem}: NOT FOUND ({src}) -- skipped'

    case_tmp = tmpdir / stem
    case_tmp.mkdir(parents=True, exist_ok=True)
    bgr, native_hw = load_rgb_downsampled(src, max_dim, case_tmp)
    for f in case_tmp.iterdir():
        f.unlink()
    case_tmp.rmdir()

    tissue = otsu_tissue(bgr, raw=raw)

    p1 = label_panel(bgr, f'{stem}  |  native {native_hw[1]}x{native_hw[0]}  '
                          f'-> preview {bgr.shape[1]}x{bgr.shape[0]}')
    label = 'raw otsu tissue (red channel)' if raw else 'otsu tissue (red channel)'
    p2 = label_panel(tint(bgr, tissue, TISSUE_COLOR),
                     f'{label}  {100 * tissue.mean():.1f}% of frame')

    out_path = outdir / f'{stem}.png'
    cv2.imwrite(str(out_path), np.hstack([p1, p2]))
    return f'{stem}: native {native_hw[1]}x{native_hw[0]}, tissue {100 * tissue.mean():.1f}% of preview frame'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', type=Path, default=DEFAULT_OUTDIR)
    ap.add_argument('--max-dim', type=int, default=DEFAULT_MAX_DIM)
    ap.add_argument('--raw', action='store_true',
                    help='bare Otsu on real pixels, skip clip_tails preprocessing')
    args = ap.parse_args()

    if args.outdir.exists():
        for f in args.outdir.glob('*.png'):
            f.unlink()
    args.outdir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix='epicardium_vips_') as tmp:
        tmpdir = Path(tmp)
        for stem, rel_path in CASES.items():
            print(process_case(stem, rel_path, args.outdir, args.max_dim, tmpdir, raw=args.raw),
                  flush=True)
    print(f'\nwrote previews to {args.outdir}')


if __name__ == '__main__':
    main()
