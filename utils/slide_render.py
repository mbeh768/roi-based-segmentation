"""Reading a slide TIFF for a panel a human looks at. Render only -- the gain
below makes a section visible on a page and would corrupt any measurement.

One global gain for all three channels, never per-channel: per-channel would
normalise away a channel imbalance, which on some slides is the defect the
figure exists to show. The gain comes from the full-resolution mean even when a
downsampled image is returned, so tone does not vary with output size. Native
aspect ratio, never padded or cropped.
"""
from pathlib import Path

import cv2
import numpy as np
import tifffile

TARGET_MEAN = 127.0


def find_slide(raw_dir, stem, kind='image'):
    """The source TIFF for `stem`. Folder names do not match the preprocessed
    stems, but file names do, so the glob keys on those."""
    hits = sorted(Path(raw_dir).glob(f'*/{stem}_{kind}.*'))
    if not hits:
        raise SystemExit(f'no raw {kind} for {stem} under {raw_dir}')
    if len(hits) > 1:
        print(f'    {stem}: {len(hits)} candidates, using {hits[0].name}')
    return hits[0]


def load_slide_rgb(path, target=None, quiet=False):
    """uint8 (H, W, 3) brightened to a mean of 127. `target` is the long side in
    pixels; None keeps the native grid.

    These files carry no pyramid and reach 5.5 GB, so CYX sources are read a
    page at a time and each plane is shrunk before the next is touched.
    """
    path = Path(path)

    def dsize(h, w):
        scale = target / max(h, w)
        return max(1, round(w * scale)), max(1, round(h * scale))

    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        axes, dtype = series.axes, series.dtype

        # CYX does not imply one page per channel: some of these files hold all
        # three in a single page, where pages[1] does not exist
        if axes.startswith('C') and len(tif.pages) >= series.shape[0]:
            H, W = series.shape[1], series.shape[2]
            total, planes = 0.0, []
            for c in range(series.shape[0]):
                plane = tif.pages[c].asarray()
                total += float(plane.mean(dtype=np.float64))
                planes.append(cv2.resize(plane, dsize(H, W),
                                         interpolation=cv2.INTER_AREA)
                              if target else plane)
                del plane
            mean = total / len(planes)
        else:
            arr = series.asarray()
            if axes.startswith('C'):
                arr = np.moveaxis(arr, 0, -1)
            H, W = arr.shape[:2]
            mean = float(arr.mean(dtype=np.float64))
            if target:
                arr = cv2.resize(arr, dsize(H, W), interpolation=cv2.INTER_AREA)
            planes = [np.ascontiguousarray(arr[..., c]) for c in range(arr.shape[-1])]
            del arr

    gain = TARGET_MEAN / max(mean, 1e-6)

    h, w = planes[0].shape
    rgb = np.empty((h, w, 3), np.uint8)
    for c in range(3):
        rgb[..., c] = np.clip(planes[c].astype(np.float32) * gain, 0, 255).astype(np.uint8)
        planes[c] = None

    if not quiet:
        print(f'    {path.name}: {W}x{H} {dtype} {axes} -> {w}x{h}  '
              f'mean {mean:.1f} x{gain:.5f} -> {rgb.mean(dtype=np.float64):.1f}')
    return rgb
