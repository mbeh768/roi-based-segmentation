"""
Generate a zoomed-in copy (-50um to +250um, the subendocardium/100um/200um band
range) of each existing radial_intensity.png / radial_ratio.png, from the same
radial_profile.csv -- no recomputation, just an x-axis-limited re-plot so the
banded region is readable without the long tail either side.

Usage:
    python quant/zoom_radial.py <intensity_dir> <stem> [bit_label]
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from quantify_intensity import save_radial_intensity, save_radial_ratio

ZOOM_LO, ZOOM_HI = -50.0, 250.0


def main(intensity_dir, stem, bit_label=None):
    d = Path(intensity_dir)
    if bit_label is None:
        p = d / 'bit_depth.txt'
        bit_label = p.read_text().strip() if p.exists() else None

    dist_um, r_mean, g_mean, b_mean, g_over_r = [], [], [], [], []
    with open(d / 'radial_profile.csv') as f:
        for row in csv.DictReader(f):
            x = float(row['distance_um'])
            if ZOOM_LO <= x <= ZOOM_HI:
                dist_um.append(x)
                r_mean.append(float(row['R_mean']))
                g_mean.append(float(row['G_mean']))
                b_mean.append(float(row['B_mean']))
                g_over_r.append(float(row['G_over_R']))

    zoom_stem = f'{stem} (zoom {ZOOM_LO:g} to {ZOOM_HI:g}um)'
    save_radial_intensity(dist_um, r_mean, g_mean, b_mean, zoom_stem,
                           d / 'radial_intensity_zoom.png', bit_label)
    save_radial_ratio(dist_um, g_over_r, zoom_stem,
                       d / 'radial_ratio_zoom.png', bit_label)


if __name__ == '__main__':
    bl = sys.argv[3] if len(sys.argv) > 3 else None
    main(sys.argv[1], sys.argv[2], bl)
