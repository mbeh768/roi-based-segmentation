"""
One-off: regenerate the 4 chart PNGs for an already-correct 8-bit-native slide
from its existing band_means.csv / radial_profile.csv, purely to add the
"[8bit]" title label introduced alongside the uint16-rescale fix. Does not
touch band_means.csv / radial_profile.csv (data is unchanged).

Usage:
    python quant/relabel_charts.py <intensity_dir> <stem>
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from quantify_intensity import (
    BAND_LABELS, save_bar_intensity, save_bar_ratio,
    save_radial_intensity, save_radial_ratio,
)


def main(intensity_dir, stem, bit_label='8bit'):
    d = Path(intensity_dir)
    (d / 'bit_depth.txt').write_text(bit_label + '\n')

    rows = []
    with open(d / 'band_means.csv') as f:
        for row in csv.DictReader(f):
            rows.append((row['band'], float(row['R_mean']), float(row['G_mean']),
                         float(row['B_mean']), float(row['G_over_R']), int(row['n_px'])))
    save_bar_intensity(rows, stem, d / 'bar_intensity.png', bit_label)
    save_bar_ratio(rows, stem, d / 'bar_ratio.png', bit_label)

    dist_um, r_mean, g_mean, b_mean, g_over_r = [], [], [], [], []
    with open(d / 'radial_profile.csv') as f:
        for row in csv.DictReader(f):
            dist_um.append(float(row['distance_um']))
            r_mean.append(float(row['R_mean']))
            g_mean.append(float(row['G_mean']))
            b_mean.append(float(row['B_mean']))
            g_over_r.append(float(row['G_over_R']))
    save_radial_intensity(dist_um, r_mean, g_mean, b_mean, stem, d / 'radial_intensity.png', bit_label)
    save_radial_ratio(dist_um, g_over_r, stem, d / 'radial_ratio.png', bit_label)
    print(f"relabeled -> {d}/")


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
