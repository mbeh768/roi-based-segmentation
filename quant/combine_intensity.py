"""
Combine the two per-half intensity quantifications (1of2 + 2of2) of a pair into a
single whole-heart-equivalent result: band_means.csv, radial_profile.csv, and the
four chart PNGs, as if quantify_intensity.py had been run once on the whole image
instead of on each half separately.

Band means are pixel-count-weighted averages (not a plain mean of the two halves'
means, since the halves can have very different pixel counts). The radial profile
is combined by matching bins on physical distance (both halves were binned at the
same um_per_px, so bin edges line up exactly) and weighted-averaging any bin present
in both halves.

Usage:
    python quant/combine_intensity.py <intensity_dir_1of2> <intensity_dir_2of2> <outdir> [--stem STEM]
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from quantify_intensity import (
    BAND_LABELS, save_bar_intensity, save_bar_ratio,
    save_radial_intensity, save_radial_ratio,
)

UM_PER_PX = 2.5562774308232


def read_bit_label(intensity_dir):
    p = Path(intensity_dir) / 'bit_depth.txt'
    return p.read_text().strip() if p.exists() else 'unknown bit depth'


def read_band_means(path):
    rows = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            rows[row['band']] = row
    return rows


def combine_band_means(rows1, rows2):
    out = []
    for band in BAND_LABELS:
        r1, r2 = rows1[band], rows2[band]
        n1, n2 = int(r1['n_px']), int(r2['n_px'])
        n = n1 + n2
        r_mean = (float(r1['R_mean']) * n1 + float(r2['R_mean']) * n2) / n
        g_mean = (float(r1['G_mean']) * n1 + float(r2['G_mean']) * n2) / n
        b_mean = (float(r1['B_mean']) * n1 + float(r2['B_mean']) * n2) / n
        g_over_r = g_mean / r_mean if r_mean else float('nan')
        out.append((band, r_mean, g_mean, b_mean, g_over_r, n))
    return out


def read_radial_profile(path, um_per_px):
    bins = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            b = round(float(row['distance_um']) / um_per_px)
            bins[b] = row
    return bins


def combine_radial_profile(bins1, bins2, um_per_px):
    all_bins = sorted(set(bins1) | set(bins2))
    dist_um, r_mean, g_mean, b_mean, g_over_r, n_px = [], [], [], [], [], []
    for b in all_bins:
        row1, row2 = bins1.get(b), bins2.get(b)
        if row1 is not None and row2 is not None:
            n1, n2 = int(row1['n_px']), int(row2['n_px'])
            n = n1 + n2
            r = (float(row1['R_mean']) * n1 + float(row2['R_mean']) * n2) / n
            g = (float(row1['G_mean']) * n1 + float(row2['G_mean']) * n2) / n
            bl = (float(row1['B_mean']) * n1 + float(row2['B_mean']) * n2) / n
        else:
            row = row1 if row1 is not None else row2
            n = int(row['n_px'])
            r, g, bl = float(row['R_mean']), float(row['G_mean']), float(row['B_mean'])
        dist_um.append(b * um_per_px)
        r_mean.append(r)
        g_mean.append(g)
        b_mean.append(bl)
        g_over_r.append(g / r if r else float('nan'))
        n_px.append(n)
    return dist_um, r_mean, g_mean, b_mean, g_over_r, n_px


def combine(dir1, dir2, outdir, stem, um_per_px=UM_PER_PX):
    dir1, dir2, outdir = Path(dir1), Path(dir2), Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    label1, label2 = read_bit_label(dir1), read_bit_label(dir2)
    bit_label = f'1of2: {label1} | 2of2: {label2}' if label1 != label2 else label1
    (outdir / 'bit_depth.txt').write_text(bit_label + '\n')

    rows1 = read_band_means(dir1 / 'band_means.csv')
    rows2 = read_band_means(dir2 / 'band_means.csv')
    rows = combine_band_means(rows1, rows2)
    with open(outdir / 'band_means.csv', 'w') as f:
        f.write('band,R_mean,G_mean,B_mean,G_over_R,n_px\n')
        for r in rows:
            f.write(f"{r[0]},{r[1]:.4f},{r[2]:.4f},{r[3]:.4f},{r[4]:.4f},{r[5]}\n")
    print(f"saved -> {outdir / 'band_means.csv'}")
    save_bar_intensity(rows, stem, outdir / 'bar_intensity.png', bit_label)
    save_bar_ratio(rows, stem, outdir / 'bar_ratio.png', bit_label)

    bins1 = read_radial_profile(dir1 / 'radial_profile.csv', um_per_px)
    bins2 = read_radial_profile(dir2 / 'radial_profile.csv', um_per_px)
    dist_um, r_mean, g_mean, b_mean, g_over_r, n_px = combine_radial_profile(bins1, bins2, um_per_px)
    with open(outdir / 'radial_profile.csv', 'w') as f:
        f.write('distance_um,R_mean,G_mean,B_mean,G_over_R,n_px\n')
        for i in range(len(dist_um)):
            f.write(f"{dist_um[i]:.3f},{r_mean[i]:.4f},{g_mean[i]:.4f},{b_mean[i]:.4f},"
                    f"{g_over_r[i]:.4f},{n_px[i]}\n")
    print(f"saved -> {outdir / 'radial_profile.csv'}")
    save_radial_intensity(dist_um, r_mean, g_mean, b_mean, stem, outdir / 'radial_intensity.png', bit_label)
    save_radial_ratio(dist_um, g_over_r, stem, outdir / 'radial_ratio.png', bit_label)

    print(f"\ndone -> {outdir}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('dir1', help='intensity dir for half 1of2')
    parser.add_argument('dir2', help='intensity dir for half 2of2')
    parser.add_argument('outdir')
    parser.add_argument('--stem', default='combined')
    parser.add_argument('--microns-per-pixel', type=float, default=UM_PER_PX)
    args = parser.parse_args()
    combine(args.dir1, args.dir2, args.outdir, args.stem, args.microns_per_pixel)
