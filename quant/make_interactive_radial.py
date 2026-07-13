"""
Generate interactive, zoomable HTML versions of each radial_intensity.png /
radial_ratio.png (drag-to-zoom, pan, hover tooltips, range slider) from the
same radial_profile.csv, so the long tail and the -50..250um band region can
both be inspected without a fixed static crop.

All generated HTML files reference a single shared quant/plotly.min.js (not
re-embedded per file) via a relative <script src>.

Usage:
    python quant/make_interactive_radial.py <intensity_dir> <stem> [bit_label]
"""
import csv
import sys
from pathlib import Path

import plotly.graph_objects as go

REPO = Path(__file__).parent.parent
PLOTLY_JS = REPO / 'quant' / 'plotly.min.js'

BAND_COLORS = {'R': '#d62728', 'G': '#2ca02c', 'B': '#1f77b4'}
BAND_EDGES = (0, 100, 200)


def read_radial_profile(path):
    dist_um, r_mean, g_mean, b_mean, g_over_r = [], [], [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            dist_um.append(float(row['distance_um']))
            r_mean.append(float(row['R_mean']))
            g_mean.append(float(row['G_mean']))
            b_mean.append(float(row['B_mean']))
            g_over_r.append(float(row['G_over_R']))
    return dist_um, r_mean, g_mean, b_mean, g_over_r


def _band_shapes():
    return [
        dict(type='line', x0=e, x1=e, y0=0, y1=1, yref='paper',
             line=dict(color='gray', width=0.8, dash='dash'), opacity=0.6)
        for e in BAND_EDGES
    ]


def _plotlyjs_src(outdir):
    import os
    return os.path.relpath(PLOTLY_JS, outdir)


def save_radial_intensity_html(dist_um, r_mean, g_mean, b_mean, stem, path, bit_label=None):
    fig = go.Figure()
    for ch, vals, color in (('R', r_mean, BAND_COLORS['R']),
                             ('G', g_mean, BAND_COLORS['G']),
                             ('B', b_mean, BAND_COLORS['B'])):
        fig.add_trace(go.Scatter(x=dist_um, y=vals, mode='lines', name=ch,
                                  line=dict(color=color, width=1.5),
                                  hovertemplate=f'{ch}: %{{y:.2f}}<extra></extra>'))
    title = f'{stem}: radial intensity profile (1px resolution)'
    if bit_label:
        title += f'<br><sup>[{bit_label}]</sup>'
    fig.update_layout(
        title=title,
        xaxis_title='signed distance from mask boundary (um)  [negative = inside mask]',
        yaxis_title='mean pixel intensity (0-255)',
        hovermode='x unified',
        shapes=_band_shapes(),
        template='plotly_white',
        xaxis=dict(rangeslider=dict(visible=True)),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )
    path = Path(path)
    fig.write_html(str(path), include_plotlyjs=_plotlyjs_src(path.parent), full_html=True)
    print(f"saved -> {path}")


def save_radial_ratio_html(dist_um, g_over_r, stem, path, bit_label=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dist_um, y=g_over_r, mode='lines', name='G/R',
                              line=dict(color=BAND_COLORS['G'], width=1.5),
                              hovertemplate='G/R: %{y:.3f}<extra></extra>'))
    title = f'{stem}: radial green/red ratio profile (1px resolution)'
    if bit_label:
        title += f'<br><sup>[{bit_label}]</sup>'
    fig.update_layout(
        title=title,
        xaxis_title='signed distance from mask boundary (um)  [negative = inside mask]',
        yaxis_title='mean-G / mean-R',
        hovermode='x unified',
        shapes=_band_shapes(),
        template='plotly_white',
        xaxis=dict(rangeslider=dict(visible=True)),
        showlegend=False,
    )
    path = Path(path)
    fig.write_html(str(path), include_plotlyjs=_plotlyjs_src(path.parent), full_html=True)
    print(f"saved -> {path}")


def main(intensity_dir, stem, bit_label=None):
    d = Path(intensity_dir)
    if bit_label is None:
        p = d / 'bit_depth.txt'
        bit_label = p.read_text().strip() if p.exists() else None

    dist_um, r_mean, g_mean, b_mean, g_over_r = read_radial_profile(d / 'radial_profile.csv')
    save_radial_intensity_html(dist_um, r_mean, g_mean, b_mean, stem, d / 'radial_intensity.html', bit_label)
    save_radial_ratio_html(dist_um, g_over_r, stem, d / 'radial_ratio.html', bit_label)


if __name__ == '__main__':
    bl = sys.argv[3] if len(sys.argv) > 3 else None
    main(sys.argv[1], sys.argv[2], bl)
