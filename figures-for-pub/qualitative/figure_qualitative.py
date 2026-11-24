"""Qualitative segmentation panels, one row per slide.

Disagreement is split by direction, which the single red of the previous version
could not show:

    black       background      covered by a tile, neither pred nor gt
    white       lumen           pred AND gt
    red         false positive  predicted lumen where the gt is background
    light blue  false negative  gt lumen the prediction missed
    grey        not scored      no tile reached it, excluded from the IoU

Grey is separate from black because coverage runs 5-38% and painting unscored
slide as correct background claims credit for area the model never saw.

Two figures from the same builder:

    --figure main   the four slides with canvases, columns image+ROI, prediction
                    with the ROI provided, prediction with the ROI replaced by
                    noise. Row label carries both IoUs.
    --figure supp   every slide with a canvas, proposed network only, columns
                    image+ROI and prediction, ordered by IoU.

Masks are resampled nearest so a class boundary never blends red into blue and
invents a colour that is in no legend; the photograph is resampled by area.

    python figures-for-pub/qualitative/figure_qualitative.py --figure main
    python figures-for-pub/qualitative/figure_qualitative.py --figure supp
"""
import argparse
import json
import shutil
import tempfile
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.patheffects import withStroke

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / 'train.py').exists())
VOXEL = ROOT / 'figures-for-pub' / 'quantification' / 'microns_per_pixel.json'
METRICS = (ROOT / 'figures-for-pub' / 'model-baseline'
           / 'k-fold-bnaug-2026-10-17_slide_metrics.json')

# indexed by code_map below. Red is the published false-positive colour; the
# blue is the infill the roi-value and density figures use for their markers.
LUT = np.array([(52, 52, 49), (0, 0, 0), (255, 255, 255),
                (230, 103, 103), (43, 140, 190)], np.uint8)
# legend order as read, not code order
KEYS = [(1, 'Background'), (2, 'Lumen'), (3, 'False positive'),
        (4, 'False negative'), (0, 'Not scored')]
ROI_RGB = (255, 214, 0)          # the ROI outline, in neither mask colour
# supplement number per arm, and where each arm's native masks live
SUPP = {'predicted': 'S1', 'sam': 'S2', 'microsam': 'S3', 'roi-fg': 'S4'}
# roi-fg is NOT stub-roi: that is the plain upsampled ROI, a sanity gate for the
# SAM export. The real arm is materialised by quantification/export_roi_fg.py.
MASK_DIR = {'sam': 'sam', 'microsam': 'microsam', 'roi-fg': 'roi-fg'}
ARM_LABEL = {'predicted': 'Predicted', 'sam': 'SAM', 'microsam': 'μSAM',
             'roi-fg': 'ROI−FG'}
# SAM and microSAM share one prompt; stub-roi is mask-only, so S4 shows none
PROMPT_ARMS = ('sam', 'microsam')
PROMPT_RGB = (255, 0, 255)       # distinct from the ROI yellow and both mask hues
CANVAS_SIDE = 8192               # the padded square the ROI and prompts index into
SURFACE, INK, INK_2 = '#fcfcfb', '#0b0b0b', '#52514e'
SURFACE_RGB = (252, 252, 251)    # the 2 px gap between panels, as pixels
WIDTH = 7.4                      # inches
NICE_UM = (500, 1000, 2000, 5000, 10000, 20000)
TITLE_PT, LEGEND_PT, BAR_PT = 7.5, 8.0, 6.5


def species(stem):
    return 'Human' if stem.startswith('AJK') else 'Pig'


def um_per_panel_px(stem, panel_w):
    """One panel pixel in microns, the unit scale_bar draws in: every panel is
    fit() to panel_w wide, so it is panel_w-th of the native width."""
    um = (json.loads(VOXEL.read_text()).get(stem) or {}).get('um_per_px')
    hw = {s: v['native_hw'] for f in json.loads(METRICS.read_text())['folds']
          for s, v in f['per_slide'].items()}.get(stem)
    if not um or not hw:
        raise SystemExit(f'{stem}: need um_per_px and native_hw to size a bar')
    return float(um) * hw[1] / panel_w


def code_map(pred, gt, covered=None):
    """0 unscored, 1 background, 2 lumen, 3 false positive, 4 false negative.

    covered is None for the whole-image arms: SAM, microSAM and ROI-FG have no
    unscored region, so nothing of theirs is grey."""
    code = (np.ones(pred.shape, np.uint8) if covered is None
            else covered.astype(np.uint8))
    code[pred & gt] = 2
    code[pred & ~gt] = 3
    code[~pred & gt] = 4
    return code


def published_iou(a, arm, stems, roi_run):
    """Native-resolution IoU as already published, never recomputed here: the
    canvas score differs in the third decimal and would drift from Table 1."""
    if arm == 'predicted':
        return {s: float(np.load(roi_run[s])['iou']) for s in stems}
    # roi-fg has no published score, so export_roi_fg.py computes it into its own file
    src = a.roi_fg_scores if arm == 'roi-fg' else a.scores
    rows = json.loads(Path(src).read_text())[MASK_DIR[arm]]['slides']
    got = {r['stem']: float(r['iou']) for r in rows}
    missing = [s for s in stems if s not in got]
    if missing:
        raise SystemExit(f'{arm}: no score for ' + ', '.join(missing))
    return got


def arm_pred(a, arm, stem, shape):
    """(pred, covered) on the canvas grid. The baselines are stored at native
    resolution, so they are area-downsampled on the same majority rule as the GT."""
    if arm == 'predicted':
        d = np.load(Path(a.canvases) / f'{stem}.npz')
        return d['pred'], d['covered']
    z = np.load(Path(a.masks) / MASK_DIR[arm] / f'{stem}.npz')
    pred = cv2.resize(z['pred'].astype(np.uint8) * 255, (shape[1], shape[0]),
                      interpolation=cv2.INTER_AREA) > 127
    return pred, None


def fit(a, width, nearest):
    h = max(1, round(a.shape[0] * width / a.shape[1]))
    return cv2.resize(a, (width, h), interpolation=cv2.INTER_NEAREST
                      if nearest else cv2.INTER_AREA)


def with_roi(image, roi, width, points=None):
    """The photograph at panel width, with the ROI it was given outlined and,
    for the SAM arms, the point prompts that went in alongside it."""
    rgb = fit(image, width, nearest=False)
    small = fit(roi.astype(np.uint8), width, nearest=True)
    contours, _ = cv2.findContours(small, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(rgb, contours, -1, ROI_RGB, 5 * max(2, width // 900))
    if points is not None and len(points):
        r = max(4, width // 220)
        for x, y in (np.asarray(points, float) * width / roi.shape[1]).round().astype(int):
            cv2.circle(rgb, (int(x), int(y)), r, PROMPT_RGB, -1)
            cv2.circle(rgb, (int(x), int(y)), r, (0, 0, 0), max(1, r // 4))
    return rgb


def load_points(a, arm):
    """stem -> point prompts in canvas pixels, or {} for the arms that use none.

    The dump is in SAM's own frame, so the factor comes from its metadata rather
    than a literal 8; stub-roi records n_points 0, so S4 draws nothing."""
    if arm not in PROMPT_ARMS:
        return {}
    d = json.loads(Path(a.points).read_text())
    k = CANVAS_SIDE / d['sam_side']
    return {s: np.asarray(v['points'], float) * k for s, v in d['slides'].items()}


def row_image(panels, gap=2):
    h = max(p.shape[0] for p in panels)
    out = []
    for i, p in enumerate(panels):
        if i:
            out.append(np.full((h, gap, 3), SURFACE_RGB, np.uint8))
        out.append(p)
    return np.hstack(out)


def scale_bar(ax, um_per_px, panel_w, img_h):
    want = 0.25 * panel_w * um_per_px
    um = min(NICE_UM, key=lambda v: abs(v - want))
    length = um / um_per_px
    x0 = 0.04 * panel_w
    y0 = img_h - 0.06 * img_h
    ax.add_patch(Rectangle((x0, y0), length, max(2.0, 0.012 * img_h),
                           facecolor='white', edgecolor='black', linewidth=0.4,
                           zorder=6))
    ax.text(x0 + length / 2, y0 - 0.012 * img_h,
            f'{um / 1000:g} mm' if um >= 1000 else f'{um:g} µm', color='white',
            fontsize=BAR_PT, ha='center', va='bottom', zorder=6,
            path_effects=[withStroke(linewidth=1.6, foreground='black')])
    return um


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--figure', choices=['main', 'supp'], default='main')
    ap.add_argument('--canvases', default=str(HERE / 'canvases'))
    ap.add_argument('--noise-canvases',
                    default=str(ROOT / 'figures-for-pub' / 'roi-value' / 'canvases'))
    ap.add_argument('--out', default=None)
    ap.add_argument('--dpi', type=int, default=600)
    ap.add_argument('--panel-px', type=int, default=2048)
    ap.add_argument('--format', choices=['png', 'pdf'], default='png',
                    help='supp only: pdf gives one <method>-supp.pdf with an index')
    ap.add_argument('--per-page', type=int, default=3,
                    help='pdf only: slide/agreement pairs per page')
    ap.add_argument('--chunks', type=int, default=4,
                    help='supp only: split across this many files, S1a, S1b, ...')
    ap.add_argument('--arm', choices=sorted(SUPP), default='predicted',
                    help='supp only: which method the prediction column shows')
    ap.add_argument('--gt-native', default=str(HERE / 'gt-native'))
    ap.add_argument('--masks',
                    default=str(ROOT / 'figures-for-pub' / 'quantification' / 'masks'))
    ap.add_argument('--scores', default=str(ROOT / 'figures-for-pub' / 'quantification'
                                            / 'sam_scores.json'))
    ap.add_argument('--points', default=str(ROOT / 'figures-for-pub' / 'quantification'
                                           / 'prompt_points.json'))
    ap.add_argument('--roi-fg-scores',
                    default=str(ROOT / 'figures-for-pub' / 'quantification'
                                / 'roi_fg_scores.json'))
    a = ap.parse_args()

    roi_run = {p.stem: p for p in sorted(Path(a.canvases).glob('*.npz'))}
    if not roi_run:
        raise SystemExit(f'no canvases in {a.canvases}')
    noise_run = {p.stem: p for p in sorted(Path(a.noise_canvases).glob('*.npz'))}

    arm = 'predicted' if a.figure == 'main' else a.arm
    score = published_iou(a, arm, list(roi_run), roi_run)
    # each supplement is ordered by its own arm's IoU, so S2-S4 are not in S1's order
    stems = sorted(roi_run, key=lambda s: -score[s])
    if a.figure == 'main':
        # best two and worst two, out of the slides that also have a noise
        # canvas -- supp exports every slide, so this cannot be "all of them"
        paired = [s for s in stems if s in noise_run]
        if len(paired) < 4:
            raise SystemExit(f'need four slides with both canvases, have '
                             f'{len(paired)}: ' + ', '.join(paired))
        stems = paired[:2] + paired[-2:]
        heads = ['Section with ROI input', 'Predicted, ROI provided',
                 'Predicted, ROI replaced by noise']
    else:
        heads = ['Section with ROI input', ARM_LABEL[a.arm]]

    points = load_points(a, arm)
    rows, used = [], set()
    for stem in stems:
        d = np.load(roi_run[stem])
        # GT from the source mask, not the canvas `gt`, which is covered-limited
        gt = np.load(Path(a.gt_native) / f'{stem}.npz')['gt']
        pred, covered = arm_pred(a, arm, stem, gt.shape)
        code = code_map(pred, gt, covered)
        used.update(np.unique(code).tolist())
        panels = [with_roi(d['image'], d['roi'], a.panel_px, points.get(stem)),
                  fit(LUT[code], a.panel_px, nearest=True)]
        label = f'{stem} · {species(stem)} · IoU {score[stem]:.3f}'
        if a.figure == 'main':
            n = np.load(noise_run[stem])
            ncode = code_map(n['pred'], gt, n['covered'])
            used.update(np.unique(ncode).tolist())
            panels.append(fit(LUT[ncode], a.panel_px, nearest=True))
            label = (f'{stem} · {species(stem)} · IoU {score[stem]:.3f} '
                     f"with ROI, {float(n['iou']):.3f} with noise")
        rows.append((stem, label, row_image(panels),
                     um_per_panel_px(stem, a.panel_px)))

    if a.figure == 'supp' and a.format == 'pdf':
        build_pdf(rows, heads, used, a)
        return

    # contiguous blocks, so the IoU ordering still reads straight through S1a..S1d
    n, c = len(rows), max(1, a.chunks)
    parts = ([rows] if a.figure == 'main'
             else [rows[k * n // c:(k + 1) * n // c] for k in range(c)])

    for k, part in enumerate(parts):
        if not part:
            continue
        suffix = '' if a.figure == 'main' else 'abcdefgh'[k]
        draw_part(part, heads, used, a, suffix)


INDEX_TOP, INDEX_DY, INDEX_PT = 0.90, 0.034, 10.0
# US Letter portrait, in inches, with the margins the pages are laid out inside
PAGE_W, PAGE_H = 8.5, 11.0
M_L, M_R, M_T, M_B = 0.45, 0.45, 0.45, 0.62
TITLE_H, HEAD_H, ROW_GAP = 0.26, 0.28, 0.14


def legend(fig, used, a, anchor):
    """A colour that never appears would only invite the reader to hunt for it, so
    the keys are the classes this figure actually contains."""
    handles = [Rectangle((0, 0), 1, 1, facecolor=LUT[i] / 255,
                         edgecolor='#c3c2b7', lw=0.4) for i, _ in KEYS if i in used]
    names = [n for i, n in KEYS if i in used]
    handles.append(Line2D([], [], color=np.array(ROI_RGB) / 255, lw=1.6))
    names.append('ROI input')
    if a.figure != 'main' and a.arm in PROMPT_ARMS:
        handles.append(Line2D([], [], linestyle='none', marker='o', markersize=6,
                              markerfacecolor=np.array(PROMPT_RGB) / 255,
                              markeredgecolor='black', markeredgewidth=0.6))
        names.append('SAM point prompt')
    fig.legend(handles, names, loc='lower center', ncol=len(names), frameon=False,
               fontsize=LEGEND_PT, labelcolor=INK_2, bbox_to_anchor=anchor)


def pack(rows):
    """Greedy fill of fixed pages: as many pairs as fit, so a tall portrait
    section can share a page with nothing and a flat one with two others.

    Yields [(row, image_w, image_h), ...] per page, in inches.
    """
    usable_w = PAGE_W - M_L - M_R
    usable_h = PAGE_H - M_T - M_B
    page, left = [], usable_h - HEAD_H       # the first row carries the headers
    for r in rows:
        aspect = r[2].shape[0] / r[2].shape[1]
        w = usable_w
        h = w * aspect
        if TITLE_H + h > usable_h - HEAD_H:  # too tall even alone: scale to fit
            h = usable_h - HEAD_H - TITLE_H
            w = h / aspect
        need = TITLE_H + h + ROW_GAP
        if page and need > left:
            yield page
            page, left = [], usable_h - HEAD_H
        page.append((r, w, h))
        left -= need
    if page:
        yield page


def draw_pdf_page(part, heads, used, a, pdf):
    """One fixed-size page. Axes are placed by hand rather than by subplots: the
    rows differ in height, so a gridspec would stretch them to a common cell."""
    fig = plt.figure(figsize=(PAGE_W, PAGE_H))
    fig.patch.set_facecolor(SURFACE)
    # centred in the margins rather than hung from the top: pages hold 1 to 3
    # pairs, and a single tall one would otherwise sit above a page of whitespace
    usable_h = PAGE_H - M_T - M_B
    block = (HEAD_H + sum(TITLE_H + h for _r, _w, h in part)
             + ROW_GAP * (len(part) - 1))
    y = M_B + (usable_h + block) / 2 - HEAD_H

    for i, ((stem, label, img, um_px), w, h) in enumerate(part):
        x = M_L + (PAGE_W - M_L - M_R - w) / 2
        ax = fig.add_axes([x / PAGE_W, (y - TITLE_H - h) / PAGE_H,
                           w / PAGE_W, h / PAGE_H])
        ax.imshow(img)
        ax.set_title(label, fontsize=TITLE_PT, color=INK, loc='left', pad=3)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        bar = scale_bar(ax, um_px, a.panel_px, img.shape[0])
        if i == 0:
            for j, head in enumerate(heads):
                cx = (j * (a.panel_px + 2) + a.panel_px / 2) / img.shape[1]
                ax.annotate(head, xy=(cx, 1.0), xycoords='axes fraction',
                            xytext=(0, 18), textcoords='offset points',
                            fontsize=TITLE_PT, color=INK_2, ha='center',
                            va='bottom', annotation_clip=False)
        y -= TITLE_H + h + ROW_GAP
        print(f'  {stem:22} {species(stem):6} {um_px:6.2f} um/px  bar {bar} um')

    legend(fig, used, a, anchor=(0.5, M_B / PAGE_H * 0.35))
    pdf.savefig(fig, facecolor=SURFACE, dpi=a.dpi)
    plt.close(fig)


def draw_index(rows, where, a, path):
    """Page 1: one line per slide. Returns each line's rectangle in PDF points,
    so the GoTo links are placed from the layout rather than by searching for the
    stem -- '21PControl-1' is a substring of '21PControl-11-1of2'."""
    fig = plt.figure(figsize=(PAGE_W, PAGE_H))   # letter, like the content pages
    fig.patch.set_facecolor(SURFACE)
    fig.text(0.06, 0.955, f'Supplementary {SUPP[a.arm]}: {ARM_LABEL[a.arm]}',
             fontsize=INDEX_PT + 3, color=INK, fontweight='bold', va='bottom')
    fig.text(0.06, 0.935, f'{len(rows)} held-out slides, ordered by IoU.',
             fontsize=INDEX_PT - 1, color=INK_2, va='bottom')

    rects, W, H = {}, PAGE_W * 72.0, PAGE_H * 72.0
    for i, (stem, label, _img, _um) in enumerate(rows):
        y = INDEX_TOP - i * INDEX_DY
        fig.text(0.06, y, f'{i + 1:>2}.  {label}', fontsize=INDEX_PT, color=INK,
                 va='center')
        fig.text(0.94, y, f'page {where[stem]}', fontsize=INDEX_PT, color=INK_2,
                 va='center', ha='right')
        half = INDEX_DY * 0.42
        rects[stem] = (0.055 * W, (1 - (y + half)) * H,
                       0.945 * W, (1 - (y - half)) * H)
    fig.savefig(path, facecolor=SURFACE)      # no bbox_inches: it would move the text
    plt.close(fig)
    return rects


def build_pdf(rows, heads, used, a):
    """Content pages of PER_PAGE pairs, an index in front, GoTo links and marks."""
    import pymupdf
    from matplotlib.backends.backend_pdf import PdfPages

    pages = list(pack(rows))
    where = {r[0][0]: n + 2 for n, part in enumerate(pages) for r in part}

    tmp = Path(tempfile.mkdtemp(prefix='supp-'))
    body, index = tmp / 'body.pdf', tmp / 'index.pdf'
    with PdfPages(body) as pdf:
        for part in pages:
            draw_pdf_page(part, heads, used, a, pdf)
    rects = draw_index(rows, where, a, index)

    doc = pymupdf.open(index)
    doc.insert_file(pymupdf.open(body))
    for stem, rect in rects.items():
        page = where[stem] - 1
        doc[0].insert_link({'kind': pymupdf.LINK_GOTO, 'from': pymupdf.Rect(*rect),
                            'page': page, 'to': pymupdf.Point(0, 0)})
    doc.set_toc([[1, f'{i + 1}. {s}', where[s]] for i, (s, *_r) in enumerate(rows)])

    out = Path(a.out) if a.out else HERE / f'{a.arm}-supp.pdf'
    doc.save(out, garbage=4, deflate=True)
    doc.close()
    shutil.rmtree(tmp, ignore_errors=True)
    print(f'wrote {out.name}   {len(pages) + 1} pages   '
          f'{out.stat().st_size / 1e6:.2f} MB   {PAGE_W:g}x{PAGE_H:g} in, {a.dpi} dpi   '
          f'pairs per page: {[len(p) for p in pages]}')


def draw_part(rows, heads, used, a, suffix, pdf=None):
    # each row takes its OWN aspect: sized from one shared maximum, a landscape
    # row sat in a cell built for a portrait one and the leftover margin showed
    # up as blank bands, the last of which trailed off the bottom
    aspects = [r.shape[0] / r.shape[1] for *_, r, _ in rows]
    fig, axes = plt.subplots(len(rows), 1, squeeze=False,
                             figsize=(WIDTH, WIDTH * sum(aspects) * 1.06),
                             gridspec_kw={'height_ratios': aspects,
                                          'hspace': 0.10})
    fig.patch.set_facecolor(SURFACE)
    axes = axes.ravel()

    for i, (ax, (stem, label, img, um_px)) in enumerate(zip(axes, rows)):
        ax.imshow(img)
        ax.set_title(label, fontsize=TITLE_PT, color=INK, loc='left', pad=3)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        bar = scale_bar(ax, um_px, a.panel_px, img.shape[0])
        if i == 0:
            # offset in points, above the row title: an axes-fraction offset
            # would scale with the row height and collide on a short row
            for j, head in enumerate(heads):
                cx = (j * (a.panel_px + 2) + a.panel_px / 2) / img.shape[1]
                ax.annotate(head, xy=(cx, 1.0), xycoords='axes fraction',
                            xytext=(0, 18), textcoords='offset points',
                            fontsize=TITLE_PT, color=INK_2, ha='center',
                            va='bottom', annotation_clip=False)
        print(f'  {stem:22} {species(stem):6} {um_px:6.2f} um/px  bar {bar} um')

    legend(fig, used, a, anchor=(0.5, 0.0))

    base = Path(a.out) if a.out else HERE / (
        'qualitative.png' if a.figure == 'main'
        else f'supplementary_{SUPP[a.arm]}.png')
    out = base if not suffix else base.with_name(f'{base.stem}{suffix}{base.suffix}')
    fig.savefig(out, dpi=a.dpi, facecolor=SURFACE, bbox_inches='tight')
    plt.close(fig)
    px = plt.imread(out).shape
    print(f'wrote {out.name}   {px[1]} x {px[0]} px   '
          f'{out.stat().st_size / 1e6:.2f} MB   ({len(rows)} rows)')


if __name__ == '__main__':
    main()
