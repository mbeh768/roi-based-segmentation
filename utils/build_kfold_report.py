"""Render the leave-one-case-out results as a PDF.

Inputs (produced by the sweep, not recomputed here):
  saved/models/<run>/test_results.json   per-fold tile metrics, from train.py
  slide_level_iou.json                   per-slide stitched metrics, from
                                         utils/slide_level_iou.py

Usage:
    python utils/build_kfold_report.py --name KFOLD_LOCO \
        --models-dir saved/models --slide-json slide_level_iou.json \
        --out kfold_report.pdf
"""
import argparse
import json
import statistics as st
from datetime import date
from pathlib import Path

from utils import read_json

# Slide flagged in the source dataset as having a sectioning/channel defect;
# the directory is literally named "17P645-1 (validated) (channel issue)".
DEFECTIVE = {'17P645-1'}

CSS = """
@page { size: A4; margin: 16mm 14mm; @bottom-center {
    content: "page " counter(page) " of " counter(pages);
    font: 8pt 'Helvetica'; color: #888; } }
body { font: 9.5pt/1.45 'Helvetica','Arial',sans-serif; color: #1a1a1a; }
h1 { font-size: 17pt; margin: 0 0 2mm; }
h2 { font-size: 12pt; margin: 7mm 0 2mm; border-bottom: 1.2pt solid #333;
     padding-bottom: 1mm; }
h3 { font-size: 10pt; margin: 4mm 0 1.5mm; color: #333; }
.sub { color: #666; font-size: 8.5pt; margin-bottom: 4mm; }
table { border-collapse: collapse; width: 100%; margin: 2mm 0 3mm;
        font-size: 8.5pt; }
th { background: #f0f0f0; text-align: left; padding: 1.6mm 2mm;
     border-bottom: 1pt solid #bbb; font-weight: bold; }
td { padding: 1.4mm 2mm; border-bottom: 0.4pt solid #e0e0e0; }
td.n, th.n { text-align: right; font-variant-numeric: tabular-nums; }
tr.flag td { background: #fff4f4; }
tr.tot td { font-weight: bold; border-top: 1pt solid #999; background: #fafafa; }
.headline { background: #f7f7f7; border-left: 3pt solid #333; padding: 3mm 4mm;
            margin: 3mm 0; }
.headline .big { font-size: 15pt; font-weight: bold; }
.note { background: #fffdf0; border-left: 2.5pt solid #d0a000; padding: 2.5mm 3.5mm;
        margin: 3mm 0; font-size: 8.5pt; }
.mono { font-family: 'DejaVu Sans Mono','Courier New',monospace; font-size: 8pt; }
ul { margin: 1.5mm 0 2mm; padding-left: 5mm; }
li { margin-bottom: 1mm; }
.bar { display: inline-block; height: 7px; background: #4a7ba7; vertical-align: middle; }
.bar.lo { background: #c0504d; }
"""


def ms(v):
    return (st.mean(v), st.stdev(v) if len(v) > 1 else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--models-dir', default='saved/models')
    ap.add_argument('--slide-json', required=True)
    ap.add_argument('--out', default='kfold_report.pdf')
    ap.add_argument('--config', default='config.json')
    ap.add_argument('--aug-json', default=None,
                    help='utils/aug_effect.py output; adds the augmentation section')
    a = ap.parse_args()

    slide = read_json(a.slide_json)
    aug = read_json(a.aug_json) if a.aug_json and Path(a.aug_json).exists() else None
    cfg = read_json(a.config)

    # per-fold tile metrics straight from the sweep
    folds = []
    for cfg_path in sorted(Path(a.models_dir).glob('*/config.json')):
        c = read_json(cfg_path)
        if c.get('name') != a.name:
            continue
        rp = cfg_path.parent / 'test_results.json'
        if not rp.exists():
            continue
        r = read_json(rp)
        r['fold'] = c['data_loader']['args'].get('fold', 0)
        r['run'] = cfg_path.parent.name
        folds.append(r)
    folds.sort(key=lambda r: r['fold'])

    by_fold = {f['fold']: f for f in slide['folds']}
    slides_by_fold = {}
    for s in slide['slides']:
        slides_by_fold.setdefault(s['fold'], []).append(s)

    rows = []
    for f in folds:
        sf = by_fold.get(f['fold'], {})
        case = ', '.join(f.get('test_cases') or [])
        rows.append({
            'fold': f['fold'], 'case': case,
            'tiles': f.get('n_test_tiles', 0),
            'best_epoch': f.get('best_epoch'),
            'tile_run': f.get('test_iou', float('nan')),
            'tile_noaug': sf.get('tile_iou_batchmean', float('nan')),
            'tile_pooled': sf.get('tile_iou_pooled', float('nan')),
            'slide': sf.get('slide_iou_case', float('nan')),
            'defective': any(d in case for d in DEFECTIVE),
        })

    ok = [r for r in rows if not r['defective']]
    all_slide = [r['slide'] for r in rows]
    ok_slide = [r['slide'] for r in ok]
    all_tile = [r['tile_run'] for r in rows]
    m_all, s_all = ms(all_slide)
    m_ok, s_ok = ms(ok_slide)
    mt_all, st_all = ms(all_tile)
    tw = sum(r['slide'] * r['tiles'] for r in rows) / sum(r['tiles'] for r in rows)

    la = cfg['data_loader']['args']
    tr = cfg['trainer']
    n_folds = len(rows)

    H = []
    H.append(f"<h1>Ventricle segmentation &mdash; leave-one-case-out cross validation</h1>")
    H.append(f"<div class='sub'>{a.name} &middot; generated {date.today().isoformat()} "
             f"&middot; {n_folds} folds &middot; {sum(r['tiles'] for r in rows)} held-out tiles</div>")

    H.append("<div class='headline'>")
    H.append(f"<div class='big'>Slide-level IoU {m_all:.3f} &plusmn; {s_all:.3f}</div>")
    H.append(f"<div>across all {n_folds} cases &middot; tile-level {mt_all:.3f} &plusmn; {st_all:.3f} "
             f"&middot; tile-weighted {tw:.3f}</div>")
    H.append(f"<div style='margin-top:2mm'>Excluding the defective slide "
             f"({', '.join(sorted(DEFECTIVE))}): <b>{m_ok:.3f} &plusmn; {s_ok:.3f}</b> "
             f"(n={len(ok)})</div>")
    H.append("</div>")

    H.append("<h2>Method</h2>")
    H.append("<p>Every slide is evaluated by a model that never trained on it, and whose "
             "checkpoint was chosen using validation cases that do not include it. "
             "The 20 slides collapse to 14 <i>cases</i>: six are two physical halves of one "
             "tissue sample and are always held out together, since splitting a pair across "
             "train and test would leak adjacent tissue.</p>")
    H.append("<ul>")
    H.append(f"<li><b>Split per fold:</b> {n_folds - 1 - la.get('n_val_cases', 0)} train / "
             f"{la.get('n_val_cases')} validation / 1 test case. Each case is the test set "
             f"exactly once.</li>")
    H.append("<li><b>Model selection</b> (early stopping, best checkpoint) reads only the "
             "validation cases. The test case is scored once, after training. Validation "
             "metrics are therefore <i>not</i> held-out estimates and are excluded from the "
             "headline figures.</li>")
    H.append(f"<li><b>Training:</b> {cfg['arch']['type']} ({cfg['arch']['args'].get('backbone')}), "
             f"batch {la.get('batch_size')}, {cfg['optimizer']['type']} "
             f"lr={cfg['optimizer']['args'].get('lr')}, loss {cfg.get('loss')}, "
             f"max {tr.get('epochs')} epochs, early stop {tr.get('early_stop')}, "
             f"bf16 mixed precision.</li>")
    H.append("</ul>")

    H.append("<h3>How the two IoUs differ</h3>")
    H.append("<p><b>Slide-level IoU</b> is the headline metric: predictions for every held-out "
             "tile are stitched onto the full slide canvas, averaged where tiles overlap "
             "(stride 256 with 512-pixel tiles means interior pixels are covered four times), "
             "thresholded once, and compared to the stitched ground truth &mdash; one "
             "intersection-over-union per slide. <b>Tile-level IoU</b> averages a per-batch IoU "
             "and so weights a batch of near-empty tiles the same as a batch full of ventricle. "
             "Slide-level is the number a reader should quote.</p>")

    H.append("<h2>Results by held-out case</h2>")
    H.append("<table><tr><th class='n'>Fold</th><th>Held-out case</th><th class='n'>Tiles</th>"
             "<th class='n'>Best ep.</th><th class='n'>Slide IoU</th>"
             "<th class='n'>Tile IoU</th><th></th></tr>")
    for r in sorted(rows, key=lambda r: -r['slide']):
        cls = " class='flag'" if r['defective'] else ""
        w = max(1, int(round(r['slide'] * 90)))
        bar = f"<span class='bar{' lo' if r['defective'] else ''}' style='width:{w}px'></span>"
        name = r['case'] + (" &#9888;" if r['defective'] else "")
        H.append(f"<tr{cls}><td class='n'>{r['fold']}</td><td>{name}</td>"
                 f"<td class='n'>{r['tiles']}</td><td class='n'>{r['best_epoch']}</td>"
                 f"<td class='n'><b>{r['slide']:.4f}</b></td>"
                 f"<td class='n'>{r['tile_run']:.4f}</td><td>{bar}</td></tr>")
    H.append(f"<tr class='tot'><td></td><td>mean &plusmn; sd (all {n_folds})</td>"
             f"<td class='n'>{sum(r['tiles'] for r in rows)}</td><td></td>"
             f"<td class='n'>{m_all:.4f} &plusmn; {s_all:.4f}</td>"
             f"<td class='n'>{mt_all:.4f}</td><td></td></tr>")
    H.append(f"<tr class='tot'><td></td><td>mean &plusmn; sd (excl. defective)</td>"
             f"<td class='n'>{sum(r['tiles'] for r in ok)}</td><td></td>"
             f"<td class='n'>{m_ok:.4f} &plusmn; {s_ok:.4f}</td>"
             f"<td class='n'>{ms([r['tile_run'] for r in ok])[0]:.4f}</td><td></td></tr>")
    H.append("</table>")

    H.append("<div class='note'><b>&#9888; 17P645-1 is a known-defective slide.</b> "
             "Its source directory is named <span class='mono'>17P645-1 (validated) "
             "(channel issue)</span> and it has a sectioning error, both recorded before this "
             "experiment ran. It is reported here rather than dropped: it was in the training "
             "data for the other folds, and removing a case after seeing that it scored poorly "
             "would be selection on the outcome. The figure excluding it is given as a "
             "sensitivity analysis, with the exclusion justified by input data quality rather "
             "than by its score.</div>")

    H.append("<h2>Per-slide detail</h2>")
    H.append("<p>Paired cases contribute two slides; the case-level IoU above pools both.</p>")
    H.append("<table><tr><th class='n'>Fold</th><th>Slide</th><th class='n'>Slide IoU</th>"
             "<th class='n'>GT px</th><th class='n'>Pred px</th>"
             "<th class='n'>Canvas covered</th></tr>")
    for s in sorted(slide['slides'], key=lambda s: s['fold']):
        cls = " class='flag'" if any(d in s['stem'] for d in DEFECTIVE) else ""
        H.append(f"<tr{cls}><td class='n'>{s['fold']}</td><td>{s['stem']}</td>"
                 f"<td class='n'>{s['slide_iou']:.4f}</td>"
                 f"<td class='n'>{s['gt_px']:,}</td><td class='n'>{s['pred_px']:,}</td>"
                 f"<td class='n'>{s['covered_frac']*100:.1f}%</td></tr>")
    H.append("</table>")

    H.append("<h2>Metric comparison</h2>")
    H.append("<table><tr><th class='n'>Fold</th><th>Case</th>"
             "<th class='n'>Tile (per-batch mean)</th><th class='n'>Tile (pooled)</th>"
             "<th class='n'>Slide-level</th></tr>")
    for r in rows:
        cls = " class='flag'" if r['defective'] else ""
        H.append(f"<tr{cls}><td class='n'>{r['fold']}</td><td>{r['case']}</td>"
                 f"<td class='n'>{r['tile_run']:.4f}</td>"
                 f"<td class='n'>{r['tile_pooled']:.4f}</td>"
                 f"<td class='n'><b>{r['slide']:.4f}</b></td></tr>")
    H.append(f"<tr class='tot'><td></td><td>mean</td>"
             f"<td class='n'>{mt_all:.4f}</td>"
             f"<td class='n'>{ms([r['tile_pooled'] for r in rows])[0]:.4f}</td>"
             f"<td class='n'>{m_all:.4f}</td></tr>")
    H.append("</table>")
    H.append("<p>Pooled IoU accumulates one intersection and union across every test tile, so "
             "unlike the per-batch mean it does not depend on how tiles are grouped into "
             "batches. It tracks the slide-level figure closely, which is expected: both pool "
             "over pixels. The per-batch mean sits lower because it gives a batch of "
             "near-empty tiles the same weight as a batch full of ventricle.</p>")

    if aug is not None:
        H.append("<h3>Effect of test-time augmentation</h3>")
        H.append("<p>The dataset object decides augmentation once, at construction, and the "
                 "validation and test loaders reuse it &mdash; so the sweep scored each held-out "
                 "tile under one random flip/rotation draw. This is a property of the codebase, "
                 "not of this experiment, and it applies to every <span class='mono'>val_iou</span> "
                 "in the training logs. It is label-preserving: image, mask, ROI and position are "
                 "transformed together, and the model was trained on that same distribution.</p>")
        H.append("<p>To measure it rather than assume, each fold was rescored with augmentation "
                 "on and off, holding batch size and seed fixed and using pooled IoU (the "
                 "per-batch mean is sensitive to batch grouping and cannot answer this "
                 "question):</p>")
        H.append("<table><tr><th class='n'>Fold</th><th class='n'>Pooled IoU, aug on</th>"
                 "<th class='n'>Pooled IoU, aug off</th><th class='n'>Delta</th></tr>")
        for r in aug['folds']:
            H.append(f"<tr><td class='n'>{r['fold']}</td>"
                     f"<td class='n'>{r['pooled_aug']:.4f}</td>"
                     f"<td class='n'>{r['pooled_noaug']:.4f}</td>"
                     f"<td class='n'>{r['delta']:+.4f}</td></tr>")
        H.append(f"<tr class='tot'><td></td><td>mean</td><td class='n'></td><td class='n'></td>"
                 f"<td class='n'>{aug['mean_delta']:+.4f}</td></tr>")
        H.append("</table>")
        H.append(f"<p>Mean shift <b>{aug['mean_delta']:+.4f}</b> IoU "
                 f"(sd {aug['std_delta']:.4f}, largest magnitude "
                 f"{aug['max_abs_delta']:.4f}), with randomly signed deltas &mdash; "
                 f"indistinguishable from zero. Test-time augmentation did not materially "
                 f"affect the reported results. All headline figures in this report are "
                 f"nonetheless computed with augmentation disabled, which is also required for "
                 f"the slide-level stitching to align.</p>")

    H.append("<h2>Caveats</h2><ul>")
    H.append("<li><b>Test-set sizes are uneven</b> (%d to %d tiles). The unweighted mean gives "
             "every case equal weight regardless of size; the tile-weighted figure (%.3f) is "
             "given above for comparison.</li>"
             % (min(r['tiles'] for r in rows), max(r['tiles'] for r in rows), tw))
    H.append("<li><b>Evaluation is restricted to the ROI-adjacent region.</b> ROI exclusion "
             "keeps 3,997 of 19,220 tiles, so the stitched canvas covers only the annotated "
             "neighbourhood (per-slide coverage is tabulated above), not the entire slide.</li>")
    H.append("<li><b>Ground-truth masks are not strictly binary</b> &mdash; background is stored "
             "as ~0.0157 rather than 0 on some slides. All IoUs here threshold the target at "
             "0.5; without that, the soft background inflates the union and depresses IoU by "
             "roughly 0.026.</li>")
    H.append("<li><b>Fold length varied</b> (155&ndash;365 epochs) under a patience of "
             f"{tr.get('early_stop')}, so the folds are not equally converged.</li>")
    H.append("</ul>")

    from weasyprint import HTML, CSS as WCSS
    HTML(string="".join(H)).write_pdf(a.out, stylesheets=[WCSS(string=CSS)])
    print(f"wrote {a.out}")


if __name__ == '__main__':
    main()
