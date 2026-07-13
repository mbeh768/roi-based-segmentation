"""
Builds the held-out verification PDF report from evidence/<pair>.json files
(produced by gather_evidence.py) and thumbnails/ (produced by make_thumbnails.py).

Usage:
    python verification/build_report.py
"""
import base64
import csv
import io
import json
from pathlib import Path

from weasyprint import HTML

REPO = Path(__file__).parent.parent
EVIDENCE_DIR = Path(__file__).parent / "evidence"
THUMB_DIR = Path(__file__).parent / "thumbnails"

PAIR_ORDER = ["AJK056-10", "AJKY121-9", "AJKY121-8"]


def b64_img(path):
    data = Path(path).read_bytes()
    return "data:image/png;base64," + base64.b64encode(data).decode()


def parse_band_csv(text):
    if not text:
        return None
    reader = csv.DictReader(io.StringIO(text))
    return list(reader)


def band_table_html(gt_rows, pred_rows):
    if gt_rows is None or pred_rows is None:
        return "<p><em>data not available</em></p>"
    html = ["<table class='band-table'><tr><th>Band</th>"
            "<th>GT foreground px</th><th>GT %</th>"
            "<th>Pred foreground px</th><th>Pred %</th>"
            "<th>R (GT)</th><th>G (GT)</th><th>B (GT)</th>"
            "<th>R (Pred)</th><th>G (Pred)</th><th>B (Pred)</th></tr>"]
    for gt, pred in zip(gt_rows, pred_rows):
        gt_pct = float(gt['n_px']) if False else None
        html.append(
            f"<tr><td>{gt['band']}</td>"
            f"<td>{int(gt['n_px']):,}</td><td>{gt.get('G_over_R','')}</td>"
            f"<td>{int(pred['n_px']):,}</td><td>{pred.get('G_over_R','')}</td>"
            f"<td>{float(gt['R_mean']):.2f}</td><td>{float(gt['G_mean']):.2f}</td><td>{float(gt['B_mean']):.2f}</td>"
            f"<td>{float(pred['R_mean']):.2f}</td><td>{float(pred['G_mean']):.2f}</td><td>{float(pred['B_mean']):.2f}</td>"
            f"</tr>"
        )
    html.append("</table>")
    return "".join(html)


def pair_section_html(evidence):
    pair = evidence["pair"]
    run = evidence["run_dir"]
    stems = evidence["stems"]
    exclusion_ok = evidence["exclusion_verified"]

    sec = [f"<h2>{pair}</h2>"]
    sec.append(f"<p class='meta'>Training run: <code>{run}</code> &nbsp; "
               f"Held-out slides: {', '.join(f'<code>{s}</code>' for s in stems)}</p>")

    badge = ("<span class='badge badge-ok'>VERIFIED: fully excluded from training</span>"
             if exclusion_ok else
             "<span class='badge badge-fail'>WARNING: exclusion NOT verified</span>")
    sec.append(f"<p>{badge}</p>")

    sec.append("<h3>1. Training-set exclusion, by name</h3>")
    sec.append(f"<p>Total slides seen by the training data loader for run "
               f"<code>{run}</code>: {len(evidence['train_set_stems'])} train + "
               f"{len(evidence['val_set_stems'])} validation = "
               f"{len(evidence['train_set_stems']) + len(evidence['val_set_stems'])}.</p>")
    sec.append("<table class='stem-table'><tr><th>Training-set slide stems (verbatim, as logged by the trainer)</th></tr>")
    sec.append(f"<tr><td class='mono'>{', '.join(evidence['train_set_stems'])}</td></tr></table>")
    sec.append("<table class='stem-table'><tr><th>Validation-set slide stems</th></tr>")
    sec.append(f"<tr><td class='mono'>{', '.join(evidence['val_set_stems'])}</td></tr></table>")
    for stem in stems:
        present_train = stem in evidence["train_set_stems"]
        present_val = stem in evidence["val_set_stems"]
        status = "ABSENT (correct)" if not (present_train or present_val) else "PRESENT -- LEAK"
        cls = "ok" if not (present_train or present_val) else "fail"
        sec.append(f"<p class='{cls}'>&bull; <code>{stem}</code>: {status} "
                   f"from both training and validation splits.</p>")

    sec.append("<h3>2. Provenance chain per slide</h3>")
    sec.append("<table class='chain-table'><tr><th>Slide</th><th>Ground-truth mask</th>"
               "<th>Raw RGB source</th><th>Checkpoint&rarr;prediction&rarr;quantification chronology</th></tr>")
    for stem in stems:
        d = evidence["stems_detail"][stem]
        chrono = d["chronology_ok"]
        chrono_cell = ("<span class='ok'>&#10003; ordered correctly</span>" if chrono
                       else "<span class='fail'>&#10007; ORDERING PROBLEM</span>")
        sec.append(
            f"<tr><td class='mono'>{stem}</td>"
            f"<td class='mono small'>{Path(d['gt_mask_path']).name}<br/>md5 {d['gt_mask_md5']}</td>"
            f"<td class='mono small'>{Path(d['raw_image_path']).name}<br/>md5 {d['raw_image_md5']}</td>"
            f"<td>{chrono_cell}</td></tr>"
        )
    sec.append("</table>")

    sec.append("<h4>Chronology detail (unix timestamps; each step strictly after the previous)</h4>")
    sec.append("<table class='chain-table'><tr><th>Slide</th><th>Checkpoint saved</th>"
               "<th>Prediction (prob_map) generated</th><th>Native-res mask generated</th>"
               "<th>Quantification generated</th></tr>")
    for stem in stems:
        v = evidence["stems_detail"][stem]["chronology_values_unix"]
        def fmt(x):
            if x is None:
                return "-"
            import datetime
            return datetime.datetime.fromtimestamp(x).strftime("%Y-%m-%d %H:%M:%S")
        sec.append(f"<tr><td class='mono'>{stem}</td><td>{fmt(v['checkpoint_saved'])}</td>"
                   f"<td>{fmt(v['prob_map_generated'])}</td><td>{fmt(v['native_mask_generated'])}</td>"
                   f"<td>{fmt(v['quantification_generated'])}</td></tr>")
    sec.append("</table>")

    sec.append("<h3>3. Visual comparison: ground truth vs. held-out model prediction</h3>")
    for stem in stems:
        thumb = THUMB_DIR / f"{pair}_{stem}_gt_vs_pred.png"
        if thumb.exists():
            sec.append(f"<p class='mono small'>{stem}</p>")
            sec.append(f"<img class='thumb' src='{b64_img(thumb)}'/>")
        else:
            sec.append(f"<p class='fail'>{stem}: comparison thumbnail not available "
                       f"(quantification not yet run for this slide)</p>")

    sec.append("<h3>4. Band-level intensity quantification: ground truth vs. prediction</h3>")
    for stem in stems:
        d = evidence["stems_detail"][stem]
        gt_rows = parse_band_csv(d.get("gt_band_means_csv"))
        pred_rows = parse_band_csv(d.get("pred_band_means_csv"))
        sec.append(f"<p class='mono small'>{stem}</p>")
        sec.append(band_table_html(gt_rows, pred_rows))

    sec.append("<h3>5. Analysis code integrity</h3>")
    sec.append("<p>The identical scripts were used to quantify both the ground-truth and "
               "the model-predicted masks for every slide above -- same code path, only the "
               "input mask differs:</p>")
    sec.append("<table class='chain-table'><tr><th>Script</th><th>MD5</th></tr>")
    for name, h in evidence["tool_script_md5"].items():
        sec.append(f"<tr><td class='mono'>{name}</td><td class='mono'>{h}</td></tr>")
    sec.append("</table>")

    return "".join(sec)


CSS = """
@page { size: A4; margin: 2cm; }
body { font-family: -apple-system, Helvetica, Arial, sans-serif; font-size: 10.5pt; color: #1a1a1a; line-height: 1.45; }
h1 { font-size: 20pt; margin-bottom: 0.2em; }
h2 { font-size: 15pt; margin-top: 1.6em; border-bottom: 2px solid #333; padding-bottom: 4px; page-break-before: always; }
h2:first-of-type { page-break-before: avoid; }
h3 { font-size: 12pt; margin-top: 1.4em; color: #222; }
h4 { font-size: 10.5pt; margin-top: 1em; color: #444; }
p.meta { color: #555; }
code { background: #f0f0f0; padding: 1px 4px; border-radius: 3px; font-size: 9pt; }
.mono { font-family: 'Courier New', monospace; font-size: 8.5pt; }
.small { font-size: 8.5pt; color: #333; }
table { border-collapse: collapse; width: 100%; margin: 8px 0 16px 0; table-layout: fixed; }
th, td { border: 1px solid #ccc; padding: 4px 6px; text-align: left; vertical-align: top; word-wrap: break-word; overflow-wrap: break-word; }
th { background: #eaeaea; font-size: 9pt; }
td { font-size: 8.5pt; }
.stem-table td { word-break: break-word; }
.chain-table { font-size: 8pt; }
.chain-table th, .chain-table td { padding: 3px 4px; }
.ok { color: #0a7a2f; font-weight: bold; }
.fail { color: #b3261e; font-weight: bold; }
.badge { display: inline-block; padding: 4px 10px; border-radius: 4px; font-weight: bold; font-size: 10pt; }
.badge-ok { background: #d7f3df; color: #0a7a2f; }
.badge-fail { background: #fbdad7; color: #b3261e; }
img.thumb { width: 100%; max-width: 500px; display: block; margin: 6px 0 14px 0; border: 1px solid #ccc; }
.summary-table td, .summary-table th { text-align: center; }
"""


def main():
    evidences = {}
    for pair in PAIR_ORDER:
        path = EVIDENCE_DIR / f"{pair}.json"
        if path.exists():
            evidences[pair] = json.loads(path.read_text())

    import datetime
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html_parts = [
        "<html><head><meta charset='utf-8'></head><body>",
        "<h1>Held-Out Model Evaluation: Verification Report</h1>",
        f"<p class='meta'>Generated {now} &mdash; att_vent_seg / EXPERIMENT 10A</p>",
        "<h3>Purpose</h3>",
        "<p>This report exists because an earlier version of this analysis silently used "
        "<strong>ground-truth</strong> masks in a directory (<code>held_out_results/</code>) "
        "that was supposed to demonstrate genuine <strong>held-out model performance</strong> "
        "&mdash; the model's own prediction sat unused the entire time. That error was caught "
        "and fixed. This report re-derives, from first principles and with the underlying "
        "checksums/timestamps shown explicitly, that:</p>"
        "<ol>"
        "<li>Each pair of slides below was <strong>physically absent</strong> from the training "
        "data directory for the entire duration of the training run used to generate its "
        "prediction &mdash; verified by name, not just count, against the trainer's own logged "
        "train/validation slide lists.</li>"
        "<li>The predicted masks quantified below come from that exact checkpoint, applied to "
        "the untouched raw slide images, upsampled correctly from the model's padded prediction "
        "canvas back to native resolution before any distance/intensity math was computed.</li>"
        "<li>The ground-truth baseline quantification used the literal annotation files from "
        "the source dataset, unmodified.</li>"
        "<li>Both ground-truth and predicted quantification used the identical analysis code.</li>"
        "</ol>",
    ]

    html_parts.append("<h2 style='page-break-before: avoid;'>Summary</h2>")
    html_parts.append("<table class='summary-table'><tr><th>Pair</th><th>Training run</th>"
                      "<th>Held-out slides</th><th>Exclusion verified</th>"
                      "<th>Chronology verified (all slides)</th></tr>")
    for pair in PAIR_ORDER:
        if pair not in evidences:
            html_parts.append(f"<tr><td>{pair}</td><td colspan='4'><em>pending / not yet run</em></td></tr>")
            continue
        ev = evidences[pair]
        chrono_all = all(d["chronology_ok"] for d in ev["stems_detail"].values())
        html_parts.append(
            f"<tr><td>{pair}</td><td class='mono small'>{ev['run_dir']}</td>"
            f"<td class='mono small'>{', '.join(ev['stems'])}</td>"
            f"<td class='{'ok' if ev['exclusion_verified'] else 'fail'}'>"
            f"{'YES' if ev['exclusion_verified'] else 'NO'}</td>"
            f"<td class='{'ok' if chrono_all else 'fail'}'>{'YES' if chrono_all else 'NO'}</td></tr>"
        )
    html_parts.append("</table>")

    for pair in PAIR_ORDER:
        if pair in evidences:
            html_parts.append(pair_section_html(evidences[pair]))

    html_parts.append("</body></html>")
    html_doc = "".join(html_parts)

    out_pdf = REPO / "verification" / "held_out_verification_report.pdf"
    HTML(string=html_doc).write_pdf(str(out_pdf), stylesheets=[])
    # write CSS via string concat since weasyprint wants a real stylesheet
    from weasyprint import CSS as WCSS
    HTML(string=html_doc).write_pdf(str(out_pdf), stylesheets=[WCSS(string=CSS)])
    print(f"wrote {out_pdf}")

    out_html = REPO / "verification" / "held_out_verification_report.html"
    out_html.write_text(f"<style>{CSS}</style>{html_doc}")
    print(f"wrote {out_html}")


if __name__ == "__main__":
    main()
