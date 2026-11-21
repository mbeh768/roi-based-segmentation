"""Per-slide held-out IoU of the model, overall and per species.

    python figures-for-pub/supplemental/iou_summary.py
"""
import csv
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / 'train.py').exists())
METRICS = (ROOT / 'figures-for-pub' / 'model-baseline'
           / 'k-fold-bnaug-2026-10-17_slide_metrics.json')
OUT = HERE / 'outputs' / 'iou_summary.csv'

iou = {s: v['iou'] for f in json.loads(METRICS.read_text())['folds']
       for s, v in f['per_slide'].items()}
sp = {s: 'Human' if s.startswith('AJK') else 'Pig' for s in iou}

rows = []
for name in ('all', 'Pig', 'Human'):
    a = np.array([v for s, v in iou.items() if name == 'all' or sp[s] == name])
    rows.append({'group': name, 'n': a.size, 'mean': a.mean(),
                 'sd': a.std(ddof=1), 'median': np.median(a),
                 'min': a.min(), 'max': a.max()})

print(f"  {'group':<8}{'n':>3}{'mean':>9}{'SD':>8}{'median':>9}{'min':>8}{'max':>8}")
for r in rows:
    print(f"  {r['group']:<8}{r['n']:>3}{r['mean']:>9.4f}{r['sd']:>8.4f}"
          f"{r['median']:>9.4f}{r['min']:>8.4f}{r['max']:>8.4f}")

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(f'\nwrote {OUT}')
print(f"  manuscript 0.94 +/- 0.04   measured {rows[0]['mean']:.4f} "
      f"+/- {rows[0]['sd']:.4f}")
